> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 6 — NLP: Tokens and Word Embeddings

**Purpose:** See how a machine reads: ==a language model scores which token comes next, text is cut into sub-word tokens with byte-pair encoding, and every token ID becomes a vector== that a network can compute with.

**Term:** Midterm (Weeks 6–10) · **Month:** 2 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch text turn into pieces, counts and vectors",
  "caption": "A language model never sees a sentence the way you do. It sees tokens and it scores which one comes next. The splits in the first two scenes are an example, not the output of a real tokenizer. The counts in the third and fourth scenes are real counts from a 45-word text, and the cosine numbers in the last scene come from illustrative vectors.",
  "scenes": [
    { "name": "One word", "kind": "tokens", "outputLabel": "Tokens (an example split)", "prompt": "unbelievably", "tokens": ["un", "believ", "ably"] },
    { "name": "A sentence", "kind": "tokens", "outputLabel": "Tokens (an example split)", "prompt": "Tokenization is fun", "tokens": ["Token", "ization", " is", " fun"] },
    { "name": "Counting", "kind": "text", "outputLabel": "Real counts from a small text", "prompt": "the canteen opens at seven. the students buy rice and fish. … (45 words in all)", "output": "The text has 45 words but only 23 different ones. The word “the” appears 9 times. It is followed by “students” 4 times, “canteen” 3 times, “teacher” once and “counter” once. So the chance that “students” comes after “the” is 4 out of 9." },
    { "name": "Next word", "kind": "bars", "outputLabel": "P(next word | the), from those counts", "prompt": "the", "bars": [ { "label": "students", "value": 0.44 }, { "label": "canteen", "value": 0.33 }, { "label": "teacher", "value": 0.11 }, { "label": "counter", "value": 0.11 } ] },
    { "name": "Word vectors", "kind": "text", "outputLabel": "Cosine similarity (illustrative vectors)", "prompt": "How alike are two words?", "output": "With the vectors used later in this lesson, cat and dog score 0.97 and cat and rice score 0.17. When every word becomes a list of numbers, “similar” becomes something you can calculate." }
  ]
}
```

## By the end, you can…

1. ==Explain what a language model is== and describe the three eras: hand-written rules, statistics (n-grams) and neural models.
2. ==Compute a bigram probability, a sentence probability and a perplexity by hand==, and say what a lower perplexity means.
3. ==Explain why n-gram tables explode== (V^n entries, most of them never seen) and how add-one smoothing and interpolation help.
4. ==Compare word, character and sub-word tokenisation== and choose between them for a vocabulary, a sequence length and a language.
5. ==Run byte-pair-encoding (BPE) merges by hand== and encode a new word with the merges you learned.
6. ==Compute a cosine similarity and explain what an embedding matrix lookup does==, including the limits and biases of word vectors.

Assessed in the Week 6 lab and quiz. Read each section, then play with the figure under it: the figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. Notes such as “Lab Part 5” tell you which part of the lab lets you try the idea for real. Week 1 already showed you tokens, the chain rule, an n-gram model and perplexity at a first-look level, so this week we do not start again: we go deeper, with exact numbers, and then we open the box one step further, to see how a token turns into a vector. Transformers and attention are the topic of Week 7, so here we only name them. Every number in the text was computed, not guessed.

## What is a language model? {rail=Language models}

{lede} A [[language model|A model that gives a probability to any piece of text by saying how likely each next word (or token) is, given the words before it.]] is a ==probability model of text==: given the words so far, it says how likely each possible next word is. Everything else in this week, tokens, counts and vectors, exists to make that one job possible.

### An everyday picture

Open a messaging app and type “Magandang”. Your phone offers “umaga”, “gabi” and “hapon” (morning, evening, afternoon). It has not understood that you want to greet someone. It has simply seen that these words very often follow “Magandang”, so it ranks them. If you type “Salamat” it may offer “po”. That ranking is a language model in its smallest form. In Week 1 you saw the same job in GPT-2, which ranks 50,257 possible tokens at every step. We write the job as one conditional probability:

P(next word | previous words)

The bar “|” reads “given”. The part before the bar is the thing we want, the part after it is what we already know.

### From one step to a whole sentence

If a model can give the probability of the next word, it can give the probability of a whole sentence by chaining the steps together. You met this in Week 1 as the [[chain rule|The rule that the probability of a sequence equals the product of each item’s probability given everything before it.]]. Select each part of the formula, then step through a new worked example.

```viz
{
  "type": "equation",
  "title": "The chain rule for a sentence (a new example)",
  "intro": "Select each symbol, then step through a four-word sentence from the school canteen.",
  "formula": [
    { "t": "P(w₁ … wₙ)", "id": "whole" },
    " = ",
    { "t": "Π", "id": "pi" },
    " ",
    { "t": "P(wᵢ | w<ᵢ)", "id": "step" }
  ],
  "parts": [
    { "id": "whole", "name": "P(w₁ … wₙ)", "meaning": "The probability of the whole sentence of n words. It is a number between 0 and 1, and for any real sentence it is very small." },
    { "id": "pi", "name": "Π (capital pi)", "meaning": "“Multiply all of these”, for i = 1, 2, … up to n. It is to multiplying what Σ is to adding." },
    { "id": "step", "name": "P(wᵢ | w<ᵢ)", "meaning": "The probability of the i-th word given every word before it (w<ᵢ means w₁ up to wᵢ₋₁). This is exactly one next-word prediction." }
  ],
  "steps": [
    { "title": "Split the sentence into steps", "expr": "“The students eat rice”: P(The) × P(students | The) × P(eat | The students) × P(rice | The students eat)", "note": "Four words, four factors. Each factor is one prediction." },
    { "title": "Put in the numbers", "expr": "0.2 × 0.1 × 0.05 × 0.2", "note": "These factors are illustrative numbers authored for teaching." },
    { "title": "Multiply", "expr": "0.2 × 0.1 = 0.02;  0.02 × 0.05 = 0.001;  0.001 × 0.2 = 0.0002", "note": "A probability of 0.0002 is 1 chance in 5,000. Each extra word makes the product smaller." }
  ],
  "note": "The four factors are made up. A real model would produce them from its parameters."
}
```

Notice that the sentence is only as likely as its weakest steps allow: one surprising word (a factor of 0.05) pulls the whole product down. ==Long sentences always get tiny probabilities==, so we need a fairer way to compare them. That is perplexity, below.

### What can you do with a probability model of text?

Two kinds of jobs follow from one model. You can use it to ==score== a piece of text (how likely is it?), or to ==generate== text (draw the next word again and again). Many products use both. Sort the eight tasks below.

```viz
{
  "type": "sorter",
  "title": "Scoring text or generating text?",
  "intro": "Drag or select each task into the bin that describes how the language model is used.",
  "bins": [ { "id": "scoring", "label": "Scoring text" }, { "id": "generating", "label": "Generating text" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "A spell checker deciding between “their” and “there” in a sentence", "answer": "scoring", "why": "It compares how likely the sentence is with each word in the gap and keeps the higher score. Nothing new is written." },
    { "name": "Phone autocomplete offering three next words", "answer": "generating", "why": "It proposes the most probable next words. Each suggestion is a small piece of generated text." },
    { "name": "A speech recogniser choosing between two sentences that sound alike", "answer": "scoring", "why": "The sound fits both. The language model says which word sequence is far more likely in the language." },
    { "name": "A chatbot writing a reply", "answer": "generating", "why": "It draws one next token at a time, as in Week 1, until the reply is finished." },
    { "name": "Computing the perplexity of a test text", "answer": "scoring", "why": "Perplexity is built from the probabilities the model gives to the text. It does not write anything." },
    { "name": "Writing a story from a one-line prompt", "answer": "generating", "why": "The prompt is the start of the text and the model continues it." },
    { "name": "A translation system that writes several candidate sentences and keeps the most fluent one", "answer": "both", "why": "Writing the candidates is generating. Ranking them by how fluent they are is scoring." },
    { "name": "A keyboard that offers next words and also flags a likely typo", "answer": "both", "why": "Suggesting words is generating. Noticing that a word is unlikely in its place is scoring." }
  ]
}
```

Notice how one table of probabilities lets you do both. That is why the language model is the engine under so many language tools.

### How good is a language model? Perplexity

To judge a model, give it a text it has never seen and ask how surprised it is. A model that gave high probabilities to the real words is a good model. But raw sentence probabilities depend on length, so we use [[perplexity|A measure of how surprised a language model is by a text: the sentence probability turned upside down and averaged per word. Lower is better.]]. Select the parts of the formula and follow the numbers.

```viz
{
  "type": "equation",
  "title": "Perplexity: PP = P^(−1/n) = exp(−(1/n) Σ ln P)",
  "intro": "Select each symbol, then step through a four-word sentence whose total probability is 0.002.",
  "formula": [
    { "t": "PP", "id": "pp" },
    " = ",
    { "t": "P(w₁ … wₙ)", "id": "whole" },
    { "t": "^(−1/n)", "id": "root" },
    " = exp( −",
    { "t": "(1/n)", "id": "mean" },
    " ",
    { "t": "Σ ln P(wᵢ | w<ᵢ)", "id": "sumlog" },
    " )"
  ],
  "parts": [
    { "id": "pp", "name": "PP (perplexity)", "meaning": "How many equally likely words the model was torn between at each step, on average. Lower means the text fits the model better. A perfectly sure model has PP = 1." },
    { "id": "whole", "name": "P(w₁ … wₙ)", "meaning": "The probability the model gives to the whole text: the product of all its next-word probabilities." },
    { "id": "root", "name": "^(−1/n)", "meaning": "Turn the probability upside down and take the n-th root. This spreads the total over the n words, so a long text and a short text can be compared." },
    { "id": "mean", "name": "1/n", "meaning": "n is the number of tokens in the text. Dividing by n takes an average per token." },
    { "id": "sumlog", "name": "Σ ln P(wᵢ | w<ᵢ)", "meaning": "Add the natural logarithms of the n next-word probabilities. Adding logs is the same as multiplying probabilities, but it avoids numbers too small for a computer." }
  ],
  "steps": [
    { "title": "The four probabilities", "expr": "0.2, 0.1, 0.25, 0.4", "note": "One probability for each word of a four-word sentence (illustrative)." },
    { "title": "The sentence probability", "expr": "P = 0.2 × 0.1 × 0.25 × 0.4 = 0.002", "note": "1 chance in 500." },
    { "title": "Perplexity by the power", "expr": "PP = 0.002^(−1/4) = 500^(1/4) ≈ 4.73", "note": "The fourth root of 500 is about 4.73." },
    { "title": "Perplexity by the logarithms", "expr": "ln 0.002 = −6.2146;  −(1/4) × (−6.2146) = 1.5537;  exp(1.5537) ≈ 4.73", "note": "The same answer. Real systems use this route." },
    { "title": "Read it in words", "expr": "PP ≈ 4.73", "note": "The model was about as unsure as if it had to pick among roughly five equally likely words at every step." }
  ],
  "note": "The four probabilities are made up. Compare perplexities only for the same text and the same tokens: a model that reads words and a model that reads sub-words are not measured on the same scale."
}
```

Now see why a better model has a lower perplexity. The next figure scores the same six-word sentence under two models. Model A finds each word fairly likely, Model B is more confused.

```viz
{
  "type": "chain",
  "title": "The same sentence under a better and a weaker model",
  "sentences": [
    { "label": "Model A (better)", "tokens": [ { "t": "The", "p": 0.2 }, { "t": "students", "p": 0.12 }, { "t": "eat", "p": 0.1 }, { "t": "rice", "p": 0.25 }, { "t": "at", "p": 0.3 }, { "t": "noon", "p": 0.4 } ], "note": "Every word is fairly likely after the ones before it." },
    { "label": "Model B (weaker)", "tokens": [ { "t": "The", "p": 0.05 }, { "t": "students", "p": 0.04 }, { "t": "eat", "p": 0.03 }, { "t": "rice", "p": 0.08 }, { "t": "at", "p": 0.1 }, { "t": "noon", "p": 0.15 } ], "note": "Every step is more of a surprise." }
  ],
  "note": "All probabilities are illustrative, authored for teaching. Model A multiplies to 0.000072 (perplexity about 4.90) and Model B to 7.2 × 10⁻⁸ (perplexity about 15.5), so A thinks the sentence is 1,000 times more likely."
}
```

Notice that both models read the same words, yet B’s perplexity is more than three times A’s. ==A lower perplexity means the model assigned higher probability to the real text, so it predicts that text better.== In Lab Part 1 you will compute a perplexity yourself for a counting model.

### Predicting is not understanding

One caution before we go on. A language model is trained to predict, and it predicts well. ==It does not follow that it understands in the way you do, or that what it says is true.== Week 1 called this “fluent is not the same as true”. A model can give a high probability to a smooth sentence that is wrong. Keep this in mind whenever a probability sounds like a fact.

### The whole path, step by step

Finally, here is the path a piece of text takes through a language model. The first and last steps are the subject of this week: how text becomes numbers, and how numbers become meaning.

```viz
{
  "type": "pipeline",
  "title": "From text to the next token",
  "intro": "Step through the path. Each step lights up the part of the system that is working.",
  "nodes": [
    { "id": "text", "label": "Text", "sub": "the students eat" },
    { "id": "tokens", "label": "Tokens", "sub": "pieces" },
    { "id": "ids", "label": "IDs", "sub": "whole numbers" },
    { "id": "model", "label": "Model", "sub": "vectors, layers" },
    { "id": "probs", "label": "Probabilities", "sub": "one per token" },
    { "id": "next", "label": "Next token", "sub": "picked" }
  ],
  "steps": [
    { "title": "1. Cut the text into tokens", "body": "A tokenizer splits the text into pieces. This is the work of the Tokens and BPE sections of this lesson.", "active": ["text", "tokens"], "example": "the | students | eat" },
    { "title": "2. Replace each token by its ID", "body": "A table called the vocabulary gives every piece a whole-number ID. A network can only work with numbers.", "active": ["tokens", "ids"], "example": "[5, 31, 12] (made-up IDs)" },
    { "title": "3. Run the model", "body": "The model turns each ID into a vector and then computes with those vectors. The first part, the vector lookup, is in the Embeddings section.", "active": ["ids", "model"], "example": "ID 31 → a vector of d numbers" },
    { "title": "4. Score every possible next token", "body": "The model gives a score to each token in the vocabulary, and softmax (Week 1) turns the scores into probabilities that add up to 1.", "active": ["model", "probs"], "example": "rice 0.25, fish 0.20, noodles 0.05, …" },
    { "title": "5. Pick one and repeat", "body": "One token is chosen, by taking the largest or by sampling as in Week 1. It is added to the text and the loop starts again.", "active": ["probs", "next", "text"], "example": "the students eat rice → …" }
  ],
  "note": "IDs and probabilities in the examples are made up."
}
```

Notice that the loop starts and ends with text, but the model in the middle only ever sees numbers. The rest of the lesson fills in those steps, starting with the history of how language models were built.

## From rules to statistics to neural models {rail=History}

{lede} Language models were built in three eras. First people wrote ==rules==, then computers ==counted word sequences==, and now ==neural networks learn a vector for every token== from huge amounts of text. Each era fixed the biggest weakness of the one before, and each brought a new weakness of its own.

### Era 1: rules

Imagine a school help desk where the rule book says: “If the student writes ‘I feel ___’, reply ‘Why do you feel ___?’; if the message contains ‘exam’, reply ‘Tell me more about the exam’.” That is a [[rule-based system|A program whose behaviour comes from rules written by hand by people, for example patterns such as “if the message contains X, reply Y”.]]. In 1966 Joseph Weizenbaum’s program ELIZA worked this way: it matched keywords and filled the matched words into a template, and some people felt it was listening. It understood nothing, and its weakness is plain: ==language is too rich for any list of rules==. Every new topic, spelling or word order needs another rule, and the rules start to fight each other. In Lab Part 2 you build a small rule-based bot with six rules and count how many inputs it can handle.

### Era 2: statistics

The next idea is to stop writing rules and ==count==. In 1948 Claude Shannon showed that text has regularities you can use: the letters or words before a spot make some choices far more likely than others. A computer can count, in a large text, how often each word follows each other word, and turn the counts into probabilities. This is the [[n-gram model|A language model that estimates the next word from counts of the last n − 1 words in a training text. A bigram model uses one previous word, a trigram model two.]], the subject of the next section. For decades, n-gram models were the language models inside speech recognisers and early statistical translation systems. They were careful and general, and they needed no linguist, only text and a counter. Their weakness: ==they only know the exact word sequences they have counted==, so they stumble on anything new.

### Era 3: neural models

The third idea gives every word a ==vector== (a short list of numbers) that the network learns, so that words used in similar ways get similar vectors. Then what the model learns about “cat” also helps it with “dog”. Bengio and colleagues published the first well-known neural language model in 2003. In 2013 Mikolov and colleagues showed with word2vec that such vectors can be learned quickly from very large text. Recurrent networks such as the LSTM read one word at a time and carry a memory. In 2017 the Transformer let a model look at all the words of a text at once, and GPT-style models, trained to predict the next token, followed from 2018. Their weakness is not accuracy but cost and opacity: they need a great deal of data and computing power, and it is hard to see why they said something. We study the Transformer in Weeks 7 and 8.

Select each event on the timeline and notice which problem it solved.

```viz
{
  "type": "timeline",
  "title": "Language models, 1948 to 2022",
  "events": [
    { "year": "1948", "title": "Shannon: text is predictable", "text": "In A Mathematical Theory of Communication, Claude Shannon showed how English can be approximated by choosing letters and words according to how often they follow the ones before. It is the seed of the n-gram idea." },
    { "year": "1966", "title": "ELIZA", "text": "Joseph Weizenbaum’s program matched keywords and filled them into templates, imitating a conversation. It used hand-written rules and understood nothing." },
    { "year": "1980s", "title": "N-grams in speech recognition", "text": "Speech recognisers began to combine a sound model with a statistical language model that counted word sequences, so that a likely word sequence beat an unlikely one that sounded the same." },
    { "year": "Early 1990s", "title": "Statistical machine translation", "text": "Researchers at IBM learned translation from pairs of sentences by counting, with a language model to keep the output fluent. Counting replaced hand-written translation rules." },
    { "year": "1997", "title": "LSTM", "text": "Hochreiter and Schmidhuber introduced the long short-term memory, a recurrent network that can keep information over many steps. For years it was the standard way for a neural network to read text one word at a time." },
    { "year": "2003", "title": "A neural probabilistic language model", "text": "Bengio and colleagues trained a network that learned a vector for every word and used the vectors to predict the next word. Similar words ended up with similar vectors." },
    { "year": "2013", "title": "word2vec", "text": "Mikolov and colleagues showed that simple, fast training gives high-quality word vectors from very large text collections, and that directions in the vector space carry meaning." },
    { "year": "2016", "title": "BPE for neural translation", "text": "Sennrich and colleagues used byte-pair encoding to split rare words into pieces the model already knows, so neural translation could read new words without a huge vocabulary." },
    { "year": "2017", "title": "The Transformer", "text": "Vaswani and colleagues introduced the Transformer, which uses attention to let every token look at every other token. You study it in Weeks 7 and 8." },
    { "year": "2018", "title": "BERT and GPT", "text": "Devlin and colleagues introduced BERT, and Radford and colleagues the first GPT. Both are Transformers trained on large text with sub-word tokens." },
    { "year": "2019", "title": "GPT-2", "text": "GPT-2 was trained to predict the next token and uses a vocabulary of 50,257 byte-level BPE tokens. You used it in the Week 1 lab." },
    { "year": "2020", "title": "GPT-3", "text": "Brown and colleagues reported GPT-3 with 175 billion parameters and showed that a very large next-token predictor can do many tasks from a few examples in its prompt." },
    { "year": "2022", "title": "ChatGPT", "text": "OpenAI released ChatGPT in November 2022, a chat assistant based on a GPT model that was further trained with human feedback. It brought language models to a very wide public." }
  ]
}
```

Notice that the job stayed the same through all of these steps: predict what comes next. What changed is how much of the text the model can use and how it represents a word.

### The same prompt, three ways

Here is the same kind of input handled by each era. Switch between the three tabs and compare what each one needs.

```viz
{
  "type": "flow",
  "title": "One prompt, three kinds of model",
  "modes": [
    {
      "name": "Rules",
      "tagline": "Hand-written patterns",
      "inputLabel": "Message",
      "modelLabel": "Pattern rules",
      "outputLabel": "Reply",
      "examples": [
        { "label": "A rule matches", "input": "I feel tired today", "output": "Why do you feel tired today?" },
        { "label": "No rule matches", "input": "Kumusta ang practice ng robotics club?", "output": "Please tell me more." }
      ],
      "note": "Outputs are made up to show the idea. The first reply works because a rule for “I feel …” exists; the second is a fallback because no rule covers the question."
    },
    {
      "name": "N-grams",
      "tagline": "Counted word sequences",
      "inputLabel": "Previous word",
      "modelLabel": "Table of counts",
      "outputLabel": "Most likely next word",
      "examples": [
        { "label": "A pair seen often", "input": "good", "output": "morning (if “good morning” is the commonest pair in the text)" },
        { "label": "A pair never seen", "input": "good robots", "output": "No counts for “robots” after “good”: probability 0" }
      ],
      "note": "Outputs are made up. The model can only repeat patterns it has counted, so a new pair gets probability zero unless smoothing is used."
    },
    {
      "name": "Neural",
      "tagline": "Learned vectors",
      "inputLabel": "Text so far",
      "modelLabel": "Network with learned vectors",
      "outputLabel": "Probabilities for the next token",
      "examples": [
        { "label": "A new combination", "input": "The dog slept in the", "output": "garden 0.30, kitchen 0.25, … (the model has never seen this exact sentence, but “dog” has a vector close to “cat”, which it has seen)" },
        { "label": "A rare word", "input": "The robotics club meets in the", "output": "room 0.35, lab 0.20, … (a distribution over the whole vocabulary)" }
      ],
      "note": "Outputs are made up. A real neural model would produce a probability for every token in its vocabulary."
    }
  ]
}
```

Notice that the neural model is the only one that gives a sensible answer to a sentence it has never seen, because similar words share similar vectors. The table below puts the three eras side by side.

| | Rules | Statistics (n-grams) | Neural models |
|---|---|---|---|
| What it needs | Experts who write the rules | A large text and a counter | A very large text, a network and many hours of training |
| How it generalises | Only to inputs a rule matches | Only to word sequences it has counted | To new combinations, because similar words get similar vectors |
| How big it is | Hundreds or thousands of rules | A table with one entry per seen n-gram | Millions to billions of parameters (GPT-2 small has 124 million) |
| What breaks | Anything outside the rules | Unseen n-grams, long-range links between words | Cost, the need for data, and being hard to inspect |

Lab Part 2 builds a tiny model of each kind on the same text and measures how each one handles input it has not seen. Try the sorter below first.

```viz
{
  "type": "sorter",
  "title": "Which era is it?",
  "intro": "Place each example in the era it belongs to.",
  "bins": [ { "id": "rules", "label": "Rules" }, { "id": "statistical", "label": "Statistical" }, { "id": "neural", "label": "Neural" } ],
  "cards": [
    { "name": "A regular expression that answers “I feel X” with “Why do you feel X?”", "answer": "rules", "why": "The behaviour is a pattern written by a person. Nothing is learned from data." },
    { "name": "Counting how often “morning” follows “good” in a large text", "answer": "statistical", "why": "It estimates a probability from counts of word sequences. That is an n-gram model." },
    { "name": "A network that gives “dog” a vector close to the vector of “cat”", "answer": "neural", "why": "The vectors are learned by training, and similar words end up with similar vectors." },
    { "name": "A speech recogniser from the 1980s that prefers likely word triples", "answer": "statistical", "why": "A trigram table of counts, combined with a sound model, ranks the candidate sentences." },
    { "name": "A hand-written grammar that lists how a sentence is built from nouns and verbs", "answer": "rules", "why": "People wrote the grammar. It breaks on any construction nobody wrote down." },
    { "name": "An LSTM that reads a sentence one word at a time and updates a memory", "answer": "neural", "why": "It is a neural network with learned weights that carries information from word to word." },
    { "name": "A table of word-pair counts with one added to every count", "answer": "statistical", "why": "Add-one smoothing is a counting trick for n-gram models, so it is statistical." },
    { "name": "A GPT-style model trained by backpropagation to predict the next token", "answer": "neural", "why": "It learns its weights by gradient descent on a huge text, using a Transformer network." },
    { "name": "A kiosk that replies “Closed” whenever the message contains “open”", "answer": "rules", "why": "A single hand-written keyword rule, with no learning. It also shows how crude such rules can be." }
  ]
}
```

Notice that the statistical and neural examples both learn from text, but only the neural ones learn vectors. ==Every era is a trade between what you must write by hand and what the machine learns from data.==

## N-gram models and their limits {rail=N-grams}

{lede} An n-gram model turns counting into prediction: ==the chance of the next word is how often it followed the same few words in the training text==. It is simple enough to do by hand, which makes it the best place to see both what a language model does and why this first idea runs out of road.

### The bigram: one word of memory

Think of a school canteen menu board that never changes. If you have watched the canteen for a month and “rice” was followed by “and” every single time, you would bet that “rice” will be followed by “and” tomorrow. A [[bigram|A pair of neighbouring words. A bigram model predicts each word from only the one word before it.]] model does exactly this bet, for every word. For a word that came before, it looks at all the words that ever followed it and uses the shares. Select each symbol, then follow the numbers on a 45-word canteen text.

```viz
{
  "type": "equation",
  "title": "The bigram probability: count(pair) ÷ count(previous word)",
  "intro": "Select each part, then step through the 45-word canteen text used in the figure below.",
  "formula": [
    { "t": "P(wₙ | wₙ₋₁)", "id": "p" },
    " = ",
    { "t": "count(wₙ₋₁ wₙ)", "id": "pair" },
    " / ",
    { "t": "count(wₙ₋₁)", "id": "prev" }
  ],
  "parts": [
    { "id": "p", "name": "P(wₙ | wₙ₋₁)", "meaning": "The chance that the next word is wₙ, given that the word just before it is wₙ₋₁." },
    { "id": "pair", "name": "count(wₙ₋₁ wₙ)", "meaning": "How many times the two words appear next to each other, in this order, in the training text." },
    { "id": "prev", "name": "count(wₙ₋₁)", "meaning": "How many times the previous word appears with some word after it. It is also the total of the counts of everything that follows it, so the probabilities add up to 1." }
  ],
  "steps": [
    { "title": "Count the previous word", "expr": "count(the) = 9", "note": "“the” appears 9 times in the text and is followed by a word each time." },
    { "title": "Count each pair", "expr": "the students 4,  the canteen 3,  the teacher 1,  the counter 1", "note": "4 + 3 + 1 + 1 = 9, as it must." },
    { "title": "Divide", "expr": "P(students | the) = 4/9 ≈ 0.444;  P(canteen | the) = 3/9 ≈ 0.333;  P(teacher | the) = 1/9 ≈ 0.111", "note": "The four probabilities add up to 1." },
    { "title": "Score a short phrase", "expr": "P(students | the) × P(eat | students) = 4/9 × 1/4 = 1/9 ≈ 0.111", "note": "“students” is followed by buy, eat, pay and go once each, so P(eat | students) = 1/4. This is the chain rule with a one-word memory." },
    { "title": "A longer phrase", "expr": "“the students eat at noon”: 4/9 × 1/4 × 1/1 × 1/5 = 1/45 ≈ 0.022", "note": "Every “eat” is followed by “at” (1/1) and “at” is followed by five different words once each (1/5). The chance of the first word, P(the), is left out." }
  ],
  "note": "The counting ignores the full stops, so the last word of one sentence counts as followed by the first word of the next. That is also what the figure below does."
}
```

Now try it. Pick a previous word and read the probabilities, then generate a few sentences.

```viz
{
  "type": "ngram",
  "title": "A bigram model from a 45-word canteen text",
  "intro": "Pick “the”, then “at”, then “and”: which words have only one follower and which have several? Then press the generate button a few times.",
  "text": "the canteen opens at seven. the students buy rice and fish. the students eat at noon. the teacher eats rice and eggs. the canteen sells rice and fish. the students pay at the counter. the canteen closes at four. the students go home at five.",
  "start": "the",
  "note": "A tiny text on purpose, so you can check every count by hand. Nothing in a real model is this small."
}
```

Notice what the model does. After “rice” it always says “and”, because that is all it has seen. After “at” it has five choices of equal weight. Generated sentences look like the canteen text for a few words and then drift or loop, for example “the students eat at the counter”, which the text never contains. The model only looks one word back, and it has no idea what a sentence is for. Lab Part 1 builds this model in plain Python on a robotics-club text, step by step.

### Longer memory: trigrams and n-grams

What if the model remembers more? An [[n-gram|A run of n neighbouring words. A trigram has three words and predicts the third from the two before it.]] model uses the previous n − 1 words. For a trigram, count the three-word sequence and divide by the count of the two-word context.

```viz
{
  "type": "equation",
  "title": "The n-gram probability: a longer context",
  "intro": "Select each part, then compare a trigram with the bigram for the same word.",
  "formula": [
    { "t": "P(wₙ | wₙ₋ₙ₊₁ … wₙ₋₁)", "id": "p" },
    " = ",
    { "t": "count(wₙ₋ₙ₊₁ … wₙ)", "id": "seq" },
    " / ",
    { "t": "count(wₙ₋ₙ₊₁ … wₙ₋₁)", "id": "ctx" }
  ],
  "parts": [
    { "id": "p", "name": "P(wₙ | wₙ₋ₙ₊₁ … wₙ₋₁)", "meaning": "The chance of the next word given the n − 1 words just before it. For n = 2 this is the bigram, for n = 3 the trigram." },
    { "id": "seq", "name": "count(wₙ₋ₙ₊₁ … wₙ)", "meaning": "How often the whole run of n words appears in the training text." },
    { "id": "ctx", "name": "count(wₙ₋ₙ₊₁ … wₙ₋₁)", "meaning": "How often the context alone (the first n − 1 words of that run) appears." }
  ],
  "steps": [
    { "title": "A bigram answer", "expr": "P(counter | the) = 1/9 ≈ 0.111", "note": "“the” is followed by counter once in nine cases." },
    { "title": "A trigram answer", "expr": "P(counter | at the) = count(at the counter) / count(at the) = 1/1 = 1", "note": "The two-word context “at the” occurs only once in the text." },
    { "title": "A trigram that is less sure", "expr": "P(fish | rice and) = 2/3 ≈ 0.667;  P(eggs | rice and) = 1/3 ≈ 0.333", "note": "“rice and” occurs three times: fish twice and eggs once." },
    { "title": "Read the two answers", "expr": "0.111  →  1.000", "note": "The longer context made the model far more confident, but only because it has seen “at the” once. A probability of 1 from a single example is a warning, not a fact." }
  ],
  "note": "Counts are from the 45-word canteen text above."
}
```

Notice the trade: ==a longer context makes the model more specific, but every context is seen less often==, so the counts become thin. A trigram sees only the last two words, a 4-gram the last three, and that still cannot link “students” at the start of a long sentence to the verb at its end.

### The sparsity problem

How big would the table be? With a vocabulary of V different words there are V × V = V² possible bigrams, V³ possible trigrams, and V^n possible n-grams. Select the symbols, then watch the numbers explode.

```viz
{
  "type": "equation",
  "title": "How many n-grams are possible? V^n",
  "intro": "Select each symbol, then step through a vocabulary of 10,000 words.",
  "formula": [
    { "t": "N", "id": "N" },
    " = ",
    { "t": "V", "id": "V" },
    { "t": "ⁿ", "id": "n" }
  ],
  "parts": [
    { "id": "N", "name": "N (possible n-grams)", "meaning": "How many different runs of n words could exist, whether or not they ever appear in any text." },
    { "id": "V", "name": "V (vocabulary size)", "meaning": "How many different words the model knows." },
    { "id": "n", "name": "n (length of the run)", "meaning": "How many words are in each n-gram: 1, 2, 3, …" }
  ],
  "steps": [
    { "title": "Unigrams (n = 1)", "expr": "10,000¹ = 10,000", "note": "One entry per word." },
    { "title": "Bigrams (n = 2)", "expr": "10,000² = 100,000,000 = 10⁸", "note": "One hundred million possible pairs." },
    { "title": "Trigrams (n = 3)", "expr": "10,000³ = 10¹²", "note": "One million million possible triples." },
    { "title": "4-grams (n = 4)", "expr": "10,000⁴ = 10¹⁶", "note": "Ten thousand million million possible 4-grams." },
    { "title": "Compare with real text", "expr": "A text of 1 billion words contains at most 10⁹ different trigrams, so at most 10⁹ / 10¹² = 0.1% of the possible trigrams can ever be seen.", "note": "The other 99.9% have a count of zero." }
  ],
  "note": "The arithmetic is exact. Real text uses only a tiny part of the possible n-grams, but the table you would have to cover them all is still enormous."
}
```

The calculator below does the same thing with sliders. Keep V at 10,000 and the text at 1,000 million words, then raise n from 1 to 4 and watch the share of n-grams that can ever appear fall off a cliff.

```viz
{
  "type": "calc",
  "kind": "ngram_space",
  "title": "How much of the n-gram table can ever be filled?"
}
```

At n = 2 a billion words can in principle fill every bigram (100%). At n = 3 at most 0.1% can be filled, and the figure shows this as 1.0e-1%, which means 0.1%. At n = 4 the share is 1.0e-5%. The same fact on a log scale:

```viz
{
  "type": "logbars",
  "title": "Possible n-grams for a 10,000-word vocabulary",
  "unit": "possible n-grams",
  "items": [
    { "label": "Unigrams (n = 1)", "value": 10000, "note": "10⁴" },
    { "label": "Bigrams (n = 2)", "value": 100000000, "note": "10⁸" },
    { "label": "Trigrams (n = 3)", "value": 1000000000000, "note": "10¹²" },
    { "label": "4-grams (n = 4)", "value": 10000000000000000, "note": "10¹⁶" }
  ]
}
```

On our tiny canteen text the same thing happens on a small scale: the 45 words use V = 23 different words, so 23 × 23 = 529 bigrams are possible, but only 35 different ones appear, about 6.6%. ==Most possible n-grams are never seen, so a model built only from counts has nothing to say about them.== Lab Part 3 measures this on the robotics-club text and on new text.

### Zero probability, and add-one smoothing

[[Sparsity|The situation in which most of the possible events (here, n-grams) never occur in the data, so their counts are zero.]] is not only about memory. If a new sentence contains one bigram that the training text never had, its count is 0, so its probability is 0. In the chain rule ==that single zero multiplies everything else to 0==, and the perplexity becomes infinite. Real text always contains new bigrams, so the unsmoothed model fails on exactly the text you care about. [[Smoothing|Changing the counts so that unseen events get a small non-zero probability, taken from the events that were seen.]] fixes this. The simplest method is add-one, or Laplace, smoothing: pretend every possible pair was seen once more than it was.

```viz
{
  "type": "equation",
  "title": "Add-one smoothing: (count + 1) ÷ (count of previous word + V)",
  "intro": "Select each part, then smooth the model for the 45-word canteen text, which has V = 23 different words.",
  "formula": [
    { "t": "P(wₙ | wₙ₋₁)", "id": "p" },
    " = (",
    { "t": "count(wₙ₋₁ wₙ)", "id": "pair" },
    " + ",
    { "t": "1", "id": "one" },
    ") / (",
    { "t": "count(wₙ₋₁)", "id": "prev" },
    " + ",
    { "t": "V", "id": "V" },
    ")"
  ],
  "parts": [
    { "id": "p", "name": "P(wₙ | wₙ₋₁)", "meaning": "The smoothed chance of the next word. It is never zero." },
    { "id": "pair", "name": "count(wₙ₋₁ wₙ)", "meaning": "The real count of the pair. It may be zero." },
    { "id": "one", "name": "+ 1", "meaning": "One imaginary extra sighting of every possible pair. This is what lifts zero counts off zero." },
    { "id": "prev", "name": "count(wₙ₋₁)", "meaning": "The real count of the previous word." },
    { "id": "V", "name": "+ V", "meaning": "We added 1 to each of the V possible next words, so the total must grow by V. This keeps the probabilities adding up to 1." }
  ],
  "steps": [
    { "title": "An unseen pair, before", "expr": "P(eat | canteen) = 0 / 3 = 0", "note": "“canteen” is followed by opens, sells and closes, never by eat." },
    { "title": "The same pair, after", "expr": "P(eat | canteen) = (0 + 1) / (3 + 23) = 1/26 ≈ 0.0385", "note": "It is no longer impossible." },
    { "title": "A seen pair, before and after", "expr": "P(eat | students):  1/4 = 0.25  →  (1 + 1) / (4 + 23) = 2/27 ≈ 0.0741", "note": "The seen pair gives up probability to pay for the unseen ones. On a tiny text the drop is large." },
    { "title": "Check that it still sums to 1", "expr": "Σ over all 23 next words = (4 + 23) / (4 + 23) = 1", "note": "The four real counts add to 4, and the 23 added ones add to 23." }
  ],
  "note": "Add-one is the simplest fix and it moves too much probability to unseen events. Better methods exist, but this one is enough to see the idea."
}
```

Notice the price: add-one gave “eat after canteen” a chance, but it also cut “eat after students” from 0.25 to 0.074. ==Smoothing trades some certainty for the ability to say something about new text.==

### Mixing models: backoff and interpolation

A better trick is to mix a precise model with a coarse one. [[Interpolation|Combining the probabilities from several n-gram orders with weights that add up to 1, for example a bigram probability and a unigram probability.]] uses both, and backoff falls back to the shorter model only when the longer one has no counts.

```viz
{
  "type": "equation",
  "title": "Interpolation: P = λ₂ P₂ + λ₁ P₁",
  "intro": "Select each part. The weights say how much to trust the precise bigram and how much the coarse unigram.",
  "formula": [
    { "t": "P", "id": "P" },
    " = ",
    { "t": "λ₂", "id": "l2" },
    " ",
    { "t": "P₂", "id": "P2" },
    " + ",
    { "t": "λ₁", "id": "l1" },
    " ",
    { "t": "P₁", "id": "P1" }
  ],
  "parts": [
    { "id": "P", "name": "P (the mixed probability)", "meaning": "The probability the model finally uses for the next word." },
    { "id": "l2", "name": "λ₂ (weight on the bigram)", "meaning": "How much we trust the bigram estimate. λ₂ + λ₁ = 1." },
    { "id": "P2", "name": "P₂ (bigram probability)", "meaning": "P(wₙ | wₙ₋₁): precise, but it may be zero." },
    { "id": "l1", "name": "λ₁ (weight on the unigram)", "meaning": "How much we trust the plain word frequency." },
    { "id": "P1", "name": "P₁ (unigram probability)", "meaning": "count(wₙ) / (number of words): the overall share of the word, ignoring the context. It is rarely zero for words in the vocabulary." }
  ],
  "steps": [
    { "title": "The unigram", "expr": "P₁(eat) = 1/45 ≈ 0.0222", "note": "“eat” appears once among the 45 words." },
    { "title": "Mix with λ₂ = 0.7, λ₁ = 0.3 for an unseen pair", "expr": "P(eat | canteen) = 0.7 × 0 + 0.3 × 0.0222 ≈ 0.0067", "note": "The bigram part is zero, but the unigram part keeps the total above zero." },
    { "title": "Mix for a seen pair", "expr": "P(eat | students) = 0.7 × 0.25 + 0.3 × 0.0222 = 0.175 + 0.0067 ≈ 0.1817", "note": "Compare 0.0741 with add-one: interpolation keeps most of the real evidence." }
  ],
  "note": "The weights 0.7 and 0.3 were picked for the example. In practice they are chosen on held-out text, as you do in Lab Part 3."
}
```

Notice that interpolation protects the seen pairs better than add-one. Lab Part 3 tries five values of λ and measures which gives the lowest perplexity.

### Another text, and what n-grams cannot do

Here is a second text, the robotics-club text you will use in the lab. It has 199 words and 81 different ones (full stops not counted), so the counts are richer.

```viz
{
  "type": "ngram",
  "title": "A bigram model from the robotics-club text",
  "intro": "Pick “the”. About how many different words follow it, and which is most common? Then pick “and” and generate a few sentences.",
  "text": "The robotics club meets every Friday in the science room. The club has twelve students and two teachers. The students build small robots and the robots follow a black line on the floor. The teachers help the students write code and the students test the code on the robots. Every robot has a motor and a sensor and a small battery. The sensor reads the black line and the motor turns the wheels. The club shares one laptop and the students take turns on the laptop. When a robot fails the students look at the code and the teachers look at the battery. The students learn from every failure and the club gets better every week. In March the club will join a contest in the city. The contest has a track and a timer and a judge. The best robot follows the line without a mistake. The students practise on the track after class and the teachers bring snacks. The robots are not perfect but the students are proud of the robots. After the contest the club will show the robots to the whole school. The younger students will ask questions and the club will answer every question.",
  "start": "the",
  "note": "The text is lower-cased by the figure. “the” appears 41 times with a following word: students 8, club 6, robots 5, teachers 3, code 2, contest 2 and 15 other words once each, so P(students | the) = 8/41 ≈ 0.195."
}
```

Notice that after “the” the model can choose among 21 different words, and the best guess, “students”, has a probability under 0.2. Even with 199 words the table is mostly empty. ==A language model of this kind cannot do four things.== It cannot use words far back: in “The students who arrived late for the robotics club yesterday were …” the verb must agree with “students”, many words earlier. It cannot see that “cat” and “dog” are alike: to a count table they are different symbols, so what it learns about one tells it nothing about the other. It cannot share evidence between “robot” and “robots”. And its table grows with the text and with n. These four limits are exactly what the neural idea of the next sections is designed to fix: a model that stores a short vector for each token, rather than a row of counts for each context, and lets similar tokens share what is learned.

## Tokens and vocabularies {rail=Tokens}

{lede} A neural network can only compute with numbers, so before any model sees a sentence, the text must be cut into pieces and each piece replaced by a whole number. ==A tokenizer cuts text into tokens, and a vocabulary is the table that gives every token its integer ID==. The choice of piece, whole words, single characters or something in between, shapes everything that follows.

### Text, tokens, IDs

Think of the school ID system: every student has an ID number, and the number is only a label. It does not say who is taller, older or better at maths. A [[token|One piece of text that a model reads as a single unit. It can be a whole word, part of a word, a single character or a punctuation mark.]] is like a student and its [[token ID|The whole number that stands for one token in the vocabulary. It is only a label: ID 5584 is not “bigger” or “better” than ID 4365.]] is the number on the card. The [[vocabulary|The fixed table of all the tokens a model knows, with one integer ID for each. The vocabulary size is how many entries it has.]] is the register. Real vocabularies are large: GPT-2’s has 50,257 entries, and BERT’s, another well-known model, has about 30,000.

A token is not always a word. It may be part of a word (“ably”), a space plus a word (“ the”), a full stop, or even one letter. Besides ordinary pieces, vocabularies hold a few [[special tokens|Reserved entries that do not stand for ordinary text, such as a marker for an unknown piece, for padding, or for the beginning or end of a text.]], for example `<unk>` for a piece the tokenizer cannot write, `<pad>` to fill shorter texts up to the same length in a batch, and markers such as `<bos>` and `<eos>` for the beginning and end of a text. GPT-2 uses a single end-of-text marker.

### Three choices: words, characters or something in between

Take the 12-letter word “unbelievably”. A tokenizer could treat it in three ways.

| Unit | Vocabulary size | Length of “unbelievably” | Main problem |
|---|---|---|---|
| Whole words | Very large, and it never stops growing | 1 token | A new, rare or misspelt word is unknown (`<unk>`) |
| Single characters | Tiny: letters, digits, punctuation (under a hundred for English) | 12 tokens | Sequences become long, and each token carries almost no meaning |
| Sub-words | A budget you choose, often tens of thousands | Several, for example un + believ + ably (an example split) | Needs a method to choose the pieces |

The pieces in the last row are an illustration, not the output of a particular tokenizer. Watch the three kinds handle the same inputs:

```viz
{
  "type": "flow",
  "title": "Three ways to cut the same text",
  "modes": [
    {
      "name": "Word-level",
      "tagline": "One token per word",
      "inputLabel": "Text",
      "modelLabel": "Word tokenizer",
      "outputLabel": "Tokens",
      "examples": [
        { "label": "Words in the vocabulary", "input": "the students played", "output": "the | students | played   (3 tokens)" },
        { "label": "A new word", "input": "the students talkers", "output": "the | students | <unk>   (the word “talkers” is lost)" }
      ],
      "note": "Illustrative outputs. A word-level vocabulary needs a separate entry for every form of a word and writes anything else as <unk>."
    },
    {
      "name": "Character-level",
      "tagline": "One token per letter",
      "inputLabel": "Text",
      "modelLabel": "Character tokenizer",
      "outputLabel": "Tokens",
      "examples": [
        { "label": "Any word", "input": "talkers", "output": "t | a | l | k | e | r | s   (7 tokens)" },
        { "label": "A short phrase", "input": "the students", "output": "t | h | e | _ | s | t | u | d | e | n | t | s   (12 tokens with the space)" }
      ],
      "note": "Illustrative outputs. Nothing is ever unknown, but the sequences are long and each letter alone says almost nothing."
    },
    {
      "name": "Sub-word",
      "tagline": "Frequent pieces",
      "inputLabel": "Text",
      "modelLabel": "Sub-word tokenizer (BPE)",
      "outputLabel": "Tokens",
      "examples": [
        { "label": "A common word", "input": "played", "output": "played   (1 token)" },
        { "label": "A new word", "input": "talkers", "output": "talk | er | s   (3 tokens, an example split)" }
      ],
      "note": "Illustrative outputs. Common words stay whole, and rare or new words are built from pieces the vocabulary already has."
    }
  ]
}
```

Notice the middle path. Sub-words keep sequences short for common words and still never lose a new word, because ==a new word can always be spelt out from smaller pieces==.

### Why not whole words?

A word-level vocabulary has four problems.

- **Rare and new words.** Names, slang, “selfie”, a new product name: each is an [[out-of-vocabulary word|A word that is not in the tokenizer’s vocabulary. A word-level tokenizer writes it as an unknown token.]] and shows up as `<unk>` until someone adds it, and then the model must be trained again.
- **Typos.** “recieve” is a different word from “receive”, so it becomes `<unk>`.
- **Word forms.** “play”, “plays”, “played” and “playing” are four separate entries, and the model learns nothing about their shared stem.
- **A huge vocabulary.** The more words you keep, the bigger the table in the model (the Embeddings section shows how many numbers that costs).

### Why not single characters?

A character-level model never meets an unknown word, but its sequences are long. A 600-word essay is roughly 3,600 characters. The model must also learn, from scratch, that “t”, “h”, “e” in a row make a word. Week 7 will show that the cost of reading a sequence with attention grows quickly with its length, so long sequences are expensive. Sub-words are a compromise: short enough to be cheap, small enough to be flexible.

### Watch the trade-off

The figure below learns BPE merges (the next section explains how) from a small corpus of 18 word forms and cuts the sentence “talkers played walking” in three ways. Slide the number of merges from 0 up to 24.

```viz
{
  "type": "vocabtrade",
  "title": "Characters, sub-words or whole words?",
  "intro": "Drag the merge slider from 0 to 24. Watch the BPE row: the number of tokens for the sentence falls, and the unseen word “talkers” is split into pieces instead of being unknown.",
  "corpus": [ ["play", 6], ["plays", 3], ["played", 5], ["playing", 4], ["player", 3], ["players", 2], ["walk", 5], ["walks", 2], ["walked", 4], ["walking", 4], ["walker", 2], ["talk", 4], ["talks", 2], ["talked", 3], ["talking", 3], ["jump", 3], ["jumped", 2], ["jumping", 2] ],
  "text": "talkers played walking",
  "maxMerges": 24,
  "merges": 16,
  "note": "A toy corpus of 18 word forms. The figure counts the end-of-word mark as a symbol. Its vocabulary number counts the pieces still used to write the 18 training words, so it can dip at first (a piece such as “al” disappears once “alk” and then “walk” are built); a real tokenizer keeps every piece it ever merged, so its table only grows.",
  "checks": [
    { "merges": 0, "vocabSize": 18, "textTokens": 23, "unk": 1, "wordVocab": 18 },
    { "merges": 12, "textTokens": 9 },
    { "merges": 16, "vocabSize": 10, "textTokens": 7 },
    { "merges": 24, "vocabSize": 16, "textTokens": 5 }
  ]
}
```

Here is what to read from it. With 0 merges the sentence needs 23 tokens, one per letter plus an end-of-word mark for each of the three words (the figure counts that mark as a symbol), from a base set of 18 symbols. At word level it needs only 3 tokens, but “talkers” is unknown, because the corpus has “talk”, “talks” and “talker” but never “talkers”, and the word vocabulary is the 18 training words. With 16 merges the sentence takes 7 tokens: talk + er + s, play + ed, walk + ing. The unseen word is built from known pieces. With 24 merges it takes 5 tokens. ==More merges give shorter sequences but a bigger set of pieces to learn.== Lab Part 5 draws this same curve for a tokenizer you build.

### Languages, spaces and numbers

A tokenizer is built from some text, so it is best at the language of that text. A tokenizer built mostly from English tends to cut a language it saw less of into more, smaller pieces. “Good morning” is two words in English; “Magandang umaga” in Tagalog and “Maayong buntag” in Cebuano are two words each too, but the number of tokens can differ widely between tokenizers. More tokens for the same sentence is a ==cost and quality issue==: the text takes more of the model’s limited memory, can cost more to process, and the model sees each word in less coherent pieces. This is a fairness question, and the reason a community such as yours should measure before trusting a tokenizer. You do exactly that in Lab Parts 4 and 7, with measured numbers rather than guesses.

Spaces and capitals matter too. In many tokenizers the space is part of the token that follows it, so “ the” (with a space) and “the” (without), and “The” with a capital, are different tokens with different IDs. This is the leading-space note from Week 1. Numbers and program code can also split in surprising ways: a long number may become several pieces, and the pieces are not always the digits you would choose. ==Always look at the actual tokens before you draw conclusions.==

### How many tokens is that text?

Models have a fixed-size window of tokens they can read at once, so you often need to convert between words and tokens. Week 1 gave the rule of thumb for English: ==one token is about three-quarters of a word==. Select the parts and use it.

```viz
{
  "type": "equation",
  "title": "Tokens from words: T = W × r",
  "intro": "Select each symbol, then convert an essay and a context window.",
  "formula": [
    { "t": "T", "id": "T" },
    " = ",
    { "t": "W", "id": "W" },
    " × ",
    { "t": "r", "id": "r" }
  ],
  "parts": [
    { "id": "T", "name": "T (tokens)", "meaning": "How many tokens the tokenizer produces for the text." },
    { "id": "W", "name": "W (words)", "meaning": "How many words the text has, counted by splitting on spaces." },
    { "id": "r", "name": "r (tokens per word)", "meaning": "The average number of tokens per word. For English, one token is about ¾ of a word, so r ≈ 1/0.75 ≈ 1.33. Other languages and kinds of text can have a larger r." }
  ],
  "steps": [
    { "title": "Tokens per word from the rule", "expr": "r = 1 / 0.75 ≈ 1.33", "note": "Three tokens cover about four words." },
    { "title": "An English essay", "expr": "W = 600:  T = 600 × 1.33 = 800 tokens", "note": "The same as 600 ÷ 0.75." },
    { "title": "A harder text, r = 2", "expr": "W = 600:  T = 600 × 2 = 1,200 tokens", "note": "A text that splits into more pieces uses 50% more tokens for the same number of words." },
    { "title": "How much fits in a window?", "expr": "GPT-2 reads at most 1,024 tokens:  1,024 × 0.75 = 768 English words;  at r = 2:  1,024 ÷ 2 = 512 words", "note": "The same window holds fewer words when r is larger." }
  ],
  "note": "The ¾ rule is an average for ordinary English. Measure r for your own text, as you do in Lab Part 4."
}
```

Notice that ==r is the number to watch when you compare tokenizers==: a lower r means the tokenizer is a better fit for that text.

### Encoding and decoding

A good tokenizer is reversible: you can turn tokens back into the original text. Step through the round trip.

```viz
{
  "type": "pipeline",
  "title": "Encode and decode: text to IDs and back",
  "intro": "Step through the trip. The IDs are made up for the example.",
  "nodes": [
    { "id": "text", "label": "Text", "sub": "unbelievably" },
    { "id": "pieces", "label": "Pieces", "sub": "split" },
    { "id": "ids", "label": "IDs", "sub": "look up" },
    { "id": "pieces2", "label": "Pieces", "sub": "look up back" },
    { "id": "text2", "label": "Text", "sub": "joined" }
  ],
  "steps": [
    { "title": "1. Split into pieces", "body": "The tokenizer cuts the text into pieces from its vocabulary.", "active": ["text", "pieces"], "example": "unbelievably → un | believ | ably (an example split)" },
    { "title": "2. Look up each ID", "body": "Each piece is replaced by its integer ID from the vocabulary table. This list of numbers is what the model receives.", "active": ["pieces", "ids"], "example": "[403, 6188, 1346] (made-up IDs)" },
    { "title": "3. The model works on the IDs", "body": "The model never sees letters or pieces, only these numbers. The Embeddings section shows what it does with them.", "active": ["ids"], "example": "[403, 6188, 1346] → … → the next ID" },
    { "title": "4. Decode: IDs back to pieces", "body": "To read the model’s output, each ID is looked up in the same table in the other direction.", "active": ["ids", "pieces2"], "example": "[403, 6188, 1346] → un | believ | ably" },
    { "title": "5. Join the pieces", "body": "The pieces are put together, with any spaces they carry, to give the text back. A byte-level tokenizer can do this for any text.", "active": ["pieces2", "text2"], "example": "un + believ + ably → unbelievably" }
  ],
  "note": "The split and the IDs are illustrative, not from a particular tokenizer."
}
```

Notice that nothing is lost in the round trip: decoding is the exact reverse of encoding. In Lab Part 4 you check this on a real tokenizer and see what a different one loses.

## Byte-pair encoding (BPE) {rail=BPE}

{lede} [[Byte-pair encoding (BPE)|A way to build a sub-word vocabulary from text. Start with single characters, then repeatedly merge the most frequent neighbouring pair into one new symbol.]] answers the question “which pieces should the vocabulary contain?” with a very simple habit: ==find the pair of neighbouring symbols that occurs most often, glue it into one symbol, and repeat==. The list of merges, in the order you made them, is the tokenizer.

### The idea in plain words

Imagine a secretary who must write the same words again and again and wants to save effort. She notices that the two-letter pair “es” comes up most often, so she invents a one-stroke sign for it. Next she sees that her new sign followed by “t” is the commonest pair, so she invents a sign for that too. After enough signs, the commonest words are one stroke each, and she can still spell a word she has never written, because she still knows the single letters. That is BPE. The recipe is:

1. **Start small.** Write every word as separate characters, followed by an end-of-word mark `</w>` (so that “est” at the end of a word is different from “est” in the middle). Keep the count of each word.
2. **Count pairs.** Count every pair of neighbouring symbols, adding up the word counts.
3. **Merge.** Take the most frequent pair (if two are tied, the one seen first) and replace it everywhere by one new symbol. It joins the vocabulary.
4. **Repeat** steps 2 and 3 for M merges, or stop earlier when no pair occurs at least twice.

Sennrich and colleagues adapted this idea to neural machine translation in 2016. The size of the result is easy to compute. Select each symbol.

```viz
{
  "type": "equation",
  "title": "Vocabulary size and tokens per word: |V| = B + M and T = tokens ÷ words",
  "intro": "Select each symbol, then count both for the corpus low ×5, lower ×2, newest ×6, widest ×3 after 12 merges.",
  "formula": [
    { "t": "|V|", "id": "V" },
    " = ",
    { "t": "B", "id": "B" },
    " + ",
    { "t": "M", "id": "M" },
    ";     ",
    { "t": "T", "id": "T" },
    " = ",
    { "t": "total tokens", "id": "tot" },
    " / ",
    { "t": "total words", "id": "words" }
  ],
  "parts": [
    { "id": "V", "name": "|V| (vocabulary size)", "meaning": "The number of entries in the final table. Every symbol ever created stays in the table, so merges only add to it." },
    { "id": "B", "name": "B (base symbols)", "meaning": "The single characters of the corpus plus the end-of-word mark. For a byte-level tokenizer B is 256." },
    { "id": "M", "name": "M (merges)", "meaning": "How many merges you chose to learn. Each merge adds exactly one new symbol." },
    { "id": "T", "name": "T (tokens per word)", "meaning": "The average number of tokens a word needs in the training corpus. Lower means shorter sequences." },
    { "id": "tot", "name": "total tokens", "meaning": "Add up, over all words, the number of symbols in the word times the word’s count." },
    { "id": "words", "name": "total words", "meaning": "The sum of the word counts. Here 5 + 2 + 6 + 3 = 16." }
  ],
  "steps": [
    { "title": "Base symbols", "expr": "B = l, o, w, e, r, n, s, t, i, d, </w> = 11", "note": "Ten letters and the end-of-word mark." },
    { "title": "Vocabulary after 12 merges", "expr": "|V| = 11 + 12 = 23", "note": "Eleven base symbols and twelve new ones." },
    { "title": "Tokens before any merge", "expr": "low 4 × 5 + lower 6 × 2 + newest 7 × 6 + widest 7 × 3 = 20 + 12 + 42 + 21 = 95", "note": "Each word has its letters plus the end mark." },
    { "title": "Tokens per word before", "expr": "T = 95 / 16 ≈ 5.94", "note": "Almost six tokens for an average word." },
    { "title": "Tokens after 12 merges", "expr": "low 1 × 5 + lower 4 × 2 + newest 1 × 6 + widest 1 × 3 = 5 + 8 + 6 + 3 = 22,  so T = 22 / 16 = 1.375", "note": "Three of the four words are now a single token." }
  ],
  "note": "The arithmetic is exact. After 12 merges the word “lower” is still low + e + r + end mark, because the pairs inside it occur only twice, less often than the merges already made."
}
```

Notice the two numbers move in opposite directions: ==each merge makes the table one entry bigger and the text shorter==. The rest of this section is about picking how far to go.

### A hand-worked trace

Start with the four words and their counts. Counting every adjacent pair, weighted by the word counts, gives the eight most frequent:

| Pair | Where it occurs | Count |
|---|---|---|
| e s | newest (6) and widest (3) | 9 |
| s t | newest (6) and widest (3) | 9 |
| t </w> | newest (6) and widest (3) | 9 |
| w e | lower (2) and newest (6) | 8 |
| l o | low (5) and lower (2) | 7 |
| o w | low (5) and lower (2) | 7 |
| n e | newest (6) | 6 |
| e w | newest (6) | 6 |

Three pairs tie at 9. We take the one seen first, “e s”. This is what happens in the first four merges. The right-hand column is the total number of tokens in the corpus, which falls by exactly the count of each merge.

| Merge | New symbol | Count | The words afterwards | Total tokens |
|---|---|---|---|---|
| start | | | low = l o w </w>, lower = l o w e r </w>, newest = n e w e s t </w>, widest = w i d e s t </w> | 95 |
| 1 | e + s → es | 9 | newest = n e w es t, widest = w i d es t | 86 |
| 2 | es + t → est | 9 | newest = n e w est, widest = w i d est | 77 |
| 3 | est + </w> → est</w> | 9 | newest = n e w est</w>, widest = w i d est</w> | 68 |
| 4 | l + o → lo | 7 | low = lo w, lower = lo w e r | 61 |

After merge 1, the pair “es t” has count 9 and so does “t </w>”; the first seen is “es t”, so merge 2 builds “est”. After merge 3, the best pair is “l o” with 7 (it ties with “o w” and is seen first). Check your hand count against the figure: drag the slider and read the last merge and its count at every step.

```viz
{
  "type": "bpe",
  "title": "Learn BPE merges from low, lower, newest, widest",
  "intro": "Drag the slider to replay the merges. Read the last merge and its count at each step, then type “lowest”, “newer” and a nonsense word such as “xyz”.",
  "corpus": [ ["low", 5], ["lower", 2], ["newest", 6], ["widest", 3] ],
  "maxMerges": 12,
  "tryWord": "lowest",
  "note": "A classic toy corpus of four words. Real tokenizers learn tens of thousands of merges from billions of words."
}
```

Notice that the first three merges build the ending “est</w>” out of the letters, because newest and widest share it. After 12 merges, three of the four words are single symbols. ==The merges list is the tokenizer; nothing else needs to be stored.==

### Encoding a word you have never seen

The corpus never contained “lowest”. To encode it, write it as characters plus `</w>` and apply the merges in the order they were learned, each one wherever it fits. For “lowest”: start with l o w e s t </w>. Merge 1 (e + s) gives l o w es t </w>. Merge 2 (es + t) gives l o w est </w>. Merge 3 gives l o w est</w>. Merge 4 (l + o) gives lo w est</w>. Merge 5 (lo + w) gives low est</w>. No later merge fits, so “lowest” is **low | est</w>**: two tokens, both known, although the word is new. “newer” becomes new | e | r | </w> (four tokens) after merges 6 and 7. A word made of letters never seen at all, like “xyz”, falls back to single characters; in our toy corpus the letter x never appeared, so a strict tokenizer would have to write it as `<unk>`. Byte-level BPE, below, removes that problem.

Decoding is the reverse: join the pieces and turn `</w>` back into a space. Now a second corpus with different pieces. Predict the first merge before you open the figure; it is the pair “u g”.

```viz
{
  "type": "bpe",
  "title": "Learn BPE merges from hug, pug, pun, bun, hugs",
  "intro": "Step through the merges. Which pair is merged first, and with what count? Then type “bug”: it never appears in the corpus.",
  "corpus": [ ["hug", 10], ["pug", 5], ["pun", 12], ["bun", 4], ["hugs", 5] ],
  "maxMerges": 11,
  "tryWord": "bug",
  "note": "A second toy corpus, with different pieces. The first merge is u + g with count 20: hug (10) + pug (5) + hugs (5)."
}
```

The first merge is “u g” with 20 (hug 10, pug 5, hugs 5), ahead of “p u” with 17, “u n” with 16 and “h u” with 15. The word “bug” is new, but it becomes b | ug | </w>: the letter b was seen in “bun” and the piece “ug” was learned from the others. Lab Part 5 asks you to write this algorithm yourself in a few lines of Python and compare it with a real tokenizer.

### Byte-level BPE

Real tokenizers face every language, emoji and typo on the internet. GPT-2 uses [[byte-level BPE|BPE that starts from the 256 possible byte values instead of letters, so any text in any language can be written without an unknown token.]]: it starts from bytes, the 256 possible values a computer uses to store text, instead of from letters. Every possible text can be written as bytes, so ==nothing is ever unknown== and `<unk>` is not needed. Its vocabulary of 50,257 entries matches our formula: 256 base symbols, 50,000 merges and one special end-of-text token. The price is that a character outside English may take several bytes, so it may need several tokens before merges join them. WordPiece (used by BERT), Unigram and SentencePiece are other ways to choose the pieces; this lesson teaches BPE.

### How many merges?

There is no single right number, because it is a trade-off, and you can watch it. The figure uses the same corpus. Drag from 0 to 12 merges and read the tokens for “lowest newer widest”.

```viz
{
  "type": "vocabtrade",
  "title": "How many merges is enough?",
  "intro": "Drag the merge slider. Notice when the word “widest”, which is in the corpus, becomes a single token, and what happens to “lowest” and “newer”, which are not.",
  "corpus": [ ["low", 5], ["lower", 2], ["newest", 6], ["widest", 3] ],
  "text": "lowest newer widest",
  "maxMerges": 12,
  "merges": 5,
  "note": "Same toy corpus as the BPE figure. The vocabulary number counts the pieces still in use, so it dips as pieces are absorbed; a real table (11 + merges) only grows.",
  "checks": [
    { "merges": 0, "vocabSize": 11, "textTokens": 20, "unk": 2, "wordVocab": 4 },
    { "merges": 5, "vocabSize": 9, "textTokens": 12 },
    { "merges": 12, "vocabSize": 7, "textTokens": 7 }
  ]
}
```

With 0 merges the three words need 20 tokens. With 5 merges they need 12, with 8 merges 10, and with 12 merges 7: “lowest” is low + est</w>, “newer” is new + e + r + the end mark, and “widest” is a single token. At word level only “widest” is known, so “lowest” and “newer” would be unknown. ==Too few merges leave long sequences made of letters; too many create rare pieces, such as the word “widest” itself, that appear only a few times in training and so are hard to learn well.== Real tokenizers choose a vocabulary size in the tens of thousands as a compromise, and measure the result on their text. Try the sorter below.

```viz
{
  "type": "sorter",
  "title": "Which tokenizer does this?",
  "intro": "Decide whether each outcome belongs to word-level, character-level or BPE tokenisation.",
  "bins": [ { "id": "word", "label": "Word-level" }, { "id": "char", "label": "Character-level" }, { "id": "bpe", "label": "BPE" } ],
  "cards": [
    { "name": "“unbelievably” becomes one token", "answer": "word", "why": "A word-level vocabulary treats every word as one entry, however long it is, if the word is in the vocabulary." },
    { "name": "“unbelievably” becomes twelve tokens", "answer": "char", "why": "There is one token for each of its twelve letters." },
    { "name": "“unbelievably” becomes un + believ + ably (an example split)", "answer": "bpe", "why": "Frequent pieces from the merges are used, and the word is built from them." },
    { "name": "The typo “recieve” becomes an unknown token", "answer": "word", "why": "The misspelling is not in the word list, so it is written as <unk>." },
    { "name": "A vocabulary of well under a hundred symbols that can write anything in English", "answer": "char", "why": "Letters, digits and punctuation are enough to write any English text, so it is tiny, but the sequences are long." },
    { "name": "Learned by merging the most frequent neighbouring pair again and again", "answer": "bpe", "why": "That is the BPE training loop, and the merges list is the tokenizer." },
    { "name": "A six-word sentence is exactly six tokens", "answer": "word", "why": "Each word is one token, so the lengths match." },
    { "name": "Common words stay whole and rare words are split into known pieces", "answer": "bpe", "why": "Frequent strings were merged into single tokens, rare ones were not, and the pieces are always available." },
    { "name": "Each token carries almost no meaning on its own, and sequences are very long", "answer": "char", "why": "A single letter says little, and a text needs many letters." }
  ]
}
```

### Tokenizer quirks to remember

- **The model never sees letters inside a token.** ==It sees the ID of a whole piece.== That is why a language model can often get a question like “how many r’s are in strawberry?” wrong: the word may reach it as a few pieces, not as ten letters. Say “often”, not “always”; you will test this in Lab Part 4.
- **Different tokenizers cut differently.** The same sentence can give different token counts for different models, so counts and costs are not comparable across them.
- **Spaces, capitals and punctuation are part of the pieces.** “ the”, “the” and “The” are different tokens in many tokenizers.
- **Numbers and code split in odd places,** so arithmetic on long numbers is harder than it looks.
- **Names, other languages and emoji** often take more tokens than plain English words.

Notice what these have in common: they all come from the pieces a tokenizer happened to learn, and from nothing the model “decided”. ==When a model behaves oddly with spelling, numbers or a language, check the tokens first.== Lab Part 7 asks you to design and test a tokenizer for a real community.

## Word embeddings: from IDs to meaning {rail=Embeddings}

{lede} A token ID is only a label, and a label cannot tell the model that “cat” is more like “dog” than like “rice”. An [[embedding|A vector of d numbers that stands for one token. The numbers are learned, so tokens used in similar ways end up with similar vectors.]] replaces each ID with a short list of numbers, and ==similar tokens get similar vectors==. This is the bridge from the tokenizer you just met to the neural networks of the coming weeks.

### IDs are labels, not meanings

Suppose the vocabulary gives “cat” the ID 5584 and “dog” the ID 4365. The fact that 5584 is larger than 4365 means nothing: it is the same as a student with ID number 5584 being no “bigger” than student 4365. If a network took the ID as an input number, it would think that tokens with close IDs are related, and they are not. We need a representation that has no false order and, ideally, a true sense of similarity.

### One-hot vectors: no similarity at all

The plain way is a [[one-hot vector|A vector as long as the vocabulary that has a 1 in the position of one token and 0 everywhere else.]]. In a vocabulary of V tokens, token number i becomes a list of V numbers with a single 1 in position i. For GPT-2 that is a list of 50,257 numbers, 50,256 of them zero. Because two different tokens have their 1 in different places, their dot product is 0, and so their cosine similarity is 0: ==in one-hot form every word is equally different from every other word==. Compare any two words below.

```viz
{
  "type": "vecsim",
  "title": "One-hot vectors: every pair is equally unrelated",
  "intro": "Choose two different words and read the dot product and the cosine. Then choose the same word twice.",
  "dims": ["cat", "dog", "rice", "king", "queen"],
  "words": [
    { "label": "cat", "v": [1, 0, 0, 0, 0] },
    { "label": "dog", "v": [0, 1, 0, 0, 0] },
    { "label": "rice", "v": [0, 0, 1, 0, 0] },
    { "label": "king", "v": [0, 0, 0, 1, 0] },
    { "label": "queen", "v": [0, 0, 0, 0, 1] }
  ],
  "a": 0,
  "b": 1,
  "note": "One-hot vectors for a five-word vocabulary. Each column is one word of the vocabulary.",
  "checks": [
    { "a": "cat", "b": "dog", "cos": 0, "dot": 0 },
    { "a": "king", "b": "queen", "cos": 0, "dot": 0 },
    { "a": "cat", "b": "cat", "cos": 1, "dot": 1 }
  ]
}
```

Notice that “king” and “queen” score exactly as low as “king” and “rice”: 0. A one-hot vector also ==wastes space==, since almost all of it is zeros.

### Dense embeddings and the embedding matrix

A [[dense vector|A short list of mostly non-zero real numbers, such as 4 or 768 of them, in which every position carries a little information.]] packs the identity of a token into d numbers, with d far smaller than V. In this lesson d = 4 so you can read them; real models use hundreds (GPT-2 small uses 768). The numbers are stored as rows of a table called the [[embedding matrix|A table E with V rows and d columns. Row i is the vector for the token with ID i. Its numbers are learned during training.]] E. Looking up a token means taking the row of its ID, which is the same as multiplying its one-hot vector by E. Select each part, then follow a five-word example.

```viz
{
  "type": "equation",
  "title": "The embedding lookup: e = E[id] = (one-hot vector) × E",
  "intro": "Select each symbol, then look up “dog” in a five-word, three-number embedding matrix.",
  "formula": [
    { "t": "e", "id": "e" },
    " = ",
    { "t": "E", "id": "E" },
    "[",
    { "t": "id", "id": "id" },
    "] = ",
    { "t": "oₜ", "id": "o" },
    " × ",
    { "t": "E", "id": "E" }
  ],
  "parts": [
    { "id": "e", "name": "e (the embedding)", "meaning": "The vector of d numbers that the model uses for this token from now on." },
    { "id": "E", "name": "E (embedding matrix)", "meaning": "A table with V rows and d columns. The numbers in it are parameters, learned like any weights. Its size is V × d." },
    { "id": "id", "name": "id", "meaning": "The token’s integer ID from the vocabulary. It picks the row." },
    { "id": "o", "name": "oₜ (one-hot vector)", "meaning": "The vector of length V with a 1 at position id. Multiplying it by E picks out one row and ignores all others." }
  ],
  "steps": [
    { "title": "The matrix, with V = 5 words and d = 3", "expr": "E rows: cat (id 0) = [0.9, 0.1, 0.3];  dog (1) = [0.8, 0.1, 0.5];  rice (2) = [0.0, 0.9, 0.2];  king (3) = [0.1, 0.0, 0.9];  queen (4) = [0.1, 0.1, 0.8]", "note": "Illustrative numbers. Row i belongs to the token with ID i." },
    { "title": "Look up “dog” by its ID", "expr": "e = E[1] = [0.8, 0.1, 0.5]", "note": "Just read row 1." },
    { "title": "The same with the one-hot vector", "expr": "[0, 1, 0, 0, 0] × E = 0×[0.9, 0.1, 0.3] + 1×[0.8, 0.1, 0.5] + 0×[0.0, 0.9, 0.2] + 0×[0.1, 0.0, 0.9] + 0×[0.1, 0.1, 0.8] = [0.8, 0.1, 0.5]", "note": "Every other row is multiplied by 0 and disappears. Real code reads the row directly, because it is much faster." },
    { "title": "Count the parameters", "expr": "V × d = 5 × 3 = 15", "note": "Fifteen numbers to learn for this tiny vocabulary." },
    { "title": "A real model", "expr": "GPT-2 small: 50,257 × 768 = 38,597,376 ≈ 38.6 million", "note": "The token embedding matrix alone has about 38.6 million parameters, learned during training." }
  ],
  "note": "The five rows are made up. In a trained model the numbers come from learning, and the rows of similar tokens come out close."
}
```

Notice the size: ==an embedding matrix has V × d entries, and every one of them is a parameter the network must learn==. The bigger the vocabulary or the vector, the more parameters. Lab Part 6 builds a small matrix and checks that the lookup equals the one-hot multiplication.

### Embeddings are learned, not designed

Nobody types the numbers of E. ==At the start they are random.== The matrix is the first layer of the network, and it is trained exactly as every other weight is trained, by backpropagation (Week 3): the network predicts, the loss measures the error, and the gradient moves each row a little. After millions of steps, tokens that help predict the same things end up with similar rows. ==The model discovers the geometry for itself.==

### You shall know a word by the company it keeps

Why would that work? The linguist J. R. Firth wrote in 1957 that ==you shall know a word by the company it keeps==. Look at the contexts of “cat” and “dog”: “the cat chased the mouse”, “the dog chased the cat”, “the cat slept in the garden”, “the dog slept in the garden”. They keep very similar company. “Rice” keeps different company: “she ate rice in the canteen”. If a network is trained to predict the words around each word, as word2vec does, then two words with similar neighbours must be given similar vectors, or the predictions would come out different. In one line, the training goal is: *for each word, predict the words that appear next to it*. A related, simpler route is to count how often words appear together and compress the table of counts; Lab Part 6 does both on a toy corpus of animals, drinks, royalty and places and compares the two.

### Measuring similarity: the cosine

How do we compare two vectors? Two measures are common. The [[dot product|Multiply two vectors position by position and add the results.]] multiplies the vectors position by position and adds the results; it grows when vectors point the same way and when they are long. The [[cosine similarity|A number from −1 to 1 that compares the directions of two vectors: the dot product divided by the product of their lengths.]] divides by the lengths and keeps only the direction. Select each part and follow two examples.

```viz
{
  "type": "equation",
  "title": "Cosine similarity: cos(a, b) = a·b ÷ (‖a‖ ‖b‖)",
  "intro": "Select each part, then compute the cosine of two three-number vectors by hand.",
  "formula": [
    { "t": "cos(a, b)", "id": "cos" },
    " = ",
    { "t": "a·b", "id": "dot" },
    " / (",
    { "t": "‖a‖", "id": "na" },
    " × ",
    { "t": "‖b‖", "id": "nb" },
    ")"
  ],
  "parts": [
    { "id": "cos", "name": "cos(a, b)", "meaning": "A number from −1 to 1. Near 1: the vectors point the same way. Near 0: they are unrelated. Near −1: they point in opposite directions." },
    { "id": "dot", "name": "a·b (dot product)", "meaning": "Multiply the two vectors position by position, then add: a₁b₁ + a₂b₂ + … It is positive when the vectors lean the same way." },
    { "id": "na", "name": "‖a‖ (length of a)", "meaning": "The square root of the sum of the squares of a’s entries. It is the vector’s length." },
    { "id": "nb", "name": "‖b‖ (length of b)", "meaning": "The same for b. Dividing by both lengths removes the effect of how long the vectors are." }
  ],
  "steps": [
    { "title": "Two vectors", "expr": "a = (2, 1, 2),  b = (1, 2, 2)", "note": "Made-up numbers chosen so the lengths are whole numbers." },
    { "title": "Dot product", "expr": "a·b = 2×1 + 1×2 + 2×2 = 2 + 2 + 4 = 8", "note": "Multiply position by position, then add." },
    { "title": "Lengths", "expr": "‖a‖ = √(4 + 1 + 4) = 3;  ‖b‖ = √(1 + 4 + 4) = 3", "note": "Both vectors have length 3." },
    { "title": "Cosine", "expr": "cos = 8 / (3 × 3) = 8/9 ≈ 0.889", "note": "Quite similar: they point mostly the same way." },
    { "title": "Length does not matter", "expr": "a = (1, 2, 2) and 2a = (2, 4, 4):  dot = 18, lengths 3 and 6, cos = 18 / 18 = 1", "note": "A vector and its double have cosine exactly 1: only direction counts." }
  ],
  "note": "Cosine similarity compares directions. It does not say that two words mean the same, only that their vectors point the same way."
}
```

Now try it on word vectors. The vectors below have four numbers each and are authored for teaching: the four columns loosely stand for “animal”, “food”, “royal” and “small”, but in a real model the columns have no names.

```viz
{
  "type": "vecsim",
  "title": "Cosine similarity of eight word vectors",
  "intro": "Compare cat with dog, then cat with rice, then king with queen. Read the dot product, the two lengths and the cosine.",
  "dims": ["animal", "food", "royal", "small"],
  "words": [
    { "label": "cat", "v": [0.9, 0.1, 0.0, 0.3] },
    { "label": "dog", "v": [0.8, 0.1, 0.0, 0.5] },
    { "label": "tiger", "v": [0.9, 0.0, 0.1, 0.8] },
    { "label": "rice", "v": [0.0, 0.9, 0.0, 0.2] },
    { "label": "bread", "v": [0.1, 0.8, 0.0, 0.3] },
    { "label": "king", "v": [0.1, 0.0, 0.9, 0.6] },
    { "label": "queen", "v": [0.1, 0.0, 0.8, 0.5] },
    { "label": "prince", "v": [0.2, 0.0, 0.7, 0.3] }
  ],
  "a": 0,
  "b": 1,
  "note": "Illustrative vectors authored for teaching, not learned from text. The three groups (animals, foods, royalty) were put in on purpose.",
  "checks": [
    { "a": "cat", "b": "dog", "cos": 0.97, "dot": 0.88 },
    { "a": "cat", "b": "rice", "cos": 0.17, "dot": 0.15 },
    { "a": "king", "b": "queen", "cos": 1.0, "dot": 1.03 },
    { "a": "king", "b": "prince", "cos": 0.97, "dot": 0.83 }
  ]
}
```

For cat and dog the dot product is 0.88 and the lengths are 0.954 and 0.949, so the cosine is 0.88 / (0.954 × 0.949) = 0.97: nearly the same direction. For cat and rice the dot product is 0.15, the lengths are 0.954 and 0.922, and the cosine is 0.17: nearly unrelated. King and queen score 1.00 (0.9995, to be exact). The whole table of cosines, all pairs at once, shows the three groups as three bright blocks.

```viz
{
  "type": "heatmap",
  "title": "Cosine similarity between all eight words",
  "rows": ["cat", "dog", "tiger", "rice", "bread", "king", "queen", "prince"],
  "cols": ["cat", "dog", "tiger", "rice", "bread", "king", "queen", "prince"],
  "corner": "word ↓ word →",
  "showValues": true,
  "W": [
    [1, 0.97, 0.91, 0.17, 0.32, 0.26, 0.27, 0.36],
    [0.97, 1, 0.98, 0.22, 0.38, 0.37, 0.37, 0.41],
    [0.91, 0.98, 1, 0.14, 0.32, 0.5, 0.5, 0.52],
    [0.17, 0.22, 0.14, 1, 0.98, 0.12, 0.11, 0.08],
    [0.32, 0.38, 0.32, 0.98, 1, 0.2, 0.2, 0.16],
    [0.26, 0.37, 0.5, 0.12, 0.2, 1, 1, 0.97],
    [0.27, 0.37, 0.5, 0.11, 0.2, 1, 1, 0.98],
    [0.36, 0.41, 0.52, 0.08, 0.16, 0.97, 0.98, 1]
  ],
  "note": "Computed from the illustrative vectors above and rounded to two decimals. The matrix is symmetric: cos(a, b) = cos(b, a)."
}
```

Notice the pattern: three blocks along the diagonal (animals, foods, royalty), with small numbers between the groups. Look also at what the numbers do not say: king–queen and king–prince are both about 1, so cosine tells us they are alike, but not how they differ. This is the point of the next map.

### Directions in the space, and their limits

When many words are placed in a space, directions can carry meaning. In the 2-D map below, the step from “man” to “king” is the same arrow as the step from “woman” to “queen”, so king − man + woman lands on queen. Select a word to see its nearest neighbours and try the two analogies.

```viz
{
  "type": "scatter",
  "title": "A 2-D map of word meaning",
  "metric": "cosine",
  "points": [
    { "label": "man", "x": 2, "y": 1, "group": "people" },
    { "label": "woman", "x": 2, "y": 3, "group": "people" },
    { "label": "king", "x": 6, "y": 1, "group": "royalty" },
    { "label": "queen", "x": 6, "y": 3, "group": "royalty" },
    { "label": "uncle", "x": 4, "y": 1, "group": "family" },
    { "label": "aunt", "x": 4, "y": 3, "group": "family" },
    { "label": "cat", "x": 3, "y": -5, "group": "animals" },
    { "label": "dog", "x": 4, "y": -4, "group": "animals" },
    { "label": "horse", "x": 5, "y": -6, "group": "animals" },
    { "label": "rice", "x": -3, "y": -5, "group": "foods" },
    { "label": "bread", "x": -4, "y": -4, "group": "foods" },
    { "label": "soup", "x": -2, "y": -6, "group": "foods" }
  ],
  "analogies": [
    { "a": "king", "b": "man", "c": "woman", "expect": "queen" },
    { "a": "uncle", "b": "man", "c": "woman", "expect": "aunt" }
  ],
  "note": "The coordinates are illustrative, drawn by hand so that the analogies work. Real embeddings come from training, have hundreds of dimensions, and are messier."
}
```

Check by hand: king − man + woman = (6, 1) − (2, 1) + (2, 3) = (6, 3), which is where queen sits. Uncle − man + woman = (4, 1) − (2, 1) + (2, 3) = (4, 3), which is aunt. In this made-up map the arrows are perfectly parallel. ==Real embeddings are not so tidy.== Treat the famous analogies with care:

- **They are cherry-picked.** King − man + woman ≈ queen is famous because it works. Many other relations come out wrong or fuzzy, and the answer is usually found by excluding the three words you started with.
- **Neighbours mirror the text.** An embedding is learned from a text collection, so it can pick up the stereotypes that text contains. Researchers have found that embeddings trained on large collections of text can place some job words closer to one gender word than to another. ==Check an embedding before you rely on it==, especially when it feeds a decision about people.
- **One vector per token, whatever the sentence.** These are [[static embeddings|Embeddings with exactly one vector per token, the same in every sentence.]]. “Bank” in “river bank” and “bank account” gets the same vector. In Week 7, attention lets a vector depend on the other words around it; the result is called a [[contextual embedding|A vector for a token that depends on the other words around it, so the same word gets different vectors in different sentences.]]. For now, only remember the difference.

Lab Part 6 tests these analogies on a toy corpus and reports honestly how many of them work.

### Where the embedding sits in the model

Finally, put the pieces in order. The embedding lookup is the first step the model takes after the tokenizer, and everything after it works with vectors.

```viz
{
  "type": "pipeline",
  "title": "From a token ID to a probability",
  "intro": "Step through the path of one token. The attention step is named only; you build it in Week 7.",
  "nodes": [
    { "id": "id", "label": "Token ID", "sub": "from the tokenizer" },
    { "id": "emb", "label": "Embedding", "sub": "row of E" },
    { "id": "att", "label": "Attention", "sub": "Week 7" },
    { "id": "logits", "label": "Logits", "sub": "one score per token" },
    { "id": "soft", "label": "Softmax", "sub": "probabilities" }
  ],
  "steps": [
    { "title": "1. A token ID arrives", "body": "The tokenizer has turned the text into integers. The model starts from one ID per token.", "active": ["id"], "example": "dog → ID 1" },
    { "title": "2. Look up its embedding", "body": "The ID selects one row of the embedding matrix E, a vector of d numbers that stands for the token.", "active": ["id", "emb"], "example": "E[1] = [0.8, 0.1, 0.5]" },
    { "title": "3. Mix with the other tokens (next week)", "body": "In a Transformer, attention lets each vector take in information from the other tokens of the sentence. We only name this step here.", "active": ["emb", "att"], "example": "“bank” near “river” gets a different vector from “bank” near “money”" },
    { "title": "4. Score every token", "body": "The model turns the final vector into one score, called a logit, for each of the V tokens of the vocabulary.", "active": ["att", "logits"], "example": "V = 50,257 scores for GPT-2" },
    { "title": "5. Softmax", "body": "Softmax (Week 1) turns the scores into probabilities that add up to 1. The next token is picked from them.", "active": ["logits", "soft"], "example": "rice 0.25, fish 0.20, …" }
  ],
  "note": "The vectors and numbers in the examples are illustrative."
}
```

Notice that the model begins and ends with one entry per token: V rows going in (the embedding matrix) and V scores coming out (the logits). Test yourself on the difference between the two kinds of vector.

```viz
{
  "type": "sorter",
  "title": "One-hot or dense?",
  "intro": "Decide which representation each statement describes.",
  "bins": [ { "id": "onehot", "label": "One-hot" }, { "id": "dense", "label": "Dense embedding" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "A vector as long as the vocabulary, with a single 1", "answer": "onehot", "why": "That is the definition of a one-hot vector." },
    { "name": "Any two different words have cosine similarity 0", "answer": "onehot", "why": "Their 1s are in different places, so the dot product is zero." },
    { "name": "Learned during training by backpropagation", "answer": "dense", "why": "The rows of the embedding matrix are weights that gradient descent updates. A one-hot vector needs no learning." },
    { "name": "A few hundred numbers per token in a real model", "answer": "dense", "why": "Real embeddings have a dimension d of the order of hundreds, far smaller than V." },
    { "name": "Words used in similar ways end up with similar vectors", "answer": "dense", "why": "Similarity is what the learning produces. One-hot vectors cannot show it." },
    { "name": "Each token is a list of numbers a model can multiply", "answer": "both", "why": "Both are vectors of numbers, so both can go through a matrix multiplication. In fact, one-hot times E gives the dense vector." },
    { "name": "Every vector has as many entries as the vocabulary has tokens", "answer": "onehot", "why": "The length of a one-hot vector is V. A dense vector has d entries with d much smaller than V." },
    { "name": "Stands for exactly one token of the vocabulary", "answer": "both", "why": "Each token has its own one-hot vector and its own row in E, so both representations identify one token." }
  ]
}
```

Notice that the two are connected: ==a lookup in E is a one-hot vector multiplied by E==. The dense vector is where learning happens. One last timeline puts the tokens and the vectors in a single story.

```viz
{
  "type": "timeline",
  "title": "How pieces and vectors arrived",
  "events": [
    { "year": "1957", "title": "The company a word keeps", "text": "J. R. Firth wrote that you shall know a word by the company it keeps. The idea that meaning comes from context is the root of word embeddings." },
    { "year": "1994", "title": "BPE as a compression method", "text": "Philip Gage described byte-pair encoding as a way to compress data by replacing the most frequent pair of bytes with a new symbol." },
    { "year": "2003", "title": "Neural language model", "text": "Bengio and colleagues learned a vector for every word as part of a network that predicts the next word." },
    { "year": "2013", "title": "word2vec", "text": "Mikolov and colleagues trained word vectors quickly from very large text and showed that directions carry meaning, such as king − man + woman ≈ queen." },
    { "year": "2014", "title": "GloVe", "text": "Pennington, Socher and Manning built word vectors from global word co-occurrence counts, another well-known way to get embeddings." },
    { "year": "2016", "title": "BPE for translation", "text": "Sennrich and colleagues used BPE to split rare words into sub-words in neural machine translation." },
    { "year": "2018", "title": "SentencePiece and BERT", "text": "Kudo and Richardson released SentencePiece, a tokenizer that works on raw text. Devlin and colleagues introduced BERT, which uses a WordPiece vocabulary of about 30,000 pieces." },
    { "year": "2019", "title": "GPT-2 and byte-level BPE", "text": "GPT-2 started BPE from bytes, so any text can be encoded, with a vocabulary of 50,257 tokens." }
  ]
}
```

## Key takeaways

- **A language model scores the next token.** ==It gives P(next | previous words), the chain rule multiplies these into a sentence probability, and a lower perplexity PP = P^(−1/n) means the text fits the model better==; language models predict, they do not understand. Three eras built them: hand-written rules, counted n-grams, and neural networks with learned vectors.
- **N-gram models count, and they run out of road.** The bigram probability is count(pair) ÷ count(previous word), but ==V^n possible n-grams means most are never seen==, one zero kills the chain-rule product, and add-one smoothing (count + 1) ÷ (count(prev) + V) or interpolation repairs it at a price.
- **Text becomes tokens, and tokens become IDs.** ==Sub-word tokens sit between words (huge vocabulary, unknown words) and characters (long sequences)==; a vocabulary is a table of pieces and IDs, and the same text can need very different numbers of tokens in different languages and tokenizers.
- **BPE learns the pieces by merging.** Start from characters plus an end mark, repeatedly ==merge the most frequent neighbouring pair==, and stop after M merges; the vocabulary is B + M, tokens per word falls as merges grow, and a new word is encoded by applying the merges in the order learned.
- **An embedding gives each token a vector.** ==The lookup e = E[id] picks a row of a V × d matrix whose V·d numbers are learned==, similar tokens get similar vectors, cosine similarity cos(a, b) = a·b ÷ (‖a‖‖b‖) measures it, and analogies, static vectors and biases show both the power and the limits.

<!-- LAB-START -->
## Lab: Imagine how a machine reads, then build the reader {rail=Laboratory}

This lab is called *Imagine how a machine reads, then build the reader*. Week 6 looks inside the first step of every language model: how text becomes numbers. You first put yourself in the place of a machine that has never seen language (what would you count? what would your reading units be?), and then ==build, step by step and from scratch, a counting language model, a tiny neural one, real tokenizers, a byte-pair-encoding tokenizer written by hand, and word embeddings learned from a toy corpus, and finish by designing a tokenizer for a real community==. It has seven parts, and every part begins with an **Imagine first** moment: you write a prediction before you run any code and compare it with what really happens. Each part points back to a section of the lesson and to one of its figures. There are no model downloads: the only things ever fetched are the small tokenizer files of `gpt2` and `bert-base-uncased` (a few hundred kilobytes), and everything runs in seconds to a couple of minutes on a CPU. Time: about 3 hours. This page contains the setup and the first three parts; the other four parts, the deliverables and the marking scheme follow in the same lab.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cell, run a smoke test | 10 |
| 1 | Be a language model: count and predict | 25 |
| 2 | From rules to counts to neural: three tiny language models | 25 |
| 3 | The limits of n-grams: sparsity | 20 |
| 4 | Tokenisation in the real world | 25 |
| 5 | Build a BPE tokenizer from scratch | 30 |
| 6 | Word embeddings from scratch | 30 |
| 7 | Design a tokenizer for a community, and reflect | 15 |

The total is 180 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week06` and a notebook `lab6.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`words`, `bigram`, `p_addone`, `perplexity`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cell and then the blocks again from the top.
2. Do not install packages: PyTorch, NumPy, Matplotlib and `transformers` are already installed. Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook. Parts 1 to 3 need no downloads at all. From Part 4 on, `load_tokenizer("gpt2")` and `load_tokenizer("bert-base-uncased")` read the two small tokenizer files from the offline Hugging Face cache on the Spark (ask the instructor if the smoke test below reports that they are missing). In Google Colab the same two calls download them once, a few hundred kilobytes in total. No model weights are ever downloaded.
3. A CPU is enough. Most blocks finish in a second or two; the slowest ones (training the small networks) take a few seconds to about a minute. The GPU of the Spark is shared by the whole class and gives no advantage on models this small, so leave it alone. The helper cell prints which device it found.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste this into the first cell and run it. It fixes the random seed (so your numbers match the ones printed in this lab), defines the empty `RESULTS` dictionary and the `check(...)` function, holds the two short texts used in Parts 1 to 3, and defines the helpers that make a toy corpus, load and show tokenizers, and compare word vectors:

```python
import os, gc, math, random, re, time, warnings
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
QUICK = False
print("Running on:", device)

RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

SCHOOL_TEXT = """
The robotics club meets every Friday in the science room. The club has twelve students and two teachers. The students build small robots and the robots follow a black line on the floor. The teachers help the students write code and the students test the code on the robots. Every robot has a motor and a sensor and a small battery. The sensor reads the black line and the motor turns the wheels. The club shares one laptop and the students take turns on the laptop. When a robot fails the students look at the code and the teachers look at the battery. The students learn from every failure and the club gets better every week. In March the club will join a contest in the city. The contest has a track and a timer and a judge. The best robot follows the line without a mistake. The students practise on the track after class and the teachers bring snacks. The robots are not perfect but the students are proud of the robots. After the contest the club will show the robots to the whole school. The younger students will ask questions and the club will answer every question.
"""
SCHOOL_TEST = """
The club meets in the science room on Friday. The students test the robots on the track and the teachers look at the code. The robot follows the line and the sensor reads the floor. The students are proud of the club.
"""

def words_of(text):
    """Lower-case words of a text, with each full stop kept as its own token '.'."""
    return re.findall(r"[a-z']+|\.", text.lower())

_ANIMALS = ["cat", "dog", "mouse", "bird", "rabbit", "fox"]
_DRINKS = ["tea", "coffee", "milk", "juice", "water", "soda"]
_ROYALS = ["king", "queen", "prince", "princess", "duke", "duchess"]
_PLACES = ["kitchen", "garden", "market", "school", "castle", "kingdom"]

def make_toy_corpus(n=3000, seed=0):
    """A list of n short sentences (lists of words) built from templates about animals, drinks, royalty and places. Words of the same group
    appear in the same kinds of sentences, so their embeddings should end up close together."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        k = r.randrange(5)
        if k == 0:   s = ["the", r.choice(_ANIMALS), r.choice(["chased", "saw", "followed"]), "the", r.choice(_ANIMALS)]
        elif k == 1: s = ["the", r.choice(_ANIMALS), r.choice(["slept", "ran", "played"]), "in", "the", r.choice(["garden", "kitchen", "market"])]
        elif k == 2: s = [r.choice(["she", "he"]), r.choice(["drank", "bought", "liked"]), r.choice(_DRINKS), "in", "the", r.choice(["kitchen", "market", "school"])]
        elif k == 3: s = ["the", r.choice(_ROYALS), r.choice(["ruled", "visited", "protected"]), "the", r.choice(["kingdom", "castle"])]
        else:        s = ["the", r.choice(_ROYALS), "lived", "in", "the", "castle", "and", "drank", r.choice(_DRINKS)]
        out.append(s)
    return out

def load_tokenizer(name):
    """Load a Hugging Face tokenizer (a small file, no model weights). Needs the internet once, or the offline cache on the school's Spark."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)

def show_tokens(tok, text):
    """Print the pieces and ids that tokenizer `tok` makes from `text`, and how many there are."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"{len(ids)} tokens:", pieces)
    print("ids:", ids)
    return ids

def cosine(a, b):
    """Cosine similarity of two vectors."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def nearest(E, vocab, word, k=5):
    """The k words whose vectors (rows of E, one per word in `vocab`) are most cosine-similar to `word`."""
    i = vocab.index(word)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sims = En @ En[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    return [(vocab[j], round(float(sims[j]), 3)) for j in order]
```

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `SCHOOL_TEXT` (215 tokens) and `SCHOOL_TEST` (46 tokens) are two short original texts about a school robotics club: the first is what the language models learn from, the second is new text to test them on;
   - `words_of(text)` returns the lower-case words of a text as a list and keeps each full stop as its own token `.`;
   - `make_toy_corpus(n, seed)` returns `n` short sentences (lists of words) about animals, drinks, royalty and places, with a vocabulary of about 40 words; Part 6 learns word embeddings from it;
   - `load_tokenizer(name)` loads a Hugging Face tokenizer (a small file, no model weights) and `show_tokens(tok, text)` prints the pieces and ids it makes from a text;
   - `cosine(a, b)` is the cosine similarity of two vectors and `nearest(E, vocab, word, k)` lists the `k` words whose vectors (rows of `E`) are closest to the vector of `word`;
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

7. Run this smoke test in the next cell. It checks that the helpers work, that your Matplotlib shows figures in the notebook, and whether the two tokenizers can be loaded:

```python
ws = words_of(SCHOOL_TEXT)
print("SCHOOL_TEXT has", len(ws), "tokens and", len(set(ws)), "different words | SCHOOL_TEST has", len(words_of(SCHOOL_TEST)), "tokens")
toy = make_toy_corpus(5, seed=0)
print("toy sentences:", [" ".join(s) for s in toy[:2] ])
print("cosine of (1, 0) and (1, 1):", round(cosine([1, 0], [1, 1]), 3))
for name in ["gpt2", "bert-base-uncased"]:
    try:
        tok = load_tokenizer(name)
        print(f"{name}: loaded, vocabulary size {len(tok):,}")
    except Exception as e:
        print(f"{name}: could not be loaded ({type(e).__name__}). On the Spark ask the instructor about the offline cache; in Colab check your internet.")
plt.figure(figsize=(3, 2)); plt.plot([0, 1, 2], [0, 1, 4], "o-"); plt.title("figures work"); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
SCHOOL_TEXT has 215 tokens and 82 different words | SCHOOL_TEST has 46 tokens
toy sentences: ['the princess ruled the castle', 'the princess lived in the castle and drank juice']
cosine of (1, 0) and (1, 1): 0.707
gpt2: loaded, vocabulary size 50,257
bert-base-uncased: loaded, vocabulary size 30,522
```

   You should see `Running on: cpu` after the first cell. The smoke test then shows the length of the two texts (215 and 46 tokens), the first two toy sentences, the cosine of (1, 0) and (1, 1) (0.707, which is the cosine of 45 degrees), the vocabulary sizes of the two tokenizers (`gpt2` has 50,257 entries and `bert-base-uncased` has 30,522), and a small figure. If a tokenizer cannot be loaded you get a clear message instead of an error, and Parts 1 to 3 still work; ask the instructor before Part 4.

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cell and the earlier blocks again, in order.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the block ends with `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42`, and that you did not run a training block twice in the same variable. The blocks that train a network set their own seed before building it, but small differences in the last digit can appear between PyTorch versions; the stated ranges and the checks are tolerant, the exact digits are not.
- A tokenizer cannot be loaded: on the Spark the files come from the offline cache, so ask the instructor; in Colab check that the runtime is connected to the internet.
- A cell seems frozen: stop it (the square button), restart the kernel and run again from the top. If a cell takes more than two minutes, you probably changed a loop count or a data size.

### Part 1. Be a language model: count and predict (25 min)

*Lesson sections: "Language models" and "N-grams".* Every chatbot you have used does one basic job: given the words so far, it says which word is likely to come next. In this part you do that job with the simplest tool there is, counting. You count the words of a short text about a school robotics club, count which word follows which, turn the counts into probabilities, score whole sentences with the chain rule, measure how surprised the model is by new text (perplexity), repair the model when it breaks (add-one smoothing), and let it write. At the end you list what counting adjacent words can never know. These are the same ideas as the lesson's n-gram figure, only now the numbers are yours.

The text is `SCHOOL_TEXT` (215 tokens, about a school robotics club) and the "new" text is `SCHOOL_TEST` (46 tokens, a different set of sentences about the same club). Both come from the setup cell. A *token* here is one lower-case word, or one full stop; `words_of` splits a text that way.

**Imagine first.** Put yourself in the place of a machine that has only ever seen the robotics-club text. Someone writes "the" and asks you for the next word. Before you run anything, write two guesses. (1) Which word do you think follows "the" most often, and what share of the time? (2) If a brand-new sentence appears that you have never seen, can your counts ever give it a probability above zero? Answer `"zero"` or `"positive"`. You will compare your guesses with the measured numbers in Steps 1.2, 1.3 and 1.4.

```python
# Imagine first: what follows "the", and can counting rate a sentence it has never seen?
my_next_word = None       # your guess for the most common word after "the", e.g. "club"
my_next_share = None      # your guess for its share, between 0 and 1
my_unseen_sentence = None # "zero" or "positive"
```

1. **Step 1.1. Read the text and count the words.** A language model starts from a *corpus*, the text it learns from. Before any model, look at the corpus like a statistician. The first block tokenises `SCHOOL_TEXT` with `words_of`, counts the total number of tokens, and counts the *distinct* words, which is the vocabulary size **V** (the same V that appears in every formula in the lesson). The second block counts how often each word occurs and prints the ten most common as a table. The third draws them as a bar chart. Counting is all a unigram model is: a table of word frequencies.

```python
words = words_of(SCHOOL_TEXT)        # the corpus as a list of tokens
vocab = sorted(set(words))            # the distinct words, in alphabetical order
V = len(vocab)
print("total tokens:", len(words), "| distinct words V:", V)
print("first 12 tokens:", words[:12])
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
total tokens: 215 | distinct words V: 82
first 12 tokens: ['the', 'robotics', 'club', 'meets', 'every', 'friday', 'in', 'the', 'science', 'room', '.', 'the']
```

```python
unigram = Counter(words)
print(f"{'rank':>4}  {'word':<10} {'count':>5}  {'share':>6}")
for rank, (w, c) in enumerate(unigram.most_common(10), 1):
    print(f"{rank:>4}  {w:<10} {c:>5}  {c / len(words):>6.1%}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
rank  word       count   share
   1  the           41   19.1%
   2  .             16    7.4%
   3  and           13    6.0%
   4  students      10    4.7%
   5  a             10    4.7%
   6  club           7    3.3%
   7  robots         6    2.8%
   8  every          5    2.3%
   9  teachers       4    1.9%
  10  on             4    1.9%
```

```python
top = unigram.most_common(10)
plt.figure(figsize=(7, 3))
plt.bar([w for w, c in top], [c for w, c in top], color="tab:blue")
plt.title("The 10 most common tokens in SCHOOL_TEXT"); plt.ylabel("count"); plt.xticks(rotation=30)
plt.tight_layout(); plt.show()
```

   You should see `total tokens: 215 | distinct words V: 82`: the text is only 215 tokens long and uses 82 different words. The first tokens are `the robotics club meets every friday in the science room . the`. The table puts **the** first with 41 occurrences (19.1% of all tokens), then the full stop (16), **and** (13), and **students** and **a** (10 each). The bar chart has one tall bar for "the" and then a fast fall: the second bar is less than half as tall. Also notice how many words are rare: 53 of the 82 words occur exactly once. A language model must somehow predict those words, yet it has seen each of them only one time. That is the root of the problems in Steps 1.4 and 3.2.

   Tips and common mistakes:

   - `words_of` keeps each full stop as its own token `.`, so a sentence of five words has six tokens. This is how the model will later learn where sentences end.
   - `set(words)` throws the order away; `sorted(...)` puts it back in a fixed order, which we need later to number the words.
   - Do not confuse *tokens* (every word occurrence) with *types* (distinct words). The first is the length of the text; the second is V.

   Try also (optional):

   - Print the 10 *least* common words. How many words occur exactly once?
   - Change `most_common(10)` to `most_common(25)` and see whether the bars fall off smoothly (they follow a long-tailed curve: a few words are very common, most are rare).

   *Exercise 1.1:* count how many times the word `"robots"` occurs in `words`, store it in `n_robots`, and count how many distinct words occur exactly **once**, store it in `n_once`. Why it matters: the words that occur once are the ones a counting model knows least about; they decide how badly the model suffers on new text.

```python
n_robots = None   # how many times "robots" occurs
n_once = None     # how many distinct words occur exactly once
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_robots is None or n_once is None:
    print("Fill in n_robots and n_once above, then run this cell again.")
else:
    check("n_robots is the count of 'robots'", n_robots == words.count("robots"), "Use words.count('robots') or unigram['robots'].")
    check("n_once is the number of words with count 1", n_once == sum(1 for c in unigram.values() if c == 1), "Loop over unigram.values() and count the ones equal to 1.")
    RESULTS["p1_tokens"] = len(words); RESULTS["p1_V"] = V; RESULTS["p1_n_once"] = n_once
```

   `n_robots` is 6 and `n_once` is 53.

2. **Step 1.2. Count which word follows which.** A unigram table cannot say what comes after "the". For that we count *pairs*. A **bigram** is two neighbouring tokens, and the table of bigram counts is the whole model. The first block builds the counts with `zip(words, words[1:])`, which pairs every token with the one after it, and also counts how often each word appears as the *first* of a pair (its *context count*). The second block lists what follows "the": the counts, and the probabilities you get by dividing each count by the context count. The probabilities of everything that can follow one context must add up to 1, and the block asserts that.

```python
bigram = Counter(zip(words, words[1:]))                 # (previous word, next word) -> how often
context = Counter(w1 for w1, w2 in zip(words, words[1:]))   # how often w1 is followed by anything
print("distinct bigrams:", len(bigram), "| bigram tokens:", sum(bigram.values()))
print("context count of 'the':", context["the"], "| unigram count of 'the':", unigram["the"])
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
distinct bigrams: 161 | bigram tokens: 214
context count of 'the': 41 | unigram count of 'the': 41
```

```python
after_the = Counter({w2: c for (w1, w2), c in bigram.items() if w1 == "the"})
print(f"{'next word':<10} {'count':>5} {'P(next | the)':>14}")
for w2, c in after_the.most_common(8):
    print(f"{w2:<10} {c:>5} {c / context['the']:>14.3f}")
total_p = sum(c / context["the"] for c in after_the.values())
print("number of different words seen after 'the':", len(after_the))
print("all probabilities after 'the' add up to", round(total_p, 6))
assert abs(total_p - 1) < 1e-9
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
next word  count  P(next | the)
students       8          0.195
club           6          0.146
robots         5          0.122
teachers       3          0.073
code           2          0.049
contest        2          0.049
robotics       1          0.024
science        1          0.024
number of different words seen after 'the': 21
all probabilities after 'the' add up to 1.0
```

   You should see `distinct bigrams: 161 | bigram tokens: 214`: the 215 tokens make 214 pairs, but only 161 of them are different (the others are repeats such as "the students"). "the" is followed by **21 different words**. The most common are **students** (8 times, P = 0.195), **club** (6, 0.146) and **robots** (5, 0.122). Compare with your Imagine-first guess: if you guessed students, you were right, but notice that even the winner gets less than one chance in five. The probabilities add up to 1 (the assert passes silently). After "the" the model is quite unsure, because "the" can introduce almost any noun. After a word like "friday" or "science" the row has a single entry and the model is certain.

   Tips and common mistakes:

   - The probability is `count(the, next) / count(the as a context)`. The unigram count of a word can be one larger than its context count when the word is the very last token of the text (no pair starts there). For "the" they happen to be equal.
   - The model has no idea what the words mean. "club" and "students" are just labels in a table.
   - A *conditional* probability P(next | the) is a probability inside one row of the table. Each row sums to 1; the whole table does not.

   Try also (optional):

   - Print the table of what follows "students" and "and". Which context is the most predictable? Which is the most uncertain?
   - Count how many contexts have exactly one possible next word. For those contexts the model is completely sure.

   *Exercise 1.2:* from the counts, compute `p_students_after_the` = P(students | the) and find `best_after_students`, the most common word after `"students"`. Why it matters: this is exactly the lookup a bigram model performs every time it predicts a word.

```python
p_students_after_the = None   # count(the, students) / count(the as context)
best_after_students = None    # a word, e.g. "club"
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if p_students_after_the is None or best_after_students is None:
    print("Fill in both answers above, then run this cell again.")
else:
    truth = bigram[("the", "students")] / context["the"]
    check("P(students | the) is right", abs(p_students_after_the - truth) < 1e-9, "Divide bigram[('the', 'students')] by context['the'].")
    best = max((c, w2) for (w1, w2), c in bigram.items() if w1 == "students")
    check("best_after_students has the highest count", bigram[("students", best_after_students)] == best[0], "Look at all pairs that start with 'students' and keep the most frequent.")
    RESULTS["p1_p_students_after_the"] = round(p_students_after_the, 3)
```

   `p_students_after_the` is 0.195 (8 divided by 41) and `best_after_students` is 'and'.

3. **Step 1.3. The probability of a whole sentence (the chain rule).** With a table of P(next | previous) you can score a whole sentence. The chain rule of Week 1 says P(w1, w2, ..., wn) is P(w1) times P(w2 | w1) times P(w3 | w2) and so on; a bigram model keeps only one word of context, and here we start the product at the second token (we do not model the first word). Multiplying many small numbers underflows, so we add logarithms: ln P = sum of ln P(next | previous). The first block defines `p_bigram` and `sentence_logprob`. The second block scores four sentences: a sentence copied from the training text, a *new* sentence made from familiar word pairs, the same new sentence scrambled, and one sentence from `SCHOOL_TEST`.

```python
def p_bigram(prev, w):
    """Unsmoothed estimate P(w | prev) = count(prev, w) / count(prev as a context). Zero if the pair was never seen."""
    return bigram[(prev, w)] / context[prev] if context[prev] else 0.0

def sentence_logprob(sentence, prob):
    """Chain rule with one word of context: the sum of ln P(w_i | w_i-1) over the bigrams of the sentence.
    Returns (log-probability, number of bigrams); the log-probability is -inf if any bigram has probability 0."""
    ws = words_of(sentence)
    lp = 0.0
    for a, b in zip(ws, ws[1:]):
        p = prob(a, b)
        if p == 0:
            return float("-inf"), len(ws) - 1
        lp += math.log(p)
    return lp, len(ws) - 1
```

```python
new_sentence = "The teachers look at the robots and the students test the code on the robots."
scrambled = " ".join(reversed(words_of(new_sentence)[:-1])) + " ."
test_sentence = "The robot follows the line and the sensor reads the floor."
tests = [("copied from the text", "The sensor reads the black line and the motor turns the wheels."),
         ("new, familiar pairs", new_sentence),
         ("the same, scrambled", scrambled),
         ("from SCHOOL_TEST", test_sentence)]
for label, s in tests:
    lp, n = sentence_logprob(s, p_bigram)
    p_txt = f"{math.exp(lp):.2e}" if lp > float("-inf") else "0"
    lp_txt = f"{lp:.2f}" if lp > float("-inf") else "-inf"
    print(f"{label:<22} bigrams {n:>2}  ln P = {lp_txt:>8}  P = {p_txt}")
lp_copy, _ = sentence_logprob(tests[0][1], p_bigram)
lp_new, _ = sentence_logprob(new_sentence, p_bigram)
lp_scr, _ = sentence_logprob(scrambled, p_bigram)
assert lp_scr < lp_new          # the scrambled sentence is less likely (here: impossible)
print("scrambled sentence:", scrambled)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
copied from the text   bigrams 12  ln P =   -18.52  P = 9.07e-09
new, familiar pairs    bigrams 15  ln P =   -19.64  P = 2.95e-09
the same, scrambled    bigrams 15  ln P =     -inf  P = 0
from SCHOOL_TEST       bigrams 11  ln P =     -inf  P = 0
scrambled sentence: robots the on code the test students the and robots the at look teachers the .
```

   You should see four lines. The sentence **copied from the text** has 12 bigrams, ln P = -18.52 and P = 9.07e-09: a very small number, as every sentence probability is, because each of the 12 factors is below 1. The **new sentence** (it is not in the text, but all of its pairs are) gets ln P = -19.64, P = 2.95e-09. So counting adjacent pairs *can* score a sentence it has never seen, as long as every pair is familiar; the new sentence is slightly less likely than the copy only because it is three bigrams longer. The **scrambled** sentence contains the same words in the wrong order, and one of its pairs ("robots the", for example) was never seen, so ln P is `-inf` and P is 0; the assert `lp_scr < lp_new` passes. Finally the sentence **from SCHOOL_TEST** also gets 0, because of a single pair, ("the", "robot"): the training text says "robots" but never "the robot". Compare with your Imagine-first guess: unseen sentences are not always impossible, but one unseen pair is enough to make them so.

   Tips and common mistakes:

   - `math.log` is the natural logarithm ln. A probability of 1e-14 has ln P of about -32: far easier to read and to add.
   - A model with zero in one factor gives zero for the whole product. This is not a bug in the code, it is a property of unsmoothed counts, and it is the subject of Step 1.4.
   - The scrambled sentence uses exactly the same words as the new sentence. Only the order is different, and order is what a bigram model measures.

   Try also (optional):

   - Write your own sentence from words in the text and see if it gets a positive probability.
   - Divide ln P by the number of bigrams to get an average per word. Which sentence is the most predictable word by word?

   *Exercise 1.3:* compute by hand (with `p_bigram`) the probability of the three-word sequence `"the students test"`, which is P(students | the) times P(test | students). Store the probability in `p_hand` and its natural logarithm in `lp_hand`. Then compare with `sentence_logprob`. Why it matters: `sentence_logprob` is nothing more than this multiplication repeated along the sentence.

```python
p_hand = None    # P(students | the) * P(test | students)
lp_hand = None   # math.log(p_hand)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if p_hand is None or lp_hand is None:
    print("Fill in p_hand and lp_hand above, then run this cell again.")
else:
    lp_fn, _ = sentence_logprob("the students test", p_bigram)
    check("lp_hand matches sentence_logprob", abs(lp_hand - lp_fn) < 1e-9, "Multiply the two bigram probabilities, then take math.log.")
    check("p_hand is the exponential of lp_hand", abs(math.exp(lp_hand) - p_hand) < 1e-12, "p_hand and lp_hand must describe the same number.")
    RESULTS["p1_p_the_students_test"] = round(p_hand, 4)
```

   `p_hand` is 0.0195 (that is 0.195 times 0.100) and `lp_hand` is -3.94.

4. **Step 1.4. Perplexity, and why the model breaks.** A single probability depends on the length of the sentence, so to compare texts we use **perplexity**: the average surprise per token. It is exp of the negative mean log-probability, which is the same as P raised to the power minus one over n (the convention of Week 1). A perplexity of 10 means the model is as unsure as if it had to choose among 10 equally likely words at every step. The first block defines `perplexity(text, prob)` and tests it on the training text and on `SCHOOL_TEST`. The second block lists the bigrams of `SCHOOL_TEST` that the model has never seen, which is the cause of the failure you will see.

```python
def perplexity(text, prob):
    """exp of the average negative ln P(w_i | w_i-1) over all bigrams of `text`; infinite if one bigram has probability 0."""
    ws = words_of(text)
    lp, n = 0.0, 0
    for a, b in zip(ws, ws[1:]):
        p = prob(a, b)
        if p == 0:
            return float("inf")
        lp += math.log(p)
        n += 1
    return math.exp(-lp / n)

ppl_train_mle = perplexity(SCHOOL_TEXT, p_bigram)
ppl_test_mle = perplexity(SCHOOL_TEST, p_bigram)
print("perplexity of the unsmoothed bigram model on SCHOOL_TEXT:", round(ppl_train_mle, 2))
print("perplexity of the unsmoothed bigram model on SCHOOL_TEST:", ppl_test_mle)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
perplexity of the unsmoothed bigram model on SCHOOL_TEXT: 3.18
perplexity of the unsmoothed bigram model on SCHOOL_TEST: inf
```

```python
test_words = words_of(SCHOOL_TEST)
test_pairs = list(zip(test_words, test_words[1:]))
unseen_pairs = [p for p in test_pairs if bigram[p] == 0]
print("bigrams in SCHOOL_TEST:", len(test_pairs), "| never seen in SCHOOL_TEXT:", len(unseen_pairs))
for a, b in unseen_pairs:
    print(f"  ({a!r}, {b!r}): context {a!r} was seen {context[a]} times, but never followed by {b!r}")
print("words of SCHOOL_TEST that are not in the vocabulary:", sorted(set(test_words) - set(vocab)))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
bigrams in SCHOOL_TEST: 45 | never seen in SCHOOL_TEXT: 8
  ('meets', 'in'): context 'meets' was seen 1 times, but never followed by 'in'
  ('room', 'on'): context 'room' was seen 1 times, but never followed by 'on'
  ('on', 'friday'): context 'on' was seen 4 times, but never followed by 'friday'
  ('friday', '.'): context 'friday' was seen 1 times, but never followed by '.'
  ('robots', 'on'): context 'robots' was seen 6 times, but never followed by 'on'
  ('code', '.'): context 'code' was seen 3 times, but never followed by '.'
  ('the', 'robot'): context 'the' was seen 41 times, but never followed by 'robot'
  ('club', '.'): context 'club' was seen 7 times, but never followed by '.'
words of SCHOOL_TEST that are not in the vocabulary: []
```

   You should see a perplexity of **3.18** on `SCHOOL_TEXT` and **inf** on `SCHOOL_TEST`. The first number looks wonderful: on its own training text the model is, on average, as unsure as if it were choosing among about three equally likely words. But it is flattered, because the model has memorised these exact pairs. On new text the perplexity is infinite. The second block shows why: of the 45 bigrams of the test text, **8** (17.8%) were never seen, for example ("meets", "in") and ("on", "friday"). Every word of the test is in the vocabulary, so the problem is not new words but new *combinations* of known words. One more thing to notice: ("the", "robot") is unseen because the training text said "robots". To the counts, "robot" and "robots" are two unrelated labels; Part 2 shows how vectors can know better.

   Tips and common mistakes:

   - Perplexity on the *training* text is flattering: the model has seen exactly these pairs. The honest number is the one on text it has never seen.
   - Lower perplexity is better. The best possible value is 1 (the model is certain of every word).
   - The breakdown is not caused by new *words* (the block shows that every word of the test is known). It is caused by new *pairs* of known words.

   Try also (optional):

   - Remove the sentence about Friday from `SCHOOL_TEST` (edit a copy of the string) and recompute: is the perplexity still infinite?
   - Which unseen pair looks the most reasonable to you, a pair that English could easily contain? What does that say about the counts?

   *Exercise 1.4:* store in `n_unseen` the number of bigrams of `SCHOOL_TEST` that have a count of zero, and in `share_unseen` their share of all the test bigrams (a number between 0 and 1). Then write in `why_inf` (one sentence, a string) why the perplexity is infinite. Why it matters: this share is the real size of the sparsity problem in this tiny corpus.

```python
n_unseen = None       # how many test bigrams were never seen in training
share_unseen = None   # that number divided by all test bigrams
why_inf = ""          # one sentence
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_unseen is None or share_unseen is None or len(why_inf.split()) < 6:
    print("Fill in n_unseen, share_unseen and a sentence for why_inf above, then run this cell again.")
else:
    truth = sum(1 for p in test_pairs if bigram[p] == 0)
    check("n_unseen is right", n_unseen == truth, "Count the test pairs p with bigram[p] == 0.")
    check("share_unseen is n_unseen / all test bigrams", abs(share_unseen - truth / len(test_pairs)) < 1e-9, "Divide by len(test_pairs).")
    RESULTS["p1_share_unseen_bigrams"] = round(share_unseen, 3); RESULTS["p1_why_inf"] = why_inf
```

   `n_unseen` is 8 and `share_unseen` is 0.178.

5. **Step 1.5. Add-one smoothing.** The repair is to never say never. **Add-one smoothing** (Laplace smoothing) pretends every possible pair was seen once more than it was: P(w | prev) = (count(prev, w) + 1) / (count(prev) + V). The V in the denominator is there so that the probabilities in one row still add up to 1. The first block defines the smoothed model and a *unigram* baseline that ignores the previous word and just uses (count(w) + 1) / (N + V), where N is the number of tokens. The second block compares the three models on the training text and on `SCHOOL_TEST` in a table. The third checks that one smoothed row still sums to 1.

```python
def p_addone(prev, w):
    """Add-one smoothed bigram: (count(prev, w) + 1) / (count(prev as context) + V)."""
    return (bigram[(prev, w)] + 1) / (context[prev] + V)

def p_unigram(prev, w):
    """Baseline that ignores `prev`: (count(w) + 1) / (N + V)."""
    return (unigram[w] + 1) / (len(words) + V)
```

```python
models = [("unigram (add-one)", p_unigram), ("bigram, unsmoothed", p_bigram), ("bigram, add-one", p_addone)]
print(f"{'model':<20} {'on SCHOOL_TEXT':>15} {'on SCHOOL_TEST':>15}")
for name, f in models:
    print(f"{name:<20} {perplexity(SCHOOL_TEXT, f):>15.2f} {perplexity(SCHOOL_TEST, f):>15.2f}")
ppl_train_addone = perplexity(SCHOOL_TEXT, p_addone)
ppl_test_addone = perplexity(SCHOOL_TEST, p_addone)
ppl_test_unigram = perplexity(SCHOOL_TEST, p_unigram)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
model                 on SCHOOL_TEXT  on SCHOOL_TEST
unigram (add-one)              41.79           35.84
bigram, unsmoothed              3.18             inf
bigram, add-one                32.56           36.19
```

```python
row = sum(p_addone("the", w) for w in vocab)
print("add-one probabilities after 'the' add up to", round(row, 6))
assert abs(row - 1) < 1e-9
print("P(club | the): unsmoothed", round(p_bigram("the", "club"), 3), "-> add-one", round(p_addone("the", "club"), 3))
print("P(robot | the): unsmoothed", p_bigram("the", "robot"), "-> add-one", round(p_addone("the", "robot"), 4))
RESULTS["p1_ppl_test_addone"] = round(ppl_test_addone, 1)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
add-one probabilities after 'the' add up to 1.0
P(club | the): unsmoothed 0.146 -> add-one 0.057
P(robot | the): unsmoothed 0.0 -> add-one 0.0081
```

   You should see this table: the unigram baseline has perplexity 41.79 on the training text and **35.84** on the test; the unsmoothed bigram 3.18 and **inf**; the add-one bigram 32.56 and **36.19**. Smoothing has repaired the infinity, but look at the test column: the add-one bigram (36.19) is no better than the unigram baseline (35.84) that ignores the previous word. Why? Adding one to all 82 cells of a row is a large gift for a small row. After "the" (seen 41 times) about **50%** of the probability now goes to the 61 words that never followed "the": P(club | the) fell from 0.146 to 0.057, and P(robot | the) rose from 0 to 0.0081. The smoothed probabilities of the row still add up to 1, so what one pair gains another loses. Smoothing must be done with care; the exercise and Step 3.3 show better ways.

   Tips and common mistakes:

   - Add-one is a blunt instrument: with V = 82 and only 41 observations of "the" it moves a lot of probability to pairs that never happen. That is why smoothed numbers for common contexts are lower than the raw ones.
   - The unigram baseline is deliberately weak. If your bigram model cannot beat it, the context is not helping.
   - Perplexities from different vocabularies are not comparable. Here every model uses the same V.

   Try also (optional):

   - Try adding 0.1 instead of 1 in the numerator and 0.1 times V in the denominator. This is called add-k smoothing.
   - Compute the smoothed probability mass that goes to words never seen after "the" (one minus the sum over the 21 seen ones).

   *Exercise 1.5:* write `p_addk(prev, w, k)` = (count(prev, w) + k) / (count(prev) + k times V) and compute the perplexity on `SCHOOL_TEST` for k in 1, 0.5, 0.1 and 0.01. Store the results in the dictionary `ppl_by_k` and the best k in `best_k`. Why it matters: smoothing has a setting, and the right setting depends on the data. You will tune settings like this all through the course.

```python
ppl_by_k = {}     # e.g. {1: 56.1, 0.5: ...}
best_k = None     # the k with the lowest perplexity
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if not ppl_by_k or best_k is None:
    print("Fill in ppl_by_k and best_k above, then run this cell again.")
else:
    def _addk(k): return lambda a, b: (bigram[(a, b)] + k) / (context[a] + k * V)
    truth = {k: perplexity(SCHOOL_TEST, _addk(k)) for k in [1, 0.5, 0.1, 0.01]}
    check("k = 1 gives the add-one perplexity", abs(ppl_by_k.get(1, -1) - ppl_test_addone) < 0.01, "With k = 1 the formula is the add-one model.")
    check("best_k has the lowest perplexity", best_k == min(truth, key=truth.get), "Pick the key with the smallest value.")
    RESULTS["p1_best_k"] = best_k
```

   `ppl_by_k` is about {1: 36.19, 0.5: 27.30, 0.1: 14.76, 0.01: 10.25} and `best_k` is 0.01. On this text, adding less than one to every cell is better (a small k trusts the counts more). Do not conclude that k should always be tiny: the test text here is similar to the training text, and a text with many more unseen pairs would punish a small k.

6. **Step 1.6. Let the model write.** A language model that can score text can also *generate* it: pick a next word at random with chances given by the model, add it, and repeat until a full stop. The first block defines `generate_bigram(start, n_words, seed, prob)`. The second block prints five sentences from the unsmoothed model and five from the add-one model, each starting from "the" with seeds 1 to 5. Read each one critically: mark what is fluent (grammatical, makes sense) and what is nonsense. The third block checks which unsmoothed sentences are copies of the training text.

```python
def generate_bigram(start, n_words, seed, prob=p_bigram):
    """Write up to n_words tokens: start with `start`, then sample each next token with probability prob(previous, token); stop at '.'."""
    r = random.Random(seed)
    out = [start]
    for _ in range(n_words - 1):
        weights = [prob(out[-1], w) for w in vocab]
        out.append(r.choices(vocab, weights=weights)[0])
        if out[-1] == ".":
            break
    return " ".join(out)
```

```python
print("--- unsmoothed bigram model ---")
for s in range(1, 6):
    print(s, generate_bigram("the", 25, s, p_bigram))
print("--- add-one bigram model ---")
for s in range(1, 6):
    print(s, generate_bigram("the", 25, s, p_addone))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
--- unsmoothed bigram model ---
1 the club will show the robots and the students and a timer and the battery .
2 the whole school .
3 the club shares one laptop and a black line and a track after the robots are not perfect but the students take turns the science
4 the club gets better every failure and the students test the robots .
5 the sensor reads the wheels .
--- add-one bigram model ---
1 the build to students every motor line robotics take at after timer laptop snacks .
2 the whole whole answer at timer show room follow question question practise build learn in show younger whole one learn failure after after march friday
3 the code on in questions questions answer a timer every every younger meets timer march robotics bring robot track motor small robots and the robotics
4 the code ask join black are judge wheels teachers small club on floor build battery class wheels the teachers take class follow reads show to
5 the robots show test whole show wheels a meets whole robotics two best march every one perfect a code fails wheels snacks bring teachers better
```

```python
text_joined = " ".join(words)
copied = [s for s in range(1, 6) if generate_bigram("the", 25, s, p_bigram) in text_joined]
print("unsmoothed sentences that appear word for word in SCHOOL_TEXT:", copied)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
unsmoothed sentences that appear word for word in SCHOOL_TEXT: [2]
```

   You should see ten lines. The **unsmoothed** model writes sentences made only of pairs it has seen: "the whole school ." (seed 2) is a perfect English sentence, and it is also a literal copy of the text; "the sensor reads the wheels ." (seed 5) is grammatical but wrong about the world; "the club will show the robots and the students and a timer and the battery ." (seed 1) starts fluently and then wanders; seed 3 runs on without stopping because it keeps finding words to continue with. The **add-one** model writes word salad ("the build to students every motor line robotics take at ..."), because after smoothing every word is a candidate in every position. The last block lists the unsmoothed sentences that are word-for-word copies: only seed 2. So the unsmoothed model is mostly *not* copying (a surprise, with a text this short), but it is also not always sensible: it has no sense of where a sentence should end or what fits together.

   Tips and common mistakes:

   - `random.Random(seed)` gives a private random generator, so the same seed gives the same sentence on every run.
   - The add-one sentences wander because each of the 82 words gets a small share of every row. Smoothing buys safety on new text and pays with fluency when you sample.
   - A sentence can be locally plausible (every pair makes sense) and still mean nothing. That is the one-word-memory problem the lesson describes.

   Try also (optional):

   - Start from "every" or "robots" instead of "the" and compare.
   - Increase `n_words` to 60 and remove the stop at "." to see a long ramble.

   *Exercise 1.6:* run the unsmoothed generator for seeds 1 to 8 starting at `"the"` and store in `n_fluent` how many of the eight sentences *you* judge fluent (a human opinion: there is no single right answer). Store in `n_copied` how many of them appear word for word in `SCHOOL_TEXT`. Why it matters: a low count of copies and a high count of fluent sentences is what we want; with counts this small you will see that the two tend to go together.

```python
n_fluent = None   # your judgement, 0 to 8
n_copied = None   # computed from the text
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_fluent is None or n_copied is None:
    print("Fill in n_fluent and n_copied above, then run this cell again.")
else:
    outs = [generate_bigram("the", 25, s, p_bigram) for s in range(1, 9)]
    truth = sum(1 for o in outs if o in text_joined)
    check("n_fluent is between 0 and 8", 0 <= n_fluent <= 8, "Count your own judgements.")
    check("n_copied is right", n_copied == truth, "Test each sentence with 'o in text_joined'.")
    RESULTS["p1_n_copied_of_8"] = n_copied
```

   `n_copied` is 1. `n_fluent` is your own opinion; looking at the eight sentences, I counted four as fluent.

7. **Step 1.7. Imagine and write: what counting cannot know.** You have now seen what a bigram model can and cannot do. In this last step of the part you write down, in your own words, **three kinds of knowledge about language that counting adjacent words can never capture**, however much text you give it. Think about meaning (does the model know that "students" and "pupils" are close?), about distance (can it connect a word to another one ten words earlier?), and about the world (can it know that a robot cannot drink tea?). Give one short example sentence for each kind. There is no code to run in this step, only a written answer in a markdown cell on the Spark (or in the starter cell in Colab). Keep your answer: Part 6 shows how word embeddings fix the first kind.

   This step has no output. A good answer names three different kinds of failure and gives an example sentence for each: (1) *meaning*, since counts treat "students" and "pupils" as unrelated labels (you saw this with "robot" and "robots" in Step 1.4); (2) *long-distance structure*, since a bigram sees only one word back (so "are" after "yesterday" cannot know that the subject was "robots", ten words earlier); (3) *knowledge of the world*, since nothing in the counts says that a robot cannot drink tea. Other good answers: negation ("not perfect" versus "perfect"), reference ("it", "they"), the order of events, or sarcasm.

   Tips and common mistakes:

   - A good answer names the kind of knowledge and gives a sentence where counting adjacent words goes wrong.
   - Think about *why* counting fails, not only that it fails: it only sees one neighbour, it treats each word as an unrelated label, and it has no model of the world.

   Try also (optional):

   - Ask a neighbour for their three kinds and see which ones you both found.

   *Exercise 1.7:* write your three kinds of knowledge, each with a short example, in the string `answer_1_7`.

```python
answer_1_7 = ""   # three kinds of knowledge, each with an example (at least 30 words)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if len(answer_1_7.split()) < 30:
    print("Write at least 30 words in answer_1_7 (three kinds, each with an example), then run this cell again.")
else:
    check("answer_1_7 has at least 30 words", True)
    RESULTS["p1_answer_1_7"] = answer_1_7
```

   The check only tests that you wrote at least 30 words; your teacher reads the content.

### Part 2. From rules to counts to neural: three tiny language models (25 min)

*Lesson section: "History".* Language technology has had three big eras: hand-written rules, statistical counts, and neural networks. In this part you build one tiny model from each era for the same small job, and you compare them on the same eight questions. The job is a "robotics club helper": a bot that answers questions about the club of Part 1. The rule-based bot is a handful of patterns; the statistical model is the bigram table you already have; the neural model is a small PyTorch network that learns a vector for every word. The comparison is deliberately small and honest: on a text of 215 tokens a neural network has no magic advantage, but it has one property that counting lacks, and Step 2.4 shows it.

**Imagine first.** Here is a sentence none of the three models has ever seen: "the teachers fix the sensor". Which of the three (a rule-based bot, a bigram model, a tiny neural model) will give a sensible answer, and which will fail or refuse? Write `"yes"` or `"no"` for each, then compare your answers with the table you build in Step 2.5. Think about what each model *needs* in order to answer: a rule that matches, a pair of words it has counted, or a vector for each word.

```python
# Imagine first: which model copes with an input it has never seen?
my_guess = {"rules": None, "bigram": None, "neural": None}   # "yes" or "no" for each
```

1. **Step 2.1. A rule-based bot (the ELIZA era).** The first language programs, in the 1960s, were lists of rules: *if the input looks like this, reply like that*. You write the rules with regular expressions from Python's `re` module. The first block defines six rules for the club helper and a function `rule_bot(text)` that tries them in order and falls back to an apology. The second block runs the bot on eight inputs: four that the rules were written for and four that ask for the same things in other words. It prints the reply and counts how many inputs the rules really handled.

```python
RULES = [
    (r"\b(hello|hi|hey)\b", "Hello! I am the club helper. Ask me about meetings, robots or the contest."),
    (r"\bwhen\b.*\bmeet", "The robotics club meets every Friday in the science room."),
    (r"\bhow many\b.*\bstudents\b", "The club has twelve students and two teachers."),
    (r"\b(contest|competition)\b", "The contest is in March in the city."),
    (r"\b(battery|motor|sensor)\b", "Every robot has a motor and a sensor and a small battery."),
    (r"\b(thanks|thank you)\b", "You are welcome!"),
]
FALLBACK = "Sorry, I do not understand. Try asking about the meeting time or the contest."

def rule_bot(text):
    """Return (reply, handled). The first rule whose pattern is found in the lower-cased text wins."""
    for pattern, reply in RULES:
        if re.search(pattern, text.lower()):
            return reply, True
    return FALLBACK, False

print("number of rules:", len(RULES))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
number of rules: 6
```

```python
INPUTS = [
    "Hello there",
    "When does the club meet?",
    "How many students are in the club?",
    "Is there a contest this year?",
    "What day do you gather?",
    "Can a beginner join the club?",
    "Who fixes the robots when they fail?",
    "Do you have snacks?",
]
handled = 0
for text in INPUTS:
    reply, ok = rule_bot(text)
    handled += ok
    print(("HANDLED " if ok else "fallback ") + f"| {text:<38} -> {reply[:48]}")
print(f"the rules handled {handled} of {len(INPUTS)} inputs")
RESULTS["p2_rules_handled"] = handled
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
HANDLED | Hello there                            -> Hello! I am the club helper. Ask me about meetin
HANDLED | When does the club meet?               -> The robotics club meets every Friday in the scie
HANDLED | How many students are in the club?     -> The club has twelve students and two teachers.
HANDLED | Is there a contest this year?          -> The contest is in March in the city.
fallback | What day do you gather?                -> Sorry, I do not understand. Try asking about the
fallback | Can a beginner join the club?          -> Sorry, I do not understand. Try asking about the
fallback | Who fixes the robots when they fail?   -> Sorry, I do not understand. Try asking about the
fallback | Do you have snacks?                    -> Sorry, I do not understand. Try asking about the
the rules handled 4 of 8 inputs
```

   You should see `number of rules: 6` and then eight lines. The first four inputs ("Hello there", "When does the club meet?", "How many students are in the club?", "Is there a contest this year?") are **HANDLED**, and each gets exactly the right reply. The other four ("What day do you gather?", "Can a beginner join the club?", "Who fixes the robots when they fail?", "Do you have snacks?") fall through to the apology, so the rules handled **4 of 8** inputs. Look at the first failure: "What day do you gather?" means the same as "When does the club meet?" but uses other words, and a regular expression has no idea that the two are the same question. To cover more you would have to write a new rule for every phrasing, then check that the new rules do not hide old ones. Rule-based bots are exact where they work and silent where they do not.

   Tips and common mistakes:

   - In a regular expression `\b` means a word boundary, `.*` means any characters, and `(a|b)` means a or b. 
   - The rules are tried in order, so order matters: a general rule placed early would hide a specific one.
   - Rules are precise and predictable. When they match, the answer is exactly right, and you can explain why. That is why rule-based systems are still used for forms and banking menus.

   Try also (optional):

   - Ask "My robot has no battery": which rule fires? Is that the reply you want?
   - Write three more inputs that a person could ask about the contest and see how many of them your single contest rule catches.

   *Exercise 2.1:* write a seventh rule that makes the bot handle `"What day do you gather?"`. Store a tuple `(pattern, reply)` in `new_rule`. Why it matters: this is what it costs to extend a rule-based system by one phrasing; you will have to repeat it for every new way people ask.

```python
new_rule = None   # a tuple (pattern, reply), e.g. (r"\bday\b", "...")
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if new_rule is None:
    print("Fill in new_rule above, then run this cell again.")
else:
    pattern, reply = new_rule
    check("the new pattern matches 'what day do you gather?'", re.search(pattern, "what day do you gather?") is not None, "Try r'\\bday\\b' or r'\\bgather'.")
    check("the reply is a sentence", len(reply.split()) >= 3, "Write a reply of at least three words.")
    RESULTS["p2_new_rule"] = new_rule
```

   Any pattern that matches `what day do you gather?` passes the check. The sample answer uses `day` followed by `gather` or `meet`; with it, the bot handles 5 of the 8 inputs.

2. **Step 2.2. The statistical model on the same inputs.** Now give the same eight inputs to the bigram model of Part 1. A bigram model does not answer questions; it predicts the **next word**. So for each input we look at its last word and ask: what does the table say comes next? If the last word was never a context in the training text, the model has no counts and cannot say anything at all (unsmoothed), which is the failure of Part 1 again. The block prints, for each input, the last word, the model's most likely next word and the probability, or `no counts`.

```python
def next_word_guess(text):
    """Most likely next word after the last word of `text` under the unsmoothed bigram model, or None if that word has no counts."""
    last = words_of(text)[-1]
    if context[last] == 0:
        return last, None, 0.0
    nxt = max(vocab, key=lambda w: p_bigram(last, w))
    return last, nxt, p_bigram(last, nxt)

no_counts = 0
for text in INPUTS:
    last, nxt, p = next_word_guess(text)
    if nxt is None:
        no_counts += 1
        print(f"{text:<38} last word {last!r:<11} -> no counts")
    else:
        print(f"{text:<38} last word {last!r:<11} -> {nxt!r} (P = {p:.2f})")
print(f"the bigram model has no counts for {no_counts} of {len(INPUTS)} inputs")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
Hello there                            last word 'there'     -> no counts
When does the club meet?               last word 'meet'      -> no counts
How many students are in the club?     last word 'club'      -> 'will' (P = 0.43)
Is there a contest this year?          last word 'year'      -> no counts
What day do you gather?                last word 'gather'    -> no counts
Can a beginner join the club?          last word 'club'      -> 'will' (P = 0.43)
Who fixes the robots when they fail?   last word 'fail'      -> no counts
Do you have snacks?                    last word 'snacks'    -> '.' (P = 1.00)
the bigram model has no counts for 5 of 8 inputs
```

   You should see that the bigram model has **no counts for 5 of the 8 inputs**: the last words "there", "meet" (the text says "meets"), "year", "gather" and "fail" are not in the vocabulary. For "club" (inputs 3 and 6) it predicts "will" with P = 0.43, because the text says "the club will join" and "the club will show" and "the club will answer"; and after "snacks" it predicts a full stop with probability 1.00. Notice what this means. The bigram model *can* continue text where its counts allow, but it does not answer any of the questions: the rule bot gave 4 correct replies, the bigram model produced 3 continuations, none of them an answer. They solve different problems, and both fail on a word they have never seen.

   Tips and common mistakes:

   - The bigram model solves a different task from the rule bot: it continues text instead of replying. Both are compared here only on whether they can work with the input at all.
   - A question mark is not in `words_of`'s pattern, so it is dropped; the last *word* is used.
   - Where the table has counts, the guess is fluent but may be useless as an answer. Fluent and helpful are different things.

   Try also (optional):

   - Smooth the model (use `p_addone`) and see that it now always names a next word, but a poorly supported one.
   - Pick an input and print the whole row of probabilities for its last word.

   *Exercise 2.2:* store in `n_no_counts` the number of the eight inputs whose *last word is not in the vocabulary* (a word outside the vocabulary can never have counts), and name one input in `example_unknown` (a string copied from `INPUTS`). Why it matters: a word outside the vocabulary is the hardest case, because no count of any kind can mention it. Part 4 shows how tokenisers avoid it.

```python
n_no_counts = None      # inputs whose last word is not in the vocabulary
example_unknown = None  # one such input
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_no_counts is None or example_unknown is None:
    print("Fill in both answers above, then run this cell again.")
else:
    unknown = [t for t in INPUTS if words_of(t)[-1] not in set(vocab)]
    check("n_no_counts is right", n_no_counts == len(unknown), "Test words_of(text)[-1] not in set(vocab) for each input.")
    check("example_unknown is one of them", example_unknown in unknown, "Copy one of the inputs from the list above.")
    RESULTS["p2_inputs_with_unknown_last_word"] = n_no_counts
```

   `n_no_counts` is 5; any of the five inputs ending in a word outside the vocabulary is a valid `example_unknown`.

3. **Step 2.3. A tiny neural language model.** Now the third era. Instead of a table with one cell for every pair of words, a neural model gives every word a short list of numbers, a **vector** (here 8 numbers), looks up the vector of the previous word, and turns it into a score for every possible next word. In PyTorch that is `nn.Embedding(V, 8)` followed by `nn.Linear(8, V)`. The loss is cross-entropy (the negative log of the probability given to the right next word), so the **perplexity** is exp of the loss, the same quantity as in Part 1. The first block turns the bigrams of both texts into pairs of word numbers. The second trains the network on `SCHOOL_TEXT` for 300 epochs with Adam. The third reports the perplexity on both texts and puts it next to the bigram models.

```python
w2i = {w: i for i, w in enumerate(vocab)}      # word -> number

def bigram_ids(text):
    """(previous-word numbers, next-word numbers) for all bigrams of a text, as tensors."""
    ws = words_of(text)
    return torch.tensor([w2i[w] for w in ws[:-1] ]), torch.tensor([w2i[w] for w in ws[1:] ])

x_tr, y_tr = bigram_ids(SCHOOL_TEXT)
x_te, y_te = bigram_ids(SCHOOL_TEST)
print("training pairs:", len(x_tr), "| test pairs:", len(x_te))

class TinyLM(nn.Module):
    """previous word -> embedding vector -> scores for every word in the vocabulary."""
    def __init__(self, V, d=8):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.out = nn.Linear(d, V)
    def forward(self, x):
        return self.out(self.emb(x))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
training pairs: 214 | test pairs: 45
```

```python
def train_lm(d=8, epochs=300, lr=0.05, seed=SEED):
    """Train a TinyLM on the training bigrams (full batch, Adam). Returns the model and its loss curve."""
    torch.manual_seed(seed)
    model = TinyLM(V, d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(30 if QUICK else epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x_tr), y_tr)
        loss.backward(); opt.step()
        losses.append(float(loss))
    return model, losses

def lm_perplexity(model, x, y):
    with torch.no_grad():
        return math.exp(float(F.cross_entropy(model(x), y)))

lm, lm_losses = train_lm(8)
print(f"loss at epoch 0: {lm_losses[0]:.3f} | at the end: {lm_losses[-1]:.3f} | parameters: {sum(p.numel() for p in lm.parameters())}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
loss at epoch 0: 4.530 | at the end: 1.158 | parameters: 1394
```

```python
ppl_nn_train = lm_perplexity(lm, x_tr, y_tr)
ppl_nn_test = lm_perplexity(lm, x_te, y_te)
print(f"{'model':<22} {'SCHOOL_TEXT':>12} {'SCHOOL_TEST':>12}")
print(f"{'bigram, add-one':<22} {ppl_train_addone:>12.2f} {ppl_test_addone:>12.2f}")
print(f"{'neural, d = 8':<22} {ppl_nn_train:>12.2f} {ppl_nn_test:>12.2f}")
plt.figure(figsize=(5, 3)); plt.plot(lm_losses); plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.title("TinyLM training"); plt.tight_layout(); plt.show()
RESULTS["p2_nn_ppl_test"] = round(ppl_nn_test, 1)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
model                   SCHOOL_TEXT  SCHOOL_TEST
bigram, add-one               32.56        36.19
neural, d = 8                  3.18        39.09
```

   You should see `training pairs: 214 | test pairs: 45`; then the training loss falling from 4.530 (close to ln 82 = 4.41, the loss of a model that guesses uniformly among the 82 words) to 1.158, with 1394 parameters (82 times 8 for the embedding, plus 8 times 82 plus 82 for the output layer). The table shows the **training** perplexity 3.18, which is exactly the number of the unsmoothed bigram model (3.18): with 8 numbers per word the network has enough room to learn every row of the bigram table. On `SCHOOL_TEST` its perplexity is **39.09**, slightly worse than the add-one bigram (36.19). Be honest about this: on a text of 215 tokens a neural model has no advantage in perplexity. It has memorised the pairs it saw and, with no smoothing, it is not clever about the pairs it did not see. What it does have is vectors, which you inspect in the next step.

   Tips and common mistakes:

   - Full-batch training (all pairs at once) is fine here because there are only 214 pairs.
   - `nn.Embedding(V, 8)` is a table of V rows with 8 numbers each, learned by gradient descent; looking up row i is the same as multiplying a one-hot vector by the table (you prove it in Part 6).
   - Perplexity is exp(loss) only because `F.cross_entropy` uses the natural logarithm and averages over the pairs, the same definition as in Step 1.4.

   Try also (optional):

   - Change the learning rate to 0.01 or 0.2 and watch the loss curve.
   - Print the five most likely next words the trained model gives after "the" with `torch.softmax(lm(torch.tensor([w2i['the'] ])), -1)`.

   *Exercise 2.3:* train the network with vector sizes d = 2, 8 and 16 using `train_lm(d)`, and store the **training** perplexity of each in the dictionary `ppl_by_d` (keys 2, 8, 16). Then store in `bigger_is_better` whether d = 16 has a lower training perplexity than d = 2 (`True` or `False`). Why it matters: the size of the vectors is the first design choice of any embedding model.

```python
ppl_by_d = {}             # {2: ..., 8: ..., 16: ...}
bigger_is_better = None   # True or False
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if not ppl_by_d or bigger_is_better is None:
    print("Fill in ppl_by_d and bigger_is_better above, then run this cell again.")
else:
    check("ppl_by_d has the keys 2, 8 and 16", set(ppl_by_d) == {2, 8, 16}, "Loop over [2, 8, 16].")
    check("the d = 8 value matches the model of this step", abs(ppl_by_d.get(8, 0) - ppl_nn_train) < 0.05, "Use train_lm(d) and lm_perplexity(model, x_tr, y_tr).")
    check("bigger_is_better matches your numbers", bigger_is_better == (ppl_by_d[16] < ppl_by_d[2]), "Compare ppl_by_d[16] with ppl_by_d[2].")
    RESULTS["p2_ppl_by_d"] = {k: round(v, 2) for k, v in ppl_by_d.items()}
```

   `ppl_by_d` is about {2: 3.92, 8: 3.18, 16: 3.18} and `bigger_is_better` is True: two numbers per word are too few to keep apart 82 words, while 8 and 16 are both enough to reach the training perplexity of the table.

4. **Step 2.4. Why vectors help: similar words get similar vectors.** Here is the property that counting lacks. After training, each word has an 8-number vector in `lm.emb.weight`. If two words are used in similar places, the network has a reason to give them similar vectors, because that makes the same next words likely after both. We measure similarity with the **cosine** of two vectors, a·b divided by the length of a times the length of b: 1 means the same direction, 0 means unrelated, negative means opposite. The first block computes the cosine of four pairs of words that a person would call related and four unrelated pairs, chosen before looking at any result. The second block shows what the bigram table sees for the first related pair: separate rows with no connection between them.

```python
E = lm.emb.weight.detach().numpy()          # one row per word, 8 numbers each
related = [("students", "teachers"), ("robot", "robots"), ("motor", "sensor"), ("track", "contest")]
unrelated = [("students", "battery"), ("robots", "friday"), ("motor", "younger"), ("laptop", "mistake")]
def pair_cos(pairs):
    return [cosine(E[w2i[a] ], E[w2i[b] ]) for a, b in pairs]
cos_rel, cos_unrel = pair_cos(related), pair_cos(unrelated)
print("related pairs")
for (a, b), c in zip(related, cos_rel): print(f"  {a:>9} / {b:<9} cosine {c:+.2f}")
print("unrelated pairs")
for (a, b), c in zip(unrelated, cos_unrel): print(f"  {a:>9} / {b:<9} cosine {c:+.2f}")
print(f"mean cosine: related {np.mean(cos_rel):+.2f} | unrelated {np.mean(cos_unrel):+.2f}")
check("on average the related pairs are closer than the unrelated pairs", np.mean(cos_rel) > np.mean(cos_unrel), "With very little training the vectors are noisy; train longer or use more text.")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
related pairs
   students / teachers  cosine +0.39
      robot / robots    cosine -0.21
      motor / sensor    cosine +0.64
      track / contest   cosine +0.15
unrelated pairs
   students / battery   cosine +0.18
     robots / friday    cosine -0.57
      motor / younger   cosine -0.09
     laptop / mistake   cosine +0.41
mean cosine: related +0.24 | unrelated -0.02
✓ on average the related pairs are closer than the unrelated pairs
```

```python
nxt_s = {w2 for (w1, w2) in bigram if w1 == "students"}
nxt_t = {w2 for (w1, w2) in bigram if w1 == "teachers"}
print("words seen after 'students':", sorted(nxt_s))
print("words seen after 'teachers':", sorted(nxt_t))
print("shared next words:", sorted(nxt_s & nxt_t))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
words seen after 'students': ['and', 'are', 'build', 'learn', 'look', 'practise', 'take', 'test', 'will', 'write']
words seen after 'teachers': ['.', 'bring', 'help', 'look']
shared next words: ['look']
```

   You should see the cosines of eight pairs. Related pairs: students / teachers +0.39, robot / robots -0.21, motor / sensor +0.64, track / contest +0.15. Unrelated pairs: from -0.57 to +0.41. On average the related pairs are closer (**+0.24** against **-0.02**), and the check prints a tick. But be honest about the noise: laptop / mistake (+0.41) is closer than three of the four "related" pairs, and robot / robots is even negative, because the text uses "robot" only in "every robot", "a robot fails" and "the best robot", and "robots" in other places, so the network had no reason to put them together. A text of 215 tokens is far too small for reliable vectors; Part 6 repeats the experiment with 3,000 sentences. The second block shows how the counting model sees the same pair: "students" is followed by 10 different words and "teachers" by 4, and they share only one ("look"). Nothing in the table tells the model that the two words are alike.

   Tips and common mistakes:

   - Cosine uses only direction, not length, so a frequent word with a long vector can still be compared fairly with a rare one.
   - With 215 tokens the vectors are noisy. Treat the pattern (related higher than unrelated) as the lesson and the individual numbers as approximate. The check in this lab is tolerant for that reason.
   - The bigram model compares two words only through the rows of the table, and those rows overlap only where the training text happened to use the same neighbour.

   Try also (optional):

   - Write the cosine of "the" and "." (two very different kinds of word). What do you expect?
   - Use `nearest(E, vocab, 'students', 5)` to list the closest words to "students" in this tiny embedding.

   *Exercise 2.4:* choose a pair of words from the vocabulary that you think should be similar and store them as a tuple in `my_pair`; store their cosine in `my_cos`, and the cosine of the first word with `"the"` in `cos_with_the`. Why it matters: a pair you picked yourself shows whether the tiny embedding agrees with your intuition, or is still noisy.

```python
my_pair = None       # a tuple of two words, e.g. ("club", "contest")
my_cos = None        # cosine of the two vectors
cos_with_the = None  # cosine of the first word with "the"
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if my_pair is None or my_cos is None or cos_with_the is None:
    print("Fill in my_pair, my_cos and cos_with_the above, then run this cell again.")
else:
    a, b = my_pair
    check("both words are in the vocabulary", a in w2i and b in w2i, "Choose words from the vocab list.")
    check("my_cos is the cosine of the two vectors", abs(my_cos - cosine(E[w2i[a] ], E[w2i[b] ])) < 1e-6, "Use cosine(E[w2i[a] ], E[w2i[b] ]).")
    check("cos_with_the is right", abs(cos_with_the - cosine(E[w2i[a] ], E[w2i["the"] ])) < 1e-6, "Compare the first word with 'the'.")
    RESULTS["p2_my_pair"] = my_pair; RESULTS["p2_my_cos"] = round(my_cos, 2)
```

   With the sample pair (students, teachers), `my_cos` is 0.39 and `cos_with_the` is -0.24. Your numbers depend on your pair.

5. **Step 2.5. The three eras side by side.** Now collect everything into one table. For each of the three models, record what it needs from you, how many numbers it stores (rules, table entries or parameters), and whether it can say something sensible about the unseen sentence "the teachers fix the sensor". The block computes the numbers from the objects you built, so they are measured and not typed. For the last column the bot and the bigram model are tested with the code of Steps 2.1 and 2.2 and the neural model by asking it for a next word after "teachers", a word it knows in a context it never saw.

```python
new_in = "the teachers fix the sensor"
rule_ok = rule_bot(new_in)[1]                                   # does any rule fire? (the word 'sensor' makes one fire)
bigram_ok = all(bigram[p] > 0 for p in zip(words_of(new_in), words_of(new_in)[1:]))   # are all its pairs in the table?
with torch.no_grad():
    probs = torch.softmax(lm(torch.tensor([w2i["teachers"] ])), dim=-1)[0]
nn_next = vocab[int(probs.argmax())]
n_params = sum(p.numel() for p in lm.parameters())
rows = [("rules (ELIZA)", "a person writes them", f"{len(RULES)} rules", rule_ok),
        ("bigram counts", "a text to count", f"{len(bigram)} table entries", bigram_ok),
        ("tiny neural model", "text + training", f"{n_params} parameters", True)]
print(f"{'model':<18} {'needs':<22} {'size':<20} {'handles the new sentence?'}")
for name, needs, size, ok in rows:
    print(f"{name:<18} {needs:<22} {size:<20} {'yes' if ok else 'no'}")
print("neural model: after 'teachers' it predicts", repr(nn_next), "although 'the teachers fix' never occurs")
RESULTS["p2_sizes"] = {"rules": len(RULES), "bigram_entries": len(bigram), "nn_params": n_params}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
model              needs                  size                 handles the new sentence?
rules (ELIZA)      a person writes them   6 rules              yes
bigram counts      a text to count        161 table entries    no
tiny neural model  text + training        1394 parameters      yes
neural model: after 'teachers' it predicts 'help' although 'the teachers fix' never occurs
```

   You should see three rows. **Rules**: 6 rules, written by a person; they "handle" the new sentence "the teachers fix the sensor" only by luck, because the word "sensor" triggers the equipment rule and the reply is about batteries, not about fixing. **Bigram counts**: 161 table entries; the sentence is **not** handled, because the pair ("teachers", "fix") was never seen. **Tiny neural model**: 1394 parameters; it can always produce a guess, here "help" after "teachers", although the pair "teachers fix" never occurs. This is the property to remember: the table has a hole for every unseen pair, while the neural model fills every cell from shared vectors (whether the fill is *good* depends on the text; on 215 tokens it is crude). So the right answers to the Imagine-first question were rules: yes (by accident), bigram: no, neural: yes.

   Tips and common mistakes:

   - "Handles" does not mean "is right". The rule bot fires for the wrong reason (it only sees the word "sensor"); the neural model always produces a guess, good or bad.
   - The bigram table has fewer entries than the neural model has parameters, but the table cannot grow smoothly: to handle a new pair it needs new counts.
   - Real language models of the neural era have billions of parameters; the idea is the same as `TinyLM` with many more words, many more layers and far more text.

   Try also (optional):

   - Compare the "size" numbers with the lesson's V = 10,000 example: how many table entries would a bigram table need?
   - Add a fourth row for the add-one bigram model. Does it handle the new sentence?

   *Exercise 2.5:* compare your `my_guess` dictionary from the start of the part with the table. Store in `n_right` how many of your three guesses match the table (`"yes"` for handles the new sentence, `"no"` for does not). Why it matters: predicting first and then checking is the habit that makes you a better engineer.

```python
n_right = None   # how many of your 3 guesses were right (0 to 3)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_right is None or any(v is None for v in my_guess.values()):
    print("Fill in my_guess (Imagine first) and n_right above, then run this cell again.")
else:
    truth = {"rules": "yes" if rule_ok else "no", "bigram": "yes" if bigram_ok else "no", "neural": "yes"}
    real = sum(1 for k in truth if my_guess.get(k) == truth[k])
    check("n_right matches your guesses against the table", n_right == real, "Compare each of the three values of my_guess with the table above.")
    RESULTS["p2_guesses_right"] = n_right
```

   The truth is rules yes (by accident), bigram no, neural yes. `n_right` is how many of your three guesses match these.

6. **Step 2.6. Imagine and write: which era would you choose?.** Last step of the part: choose the right tool. For each of three products write one sentence that names the era you would choose (rules, counts or neural) and gives one reason: (a) a **school canteen menu bot** that answers questions about today's menu, (b) **autocomplete on a phone keyboard**, which must run instantly on the phone, and (c) a **chatbot that must answer anything** a student can ask. There is no single right answer, but there are weak reasons and strong ones. A strong reason mentions what the product must handle (a closed list of questions, speed and privacy on a small device, open-ended input) and what each era needs.

   This step has no output. A strong answer names an era and ties it to the product: for the **canteen menu bot** rules are a good choice, because the topic is closed, the questions are few, and a rule is exact and easy to fix when the menu changes; for **phone autocomplete** counts or a very small neural model are right, because the model must answer in milliseconds on a small device and learn from the user's own typing; for a chatbot that **must answer anything**, only a large neural model trained on a huge amount of text can cope with questions nobody foresaw.

   Tips and common mistakes:

   - Weak reasons are vague: "because it is modern". Strong reasons point to a property: closed topic, small device, open input.
   - You may choose a mixture (rules for the menu, a neural model for the rest). If you do, say which part is which.

   Try also (optional):

   - Ask which choice would change if the canteen bot had to understand Filipino and English mixed in one sentence.

   *Exercise 2.6:* fill the dictionary `answer_2_6` with one sentence for each of the keys `"canteen"`, `"autocomplete"` and `"anything"`.

```python
answer_2_6 = {"canteen": "", "autocomplete": "", "anything": ""}   # one sentence each, naming the era and a reason
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if any(len(answer_2_6.get(k, "").split()) < 8 for k in ["canteen", "autocomplete", "anything"]):
    print("Write at least 8 words for each of the three products, then run this cell again.")
else:
    check("all three answers are written", True)
    RESULTS["p2_answer_2_6"] = answer_2_6
```

   The check only tests that each answer has at least 8 words.

### Part 3. The limits of n-grams: sparsity (20 min)

*Lesson section: "N-grams".* In Parts 1 and 2 you met the main weakness of counting: the model only knows the pairs it has seen. In this part you measure that weakness precisely. You count how many n-grams are possible and how many are seen (the sparsity problem of the lesson's `ngram_space` figure), how many n-grams of new text are unseen, how mixing a bigram with a unigram model repairs part of the problem, and what happens when you give the model more context: it stops being creative and starts to copy. At the end you explain, in words, how a neural model avoids the huge table.

**Imagine first.** `SCHOOL_TEST` is new text about the same club, written in the same style. Before you run anything, guess what **share of its bigrams** (pairs of neighbouring tokens) never occurred in `SCHOOL_TEXT`, as a number between 0 and 1. You already saw in Step 1.4 that some pairs were unseen; here you also guess how fast the share grows for trigrams and 4-grams. Write your guesses and compare them with Step 3.2.

```python
# Imagine first: the share of new n-grams that the training text never contained.
my_unseen_bigram = None    # your guess for bigrams, between 0 and 1
my_unseen_fourgram = None  # your guess for 4-grams, between 0 and 1
```

1. **Step 3.1. Possible versus seen.** An n-gram is n neighbouring tokens. With a vocabulary of V words there are V to the power n possible n-grams: 82 possible bigrams for each of the 82 first words, and so on. But a text of 215 tokens can contain at most 215 minus n plus 1 of them. The first block defines the helper `ngrams` and counts, for n = 1 to 4, how many distinct n-grams `SCHOOL_TEXT` contains, how many are possible, and the share seen. The second block plots the two numbers on a log scale. The third repeats the arithmetic for the lesson's example of V = 10,000.

```python
def ngrams(ws, n):
    """All runs of n neighbouring tokens of the list ws, as tuples."""
    return list(zip(*[ws[i:] for i in range(n)]))

rows = []
print(f"{'n':>2} {'distinct seen':>14} {'possible V^n':>16} {'share seen':>12}")
for n in range(1, 5):
    seen = len(set(ngrams(words, n)))
    possible = V ** n
    rows.append((n, seen, possible))
    print(f"{n:>2} {seen:>14} {possible:>16,} {seen / possible:>12.2e}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 n  distinct seen     possible V^n   share seen
 1             82               82     1.00e+00
 2            161            6,724     2.39e-02
 3            201          551,368     3.65e-04
 4            212       45,212,176     4.69e-06
```

```python
ns_ = [r[0] for r in rows]
plt.figure(figsize=(5.5, 3.2))
plt.plot(ns_, [r[2] for r in rows], "o-", label="possible (V^n)", color="tab:red")
plt.plot(ns_, [r[1] for r in rows], "o-", label="seen in SCHOOL_TEXT", color="tab:blue")
plt.yscale("log"); plt.xticks(ns_); plt.xlabel("n"); plt.ylabel("number of n-grams (log scale)"); plt.legend()
plt.tight_layout(); plt.show()
```

```python
V_big = 10_000
for n in range(1, 5):
    print(f"V = 10,000, n = {n}: {V_big ** n:,} possible n-grams")
print("a corpus of one million words contains at most", f"{1_000_000:,}", "bigram tokens, so at most", f"{1_000_000 / V_big ** 2:.0%}", "of the possible bigrams can ever be seen")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
V = 10,000, n = 1: 10,000 possible n-grams
V = 10,000, n = 2: 100,000,000 possible n-grams
V = 10,000, n = 3: 1,000,000,000,000 possible n-grams
V = 10,000, n = 4: 10,000,000,000,000,000 possible n-grams
a corpus of one million words contains at most 1,000,000 bigram tokens, so at most 1% of the possible bigrams can ever be seen
```

   You should see this table. For n = 1 all 82 words are seen (share 1). For n = 2 there are 161 distinct bigrams out of 6,724 possible (**2.4%**); for n = 3, 201 out of 551,368 (**0.036%**); for n = 4, 212 out of 45,212,176 (about **0.00047%**). The plot has two lines on a log scale: the blue line (seen) is almost flat, rising from 82 to 212, while the red line (possible) climbs from 82 to over 45 million. The gap between the lines is the sparsity problem. With the lesson's V = 10,000 the possible bigrams already number 100,000,000, trigrams 10^12 and 4-grams 10^16; even a corpus of a million tokens could contain at most 1% of the bigrams.

   Tips and common mistakes:

   - The number of distinct n-grams seen grows almost linearly with n (each new position is a new item), but the number of possible ones explodes exponentially.
   - The vertical axis is logarithmic: each gridline is ten times the one below. A straight line on a log plot is an exponential.
   - For the real vocabulary of a modern model (tens of thousands of tokens) even a bigram table is mostly empty.

   Try also (optional):

   - Plot the share seen on its own log axis.
   - Count how many *distinct* trigrams occur only once in `SCHOOL_TEXT`.

   *Exercise 3.1:* with V = 10,000, store in `max_share_bigrams` the largest possible share (a number between 0 and 1) of all possible bigrams that a corpus of one million tokens could contain, and in `max_share_trigrams` the same for trigrams. Why it matters: even a huge corpus leaves most of the n-gram space empty, which is why counting alone cannot reach long contexts.

```python
max_share_bigrams = None
max_share_trigrams = None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if max_share_bigrams is None or max_share_trigrams is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("bigrams: 1,000,000 / 10,000^2 = 0.01", abs(max_share_bigrams - 0.01) < 1e-9, "Divide the number of tokens by V ** 2.")
    check("trigrams: 1,000,000 / 10,000^3 = 1e-06", abs(max_share_trigrams - 1e-6) < 1e-12, "Divide the number of tokens by V ** 3.")
    RESULTS["p3_max_share_trigrams"] = max_share_trigrams
```

   `max_share_bigrams` is 0.01 and `max_share_trigrams` is 1e-06.

2. **Step 3.2. Unseen n-grams in new text.** The number that decides whether a counting model works is not how much of the space was seen, but how much of *new text* falls inside what was seen. For n = 1 to 4, the first block takes every n-gram of `SCHOOL_TEST` and checks whether it occurs anywhere in `SCHOOL_TEXT`; it prints the share that was never seen. The second block draws the shares as bars. Compare with your Imagine-first guesses.

```python
train_sets = {n: set(ngrams(words, n)) for n in range(1, 5)}
unseen_share = {}
print(f"{'n':>2} {'test n-grams':>13} {'never seen':>11} {'share unseen':>13}")
for n in range(1, 5):
    grams = ngrams(test_words, n)
    miss = sum(1 for g in grams if g not in train_sets[n])
    unseen_share[n] = miss / len(grams)
    print(f"{n:>2} {len(grams):>13} {miss:>11} {unseen_share[n]:>13.1%}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 n  test n-grams  never seen  share unseen
 1            46           0          0.0%
 2            45           8         17.8%
 3            44          21         47.7%
 4            43          31         72.1%
```

```python
plt.figure(figsize=(5, 3))
plt.bar([str(n) for n in unseen_share], list(unseen_share.values()), color="tab:orange")
plt.ylim(0, 1); plt.xlabel("n"); plt.ylabel("share of test n-grams never seen"); plt.title("New text breaks long contexts")
plt.tight_layout(); plt.show()
RESULTS["p3_unseen_share"] = {n: round(v, 3) for n, v in unseen_share.items()}
```

   You should see: n = 1: 0 of 46 unseen (0.0%); n = 2: 8 of 45 (**17.8%**); n = 3: 21 of 44 (**47.7%**); n = 4: 31 of 43 (**72.1%**). The bars rise steeply: nearly one in five bigrams of new text was never seen, nearly half of the trigrams and almost three quarters of the 4-grams. Compare with your Imagine-first guesses. If you guessed much less than this for bigrams, or for 4-grams, you are in good company: the shares are higher than most people expect. The reason is that the test text is not a copy of the training text; it recombines familiar words in new ways, as every new sentence does, and a longer n-gram is a more specific combination.

   Tips and common mistakes:

   - For n = 1 the share is 0 because every test word occurs in the training text (Step 1.4 showed it). For n = 2 it matches the 8 of 45 you counted before.
   - The test text is tiny, so each share rests on only 43 to 46 n-grams; the jumps between n = 2, 3 and 4 are real, but the exact values are approximate.
   - In real systems the same effect appears: the longer the context, the more often it has never been seen.

   Try also (optional):

   - Print the unseen trigrams of the test text.
   - Compute the same shares when the training text is only its first half. Does less training text increase the unseen share?

   *Exercise 3.2:* split `words` into a first half (training) and a second half (test), and compute the share of **unseen bigrams** of the second half given the first half. Store it in `unseen_half`. Why it matters: this is a test on text from the *same* source as the training; compare it with the share for `SCHOOL_TEST`.

```python
unseen_half = None   # share between 0 and 1
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if unseen_half is None:
    print("Fill in unseen_half above, then run this cell again.")
else:
    mid = len(words) // 2
    first, second = set(ngrams(words[:mid], 2)), ngrams(words[mid:], 2)
    truth = sum(1 for g in second if g not in first) / len(second)
    check("unseen_half is right", abs(unseen_half - truth) < 1e-9, "Train on words[:mid], test on words[mid:], and use bigrams (n = 2).")
    RESULTS["p3_unseen_half"] = round(unseen_half, 3)
```

   `unseen_half` is 0.738 (73.8%). That is higher than the 17.8% of `SCHOOL_TEST`, because each half has only 107 tokens: the less training text, the more is new.

3. **Step 3.3. Interpolation: mix a bigram with a unigram.** Add-one smoothing is blunt. A better repair is to *mix* models: use the bigram when it knows something and fall back on the unigram otherwise. **Linear interpolation** does it with one number lambda between 0 and 1: P = lambda times P_bigram + (1 minus lambda) times P_unigram, where both parts use add-one smoothing so that nothing is ever zero. At lambda = 1 you have the add-one bigram of Step 1.5, at lambda = 0 the unigram baseline. The block computes the perplexity on `SCHOOL_TEST` for five lambdas, asserts that the two ends match the earlier numbers, and plots perplexity against lambda.

```python
def p_interp(lam):
    """Return a probability function: lam * add-one bigram + (1 - lam) * add-one unigram."""
    return lambda prev, w: lam * p_addone(prev, w) + (1 - lam) * p_unigram(prev, w)

lams = [0, 0.3, 0.6, 0.9, 1]
ppl_lam = {lam: perplexity(SCHOOL_TEST, p_interp(lam)) for lam in lams}
for lam, p in ppl_lam.items():
    print(f"lambda = {lam:<4} perplexity on SCHOOL_TEST = {p:.2f}")
assert abs(ppl_lam[1] - ppl_test_addone) < 1e-9 and abs(ppl_lam[0] - ppl_test_unigram) < 1e-9
best_lam = min(ppl_lam, key=ppl_lam.get)
print("best lambda among these:", best_lam)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
lambda = 0    perplexity on SCHOOL_TEST = 35.84
lambda = 0.3  perplexity on SCHOOL_TEST = 32.54
lambda = 0.6  perplexity on SCHOOL_TEST = 31.99
lambda = 0.9  perplexity on SCHOOL_TEST = 34.21
lambda = 1    perplexity on SCHOOL_TEST = 36.19
best lambda among these: 0.6
```

```python
plt.figure(figsize=(5, 3))
plt.plot(list(ppl_lam), list(ppl_lam.values()), "o-")
plt.xlabel("lambda (weight of the bigram model)"); plt.ylabel("perplexity on SCHOOL_TEST"); plt.title("Mixing bigram and unigram")
plt.tight_layout(); plt.show()
RESULTS["p3_best_lambda"] = best_lam
```

   You should see five perplexities: lambda = 0 (the unigram) 35.84; 0.3: 32.54; 0.6: **31.99**; 0.9: 34.21; lambda = 1 (add-one bigram) 36.19. The two ends reproduce the numbers of Step 1.5 (the assert checks it), and the curve is a shallow bowl: **the mixture is better than either model alone**. The best of the five values is lambda = 0.6, about 11% lower than the unigram. A bigram model that knows its pairs and a unigram model that knows its words complement each other: the first is specific, the second is safe. The improvement is modest, because the add-one parts still waste probability; the idea of mixing a specific model with a safe one is the basis of several real smoothing methods.

   Tips and common mistakes:

   - Interpolation is the idea behind several classic smoothing methods. The weights can themselves be tuned on held-out text, as you do in the exercise.
   - The mixture is still a proper probability: both parts sum to 1 over the vocabulary, so the mixture does too.
   - Tuning lambda on the same test text you report is a mild form of cheating. With a text this small there is no better option, but in a real project you would tune on a separate validation text.

   Try also (optional):

   - Mix three models: trigram, bigram and unigram.
   - Mix with an unsmoothed bigram (`p_bigram`) and see what happens when a pair is unseen.

   *Exercise 3.3:* search lambda in 0, 0.1, 0.2, ..., 1.0 (eleven values) and store the lambda with the lowest perplexity in `best_lam_fine` and that perplexity in `best_ppl_fine`. Why it matters: a coarse grid can miss the sweet spot, and the curve is flat near its minimum.

```python
best_lam_fine = None
best_ppl_fine = None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if best_lam_fine is None or best_ppl_fine is None:
    print("Fill in best_lam_fine and best_ppl_fine above, then run this cell again.")
else:
    grid = {round(i / 10, 1): perplexity(SCHOOL_TEST, p_interp(i / 10)) for i in range(11)}
    check("best_lam_fine is the best of the 11 values", round(best_lam_fine, 1) == min(grid, key=grid.get), "Evaluate perplexity(SCHOOL_TEST, p_interp(lam)) for all 11 values.")
    check("best_ppl_fine is its perplexity", abs(best_ppl_fine - min(grid.values())) < 1e-6, "Keep the lowest value.")
    assert best_ppl_fine <= min(ppl_lam.values()) + 1e-9      # a finer grid cannot be worse
    RESULTS["p3_best_lambda_fine"] = round(best_lam_fine, 1)
```

   `best_lam_fine` is 0.5 and `best_ppl_fine` is 31.92: a finer grid finds a slightly better value than the five of the block (31.99 at 0.6).

4. **Step 3.4. Longer context, more copying.** Longer contexts make a model more specific, but with a tiny corpus they leave it with almost nothing to choose from: in most contexts only one continuation was ever seen, so the model repeats the training text. You measure this with the **copy rate**, the share of 3-word sequences in the generated text that also occur in the training text. The first block defines `train_ngram` and `generate_ngram` for any n (an unsmoothed model: the context is the n minus 1 previous tokens; when a context has no continuation the generator starts again after a full stop). The second block generates 300 tokens for n = 2, 3 and 4 and prints the copy rate and the number of distinct words used.

```python
def train_ngram(ws, n):
    """Dictionary: context (a tuple of n-1 tokens) -> Counter of the tokens that followed it."""
    model = defaultdict(Counter)
    for g in ngrams(ws, n):
        model[g[:-1] ][g[-1] ] += 1
    return model

def generate_ngram(model, n, length, seed):
    """Sample `length` tokens. The text starts right after a full stop; at a dead end (a context with no continuation) it starts again."""
    r = random.Random(seed)
    starts = [c for c in model if c[0] == "."]      # contexts that begin with a full stop
    out = list(r.choice(starts))
    while len(out) < length + 1:
        ctx = tuple(out[-(n - 1):])
        if ctx not in model:
            out.extend(r.choice(starts)); continue
        nxt = model[ctx]
        out.append(r.choices(list(nxt), weights=list(nxt.values()))[0])
    return out[1:length + 1]
```

```python
train3 = set(ngrams(words, 3))
def copy_rate(n, length=300, seed=3):
    out = generate_ngram(train_ngram(words, n), n, length, seed)
    tri = ngrams(out, 3)
    return sum(1 for g in tri if g in train3) / len(tri), len(set(out))

copy = {}
print(f"{'n':>2} {'copy rate':>10} {'distinct words used':>20}")
for n in [2, 3, 4]:
    rate, distinct = copy_rate(n)
    copy[n] = (rate, distinct)
    print(f"{n:>2} {rate:>10.1%} {distinct:>20} of {V}")
RESULTS["p3_copy_rate"] = {n: round(v[0], 3) for n, v in copy.items()}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 n  copy rate  distinct words used
 2      67.1%                   72 of 82
 3      99.3%                   64 of 82
 4      99.3%                   67 of 82
```

   You should see three rows. With n = 2 the copy rate of 3-word runs is **67.1%**: about a third of the three-word runs in the output are new combinations of overlapping pairs, and 72 of the 82 words are used. With n = 3 the copy rate jumps to **99.3%** and only 64 different words appear; with n = 4 it is also 99.3% with 67 words. In plain words: from n = 3 on, the model is a photocopier. In most contexts it has seen only one continuation, so it has no choice but to repeat the text. The one-in-a-hundred exceptions come from the single place where the generator reached the end of the training text, found no continuation and started again. This is the trade-off of the lesson: a bigger n is more specific and needs far more text to have any freedom left. This is the same pattern you saw in Week 1 with the club text.

   Tips and common mistakes:

   - The copy rate of 3-grams for n = 2 is not zero: a bigram model can produce a three-word run that happens to be in the text, because two overlapping pairs fit together.
   - The generator starts again after a full stop when a context is unknown, which happens at the end of the training text, where the last context has no continuation.
   - Try other seeds. The values change a little, but the order stays: more context, more copying.

   Try also (optional):

   - Generate with n = 1 (context of zero words) by treating each token independently and compute the copy rate.
   - Print one generated run for each n and read it aloud.

   *Exercise 3.4:* compute the copy rate of the generated text for n = 5 and n = 6 with `copy_rate(n)`. Store them in `copy5` and `copy6` (numbers between 0 and 1). Then store in `trend_ok` whether the copy rate is non-decreasing from n = 2 to n = 6 (`True` or `False`). Why it matters: it shows that the creative freedom of a counting model disappears as the context grows.

```python
copy5 = None
copy6 = None
trend_ok = None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if copy5 is None or copy6 is None or trend_ok is None:
    print("Fill in copy5, copy6 and trend_ok above, then run this cell again.")
else:
    check("copy5 is right", abs(copy5 - copy_rate(5)[0]) < 1e-9, "Use copy_rate(5)[0].")
    check("copy6 is right", abs(copy6 - copy_rate(6)[0]) < 1e-9, "Use copy_rate(6)[0].")
    seq = [copy_rate(n)[0] for n in range(2, 7)]
    check("trend_ok matches the five rates", trend_ok == all(a <= b + 1e-12 for a, b in zip(seq, seq[1:])), "Compare each value with the next.")
    RESULTS["p3_copy5_copy6"] = (round(copy5, 3), round(copy6, 3))
```

   `copy5` is 0.993 and `copy6` is 0.980. `trend_ok` is False: the rate reaches about 99% from n = 3 and then wiggles by a hair, so it is not strictly non-decreasing; the wiggle comes from the place where the generator reaches the end of the training text and starts again.

5. **Step 3.5. Imagine and write: how a neural model avoids the table.** You have seen that an n-gram model needs a table of up to V to the power n entries, and that most of the table is empty. A neural model does not store a table: in Step 2.3 you built one with only 1,394 numbers that could answer for any pair of the 82 words. In a paragraph of your own, explain **how a neural model avoids the table of V to the power n entries**. Use these words: *vector*, *parameters*, *similar words*, *generalise*. If you can, say what a neural model still cannot do with a very long context (hint: its parameters do not grow with n, but its attention or memory does).

   This step has no output. A good answer says: a neural model stores a **vector** for each word and a set of weights (the **parameters**), so its size depends on the vocabulary and the size of the vectors, not on V to the power n; and because **similar words** get similar vectors, what the model learns about one word carries over to others, so it can **generalise** to a context that never occurred instead of finding an empty cell. Your Step 2.3 model needed 1,394 numbers; a bigram table for the same words has 6,724 cells, and a 4-gram table 45 million. A good answer may add the limit: the neural model's cost grows with the length of context it reads (Week 7 explains attention).

   Tips and common mistakes:

   - A good answer says what is stored (vectors and weights), why the size does not explode with n, and why this helps with unseen contexts.
   - It is fine to be unsure about long contexts: Weeks 7 and 8 explain how transformers read thousands of tokens.

   Try also (optional):

   - Estimate how many parameters a neural bigram model with V = 10,000 words and d = 100 needs, and compare it with 10,000 squared.

   *Exercise 3.5:* write your paragraph in the string `answer_3_5` (at least 40 words).

```python
answer_3_5 = ""   # at least 40 words
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if len(answer_3_5.split()) < 40:
    print("Write at least 40 words in answer_3_5, then run this cell again.")
else:
    check("answer_3_5 has at least 40 words", True)
    check("it uses at least two of the key words", sum(k in answer_3_5.lower() for k in ["vector", "parameter", "similar", "generali"]) >= 2, "Use the words vector, parameters, similar words, generalise.")
    RESULTS["p3_answer_3_5"] = answer_3_5
```

   The check tests the length and that you used at least two of the four key words.

### Part 4. Tokenisation in the real world (25 min)



A language model never sees letters or words: it sees a list of whole numbers, and the program that turns your text into that list is the **tokenizer**. Until now we used words as the reading unit, because they were easy to count. Real systems cut text into **tokens**, and a token is not always a word: it can be a whole word, a piece of a word, a single character, a space plus a word, even one byte of an emoji. In this part you cut text into words and characters by hand, count what each choice costs, and then meet two real tokenizers: the byte-level BPE tokenizer of GPT-2 and the WordPiece tokenizer of BERT. ==The way text is cut into tokens decides how long a text is for the model, how big its vocabulary must be, and how fairly it treats different languages.==

Only the small tokenizer files of `gpt2` and `bert-base-uncased` are used (no model weights). On the school's DGX Spark they are read from the offline cache; in Colab they download once (a few hundred kilobytes). If the files cannot be loaded, the loading cell prints a clear message and the cells that need them stop with that message, while everything else keeps working.

**Imagine first.** Look at this sentence: *Unbelievably, the students' robots weren't broken!* It has 6 words when you split on spaces, and a certain number of characters (letters, spaces and punctuation marks, every one counted). How many **tokens** will it need at the word level, at the character level, and for the GPT-2 tokenizer? Write three whole numbers before you run anything. Hint: a tokenizer for words must decide what to do with the comma, the apostrophes and the exclamation mark, and GPT-2 usually keeps common short words whole but splits rare long ones. You compare your guesses with the measured numbers in Step 4.2.

```python
S1 = "Unbelievably, the students' robots weren't broken!"
guess_word = None     # YOUR TURN: how many word-level tokens (words and punctuation marks counted separately)?
guess_char = None     # YOUR TURN: how many characters (letters, spaces and punctuation, all counted)?
guess_gpt2 = None     # YOUR TURN: how many GPT-2 tokens?
print("Your guesses  word:", guess_word, "| char:", guess_char, "| gpt2:", guess_gpt2)
print("The sentence has", len(S1.split()), "space-separated words.")
```

1. **Step 4.1. Three granularities by hand.** First we cut text ourselves. A **word tokenizer** here is a regular expression that keeps a word (with an inner apostrophe, like *weren't*) together and makes every punctuation mark its own token. A **character tokenizer** is simply `list(text)`. For four sentences (plain English, one with typing mistakes, one with brand-new words, and the Imagine-first sentence) the code counts the tokens, and also the number of *distinct* units, which is the size of the vocabulary this sentence alone would need. Then we build a word-level vocabulary from `SCHOOL_TEXT`, with the special entry `<unk>` for every word it has never seen, and count how many `<unk>` appear in other texts.

```python
SENTS = {
    "plain": "The students test the robots on the track.",
    "typo":  "The studnets tset the robtos on the track.",
    "new":   "The students zorbified the hyperquantum robots.",
    "S1":    S1,
}

def word_tokens(text):
    """Words (an inner apostrophe stays inside the word) and single punctuation marks."""
    return re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?|[^\sA-Za-z0-9]", text)

def char_tokens(text):
    """Every character is a token, spaces included."""
    return list(text)

print(f"{'sentence':8s} {'words':>5s} {'word tok':>8s} {'distinct':>8s} {'char tok':>8s} {'distinct':>8s}")
for name, text in SENTS.items():
    w, c = word_tokens(text), char_tokens(text)
    print(f"{name:8s} {len(text.split()):5d} {len(w):8d} {len(set(w)):8d} {len(c):8d} {len(set(c)):8d}")
print("S1 word tokens:", word_tokens(S1))
RESULTS["p4_S1_word_char"] = (len(word_tokens(S1)), len(char_tokens(S1)))
```

```python
school_vocab = ["<unk>"] + sorted(set(words_of(SCHOOL_TEXT)))     # id 0 is <unk>
word_to_id = {w: i for i, w in enumerate(school_vocab)}
print("word-level vocabulary built from SCHOOL_TEXT:", len(school_vocab), "entries (including <unk>)")

def encode_words(text, to_id=word_to_id):
    """Word-level ids; every word outside the vocabulary becomes 0, which is <unk>."""
    return [to_id.get(w, 0) for w in words_of(text)]

def unk_count(text):
    ids = encode_words(text)
    return sum(1 for i in ids if i == 0), len(ids)

for name, text in [("SCHOOL_TEST", SCHOOL_TEST)] + list(SENTS.items()):
    u, n = unk_count(text)
    print(f"{name:12s} {u} unknown of {n} tokens  ->  {encode_words(text)[:12]}")
RESULTS["p4_unk_new"] = unk_count(SENTS["new"])[0]
```

   **What you should see.** The first table (your numbers must be identical, nothing here is random):

   @@OUT 4.1a 1-6
   The *plain* sentence has 8 space-separated words but 9 word tokens, because the full stop is a token of its own. The Imagine-first sentence has 6 words, 9 word tokens (the comma, the apostrophe after *students* and the exclamation mark are tokens of their own, while *weren't* stays together) and 50 characters. Typing mistakes change nothing at the character level, but at the word level they are new units. The distinct-unit columns show the trade-off already: a character vocabulary stays tiny (at most a few dozen symbols) while a word vocabulary grows with every new word.

   The vocabulary and the unknown counts:

   @@OUT 4.1b
   The word vocabulary of `SCHOOL_TEXT` has only 82 words (83 entries with `<unk>`). `SCHOOL_TEST` was written with the same words, so it has no unknown word at all, and so does the *plain* sentence. The sentence with mistakes loses 3 of its 9 tokens to `<unk>` (*studnets*, *tset* and *robtos*) and the sentence with new words loses 2 (*zorbified* and *hyperquantum*). A word-level model cannot tell *tset* from *zorbified*: both are the same id 0.

   **Try also (optional):**

   - Add a fifth sentence with a name from your town, with an emoji, and with a number such as 3,450. Which tokenizer notices them?
   - Change `word_tokens` so that it also lower-cases everything. How many distinct word tokens does `plain` have now?
   - Print `school_vocab[:15]` and `school_vocab[-5:]`: the ids are just the positions in the sorted list. Is id 7 "bigger" than id 3 in any meaningful way?

   *Exercise 4.1:* Using the word-level vocabulary of `SCHOOL_TEXT` and the function `unk_count`, find how many `<unk>` the *typo* sentence and the *new* sentence produce, and how many *distinct* characters the *new* sentence needs (use `char_tokens`). Why it matters: an unknown word carries no information, while a character-level model can still read a word it has never seen.

```python
n_unk_typo = None      # YOUR CODE: number of <unk> in SENTS["typo"] (use unk_count)
n_unk_new = None       # YOUR CODE: number of <unk> in SENTS["new"]
n_chars_new = None     # YOUR CODE: number of DISTINCT characters in SENTS["new"]
if None in (n_unk_typo, n_unk_new, n_chars_new):
    print("Fill in your three numbers above, then run this cell again.")
else:
    check("typo sentence: unknown words", n_unk_typo == unk_count(SENTS["typo"])[0], "call unk_count(SENTS['typo'])[0]")
    check("new sentence: unknown words", n_unk_new == unk_count(SENTS["new"])[0], "call unk_count(SENTS['new'])[0]")
    check("distinct characters of the new sentence", n_chars_new == len(set(char_tokens(SENTS["new"]))), "len(set(char_tokens(...)))")
```

   *What it means.* The typo sentence has 3 unknown words, the new-words sentence has 2, and the new sentence needs 21 different characters. A word-level vocabulary is blind to everything outside its list, while a character vocabulary of a few dozen symbols covers any English sentence. The price of characters is length: 47 character tokens instead of 7 word tokens.

2. **Step 4.2. Real tokenizers: GPT-2 and BERT.** Now the real thing. `load_tokenizer("gpt2")` gives the byte-level BPE tokenizer of GPT-2 (vocabulary 50,257) and `load_tokenizer("bert-base-uncased")` gives the WordPiece tokenizer of BERT (vocabulary 30,522, all lower-case). We show the pieces they make for the four sentences. Look for the marks: in GPT-2 the letter `Ġ` at the start of a piece means *this piece started after a space*; in BERT the prefix `##` means *this piece continues the previous one, with no space*. Then compare the measured numbers with your Imagine-first guesses.

```python
def load_both():
    """Load the two tokenizers; if the files are not available, say so clearly instead of crashing."""
    try:
        return load_tokenizer("gpt2"), load_tokenizer("bert-base-uncased")
    except Exception as e:
        print("Could not load the tokenizers (" + type(e).__name__ + ").")
        print("On the Spark, ask your instructor to check the offline cache; in Colab, check the internet connection and run this cell again.")
        return None, None

gpt2, bert = load_both()

def need_tokenizers():
    if gpt2 is None or bert is None:
        raise RuntimeError("The tokenizers are not loaded: fix the message printed by the loading cell, then run it again.")

need_tokenizers()
print("vocabulary sizes: GPT-2", len(gpt2), "| BERT", len(bert))
def pieces(tok, text):
    """The token pieces of `text` (without the special [CLS] and [SEP] tokens)."""
    return tok.convert_ids_to_tokens(tok(text, add_special_tokens=False)["input_ids"])

print("GPT-2 on S1:")
ids_s1 = show_tokens(gpt2, S1)
print("BERT on S1:", pieces(bert, S1))
```

```python
print(f"{'sentence':8s} {'words':>5s} {'GPT-2':>6s} {'BERT':>5s} {'chars':>6s}")
counts = {}
for name, text in SENTS.items():
    counts[name] = (len(pieces(gpt2, text)), len(pieces(bert, text)))
    print(f"{name:8s} {len(text.split()):5d} {counts[name][0]:6d} {counts[name][1]:5d} {len(text):6d}")
for name in ["typo", "new"]:
    print(name, "GPT-2:", pieces(gpt2, SENTS[name]))
    print(name, "BERT :", pieces(bert, SENTS[name]))
print("Your guesses were  word:", guess_word, "char:", guess_char, "gpt2:", guess_gpt2,
      "| measured  word:", len(word_tokens(S1)), "char:", len(S1), "gpt2:", counts["S1"][0])
RESULTS["p4_S1_tokens"] = {"word": len(word_tokens(S1)), "char": len(S1), "gpt2": counts["S1"][0], "bert": counts["S1"][1]}
assert len(gpt2) == 50257 and len(bert) == 30522, "unexpected vocabulary sizes"
```

   **What you should see.** The loading cell prints no message (if you see "Could not load", read it and ask for help), then:

   @@OUT 4.2a 1-4
   The Imagine-first sentence needs **13 GPT-2 tokens** and **15 BERT tokens**: far more than its 6 words, far fewer than its 50 characters. GPT-2 cuts the rare word into `Un`, `bel`, `iev`, `ably`, keeps `Ġthe`, `Ġstudents` and `Ġrobots` whole (each with the leading-space mark `Ġ`), and splits the contraction into `Ġweren` and `'t`. BERT writes everything in lower-case, cuts *unbelievably* into five pieces (`un ##bel ##ie ##va ##bly`) and cuts *weren't* into three tokens (`weren`, `'` and `t`).

   @@OUT 4.2b
   The sentences with ordinary words need about one token per word (9 for the *plain* sentence in both tokenizers, which counts the full stop). The typing mistakes cost GPT-2 13 tokens instead of 9 and BERT 12 instead of 9: *studnets* and *robtos* become several small pieces. The two invented words cost GPT-2 11 tokens and BERT 13 for 6 words. Neither tokenizer ever produces an "unknown" for these words, which is the point of sub-word tokenisation: any word can be spelled with pieces.

   **Try also (optional):**

   - Print `gpt2.convert_ids_to_tokens([0, 1, 2, 50256])` and `bert.convert_ids_to_tokens([0, 100, 101, 102, 103])` to see the special tokens at the ends of the vocabularies.
   - Look at the ids of the three occurrences of the word *the* in the *plain* sentence with `show_tokens(gpt2, SENTS["plain"])`. Is the id of `The` (capital, no space) the same as the id of `Ġthe`?
   - Tokenise the same sentence in capitals with `pieces(gpt2, S1.upper())`. What happens to the count?

   *Exercise 4.2:* *Fertility* is the number of tokens a tokenizer needs per word. Compute it as the number of tokens divided by the number of space-separated words (`len(text.split())`) for the *plain* and the *new* sentence, with both tokenizers, and decide which sentence is more expensive for GPT-2. Why it matters: fertility is how you compare tokenizers fairly, and it turns directly into cost and into how much text fits in the model's context.

```python
fert_plain_gpt2 = None    # YOUR CODE: tokens per word of SENTS["plain"] with gpt2
fert_new_gpt2 = None      # YOUR CODE: tokens per word of SENTS["new"] with gpt2
fert_new_bert = None      # YOUR CODE: tokens per word of SENTS["new"] with bert
costlier = ""             # YOUR ANSWER: "plain" or "new" (for GPT-2)
if None in (fert_plain_gpt2, fert_new_gpt2, fert_new_bert) or not costlier:
    print("Fill in your three numbers and `costlier` above, then run this cell again.")
else:
    f = lambda tok, k: len(pieces(tok, SENTS[k])) / len(SENTS[k].split())
    check("plain, GPT-2", abs(fert_plain_gpt2 - f(gpt2, "plain")) < 0.01, "tokens / words = " + str(len(pieces(gpt2, SENTS['plain']))) + " / " + str(len(SENTS['plain'].split())))
    check("new, GPT-2", abs(fert_new_gpt2 - f(gpt2, "new")) < 0.01, "count the pieces and divide by the number of words")
    check("new, BERT", abs(fert_new_bert - f(bert, "new")) < 0.01, "same with the BERT tokenizer")
    check("which sentence is costlier", costlier == ("new" if f(gpt2, "new") > f(gpt2, "plain") else "plain"), "compare the two fertilities")
```

   *What it means.* The plain sentence costs GPT-2 9 tokens for 8 words (1.125 tokens per word, the full stop being a token). The sentence with invented words costs 11 tokens for 6 words (1.83), and BERT needs 13 for the same 6 words (2.17). Familiar words are cheap, rare words are expensive: the cost depends on how often the tokenizer's training text contained the piece.

3. **Step 4.3. The round trip: encode, then decode.** A good tokenizer is **reversible**: `decode(encode(text))` should give back exactly the text you started with, because the model's answer is a list of ids that must be turned into text again. We test four awkward texts (a doubled space and a newline, accents, a number with a comma, and an emoji) with both tokenizers. GPT-2 works on bytes, so it can encode any text. BERT lower-cases, strips accents and has no entry for many characters, so its round trip is lossy.

```python
texts = ["Hello   World\nnew line", "Café Müller costs 5 pesos!", "The contest has 12 teams and 3,450 points.", "I love robots 🤖"]
for t in texts:
    back_gpt2 = gpt2.decode(gpt2.encode(t))
    back_bert = bert.decode(bert.encode(t, add_special_tokens=False))
    print(repr(t))
    print("   GPT-2 round trip exact:", back_gpt2 == t)
    print("   BERT gives back       :", repr(back_bert))
assert all(gpt2.decode(gpt2.encode(t)) == t for t in texts), "GPT-2 should rebuild every text exactly"
RESULTS["p4_gpt2_roundtrip"] = True
```

   **What you should see.** GPT-2 rebuilds all four texts exactly (the assert passes), including the three spaces, the newline, the accented letters and the emoji. BERT returns `'hello world new line'` (lower-case, the extra spaces and the newline gone), `'cafe muller costs 5 pesos!'` (the accents removed), `'the contest has 12 teams and 3, 450 points.'` (a space inserted after the comma of 3,450, and the capital letter gone) and `'i love robots [UNK]'` (the emoji has no entry, so it is replaced by the unknown token). BERT was designed to understand sentences, not to write them back, so its tokenizer throws information away on purpose.

   **Try also (optional):**

   - Encode with `bert.encode(t)` (special tokens included) and print the first and last ids: they are `[CLS]` (101) and `[SEP]` (102).
   - Use `gpt2(texts[0])["input_ids"]` and `gpt2.decode([id])` for every id, one at a time. Which piece holds the newline?
   - Try `gpt2.decode(gpt2.encode(t)[:3])` on a sentence: the pieces can stop in the middle of a word.

   *Exercise 4.3:* Write a short text of your own (any language you like) for which the BERT round trip differs from the text but the GPT-2 round trip is exact. Why it matters: before you trust a tokenizer for a task that must keep the original text (a translation, a quote, a name), you test whether it loses anything.

```python
my_text = ""      # YOUR TEXT: something BERT (uncased) cannot reproduce exactly, e.g. with capitals, accents or an emoji
if not my_text:
    print("Type a text in `my_text` above, then run this cell again.")
else:
    bert_ok = bert.decode(bert.encode(my_text, add_special_tokens=False)) == my_text
    gpt2_ok = gpt2.decode(gpt2.encode(my_text)) == my_text
    check("GPT-2 gives your text back exactly", gpt2_ok, "this should be true for any text")
    check("BERT does NOT give it back", not bert_ok, "BERT is lower-case only: add a capital letter, an accent or an emoji")
```

   *What it means.* Any capital letter is enough, because bert-base-uncased lower-cases everything; accents and emoji lose even more. GPT-2 reads bytes, so every text survives the round trip. Keeping the original text is the reason many modern models use byte-level tokenizers.

4. **Step 4.4. Quirks: numbers, spaces, capitals, code, emoji and other languages.** Tokenizers were trained on text from the internet, so they treat familiar things well and unfamiliar things in surprising ways. This step lists twelve small texts and prints the GPT-2 pieces and the BERT pieces side by side: a long number, the same word with and without a leading space and with capitals, a line of Python, an emoji, three short greetings in Filipino and Cebuano, and the word *strawberry*. Read the output slowly; the point is to notice, not to memorise. ==Only the pieces printed on your screen are facts here: do not assume anything about other words you have not tested.==

```python
quirks = [
    ("long number", "3141592653589793"),
    ("no space", "hello"), ("leading space", " hello"), ("capital", "Hello"), ("all capitals", "HELLO"),
    ("python", "for i in range(10): print(i**2)"),
    ("emoji", "I love robots 🤖"),
    ("Filipino 1", "Kumusta ka na?"), ("Cebuano", "Maayong buntag"), ("Filipino 2", "Salamat po"),
    ("strawberry", "strawberry"),
]
print(f"{'case':14s} {'GPT-2':>5s} {'BERT':>5s}   GPT-2 pieces")
for name, text in quirks:
    pg, pb = pieces(gpt2, text), pieces(bert, text)
    print(f"{name:14s} {len(pg):5d} {len(pb):5d}   {pg}")
print("BERT pieces of the number   :", pieces(bert, "3141592653589793"))
print("BERT pieces of the emoji    :", pieces(bert, "I love robots 🤖"))
print("BERT pieces of 'Kumusta ka na?':", pieces(bert, "Kumusta ka na?"))
```

   **What you should see.** The table (the GPT-2 pieces are shown in full):

   @@OUT 4.4 1-12
   Read it like a detective:

   - The 16-digit number is cut into six pieces of one to three digits, with no regard for place value; BERT cuts it into ten.
   - *hello* is one token in every spelling, but the leading space gives a *different* token (`Ġhello`), and `HELLO` in capitals needs three tokens (`HE`, `LL`, `O`). For the model these are different symbols, so it must learn separately that they mean the same.
   - The line of Python becomes 13 tokens; the two asterisks of `i**2` stay one token in GPT-2 and two in BERT.
   - In the emoji sentence GPT-2 spends three of its six tokens on the robot emoji (`ĠðŁ`, `¤`, `ĸ`: pieces of its bytes), and BERT writes `[UNK]`.
   - The Filipino and Cebuano greetings are cut into short pieces that carry little meaning in themselves: `Kumusta ka na?` is 7 tokens (`K`, `um`, `ust`, `a`, `Ġka`, `Ġna`, `?`), `Maayong buntag` is 6 and `Salamat po` is 4. This does not say anything about the quality of those languages; it says GPT-2's tokenizer saw relatively little of them.
   - *strawberry* is `st`, `raw`, `berry`: a model that sees three tokens never sees the letters one by one, which is one reason why language models can be unreliable at counting letters.

   **Try also (optional):**

   - Test your own name, the name of your town and the name of your school. Are they one token or several?
   - Compare `pieces(gpt2, "2024")`, `pieces(gpt2, "12345")` and `pieces(gpt2, "1,000,000")`. How are numbers cut?
   - Compare "colour" and "color", "photo" and "photograph". Does the spelling change the count?

   *Exercise 4.4:* Compare three Filipino and Cebuano greetings with their English meanings. For GPT-2, compute the total number of tokens of `["Kumusta ka na?", "Maayong buntag", "Salamat po"]` and of `["How are you?", "Good morning", "Thank you"]`, and the ratio of the first total to the second. Then write one sentence about what surprised you (only about what you measured). Why it matters: if the same message costs twice as many tokens in one language, it costs twice as much, fills the context twice as fast and gives the model less to work with.

```python
tokens_fil = None      # YOUR CODE: total GPT-2 tokens of the three Filipino / Cebuano greetings
tokens_eng = None      # YOUR CODE: total GPT-2 tokens of the three English greetings
ratio = None           # YOUR CODE: tokens_fil / tokens_eng
surprise = ""          # YOUR ANSWER: one sentence about what surprised you (about measured numbers only)
if None in (tokens_fil, tokens_eng, ratio) or not surprise:
    print("Fill in your three numbers and the sentence above, then run this cell again.")
else:
    fil = ["Kumusta ka na?", "Maayong buntag", "Salamat po"]; eng = ["How are you?", "Good morning", "Thank you"]
    tf, te = sum(len(pieces(gpt2, s)) for s in fil), sum(len(pieces(gpt2, s)) for s in eng)
    check("Filipino and Cebuano total", tokens_fil == tf, "add the three counts from the table")
    check("English total", tokens_eng == te, "count pieces of the three English greetings")
    check("ratio", abs(ratio - tf / te) < 0.01, "tokens_fil / tokens_eng")
    check("your sentence has at least 8 words", len(surprise.split()) >= 8, "write a real sentence")
    RESULTS["p4_fil_eng_ratio"] = round(tf / te, 3)
```

   *What it means.* GPT-2 needs 17 tokens for the three Filipino and Cebuano greetings and 8 for the three English ones: a ratio of 2.125, so the same short messages cost a little more than twice as much. This is a measurement of one tokenizer on three short texts, not a general law, but it shows why the choice of tokenizer is also a question of fairness.

5. **Step 4.5. Token cost and the three-quarters rule of thumb.** Week 1 gave a rule of thumb: one token is about three quarters of an English word. You now test it. First, five very different texts (an ordinary sentence, a technical sentence, a Filipino sentence, a line of code and a sentence with numbers) go through both tokenizers; a bar chart shows the tokens per space-separated word. Then you measure the ratio words per token on all of `SCHOOL_TEXT`, which is plain English about a school club.

```python
costs = {
    "ordinary": "The students test the robots on the track.",
    "technical": "Backpropagation computes gradients with the chain rule.",
    "Filipino": "Masaya ang mga estudyante sa robotics club.",
    "code": "for i in range(10): print(i**2)",
    "numbers": "The contest has 12 teams and 3,450 points.",
}
tpw = {k: (len(pieces(gpt2, t)) / len(t.split()), len(pieces(bert, t)) / len(t.split())) for k, t in costs.items()}
for k, (g, b) in tpw.items():
    print(f"{k:10s} GPT-2 {g:5.2f}   BERT {b:5.2f}   tokens per word")
fig, ax = plt.subplots(figsize=(7, 3.2))
xs = np.arange(len(tpw))
ax.bar(xs - 0.2, [v[0] for v in tpw.values()], 0.4, label="GPT-2")
ax.bar(xs + 0.2, [v[1] for v in tpw.values()], 0.4, label="BERT")
ax.set_xticks(xs); ax.set_xticklabels(list(tpw)); ax.set_ylabel("tokens per word"); ax.legend(); ax.axhline(1, color="gray", ls=":")
plt.tight_layout(); plt.show()

n_words = len(SCHOOL_TEXT.split())
n_tok = len(gpt2(SCHOOL_TEXT, add_special_tokens=False)["input_ids"])
print(f"SCHOOL_TEXT: {n_words} words, {n_tok} GPT-2 tokens -> {n_words / n_tok:.3f} words per token ({n_tok / n_words:.3f} tokens per word)")
RESULTS["p4_words_per_token"] = round(n_words / n_tok, 3)
```

   **What you should see.**

   @@OUT 4.5
   The ordinary sentence costs about one token per word (1.12 with GPT-2, because of the full stop); the technical sentence costs 1.86 because *backpropagation* is cut into pieces; the Filipino sentence costs 1.86 with GPT-2 and 1.71 with BERT; the sentence with numbers costs 1.38; the line of code costs 2.6 tokens per space-separated chunk with GPT-2 and 3.0 with BERT (13 and 15 tokens for 5 chunks). On the plain English text of the club, the ratio is **0.917 words per token** (199 words, 217 tokens): higher than the rule of thumb's 0.75, because this text is made of short, very common words. The rule of thumb is an average over many kinds of text; for technical, multilingual or code-heavy text the number of tokens per word is larger, so a limit of 1,000 tokens holds fewer words.

   **Try also (optional):**

   - Repeat the ratio for `SCHOOL_TEST` and for one paragraph of your own writing in English. Does 0.917 hold?
   - Estimate how many tokens a 500-word essay costs at 0.75 and at 0.917 words per token. How big is the difference?
   - Tokenise a paragraph of Filipino (for example the first lines of a song or a news article that you may freely copy) and compute the same ratio.

   *Exercise 4.5:* Use the ratio you measured on `SCHOOL_TEXT` (words per token) to estimate how many GPT-2 tokens a 1,000-word essay in this simple style needs, and compare with the rule of thumb of 0.75 words per token. Why it matters: prices and context limits of language models are quoted in tokens, so you must be able to convert.

```python
tokens_measured_rule = None    # YOUR CODE: tokens for 1000 words if the ratio is the one measured on SCHOOL_TEXT (round to a whole number)
tokens_three_quarters = None   # YOUR CODE: tokens for 1000 words at 0.75 words per token (round to a whole number)
if None in (tokens_measured_rule, tokens_three_quarters):
    print("Fill in your two numbers above, then run this cell again.")
else:
    ratio_wpt = n_words / n_tok
    check("with the measured ratio", abs(tokens_measured_rule - 1000 / ratio_wpt) <= 2, "1000 / (words per token)")
    check("with the three-quarters rule", abs(tokens_three_quarters - 1000 / 0.75) <= 2, "1000 / 0.75")
```

   *What it means.* With 0.917 words per token a 1,000-word essay needs about 1,090 tokens, with the three-quarters rule about 1,333. The rule of thumb is a safe, slightly pessimistic estimate for English prose; for another language or for code, measure with the real tokenizer.

### Part 5. Build a BPE tokenizer from scratch (30 min)



Byte-pair encoding (BPE) is the algorithm that produced the pieces you saw in Part 4, and it is surprisingly short. You start with every word spelled as single characters plus an end-of-word mark `</w>`; then you repeat one move: count which two neighbouring symbols occur together most often (weighted by how often each word occurs) and glue them into one new symbol. The list of merges, in the order they were learned, **is** the tokenizer. In this part you write the whole algorithm yourself: counting pairs, merging, learning, encoding a new word and decoding. You check your merges against the lesson's own example, train on a bigger text, and compare with the real GPT-2. ==You will have written a working sub-word tokenizer in about forty lines of plain Python.==

The lesson's corpus is `low 5, lower 2, newest 6, widest 3`: four words, with how many times each occurs. The rules are the lesson's: a word is split into characters and ends with `</w>`; the pair with the highest total count is merged; ties are broken by the pair that appears first when you scan the words in order; and a merge is made only if its count is at least 2 (a pair that occurs once teaches the tokenizer nothing it could reuse).

**Imagine first.** Write the corpus on paper as characters: `l o w </w>` (five times), `l o w e r </w>` (twice), `n e w e s t </w>` (six times), `w i d e s t </w>` (three times). Count the neighbouring pairs yourself for a minute (for example, how many times does `e s` occur?). Then write down the **first three merges** you expect, as pairs of symbols with your guess of the count, for example `("e", "s")`. Hint: a pair counts once for every word it occurs in, multiplied by how many times that word occurs. You compare with the program in Step 5.2.

```python
LESSON_CORPUS = [("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)]
guess_merges = []     # YOUR TURN: your three guesses, e.g. [("e", "s"), ("l", "o"), ("n", "e")]
print("Your guesses:", guess_merges if guess_merges else "(nothing yet)")
```

1. **Step 5.1. Count the adjacent pairs.** We represent the corpus as a dictionary that maps a word, written as a tuple of symbols, to its count. `split_word("low")` gives `("l", "o", "w", "</w>")`. The function `pair_counts` slides over every word and adds the word's count to each neighbouring pair. We print the five biggest pairs and look at the tie. Python's `max` returns the first of several equal maxima in the order the pairs were first seen, which is exactly the lesson's tie-break rule.

```python
def split_word(word):
    """'low' -> ('l', 'o', 'w', '</w>'): characters plus the end-of-word mark."""
    return tuple(word) + ("</w>",)

def make_words(corpus):
    """[(word, count), ...] -> {symbols tuple: count}."""
    return {split_word(w): c for w, c in corpus}

def pair_counts(words):
    """Count every pair of neighbouring symbols, weighted by the count of its word."""
    pc = Counter()
    for symbols, c in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pc[(a, b)] += c
    return pc

words0 = make_words(LESSON_CORPUS)
print(words0)
pc = pair_counts(words0)
for pair, c in pc.most_common(5):
    print(pair, c)
top = max(pc.values())
print("pairs with the top count", top, ":", [p for p, c in pc.items() if c == top])
print("number of distinct pairs:", len(pc))
```

   **What you should see.**

   @@OUT 5.1
   Three pairs tie at **9**: `("e", "s")` (6 times in *newest*, 3 in *widest*), `("s", "t")` and `("t", "</w>")`. The scan meets `e s` first (in *newest*, before the others), so the first merge is `e s`. Then come `("w", "e")` with 8 (6 in *newest* plus 2 in *lower*) and `("l", "o")` with 7. Notice that a pair such as `("w", "</w>")` counts only the 5 occurrences of *low*. There are 14 distinct pairs in all.

   **Try also (optional):**

   - Change the count of *lower* from 2 to 10 and run again. Which pair becomes the top one?
   - Add the word `("slow", 4)` to the corpus. Does the first merge change?
   - Print `pc[("w", "e")]` and explain the number 8 with the words in which `w e` occurs.

   *Exercise 5.1:* Without printing the whole counter, find with `pair_counts` how many times the pair `("w", "e")` occurs, how many distinct pairs have the top count, and compute by hand (then check) the count of `("o", "w")`. Why it matters: counts are the only thing BPE looks at; if you can read them, you can predict the algorithm.

```python
count_we = None        # YOUR CODE: the count of the pair ("w", "e")
n_tied = None          # YOUR CODE: how many pairs share the highest count?
count_ow = None        # YOUR CODE: the count of ("o", "w") (think: which words contain o followed by w?)
if None in (count_we, n_tied, count_ow):
    print("Fill in your three numbers above, then run this cell again.")
else:
    pc_ = pair_counts(make_words(LESSON_CORPUS))
    check("count of (w, e)", count_we == pc_[("w", "e")], "look at newest and lower")
    check("number of tied pairs", n_tied == sum(1 for c in pc_.values() if c == max(pc_.values())), "count the pairs whose count equals the maximum")
    check("count of (o, w)", count_ow == 7, "low (5) and lower (2) both contain o w")
```

   *What it means.* `w e` occurs in *newest* (6) and *lower* (2): 8. Three pairs tie at 9. `o w` occurs in *low* (5) and *lower* (2): 7, the same as `l o`, because every `o` here sits between an `l` and a `w`.

2. **Step 5.2. One merge, and then three more.** `merge_pair(words, pair)` replaces every neighbouring `a b` in every word by the single symbol `ab`. We apply it four times in a loop, printing the pair we merged and its count, and the new spelling of the four words. At the end we assert that the first four merges are the lesson's: `e s` (9), `es t` (9), `est </w>` (9) and `l o` (7).

```python
def merge_pair(words, pair):
    """Replace every occurrence of the neighbouring symbols `pair` by one new symbol."""
    a, b = pair
    new = {}
    for symbols, c in words.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        new[tuple(out)] = new.get(tuple(out), 0) + c
    return new

words = make_words(LESSON_CORPUS)
history = []
for step in range(1, 5):
    pc = pair_counts(words)
    best = max(pc, key=pc.get)             # the first pair with the highest count
    history.append((best[0], best[1], pc[best]))
    words = merge_pair(words, best)
    print(f"merge {step}: {best}  count {pc[best]}")
    for symbols, c in words.items():
        print(f"      {c} x  {' '.join(symbols)}")
lesson_first4 = [("e", "s", 9), ("es", "t", 9), ("est", "</w>", 9), ("l", "o", 7)]
assert history == lesson_first4, "your first four merges should match the lesson"
print("The first four merges match the lesson.")
print("Your guesses:", guess_merges, "| the first three merges:", [(a, b) for a, b, _ in history[:3] ])
RESULTS["p5_first_merge"] = history[0]
```

   **What you should see.**

   @@OUT 5.2
   After merge 1 the words *newest* and *widest* contain `es`; after merge 2 they contain `est`; after merge 3 the whole ending `est</w>` is one symbol, which is what makes the end-of-word mark useful: `est</w>` means *the ending est*, not the middle of a word. Merge 4 joins `l o`, and *low* and *lower* both start with `lo`. The assert passes. Compare with your guesses: few people guess `e s` first, because the three pairs `e s`, `s t` and `t </w>` tie, and the order in which the words are scanned decides.

   **Try also (optional):**

   - Break the tie differently: replace `max(pc, key=pc.get)` by `max(pc, key=lambda p: (pc[p], p))` (largest, then alphabetical order) and print the first four merges. Do they differ from the lesson's? What does it tell you about reproducing a tokenizer?
   - Print `len(words)` after every merge: does the number of different words ever change?
   - Run five more merges by hand on paper and then in code.

   *Exercise 5.2:* Using the dictionary `words` after the four merges (the variable keeps its value), compute the total number of tokens the corpus needs when each word counts as often as it occurs, and the average number of tokens per word (tokens divided by the 16 word occurrences). Why it matters: this average is the quantity a tokenizer tries to lower by learning merges.

```python
total_tokens_4 = None      # YOUR CODE: sum over the words of (number of symbols) * (count)
tokens_per_word_4 = None   # YOUR CODE: total_tokens_4 divided by the total number of word occurrences
if None in (total_tokens_4, tokens_per_word_4):
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("total tokens after 4 merges", total_tokens_4 == 61, "low: 3 symbols x 5, lower: 5 x 2, newest: 4 x 6, widest: 4 x 3")
    check("tokens per word", abs(tokens_per_word_4 - 61 / 16) < 1e-9, "61 / 16")
```

   *What it means.* After four merges *low* is `lo w </w>` (3 symbols), *lower* is `lo w e r </w>` (5), *newest* is `n e w est</w>` (4) and *widest* is `w i d est</w>` (4): 15 + 10 + 24 + 12 = 61 tokens for 16 words, 3.8125 per word. Before any merge the same corpus needs 95 tokens (5.9375 per word).

3. **Step 5.3. Learn the merges: the whole algorithm.** Now we put the loop into one function. `learn_bpe(corpus, num_merges)` repeats count, pick, merge until it has made `num_merges` merges or no pair occurs at least twice. It returns the list of merges (as `(left, right, count)`), the final spelling of the words and the average number of tokens per word after every merge. The size of the vocabulary of the tokenizer is the number of base symbols plus the number of merges: each merge adds exactly one new symbol.

```python
def tokens_per_word(words):
    """Average number of symbols per word occurrence."""
    return sum(len(s) * c for s, c in words.items()) / sum(words.values())

def learn_bpe(corpus, num_merges, min_count=2):
    """Learn up to num_merges merges. Returns (merges, final_words, curve) where curve[m] is the tokens per word after m merges."""
    words = make_words(corpus)
    merges, curve = [], [tokens_per_word(words)]
    for _ in range(num_merges):
        pc = pair_counts(words)
        if not pc:
            break
        best = max(pc, key=pc.get)
        if pc[best] < min_count:
            break
        merges.append((best[0], best[1], pc[best]))
        words = merge_pair(words, best)
        curve.append(tokens_per_word(words))
    return merges, words, curve

merges10, words10, curve10 = learn_bpe(LESSON_CORPUS, 10)
for i, (a, b, c) in enumerate(merges10, 1):
    print(f"{i:2d}. {a} + {b} -> {a + b}   (count {c})")
base_symbols = sorted({s for w, _ in LESSON_CORPUS for s in split_word(w)})
print("base symbols:", base_symbols, "->", len(base_symbols))
vocab_size = len(base_symbols) + len(merges10)
print("vocabulary size = base symbols + merges =", len(base_symbols), "+", len(merges10), "=", vocab_size)
print("tokens per word after 0, 4, 10 merges:", [round(curve10[m], 3) for m in (0, 4, 10)])
print("final spelling:", {" ".join(s): c for s, c in words10.items()})
assert vocab_size == 21 and len(merges10) == 10
RESULTS["p5_vocab_size_10"] = vocab_size
```

   **What you should see.**

   @@OUT 5.3
   The ten merges are the lesson's: `e s`, `es t`, `est </w>` (9 each), `l o`, `lo w` (7), `n e`, `ne w`, `new est</w>` (6), `low </w>` (5) and `w i` (3). There are 11 base symbols (ten letters and the end mark), so the vocabulary has 11 + 10 = **21** entries. The average falls from 5.9375 tokens per word (all characters) to 3.8125 after four merges, and to 1.75 after ten: *low*, *newest* and *widest* are almost whole words now.

   **Try also (optional):**

   - Ask for 100 merges. How many does the function really make, and why does it stop? (Hint: look at the counts of the last merges.)
   - Set `min_count=1`. Does the average tokens per word reach 1.0? Why is that a bad idea with a large corpus?
   - Print `curve10`: is the decrease from merge to merge always the same size?

   *Exercise 5.3:* Ask `learn_bpe` for 100 merges on the lesson corpus. Report how many merges it makes, the tokens per word at the end, and the vocabulary size (base symbols plus merges). Then say in one sentence why it cannot make more. Why it matters: a merge budget is an upper bound; what the text allows depends on the counts.

```python
n_made = None          # YOUR CODE: how many merges does learn_bpe(LESSON_CORPUS, 100) really make?
final_tpw = None       # YOUR CODE: tokens per word at the end
final_vocab = None     # YOUR CODE: base symbols + merges
why_stop = ""          # YOUR ANSWER: one sentence
if None in (n_made, final_tpw, final_vocab) or not why_stop:
    print("Fill in your three numbers and the sentence above, then run this cell again.")
else:
    m_, w_, c_ = learn_bpe(LESSON_CORPUS, 100)
    check("number of merges", n_made == len(m_), "len(merges)")
    check("tokens per word at the end", abs(final_tpw - c_[-1]) < 1e-9, "the last element of the curve")
    check("vocabulary size", final_vocab == len(base_symbols) + len(m_), "11 base symbols + the merges")
    check("your sentence has at least 8 words", len(why_stop.split()) >= 8, "write a real sentence")
```

   *What it means.* It makes 15 merges and stops when every word is one symbol: the tokens per word are 1.0 and the vocabulary has 11 + 15 = 26 entries. The last three merges (`low e`, `lowe r`, `lower </w>`) each have count 2, just enough to pass the threshold. With a real corpus you stop long before: you choose the number of merges.

4. **Step 5.4. Encode a new word, and decode.** The merges are a recipe. To encode a word, spell it in characters plus `</w>` and apply the merges **in the order they were learned**, each one wherever it fits. Words that were in the corpus come out as their learned pieces. A word that was not, such as *lowest*, is built from pieces that exist, and a word full of unknown letters falls back to characters. Decoding is simple: glue the pieces together and read `</w>` as the end of a word.

```python
def encode_bpe(word, merges):
    """Apply the merges, in the order they were learned, to one word. Returns the list of pieces."""
    symbols = list(split_word(word))
    for a, b, _ in merges:
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = out
    return symbols

def decode_bpe(pieces):
    """Glue the pieces and turn each </w> back into a space: the original text."""
    return "".join(pieces).replace("</w>", " ").strip()

for w in ["low", "lower", "lowest", "newer", "widen", "xyz"]:
    p = encode_bpe(w, merges10)
    print(f"{w:7s} -> {p}   decoded: {decode_bpe(p)!r}")
assert all(decode_bpe(encode_bpe(w, merges10)) == w for w in ["low", "lowest", "newer", "widen", "xyz"])
```

   **What you should see.**

   @@OUT 5.4
   *low* and *lower* were in the corpus: `low</w>` is one piece and *lower* is `low e r </w>`. The unseen word *lowest* becomes `low` + `est</w>`: two pieces that the tokenizer learned from *low* and *newest*, so it can say something sensible about a word it never saw. *newer* is `new e r </w>`, *widen* is `wi d e n </w>`, and *xyz* uses none of the merges and falls back to the characters `x y z </w>`. The decoding always gives the word back (the assert passes). Here `x`, `y` and `z` are not even in the base alphabet, which shows why real tokenizers start from bytes: a byte-level alphabet of 256 symbols can spell anything.

   **Try also (optional):**

   - Encode a short sentence by encoding each word separately and joining the pieces. Decode it.
   - Does the order of the merges matter? Apply the merges of `merges10` in reverse order to *newest* and compare.
   - Encode the word *flower*. Which merges fire, and which letters stay alone?

   *Exercise 5.4:* By hand, apply the ten merges to the unseen word *slowest* and write the list of its pieces. Then check with `encode_bpe`. Also answer: can this tokenizer ever produce an unknown token for a word made of letters it has seen? Why it matters: encoding a new word is how a tokenizer is used every second in a real model.

```python
pieces_slowest = []        # YOUR ANSWER: the list of pieces, e.g. ["a", "bc", "d</w>"]
unknown_possible = None    # YOUR ANSWER: True or False
if not pieces_slowest or unknown_possible is None:
    print("Fill in `pieces_slowest` and `unknown_possible` above, then run this cell again.")
else:
    check("pieces of slowest", pieces_slowest == encode_bpe("slowest", merges10), "apply e s, es t, est </w>, l o, lo w, ... in this order")
    check("no unknown token is needed", unknown_possible is False, "single characters are always available as a fallback")
```

   *What it means.* *slowest* starts as `s l o w e s t </w>`. The merges `e s`, `es t` and `est </w>` build `est</w>`; `l o` and `lo w` build `low`; the `s` has no partner and stays alone: `s`, `low`, `est</w>`. A tokenizer that keeps all single characters (or bytes) in its vocabulary can always fall back to them, so it never needs an unknown token.

5. **Step 5.5. A bigger corpus: SCHOOL_TEXT.** Four words are a toy. Now we learn BPE from all the words of `SCHOOL_TEXT` (82 different words, 215 word occurrences, counted with `words_of`, so the full stop counts as a word). We learn 60 merges, print the first 15 and some of the longest pieces, and measure the average number of tokens per word after 0, 10, 20, 40 and 60 merges on the training text. Then we measure the same thing on `SCHOOL_TEST`, a text the tokenizer was not trained on. The plot shows the central trade-off of the lesson: more merges give shorter sequences but a bigger vocabulary.

```python
wc = Counter(words_of(SCHOOL_TEXT))
school_corpus = list(wc.items())
print("different words:", len(school_corpus), "| word occurrences:", sum(wc.values()))
merges60, words60, curve60 = learn_bpe(school_corpus, 60)
print("merges made:", len(merges60))
for i, (a, b, c) in enumerate(merges60[:15], 1):
    print(f"{i:2d}. {a} + {b} -> {a + b}   (count {c})")
learned_pieces = sorted({s for syms in words60 for s in syms}, key=lambda s: (-len(s), s))
print("some of the longest pieces in use:", learned_pieces[:10])
```

```python
base_school = sorted({s for w, _ in school_corpus for s in split_word(w)})
print("base symbols:", len(base_school))
print(f"{'merges':>6s} {'vocabulary':>10s} {'tokens/word (train)':>20s}")
for m in (0, 10, 20, 40, 60):
    print(f"{m:6d} {len(base_school) + m:10d} {curve60[m]:20.3f}")

test_words = words_of(SCHOOL_TEST)
for m in (0, 60):
    n_tok = sum(len(encode_bpe(w, merges60[:m])) for w in test_words)
    print(f"SCHOOL_TEST with {m:2d} merges: {n_tok} tokens for {len(test_words)} words -> {n_tok / len(test_words):.3f} per word")

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(range(len(curve60)), curve60, color="C0")
ax.set_xlabel("number of merges"); ax.set_ylabel("tokens per word (training text)", color="C0")
ax2 = ax.twinx(); ax2.plot(range(len(curve60)), [len(base_school) + m for m in range(len(curve60))], color="C3")
ax2.set_ylabel("vocabulary size", color="C3")
plt.title("More merges: shorter sequences, bigger vocabulary"); plt.tight_layout(); plt.show()
RESULTS["p5_tpw_curve"] = {m: round(curve60[m], 3) for m in (0, 10, 20, 40, 60)}
assert curve60[0] > curve60[10] > curve60[20] > curve60[40] > curve60[60]
```

   **What you should see.**

   @@OUT 5.5a
   The first merges are the most common patterns of English in this text: the end `e</w>` (59 times), `th` (42), `the</w>` (41), the plural ending `s</w>` (36). After 60 merges, whole words like `students</w>`, `teachers</w>`, `robots</w>`, `every</w>` and `club</w>` are single pieces. Next, the trade-off table:

   @@OUT 5.5b 1-6
   The average falls from 5.079 tokens per word with characters only to 3.791 after 10 merges, 3.274 after 20, 2.614 after 40 and 2.256 after 60, while the vocabulary grows from 26 to 86 entries. Each new merge buys less than the one before (the first ten merges save 1.3 tokens per word, the last twenty only 0.36): that is the **diminishing return** that makes a middle choice sensible. Out of the training text, `SCHOOL_TEST` goes from 4.8 tokens per word (character level) to 1.98 with 60 merges. Because `SCHOOL_TEST` uses almost the same words as the training text it benefits even more than the training text does; on a text about another topic, the gain would be smaller.

   **Try also (optional):**

   - Train on only the first half of `SCHOOL_TEXT` and encode the second half. How does the tokens-per-word figure change compared with training on everything?
   - Print the 20 most frequent pieces in the encoded training text (use `Counter` on the pieces).
   - Learn 200 merges. Does the function stop before 200? What does the curve look like at the end?

   *Exercise 5.5:* From the list `curve60`, find the smallest number of merges for which the training text needs at most 3.0 tokens per word, and compute how many tokens per word you save between 40 and 60 merges (one decimal). Why it matters: choosing the number of merges is choosing a point on this curve.

```python
merges_for_3 = None       # YOUR CODE: the smallest m with curve60[m] <= 3.0
saving_40_60 = None       # YOUR CODE: curve60[40] - curve60[60], rounded to 3 decimals
if None in (merges_for_3, saving_40_60):
    print("Fill in your two numbers above, then run this cell again.")
else:
    true_m = next(m for m, t in enumerate(curve60) if t <= 3.0)
    check("smallest m with 3.0 tokens per word or fewer", merges_for_3 == true_m, "loop over the curve and stop at the first value <= 3.0")
    check("saving between 40 and 60 merges", abs(saving_40_60 - (curve60[40] - curve60[60])) < 0.002, "curve60[40] - curve60[60]")
```

   *What it means.* The curve first reaches 3.0 tokens per word after 27 merges, and the last 20 merges (from 40 to 60) save only 0.358 tokens per word, against 1.288 for the first ten. A vocabulary that is only 60 symbols bigger already halves the token count; going much further gives smaller and smaller gains and makes rarer pieces.

6. **Step 5.6. Compare with the real thing.** How does your small BPE (60 merges, learned from 215 words) compare with GPT-2's (about 50,000 merges, learned from a huge amount of text, and starting from bytes)? We count the tokens that both make for six examples, all in lower-case so that the comparison is fair. For your tokenizer a sentence is cut word by word, each word with its own `</w>`. GPT-2 needs the internet or the offline cache; the cell says clearly if it is missing.

```python
try:
    gpt2 = load_tokenizer("gpt2")
except Exception as e:
    gpt2 = None
    print("Could not load the GPT-2 tokenizer (" + type(e).__name__ + "). On the Spark ask your instructor about the offline cache; in Colab check the internet.")
assert gpt2 is not None, "The GPT-2 tokenizer is not loaded: fix the message above, then run this cell again."

def my_tokens(text, merges=merges60):
    return [p for w in words_of(text) for p in encode_bpe(w, merges)]

examples = ["robots", "students", "teachers", "unbelievably", "tokenization", "the robot follows the line"]
print(f"{'text':28s} {'mine':>5s} {'GPT-2':>6s}   my pieces  |  GPT-2 pieces")
rows5 = []
for t in examples:
    mine, theirs = my_tokens(t), gpt2.convert_ids_to_tokens(gpt2(t, add_special_tokens=False)["input_ids"])
    rows5.append((t, len(mine), len(theirs)))
    print(f"{t:28s} {len(mine):5d} {len(theirs):6d}   {mine if len(mine) < 8 else str(mine[:7]) + '...'}  |  {theirs}")
print("totals: mine", sum(r[1] for r in rows5), "| GPT-2", sum(r[2] for r in rows5))
RESULTS["p5_vs_gpt2"] = {r[0]: (r[1], r[2]) for r in rows5}
```

   **What you should see.**

   @@OUT 5.6
   For words from the club (*robots*, *students*, *teachers*) your tokenizer is better: 1 token each, against 2 for GPT-2, because those words fill your small training text. For words outside your text it is much worse: *unbelievably* costs 11 tokens (nearly one per letter) against 4 for GPT-2 (`un`, `bel`, `iev`, `ably`), and *tokenization* costs 9 against 2. The sentence of the club words costs 11 tokens, because *follows* is not in your text, against 5 for GPT-2. Over the six examples the totals are 34 for your tokenizer and 17 for GPT-2. The reasons: GPT-2's merges were learned from a huge text, so its pieces fit the whole language; yours were learned from 215 word occurrences, so they fit one topic; and a bigger number of merges (about 50,000 against 60) means longer pieces.

   **Try also (optional):**

   - Add more words to your training text (copy a paragraph of your own English writing into the corpus) and relearn 100 merges. Does *unbelievably* get cheaper?
   - Compare capital letters: encode "Robots" with `my_tokens` (the function lower-cases because `words_of` does) and with GPT-2.
   - Encode the Filipino sentence of Part 4 with your tokenizer. What do you see?

   *Exercise 5.6:* From the table, compute the ratio of your total tokens to GPT-2's total (one decimal), and name the one example where your tokenizer is *better* than GPT-2 by the largest difference (you may name more than one if tied). Then write two sentences explaining the pattern. Why it matters: a tokenizer is only as good as the text it was trained on and the text it is used on.

```python
ratio_total = None        # YOUR CODE: (total tokens of your tokenizer) / (total tokens of GPT-2), rounded to 1 decimal
best_for_mine = ""        # YOUR ANSWER: the example where your tokenizer saves the most tokens compared with GPT-2 (copy its text)
explain = ""              # YOUR ANSWER: two sentences about training text, number of merges and bytes
if ratio_total is None or not best_for_mine or not explain:
    print("Fill in the three answers above, then run this cell again.")
else:
    tot_mine, tot_gpt = sum(r[1] for r in rows5), sum(r[2] for r in rows5)
    check("ratio of totals", abs(ratio_total - tot_mine / tot_gpt) < 0.06, "sum of the 'mine' column over the sum of the 'GPT-2' column")
    gains = {r[0]: r[2] - r[1] for r in rows5}
    check("a word where yours saves the most", gains.get(best_for_mine, -9) == max(gains.values()), "look for the largest (GPT-2 minus mine)")
    check("your explanation has at least 20 words", len(explain.split()) >= 20, "two sentences")
```

   *What it means.* The ratio is 2.0 (34 against 17): your tokenizer needs twice as many tokens over this mixed list. It wins on the club words (1 against 2 tokens; three words tie, the first is *robots*) and loses badly on the rest. A tokenizer is a compression of the text it was trained on.

7. **Step 5.7. Imagine and write: how many merges for a language with long words?.** A last thinking task, without code. In Filipino a root word can take prefixes, infixes and repeated syllables, which makes many long forms from one root. A careful general statement: when words are long and built from recurring pieces, the training text must be large enough to see each piece many times. Think about the trade-off of Step 5.5 and write a short plan. Do not claim facts about token counts that you did not measure.

```python
merge_plan = ""      # YOUR ANSWER, 3 to 5 sentences: how would you choose the number of merges for a language with long words? What would you measure (tokens per word, how often rare pieces occur)? What text would you need?
if not merge_plan:
    print("Write your plan in `merge_plan` above (at least 30 words), then run this cell again.")
else:
    check("your plan has at least 30 words", len(merge_plan.split()) >= 30, "write 3 to 5 sentences")
    RESULTS["p5_merge_plan"] = merge_plan
```

   **What a good answer contains.** (1) You try several numbers of merges (say 100, 1,000, 5,000) on a **held-out** text, not on the training text, and plot tokens per word against the vocabulary size as in Step 5.5. (2) You pick the point where the curve flattens, because after it extra merges mostly make rare pieces that appear only a few times in training. (3) You make sure the training text is large and varied enough (news, stories, chat) so that frequent pieces such as common prefixes and word endings are learned. (4) You check what happens to names and to words from other languages mixed in (Taglish). (5) You remember that a bigger vocabulary means a bigger embedding table (Part 6), so every extra merge has a cost in parameters as well as a gain in length.

### Part 6. Word embeddings from scratch (30 min)



After tokenisation a text is a list of ids, and an id is only a label: id 5584 is not "bigger" or "more similar" to id 4365 in any useful way. To give a model something it can compute with, every token gets a vector of numbers, its **embedding**, and words that are used in similar ways should get similar vectors. In this part you build that idea from scratch on a toy corpus of 3,000 simple sentences about animals, drinks, royalty and places. You first see why one-hot vectors cannot express similarity; then you build embeddings in two ways: by counting which words occur near each other and compressing the counts (PPMI and SVD), and by training a small skip-gram neural network that predicts the neighbours of a word. You measure whether words of the same group end up close, try analogies honestly, and see how a skew in the text becomes a skew in the vectors. ==An embedding is not given to the model: it is learned from the company words keep.==

Everything is computed on the CPU in seconds, and nothing is downloaded (except in the optional last step). The corpus comes from `make_toy_corpus(3000, seed=0)`: sentences such as *the cat chased the dog*, *she drank tea in the kitchen* and *the queen lived in the castle and drank milk*.

**Imagine first.** Take a sheet of paper and draw a square map, 10 by 10. Place six words on it so that words that *mean something similar or are used in similar sentences* are close: **cat, dog, tea, coffee, king, queen**. Write the position of each word as an (x, y) pair. Before you look at any numbers, also answer in your head: will your map put *cat* nearer to *dog* or nearer to *tea*? Nearer to *tea* or to *coffee*? In Step 6.3 a computer builds a map from nothing but counts, and you compare it with yours.

```python
guess_map = {"cat": None, "dog": None, "tea": None, "coffee": None, "king": None, "queen": None}
# YOUR TURN: replace each None by an (x, y) pair between 0 and 10, for example "cat": (2, 7)
print("Your map:", guess_map)

def judge_map(gm):
    """Say what your map claims about three comparisons."""
    if any(v is None for v in gm.values()):
        print("(fill in all six positions to get a verdict)"); return
    d = lambda a, b: math.dist(gm[a], gm[b])
    print("cat closer to dog than to tea?  ", d("cat", "dog") < d("cat", "tea"))
    print("tea closer to coffee than to king?", d("tea", "coffee") < d("tea", "king"))
    print("king closer to queen than to dog? ", d("king", "queen") < d("king", "dog"))
```

1. **Step 6.1. One-hot versus dense vectors.** The simplest vector for a word is **one-hot**: a vector as long as the vocabulary with a single 1 at the word's position and 0 everywhere else. Any two different one-hot vectors are at a right angle, so their cosine similarity is 0: cat is as far from dog as it is from tea. A **dense** embedding gives each word a short vector of real numbers instead (here 4 numbers). The lookup `nn.Embedding(V, d)` is just a table with one row per word, and picking row `i` gives exactly the same result as multiplying the one-hot vector of word `i` by the table. The number of parameters of the table is V times d.

```python
corpus = make_toy_corpus(3000, seed=0)
vocab = sorted({w for s in corpus for w in s})
V = len(vocab)
idx = {w: i for i, w in enumerate(vocab)}
print("sentences:", len(corpus), "| word occurrences:", sum(len(s) for s in corpus), "| different words V =", V)
for s in corpus[:3]:
    print("   ", " ".join(s))

onehot = np.eye(V)                                   # row i is the one-hot vector of word i
print("one-hot cat:", onehot[idx["cat"] ].astype(int)[:12], "...")
print("cosine(cat, dog) =", cosine(onehot[idx["cat"] ], onehot[idx["dog"] ]), "| cosine(cat, tea) =", cosine(onehot[idx["cat"] ], onehot[idx["tea"] ]))
off_diagonal = [cosine(onehot[i], onehot[j]) for i in range(V) for j in range(i + 1, V)]
print("largest cosine between two different one-hot vectors:", max(off_diagonal))

torch.manual_seed(0)
emb = nn.Embedding(V, 4)                             # a table of V rows and 4 columns, random at the start
i = idx["cat"]
by_lookup = emb(torch.tensor(i)).detach().numpy()
by_product = onehot[i] @ emb.weight.detach().numpy()  # one-hot times the table
assert np.allclose(by_lookup, by_product), "a lookup is a one-hot times the table"
print("embedding of cat (random, not trained yet):", by_lookup.round(3))
print("parameters of the table:", emb.weight.numel(), "= V x d =", V, "x", 4)
RESULTS["p6_V"] = V
```

   **What you should see.**

   @@OUT 6.1
   The toy corpus has 3,000 sentences and 18,576 word occurrences made from only **42** different words. Every pair of different one-hot vectors has cosine 0.0 (the largest value over all 861 pairs is 0.0), and a word has cosine 1.0 with itself: one-hot vectors cannot say that *cat* is more like *dog* than like *tea*. The lookup `emb(torch.tensor(i))` and the product `onehot[i] @ table` agree (the assert passes), so an embedding layer is a fast way to multiply by a one-hot vector. The numbers in the cat vector are random for now; training will give them meaning. The table has 42 x 4 = **168** parameters.

   **Try also (optional):**

   - Print `vocab` and group the words yourself into animals, drinks, royals, places, verbs and function words.
   - Build `nn.Embedding(V, 8)` and check that its table has twice as many parameters.
   - Compute the cosine between the random embeddings of *cat* and *dog*. Is it near 0? Is it near 1? What would you expect for two random vectors of 4 numbers?

   *Exercise 6.1:* How many numbers does an embedding table need for our toy vocabulary with d = 8? And for a GPT-2-sized vocabulary of 50,257 tokens with d = 768? Why it matters: the embedding table is often the first big block of parameters of a language model, and it grows with the vocabulary you chose in Part 5.

```python
params_toy = None      # YOUR CODE: V * 8 for the toy vocabulary
params_gpt2 = None     # YOUR CODE: 50257 * 768
if None in (params_toy, params_gpt2):
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("toy table with d = 8", params_toy == nn.Embedding(V, 8).weight.numel(), "V x d")
    check("GPT-2-sized table", params_gpt2 == 50257 * 768, "50257 x 768")
```

   *What it means.* The toy table needs 42 x 8 = 336 numbers. A GPT-2-sized table needs 50,257 x 768 = 38,597,376, almost 38.6 million numbers, and this is only the first layer. Doubling the vocabulary doubles this block, one more reason to keep the number of BPE merges moderate.

2. **Step 6.2. Co-occurrence counts and PPMI.** The **distributional idea** says that a word is known by the company it keeps. We make it concrete: for every word we count which words occur within **2 positions** of it (before or after) in a sentence. The result is a V by V table `C` where `C[i][j]` is how many times word `j` occurred near word `i`. Raw counts are dominated by frequent words like *the*, which are near everything, so we transform them into **PPMI**, the positive pointwise mutual information: it asks, for each pair, *do these two words occur together more than they would if they were independent?*

```python
def cooc_matrix(corpus, vocab, window=2):
    """C[i][j] = how many times word j occurs within `window` positions of word i (in the same sentence)."""
    ix = {w: k for k, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for s in corpus:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    C[ix[w], ix[s[j] ] ] += 1
    return C

C = cooc_matrix(corpus, vocab)
print("shape", C.shape, "| total counts", int(C.sum()), "| symmetric:", np.allclose(C, C.T))
rows_show = ["cat", "dog", "fox", "tea", "coffee", "milk", "king", "queen", "duke"]
cols_show = ["the", "chased", "followed", "slept", "played", "drank", "liked", "ruled", "visited", "lived", "in"]
sub = C[ [idx[w] for w in rows_show] ][:, [idx[w] for w in cols_show] ]
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(sub, cmap="Blues")
ax.set_xticks(range(len(cols_show))); ax.set_xticklabels(cols_show, rotation=60)
ax.set_yticks(range(len(rows_show))); ax.set_yticklabels(rows_show)
plt.colorbar(im, label="times seen within 2 words"); plt.title("A slice of the co-occurrence table"); plt.tight_layout(); plt.show()
print("cat near chased:", int(C[idx["cat"], idx["chased"] ]), "| cat near drank:", int(C[idx["cat"], idx["drank"] ]),
      "| cat near dog:", int(C[idx["cat"], idx["dog"] ]), "| cat near the:", int(C[idx["cat"], idx["the"] ]))
```

```python
def ppmi_matrix(C):
    """Positive pointwise mutual information of a co-occurrence table.

    P(w, c) = C[w, c] / N          (how often the pair occurs)
    P(w)    = row sum / N          (how often w is a centre word)
    P(c)    = column sum / N       (how often c is a neighbour)
    PMI     = log( P(w, c) / (P(w) P(c)) )   -> 0 if w and c are independent, > 0 if they come together more than by chance
    PPMI    = max(0, PMI)          -> negative values and log(0) are set to 0
    """
    N = C.sum()
    pw = C.sum(axis=1, keepdims=True) / N
    pc = C.sum(axis=0, keepdims=True) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / N) / (pw * pc))
    return np.where(np.isfinite(pmi), np.maximum(pmi, 0), 0.0)

P = ppmi_matrix(C)
print("PPMI shape", P.shape, "| share of zero cells:", round(float((P == 0).mean()), 3))
print(f"{'pair':22s} {'count':>6s} {'PPMI':>6s}")
for a, b in [("cat", "chased"), ("cat", "the"), ("cat", "drank"), ("tea", "drank"), ("tea", "the"), ("king", "ruled")]:
    print(f"{a + ' / ' + b:22s} {int(C[idx[a], idx[b] ]):6d} {P[idx[a], idx[b] ]:6.2f}")
```

   **What you should see.** The co-occurrence table is symmetric (if *cat* is near *dog*, then *dog* is near *cat*) and has 56,304 counts in total.

   @@OUT 6.2a
   The heat map shows blocks. The animals (top rows) are near *chased*, *followed*, *slept* and *played* and never near *drank*, *liked* or *ruled*; the drinks are near *drank* and *liked*; the royals are near *ruled*, *visited* and *lived*; and the columns *the* and *in* are lit for almost every row (the animals and the royals most strongly). In the printed line, *cat* is near *chased* 76 times, near *the* 412 times, and never near *drank* or *dog* (in *the cat chased the dog* there are two words between them, so *dog* lies outside the window of 2). The animals share their company, not each other. Then the PPMI:

   @@OUT 6.2b
   About 0.81 of the cells are zero. The pair *cat / the* has the biggest raw count (412) but only a PPMI of 0.70, and *tea / the* (118 times) has a PPMI of 0.0, because *the* is near almost everything and so its company says little; *cat / chased* (76 times) has a PPMI of 1.87 and *tea / drank* (137 times) 1.59, because those pairs occur together far more than chance would give. *cat / drank* is 0.0: never together. PPMI keeps what is surprising and removes what is merely frequent. That is why it is used instead of raw counts.

   **Try also (optional):**

   - Change `window=2` to `window=1` or `window=4` and compare the PPMI of *cat / the*.
   - Find the row of `P` for *tea* and list the five words with the highest PPMI.
   - Count how many cells of `C` are exactly zero, and explain why a table for a real vocabulary of 50,000 words would have a far larger share of zeros.

   *Exercise 6.2:* Compute the pointwise mutual information of the pair (*cat*, *chased*) by hand from `C`: the probability of the pair (count divided by the total), the probability of *cat* as centre word (its row sum divided by the total), the probability of *chased* as neighbour (its column sum divided by the total), then the logarithm of the ratio (natural log). Then clip at 0 and compare with `P`. Why it matters: PPMI is only three counts and a logarithm; once you have done it by hand it stops being magic.

```python
pmi_cat_chased = None      # YOUR CODE: log( P(cat, chased) / (P(cat) * P(chased)) ) computed from C with np.log
ppmi_cat_chased = None     # YOUR CODE: max(0, pmi_cat_chased)
if None in (pmi_cat_chased, ppmi_cat_chased):
    print("Fill in your two numbers above, then run this cell again.")
else:
    check("PPMI agrees with the matrix P", abs(ppmi_cat_chased - P[idx["cat"], idx["chased"] ]) < 1e-6, "N = C.sum(); P(w,c) = C[i,j]/N; P(w) = C[i].sum()/N; P(c) = C[:,j].sum()/N")
    check("PMI is positive for this pair", pmi_cat_chased > 0, "cat and chased occur together more than by chance")
```

   *What it means.* The pair occurs more often than chance: the PMI is about 1.87 (natural logarithm), a ratio of about 6.5 to 1 between the observed and the expected probability. Because PMI is positive, the PPMI is the same number. A negative PMI would be set to 0: the table keeps only the evidence that two words attract each other.

3. **Step 6.3. Embeddings by SVD: the words find their places.** The PPMI table has 42 numbers per word, most of them redundant (all animals have nearly the same row). The **SVD** (singular value decomposition) is a linear-algebra tool that rewrites a table as a product of simpler parts and orders them by importance. Keeping the first `d` columns of the left part, scaled by the square roots of the singular values, gives every word a vector of `d` numbers that keeps the strongest structure of the table. We keep d = 8 for the neighbours and d = 2 for a map. Then we measure how well the **groups** (animals, drinks, royals, places) come out: the average cosine between two words of the same group (*within*) against two words of different groups (*between*).

```python
U, S, Vt = np.linalg.svd(P)
print("first singular values:", S[:10].round(2))
E8 = U[:, :8] * np.sqrt(S[:8])           # 8 numbers per word
E2 = U[:, :2] * np.sqrt(S[:2])           # 2 numbers per word, for the map
for w in ["cat", "tea", "king", "kitchen", "she", "school"]:
    print(f"{w:8s} nearest:", nearest(E8, vocab, w, 4))
```

```python
GROUPS = {
    "animals": ["cat", "dog", "mouse", "bird", "rabbit", "fox"],
    "drinks": ["tea", "coffee", "milk", "juice", "water", "soda"],
    "royals": ["king", "queen", "prince", "princess", "duke", "duchess"],
    "places": ["kitchen", "garden", "market", "school", "castle", "kingdom"],
}
group_of = {w: g for g, ws in GROUPS.items() for w in ws}

def group_gap(E, vocab=vocab):
    """Mean cosine of two words of the same group (within) and of two words of different groups (between)."""
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    cs = En @ En.T
    within, between = [], []
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            if i < j and a in group_of and b in group_of:
                (within if group_of[a] == group_of[b] else between).append(cs[i, j])
    return float(np.mean(within)), float(np.mean(between))

for d in (2, 4, 8, 16):
    w_, b_ = group_gap(U[:, :d] * np.sqrt(S[:d]))
    print(f"d = {d:2d}: within {w_:.3f}  between {b_:.3f}  gap {w_ - b_:.3f}")
w8, b8 = group_gap(E8)
assert w8 > b8, "words of the same group should be closer than words of different groups"
RESULTS["p6_svd_gap"] = (round(w8, 3), round(b8, 3))

colors = {"animals": "C0", "drinks": "C1", "royals": "C2", "places": "C3"}
fig, ax = plt.subplots(figsize=(6.4, 5))
for w in vocab:
    x, y = E2[idx[w] ]
    c = colors.get(group_of.get(w), "gray")
    ax.scatter(x, y, color=c, s=25); ax.annotate(w, (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
ax.set_title("The 2-D map from SVD (colours = the four groups, grey = other words)"); plt.tight_layout(); plt.show()
judge_map(guess_map)
```

   **What you should see.** The singular values and the neighbours:

   @@OUT 6.3a
   The singular values fall from 13.9 to 7.9 over the first six, then drop (6.0, 3.5, 3.5, 2.9): about seven directions carry the structure of this toy language. The nearest neighbours of *cat* are the other animals (*mouse*, *dog*, *fox* and *bird* come first, with cosine near 1.0), of *tea* the other drinks, of *king* the other royals, and of *kitchen* and *school* the other places. *she* has only one close word, *he*. Then the group measurements:

   @@OUT 6.3b 1-4
   With d = 8 the words of the same group have a mean cosine of **0.828** and words of different groups only **0.093**: a gap of 0.735 (the assert passes). The gap grows with d (0.456 for d = 2, 0.524 for d = 4, 0.735 for d = 8, 0.785 for d = 16). In the map the animals, the royals, the drinks and the places *kitchen*, *garden*, *market* and *school* each sit in a tight cluster, with the verbs and function words in between; but with two dimensions all the words lie in a narrow band, so even words of different groups keep a cosine of 0.525 and the gap is smaller. *castle* and *kingdom* are labelled as places but sit apart from the other four places, because they are used with the royals. Compare the map with your Imagine-first sketch: the computer knew nothing about animals or kings, and it found the groups from counts alone.

   **Try also (optional):**

   - Print `nearest(E2, vocab, "cat", 4)`: with only two dimensions, are the neighbours the same?
   - Use the columns 3 and 4 of `U` for the map instead of 1 and 2 (`U[:, 2:4] * np.sqrt(S[2:4])`). Which groups separate now?
   - Remove the scaling `np.sqrt(S[:8])` and compute the gap again. Does it change?

   *Exercise 6.3:* Compute the within-minus-between gap (rounded to 3 decimals) for d = 3 and d = 6 with `U[:, :d] * np.sqrt(S[:d])` and `group_gap`, and name the dimension from (2, 3, 4, 6, 8, 16) with the largest gap. Why it matters: the size of an embedding is a design choice, and you can measure its effect.

```python
gap_3 = None          # YOUR CODE: within - between for d = 3
gap_6 = None          # YOUR CODE: within - between for d = 6
best_d = None         # YOUR CODE: the d among (2, 3, 4, 6, 8, 16) with the largest gap
if None in (gap_3, gap_6, best_d):
    print("Fill in your three values above, then run this cell again.")
else:
    gaps_ = {d: np.subtract(*group_gap(U[:, :d] * np.sqrt(S[:d]))) for d in (2, 3, 4, 6, 8, 16)}
    check("gap for d = 3", abs(gap_3 - gaps_[3]) < 0.002, "w, b = group_gap(U[:, :3] * np.sqrt(S[:3])); w - b")
    check("gap for d = 6", abs(gap_6 - gaps_[6]) < 0.002, "same with d = 6")
    check("best d", best_d == max(gaps_, key=gaps_.get), "the largest gap among the six values")
```

   *What it means.* The gap is 0.594 for d = 3 and 0.752 for d = 6: it grows from d = 2 (0.456) up to the 6 or 7 dimensions that carry the real structure, then stays about the same (0.735 for d = 8, 0.785 for d = 16, the best of the six by a small margin). Too few dimensions mix the groups together, but beyond the structure of the data more dimensions add little.

4. **Step 6.4. Skip-gram: let a small network learn the vectors.** The second way to get embeddings is to *train* them. A **skip-gram** model takes a centre word and tries to predict a word that occurs within 2 positions of it. The model has an `nn.Embedding(V, 8)` (the vectors we want) followed by an `nn.Linear(8, V)` that turns a vector into one score per vocabulary word; the loss is cross-entropy as in Week 1. Because the vocabulary has only 42 words we can use the full softmax. After training, the rows of the embedding table are the word vectors: words that predict the same neighbours are pushed to similar vectors.

```python
def make_pairs(corpus, window=2):
    """All (centre word id, neighbour word id) pairs within `window` positions."""
    pairs = []
    for s in corpus:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    pairs.append((idx[w], idx[s[j] ]))
    return torch.tensor(pairs)

pairs = make_pairs(corpus)
print("training pairs:", len(pairs), "| e.g.", [(vocab[a], vocab[b]) for a, b in pairs[:4].tolist()])

def train_skipgram(d=8, epochs=10, lr=0.02, batch=512, seed=0):
    """Train a skip-gram model (Embedding -> Linear -> softmax) and return (embedding table as NumPy, list of epoch losses)."""
    torch.manual_seed(seed)
    emb_ = nn.Embedding(V, d); out_ = nn.Linear(d, V)
    opt = torch.optim.Adam(list(emb_.parameters()) + list(out_.parameters()), lr=lr)
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(pairs)); total = 0.0
        for b in range(0, len(pairs), batch):
            ids = perm[b:b + batch]
            opt.zero_grad()
            loss = F.cross_entropy(out_(emb_(pairs[ids, 0])), pairs[ids, 1])
            loss.backward(); opt.step(); total += loss.item() * len(ids)
        losses.append(total / len(pairs))
    return emb_.weight.detach().numpy().copy(), losses

t0 = time.time()
E_sg, losses = train_skipgram(d=8, epochs=10)
print("loss per epoch:", [round(l, 3) for l in losses], f"({time.time() - t0:.1f} s)")
print("uniform guessing would give a loss of ln(42) =", round(math.log(V), 3))
assert losses[-1] < losses[0] < math.log(V) + 0.5, "the loss should fall below the starting level"
```

```python
for w in ["cat", "tea", "king", "kitchen"]:
    print(f"{w:8s} nearest:", nearest(E_sg, vocab, w, 4))
w_sg, b_sg = group_gap(E_sg)
print(f"{'method':14s} {'within':>7s} {'between':>8s} {'gap':>6s}")
print(f"{'SVD (d = 8)':14s} {w8:7.3f} {b8:8.3f} {w8 - b8:6.3f}")
print(f"{'skip-gram':14s} {w_sg:7.3f} {b_sg:8.3f} {w_sg - b_sg:6.3f}")
print("parameters of the skip-gram model:", V * 8 + (8 * V + V), "(embedding table + output layer)")
RESULTS["p6_skipgram_gap"] = (round(w_sg, 3), round(b_sg, 3))
assert w_sg > b_sg
```

   **What you should see.**

   @@OUT 6.4a
   The model sees 56,304 pairs per epoch. The loss falls from 2.76 in the first epoch to about 2.26 at the end. It cannot go to zero, because the next-door word is not determined by the centre word (after *the* either an animal, a royal or a place may come), and the uniform loss is ln(42) = 3.74. The nearest neighbours and the comparison:

   @@OUT 6.4b
   The skip-gram vectors find the same groups: *cat* has *mouse*, *rabbit* and *bird* nearby (cosine 0.99, 0.97, 0.96) and *tea* has *milk*, *water*, *soda*. The group gap is a bit smaller than for SVD (within 0.841 and between 0.268 against 0.828 and 0.093): the trained vectors are not as sharply separated as the SVD ones, probably because a trained network is not forced to give all the words of a group the same vector, as the algebra of the SVD nearly does. The two methods are close cousins: both compress the same co-occurrence information, one by algebra and one by gradient descent.

   **Try also (optional):**

   - Train for 30 epochs: does the loss keep falling, and does the gap grow?
   - Change the window to 1 in `make_pairs` and compare the nearest neighbours of *kitchen*.
   - Print the cosine between *she* and *he* for both methods.

   *Exercise 6.4:* Train a skip-gram model with only d = 2 numbers per word (`train_skipgram(d=2, epochs=10)`) and report its within-minus-between group gap (3 decimals) and its parameter count (embedding table plus output layer). Then say whether you would still call the words of one group close. Why it matters: a smaller embedding is cheaper but has less room to separate meanings.

```python
gap_sg2 = None        # YOUR CODE: within - between for the d = 2 skip-gram embedding
params_sg2 = None     # YOUR CODE: parameters of the d = 2 model (V*2 for the table + 2*V + V for the Linear layer)
if None in (gap_sg2, params_sg2):
    print("Fill in your two numbers above, then run this cell again.")
else:
    E_check, _ = train_skipgram(d=2, epochs=10)
    w_c, b_c = group_gap(E_check)
    check("gap of the d = 2 model", abs(gap_sg2 - (w_c - b_c)) < 0.03, "train_skipgram(d=2, epochs=10), then group_gap")
    check("parameters of the d = 2 model", params_sg2 == V * 2 + (2 * V + V), "42*2 + (2*42 + 42)")
```

   *What it means.* The d = 2 model has 210 parameters (against 714 for d = 8) and a gap of 0.501 (within 0.991, between 0.490) against 0.573 for d = 8. With two numbers per word there is less room to push the groups apart, so words of different groups keep a clearly higher cosine than in the d = 8 model; the groups are still there, which shows how simple the structure of this toy language is.

5. **Step 6.5. Cosine similarity by hand.** The cosine of two vectors is their dot product divided by the product of their lengths: `cos(a, b) = a . b / (|a| |b|)`. It is 1 for vectors pointing the same way, 0 for vectors at a right angle and -1 for opposite vectors; it ignores how long the vectors are, which is what we want when a frequent word has a longer vector than a rare one. We compute it for three pairs of the SVD embeddings in three ways: step by step, with the helper `cosine`, and with `F.cosine_similarity` from PyTorch, and assert that they agree.

```python
rows6 = []
for a, b in [("cat", "dog"), ("cat", "tea"), ("king", "queen")]:
    va, vb = E8[idx[a] ], E8[idx[b] ]
    dot = float(np.dot(va, vb))
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    by_hand = dot / (na * nb)
    by_helper = cosine(va, vb)
    by_torch = float(F.cosine_similarity(torch.tensor(va), torch.tensor(vb), dim=0))
    assert abs(by_hand - by_helper) < 1e-9 and abs(by_hand - by_torch) < 1e-6
    rows6.append((a, b, dot, na, nb, by_hand))
    print(f"{a:5s} {b:6s} dot {dot:7.3f}  |a| {na:6.3f}  |b| {nb:6.3f}  cosine {by_hand:6.3f}")
RESULTS["p6_cosines"] = {f"{a}-{b}": round(c, 3) for a, b, _, _, _, c in rows6}
```

   **What you should see.**

   @@OUT 6.5
   *cat* and *dog* have cosine 1.000 (their rows of PPMI are almost the same, so their vectors point the same way), *cat* and *tea* have a cosine of about 0 (the two groups never share a neighbour) and *king* and *queen* are at 0.997. The three methods give the same number to many decimals (the assert passes). The dot product alone would not do: it also depends on the lengths of the vectors, which is why the cosine divides them out.

   **Try also (optional):**

   - Compute the cosine of *cat* with every other word and print the six smallest values. What do the words at the bottom have in common?
   - Compute the cosine of a vector with itself and with its negative.
   - Compute the plain dot product of *cat* with *dog* and with *the*. Is the dot product larger for the more frequent word?

   *Exercise 6.5:* Compute the cosine of (*tea*, *coffee*) and of (*tea*, *king*) from `E8` using only `np.dot` and `np.linalg.norm`, and the cosine of a vector with its double (`2 * E8[idx["tea"] ]`). Why it matters: the formula is simple, and doing it once with your own hands is the best protection against using it wrongly later.

```python
cos_tea_coffee = None     # YOUR CODE with np.dot and np.linalg.norm
cos_tea_king = None       # YOUR CODE
cos_with_double = None    # YOUR CODE: cosine of E8[idx["tea"] ] with 2 * E8[idx["tea"] ]
if None in (cos_tea_coffee, cos_tea_king, cos_with_double):
    print("Fill in your three numbers above, then run this cell again.")
else:
    check("tea and coffee", abs(cos_tea_coffee - cosine(E8[idx["tea"] ], E8[idx["coffee"] ])) < 1e-6, "dot / (norm * norm)")
    check("tea and king", abs(cos_tea_king - cosine(E8[idx["tea"] ], E8[idx["king"] ])) < 1e-6, "dot / (norm * norm)")
    check("a vector and its double", abs(cos_with_double - 1.0) < 1e-6, "the cosine ignores length")
```

   *What it means.* *tea* and *coffee* are in the same group and have a cosine near 1; *tea* and *king* are near 0. Doubling a vector does not change the direction, so the cosine with its double is exactly 1: the cosine measures direction, not length.

6. **Step 6.6. Analogies, honestly, and a word about bias.** The famous example of word embeddings is *king - man + woman = queen*. It works sometimes on embeddings learned from billions of words, and it deserves a careful test. Our corpus has no *man* and *woman*, but it has three male-female pairs: *king / queen*, *prince / princess* and *duke / duchess*. An analogy `a : b :: c : d` is a **hit** if the word closest (by cosine) to `b - a + c` is `d`, ignoring `a`, `b` and `c`. With three pairs we can make 12 analogies (6 forward and 6 backward). We test them on the corpus, on five different corpora, and on a control with mismatched pairs, to see what a "hit" is worth. Then a second experiment on bias: we add 3,000 sentences in which *she* usually drinks tea, juice or milk and *he* coffee, soda or water, and see what happens to the vectors.

```python
def ppmi_svd(corp, d=8, window=2):
    """The whole pipeline of Steps 6.2 and 6.3 in one call: returns (embedding matrix, vocabulary)."""
    vc = sorted({w for s in corp for w in s})
    Pm = ppmi_matrix(cooc_matrix(corp, vc, window))
    Um, Sm, _ = np.linalg.svd(Pm)
    return Um[:, :d] * np.sqrt(Sm[:d]), vc

def analogy_hits(E, vc, pairs_):
    """Test every a:b::c:d among the pairs (forward and backward). Returns (hits, total, list of misses)."""
    ix = {w: k for k, w in enumerate(vc)}
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    hits, total, misses = 0, 0, []
    for direction in (pairs_, [(b, a) for a, b in pairs_]):
        for a, b in direction:
            for c, d in direction:
                if (a, b) == (c, d):
                    continue
                v = En[ix[b] ] - En[ix[a] ] + En[ix[c] ]
                sims = En @ (v / np.linalg.norm(v))
                for w in (a, b, c):
                    sims[ix[w] ] = -9                  # the three given words may not be the answer
                guess = vc[int(np.argmax(sims))]
                total += 1; hits += guess == d
                if guess != d:
                    misses.append((f"{a}:{b}::{c}:?", guess))
    return hits, total, misses

true_pairs = [("king", "queen"), ("prince", "princess"), ("duke", "duchess")]
control_1 = [("king", "princess"), ("prince", "duchess"), ("duke", "queen")]     # mismatched on purpose
control_2 = [("king", "duchess"), ("prince", "queen"), ("duke", "princess")]
h, t, misses = analogy_hits(E8, vocab, true_pairs)
print(f"king-queen style analogies on the SVD embeddings: {h} of {t} hits | by chance among the 3 words left: about {t / 3:.0f}")
print("some misses:", misses[:3])
for name, ps in [("control 1", control_1), ("control 2", control_2)]:
    hh, tt, _ = analogy_hits(E8, vocab, ps)
    print(f"{name} (mismatched pairs): {hh} of {tt}")
seed_hits = []
for sd in range(5):
    Es, vs = ppmi_svd(make_toy_corpus(3000, seed=sd))
    seed_hits.append(analogy_hits(Es, vs, true_pairs)[0])
print("hits of 12 for five different toy corpora (seeds 0 to 4):", seed_hits)
RESULTS["p6_analogy_hits"] = seed_hits
```

```python
def skewed_corpus(n=3000, seed=2, strength=0.85):
    """Sentences like 'she drank tea in the kitchen': with probability `strength` she drinks tea, juice or milk and he coffee, soda or water."""
    r = random.Random(seed); out = []
    for _ in range(n):
        she = r.random() < 0.5
        pref = ["tea", "juice", "milk"] if she else ["coffee", "soda", "water"]
        drink = r.choice(pref) if r.random() < strength else r.choice(GROUPS["drinks"])
        out.append(["she" if she else "he", "drank", drink, "in", "the", "kitchen"])
    return out

def she_he_gap(corp, drink):
    """cosine(she, drink) minus cosine(he, drink) in the SVD embeddings learned from `corp`."""
    Ec, vc = ppmi_svd(corp); ix = {w: k for k, w in enumerate(vc)}
    return cosine(Ec[ix["she"] ], Ec[ix[drink] ]) - cosine(Ec[ix["he"] ], Ec[ix[drink] ])

print(f"{'drink':8s} {'plain corpus':>13s} {'skewed corpus':>14s}    (cosine(she, drink) - cosine(he, drink))")
for dr in ["tea", "juice", "milk", "coffee", "soda", "water"]:
    print(f"{dr:8s} {she_he_gap(corpus, dr):13.3f} {she_he_gap(corpus + skewed_corpus(), dr):14.3f}")
```

   **What you should see.**

   @@OUT 6.6a
   On the corpus of this lab, 9 of the 12 analogies are hits, which looks impressive until you compare: among the three royals left, a random guess would hit about 4 of 12, and the two controls with mismatched pairs get only 2 and 1. But then look at the five corpora: the hits go **9, 6, 0, 5, 5** (your numbers must match). The same code, on a corpus written by the same generator, gives anything from nothing to 9 out of 12. The reason is that in this toy text the male and female royals appear in exactly the same kinds of sentences, so there is no "gender direction" to find: what the test measures is mostly small random differences in the counts. On large real corpora a real direction can exist, and analogies sometimes work, but they are also chosen after the fact, and many fail. Treat analogies as a curiosity, not as a proof that the vectors "understand".

   @@OUT 6.6b
   In the plain corpus the gap is about 0.003 for every drink: *she* and *he* are used in the same way. In the skewed corpus the vectors changed: *she* is now closer to *tea*, *juice* and *milk* (gaps +0.14 to +0.17) and *he* closer to *coffee*, *soda* and *water* (gaps -0.04 to -0.05). Nothing in the algorithm is "biased"; it faithfully copied the company the words keep in the text. That is how embeddings learned from real text pick up stereotypes: a pattern in the text, true or not, fair or not, becomes a geometry that a model will use unless people notice and correct it.

   **Try also (optional):**

   - Set `strength=0.55` and see how the gaps shrink; then `strength=1.0`.
   - Use `ppmi_svd` with d = 16 and count the analogy hits for the five seeds. Is the number steadier?
   - Add the sentence template "the king said he ... " to the corpus (a short function that writes 1,500 such sentences, with *she* for the queen, princess and duchess) and look again at the analogies *king - he + she*.

   *Exercise 6.6:* Using the five hit counts in `seed_hits`, report the smallest, the largest and the average hit count (one decimal). Then measure the bias gap for *tea* in the skewed corpus and say in one sentence whether you would trust an analogy test as proof that a set of vectors is good. Why it matters: a number without its spread, or a demo without a control, is not evidence.

```python
hits_min = None         # YOUR CODE: smallest of seed_hits
hits_max = None         # YOUR CODE: largest of seed_hits
hits_mean = None        # YOUR CODE: average of seed_hits, rounded to 1 decimal
tea_gap = None          # YOUR CODE: she_he_gap(corpus + skewed_corpus(), "tea"), rounded to 3 decimals
trust = ""              # YOUR ANSWER: one sentence about whether analogy hits prove quality
if None in (hits_min, hits_max, hits_mean, tea_gap) or not trust:
    print("Fill in your four numbers and the sentence above, then run this cell again.")
else:
    check("smallest hit count", hits_min == min(seed_hits), "min(seed_hits)")
    check("largest hit count", hits_max == max(seed_hits), "max(seed_hits)")
    check("average", abs(hits_mean - sum(seed_hits) / len(seed_hits)) < 0.06, "sum / 5")
    check("gap for tea", abs(tea_gap - she_he_gap(corpus + skewed_corpus(), "tea")) < 0.002, "cosine(she, tea) - cosine(he, tea)")
    check("your sentence has at least 10 words", len(trust.split()) >= 10, "write a real sentence")
```

   *What it means.* The hits range from 0 to 9 with an average of 5.0; the bias gap for *tea* is about 0.154. A single run of 9 of 12 would have looked like success; five runs show it is fragile.

7. **Step 6.7. Optional stretch: embeddings of whole sentences.** Our vectors are one per word, whatever the sentence: *bank* gets the same vector in *river bank* and *bank account*. The next weeks fix that with attention. A pretrained sentence model does a related job at scale: `sentence-transformers/all-MiniLM-L6-v2` turns a whole sentence into a vector so that sentences with similar meaning have a high cosine. This cell **needs a download of about 90 MB** and the `sentence-transformers` package, so it runs only in Colab, or on the Spark if your instructor has put the model in the cache. It is wrapped in `try` and prints a message if it cannot run. We did not run it for this lab, so no numbers are quoted: you read your own output.

```python
# nb-test: skip
sentences = ["The students build robots after class.", "Learners make small machines in the afternoon.", "The cat sleeps in the kitchen.",
             "A kitten naps on the kitchen floor.", "The contest is in March.", "Bananas are rich in potassium."]
try:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = st_model.encode(sentences)
    cm = np.array([ [cosine(a, b) for b in vecs] for a in vecs ])
    print(np.round(cm, 2))
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="viridis"); plt.colorbar(); plt.title("cosine between sentence vectors"); plt.show()
except Exception as e:
    print("Skipped (needs sentence-transformers and a 90 MB download):", type(e).__name__)
```

   **What to look for.** Sentence 1 and sentence 2 say the same in different words (*students build robots* and *learners make small machines*); sentences 3 and 4 do as well (*cat sleeps* and *kitten naps*). If the model works as advertised, the cosines inside these two pairs are clearly higher than the cosines between unrelated sentences such as 1 and 6. A bag of one-hot word vectors could not do this: the pair 1 and 2 shares almost no words.

### Part 7. Design a tokenizer for a community, and reflect (15 min)



Now you use what you built. A tokenizer is not neutral: it was designed by people, trained on some text, and it works better for the language and the style of that text. In this part you imagine an app for your own school or community, write a design card for its tokenizer, measure how two real tokenizers treat sentences of your language, train your own small BPE on text that you write, choose a number of merges and justify it, and finish with four reflection questions. ==A good design is one whose choices you can defend with a number you measured yourself.== The code is short: most of the work is thinking and writing.

The default text below is in Filipino, about a homework helper app, so that the cells run as they are; **replace it with your own language, your own app and your own sentences** (a Cebuano voice assistant, a jeepney route chatbot, an SMS spam filter, a canteen menu bot). Only write sentences you are sure are correct, and write the new text yourself.

**Imagine first.** You are designing a homework-helper app for students who write in Filipino, often mixed with English words (Taglish). Before you measure anything, guess: if GPT-2's tokenizer reads a Filipino sentence and its English translation, how many **times more tokens per word** will the Filipino sentence need on average? Write a number such as 1.0 (the same), 1.5 or 2.5. You compare with the measurement in Step 7.2.

```python
guess_ratio = None      # YOUR TURN: Filipino tokens per word divided by English tokens per word, with GPT-2 (for example 1.0, 1.5, 2.5)
print("Your guess for the ratio (Filipino / English tokens per word, GPT-2):", guess_ratio)
```

1. **Step 7.1. The design card.** A **design card** is a one-page decision record: who the tokenizer is for, what text it will read, which of the three granularities you choose and why, how big the vocabulary should be, which special tokens you need, how you handle names, numbers, emoji and code-switching, and how you will check quality. The code below only stores the answers in a dictionary and prints them as a card. It holds an example for the Filipino homework helper; your task in Exercise 7.1 is to write your own.

```python
SPEC_KEYS = ["name", "language", "typical_text", "granularity", "why", "vocab_target", "special_tokens", "names_numbers_emoji", "code_switching", "quality_metrics"]

example_spec = {
    "name": "Tulong-Aral: a Filipino homework helper (example)",
    "language": "Filipino, with English words mixed in (Taglish)",
    "typical_text": "short questions and instructions such as 'Pakipaliwanag ang aralin sa matematika.' and 'Paki-explain nga yung photosynthesis.'",
    "granularity": "BPE (sub-word), learned on the app's own text and on public Filipino text",
    "why": "words are long and have many forms (prefixes and repeated syllables), so a word list would meet unknown words all the time; characters would make every question very long; BPE keeps common words whole and spells new ones from pieces",
    "vocab_target": 8000,
    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<helper>"],
    "names_numbers_emoji": "names and emoji must never become <unk>: start from bytes; split numbers into single digits so that arithmetic questions look alike",
    "code_switching": "learn from text that mixes Filipino and English, so that 'explain', 'homework' and 'photosynthesis' are cheap too",
    "quality_metrics": "tokens per word on held-out sentences (aim: low and close to English), unknown rate (aim: 0 percent), exact round trip decode(encode(text)) equal to text",
}

def print_card(spec):
    print("=" * 74)
    for k in SPEC_KEYS:
        v = spec.get(k, "")
        print(f"{k:20s}: {v if v not in ('', None) else '(empty)'}")
    print("=" * 74)

print_card(example_spec)
assert all(example_spec[k] not in ("", None) for k in SPEC_KEYS)
RESULTS["p7_example_spec"] = example_spec["name"]
```

   **What you should see.** A framed card with ten lines (the card above is an example; yours will describe your own app). A good card is *specific*: it names the kinds of text, it gives a number for the vocabulary, and each choice has a reason. Notice that the example's reasons are about the language (long words with many forms) and about the users (names, numbers and emoji are common in chat), not about what is fashionable.

   **Try also (optional):**

   - Write a second card for a different app (an SMS spam filter for English and Filipino messages) and compare the two choices of granularity.
   - Change `vocab_target` and explain, with the embedding-table formula of Step 6.1, what each extra 1,000 tokens costs in parameters for d = 256.
   - Add a line "risks" to the card: what could go wrong for users who write in a dialect your training text did not contain?

   *Exercise 7.1:* Fill in the dictionary `my_spec` for **your** app: all ten entries, in your own words. `vocab_target` must be a whole number. Be specific: name the kinds of text, say which granularity you chose and the reason, and say how you will measure quality. Why it matters: writing the choices down is what lets someone else (or you, in a month) judge them.

```python
my_spec = {k: "" for k in SPEC_KEYS}
# YOUR TURN: fill in each entry, e.g. my_spec["name"] = "Sakay-Bot: a jeepney route chatbot"; my_spec["vocab_target"] = 5000 (a whole number)
print_card(my_spec)
empty = [k for k in SPEC_KEYS if my_spec.get(k) in ("", None, [])]
if len(empty) == len(SPEC_KEYS):
    print("Fill in the entries of `my_spec` above, then run this cell again.")
else:
    check("all ten entries are filled", not empty, "still empty: " + ", ".join(empty))
    check("vocab_target is a whole number between 500 and 200000", isinstance(my_spec["vocab_target"], int) and 500 <= my_spec["vocab_target"] <= 200000, "write a number such as 8000")
    check("the reason has at least 12 words", len(str(my_spec["why"]).split()) >= 12, "explain why you chose that granularity")
    RESULTS["p7_spec"] = dict(my_spec)
```

   *What it means.* A card with all ten entries, a whole-number vocabulary and a reason of at least a dozen words passes. What matters more than the check is the *reason*: it should be about your language and your users, and the metrics must be things you can measure in Steps 7.2 and 7.3.

2. **Step 7.2. Measure it: tokens per word with GPT-2 and BERT.** Now measure your design's text with the two real tokenizers. The list `my_sentences` holds eight sentences in the target language (Filipino and Taglish in the default), and `english_baseline` holds their English meanings, in the same order. For each sentence we compute the **fertility** (tokens per space-separated word) with GPT-2 and with BERT, print a table, draw a bar chart and compare the averages over all words: total tokens divided by total words. If the tokenizers are not available, the cell says so clearly.

```python
try:
    gpt2 = load_tokenizer("gpt2"); bert = load_tokenizer("bert-base-uncased")
except Exception as e:
    gpt2 = bert = None
    print("Could not load the tokenizers (" + type(e).__name__ + "). On the Spark ask your instructor about the offline cache; in Colab check the internet.")
assert gpt2 is not None and bert is not None, "The tokenizers are not loaded: fix the message above, then run this cell again."

my_sentences = [
    "Kumusta ka na?",
    "Pakipaliwanag ang aralin sa matematika.",
    "Ano ang ibig sabihin ng salitang ito?",
    "Salamat po sa tulong.",
    "Paki-explain nga yung photosynthesis.",
    "Gumagawa ang mga estudyante ng maliliit na robot.",
    "Nag-eensayo kami pagkatapos ng klase.",
    "Maraming salamat po sa mga guro.",
]
english_baseline = [
    "How are you?",
    "Please explain the lesson in mathematics.",
    "What does this word mean?",
    "Thank you for the help.",
    "Please explain photosynthesis.",
    "The students make small robots.",
    "We practise after class.",
    "Thank you very much to the teachers.",
]

def n_tok(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])

print(f"{'sentence':44s} {'words':>5s} {'GPT-2':>6s} {'BERT':>5s}   | English: {'words':>5s} {'GPT-2':>6s} {'BERT':>5s}")
for fs, es in zip(my_sentences, english_baseline):
    print(f"{fs[:43]:44s} {len(fs.split()):5d} {n_tok(gpt2, fs):6d} {n_tok(bert, fs):5d}   |          {len(es.split()):5d} {n_tok(gpt2, es):6d} {n_tok(bert, es):5d}")

def fertility(tok, sents):
    return sum(n_tok(tok, s) for s in sents) / sum(len(s.split()) for s in sents)

fert = {("target", "GPT-2"): fertility(gpt2, my_sentences), ("target", "BERT"): fertility(bert, my_sentences),
        ("English", "GPT-2"): fertility(gpt2, english_baseline), ("English", "BERT"): fertility(bert, english_baseline)}
for (lang, name), v in fert.items():
    print(f"{lang:8s} {name:6s} tokens per word: {v:.3f}")
ratio_gpt2 = fert[("target", "GPT-2")] / fert[("English", "GPT-2")]
print(f"ratio target / English with GPT-2: {ratio_gpt2:.2f}   | with BERT: {fert[('target', 'BERT')] / fert[('English', 'BERT')]:.2f}   | your guess: {guess_ratio}")
fig, ax = plt.subplots(figsize=(6, 3.2))
labels = [("target", "GPT-2"), ("English", "GPT-2"), ("target", "BERT"), ("English", "BERT")]
ax.bar([f"{a}\n{b}" for a, b in labels], [fert[k] for k in labels], color=["C0", "C2", "C0", "C2"])
ax.set_ylabel("tokens per word"); ax.axhline(1, color="gray", ls=":"); plt.tight_layout(); plt.show()
RESULTS["p7_fertility"] = {f"{a}-{b}": round(v, 3) for (a, b), v in fert.items()}
RESULTS["p7_ratio_gpt2"] = round(ratio_gpt2, 2)
```

   **What you should see.** A table with one line per sentence (the first three Filipino lines read *Kumusta ka na?*, *Pakipaliwanag ang aralin sa matematika.* and *Ano ang ibig sabihin ng salitang ito?*), then the four averages:

   @@OUT 7.2
   For these eight sentences GPT-2 needs 2.48 tokens per word in Filipino against 1.24 for the English meanings: a ratio of **2.00**; BERT needs 2.31 against 1.32, a ratio of 1.76. So the same message costs about twice as many tokens in Filipino as in English with GPT-2's tokenizer, and that is a *cost and quality* issue, not a statement about the language. Look at single lines too: *Kumusta ka na?* needs 7 tokens against 4 for *How are you?*, and *Nag-eensayo kami pagkatapos ng klase.* needs 18 tokens for 5 words. Compare with your guess. The bar chart makes the gap visible. Remember that eight sentences are a small sample: the exact numbers would change with other sentences, but the direction (a longer sequence for a language with less training text in the tokenizer) is the point.

   **Try also (optional):**

   - Replace the sentences with your own (at least three in the target language, with their English meanings) and run the cell again.
   - Print `gpt2.convert_ids_to_tokens(gpt2(my_sentences[1])["input_ids"])` and read the pieces of *Pakipaliwanag*. Which pieces carry some meaning and which are accidents?
   - Add a Taglish sentence with an English word you often use. Is the English word cheaper than the Filipino words around it?

   *Exercise 7.2:* Replace `my_sentences` and `english_baseline` above with **your own** sentences (eight is ideal; at least three must be in your target language, and the English baseline must have the same meanings), run the cell again, and then fill in your result: the GPT-2 ratio (2 decimals) and which of the two tokenizers (`"gpt2"` or `"bert"`) is less expensive for your sentences. Why it matters: before you trust a tokenizer for a community, you measure it on the community's own text.

```python
my_ratio = None         # YOUR CODE: fertility(gpt2, my_sentences) / fertility(gpt2, english_baseline), rounded to 2 decimals
cheaper = ""            # YOUR ANSWER: "gpt2" or "bert": which tokenizer needs fewer tokens per word on YOUR sentences?
if my_ratio is None or not cheaper:
    print("Fill in `my_ratio` and `cheaper` above, then run this cell again.")
else:
    check("at least 3 target-language sentences", len(my_sentences) >= 3 and len(my_sentences) == len(english_baseline), "my_sentences and english_baseline need the same length")
    check("ratio matches your sentences", abs(my_ratio - fertility(gpt2, my_sentences) / fertility(gpt2, english_baseline)) < 0.01, "divide the two fertilities")
    true_cheaper = "gpt2" if fertility(gpt2, my_sentences) < fertility(bert, my_sentences) else "bert"
    check("the cheaper tokenizer", cheaper == true_cheaper, "compare fertility(gpt2, ...) with fertility(bert, ...)")
    RESULTS["p7_my_ratio"] = my_ratio
```

   *What it means.* With the default Filipino sentences the ratio is 2.00, and BERT (WordPiece, lower-case) is the cheaper of the two for them (2.31 against 2.48 tokens per word; for the English meanings the order is the other way round: 1.32 for BERT and 1.24 for GPT-2). That says nothing about which tokenizer is better in general: BERT cannot give back the exact text (Step 4.3), and your own sentences may behave differently.

3. **Step 7.3. Train your own BPE on your own text.** Finally, train a BPE tokenizer on text you wrote yourself, with the functions of Part 5 (`learn_bpe`, `encode_bpe`). The default `my_text` is a Filipino paragraph about a school robotics club of more than 100 words; replace it with your own (at least 100 words, written by you, in your target language). The cell learns up to 150 merges and measures, on the eight held-out sentences of Step 7.2 (which are *not* in `my_text`), the tokens per word for 0, 20, 40, 80 and all merges, and compares with a word-level vocabulary built from the same text, whose unknown rate on those sentences you also measure. Then the curve and your choice.

```python
my_text = """
Ang aming paaralan ay may robotics club. Tuwing Biyernes, nagkikita ang mga estudyante sa silid-aralan. Gumagawa sila ng maliliit na robot at sumusulat ng code.
Tinutulungan sila ng mga guro. Masaya ang mga estudyante kapag sumusunod ang robot sa itim na linya. Kapag nagkamali ang robot, tinitingnan nila ang code at ang baterya.
Natututo sila sa bawat pagkakamali. Sa Marso, sasali ang club sa paligsahan sa lungsod. Nag-eensayo ang mga estudyante pagkatapos ng klase sa silid-aralan.
Ipinagmamalaki ng paaralan ang kanilang mga robot. Sa susunod na linggo, ipapakita ng club ang mga robot sa buong paaralan. Magtatanong ang mga mas batang estudyante, at sasagot ang mga miyembro.
Mahalaga ang pagtutulungan sa bawat proyekto. Maraming salamat sa mga guro at sa mga magulang na tumutulong sa club. Masaya ang bawat estudyante kapag gumagana ang robot.
"""
text_words = words_of(my_text)
print("words in my_text:", len(text_words), "| different words:", len(set(text_words)))
assert len(text_words) >= 100, "write at least 100 words of your own text"

corp7 = list(Counter(text_words).items())
merges7, words7, curve7 = learn_bpe(corp7, 150)
print("merges learned:", len(merges7), "| first five:", [(a, b) for a, b, _ in merges7[:5] ])
```

```python
heldout = [w for s in my_sentences for w in words_of(s)]
print("held-out words:", len(heldout))
base7 = sorted({s for w, _ in corp7 for s in split_word(w)})
print(f"{'merges':>6s} {'vocabulary':>10s} {'tokens/word (train)':>20s} {'tokens/word (held-out)':>23s}")
curve_held = []
for m in range(len(merges7) + 1):
    curve_held.append(sum(len(encode_bpe(w, merges7[:m])) for w in heldout) / len(heldout))
for m in sorted({0, 20, 40, 80, len(merges7)}):
    print(f"{m:6d} {len(base7) + m:10d} {curve7[m]:20.3f} {curve_held[m]:23.3f}")

known = set(text_words)
unk_rate = sum(1 for w in heldout if w not in known) / len(heldout)
print(f"word-level vocabulary from my_text: {len(known)} words -> unknown rate on held-out words: {unk_rate:.2f}")
print("BPE has no unknown words: every held-out word can be spelled with pieces (checked below).")
assert all(decode_bpe(encode_bpe(w, merges7)) == w for w in heldout)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(curve7, label="training text"); ax.plot(curve_held, label="held-out sentences")
ax.set_xlabel("number of merges"); ax.set_ylabel("tokens per word"); ax.legend(); plt.title("Tokens per word against merges"); plt.tight_layout(); plt.show()
RESULTS["p7_curve"] = {m: round(curve_held[m], 3) for m in sorted({0, 20, 40, 80, len(merges7)})}
```

   **What you should see.**

   @@OUT 7.3a
   The paragraph has 152 words and 66 different ones (`words_of` counts the full stops as words), and the loop learns 98 merges before no pair occurs twice any more. Then:

   @@OUT 7.3b
   On the 50 held-out words the tokens per word fall from 5.56 with letters only to 3.80 after 20 merges, 3.46 after 40, 2.92 after 80 and 2.84 after all 98 merges. The training text falls much further (from 5.68 to 1.84): the gap between the two curves is the tokenizer memorising its own small text, and it grows with every merge. The first 20 merges (common endings and short words such as *ang*) save 1.76 tokens per word on new sentences; the next 78 save only 0.96 together. The word-level vocabulary of the same text would leave **38 percent** of the held-out words unknown (0.38), while BPE never fails (the assert checks that every held-out word is rebuilt exactly). This is a small experiment: with a few thousand words of training text the held-out curve would keep falling for much longer.

   **Try also (optional):**

   - Write a second paragraph of 100 more words, add it to `my_text`, and see how far the held-out curve falls.
   - Print the 20 longest pieces learned. Do they look like roots, prefixes, endings or whole words?
   - Plot the *marginal gain* `curve_held[m] - curve_held[m + 1]` against `m`.

   *Exercise 7.3:* Look at the held-out curve and choose a number of merges for your tokenizer (a whole number between 0 and the number of merges learned). Write two sentences of justification that use at least two numbers from the table: what you gain, what you pay (vocabulary size, rare pieces), and what you would do with more text. Why it matters: choosing the number of merges is the single most important decision when you train a BPE tokenizer.

```python
chosen_merges = None      # YOUR CHOICE: a whole number between 0 and len(merges7)
justification = ""        # YOUR ANSWER: two sentences that use numbers from the table
if chosen_merges is None or not justification:
    print("Fill in `chosen_merges` and `justification` above, then run this cell again.")
else:
    check("a valid number of merges", isinstance(chosen_merges, int) and 0 <= chosen_merges <= len(merges7), "choose between 0 and " + str(len(merges7)))
    check("your justification has at least 20 words", len(justification.split()) >= 20, "two sentences")
    check("it quotes at least two numbers", sum(ch.isdigit() for ch in justification) >= 2, "include numbers from the table, e.g. the tokens per word and the vocabulary size")
    if isinstance(chosen_merges, int) and 0 <= chosen_merges <= len(merges7):
        print(f"With {chosen_merges} merges: vocabulary {len(base7) + chosen_merges}, held-out tokens per word {curve_held[chosen_merges]:.3f}")
        RESULTS["p7_chosen_merges"] = chosen_merges
```

   *What it means.* The numbers of the default paragraph show diminishing returns from about 20 to 40 merges: most of the gain on new sentences is already made (by 40 merges the held-out cost has fallen from 5.56 to 3.46, 2.10 of the whole drop of 2.72 down to 2.84), and the training curve, which keeps falling, is partly fitting the paragraph. Your own text may give a different elbow, and that is fine as long as the number you choose comes from a curve you looked at.

4. **Step 7.4. Your results, together.** The dictionary `RESULTS` has been collecting your key numbers since Part 4. Print it as your summary: it is the first thing a reader of your notebook should see. Entries from parts you skipped are simply missing.

```python
for k in sorted(RESULTS):
    v = RESULTS[k]
    print(f"{k:24s} {str(v)[:90]}")
print("\nEntries:", len(RESULTS))
assert len(RESULTS) >= 8, "run the earlier parts so that RESULTS holds your numbers"
```

   **What you should see.** One line per entry, for example `p4_S1_tokens` with the four counts of Step 4.2, `p5_first_merge` with `('e', 's', 9)`, `p5_vs_gpt2` with the six comparisons, `p6_svd_gap` with `(0.828, 0.093)`, `p6_analogy_hits` with `[9, 6, 0, 5, 5]`, `p7_ratio_gpt2` with the GPT-2 ratio of Step 7.2 (2.0) and `p7_curve` with the held-out curve of Step 7.3. Check that the numbers agree with what you wrote in your answers.

5. **Step 7.5. Four reflection questions.** Answer in your own words, in at least 25 words each, using at least one number or example from this lab in each answer. There are no trick questions: the goal is to say clearly what you understood. If you cannot answer one, go back to the step named in brackets.

```python
reflection = {
    "1_language_model": "",      # What is a language model? (Part 1)
    "2_subwords": "",            # Why do language models use sub-word tokens rather than words or characters? (Parts 4 and 5)
    "3_embedding_matrix": "",    # What does an embedding matrix do, and where do its numbers come from? (Part 6)
    "4_new_language": "",        # What would you check before trusting a tokenizer on a language you have not tested? (Parts 4 and 7)
}
short = [k for k, v in reflection.items() if len(v.split()) < 25]
if len(short) == len(reflection):
    print("Write your four answers in `reflection` above (at least 25 words each), then run this cell again.")
else:
    check("all four answers have at least 25 words", not short, "still short: " + ", ".join(short))
RESULTS["p7_reflection"] = reflection
```

   **What a good answer contains.**

   - *Language model:* a model of P(next token given the tokens before it), learned from text, which scores or generates text; it predicts, it does not know that what it says is true. Cite perplexity or the sparsity of the n-gram tables.
   - *Sub-words:* words give a huge vocabulary and unknown words; characters give very long sequences; sub-words keep common words whole and spell rare ones from pieces; cite the tokens per word of Step 5.5, or the GPT-2 pieces of *unbelievably*.
   - *Embedding matrix:* a table with one row per token; the lookup picks a row (one-hot times the table); the numbers start random and are learned, by counts and SVD or by gradient descent, so that tokens that appear in similar contexts get similar vectors; cite the 38.6 million parameters of a GPT-2-sized table.
   - *New language:* measure tokens per word on real sentences against an English baseline, look at the pieces for sense, check the round trip and the unknown rate, test names, numbers and code-switching, and ask speakers of the language to read the output; cite the ratio of Step 7.2.

   **Common mistakes** (read before you submit):

   - Running a cell that uses `gpt2` or `bert` when the loading cell printed "Could not load": the later cells stop with a message on purpose. Fix the loading first.
   - Using a different tie-break in `learn_bpe` and then wondering why the merges differ from the lesson's: ties are broken by the pair seen first (Step 5.2).
   - Forgetting the end-of-word mark `</w>`: without it *est* in the middle of a word and *est* at the end of a word are the same, and the merges differ from the lesson's.
   - Counting tokens of a sentence with `len(text.split())` when you meant tokens: words and tokens are different (Step 4.5).
   - Comparing tokenizers on one sentence. Use several sentences and the total tokens divided by the total words (Step 7.2).
   - Reading a within-versus-between gap from one run as a law: change the seed or the corpus and look again (Step 6.6).
   - Taking an analogy hit as proof of quality: always compare with a control (Step 6.6).
   - Trusting a tokenizer curve measured on the training text only: measure on held-out text (Step 7.3).
   - Writing sentences in a language you are not sure about and then reading the numbers as a fact about that language. Write only what you know to be correct, or ask a speaker.

### Deliverables

A short checklist before you submit: every exercise has a written answer under its block; every `None` and every empty string in a starter cell has been replaced; the code blocks ran from the first to the last without an error; and the printed `RESULTS` summary of Step 7.4 is included.

Submit your work the way your instructor asks (the notebook with outputs, or your JupyterLab files with the printed output and your written answers). For Parts 4 to 7 it must include:

- **Part 4 (14 points):** the word and character counts for the four sentences and your `<unk>` counts (4.1), the GPT-2 and BERT pieces and token counts with your Imagine-first guesses compared (4.2), the round trips and your text that BERT cannot give back (4.3), the quirks table with the greeting comparison and your sentence about what surprised you (4.4), and the token-cost chart with the words-per-token ratio and your two estimates (4.5).
- **Part 5 (20 points):** the pair counts and the three tied pairs (5.1), the four merges checked against the lesson with your Imagine-first guesses compared (5.2), `learn_bpe` with ten merges and the vocabulary size 21 (5.3), `encode_bpe` and `decode_bpe` with the five encodings and your prediction for *slowest* (5.4), the 60 merges learned from `SCHOOL_TEXT` with the trade-off table and plot (5.5), the comparison with GPT-2 on six examples and your explanation (5.6), and your merge plan for a language with long words (5.7).
- **Part 6 (20 points):** the one-hot and lookup demonstration with the parameter counts (6.1), the co-occurrence heat map and the PPMI table with your PMI by hand (6.2), the SVD neighbours, the 2-D map, the within and between measurements and your comparison with the Imagine-first map (6.3), the skip-gram training loss, neighbours and the comparison table with the d = 2 model (6.4), the three cosines by hand (6.5), and the analogy test with its controls, the five-corpus spread, the skew experiment and your sentence about trusting analogies (6.6).
- **Part 7 (10 points):** your design card (7.1), your measurement on your own sentences with the ratio and your guess compared (7.2), the BPE trained on your own text with the curve, the chosen number of merges and your justification (7.3), the printed `RESULTS` summary (7.4), and your four reflection answers (7.5).

Together with Parts 1 to 3 the lab is marked out of **100 points**: Part 1 is worth 12, Part 2 is worth 14, Part 3 is worth 10, Part 4 is worth 14, Part 5 is worth 20, Part 6 is worth 20 and Part 7 is worth 10. The rubric is in the section "How Week 6 is assessed". The work graded is the one your instructor specifies.

**Stretch goals.** Write `encode_bpe` so that it works on bytes (start from the 256 byte values of the UTF-8 encoding) and check that it never fails on an emoji; implement WordPiece-style greedy longest-match encoding with a vocabulary of your own and compare the pieces with BPE for the same words; train the skip-gram model with negative sampling (a handful of random words instead of the full softmax) and compare the neighbours with the full softmax; build a corpus in which a gender direction really exists (sentences such as *the king said he* and *the queen said she*), and test whether the analogy hits become steadier; measure the bias gap of Step 6.6 for several strengths and plot it; for your own design card, write a one-page checklist of what you would test with ten speakers of your language before releasing the tokenizer.
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions cover the six outcomes. Eleven of them need a short calculation (a bigram probability, a sentence probability, a perplexity, a number of n-grams, a smoothed probability, a token count, a BPE pair count, a vocabulary size, a cosine, a parameter count), so keep a pencil nearby; the explanations show the arithmetic.

```quiz
{
  "id": "week-06-check",
  "title": "Week 6 quiz: Language models, tokens and word embeddings",
  "passMark": 70,
  "questions": [
    {
      "q": "Which statement best describes a language model?",
      "options": [
        "A program that translates each word into another language with a dictionary",
        "A table that gives every word one fixed meaning",
        "A model that gives a probability to the next word (or token) given the words before it",
        "A rule book that lists every correct sentence of a language"
      ],
      "answer": 2,
      "why": "A language model estimates P(next | previous words). Chaining these predictions gives the probability of a whole text and lets the model write text one token at a time. A dictionary, a fixed table of meanings and a rule book are not probability models."
    },
    {
      "q": "In a text, the word “the” is followed by another word 9 times: “students” 4 times, “canteen” 3 times, “teacher” once and “counter” once. What is the bigram probability P(students | the)?",
      "options": [
        "4/9 ≈ 0.44",
        "3/9 ≈ 0.33",
        "1/4 = 0.25",
        "9/4 = 2.25"
      ],
      "answer": 0,
      "why": "P(students | the) = count(the students) / count(the) = 4/9 ≈ 0.444. 3/9 is the probability of “canteen”. 1/4 would be right only if the four followers were equally common. 9/4 is greater than 1, so it cannot be a probability."
    },
    {
      "q": "A model gives P(Students) = 0.1, P(eat | Students) = 0.2 and P(rice | Students eat) = 0.5. What probability does the chain rule give for the sentence “Students eat rice”?",
      "options": [
        "0.8",
        "0.27",
        "0.1",
        "0.01"
      ],
      "answer": 3,
      "why": "The chain rule multiplies the steps: 0.1 × 0.2 × 0.5 = 0.01. Adding them gives 0.8 and averaging them gives about 0.27, but probabilities of successive steps are multiplied, not added. 0.1 is only the first factor."
    },
    {
      "q": "A three-word sentence has a total probability of 0.001 under a language model. What is its perplexity, PP = P^(−1/n)?",
      "options": [
        "333",
        "10",
        "1,000",
        "0.1"
      ],
      "answer": 1,
      "why": "PP = 0.001^(−1/3). Since 0.001 = 10⁻³, raising it to the power −1/3 gives 10¹ = 10: the model was as unsure as if it chose among about 10 equally likely words at each step. 333 divides 1000 by 3 instead of taking a cube root, 1,000 is 1/P, and 0.1 forgets the minus sign in the exponent."
    },
    {
      "q": "A vocabulary has V = 5,000 words. How many different trigrams (three-word sequences) are possible?",
      "options": [
        "15,000",
        "25,000,000",
        "125,000,000,000",
        "5,000"
      ],
      "answer": 2,
      "why": "The number of possible n-grams is V^n, so 5,000³ = 125,000,000,000 = 1.25 × 10¹¹. 25,000,000 is V², the number of bigrams. 15,000 is 3 × V, and 5,000 is V itself. This explosion is why most trigrams are never seen in any text."
    },
    {
      "q": "In a training text the bigram “robots eat” never occurs. The word “robots” appears 12 times with a word after it, and the vocabulary has V = 88 words. What is the add-one smoothed probability P(eat | robots)?",
      "options": [
        "0",
        "1/12 ≈ 0.083",
        "1/88 ≈ 0.011",
        "1/100 = 0.01"
      ],
      "answer": 3,
      "why": "Add-one smoothing gives (count + 1) / (count(prev) + V) = (0 + 1) / (12 + 88) = 1/100 = 0.01. Zero is the unsmoothed answer, the problem smoothing fixes. 1/12 forgets that the denominator grows by V, and 1/88 forgets the 12 real occurrences in the denominator."
    },
    {
      "q": "What is the main weakness of a rule-based system such as ELIZA?",
      "options": [
        "Hand-written rules cannot cover the variety of real language, so new inputs break it",
        "It needs billions of parameters and a lot of training data",
        "It stores a count for every possible n-gram, which is too large",
        "It gives every word the same vector"
      ],
      "answer": 0,
      "why": "A rule-based system does only what people wrote rules for, and language is too varied for any list of rules. Needing billions of parameters is a trait of large neural models, a table of n-gram counts belongs to the statistical era, and the vector issue belongs to word embeddings."
    },
    {
      "q": "Using the rule of thumb of about three-quarters of an English word per token, about how many tokens is a 300-word English text?",
      "options": [
        "225",
        "400",
        "300",
        "900"
      ],
      "answer": 1,
      "why": "If one token is about 0.75 words, then T = W ÷ 0.75 = 300 ÷ 0.75 = 400 tokens. 225 multiplies by 0.75 instead of dividing, which would make the tokens fewer than the words. 300 assumes one token per word, which is true only for a word-level tokenizer."
    },
    {
      "q": "A student proposes using whole words as the tokens, “since words are what people read”. Which problem will they meet?",
      "options": [
        "Every sentence becomes much longer than it would with characters",
        "The vocabulary is tiny, so the model has nothing to learn from",
        "New, rare or misspelt words are not in the vocabulary and become unknown tokens",
        "Words cannot be turned into integer IDs"
      ],
      "answer": 2,
      "why": "A word-level vocabulary has no way to write a word it has not listed, so names, new words and typos become <unk>. Sequences are shorter with words than with characters, not longer. The vocabulary is very large, not tiny, and any entry in a table can have an ID."
    },
    {
      "q": "A corpus is hug ×10, pug ×5, pun ×12, bun ×4, hugs ×5, with every word written as letters plus an end-of-word mark. How many times does the pair “u g” occur?",
      "options": [
        "20",
        "17",
        "16",
        "15"
      ],
      "answer": 0,
      "why": "The pair “u g” appears in hug (10), pug (5) and hugs (5): 10 + 5 + 5 = 20, which is why it is the first merge. 17 is the count of “p u” (pug 5, pun 12), 16 is “u n” (pun 12, bun 4), and 15 is “h u” (hug 10, hugs 5)."
    },
    {
      "q": "A BPE tokenizer starts from the 26 lower-case letters plus an end-of-word mark and learns 500 merges. How large is its vocabulary?",
      "options": [
        "500",
        "526",
        "26",
        "527"
      ],
      "answer": 3,
      "why": "|V| = B + M. The base is 26 letters + 1 end-of-word mark = 27 symbols, and each merge adds one new symbol, so 27 + 500 = 527. 526 forgets the end mark. 500 counts only the merges, and 26 only the letters."
    },
    {
      "q": "A tokenizer learned these merges in this order: 1. e + s, 2. es + t, 3. est + </w>, 4. l + o, 5. lo + w. How many tokens does the new word “lowest” become?",
      "options": [
        "7",
        "2",
        "3",
        "1"
      ],
      "answer": 1,
      "why": "Start from l o w e s t </w> (7 symbols). Merge 1 gives es, merge 2 gives est, merge 3 gives est</w>, merge 4 gives lo, merge 5 gives low. The result is low + est</w>: 2 tokens. 7 is the count before any merge, and the merges must be applied in the order they were learned."
    },
    {
      "q": "Two word vectors are a = (6, 8) and b = (8, 6). What is their cosine similarity?",
      "options": [
        "0.48",
        "1.00",
        "0.96",
        "0.75"
      ],
      "answer": 2,
      "why": "The dot product is 6×8 + 8×6 = 96. Both vectors have length √(36 + 64) = 10, so cos = 96 / (10 × 10) = 0.96. They point in almost the same direction but not exactly, so the cosine is below 1 even though the lengths are equal. 0.75 is only 6/8."
    },
    {
      "q": "A model has a vocabulary of 30,000 tokens and an embedding size of 512. How many parameters are in its embedding matrix E?",
      "options": [
        "15,360,000",
        "30,512",
        "15,360",
        "61,440,000"
      ],
      "answer": 0,
      "why": "E has V rows and d columns, so it holds V × d = 30,000 × 512 = 15,360,000 parameters. 30,512 adds the two sizes instead of multiplying. 15,360 is off by a factor of 1,000. 61,440,000 would be the size in bytes with 4 bytes per number, not the number of parameters."
    },
    {
      "q": "Which statement about word embeddings is correct?",
      "options": [
        "Linguists type in the numbers of E before training starts",
        "Looking up a row of E is a different operation from multiplying a one-hot vector by E",
        "A cosine similarity of 1 means two words mean the same in every context",
        "A static embedding gives “bank” the same vector in every sentence, and vectors can reflect stereotypes in the text they were learned from"
      ],
      "answer": 3,
      "why": "A static embedding has one vector per token, so “river bank” and “bank account” share it (Week 7’s attention makes vectors depend on context), and because the vectors are learned from text they can carry its stereotypes. The numbers of E are learned by backpropagation, a row lookup equals a one-hot multiplication, and a cosine of 1 only says that two vectors point the same way."
    }
  ]
}
```

## How Week 6 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: Be a language model: count and predict (word counts, bigram table, chain rule, perplexity, add-one smoothing, generation) | 12 | Words counted and the top-10 table and chart shown; bigram counts and the probabilities after "the" checked to sum to 1; sentence probabilities by the chain rule shown as a product and as a log, with the scrambled sentence explained; the unseen bigrams of the test text found and the infinite perplexity explained; add-one smoothing implemented, perplexities compared with a unigram baseline and a probability row checked; sentences generated and judged; three kinds of knowledge that counting cannot capture written | Steps run and numbers are right; the explanation of the unseen bigrams or of smoothing is brief | Counts or probabilities wrong; no perplexity; the zero-probability problem not identified; conclusions not backed by the printed numbers |
| Part 2: From rules to counts to neural: three tiny language models (rule-based bot, bigram model, neural model, embeddings, comparison) | 14 | A rule-based bot with six rules run on eight inputs and its coverage counted; the bigram model on the same inputs with the gaps noted; the neural model trained with loss and perplexities on both texts and compared honestly with the smoothed bigram model; cosine of four word pairs measured and read; the summary table of the three eras with sizes in numbers; a reasoned choice of era for the three uses | Everything runs and the checks pass; the comparison of the three models or the reasons for the three uses are general | A model missing; perplexities not compared or the neural model's result misreported; no measured cosines |
| Part 3: The limits of n-grams: sparsity (possible against seen, unseen in new text, interpolation, copying, neural alternative) | 10 | The table and log plot of distinct, possible and seen n-grams for n = 1 to 4 related to the lesson's figure; the share of unseen n-grams in the test text measured and plotted; interpolation tried for five values of lambda with the best value chosen from the plot; the copy rate measured for n = 2, 3 and 4 and explained; a clear paragraph on how a neural model avoids the table of V to the power n entries | Everything runs; the reading of the plots is general | Possible counts wrong; unseen share or interpolation absent; no explanation of the neural alternative |
| Part 4: Tokenisation in the real world (words and characters by hand, GPT-2 and BERT, round trip, quirks, token cost) | 14 | Word and character tokens counted for four sentences and the `<unk>` counts of the word vocabulary found; GPT-2 (13 tokens) and BERT (15 tokens) pieces for the Imagine-first sentence compared with the guesses; the round trip asserted for GPT-2 and a text that BERT cannot give back; the quirks table read carefully, with the Filipino and Cebuano greetings costing 2.125 times the English ones and a sentence about what surprised you that states only what was measured; the words-per-token ratio of 0.917 measured and the essay estimates computed | Steps run and numbers are right; the reading of the quirks or the cost comparison is brief | Counts wrong; no comparison with the guesses; claims about languages or tokenizers that were not measured |
| Part 5: Build a BPE tokenizer from scratch (pair counts, merges, learning, encoding, bigger corpus, comparison with GPT-2) | 20 | `pair_counts`, `merge_pair`, `learn_bpe`, `encode_bpe` and `decode_bpe` written and working; the first four merges equal to the lesson's (`e s`, `es t`, `est </w>`, `l o`) by assert, with the three-way tie explained; ten merges and the vocabulary size 21; unseen words encoded and decoded, with the character fallback explained; 60 merges learned from `SCHOOL_TEXT` with the trade-off table and plot (5.079 down to 2.256 tokens per word); the comparison with GPT-2 on six examples explained by training text, merges and bytes; a reasoned plan for the number of merges for a language with long words | Everything runs and the checks pass; the explanation of the tie-break, of the trade-off or of the comparison with GPT-2 is brief | A function missing or its merges different from the lesson's with no explanation; the end-of-word mark ignored; no comparison with a real tokenizer |
| Part 6: Word embeddings from scratch (one-hot, co-occurrence and PPMI, SVD, skip-gram, cosine, analogies and bias) | 20 | One-hot cosines of 0 and the lookup equal to one-hot times the table, with parameter counts (336 and 38,597,376); the co-occurrence heat map and the PPMI by hand; SVD neighbours and the 2-D map compared with the Imagine-first sketch, within and between cosines measured (0.828 against 0.093) with the assert; a skip-gram network trained and compared with SVD by the same measure; three cosines by hand equal to NumPy and PyTorch; analogies reported honestly with the chance level, the controls and the spread over five corpora; the skew experiment read as an effect of the text | Everything runs and the checks pass; the reading of the maps, of the analogies or of the bias experiment is general | Embeddings not trained or not measured; cosine wrong; analogy hits reported without a control or as proof; no discussion of bias |
| Part 7: Design a tokenizer for a community, and reflect (design card, measurement, own BPE, results, reflection) | 10 | A specific design card for your own app with a vocabulary number and reasons about the language and the users; tokens per word measured with GPT-2 and BERT on your own sentences with an English baseline and your guess compared; a BPE trained on at least 100 words of your own text with the held-out curve, a chosen number of merges and a justification that uses numbers; `RESULTS` printed; four reflection answers that cite numbers or examples from the lab | Everything present; the design is generic or the answers are general | Card or reflection missing; no measurement on your own text; no justification for the number of merges |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave (your instructor will use them to open Week 7):
1. In the canteen text, “the” is followed by a word 9 times, and “the teacher” occurs once. The vocabulary has V = 23 words. Give the unsmoothed P(teacher | the) and the add-one smoothed value, and say what the smoothing bought us.
2. With the merges e + s, es + t, est + </w>, l + o, lo + w, n + e, ne + w, new + est</w>, encode the word “newer”. How many tokens is it, and which merges did you use?
3. An embedding matrix has 10,000 rows and 256 columns. How many parameters does it hold, and what does the lookup for the token with ID 7 return?

**Homework / self-study**
- Finish the lab, including the written answers and the Part 7 design.
- Re-read the `equation` figures in this lesson (the chain rule, perplexity, the bigram and n-gram probabilities, V^n, add-one smoothing, interpolation, tokens from words, BPE size and tokens per word, the embedding lookup and cosine similarity) and, for each, write one sentence in your own words about what every symbol means.
- Preview Week 7: attention. You saw that a static embedding gives “bank” one vector in every sentence; next week you learn how a model lets a token’s vector depend on the other words in the sentence.
