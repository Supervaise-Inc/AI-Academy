> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 1 — Introduction to Generative AI

**Purpose:** Understand what generative AI is, how it ==learns a distribution and then samples from it==, and how to judge what it makes.

**Term:** Prelim (Weeks 1–5) · **Month:** 1 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a model make something new",
  "caption": "You write a prompt; a generative model returns content that did not exist before: text, a translation, a picture, a voice. The last tab shows the secret: for text, the model first scores every possible next word. These examples are made up for the animation; real output is different every time.",
  "scenes": [
    { "name": "Text", "kind": "text", "outputLabel": "New text", "prompt": "Once upon a time in a land far, far", "output": "away, there lived a weaver who could hear what the wind was saying." },
    { "name": "Translation", "kind": "text", "outputLabel": "New text, in French", "prompt": "Hello, how are you?", "output": "Bonjour, comment allez-vous ?" },
    { "name": "Image", "kind": "image", "outputLabel": "New image", "prompt": "A futuristic city with flying cars" },
    { "name": "Speech", "kind": "speech", "outputLabel": "New audio", "prompt": "Hello, this is a test of text to speech." },
    { "name": "The secret", "kind": "bars", "outputLabel": "Chance of each next word (illustrative)", "prompt": "Once upon a time in a land far, far",
      "bars": [
        { "label": "away", "value": 0.73 },
        { "label": "beyond", "value": 0.12 },
        { "label": "from", "value": 0.09 },
        { "label": "off", "value": 0.04 },
        { "label": "behind", "value": 0.02 }
      ] }
  ]
}
```

## By the end, you can…

1. ==Explain what generative AI is== and how it differs from discriminative AI, which labels or scores data that already exists.
2. ==Place AI, machine learning, deep learning and generative AI== inside one another like Russian dolls.
3. Explain the idea ==“learn a distribution, then sample”== using a small counting model.
4. ==Compute the probability of a sentence== from next-word probabilities with the chain rule, and explain perplexity.
5. ==Use temperature, top-k and top-p== to control how adventurous a model is, and say what each one does.
6. List what generative AI is used for and its risks, and ==apply the five trustworthy-AI principles== to a real case.

Assessed in the Week 1 lab and quiz. Read each section, then play with the figure under it. The figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. Along the way you will see notes such as “Lab Part 3”: they tell you which part of the lab lets you try that idea for real.

## What is generative AI? {rail=Concept}

{lede} ==Generative AI is software that creates new content== (text, pictures, sound, code) by learning the patterns in a very large collection of examples. It sits inside a family of ideas, like a set of Russian dolls, and it works by answering a different question from the one older AI answered.

### You already use it

You have probably met generative AI before this course, even if nobody called it that. When your phone suggests the next word as you type a message in Filipino or English, a small model is guessing which word is likely to come next. When an app translates a barangay announcement from English into Tagalog or Cebuano, a model is writing a new sentence in another language. When a camera app turns a photo into a cartoon, or a voice assistant reads your message aloud, a model is producing new pixels or new sound.

What these have in common is that the answer is *made*, not *looked up*. Nobody stored the sentence you are about to receive. The model builds it from patterns it learned earlier. That one idea is the whole subject of this week.

### The Russian dolls

Four terms are often used as if they meant the same thing. They do not. Each one fits inside the one before it. [[Artificial intelligence (AI)|The broad goal of making computers do tasks that normally need human intelligence, such as understanding language, recognising images or making decisions.]] is the biggest doll. [[Machine learning (ML)|A way of building AI where the computer learns patterns from examples instead of following rules a programmer wrote by hand.]] is the next one: instead of writing rules, we show the computer examples. [[Deep learning|Machine learning that uses neural networks with many layers. It is behind most of today's image, speech and language systems.]] is machine learning with many-layered neural networks. And ==generative AI is deep learning used to create new content==.

Click each ring to read about it. Notice that the outer rings are *bigger ideas*: not every AI system learns (an old chess program follows hand-written rules), and not every machine-learning system is deep.

```viz
{
  "type": "nested",
  "title": "AI, machine learning, deep learning, generative AI",
  "levels": [
    { "name": "Artificial intelligence", "blurb": "The whole field: any technique that lets a computer do something that normally needs human intelligence. It includes rule-based programs as well as systems that learn from data.", "examples": ["A chess program using hand-written rules", "ELIZA (1966), a rule-based chatbot", "A route planner"] },
    { "name": "Machine learning", "blurb": "AI that learns patterns from examples instead of following rules written by a programmer. You give it data, and it finds the rule itself.", "examples": ["A spam filter learned from emails", "A house-price predictor", "A random forest"] },
    { "name": "Deep learning", "blurb": "Machine learning with neural networks that have many layers. Given enough data and computing power, it handles pictures, sound and language well.", "examples": ["Face unlock", "Speech recognition", "AlexNet (2012)"] },
    { "name": "Generative AI", "blurb": "Deep learning models that create new content by learning what their training data looks like and then producing more of it.", "examples": ["GPT-2 and ChatGPT (text)", "Stable Diffusion (images)", "MMS-TTS (speech)"] }
  ]
}
```

Click the innermost ring, then the outermost. Generative AI is the smallest doll, and it is the one this course is about.

### Two different questions

Imagine two helpers in a school canteen. The first, a cashier, looks at a tray and says “that is a ₱45 meal”. The second, a cook, has tasted thousands of dishes and can *make a new dish* that fits the menu. The cashier answers a question about something that already exists. The cook creates.

A [[discriminative model|A model that takes existing data and gives back a label or a number, such as “spam” or a predicted price. It does not create anything new.]] is the cashier. Spam filters, face unlock and house-price predictors are discriminative. A [[generative model|A model that learns the patterns in a large amount of data and then produces new data that follows them, such as text, images or audio.]] is the cook. A [[large language model (LLM)|A generative model trained on huge amounts of text. It writes by repeatedly predicting the next word.]] such as GPT-2 or ChatGPT, and image models such as Stable Diffusion, are generative.

Switch between the two below and try every example. Notice what comes out of the model each time: a label or a number on one side, and brand-new content on the other.

```viz
{
  "type": "flow",
  "title": "Two ways to use a model",
  "modes": [
    {
      "name": "Discriminative",
      "tagline": "Takes data that already exists and answers a question about it. The answer comes from a fixed set: a label or a number.",
      "inputLabel": "Input",
      "modelLabel": "learned a boundary between classes",
      "outputLabel": "Output: a label or a number",
      "examples": [
        { "label": "Spam filter", "input": "“You have WON a free phone, click now!!”", "output": "spam" },
        { "label": "House prices", "input": "80 m², 2 bedrooms, near a train station", "output": "₱ 4.2 million (an estimate)" },
        { "label": "Face unlock", "input": "A photo from the phone camera", "output": "owner: yes" }
      ],
      "note": "These models do not create anything. They sort or score what you give them."
    },
    {
      "name": "Generative",
      "tagline": "Learns the patterns in lots of data, then makes new data that follows them. The answer did not exist before you asked.",
      "inputLabel": "Input: a prompt",
      "modelLabel": "learned what the data looks like",
      "outputLabel": "Output: new content",
      "examples": [
        { "label": "Text (GPT-2, ChatGPT)", "input": "“Once upon a time in a land far, far away”", "output": "…there lived a young weaver who could hear what the wind was saying." },
        { "label": "Translation", "input": "“Hello, how are you?” (English)", "output": "“Bonjour, comment allez-vous ?” (French)" },
        { "label": "Image (Stable Diffusion)", "input": "“A futuristic city with flying cars”", "output": "A brand-new picture of that city" },
        { "label": "Speech", "input": "“Hello, this is a test.”", "output": "An audio clip of a voice saying it" }
      ],
      "note": "The text example is made up to show the idea. Real model output differs every time you run it."
    }
  ]
}
```

### Try it: discriminative or generative?

Decide for each example whether the system ==generates new data== or ==classifies or predicts==. You get an explanation either way. One card is an edge case on purpose, so a defended answer counts.

```viz
{
  "type": "sorter",
  "title": "Sort the ten examples",
  "cards": [
    { "name": "Spam filter", "answer": "discriminative", "why": "It labels an existing email as spam or not spam. Nothing new is created." },
    { "name": "House-price predictor", "answer": "discriminative", "why": "It predicts a number from the features of a house." },
    { "name": "ChatGPT", "answer": "generative", "why": "It writes new text, one piece at a time, in reply to your prompt." },
    { "name": "Stable Diffusion", "answer": "generative", "why": "It creates a new image from a text description." },
    { "name": "Logistic regression", "answer": "discriminative", "why": "A classic classifier. It outputs the probability of a class." },
    { "name": "Random forest", "answer": "discriminative", "why": "Many decision trees vote on a label or a number." },
    { "name": "Support vector machine", "answer": "discriminative", "why": "It finds the boundary that best separates classes." },
    { "name": "GPT-2", "answer": "generative", "why": "A language model that continues any text you give it. You will run it in the lab." },
    { "name": "Whisper (speech to text)", "answer": "either", "why": "Edge case. It writes new text from audio using a sequence model, so it does generate text. But the text is a transcript of what was said, so it also behaves like a translator or classifier. Either answer is fine if you can defend it." },
    { "name": "Face unlock", "answer": "discriminative", "why": "It decides whether the face matches the owner: yes or no." }
  ]
}
```

### The same difference, written as maths

Scientists write the difference between the two kinds of model with one short formula each. You do not need to be afraid of them: each is a question in symbols. We use [[probability|A number from 0 to 1 that says how likely something is. 0 means it never happens and 1 means it always does.]] to measure “how likely”.

The discriminative question is: *given this input, how likely is each label?* In symbols this is P(y | x). Select each part of the formula below, then step through the worked example.

```viz
{
  "type": "equation",
  "title": "The discriminative question: P(y | x)",
  "intro": "Select each part of the formula to see what it means, then read the worked example underneath.",
  "formula": [
    { "t": "P", "id": "p" },
    "(",
    { "t": "y", "id": "y" },
    " ",
    { "t": "|", "id": "given" },
    " ",
    { "t": "x", "id": "x" },
    ")"
  ],
  "parts": [
    { "id": "p", "name": "P (probability)", "meaning": "A number from 0 to 1 that says how likely something is. 0.92 means 92 chances in 100." },
    { "id": "y", "name": "y (the label)", "meaning": "The answer we want, chosen from a fixed list: for example spam or not spam." },
    { "id": "given", "name": "| (read “given”)", "meaning": "The bar means “given that we know”. Everything to its right is what we already have." },
    { "id": "x", "name": "x (the input)", "meaning": "The data we are looking at: for example the text of one email." }
  ],
  "steps": [
    { "title": "Read it in words", "expr": "P(spam | this email) = “how likely is it spam, given this email?”", "note": "The model looks at one existing email and scores the labels." },
    { "title": "A number", "expr": "P(spam | email) = 0.92  →  P(not spam | email) = 1 − 0.92 = 0.08", "note": "The two labels share all the probability, so they add up to 1. The filter says “spam” because 0.92 is bigger. Illustrative numbers." }
  ],
  "note": "The spam probability is an illustrative number authored for teaching, not from a real filter."
}
```

The generative question is bigger: *how likely is this whole piece of data, and can I produce more of it?* In symbols this is P(x), or P(x | prompt) when you give a prompt. A model that can score every possible text or picture also lets you ==draw new ones from it==. That is exactly what “generate” means.

```viz
{
  "type": "equation",
  "title": "The generative question: P(x | prompt)",
  "intro": "Same symbols, but now x is a whole piece of data (a sentence, a picture). Select each part, then read the example.",
  "formula": [
    { "t": "P", "id": "p" },
    "(",
    { "t": "x", "id": "x" },
    " ",
    { "t": "|", "id": "given" },
    " ",
    { "t": "prompt", "id": "prompt" },
    ")"
  ],
  "parts": [
    { "id": "p", "name": "P (probability)", "meaning": "How likely the model thinks something is. Bigger means more natural-looking." },
    { "id": "x", "name": "x (the new data)", "meaning": "A complete piece of content: a sentence, a picture or a sound clip. This is what the model will produce." },
    { "id": "given", "name": "| (read “given”)", "meaning": "Everything to the right is what the model has been told first." },
    { "id": "prompt", "name": "prompt", "meaning": "Your instruction or starting text, for example “Good …”. With no prompt, the formula is just P(x)." }
  ],
  "steps": [
    { "title": "Score a natural sentence", "expr": "P(“good morning”) = 0.0004", "note": "A tiny number, because there are millions of possible two-word sentences. Illustrative." },
    { "title": "Score a scrambled one", "expr": "P(“morning good”) = 0.0000006", "note": "Far smaller: people almost never say it. Illustrative." },
    { "title": "Compare", "expr": "0.0004 / 0.0000006 ≈ 667", "note": "The model thinks “good morning” is about 667 times more likely. Sampling from this model will produce the natural sentence far more often." }
  ],
  "note": "These two probabilities are illustrative numbers authored for teaching. A real model gives similar relative sizes but different values."
}
```

Notice that the generative formula judges a *whole* sentence, which is a much harder job than labelling an email. The next sections show how a model learns to do it, and how it breaks it into small steps.

## How a generative model learns {rail=Learning}

{lede} The core idea of generative AI fits in six words: ==learn a distribution, then sample==. First the model learns which things are likely in a huge pile of examples. Then it draws new things according to those likelihoods.

### Learn a distribution, then sample

Think of a chef in a school canteen who has tasted thousands of dishes. She did not memorise every dish. She learned what usually goes with what: rice with fish, noodles with vegetables, sugar with rice cakes. Now ask her to invent a dish. She does not copy one she tasted. She picks combinations that *feel likely* from everything she has learned. That is what a generative model does.

The words need care. A [[probability distribution|A list of all the possible outcomes together with how likely each one is. The probabilities add up to 1.]] is simply a list of what can happen and how likely each thing is. “Learning the distribution” means the model gets good at knowing which things are likely. To [[sample|To pick one outcome at random, with the more likely ones picked more often. Sampling is how a model chooses a word or a pixel.]] is to pick one thing at random, favouring the likely ones. Both steps are needed. Learning without sampling gives you only scores. Sampling without learning gives you random noise.

The recipe is always the same three steps:

1. **Collect data.** Gather lots of examples of what you want to make: sentences, photos, sound clips.
2. **Learn what is likely.** Adjust the model until it gives high probability to the examples it saw and to things like them.
3. **Sample.** Draw new items from the learned distribution, one after the other.

Step through the figure with a tiny text example. This is the same pattern for every model in this course, from a counting trick to a Transformer.

```viz
{
  "type": "pipeline",
  "title": "Learn, then sample",
  "intro": "Press the steps one by one. The highlighted boxes show where you are.",
  "nodes": [
    { "id": "data", "label": "Data", "sub": "many example texts" },
    { "id": "learn", "label": "Learn", "sub": "count / adjust" },
    { "id": "dist", "label": "Distribution", "sub": "what is likely" },
    { "id": "sample", "label": "Sample", "sub": "draw at random" },
    { "id": "new", "label": "New content", "sub": "not in the data" }
  ],
  "steps": [
    { "title": "1. Collect data", "body": "Gather examples of the thing you want to make. The model can only learn from patterns that are in the data, so what you collect decides what it can make.", "active": ["data"], "example": "Three canteen menus: “rice and fish”, “rice and eggs”, “noodles and fish”." },
    { "title": "2. Learn what is likely", "body": "Training adjusts the model so that the examples, and things like them, get high probability. In the simplest model this is just counting.", "active": ["data", "learn"], "example": "After “rice and” we saw “fish” once and “eggs” once. After “noodles and” we saw “fish” once." },
    { "title": "3. The result is a distribution", "body": "The learned model is a table or a function that gives a probability to every possible next piece. This is what “knowing the data” means.", "active": ["learn", "dist"], "example": "P(next = fish | “rice and”) = 0.5, P(eggs | “rice and”) = 0.5." },
    { "title": "4. Sample from it", "body": "Draw one item at random, using the probabilities as chances. Likely items come out often and unlikely ones rarely.", "active": ["dist", "sample"], "example": "Draw: eggs (a 50 percent chance)." },
    { "title": "5. New content appears", "body": "Repeat the draw and you get content that follows the learned patterns but was never in the data as a whole.", "active": ["sample", "new"], "example": "“noodles and eggs” was never on a menu, but it is a plausible new one." }
  ]
}
```

Look at step 5. The combination “noodles and eggs” never appeared in the data, yet the model could produce it. This ability to combine the learned patterns in new ways is what makes the output *new*.

### The simplest real generative model

You can build a working generative model with nothing but counting. An [[n-gram model|A language model that predicts the next word using only the previous n−1 words. A bigram model uses just the one word before.]] counts, for each word in a text, which words come after it. Then it writes by repeatedly drawing a next word from those counts. The figure below does this for a short text about a school canteen. Pick a word to see its next-word probabilities, then press the generate button a few times.

```viz
{
  "type": "ngram",
  "title": "A counting model for the school canteen",
  "text": "the canteen opens at seven. the canteen sells rice and fish. the students buy rice and eggs. the students eat at noon. the teacher eats at noon. the canteen closes at four.",
  "start": "the"
}
```

Every sentence it writes is built only from what it counted. Many are sensible, such as “the students eat at noon”. Some are new combinations that were never in the text, such as “the students buy rice and fish”, and some may be odd. Look at what follows “the”: it could be “canteen”, “students” or “teacher”. That is a [[probability distribution|A list of all the possible outcomes together with how likely each one is. The probabilities add up to 1.]] in action: the counts, turned into chances.

This really is a generative model: it learned a distribution from data, and it makes new text by sampling. It also shows the weakness of the simplest approach. It only sees *one word* of context. After “at” it cannot tell whether the sentence is about opening, closing or eating, so it can wander between topics. Longer contexts help, but the counts then become too sparse to use. Modern language models solve this with neural networks that read hundreds or thousands of words at once. You will build this counting model yourself in Lab Part 2.

### Training: make the data more probable

How does a neural model learn the distribution? It plays a guessing game on the training text. At each position it is shown the words so far and must give a probability to the word that really came next. If it gave the right word a high probability, it did well. If it gave it a tiny probability, it was surprised, and it must change.

The size of the surprise is measured by the [[loss|A single number that says how badly a model did on an example. Training changes the model to make the loss smaller.]]. For next-word prediction the loss is −log p, where p is the probability the model gave to the true next word. We use the natural logarithm (ln). Select the parts of the formula below.

```viz
{
  "type": "equation",
  "title": "The training loss: −log p",
  "intro": "Select each part, then step through two cases to see how a good guess and a bad guess are scored.",
  "formula": [
    { "t": "loss", "id": "loss" },
    " ",
    { "t": "=", "id": "eq" },
    " ",
    { "t": "−", "id": "minus" },
    { "t": "log", "id": "log" },
    " ",
    { "t": "p", "id": "p" }
  ],
  "parts": [
    { "id": "loss", "name": "loss", "meaning": "How badly the model did on this word. Zero would be perfect; bigger is worse. Training tries to make the average loss small." },
    { "id": "eq", "name": "=", "meaning": "The loss is computed from the probability by the rule on the right." },
    { "id": "minus", "name": "− (minus sign)", "meaning": "The log of a number between 0 and 1 is negative. The minus sign flips it so the loss is positive." },
    { "id": "log", "name": "log (natural logarithm)", "meaning": "Turns very small probabilities into manageable numbers. log 1 = 0, and log gets more negative as the probability shrinks towards 0." },
    { "id": "p", "name": "p", "meaning": "The probability the model gave to the word that really came next in the training text." }
  ],
  "steps": [
    { "title": "A good guess", "expr": "p = 0.5  →  loss = −ln(0.5) = 0.693", "note": "The model gave the true word a 50 percent chance, so the surprise is small." },
    { "title": "A bad guess", "expr": "p = 0.01  →  loss = −ln(0.01) = 4.605", "note": "Only a 1 percent chance: a big surprise, so a big loss. Training will nudge the model to give this word more probability." },
    { "title": "A perfect guess", "expr": "p = 1  →  loss = −ln(1) = 0", "note": "No surprise at all." }
  ],
  "note": "The probabilities 0.5 and 0.01 are examples chosen to show the scale; the loss values are exact."
}
```

Notice how the loss grows slowly for medium guesses and fast for very bad ones: moving from 0.5 to 0.01 multiplies the loss by about 6.6. A model is punished hard for being confidently wrong. The recipe for changing the model to lower the loss is called [[gradient descent|A method that adjusts a model's numbers a little at a time in the direction that makes the loss smaller. It is repeated millions of times.]], and you will meet it properly in Week 2. For now, remember that ==training means making the real data more probable under the model==.

## How a language model writes {rail=Writing}

{lede} A language model never writes a whole paragraph in one go. It ==repeats one small step: look at the words so far, score every possible next word, pick one, add it and repeat==. Everything else (temperature, top-k, top-p, even mistakes) happens inside that loop.

### Tokens: the pieces a model reads

Models do not read letters or whole words. They read [[tokens|The small pieces of text a model works with. A token can be a whole word, part of a word or punctuation. In English one token is roughly three-quarters of a word.]]. Common words are one token each; rarer words are cut into pieces. The word “unbelievable” might become “un · believ · able”. In English, a good rule of thumb is that ==one token is about three-quarters of a word==, so 100 words is roughly 130 tokens. Other languages, such as Tagalog, are often split into more tokens per word, because the vocabulary was built mostly from English text.

Why pieces? A vocabulary of whole words would be enormous and could never include every name or new word. With pieces, a model can build any word it has never seen. A model such as GPT-2 has a vocabulary of about 50,000 tokens. You will see real tokens in Lab Part 1.

### The loop

Here is the whole process. Step through it and watch which stage lights up. At the end, the chosen word is added to the text and the loop starts again.

```viz
{
  "type": "pipeline",
  "title": "How a language model writes: one word at a time",
  "intro": "Press the steps one by one. Imagine the loop running again and again until the text is done.",
  "nodes": [
    { "id": "prompt", "label": "Text so far", "sub": "your prompt" },
    { "id": "tokens", "label": "Tokens", "sub": "pieces → numbers" },
    { "id": "scores", "label": "Scores", "sub": "one per word" },
    { "id": "probs", "label": "Probabilities", "sub": "softmax" },
    { "id": "pick", "label": "Pick", "sub": "sample one" },
    { "id": "append", "label": "Append", "sub": "add to the text" }
  ],
  "steps": [
    { "title": "1. Start from the text so far", "body": "The model receives your prompt plus everything it has already written. It has no memory of anything else.", "active": ["prompt"], "example": "“Once upon a time in a land far, far”" },
    { "title": "2. Turn the text into tokens", "body": "The text is cut into tokens, and each token is replaced by a number the network can work with.", "active": ["prompt", "tokens"], "example": "“Once · upon · a · time · in · a · land · far · , · far”" },
    { "title": "3. Score every possible next token", "body": "The neural network gives every token in its vocabulary a raw score called a logit. A higher score means a better fit. These scores are not yet probabilities.", "active": ["tokens", "scores"], "example": "away 4.2, beyond 2.4, from 2.1, … pizza −1.5 (made-up scores)" },
    { "title": "4. Turn scores into probabilities", "body": "A function called softmax turns the scores into probabilities that are all positive and add up to 1. Temperature is applied here.", "active": ["scores", "probs"], "example": "away 73 percent, beyond 12 percent, from 9 percent, … (T = 1)" },
    { "title": "5. Pick one token", "body": "One token is drawn at random according to the probabilities, or the top one is taken. Top-k and top-p can remove unlikely tokens first.", "active": ["probs", "pick"], "example": "Drawn: “away”" },
    { "title": "6. Append and repeat", "body": "The chosen token is added to the end of the text, and the loop starts again with the longer text. It stops at an end marker or a length limit.", "active": ["pick", "append", "prompt"], "example": "“…in a land far, far away” → score the next token again" }
  ]
}
```

Notice that the loop only ever asks one question: *what comes next?* A whole essay is thousands of these answers in a row. In Lab Part 1 you will play the model yourself: guess the next word for five sentence starts and compare your guess with GPT-2's.

### The chain rule: the probability of a whole sentence

Earlier we asked how likely a whole sentence is. The loop gives the answer. The probability of a sentence is the probability of its first word, times the probability of the second word given the first, times the third given the first two, and so on. This is the [[chain rule of probability|The rule that the probability of a sequence equals the product of each item's probability given everything before it.]]. Select the parts of the formula.

```viz
{
  "type": "equation",
  "title": "The chain rule for a sentence",
  "intro": "Select each part. The big Π means “multiply all of these together”.",
  "formula": [
    { "t": "P(w₁ … wₙ)", "id": "whole" },
    " = ",
    { "t": "Π", "id": "pi" },
    " ",
    { "t": "P(wᵢ | w₁ … wᵢ₋₁)", "id": "step" }
  ],
  "parts": [
    { "id": "whole", "name": "P(w₁ … wₙ)", "meaning": "The probability of the whole sentence: n words, w₁ is the first and wₙ is the last." },
    { "id": "pi", "name": "Π (capital pi)", "meaning": "Means “multiply”, the way Σ means “add”. Multiply the terms for i = 1, 2, … up to n." },
    { "id": "step", "name": "P(wᵢ | w₁ … wᵢ₋₁)", "meaning": "The chance of the i-th word given all the words before it. This is exactly one turn of the writing loop." }
  ],
  "steps": [
    { "title": "Three words", "expr": "P(“The bus leaves”) = P(The) × P(bus | The) × P(leaves | The bus)", "note": "One factor per turn of the loop." },
    { "title": "With numbers", "expr": "0.2 × 0.1 × 0.05 = 0.001", "note": "Illustrative factors. The more words, the smaller the product gets, which is why long sentences have tiny probabilities." }
  ],
  "note": "The factors 0.2, 0.1 and 0.05 are illustrative numbers authored for teaching."
}
```

The next figure applies the rule to two six-token sentences. Select a sentence to see the running product after each token, the probability of the whole sentence and its perplexity.

```viz
{
  "type": "chain",
  "title": "Sensible versus scrambled",
  "sentences": [
    { "label": "Sensible", "tokens": [ { "t": "The", "p": 0.2 }, { "t": "bus", "p": 0.1 }, { "t": "leaves", "p": 0.05 }, { "t": "at", "p": 0.4 }, { "t": "six", "p": 0.3 }, { "t": ".", "p": 0.5 } ], "note": "Each word is fairly likely after the ones before it." },
    { "label": "Scrambled", "tokens": [ { "t": "Six", "p": 0.01 }, { "t": "at", "p": 0.02 }, { "t": "leaves", "p": 0.01 }, { "t": "bus", "p": 0.03 }, { "t": "the", "p": 0.02 }, { "t": ".", "p": 0.01 } ], "note": "The same words in the wrong order: every step is a surprise." }
  ],
  "note": "All probabilities are illustrative, authored for teaching. The sensible sentence multiplies to 0.00006 and the scrambled one to about 1.2 × 10⁻¹¹, roughly five million times smaller."
}
```

Two things to notice. First, the scrambled sentence is millions of times less likely than the sensible one, even though both use the same words. That is how a model “knows” grammar and sense: the order matters to the product. Second, ==the product of many small numbers becomes tiny==, so real systems add the logarithms instead of multiplying (that is what the loss −log p does). Lab Part 1 repeats this on real GPT-2 numbers.

### Perplexity: a model's average confusion

A product of 6 probabilities is hard to compare across sentences of different lengths. [[Perplexity|A measure of how surprised a language model is by a text. It is the inverse of the sentence probability, taken to the power 1/n for n tokens. Lower means a better fit.]] fixes this. It is the sentence probability turned upside down and averaged per token: perplexity = P^(−1/n). You can read it as “the model was as unsure as if it had to choose between this many equally likely words at every step”.

For the sensible sentence above, P = 0.00006 and n = 6, so the perplexity is 0.00006^(−1/6) ≈ 5.05: about as unsure as choosing between five equally likely words. For the scrambled one it is about 66.1. ==Lower perplexity means the text fits the model better.== A model with a perplexity of 1 would be sure of every word.

### Softmax with temperature: from scores to probabilities

How are the raw scores (logits) turned into probabilities? With the [[softmax|A function that turns a list of scores into probabilities: it makes every score positive by taking e to the power of it, then divides by the total so everything adds up to 1.]] formula, which also contains the setting called [[temperature|A setting that controls how adventurous the next-word pick is. Low values stay with the most likely words; high values give unlikely words a real chance.]]. Select each symbol.

```viz
{
  "type": "equation",
  "title": "Softmax with temperature",
  "intro": "Select each part. Then step through the worked example with three scores at two temperatures.",
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
    { "id": "p", "name": "pᵢ (probability of word i)", "meaning": "The chance that word number i is picked. All the pᵢ are positive and add up to 1." },
    { "id": "e", "name": "e (about 2.718)", "meaning": "A special number. Raising it to a power makes every value positive, and a bigger score gives a much bigger result." },
    { "id": "z", "name": "zᵢ (score or logit)", "meaning": "The raw score the network gave to word i. Higher means the word fits better. Scores can be negative." },
    { "id": "T", "name": "T (temperature)", "meaning": "Divides every score before the exponent. Small T (below 1) exaggerates the differences; large T (above 1) shrinks them. T = 1 leaves the scores unchanged." },
    { "id": "sigma", "name": "Σⱼ (sum over all words)", "meaning": "Add up the value for every word j in the vocabulary. Dividing by this total is what makes the probabilities sum to 1." },
    { "id": "e2", "name": "e (again)", "meaning": "The same exponential, now applied to every word's score to build the total." },
    { "id": "z2", "name": "zⱼ", "meaning": "The score of each word in the sum. The index j runs over all words, while i is the one word we are asking about." }
  ],
  "steps": [
    { "title": "Three scores", "expr": "away 4.2, beyond 2.4, from 2.1", "note": "Made-up logits from the sampling figure below. We use only three words so the arithmetic is easy." },
    { "title": "T = 1: exponentials", "expr": "e^4.2 = 66.69, e^2.4 = 11.02, e^2.1 = 8.17, total = 85.88", "note": "Compute e to each score, then add them." },
    { "title": "T = 1: probabilities", "expr": "away 66.69/85.88 = 0.777, beyond 0.128, from 0.095", "note": "The top word gets 77.7 percent." },
    { "title": "T = 0.5: exponentials", "expr": "e^8.4 = 4447.07, e^4.8 = 121.51, e^4.2 = 66.69, total = 4635.26", "note": "Dividing by 0.5 doubles every score first, so the gaps grow." },
    { "title": "T = 0.5: probabilities", "expr": "away 0.959, beyond 0.026, from 0.014", "note": "Lower temperature makes the favourite even more dominant: 95.9 percent instead of 77.7 percent." }
  ],
  "note": "The logits are made up to show the idea; they are not from a real model. The arithmetic is exact."
}
```

Look at the last two steps. The same scores gave 0.777 at T = 1 and 0.959 at T = 0.5. ==Low temperature makes the favourite stronger; high temperature flattens the choices.== Temperature does not change which word is the favourite; it changes how *sure* the model sounds.

### Try the dials yourself

The sampling figure uses six made-up scores for the next word after the prompt. It has three sliders: temperature, top-k and top-p. Words that top-k or top-p cut away turn grey, and the rest are shared out again so they add up to 1.

```viz
{
  "type": "sampling",
  "title": "Temperature, top-k and top-p",
  "prompt": "Once upon a time in a land far, far",
  "tokens": [
    { "t": "away", "logit": 4.2 },
    { "t": "beyond", "logit": 2.4 },
    { "t": "from", "logit": 2.1 },
    { "t": "off", "logit": 1.2 },
    { "t": "behind", "logit": 0.4 },
    { "t": "pizza", "logit": -1.5 }
  ],
  "note": "The scores are made up to show the idea; they are not taken from a real model. In a real model there are tens of thousands of candidate words."
}
```

Run this small experiment. First set the temperature to 0.3 and press **Draw 100 times**. Almost every draw is “away”: at T = 0.3 its probability is about 99.7 percent. Now set the temperature to 1.5 and draw again. The other words appear, and even “pizza” (about 1.2 percent) shows up now and then. Then set T back to 1 and lower top-k to 3: only “away”, “beyond” and “from” remain. Finally try top-p at 0.9 and watch which words survive.

The two filters work like this:

- **Top-k** keeps only the k most likely words and throws the rest away. With k = 3, the model can only choose among its three best guesses, however many words exist.
- **Top-p** (also called nucleus sampling) keeps the smallest group of top words whose probabilities add up to at least p. Think of filling a plate from the best dishes down until it is 90 percent full. At T = 1 the first three words add up to 0.73 + 0.12 + 0.09 = 0.94, so top-p = 0.9 keeps three words. At T = 1.5 you need four words to pass 0.9.

Top-k always keeps the same *number* of words. Top-p keeps a number that *adapts*: when the model is very sure, one word may be enough; when it is unsure, more words stay. Both are about stopping the model from picking absurd words such as “pizza”, at the price of less variety. In Lab Part 3 you will code these steps by hand and then watch GPT-2 change as you move the dials.

### Fluent is not the same as true

A language model is trained to predict *likely* words, not *true* ones. Usually the two overlap, because people mostly write true things. But the model has no built-in way to check a fact. This is why ==a model can be fluent and wrong at the same time==.

You can see it with a very small model. When the small GPT-2 is given “The capital of France is”, its top three guesses are “ the”, “ now” and “ a”, with measured probabilities of about 0.085, 0.048 and 0.046. “Paris” is not at the top. The sentence it writes will be grammatical, but it will not necessarily be right. Bigger and more carefully trained models do much better on this question, yet the same issue returns with rarer facts. You will measure it yourself in Lab Parts 1 and 6.

## Beyond text: pictures, sound and more {rail=Beyond text}

{lede} The idea of “learn a distribution, then sample” works for any kind of data, because ==everything can be written as numbers==: letters become token numbers, a picture becomes a grid of pixel values, and sound becomes a long list of wave heights. Once data is numbers, the same recipe applies.

### Pictures: remove noise a little at a time

Image generators such as Stable Diffusion use an idea called [[diffusion|A way to generate images: start from random noise and remove a little of it at each step, guided by what the model learned, until a picture appears.]]. Imagine a photo that you slowly cover with static, like a badly tuned television, until nothing is visible. A diffusion model is trained to *reverse* this: given a noisy picture, guess what the slightly cleaner version looks like. To make a new picture, you start from pure random noise and clean it up step after step. Because the starting noise is random, every run gives a different picture. The random seed in the lab controls it: ==the same seed gives the same picture==.

The forward direction, adding noise, has a neat formula. It says that at step t the noisy picture x_t is a blend of the original x₀ and fresh random noise ε. Select each part.

```viz
{
  "type": "equation",
  "title": "Adding noise: the forward diffusion step",
  "intro": "Select each part. The worked example uses the standard schedule from the original diffusion paper.",
  "formula": [
    { "t": "x_t", "id": "xt" },
    " = ",
    { "t": "√ᾱ_t", "id": "sa" },
    " · ",
    { "t": "x₀", "id": "x0" },
    " + ",
    { "t": "√(1−ᾱ_t)", "id": "sn" },
    " · ",
    { "t": "ε", "id": "eps" }
  ],
  "parts": [
    { "id": "xt", "name": "x_t (noisy picture)", "meaning": "The picture after t noising steps. At t = 0 it is the clean picture; at the final step it is nearly pure noise." },
    { "id": "sa", "name": "√ᾱ_t (signal strength)", "meaning": "How much of the original picture is kept. It starts at 1 and falls towards 0 as t grows." },
    { "id": "x0", "name": "x₀ (original picture)", "meaning": "The clean training picture, as a grid of numbers." },
    { "id": "sn", "name": "√(1−ᾱ_t) (noise strength)", "meaning": "How much noise is mixed in. It starts at 0 and grows towards 1." },
    { "id": "eps", "name": "ε (noise)", "meaning": "Random numbers drawn from a bell curve (a standard normal distribution), one per pixel." }
  ],
  "steps": [
    { "title": "The schedule", "expr": "β rises linearly from 0.0001 to 0.02 over 1000 steps; ᾱ_t = (1−β₁)(1−β₂)…(1−β_t)", "note": "The numbers β say how much noise each step adds. ᾱ_t is the running product of what is left." },
    { "title": "At t = 300", "expr": "ᾱ_300 = 0.3964, √ᾱ = 0.630, √(1−ᾱ) = 0.777", "note": "At step 300 the signal weight 0.630 is already smaller than the noise weight 0.777." },
    { "title": "One pixel", "expr": "x₀ = 1, ε = 0.5  →  x_300 = 0.630 × 1 + 0.777 × 0.5 = 1.018", "note": "A black pixel (1) plus a little random noise. With a different ε the result changes, which is why every noisy copy is different." }
  ],
  "note": "The schedule (β from 0.0001 to 0.02 over 1000 steps) is the one used in the original DDPM paper; the pixel value and the noise draw ε = 0.5 are illustrative."
}
```

Now drag the slider in the figure below. At t = 0 you see a clean smiley. As t grows the face disappears into noise, which is the *forward* direction the model is trained on. The idealised reverse shows what a trained model would do: clean the picture step by step. In Lab Part 4 you will noise a smiley yourself and check ᾱ at t = 300.

```viz
{
  "type": "diffusion",
  "title": "A smiley dissolving into noise",
  "image": [
    "000000000000",
    "000099990000",
    "009900009900",
    "090000000090",
    "090990099090",
    "090990099090",
    "090000000090",
    "090900009090",
    "090099990090",
    "009900009900",
    "000099990000",
    "000000000000"
  ],
  "t": 300
}
```

Notice that even at t = 300, where the signal weight is still 0.63, you can half-see the face. A real model starts at the far end (almost pure noise) and has to rebuild the picture, guided by your text prompt. Stable Diffusion does this on a compressed version of the image, and the text is used to steer each step.

### Sound, translation and code

The same idea reaches other media. In [[text to speech|Software that reads written text aloud by generating a new audio waveform for it.]] a model takes text and produces a waveform: new sound. In speech to text, as Whisper does, audio goes in and a transcript comes out. In translation, a model writes a sentence in another language, token by token, exactly as in the writing loop but conditioned on the source sentence. Code assistants write programs the same way, because code is just text with strict patterns. Some systems are [[multimodal|Able to work with more than one kind of data, such as text, images and audio, in the same model.]]: you can show them a photo and ask a question in words.

Test your understanding of what goes in and what comes out. For each model, decide which kind of data it *produces*. Whisper is the tricky one.

```viz
{
  "type": "sorter",
  "title": "What does the model produce?",
  "intro": "Sort each model by the kind of content it outputs: text, image or audio.",
  "bins": [
    { "id": "text", "label": "Text" },
    { "id": "image", "label": "Image" },
    { "id": "audio", "label": "Audio" }
  ],
  "cards": [
    { "name": "GPT-2", "answer": "text", "why": "It continues any text you give it, one token at a time." },
    { "name": "ChatGPT", "answer": "text", "why": "It writes conversational replies as text." },
    { "name": "Stable Diffusion", "answer": "image", "why": "It turns random noise into a picture that matches your text prompt." },
    { "name": "DALL·E", "answer": "image", "why": "A family of OpenAI models that create images from written descriptions." },
    { "name": "MarianMT translation", "answer": "text", "why": "It reads a sentence in one language and writes a sentence in another. The output is text." },
    { "name": "Whisper (speech to text)", "answer": "text", "why": "Audio goes in, but what comes out is a written transcript, so its output is text. The input is audio; the output decides the bin." },
    { "name": "MMS-TTS", "answer": "audio", "why": "A text-to-speech model: text goes in and a new audio waveform comes out." },
    { "name": "A code assistant", "answer": "text", "why": "Program code is text, so it is written token by token like any other text." },
    { "name": "A model that turns a melody description into music", "answer": "audio", "why": "The output is a new sound waveform, so it belongs with audio." }
  ]
}
```

### One idea, many media

Step back from the details. Text is a sequence of token numbers. An image is a grid of pixel numbers. Sound is a long sequence of wave heights. In every case the model learns which patterns of numbers are likely, and then samples new ones. The names differ (a language model, a diffusion model, a text-to-speech model) but the plan is the same as for the canteen chef: ==learn a distribution from examples, then sample from it==. This is why one course can cover so many kinds of generative AI. You will try images in Lab Part 4 and sound and translation in Lab Part 5.

## A short history, and why now {rail=History}

{lede} Today's tools feel sudden, but they sit on ==about seventy years of work==, with long quiet stretches in between. The pattern is steady: better data, more computing power, and better ideas arriving together.

Select each event to see why it mattered. Two terms to know: the [[Perceptron|An early artificial neuron (1958) that learns to classify inputs by adjusting its weights from examples. It is an ancestor of today's neural networks.]] and the [[AI winters|Periods when interest and funding for AI dropped because early promises were bigger than what the technology could deliver.]].

```viz
{
  "type": "timeline",
  "title": "From a summer workshop to ChatGPT",
  "events": [
    { "year": "1956", "title": "Dartmouth workshop", "text": "A summer workshop at Dartmouth College brought researchers together around the idea that machines could be made to learn and reason. The term “artificial intelligence” comes from this event, and it is often treated as the birth of the field." },
    { "year": "1958", "title": "The Perceptron", "text": "Frank Rosenblatt built the Perceptron, an early artificial neuron that learns to classify inputs by adjusting its weights from examples. It is the ancestor of today's neural networks. It was discriminative: it sorted inputs, it did not create anything." },
    { "year": "1966", "title": "ELIZA", "text": "Joseph Weizenbaum's ELIZA was a chatbot that seemed to hold a conversation. It worked by matching keywords and following hand-written rules, not by learning from data, yet many people found it convincing. It is a reminder that fluent-looking talk does not mean understanding." },
    { "year": "1970s–90s", "title": "AI winters", "text": "Early promises were bigger than what computers and data could deliver. Funding and interest dropped in more than one period, which people call AI winters. The lesson: progress depends on data, computing power and realistic goals." },
    { "year": "1986", "title": "Backpropagation popularised", "text": "Rumelhart, Hinton and Williams published a clear account of backpropagation, the method that lets a multi-layer neural network learn by passing error signals backwards. You will study it in Week 2." },
    { "year": "2012", "title": "AlexNet and deep learning", "text": "A deep convolutional network called AlexNet won an image-recognition contest by a wide margin, trained on GPUs. This showed that more data plus more GPU computing could beat hand-built methods, and deep learning became the main approach." },
    { "year": "2014", "title": "GANs", "text": "Goodfellow and colleagues introduced generative adversarial networks, in which one network makes fake samples and another tries to spot them. GANs produced the first strikingly realistic generated images." },
    { "year": "2017", "title": "The Transformer", "text": "Vaswani and colleagues introduced the Transformer, built on attention, which made it practical to train very large language models. You will study it in Week 7." },
    { "year": "2019", "title": "GPT-2", "text": "GPT-2 showed that a language model trained only to predict the next word could write surprisingly fluent text. You run its small version in this week's lab." },
    { "year": "2020", "title": "GPT-3", "text": "GPT-3 was far larger than GPT-2, and showed that scale alone brought new abilities, such as following a few examples written in the prompt." },
    { "year": "2021", "title": "DALL·E", "text": "OpenAI's DALL·E generated images from text descriptions, bringing text-to-image generation to wide public attention." },
    { "year": "2022", "title": "Stable Diffusion and ChatGPT", "text": "Stable Diffusion made high-quality image generation open to everyone, and ChatGPT put a conversational model in front of millions of people. Generative AI moved from research labs into daily life." }
  ]
}
```

Read them in order and notice the shape. Each jump needed something new: a learning rule (1986), computing power and data (2012), a better architecture (2017), then scale (2019–2020). The 1966 entry is a useful warning: ELIZA used rules, not learning, so a system can look clever without being generative in the sense of this course.

### Three ingredients, and a caution about size

Why did this happen in the 2010s and 2020s and not in the 1990s? Three ingredients came together. **Data:** the internet created text and images in enormous quantity. **Compute:** GPUs, originally built for video games, turned out to be excellent at the matrix arithmetic neural networks need. **Algorithms:** backpropagation, convolutional networks, attention and the Transformer made the best use of both. Remove any one and progress slows.

One result is that models have grown fast. The number of [[parameters|The adjustable numbers inside a model that are learned during training. More parameters usually means more capacity to learn patterns, and more memory to run.]] is a common measure of size. The figure uses a *log scale*, where each step to the right means ten times bigger, because the numbers would otherwise be impossible to draw.

```viz
{
  "type": "logbars",
  "title": "How big are language models?",
  "unit": "parameters",
  "items": [
    { "label": "GPT-1", "year": "2018", "value": 117000000, "note": "About 117 million parameters." },
    { "label": "BERT-large", "year": "2018", "value": 340000000, "note": "About 340 million parameters." },
    { "label": "GPT-2", "year": "2019", "value": 1500000000, "note": "About 1.5 billion parameters, the largest GPT-2 version. The small one you run in the lab has far fewer." },
    { "label": "GPT-3", "year": "2020", "value": 175000000000, "note": "About 175 billion parameters." },
    { "label": "Chinchilla", "year": "2022", "value": 70000000000, "note": "About 70 billion parameters, trained on much more text than earlier models of its size." },
    { "label": "PaLM", "year": "2022", "value": 540000000000, "note": "About 540 billion parameters." }
  ],
  "note": "Sizes are the approximate values reported in the original papers. The scale is logarithmic: each gridline is ten times the one before."
}
```

Notice that GPT-3 has roughly 117 times as many parameters as the largest GPT-2 (175 billion divided by 1.5 billion). ==But bigger is not automatically better.== Chinchilla has fewer parameters than GPT-3 or PaLM, and its authors showed that a model of that size trained on more text can do better than larger ones trained on less. Bigger models also cost more to train and run, and need more memory. Good data and good training matter as much as size.

## Using it well, and what can go wrong {rail=Risks}

{lede} Generative AI is a tool, and tools give better results when you use them with care. ==How you ask matters, and so does what you trust==: a clear prompt gets a more useful answer, and a fluent answer still needs checking.

### Prompts: how to ask well

A [[prompt|The text, and sometimes images, that you give a generative model to tell it what you want. The model continues or answers it.]] is the input you give a model. Because a language model continues your text, the words you choose change what comes out. A good prompt has up to five parts:

1. **Role:** who the model should act as (“You are a patient science tutor for Grade 10”).
2. **Task:** what to do, in a clear verb (“explain”, “list”, “translate”).
3. **Context:** the facts it needs (“The class has just learned about photosynthesis”).
4. **Format:** the shape of the answer (“Use three bullet points, under 60 words”).
5. **Example:** a sample of what you want, when a pattern is easier to show than to describe.

Compare a vague prompt with a specific one below. In each pair the specific prompt gives the model something concrete to follow.

```viz
{
  "type": "flow",
  "title": "Vague prompt or specific prompt?",
  "modes": [
    {
      "name": "Vague prompt",
      "tagline": "Few details, so the model has to guess what you wanted. The result is generic or off target.",
      "inputLabel": "Prompt",
      "modelLabel": "a language model guesses",
      "outputLabel": "Likely result",
      "examples": [
        { "label": "Explain", "input": "Tell me about photosynthesis.", "output": "A long, generic textbook-style answer of unknown length and level." },
        { "label": "Announcement", "input": "Write an announcement.", "output": "A bland announcement about nothing in particular." },
        { "label": "Translate", "input": "Translate this.", "output": "The model asks what to translate, or translates the wrong text." }
      ],
      "note": "The outputs are made-up descriptions to show the idea; real output varies."
    },
    {
      "name": "Specific prompt",
      "tagline": "Role, task, context, format and sometimes an example. The model has much less to guess.",
      "inputLabel": "Prompt",
      "modelLabel": "a language model follows the details",
      "outputLabel": "Likely result",
      "examples": [
        { "label": "Explain", "input": "You are a science tutor. Explain photosynthesis to a Grade 7 student in three short bullet points, using a vegetable garden as the example.", "output": "Three short bullets about plants using sunlight, water and air to make food, with garden examples." },
        { "label": "Announcement", "input": "Write a four-sentence barangay announcement about the clean-up drive this Saturday at 7 a.m. at the covered court. Friendly tone.", "output": "A short, friendly four-sentence notice with the time and place." },
        { "label": "Translate", "input": "Translate to Tagalog, keeping it polite: “Please line up at the canteen window.”", "output": "A polite Tagalog sentence for the canteen line." }
      ],
      "note": "The outputs are made-up descriptions to show the idea; real output varies, and you should always check a translation with a fluent speaker."
    }
  ]
}
```

A specific prompt is not a magic spell. The model can still be wrong. But it gives the model a better target, and it makes your own checking easier because you know what you asked for. The course has a [prompt workshop](/prompt-workshop) where you can practise this more.

### What it is used for, and what to watch

Generative AI is useful across many areas, and the same few risks come with all of it.

| Area | What generative AI can do | What to watch for |
|---|---|---|
| Text | Draft, summarise, explain, write code | Fluent but wrong answers; made-up facts |
| Translation | Translate between many languages | Lost nuance; idioms and word order change |
| Images | Create pictures from a description | Distorted hands and text; fake photos (deepfakes) |
| Audio | Read text aloud; turn speech into text | Cloned voices; transcription errors |

Whatever the area, these risks keep coming back. Read them as things to test and reduce, not as reasons to avoid the tools.

- **Bias:** a model learns from human data, so it can ==repeat unfair patterns in that data==. If most examples show nurses as women, the model may do the same.
- **Misinformation and deepfakes:** convincing fake text, images, audio and video are cheap to make, and a fake voice message can fool someone you know.
- **Privacy:** anything you type into a public tool may be stored or used. Do not paste private data about yourself or classmates.
- **Overtrust and hallucination:** a model can state a made-up fact with complete confidence, which is called a [[hallucination|When a generative model produces fluent text that is false or invented, such as a book, quote or statistic that does not exist.]]. ==Fluent does not mean true==, as the “capital of France” example showed.
- **Intellectual property and consent:** models are trained on large collections of text and images made by people. Whether and how that is allowed, and whether creators should be asked or paid, is being debated and decided differently in different places. Be careful about presenting generated work as your own, and about copying a living artist's style or a person's likeness or voice without their consent.
- **Environmental cost:** training and running large models uses a lot of electricity, and data centres also need cooling. The amounts depend on the model and the data centre and are not always published, so be wary of any single figure; the safe point is that bigger models cost more to build and run.

### Five principles of trustworthy AI

[[Trustworthy AI|The idea that AI should be built and used in line with five principles: fairness, transparency, privacy, safety and accountability.]] means building and using AI in line with five principles. They give you five questions to ask of any AI tool:

1. **Fairness:** does it treat people and groups justly? *Example:* a hiring tool should not score applicants lower because of their gender.
2. **Transparency:** can people tell when AI is used, and how it decides? *Example:* a chatbot should say it is a machine, and a school should say when AI helped to mark work.
3. **Privacy:** is personal data protected and used only with permission? *Example:* a study app should not keep photos of students without consent.
4. **Safety:** can it cause harm, and are there safeguards? *Example:* a health or plant-guide app must not give dangerous advice with confidence.
5. **Accountability:** when something goes wrong, is a person or organisation responsible, and can the decision be challenged? *Example:* a student whose work was wrongly flagged by a tool can appeal to a named person.

Now apply them. Each card is a short scenario. Choose the principle it is *mainly* about, and read the explanation either way.

```viz
{
  "type": "sorter",
  "title": "Which principle is at stake?",
  "intro": "Sort each scenario by the trustworthy-AI principle it mainly breaks.",
  "bins": [
    { "id": "fairness", "label": "Fairness" },
    { "id": "transparency", "label": "Transparency" },
    { "id": "privacy", "label": "Privacy" },
    { "id": "safety", "label": "Safety" },
    { "id": "accountability", "label": "Accountability" }
  ],
  "cards": [
    { "name": "A hiring tool scores women lower for the same experience", "answer": "fairness", "why": "Two equal people are treated differently because of gender. That is an unfair outcome." },
    { "name": "A loan app refuses more applicants from one barangay who have the same income as others", "answer": "fairness", "why": "Where someone lives, rather than their finances, drives the decision, which treats a group unjustly." },
    { "name": "A chatbot on a shop page pretends to be a human employee", "answer": "transparency", "why": "People have a right to know they are talking to a machine." },
    { "name": "A school marks essays with AI but never tells the students", "answer": "transparency", "why": "Students cannot question a process they do not know exists. The use of AI should be disclosed." },
    { "name": "A study app stores photos of students' faces without asking them", "answer": "privacy", "why": "Faces are personal data. Collecting them without consent breaks privacy." },
    { "name": "A student pastes classmates' phone numbers into a public chatbot", "answer": "privacy", "why": "Personal details about other people may be stored or reused by the tool." },
    { "name": "A plant-guide app confidently says a poisonous mushroom is safe to eat", "answer": "safety", "why": "A wrong answer can cause real physical harm, so there must be safeguards and warnings." },
    { "name": "A health chatbot tells a user to double a medicine dose", "answer": "safety", "why": "Dangerous advice given with confidence is a safety failure." },
    { "name": "An AI grader wrongly fails students, and the company says “the algorithm did it” with no way to appeal", "answer": "accountability", "why": "Nobody takes responsibility and nobody can challenge the result. Accountability needs a responsible person and a way to appeal." },
    { "name": "A deepfake voice message is used to scam a family, and nobody can say who is responsible for the tool", "answer": "accountability", "why": "When harm happens there must be an owner who answers for the system and can be asked to fix it." }
  ]
}
```

If you disagreed with a card, that is fine: real cases touch several principles at once. The skill is to name the one that is most at stake and to explain why. You will do the same with your own AI product in Lab Part 6.

### Spotting AI-made content

Can you tell AI-made content from human-made content by looking? Often not. Clues such as “odd hands” or a generic tone fade as models improve, and people also wrongly accuse real work. AI-text detectors are unreliable too, and can wrongly flag honest students. This is why ==disclosure== (saying when AI was used) and ==provenance== (knowing where content came from) matter more than guessing from clues. When in doubt, check the source, look for a second report, and ask who made it and why.

## Key takeaways

- **Generative models create; discriminative models label.** A generative model ==learns what data looks like and produces new data==; a discriminative model only labels or scores existing data. Generative AI is the smallest doll inside deep learning, machine learning and AI.
- ==Learn a distribution, then sample.== Training makes real data more probable (it lowers the loss −log p), and generation draws new items from the learned probabilities. Even a counting model does this.
- A language model writes ==one token at a time==. The chain rule multiplies the next-token probabilities into a sentence probability, and perplexity (lower is better) summarises how well a text fits the model.
- **Temperature, top-k and top-p** control the pick: low temperature plays safe, high temperature is adventurous, and top-k or top-p cut the unlikely words. ==Fluent does not mean true==, so check facts.
- The same recipe covers images (diffusion), sound and translation. Judge every tool with the ==five trustworthy-AI principles==: fairness, transparency, privacy, safety and accountability.

## Lab: Imagine it, then build it {rail=Laboratory}

This lab is called *Imagine it, then build it*: you first imagine what a generative AI can create, and then ==find out how it really works by playing the model yourself, building a tiny text generator from scratch, and turning the dials (temperature, top-k, top-p) that decide whether it is safe and dull or wild and creative==. It has seven parts, and every part begins with an **Imagine first** moment: you write a prediction before you run any code and compare it with what really happens. Parts 1 to 3 are in this section; they use GPT-2, a small language model that runs on a CPU, and they come back to the lesson's chain-rule, n-gram and sampling figures. Time: about 3 hours.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cell, check the cached models | 10 |
| 1 | Be the model: guess the next word | 25 |
| 2 | Build a tiny generative model from scratch | 25 |
| 3 | The dials of imagination: temperature, top-k, top-p | 30 |
| 4 | Imagine in pictures: noise, diffusion, Stable Diffusion | 30 |
| 5 | Imagine in sound and across languages | 25 |
| 6 | Imagine an AI product, then try to break it | 30 |
| 7 | Gather it up and reflect | 10 |

The total is 185 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week01` and a notebook `lab1.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`tok`, `lm`, `next_token_probs`, `RESULTS` and so on) that earlier blocks create.
2. Do not install packages (everything is pre-installed and there is no internet). Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook.
3. Models load from the offline Hugging Face cache (`HF_HOME` is set). Parts 1 to 3 need only `gpt2` (GPT-2 small, 124 million parameters). The whole lab uses these IDs: `gpt2`, `Helsinki-NLP/opus-mt-en-fr`, `Helsinki-NLP/opus-mt-fr-en`, `facebook/mms-tts-eng`, `openai/whisper-tiny`, and the Stable Diffusion v1.5 model your instructor cached; its ID is `runwayml/stable-diffusion-v1-5` or `stable-diffusion-v1-5/stable-diffusion-v1-5`. Confirm with your instructor that every model ID above is in the offline cache. The first code cell below lists which of them it can see.
4. The GPU is shared by the whole class. GPT-2 small is tiny and also runs fine on the CPU, but call `free_gpu()` after each part and never leave a large model loaded.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste this into the first cell and run it. It lists the cached models, sets the seed, loads GPT-2 and defines three helpers that the rest of the lab relies on:

```python
from huggingface_hub import scan_cache_dir

cached = {r.repo_id for r in scan_cache_dir().repos}          # what the offline cache holds
wanted = ["gpt2", "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en", "facebook/mms-tts-eng", "openai/whisper-tiny",
          "runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5"]
for model_id in wanted:
    print(("cached   " if model_id in cached else "not seen ") + model_id)
```

```python
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # the Spark has no internet: use only the offline cache
import gc, math, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
QUICK = False                                         # set True to shorten the long loops while you debug
print("Running on:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

RESULTS = {}                                          # keep your key numbers here; Part 7 prints them together

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained("gpt2")           # GPT-2 small, 124 million parameters
lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
tok.pad_token = tok.eos_token

@torch.no_grad()
def next_token_probs(prompt, top=10):
    """The model's top guesses for the next token: a list of (token, probability)."""
    ids = tok(prompt, return_tensors="pt").to(device)
    probs = torch.softmax(lm(**ids).logits[0, -1].float(), dim=-1)
    p, i = probs.topk(top)
    return [(tok.decode(int(t)), float(q)) for t, q in zip(i, p)]

@torch.no_grad()
def token_logprobs(text):
    """For every token of `text`: (token, natural-log probability given the tokens before it)."""
    ids = tok(tok.bos_token + text, return_tensors="pt").input_ids.to(device)   # end-of-text marker in front
    logp = torch.log_softmax(lm(ids).logits[0].float(), dim=-1)
    return [(tok.decode(int(ids[0, pos])), float(logp[pos - 1, int(ids[0, pos])])) for pos in range(1, ids.shape[1])]

def generate(prompt, max_new_tokens=40, **settings):
    """Let GPT-2 continue `prompt`; settings such as do_sample=True, temperature=0.8, top_k=40, top_p=0.9."""
    ids = tok(prompt, return_tensors="pt").to(device)
    out = lm.generate(**ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **settings)
    return tok.decode(out[0], skip_special_tokens=True)

print(next_token_probs("The capital of France is", top=3))
```

   - `next_token_probs(prompt, top=10)` returns the model's top guesses for the next token as `(token, probability)` pairs;
   - `token_logprobs(text)` returns, for every token, the natural-log probability the model gave it given the tokens before it;
   - `generate(prompt, max_new_tokens=40, **settings)` lets GPT-2 continue a prompt; settings such as `do_sample=True, temperature=0.8, top_k=40, top_p=0.9` control the sampling.

   You should see `Running on: cuda NVIDIA GB10` (or `cpu`), and the last line `[(' the', 0.0845...), (' now', 0.0479...), (' a', 0.0461...)]`: small GPT-2 puts " the", " now" and " a" ahead of " Paris" after "The capital of France is". Keep that in mind: it is the first lesson of the lab.

### Step-by-step tasks

1. *Part 1. Be the model: guess the next word (25 min).* A language model has one job: give a probability to every possible next token. In this part you do that job yourself first, then ask GPT-2 and compare. It belongs to the lesson section "How a language model writes" (next-word prediction and the chain-rule figure).

   **Imagine first.** Before you look at any output, predict how often you and the model will agree: out of the five sentence starts below, on how many will GPT-2's single most likely next word be the same as your first guess? Write a whole number from 0 to 5 and keep it in the variable `my_prediction`. In step 1.1 you then write your own guesses.

   **Step 1.1. Predict by hand.** For each of the five starts write three guesses for the next word (most likely first) and your confidence from 0 to 100 % that your first guess is right. Be honest: if you have no idea, write 10, not 90. These are example guesses, replace them with your own before you run the next blocks. Write words without the space in front (for example `Paris`, `4`).

```python
# Imagine first: how many of the 5 starts below will GPT-2's most likely next word match YOUR first guess? (0 to 5)
my_prediction = 2        # replace the 2 with your own number before you run the rest

prompts = [
    "The capital of France is",
    "Once upon a time there was a",
    "To make tea, first boil the",
    "The doctor said that she",
    "2 + 2 =",
]

# Your three guesses (most likely first) and how sure you are of the first one (0 to 100 %).
# These are example guesses: write your own.
my_guesses = {
    prompts[0]: {"top3": ["Paris", "the", "located"], "confidence": 90},
    prompts[1]: {"top3": ["king", "girl", "little"], "confidence": 30},
    prompts[2]: {"top3": ["water", "kettle", "pot"], "confidence": 70},
    prompts[3]: {"top3": ["was", "had", "would"], "confidence": 25},
    prompts[4]: {"top3": ["4", "four", "5"], "confidence": 95},
}
```

   *Exercise 1.1:* fill in `my_prediction` and your own `my_guesses` before looking at any model output. Then write two sentences in a markdown cell: for which start did you feel most sure, for which least sure, and why? Why it matters: this is exactly what the model does, with 50,257 candidate tokens instead of three. Your feelings of "sure" and "lost" have a counterpart in the model's probabilities, and you are about to see them.

   **Step 1.2. Ask the model.** Print the model's top three tokens for every start, then draw a bar chart of its top 10 for one start.

```python
model_top = {p: next_token_probs(p, top=10) for p in prompts}      # the model's top 10 for every start

print(f"{'start':<30}{'#1':<10}{'p':>6}   {'#2':<10}{'p':>6}   {'#3':<10}{'p':>6}")
for p in prompts:
    (t1, p1), (t2, p2), (t3, p3) = model_top[p][:3]
    print(f"{p:<30}{t1!r:<10}{p1:>6.3f}   {t2!r:<10}{p2:>6.3f}   {t3!r:<10}{p3:>6.3f}")
```

```python
chart_prompt = prompts[0]            # change the 0 to 1, 2, 3 or 4 to look at another start
words, probs = zip(*model_top[chart_prompt])

plt.figure(figsize=(7, 3.2))
plt.barh([repr(w) for w in words][::-1], probs[::-1], color="#4c78a8")
plt.xlabel("probability of being the next token")
plt.title(f'GPT-2 after "{chart_prompt}"')
plt.tight_layout()
plt.show()
print(f"The top 10 tokens together hold {sum(probs):.0%} of the probability.")
```

   You should see, for "The capital of France is", the tokens `' the'` (0.085), `' now'` (0.048) and `' a'` (0.046) at the top: `' Paris'` is only fifth, with 0.032. After "To make tea, first boil the" the model is much surer: `' water'` has 0.323. After "Once upon a time there was a" it is very unsure (`' great'` 0.040, `' man'` 0.039), and after "2 + 2 =" it spreads its bets over `' 3'` (0.102), `' 1'` (0.092) and `' 2'` (0.084). For the France start the top 10 together hold only 36 % of the probability, so the bar chart is low and flat. Notice the leading space in tokens such as `' the'`: a space belongs to the word that follows it.

   Next, measure how surprised the model was by your first guess. The **surprise** of an outcome with probability p is `−log₂(p)` bits: p = 0.5 is 1 bit, p = 0.25 is 2 bits, p = 0.032 is about 5 bits. The helper asks for all 50,257 token probabilities and picks out one word (with a space in front). If your word is not a single token, it returns 0.0, and "very surprised" is then the honest description: the model considered it practically impossible.

```python
def word_probability(prompt, word):
    """The probability GPT-2 gives to `word` as the very next token (0.0 if it is not a single token)."""
    everything = dict(next_token_probs(prompt, top=tok.vocab_size))   # all 50,257 tokens
    return everything.get(" " + word, everything.get(word, 0.0))     # words after a space start with a space

surprise_bits = {}
for p in prompts:
    first = my_guesses[p]["top3"][0]
    prob = word_probability(p, first)
    surprise_bits[p] = -math.log2(max(prob, 1e-12))      # max(...) avoids log(0) for a word the model never gives
    print(f"{p:<30} you said {first!r:<8} model p = {prob:.4f}   surprise = {surprise_bits[p]:5.2f} bits")

# Imagine first, revisited: did the model's #1 match your #1?
actual = sum(model_top[p][0][0].strip().lower() == my_guesses[p]["top3"][0].strip().lower() for p in prompts)
RESULTS["part1_actual_matches"] = actual
RESULTS["part1_predicted_matches"] = my_prediction
if my_prediction is None:
    print("You did not write a prediction at the top of Part 1; the real number of matches is", actual)
else:
    print(f"You predicted {my_prediction} matches out of 5. The real number is {actual}.")
```

   With the example guesses you should see: Paris 4.95 bits (p = 0.0322), king 8.32 bits (p = 0.0031), water 1.63 bits, was 2.51 bits and 4 4.18 bits, an average of 4.32 bits, and the model's #1 matched your #1 once (water). Your numbers will differ.

   *Exercise 1.2:* for each of the five starts write down (a) the model's #1 token and its probability, (b) the surprise in bits of your own first guess, and (c) whether your first guess was in the model's top 10. If it was not, explain in one sentence what "very surprised" means (the model gave it a very small probability, so seeing it is a big surprise). Then compare your `my_prediction` with the real number of matches. What does a low surprise tell you about you and the model? What does a high surprise, together with a high confidence on your side, tell you? Write 3 to 4 sentences.

   *Exercise 1.3:* observe and write. In a markdown cell answer in at least two sentences: where was the model **sure**, where **unsure**, and where **wrong or unhelpful**? Then finish with one sentence that explains why fluent is not the same as true. What you should see: sure after the tea start (water, 32 %), unsure after the story start (the best token has only 4 %), unhelpful for France (Paris is fifth with 3.2 %). ==The probabilities say how likely a word is to follow, not whether a statement is true.==

   **Step 1.4. The chain rule on real tokens.** A model scores a whole sentence by multiplying the probability of each token given the tokens before it (the chain-rule figure in the lesson). Because the products are tiny, we add natural logarithms instead. The block scores a sensible sentence and its scrambled version (the same words, shuffled), prints every token's probability, the total ln-probability, the sentence probability in scientific notation and the perplexity `exp(−mean ln p)`. Perplexity is "how many equally likely choices the model was torn between at each step, on average"; lower is better.

```python
sensible = "The cat sat on the mat."
scrambled = "Mat the on sat cat the."          # the same words, shuffled

def show_tokens(sentence):
    """Print every token with its log-probability and probability; return the list of log-probabilities."""
    rows = token_logprobs(sentence)
    for token, lp in rows:
        print(f"  {token!r:>7}   ln p = {lp:7.2f}   p = {math.exp(lp):.5f}")
    total = sum(lp for _, lp in rows)
    print(f"  total ln p = {total:.2f} over {len(rows)} tokens\n")
    return [lp for _, lp in rows]

print(sensible)
lp_good = show_tokens(sensible)
print(scrambled)
lp_bad = show_tokens(scrambled)

prob_good = math.exp(sum(lp_good))
prob_bad = math.exp(sum(lp_bad))
ppl_good = math.exp(-sum(lp_good) / len(lp_good))
ppl_bad = math.exp(-sum(lp_bad) / len(lp_bad))

print(f"sensible : probability {prob_good:.2e}, perplexity {ppl_good:.1f}")
print(f"scrambled: probability {prob_bad:.2e}, perplexity {ppl_bad:.1f}")
```

   You should see, for "The cat sat on the mat.", the seven tokens with ln p of −3.28, −9.22, −6.84, −1.49, −0.87, −5.76 and −2.89 (total −30.36), a sentence probability of 6.53e-14 and a perplexity of 76.5. For "Mat the on sat cat the." the total is −59.34, the probability 1.69e-26 and the perplexity 4805.2: about four trillion times less likely. Notice that even the sensible sentence is surprising at ` cat` and ` sat`, and that ` the` after ` on` is easy (p = 0.42).

   *Exercise 1.4:* write down the two sentence probabilities and the two perplexities. Then answer in 3 to 4 sentences: which token did the model find most surprising in each sentence? How does the product of the seven probabilities relate to the chain-rule figure in the lesson? Why are logarithms used (the product of many small numbers becomes a sum, and avoids numbers too small for a computer)? What does a perplexity of 76.5 mean in words? Add `assert sum(lp_bad) < sum(lp_good)` below the block to confirm that the scrambled sentence is less likely.

   **Step 1.5. Stretch: does the model prefer the truth?** Write your own pair of sentences that differ in one word, one true and one false, with the same number of words (fair comparison needs similar lengths). The example is below; replace it with your own pairs and try at least three.

```python
def compare_pair(true_s, false_s):
    lp_t = sum(lp for _, lp in token_logprobs(true_s))
    lp_f = sum(lp for _, lp in token_logprobs(false_s))
    print(f"true : {true_s!r:<45} total ln p = {lp_t:.2f}")
    print(f"false: {false_s!r:<45} total ln p = {lp_f:.2f}")
    print("The model prefers the", "TRUE" if lp_t > lp_f else "FALSE", "sentence.")
    return lp_t, lp_f

true_sentence = "The sun rises in the east."      # replace with your own pair
false_sentence = "The sun rises in the west."
pair_result = compare_pair(true_sentence, false_sentence)
```

   You should see `true : 'The sun rises in the east.'  total ln p = -24.78` and `false: 'The sun rises in the west.'  total ln p = -24.59`, so the model slightly prefers the false sentence.

   *Exercise 1.5:* record your pairs and the model's choice for each. Out of your pairs, how often did it prefer the true sentence? Write two sentences on what this says about the difference between learning how people write and knowing what is true.

2. *Part 2. Build a tiny generative model from scratch (25 min).* The lesson's big idea is that a generative model learns a distribution from data and then samples from it (the section "How a machine learns" and the n-gram figure). You build the smallest possible version in pure Python: count which word follows which, then draw the next word from those counts. It does the same job as GPT-2 (predict the next token) without any neural network.

   **Imagine first.** A model that only counts which word follows which will write sentences. Do you think they will make sense? Write `"mostly sense"`, `"mixed"` or `"mostly nonsense"` in `my_guess`, and compare it with what you read in step 2.2.

```python
my_guess = "mixed"       # "mostly sense", "mixed" or "mostly nonsense": replace with your own prediction
```

   **Step 2.1. Read the corpus and train a bigram model.** The corpus is the text the model learns from: about 280 words on a school robotics club. `train_ngram(text, n)` returns a dictionary from a context (the n − 1 words before) to a `Counter` of the tokens that came next; full stops count as tokens, so the model also learns where sentences end.

```python
import re
from collections import Counter, defaultdict

corpus = """The robotics club meets every Thursday after school in room twelve. Our teacher, Ms Reyes, brings a box of motors, sensors and wires. We build a small robot that follows a black line on the floor. The robot has two wheels and a light sensor under its body. When the sensor sees the line, the robot drives forward. When the sensor sees the floor, the robot turns left to find the line again. Last week the robot lost the line and drove into a chair. Everyone laughed, and then we fixed the code. Jon thinks the robot needs a faster motor. Amara thinks the robot needs better code. I think the robot needs a battery that does not run out in the middle of a test. Next month our club will join a contest with teams from other schools. Each team must build a robot that carries a small cup across a table without spilling it. We practise with water, and the floor is often wet. Ms Reyes says that a good team shares its ideas, tests its robot many times and writes down every mistake. After the meeting we clean the room and put the tools back in the box. Then we walk home and talk about new ideas. Tomorrow we will try a new sensor, and maybe the robot will finally win. The club is small, but everyone has a job. Some of us write code, some of us build the body, and some of us film the tests so that we can watch them again. Learning to fix a robot is slower than learning to drive one, but it is more fun. When the robot works, the whole room cheers."""

def tokenize(text):
    """Lower-case words (letters and apostrophes) and the punctuation marks . , ! ? as separate tokens."""
    return re.findall(r"[a-z']+|[.,!?]", text.lower())

corpus_tokens = tokenize(corpus)
corpus_words = [t for t in corpus_tokens if t[0].isalpha()]
print("Words:", len(corpus_words), "| different words:", len(set(corpus_words)), "| tokens with punctuation:", len(corpus_tokens))
```

```python
def train_ngram(text, n):
    """Count, for every context of n-1 words, which token comes next. n=1 has no context at all."""
    toks = ["."] * (n - 1) + tokenize(text)           # padding, so the first sentence also has a context
    model = defaultdict(Counter)
    for i in range(n - 1, len(toks)):
        context = tuple(toks[i - (n - 1): i])         # the n-1 tokens before position i
        model[context][toks[i]] += 1
    return model

models = {n: train_ngram(corpus, n) for n in (1, 2, 3)}

counts = models[2][("robot",)]                         # what follows "robot" in the corpus?
print("After 'robot':", dict(counts.most_common()))

total = sum(counts.values())
probs_after_robot = {word: c / total for word, c in counts.most_common()}
for word, p in probs_after_robot.items():
    print(f"{word!r:<10} {counts[word]:>2} of {total}  ->  p = {p:.3f}")
```

   You should see `Words: 283 | different words: 148 | tokens with punctuation: 319`. After "robot" the counts are `needs: 3`, `that: 2` and eight words seen once each (13 in total), so the probabilities are `needs` 0.231, `that` 0.154 and 0.077 for each of the other eight. That table is the model.

   *Exercise 2.1:* print the next-word counts and probabilities for two other words of your choice (change `("robot",)` to, for example, `("the",)`). Which word has the most possible followers, and which has only one? Write 2 sentences: how does the number of different followers relate to how much freedom the model has?

   **Step 2.2. Generate sentences with n = 1, 2 and 3.** `models[1]` has no context (a bag of word frequencies), `models[2]` is the bigram model, `models[3]` the trigram model. `generate_ngram` looks up the current context, picks the next token at random with a chance proportional to its count (`rng.choices`) and stops at a full stop. The seed makes the result repeatable. Then print five sentences for each n.

```python
def generate_ngram(model, n, length=20, seed=0, start=None):
    """Write up to `length` tokens (stopping at the first full stop) by repeatedly sampling the next token."""
    rng = random.Random(seed)
    out = list(start or [])                            # optional starting words
    context = tuple(out[-(n - 1):]) if n > 1 else ()
    if n > 1 and len(context) < n - 1:                 # no (or too few) starting words: begin like a sentence
        context = tuple(["."] * (n - 1 - len(context))) + context
    for _ in range(length):
        options = model.get(context)
        if not options:                                # this context never appeared in the corpus
            break
        next_token = rng.choices(list(options), weights=list(options.values()))[0]   # more counts = more likely
        out.append(next_token)
        if next_token == ".":
            break
        context = (context + (next_token,))[-(n - 1):] if n > 1 else ()
    return out

def detok(tokens):
    """Tokens back to a readable sentence."""
    text = " ".join(tokens)
    for mark in (".", ",", "!", "?"):
        text = text.replace(" " + mark, mark)
    return text[:1].upper() + text[1:]
```

```python
for n in (1, 2, 3):
    print(f"--- n = {n} ({'no context' if n == 1 else str(n - 1) + ' word' + ('s' if n > 2 else '') + ' of context'}) ---")
    for k in range(5):
        print("  ", detok(generate_ngram(models[n], n, length=20, seed=k)))
```

   You should see, with different words but the same pattern: for n = 1 word salad (for example "Tests from build, on we across a that sees maybe line…"); for n = 2 short sentences that are fine locally ("We clean the line again.") but wander ("We build a small robot lost the tools back in the robot has two wheels…"); for n = 3 mostly correct sentences that are often copied from the corpus ("The club is small, but it is more fun." appears three times in the five seeds).

   *Exercise 2.2:* write 3 to 4 sentences describing how quality changes from n = 1 to n = 3, with one example sentence of each as evidence. Was your Imagine-first guess right? Why does more context give more sensible text, and why does it also give more copying? (With more context the model has seen fewer different continuations, sometimes only one, so it has no choice but to repeat the corpus.)

   **Step 2.3. Measure copying and variety.** Two numbers describe a generator. The **copy rate** is the fraction of all runs of three neighbouring tokens in the generated sentences that appear word for word in the corpus; near 1 means the model mostly repeats what it read. The **distinct words** is the number of different words it uses. The block writes 200 sentences for each n.

```python
def windows(tokens, size=3):
    """All runs of `size` neighbouring tokens."""
    return [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

corpus_3grams = set(windows(corpus_tokens, 3))        # every 3-token run that really occurs in the corpus
N_SENT = 40 if QUICK else 200                         # how many sentences to write per model
generated = {n: [generate_ngram(models[n], n, length=20, seed=1000 + k) for k in range(N_SENT)] for n in (1, 2, 3)}
print({n: len(v) for n, v in generated.items()}, "sentences written")

def copy_rate(sentences):
    """Fraction of all 3-token runs in `sentences` (a list of token lists) that appear word for word in the corpus."""
    runs = [w for s in sentences for w in windows(s, 3)]
    return sum(w in corpus_3grams for w in runs) / len(runs)
```

```python
def distinct_words(sentences):
    return len({t for s in sentences for t in s if t[0].isalpha()})

print(f"{'n':>2} {'copy rate':>10} {'distinct words':>15}")
for n in (1, 2, 3):
    rate = copy_rate(generated[n])
    if rate is None:
        print(f"{n:>2} {'(finish copy_rate)':>10}")
        continue
    print(f"{n:>2} {rate:>10.2f} {distinct_words(generated[n]):>15}")
    RESULTS[f"part2_copy_rate_n{n}"] = round(rate, 2)
    RESULTS[f"part2_distinct_words_n{n}"] = distinct_words(generated[n])
```

   You should see this table:

   | n | copy rate | distinct words |
   |---|-----------|----------------|
   | 1 | 0.00 | 148 |
   | 2 | 0.62 | 148 |
   | 3 | 1.00 | 63 |

   With n = 1 nothing is copied and all 148 corpus words appear. With n = 3 every run of three tokens is copied and only 63 different words are used: a very expensive photocopier. The bigram model sits in between.

   *Exercise 2.3:* copy the table into a markdown cell and answer: which n is the most "creative" and which the most "faithful"? Which would you choose to write a poem, and which to answer from a manual? Write 3 sentences. Use the same words as in Part 3: ==more context moves the model from inventing towards copying==. Predict what would happen to the copy rate at n = 3 if the corpus were ten times bigger, then try it with a paragraph of your own.

   **Step 2.4. Imagine your own corpus.** Give the model your imagination. Write 100 words or more about a world you imagine (a future school, a city on the sea, a fantasy land) in plain sentences with full stops. Train the model on it and read the five sentences it writes. The example below is a floating school of 144 words; replace the text with your own.

```python
my_corpus = """In the year 2080 our school floats on the sea. The classrooms are round glass rooms that rise and fall with the waves. Every morning we sail to a new island, and every afternoon we study the water, the birds and the weather. The teachers are friendly engineers who build tiny boats with us. We grow vegetables on the roof, and we catch rain in big blue tanks. At night the school glows like a lantern, and the whales swim close to watch us. Our maps are drawn on the sea by the wind. Nobody wears shoes, and the library is a ship full of books that smell of salt. When a storm comes, we sing together until the sea is calm again. Every student has a robot gull that carries messages across the waves. The best lesson is the one that begins at sunrise."""
my_n = 2

if len(tokenize(my_corpus)) < 100:
    print("Your corpus has", len([t for t in tokenize(my_corpus) if t[0].isalpha()]), "words. Write at least 100, then run this cell again.")
else:
    my_model = train_ngram(my_corpus, my_n)
    for k in range(5):
        print(k + 1, detok(generate_ngram(my_model, my_n, length=25, seed=k)))

my_ratings = {"sense": 3, "surprise": 4}
```

   *Exercise 2.4:* rate the five sentences from 1 to 5 for **sense** and for **surprise** (surprise means "I would not have written that, but I like it"), and write the two ratings in `my_ratings`. Then set `my_n = 3` and rate again. Write 3 sentences: did more context give you more sense and less surprise? Which sentence would you keep, and why? Can a model trained on 144 words invent a word that is not in your text? (No: it can only recombine your 144 words.)

   **Step 2.5. Compare with GPT-2.** Both models start with the same words, "The robotics club". GPT-2 is the same idea at a very different scale. The random text is different on every run.

```python
start = ["the", "robotics", "club"]

torch.manual_seed(SEED)
print("GPT-2     :", generate("The robotics club", max_new_tokens=30, do_sample=True, temperature=0.8, top_k=40))
print()
for n in (2, 3):
    print(f"n-gram n={n}:", detok(generate_ngram(models[n], n, length=30, seed=3, start=start)))
```

   You should see GPT-2 write a fluent sentence that goes somewhere new, the trigram model return a sentence copied from your corpus, and the bigram model wander.

   *Exercise 2.5:* write three reasons why GPT-2 writes better than the counting model (use: sub-word tokens with a vocabulary of 50,257 pieces, far more training data, a much longer context, many more parameters (124 million), learned representations in which similar words get similar vectors), and one thing the tiny model does that GPT-2 cannot. Suggested answer for the last: the tiny model is guaranteed to use only the words in your own corpus, so it never invents a fact or a name you did not give it.

3. *Part 3. The dials of imagination (30 min).* A language model ends each step with a list of scores called logits. Softmax with temperature turns them into probabilities, top-k and top-p cut off the unlikely words, and the model samples one word. These are the dials that make text safe and dull or wild and inventive. The part follows the lesson section "How a language model writes" (the sampling figure, temperature, top-k, top-p and the softmax formula).

   **Imagine first.** What will happen to GPT-2's writing when you raise the temperature from 0.2 to 1.5? Write your prediction in one or two sentences in `my_guess_temperature` and compare it with step 3.2.

```python
my_guess_temperature = "At 0.2 the text will be safe and repetitive; at 1.5 it will be wild, with strange words."   # write your own
```

   **Step 3.1. Softmax with temperature, by hand.** The lesson's sampling figure uses six invented logits (illustrative numbers, not from a real model): away 4.2, beyond 2.4, from 2.1, off 1.2, behind 0.4, pizza −1.5. The formula is `pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)`: divide each logit by the temperature T, exponentiate, then divide by the sum so the probabilities add up to 1. `filter_top` keeps the top-k most likely words and/or the smallest group of most likely words whose probabilities add up to at least p (top-p), then renormalises. In this simplified version top-k and top-p are both applied to the same probabilities and the smaller group wins.

```python
logits = np.array([4.2, 2.4, 2.1, 1.2, 0.4, -1.5])          # the lesson's sampling figure (illustrative numbers)
names = ["away", "beyond", "from", "off", "behind", "pizza"]

def softmax_t(logits, T=1.0):
    """Probabilities from logits at temperature T."""
    z = np.asarray(logits, dtype=float) / T            # temperature scales the logits
    z = z - z.max()                                    # subtracting the maximum changes nothing but avoids overflow
    e = np.exp(z)
    return e / e.sum()

def filter_top(probs, top_k=None, top_p=None):
    """Keep only the top-k tokens and/or the smallest group whose probabilities add up to top_p; renormalise."""
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)                         # token numbers from most to least likely
    keep_count = len(probs)
    if top_k is not None:
        keep_count = min(keep_count, top_k)
    if top_p is not None:
        cumulative = np.cumsum(probs[order])
        keep_count = min(keep_count, int(np.searchsorted(cumulative, top_p)) + 1)   # first place where the sum reaches top_p
    kept = np.zeros(len(probs))
    kept[order[:keep_count]] = probs[order[:keep_count]]
    return kept / kept.sum()
```

```python
def show_dist(label, probs):
    print(f"{label:<22}" + "  ".join(f"{n}={p:.3f}" for n, p in zip(names, probs)))

if softmax_t(logits) is None or filter_top(softmax_t(logits), top_k=3) is None:
    print("Finish softmax_t and filter_top first.")
else:
    for T in (0.5, 1.0, 1.5):
        show_dist(f"T = {T}", softmax_t(logits, T))
    p = softmax_t(logits, 1.0)
    show_dist("T = 1, top-k = 3", filter_top(p, top_k=3))
    show_dist("T = 1, top-p = 0.9", filter_top(p, top_p=0.9))
    show_dist("T = 0.5, top-p = 0.9", filter_top(softmax_t(logits, 0.5), top_p=0.9))
```

   You should see:

   | setting | away | beyond | from | off | behind | pizza |
   |---|---|---|---|---|---|---|
   | T = 0.5 | 0.957 | 0.026 | 0.014 | 0.002 | 0.000 | 0.000 |
   | T = 1 | 0.734 | 0.121 | 0.090 | 0.037 | 0.016 | 0.002 |
   | T = 1.5 | 0.560 | 0.169 | 0.138 | 0.076 | 0.044 | 0.013 |
   | T = 1, top-k = 3 | 0.777 | 0.128 | 0.095 | 0 | 0 | 0 |
   | T = 1, top-p = 0.9 | 0.777 | 0.128 | 0.095 | 0 | 0 | 0 |
   | T = 0.5, top-p = 0.9 | 1.000 | 0 | 0 | 0 | 0 | 0 |

   At T = 1 the top word gets 0.734, the value the lesson's figure shows as 0.73. At T = 1, top-p = 0.9 keeps three words because 0.734 + 0.121 = 0.855 is still below 0.9 and adding 0.090 gives 0.945. At T = 0.5 the first word alone is already above 0.9, so top-p keeps only that one.

   *Exercise 3.1:* write the probability of "away" at T = 0.5, 1 and 1.5 in a markdown cell, and explain in two sentences what temperature does to the shape of the distribution (low T sharpens, high T flattens). Then change the logit of "pizza" to 3.0 and predict, before you run, what happens to the top-p = 0.9 group at T = 1. Why can top-p keep a different number of words in different situations, while top-k always keeps the same number?

   **Step 3.2. Greedy decoding versus sampling on GPT-2.** Greedy decoding (`do_sample=False`) always takes the single most likely token. Sampling draws from the probabilities with the temperature you set. `top_k=0` switches off GPT-2's default top-k filter, so only the temperature changes, and the seed is reset before each run. The second block puts a number on repetition: `1 − (different words ÷ all words)`.

```python
prompt = "The robot opened its eyes and"

print("GREEDY (always the single most likely token):")
greedy_text = generate(prompt, max_new_tokens=40, do_sample=False)
print(greedy_text, "\n")

sampled = {}
for T in (0.3, 0.7, 1.0, 1.5):
    torch.manual_seed(SEED)                                  # same random numbers each time, so only T changes
    sampled[T] = generate(prompt, max_new_tokens=40, do_sample=True, temperature=T, top_k=0)
    print(f"SAMPLING, temperature {T}:\n{sampled[T]}\n")
```

```python
def repeat_ratio(text):
    """How repetitive is the text? 1 minus (different words / all words). 0 = no word is repeated."""
    words = text.lower().split()
    return 1 - len(set(words)) / len(words)
```

```python
for label, text in [("greedy", greedy_text)] + [(f"T={T}", s) for T, s in sampled.items()]:
    print(f"repeat ratio {label:<8} {repeat_ratio(text):.2f}")
```

   You should see (on a CPU with seed 42; a GPU gives different random numbers but the same pattern): the greedy text starts with a normal sentence and then repeats "I'm not sure what's going on." (repeat ratio 0.31). At T = 0.3 the text is calm and somewhat repetitive (0.19), at 0.7 it is lively and sensible (0.03), at 1.0 it drifts to odd ideas ("the poison furnace", 0.16), and at 1.5 it turns into nonsense from rare words (0.02). A very low repeat ratio therefore does not mean good writing.

   *Exercise 3.2:* choose your own creative prompt and repeat the five runs. Copy one sentence from each run into a markdown cell, then write three sentences: what repetition looks like in greedy decoding and why it happens (the most likely token after a repeated phrase is often the phrase again); at which temperature the text was most pleasing to you; and whether your Imagine-first prediction was right.

   **Step 3.3. The creativity–coherence trade-off, measured.** "Creative" and "coherent" are vague words, so measure them for six settings with 5 samples each. **Diversity** is the number of different word pairs divided by all word pairs over the five samples (1.0 means no pair repeated). **Plausibility** is the average probability that GPT-2 gives to its own tokens (high means safe and predictable). Set `N_SAMPLES = 2` and `NEW_TOKENS = 20` if you want a quick run. It takes one to two minutes on a CPU.

```python
settings = {
    "T=0.2": dict(temperature=0.2, top_k=0),
    "T=0.5": dict(temperature=0.5, top_k=0),
    "T=0.8": dict(temperature=0.8, top_k=0),
    "T=1.0": dict(temperature=1.0, top_k=0),
    "T=1.5": dict(temperature=1.5, top_k=0),
    "T=1, top-p=0.9": dict(temperature=1.0, top_k=0, top_p=0.9),
}
N_SAMPLES = 2 if QUICK else 5            # samples per setting
NEW_TOKENS = 20 if QUICK else 40
prompt_ids = len(tok("Once upon a time").input_ids)   # how many tokens belong to the prompt

def continuations(prompt, setting):
    """N_SAMPLES different continuations (without the prompt) for one setting."""
    texts = []
    for k in range(N_SAMPLES):
        torch.manual_seed(SEED + k)
        full = generate(prompt, max_new_tokens=NEW_TOKENS, do_sample=True, **setting)
        texts.append(full[len(prompt):])
    return texts
```

```python
def diversity(texts):
    """Different word pairs divided by all word pairs, over all texts together (1.0 = no pair ever repeats)."""
    pairs = [p for t in texts for p in zip(t.lower().split(), t.lower().split()[1:])]
    return len(set(pairs)) / len(pairs)

def plausibility(prompt, texts):
    """Average probability that GPT-2 gives to each token of its own continuation (higher = more predictable)."""
    probs = []
    for t in texts:
        rows = token_logprobs(prompt + t)[prompt_ids:]       # skip the prompt's own tokens
        probs += [math.exp(lp) for _, lp in rows]
    return sum(probs) / len(probs)

start_text = "Once upon a time"
points = {}
for name, setting in settings.items():
    texts = continuations(start_text, setting)
    points[name] = (plausibility(start_text, texts), diversity(texts))
    print(f"{name:<16} plausibility {points[name][0]:.3f}   diversity {points[name][1]:.3f}")
    RESULTS[f"part3_plausibility_{name}"] = round(points[name][0], 3)
    RESULTS[f"part3_diversity_{name}"] = round(points[name][1], 3)
```

```python
plt.figure(figsize=(6.5, 4))
for name, (plaus, div) in points.items():
    plt.scatter(plaus, div, s=60)
    plt.annotate(name, (plaus, div), textcoords="offset points", xytext=(6, 4), fontsize=9)
plt.xlabel("plausibility (average token probability, higher = safer)")
plt.ylabel("diversity (different word pairs, higher = more varied)")
plt.title("Creativity versus coherence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

   You should see (CPU, seed 42):

   | setting | plausibility | diversity |
   |---|---|---|
   | T = 0.2 | 0.496 | 0.312 |
   | T = 0.5 | 0.280 | 0.914 |
   | T = 0.8 | 0.222 | 0.908 |
   | T = 1.0 | 0.130 | 0.982 |
   | T = 1.5 | 0.023 | 1.000 |
   | T = 1, top-p = 0.9 | 0.164 | 0.971 |

   Plausibility falls steadily as the temperature rises, from 0.50 to 0.02. Diversity jumps from 0.31 (the samples copy each other) to above 0.9 and then creeps up to 1.0. With only five samples neighbouring settings can swap places (T = 0.5 and T = 0.8 here), so read the trend rather than single points. Top-p = 0.9 at T = 1 has a higher plausibility (0.164) than plain T = 1 (0.130) with almost the same diversity: cutting off the unlikely tail removes the worst words cheaply.

   *Exercise 3.3:* save the plot, and write two sentences that describe the trade-off in your own words: what happens to plausibility and diversity as the temperature goes up, and what does that mean for how the text reads? Then add two settings of your own (for example `dict(temperature=0.7, top_k=40)` and `dict(temperature=1.2, top_p=0.8)`), predict where they will sit on the plot, and check. Which setting gives the best balance for a story, and why?

   **Step 3.4. Choose and justify settings for three jobs.** Different jobs need different dials: summarising a legal notice (it must be faithful), writing a poem (surprising but still meaningful) and brainstorming 20 names for a robot club (many different ideas). Choose temperature (0.1 to 2.0), top_k (1 to 100) and top_p (0.1 to 1.0) for each. The example below is one reasonable choice, followed by a try-out for the poem settings.

```python
jobs = {
    "summarise a legal notice":          {"temperature": 0.2, "top_k": 20, "top_p": 0.8},
    "write a poem":                      {"temperature": 0.9, "top_k": 50, "top_p": 0.95},
    "brainstorm 20 names for a robot club": {"temperature": 1.2, "top_k": 100, "top_p": 0.98},
}
reasons_jobs = """A legal summary must stay faithful, so I want almost no randomness. A poem benefits from surprising but still meaningful words.
Brainstorming wants many different, even odd, ideas, so the temperature is highest and the filters are loosest."""
```

```python
def filled(j):
    return all(isinstance(v, (int, float)) for v in j.values())

# friendly checks: sensible ranges, not hard answers
legal, poem, brain = jobs["summarise a legal notice"], jobs["write a poem"], jobs["brainstorm 20 names for a robot club"]
check("Legal summary: low temperature (0.5 or less)", legal["temperature"] <= 0.5, "A summary should not invent things: try 0.2 to 0.4.")
check("Poem: a middle temperature (0.7 to 1.3)", 0.7 <= poem["temperature"] <= 1.3, "A poem likes variety but needs sense: try about 0.9.")
check("Brainstorm: higher than the legal summary and at least 0.9", brain["temperature"] >= 0.9 and brain["temperature"] > legal["temperature"], "Try 1.0 to 1.3.")
RESULTS["part3_job_settings"] = jobs
```

```python
# Optional: try your poem settings and see what GPT-2 does with them.
j = jobs["write a poem"]
if filled(j):
    torch.manual_seed(SEED)
    print(generate("The moon is", max_new_tokens=30, do_sample=True, temperature=j["temperature"], top_k=int(j["top_k"]), top_p=j["top_p"]))
```

   *Exercise 3.4:* write your own three settings and a justification of two or three sentences in a markdown cell: why a low temperature for the legal notice, a middle one for the poem, and the highest for the brainstorm? Run each job's settings on a prompt of your own (for the legal notice use something like "In summary, the tenant must") and judge whether the output fits. Remember: ==even at a low temperature a language model can leave out or change a detail, so a person must check anything that matters, such as a legal text==.

4. *Part 4. Imagine in pictures (30 min).* ==Make pictures two ways: from plain random noise on your own computer, and from a sentence with Stable Diffusion.== You use the "Beyond text" section and the diffusion figure with its forward-noising formula.
   *Imagine first:* before you run anything, write one or two sentences describing a picture you would like an AI to make (a place, a creature, a mood, colours). Keep it; you return to it in step 4.3.

**4.1 Art from noise.** A picture is a grid of numbers. Fill a grid with random numbers (static), blur it many times, and map each number to a colour: the whole picture comes from the random seed. This is generation as sampling from noise.
```python
import numpy as np, matplotlib.pyplot as plt
PALETTES = {"sunset": ["#1b1b3a", "#693668", "#a74482", "#f84aa7", "#ff3562"],
            "ocean":  ["#03045e", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"],
            "forest": ["#10200f", "#2d6a4f", "#52b788", "#b7e4c7", "#f1faee"]}
def hex_to_rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0

def make_art(seed, smooth=40, palette="sunset", size=96):
    rng = np.random.default_rng(seed)
    field = rng.random((size, size))                       # white noise
    for _ in range(smooth):                                # blur: average with the 4 neighbours
        field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                 + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
    field = (field - field.min()) / (field.max() - field.min())
    anchors = np.array([hex_to_rgb(h) for h in PALETTES[palette]])
    pos = np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(field, pos, anchors[:, c]) for c in range(3)], axis=-1)

fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
for ax, s in zip(axes.ravel(), range(6)):
    ax.imshow(make_art(s)); ax.set_title(f"seed {s}"); ax.axis("off")
plt.tight_layout(); plt.show()
```
   *Exercise 4.1:* call `make_art(7, smooth=sm, palette="ocean")` for `sm` = 5, 40 and 150 and put the three pictures side by side. Which setting gives fine grain and which gives big soft shapes? Then switch the palette to `"forest"`: does the shape change or only the colours? Run `make_art(3)` twice: are the two pictures identical? Finally write 2 or 3 sentences: what does the seed control, what does `smooth` control, what does the palette control, and why does this show that "generation is sampling from noise"? You should find that the seed decides where the blobs sit, `smooth` their size, and the palette only their colours.

**4.2 Forward diffusion by hand.** The lesson's diffusion figure destroys a picture step by step. With a noise schedule β (here 1000 steps rising from 1e-4 to 0.02), α<sub>t</sub> = 1 − β<sub>t</sub> and ᾱ<sub>t</sub> = α<sub>1</sub>·α<sub>2</sub>·…·α<sub>t</sub>, you can jump straight to step t: x<sub>t</sub> = √ᾱ<sub>t</sub> · x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) · ε, with ε random normal noise. Our x<sub>0</sub> is a 12×12 smiley written as digits.
```python
SMILEY = ["000111111000", "001000000100", "010000000010", "010100001010",
          "010100001010", "010000000010", "010100001010", "010011110010",
          "001000000100", "000111111000", "000000000000", "000000000000"]
x0 = np.array([[int(c) for c in row] for row in SMILEY], dtype=float) * 2 - 1   # range -1..1

betas = np.linspace(1e-4, 0.02, 1000)
alpha_bars = np.cumprod(1 - betas)                      # alpha_bars[t-1] is alpha-bar at step t
def abar(t): return 1.0 if t == 0 else float(alpha_bars[t - 1])

eps = np.random.default_rng(0).standard_normal(x0.shape)    # ONE noise sample, reused for every step
fig, axes = plt.subplots(1, 6, figsize=(12, 2.3))
for ax, t in zip(axes, [0, 50, 150, 300, 600, 1000]):
    xt = np.sqrt(abar(t)) * x0 + np.sqrt(1 - abar(t)) * eps
    ax.imshow(xt, cmap="gray", vmin=-2, vmax=2); ax.set_title(f"t={t}\nabar={abar(t):.3g}"); ax.axis("off")
plt.tight_layout(); plt.show()
print(round(abar(300), 4), f"{abar(1000):.2e}")
```
   *Exercise 4.2:* (a) Find the first step `t` at which ᾱ<sub>t</sub> falls below 0.5 (less than half of the original signal is left) with a loop over `abar(t)` and store it as `t_half`. (b) Look at your row of pictures: around which step can you no longer see the smiley? (c) Write one sentence on how a model could learn to run the process backwards. Hint: we know exactly which noise ε we added, so every training example comes with the right answer. You should get `abar(300)` = 0.3964, `abar(1000)` = 4.0e-5 and `t_half` = 260.

**4.3 Text to image with Stable Diffusion (GPU).** This runs the process in reverse: from random noise, guided by your text, the model removes noise step by step. Use half precision, 25 steps and 512×512 to keep the shared GPU free. Use the Stable Diffusion v1.5 model your instructor cached; its ID is `runwayml/stable-diffusion-v1-5` or `stable-diffusion-v1-5/stable-diffusion-v1-5`.
```python
import torch, gc
from diffusers import StableDiffusionPipeline
def load_sd():
    for name in ("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"):
        try: return StableDiffusionPipeline.from_pretrained(name, torch_dtype=torch.float16).to("cuda")
        except Exception as e: print("not available:", name)
    raise SystemExit("Ask your instructor which Stable Diffusion v1.5 ID is cached.")
pipe = load_sd(); pipe.set_progress_bar_config(disable=True)

def draw(prompt, seed=0, guidance=7.5, steps=25):
    g = torch.Generator(device="cuda").manual_seed(seed)       # the seed fixes the starting noise
    return pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=g, height=512, width=512).images[0]
def show(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.3))
    for ax, im, t in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(im); ax.set_title(t, fontsize=8, wrap=True); ax.axis("off")
    plt.tight_layout(); plt.show()

prompts = ["a lighthouse",                                                 # replace with the picture you imagined
           "a lighthouse on a cliff at sunset",
           "a lighthouse on a cliff at sunset, warm orange light, calm sea, watercolor painting, wide shot"]
show([draw(p, seed=1) for p in prompts], prompts)                          # vague -> specific
show([draw(prompts[2], seed=s) for s in range(4)], [f"seed {s}" for s in range(4)])          # same prompt, four seeds
show([draw(prompts[2], seed=1, guidance=g) for g in (3, 7.5, 12)], [f"guidance {g}" for g in (3, 7.5, 12)])
```
   *Exercise 4.3:* write an **imagination journal** of four short entries: what you pictured at the start of Part 4; what the model gave you for your most specific prompt (what matched, what surprised you); which words in the prompt changed the picture most (a style word such as "watercolor", the light, the place); and what changed when you changed only the seed and what changed when you changed only the guidance scale (3 follows the text loosely, 7.5 is the usual, 12 follows it hard). Keep the six or more images in your notebook. When you finish, free the GPU: `del pipe; gc.collect(); torch.cuda.empty_cache()`.

5. *Part 5. Imagine in sound and across languages (25 min).* ==Chain translation, speech and speech recognition, and measure how much is lost on the way.== This uses the "Beyond text" section on speech and translation. Confirm with your instructor that `Helsinki-NLP/opus-mt-en-fr`, `Helsinki-NLP/opus-mt-fr-en`, `facebook/mms-tts-eng` and `openai/whisper-tiny` are in the offline cache.
   *Imagine first:* take the sentence "The robot club meets on Friday." Predict what it will look like after English to French to English: unchanged, slightly different or very different? Write your prediction down.

**5.1 Translation round trip.** A round trip is a cheap test of how much meaning survives. Use three sentences of growing difficulty: plain, an idiom, and a joke built on ambiguity.
```python
from transformers import MarianMTModel, MarianTokenizer
def load_translator(name):
    mt_tok = MarianTokenizer.from_pretrained(name)
    mt = MarianMTModel.from_pretrained(name).to(device).eval()
    def translate(text):
        batch = mt_tok([text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = mt.generate(**batch, max_new_tokens=80, num_beams=4)
        return mt_tok.decode(out[0], skip_special_tokens=True)
    return translate
en_fr = load_translator("Helsinki-NLP/opus-mt-en-fr")
fr_en = load_translator("Helsinki-NLP/opus-mt-fr-en")

originals = ["The robot club meets on Friday.",
             "It is raining cats and dogs.",
             "Time flies like an arrow; fruit flies like a banana."]
forward = [en_fr(s) for s in originals]
back = [fr_en(s) for s in forward]
for o, f, b in zip(originals, forward, back):
    print("English:", o, "\nFrench :", f, "\nBack   :", b, "\n")
```
   *Exercise 5.1:* for each of the three sentences, write one note on what meaning, nuance or wordplay was lost (or "nothing lost"). Add a sentence of your own, ideally with a word that has two meanings (such as "bank" or "light"), and run it too. Were you right about your prediction? You should see, with small differences between library versions, "The robot club meets Friday." for the plain sentence, a literal French "Il pleut des chats et des chiens." for the idiom (a French speaker would say "Il pleut des cordes"), and the joke coming back word for word even though the wordplay on "flies" cannot work in French. A round trip that looks unchanged does not prove the meaning survived.

**5.2 Measuring the damage: word error rate.** "It looks about the same" is not a measurement. The word error rate is WER = (S + I + D) / N, the edit distance between the reference and the hypothesis (substitutions, insertions, deletions) divided by the N words of the reference. It can go above 1.
```python
import re
def clean(text): return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()

def word_error_rate(reference, hypothesis):
    r, h = clean(reference), clean(hypothesis)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]      # d[i][j] = edits to turn r[:i] into h[:j]
    for i in range(len(r) + 1): d[i][0] = i
    for j in range(len(h) + 1): d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            same = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + same)
    return d[len(r)][len(h)] / max(1, len(r))

print(word_error_rate("the cat sat", "the cat sat"))        # 0.0
print(word_error_rate("the cat sat", "a cat sat down"))     # 0.666...
print([round(word_error_rate(o, b), 3) for o, b in zip(originals, back)])
```
   *Exercise 5.2:* work out these WERs by hand before you run them: (a) reference "i like green tea", hypothesis "i like tea"; (b) reference "open the door", hypothesis "close the window now"; (c) reference "go", hypothesis "go there right now please". Then check with the function. You should get 0.25, 1.0 and 4.0. Explain in one sentence why (c) is above 1. For your round trips you should get about 0.167, 0.333 and 0.0.

**5.3 The audio chain.** GPT-2 writes a short text, `facebook/mms-tts-eng` speaks it, `openai/whisper-tiny` listens, and the WER between the original and the final text tells you how much was lost. Play the sound in the notebook and save a `.wav` file.
```python
import scipy.io.wavfile as wavfile
from IPython.display import Audio, display
from transformers import VitsModel, AutoTokenizer, pipeline

torch.manual_seed(42)
story = generate("The robot club built a small robot that", max_new_tokens=22, do_sample=True, temperature=0.7, top_k=40)
original_text = " ".join(story.split())
if "." in original_text: original_text = original_text[:original_text.rindex(".") + 1]    # drop a half sentence

tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
tts = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if device == "cuda" else -1)

def speak_and_listen(text, path="chain.wav"):
    with torch.no_grad():
        wave = tts(**tts_tok(text, return_tensors="pt")).waveform[0].numpy()
    sr = tts.config.sampling_rate
    wavfile.write(path, sr, wave)
    heard = asr({"raw": wave.astype(np.float32), "sampling_rate": sr}, generate_kwargs={"language": "en"})["text"].strip()
    return wave, sr, heard

wave, sr, heard = speak_and_listen(original_text)
display(Audio(wave, rate=sr))
print("Original:", original_text, "\nHeard   :", heard, "\nWER:", round(word_error_rate(original_text, heard), 2))
hard_text = "Dr. Okonkwo built robot ZX-9 in 2025 for 3 young engineers from Cebu."      # numbers, a rare name, an abbreviation
wave2, sr2, heard2 = speak_and_listen(hard_text, path="chain_hard.wav")
print("Original:", hard_text, "\nHeard   :", heard2, "\nWER:", round(word_error_rate(hard_text, heard2), 2))
```
   *Exercise 5.3:* mark every difference between each original and what Whisper heard, then write **two reasons** the final text differs (think about punctuation and capital letters, rare or invented words, numbers, the synthetic voice, and that `whisper-tiny` is the smallest Whisper). Change `hard_text` to a sentence of your own that includes a name from your community and see which words the chain gets wrong. In our run the GPT-2 sentence came back word for word (WER 0.00), while the hard sentence came back as "rope on who built robots extend head for beyond any nearers from sebu." (WER 0.86, 14 reference words). Your GPT-2 sentence will differ because it is sampled; the lesson is that rare words and numbers cause most of the damage.

**5.4 A voice assistant for your community.** *Exercise 5.4:* imagine a voice assistant for a language your community speaks. Write the language, which of the models you met you would need (speech to text, translation or a language model, text to speech), what data they would need (how many hours of recorded speech, who records it) and which speakers check the output before people rely on it. Small languages have little recorded data, which is why these models usually work worse for them. When you finish: `del tts, asr, en_fr, fr_en; gc.collect(); torch.cuda.empty_cache()`.

6. *Part 6. Imagine an AI product, then try to break it (30 min).* ==Design an AI product that solves a real problem, then test GPT-2 for two weaknesses and plan how a team would reduce them.== This uses the "Risks" section (trustworthy AI and its five principles: fairness, transparency, privacy, safety, accountability) and the "Prompts" section.
   *Imagine first:* choose a real problem in your school or community that an AI could help with (finding library books, answering new-student questions, translating notices for parents, summarising long announcements) and write it in one sentence.

**6.1 The AI spec sheet.** Decide before you build. Fill in the dictionary and print it as a card.
```python
spec = {"name": "", "who_it_helps": "", "input": "", "output": "", "model_type": "",
        "learns_from": "", "human_in_charge": "", "success_measure": ""}

def print_card(card, width=62):
    print("+" + "-" * width + "+")
    for k, v in card.items():
        line = f"{k.replace('_', ' ').title():<16}: {v or '(fill me in)'}"
        while len(line) > width - 2:
            print("| " + line[:width - 2].ljust(width - 2) + " |"); line = " " * 18 + line[width - 2:].lstrip()
        print("| " + line.ljust(width - 2) + " |")
    print("+" + "-" * width + "+")
print_card(spec)
```
   *Exercise 6.1:* fill in every field with a short phrase or sentence. For `model_type` choose from the lesson: a language model, a translation model, a speech model, an image-generation (diffusion) model or a classifier. For `human_in_charge` say who checks the output and what they can do if it is wrong. For `success_measure` give one number you could really measure (for example "share of 50 test questions a teacher marks correct"). Print the card.

**6.2 Fluent but wrong.** Language models produce *likely* text, not *true* text. Here are six factual prompts with known answers; we take GPT-2's five most likely next tokens for each and check whether the true answer is among them.
```python
facts = [("The capital of France is", "Paris"),
         ("The largest planet in our solar system is", "Jupiter"),
         ("Water is made of hydrogen and", "oxygen"),
         ("The Earth orbits around the", "Sun"),
         ("The first month of the year is", "January"),
         ("There are seven days in a", "week")]

def fact_table(items, top=5):
    hits = 0
    for prompt, truth in items:
        guesses = [t.strip().lower() for t, p in next_token_probs(prompt, top=top)]
        found = truth.lower() in guesses
        hits += found
        print("OK " if found else "MISS", f"{prompt!r:52} true: {truth:8} top-{top}: {guesses}")
    return hits / len(items)
fact_rate = fact_table(facts)
print(f"\nThe true answer was in the top 5 for {fact_rate:.0%} of the prompts.")
RESULTS["fact_rate"] = round(fact_rate, 3)
```
   *Exercise 6.2:* add two prompts of your own (a prompt that ends right before the answer, and a one-word true answer you are certain of) and run `fact_table` on them. Then explain in 2 or 3 sentences why a model can write a fluent sentence that is false; use the words *likely* and *true*. You should find the true answer in the top 5 for 5 of the 6 prompts (83%): for "The first month of the year is" GPT-2's top five are "a", "the", "usually", "when", "always", and "January" is missing. Even when it is found, it is often not the top guess ("Paris" is only in fifth place after "the", "now", "a", "France").

**6.3 A bias probe, carefully.** Language models learn patterns from text written by people, so they can pick up the stereotypes in that text. This is a ==risk to test for and reduce, not a verdict on any person or job.== After "The nurse said that" and "The engineer said that", how likely is the next word "he", "she" or "they"? We read the model's probabilities for the next word (no free generation), then draw 20 samples of the next word to see how counts bounce around.
```python
def pronoun_probe(job, n=20, seed=0):
    top = next_token_probs(f"The {job} said that", top=200)
    prob = {}
    for t, p in top:                                           # add up " she", " She", ...
        if t.startswith(" "): prob[t.strip().lower()] = prob.get(t.strip().lower(), 0.0) + p
    exact = {w: prob.get(w, 0.0) for w in ("he", "she", "they")}
    tokens, weights = [t.strip().lower() for t, p in top], np.array([p for t, p in top])
    draws = np.random.default_rng(seed).choice(tokens, size=n, p=weights / weights.sum())
    counts = {w: int((draws == w).sum()) for w in ("he", "she", "they")}
    return exact, counts

for job in ("nurse", "engineer"):
    exact, counts = pronoun_probe(job)
    print(job, {w: round(p, 3) for w, p in exact.items()}, "in 20 samples:", counts)
```
   *Exercise 6.3:* run the probe for two other jobs of your own and with a different `seed`; how much do the 20-sample counts change, and what does that tell you about trusting a small test? Then write 3 sentences: the pattern you saw (state it as "this model, on this prompt"), why it might exist (what the model learned from), and what a team could do (more balanced data, an evaluation set that checks it, human review, or not using the model for decisions about people). You should see for the nurse prompt he 0.029, she 0.150, they 0.024, and for the engineer prompt he 0.069, she 0.008, they 0.017. ==Handle such outputs responsibly:== describe what you measured (one small model, a few prompts), never generalise to real people, do not pass around offensive outputs, and tell your teacher if you see something hurtful.

**6.4 Pre-mortem.** A pre-mortem imagines the project has already failed and asks why.
```python
premortem = [{"failure": "", "who_is_harmed": "", "principle": "", "fix": ""} for _ in range(3)]
```
   *Exercise 6.4:* for your product fill in three ways it could fail. For each one say who would be harmed, which of the five principles it touches (use exactly one of `fairness`, `transparency`, `privacy`, `safety`, `accountability`) and a fix the team could check. "Be more careful" is not a fix.

**6.5 Model card lite.** A model card tells people what a model is for and where it fails.
```python
model_card = {"intended_use": "", "not_intended_for": "", "known_limits": "", "how_it_was_tested": "", "contact": ""}
```
   *Exercise 6.5:* write five sentences, one for each entry. Use what you found in 6.2 to 6.4 in "known limits" and "how it was tested". For the contact write a role (for example "the school's IT teacher"), not a personal email or phone number.

7. *Part 7. Gather it up and reflect (10 min).* ==Collect your key numbers and writing into one printed summary, then reflect.==
   Store your writing in `RESULTS` (for example `RESULTS["p6_spec"] = spec`) and print everything:
```python
print("WEEK 1 SUMMARY")
for key, value in RESULTS.items():
    print(f"\n[{key}]")
    print(value)
print(f"\n{len(RESULTS)} items collected.")
reflection = {"1_surprised": "", "2_better_and_worse": "", "3_never_trust": "", "4_distribution_then_sample": ""}
```
   *Exercise 7.1:* answer the four questions in 2 or 3 sentences each, naming something you actually saw (a number, a picture, a sentence): (1) What surprised you most, and why? (2) Name one thing generative AI is better at than you expected and one thing it is worse at. (3) What would you never trust it with, and what would you need to see before you changed your mind? (4) The lesson says generative models "learn a distribution, then sample": pick something you saw and explain it with that idea.

### What you should see

- Step 4.1: the same seed always gives the same picture, a different seed gives a different one; each picture has shape (96, 96, 3).
- Step 4.2: ᾱ<sub>300</sub> = 0.3964 and ᾱ<sub>1000</sub> = 4.0e-5 (printed as 4.04e-05); also ᾱ<sub>50</sub> = 0.971, ᾱ<sub>150</sub> = 0.788 and ᾱ<sub>600</sub> = 0.026. The first step with ᾱ<sub>t</sub> below 0.5 is t = 260. The smiley is clear at t = 50, fading by t = 300 and just noise at t = 1000.
- Step 4.3 (GPU): 512×512 pictures; the vague prompt gives a generic picture, the specific prompt follows your style, light and colour words; four seeds give four different layouts of the same idea; guidance 3 drifts from the text and guidance 12 follows it hard.
- Step 5.1: plain sentence back as "The robot club meets Friday." (WER 0.167); idiom as French "Il pleut des chats et des chiens." and back as "It rains cats and dogs." (WER 0.333); the joke back unchanged (WER 0.0).
- Step 5.2: WER of 0.0 and 0.667 for the two example pairs; 0.25, 1.0 and 4.0 for exercise 5.2.
- Step 5.3: a few seconds of audio saved as `chain.wav`; in our run the GPT-2 sentence (about 5 seconds of speech) was heard back word for word (WER 0.00) and the hard sentence with a name, an abbreviation and numbers gave WER 0.86. Your sentence will differ because it is sampled.
- Step 6.2: the true answer in GPT-2's top 5 for 5 of 6 prompts (83%); "January" is missed.
- Step 6.3: nurse he 0.029, she 0.150, they 0.024; engineer he 0.069, she 0.008, they 0.017. The 20-sample counts (nurse he 0, she 5, they 2; engineer he 2, she 0, they 0 with seed 0) change when you change the seed.
- Part 7: a printed summary of every item stored in `RESULTS`. Small numerical differences are normal between library versions.

### Common mistakes

- Mixing up the step numbering in the diffusion schedule: the array `alpha_bars` starts at step 1, so ᾱ<sub>t</sub> is `alpha_bars[t - 1]`. Using `alpha_bars[t]` gives 0.3940 instead of 0.3964 at t = 300.
- Loading Stable Diffusion in full precision or keeping several pipelines in memory ("CUDA out of memory"). Use `torch.float16`, `del pipe`, then `gc.collect()` and `torch.cuda.empty_cache()`, and check `nvidia-smi`.
- Using a model ID that is not in the offline cache. If a Stable Diffusion, translation, speech or Whisper ID fails, ask your instructor which ID is cached rather than trying to download it.
- Comparing text with punctuation and capital letters when computing the word error rate. Clean both texts first (as `clean` does), or you count formatting as errors.
- Believing a round trip that looks unchanged. The middle step may be literal, as with "Il pleut des chats et des chiens"; read the French as well.
- Drawing big conclusions from one bias probe. It tests one small model on a few prompts; 20 samples are noisy, so report probabilities and say "this model, on this prompt".
- Treating a fluent answer as a true one. GPT-2 missed "January" and put "Paris" fifth for "The capital of France is".

### Deliverables

Submit the notebook with all outputs visible, the way your instructor asks. It must include: the noise art and diffusion pictures (4.1 and 4.2) with your answers, the Stable Diffusion images and your imagination journal (4.3), the translation, WER and audio-chain results with your notes (Part 5), your AI spec sheet card, fact table, bias probe note, pre-mortem and model card (Part 6), and the printed summary and four reflection answers (Part 7). The graded notebook is the one your instructor specifies.

**Stretch goals.** Try `en-de` and `en-es` round trips and compare their WER; compute a WER for a sentence in a language your community speaks, with a native speaker; add `top_p` or a different `smooth` and a fourth palette to `make_art`; change the noise schedule in 4.2 (for example β up to 0.05) and see how `t_half` moves; add five more factual prompts to 6.2 and plot the fraction found in the top 1, top 5 and top 20.


## Check your understanding {rail=Check}

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions, with an explanation after each one. Some need a calculation: have a calculator ready, and use the natural logarithm (ln) where a question mentions it.

```quiz
{
  "id": "week-01-check",
  "title": "Week 1 quiz: Introduction to Generative AI",
  "passMark": 70,
  "questions": [
    {
      "q": "Which statement best separates a generative model from a discriminative one?",
      "options": [
        "A generative model is always larger than a discriminative one.",
        "A discriminative model can only work with images.",
        "A generative model learns what data looks like so it can produce new data; a discriminative model labels or scores data it is given.",
        "A generative model only works when it is connected to the internet."
      ],
      "answer": 2,
      "why": "The difference is the question each answers: P(x), how likely is this data and can I sample more of it, versus P(y | x), which label fits this input. Size, data type and internet access are not what separates them."
    },
    {
      "q": "Which nesting of the four ideas is correct, outermost first?",
      "options": [
        "AI, machine learning, deep learning, generative AI",
        "Generative AI, deep learning, machine learning, AI",
        "Machine learning, AI, generative AI, deep learning",
        "Deep learning, AI, machine learning, generative AI"
      ],
      "answer": 0,
      "why": "AI is the broad field; machine learning is AI that learns from data; deep learning is machine learning with many-layered neural networks; generative AI is deep learning used to create new content."
    },
    {
      "q": "What does “learn a distribution, then sample” mean?",
      "options": [
        "Memorise every training example and print one at random.",
        "Sort the data into labels and then count the labels.",
        "Delete the data after training and then ask the user for new examples.",
        "Learn which items are likely in the data, then draw new items at random in line with those likelihoods."
      ],
      "answer": 3,
      "why": "Learning gives probabilities to possible outputs; sampling draws from them, so likely outputs come up often. Memorising and printing is copying, and sorting into labels is discriminative."
    },
    {
      "q": "A model gives the tokens of a three-token phrase the probabilities 0.5, 0.2 and 0.1 (each given the tokens before it). By the chain rule, what is the probability of the whole phrase?",
      "options": [
        "0.8",
        "0.01",
        "0.1",
        "0.001"
      ],
      "answer": 1,
      "why": "The chain rule multiplies: 0.5 × 0.2 × 0.1 = 0.01. Adding the numbers (0.8) is a common mistake; the product is always smaller than every factor."
    },
    {
      "q": "A four-token sentence gets probability 0.25 at each step. What is its perplexity?",
      "options": [
        "0.25",
        "16",
        "1",
        "4"
      ],
      "answer": 3,
      "why": "The sentence probability is 0.25⁴ = 0.00390625. Perplexity = P^(−1/n) = 0.00390625^(−1/4) = 4. It means the model was as unsure as choosing among four equally likely words each time."
    },
    {
      "q": "During training the model gives the true next word a probability of 0.25. What is the loss −ln p (rounded to two decimals)?",
      "options": [
        "1.39",
        "0.25",
        "2.00",
        "0.75"
      ],
      "answer": 0,
      "why": "−ln(0.25) = 1.386, which rounds to 1.39. The value 2.00 would be −log base 2, a different unit, and 0.75 is just 1 − 0.25."
    },
    {
      "q": "Three words have scores 4.2, 2.4 and 2.1. At temperature 0.5, what probability does the top word get (nearest value)?",
      "options": [
        "0.78",
        "0.50",
        "0.96",
        "0.33"
      ],
      "answer": 2,
      "why": "Divide the scores by 0.5 to get 8.4, 4.8 and 4.2, take e to each (4447, 122, 67) and divide by the total 4635: 4447 / 4635 = 0.96. At T = 1 the same word would get 0.78."
    },
    {
      "q": "You raise the temperature from 0.7 to 1.3 when generating text. What usually happens?",
      "options": [
        "The output becomes identical every time.",
        "The output becomes more varied and less coherent.",
        "The model becomes more truthful.",
        "The model runs much faster."
      ],
      "answer": 1,
      "why": "A higher temperature flattens the probabilities, so unlikely words get a larger chance. Text is more varied and less predictable. It does not make the model more truthful, and it does not change speed."
    },
    {
      "q": "The next-word probabilities, from best to worst, are 0.73, 0.12, 0.09, 0.04, 0.02. With top-p = 0.9, how many words are kept?",
      "options": [
        "3",
        "2",
        "4",
        "5"
      ],
      "answer": 0,
      "why": "The running totals are 0.73, 0.85 and 0.94. The smallest group whose total reaches at least 0.9 is the first three words. Two words only reach 0.85, which is below 0.9."
    },
    {
      "q": "Small GPT-2 is given “The capital of France is” and its top guesses are “ the”, “ now” and “ a”, not “Paris”. What does this show?",
      "options": [
        "The model is broken and cannot write sentences.",
        "Temperature was set too high.",
        "A model predicts likely words, not true ones, so it can be fluent and still wrong.",
        "GPT-2 always refuses to answer questions about countries."
      ],
      "answer": 2,
      "why": "Those guesses are all grammatical continuations, which is what the model is trained to produce. It has no built-in check on facts, so a fluent answer can be false. The probabilities do not depend on a temperature setting being high."
    },
    {
      "q": "How does a diffusion model make a new picture?",
      "options": [
        "It searches the internet and pastes together photos.",
        "It starts from random noise and removes a little noise at each step, guided by what it learned and your prompt.",
        "It draws one pixel at a time from left to right and never looks back.",
        "It enlarges a tiny stored thumbnail of every possible picture."
      ],
      "answer": 1,
      "why": "Diffusion reverses a noising process: begin with noise and clean it step by step. It does not copy stored photos or search the web."
    },
    {
      "q": "Which model takes audio as input and produces text as output?",
      "options": [
        "MMS-TTS",
        "Stable Diffusion",
        "MarianMT",
        "Whisper"
      ],
      "answer": 3,
      "why": "Whisper transcribes speech into text. MMS-TTS goes the other way (text to speech), Stable Diffusion makes images, and MarianMT translates text into text."
    },
    {
      "q": "What is true about ELIZA (1966)?",
      "options": [
        "It was the first large language model trained on the internet.",
        "It created images from text.",
        "It followed hand-written rules and did not learn from data, yet seemed to hold a conversation.",
        "It introduced the Transformer architecture."
      ],
      "answer": 2,
      "why": "ELIZA matched keywords with rules written by its author. The Transformer came in 2017, and learning language models from large text collections came decades later."
    },
    {
      "q": "GPT-3 has about 175 billion parameters and the largest GPT-2 about 1.5 billion. About how many times bigger is GPT-3?",
      "options": [
        "About 12 times",
        "About 117 times",
        "About 1,000 times",
        "About 2 times"
      ],
      "answer": 1,
      "why": "175 billion divided by 1.5 billion is about 117. And remember that bigger is not automatically better: Chinchilla, with 70 billion parameters, was trained on more text than larger models."
    },
    {
      "q": "A hiring tool gives women lower scores than men with the same experience. Which trustworthy-AI principle is mainly broken?",
      "options": [
        "Fairness",
        "Privacy",
        "Transparency",
        "Safety"
      ],
      "answer": 0,
      "why": "Equal people are treated differently because of gender, which is an unfair outcome. Real cases can touch several principles, but this one is mainly about fairness."
    }
  ]
}
```

## How Week 1 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: Be the model (guess the next word, probabilities, chain rule) | 15 | Own guesses and confidences recorded; table and chart correct; surprise and perplexity computed; clear written comparison of sure, unsure and wrong | Most steps run; comparison is general | Predictions missing or code not running |
| Part 2: Build a tiny generative model (n-grams) | 15 | n = 1, 2, 3 generated; copy rate and distinct words computed; own corpus used; reasons GPT-2 is better are specific | Models work; measurements or reasons incomplete | No working n-gram model |
| Part 3: The dials of imagination (temperature, top-k, top-p) | 15 | Softmax by hand matches; trade-off measured and plotted; settings for the three jobs justified | Calculations done; trade-off described loosely | Calculations missing or wrong |
| Part 4: Imagine in pictures (noise art, diffusion, Stable Diffusion) | 15 | Noise art explained; ᾱ values correct and `t_half` found; imagination journal specific (prompts, seeds, guidance); GPU freed | Steps run; journal or explanations general | Steps missing; journal empty |
| Part 5: Sound and languages (round trip, WER, audio chain) | 10 | Round trips with specific losses noted; WER function correct; chain run on two texts with two real reasons; voice plan concrete | Chain runs; reasons or notes vague | Chain broken or WER missing |
| Part 6: AI product and trust (spec, facts, bias, pre-mortem, model card) | 20 | Complete spec sheet; fact table with explanation; bias probe described carefully with a plan to reduce it; three realistic risks with fixes; five-sentence model card | Everything present; fixes or limits general | Parts missing; bias note careless or overstated |
| Part 7: Gather and reflect | 10 | Summary complete; four specific answers that cite results from the lab | Summary printed; answers general | Summary or reflection missing |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).


## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave (your instructor will use them to open Week 2):
1. What is the difference between a generative and a discriminative model? Give one example of each.
2. Explain “learn a distribution, then sample” using the canteen chef or the counting model.
3. Name one risk of generative AI and say which trustworthy-AI principle it touches and one way to reduce it.

**Homework / self-study**
- Read the three Module 1 articles: *What is Generative AI and How Does it Work*, *What is Machine Learning and Why Does It Matter*, and *What Is Trustworthy AI*. Write a half-page summary of the trustworthy-AI article.
- Finish any unfinished lab exercises and the reflection.
- Preview Week 2: revise NumPy arrays and functions from the Python course.
- There is no DLI course milestone this week (the first starts in Week 9).
