> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 11 — Multimodal AI

**Purpose:** Learn how one family of models can work with pictures and words together: ==a Vision Transformer cuts an image into patches and treats them as tokens, position embeddings tell the model where each patch came from, and CLIP trains pictures and sentences to share one space== so that they can be compared.

**Term:** Pre-final (Weeks 11–15) · **Month:** 3 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a model see and read",
  "caption": "The caption and the match scores are made up to show the idea; a real model gives different words and numbers. The patch count and the attention counts are exact arithmetic that you will redo in this lesson.",
  "scenes": [
    { "name": "A picture, a sentence", "kind": "text", "outputLabel": "A caption written by a model (illustrative)", "prompt": "Photo: a blue jeepney with the signboard “Divisoria via Recto” stopped at a corner, passengers boarding at the back.", "output": "A blue jeepney stopped at a street corner while passengers climb in at the back. The signboard says Divisoria via Recto. (This sentence is hand-written to show the idea; a real captioning model might say it differently, or get a detail wrong.)" },
    { "name": "Pieces of a picture", "kind": "tokens", "outputLabel": "A 224 × 224 colour photo cut into 16 × 16 squares (computed)", "prompt": "A Transformer reads tokens, so cut the photo into squares: 224 ÷ 16 = 14 squares across and 14 down, and 14 × 14 = 196.", "tokens": ["patch 1", "patch 2", "patch 3", "patch 4", "patch 5", "…", "patch 195", "patch 196"] },
    { "name": "Which sentence fits?", "kind": "bars", "outputLabel": "Match scores for the jeepney photo (illustrative)", "prompt": "Photo of a jeepney. Which of these four sentences fits it best?", "bars": [ { "label": "a photo of a jeepney", "value": 0.86 }, { "label": "a photo of a tricycle", "value": 0.1 }, { "label": "a photo of a carabao", "value": 0.03 }, { "label": "a photo of a mango", "value": 0.01 } ] },
    { "name": "Why patches?", "kind": "text", "outputLabel": "Computed: attention scores per head in one layer", "prompt": "Self-attention compares every token with every other token. How many comparisons for one 224 × 224 photo?", "output": "With 197 tokens (196 patches plus 1 CLS token) there are 197 × 197 = 38,809 scores. With one token per pixel there would be 50,176 tokens and 50,176 × 50,176 = 2,517,630,976 scores: about 64,872 times more. Patches make a Transformer on images affordable." }
  ]
}
```

## By the end, you can…

1. ==Define a modality and a multimodal model== and give three real applications, each with its input and output modalities.
2. ==Compute the number of patch tokens, the length of one patch vector and the patch-embedding parameters== of a Vision Transformer (ViT) from the image size, the patch size and the hidden size.
3. ==Explain why a bag of patches needs position embeddings==, using the idea that self-attention alone does not know the order of its inputs.
4. ==Trace an image through a ViT==: patchify, linear projection, CLS token, position embeddings, Transformer encoder and head.
5. ==Compute a cosine-similarity matrix, zero-shot probabilities and the contrastive loss== for a small batch of image–text pairs, in the style of CLIP.
6. ==Name three ways of combining modalities== (a shared space, a projector into a language model, cross-attention) and one risk of each.

Assessed in the Week 11 lab (“Imagine a model that sees and reads”) and the quiz. Read each section, then play with the figure under it: the figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. The lab refers back to this lesson: look for the pointers such as “Lab Part 2”.

## What is a multimodal model? {rail=Multimodal}

{lede} A [[modality|One kind of data, such as text, images, audio, video or sensor readings. Each kind is stored as numbers in its own shape.]] is one kind of data, and a [[multimodal model|A model that takes in, produces or links more than one modality, for example pictures and text together.]] is ==a model that takes in, produces or links more than one kind of data==. Until now in this course you have met models that read text (Weeks 6 to 9) or small pictures (Week 4). This week they meet.

### Every modality is numbers in a shape

A computer never sees a word, a face or a song. ==It sees numbers.== What makes one modality different from another is ==the shape of those numbers==, and the shape decides which tools fit. You already know two shapes.

- **Text** is a list of integer token ids (Week 6). A 12-token sentence is 12 integers, one after the other.
- **An image** is a [[tensor|A grid of numbers with any number of dimensions: a list is a 1-D tensor, a table is 2-D, and a colour image is 3-D.]] of shape (C, H, W): C colour channels, H rows and W columns of pixels. A 224 × 224 colour photo has C = 3 (red, green, blue), so its shape is (3, 224, 224).

Two shapes are new today. **Audio** is a long list of measurements of air pressure, called samples, taken many times per second. The number of samples per second is the [[sampling rate|How many measurements of a sound are taken each second, in hertz (Hz). Many speech models use 16,000 Hz.]]: speech models commonly use 16,000 samples per second, so one second of audio is a list of 16,000 numbers. **Video** is a stack of images over time, called frames, so it is a tensor with one more dimension.

| Modality | How it is stored | One example | Numbers in it |
|---|---|---|---|
| Text | a list of token ids | a 12-token sentence | 12 |
| Image | a tensor (C, H, W) | a 224 × 224 colour photo, shape (3, 224, 224) | 150,528 |
| Audio | a list of samples | 1 second at 16,000 samples per second | 16,000 |
| Audio | a list of samples | a 30-second clip at the same rate | 480,000 |
| Video | a stack of frames | 30 frames of 3 × 224 × 224 (one second at 30 frames per second) | 4,515,840 |
| Sensor readings | a short list per time step | a 3-axis accelerometer read 50 times per second | 150 per second |

Notice how different the sizes are. A sentence is a dozen numbers and one second of video is over four million. This is the first practical lesson of multimodal work: ==the modalities do not arrive in the same shape or the same size, so a model must turn each of them into a common currency== before it can combine them. That common currency, as you will see, is a sequence of vectors, which a Transformer can read.

### How many numbers is a picture?

Here is the formula behind the image row of the table. Select each part to read what it means, then follow the worked example.

```viz
{
  "type": "equation",
  "title": "The number of values in an image",
  "intro": "Select a part of the formula to read what it means, then read the worked example below it.",
  "formula": [ { "t": "n", "id": "n" }, " = ", { "t": "C", "id": "c" }, " × ", { "t": "H", "id": "h" }, " × ", { "t": "W", "id": "w" } ],
  "parts": [
    { "id": "n", "name": "n (how many numbers)", "meaning": "The total count of numbers the computer stores for the picture." },
    { "id": "c", "name": "C (channels)", "meaning": "How many numbers describe each pixel: 3 for a colour photo (red, green, blue), 1 for a grey picture." },
    { "id": "h", "name": "H (height)", "meaning": "The number of rows of pixels." },
    { "id": "w", "name": "W (width)", "meaning": "The number of columns of pixels." }
  ],
  "steps": [
    { "title": "A small grey picture", "expr": "n = 1 × 16 × 16 = 256", "note": "The 16 × 16 grey pictures you will draw in the lab hold 256 numbers each." },
    { "title": "A model-sized colour photo", "expr": "n = 3 × 224 × 224 = 150,528", "note": "The size most vision models take: 224 rows, 224 columns, 3 colours." },
    { "title": "A phone photo", "expr": "n = 3 × 4000 × 3000 = 36,000,000", "note": "A 12-megapixel photo has 36 million numbers, which is why images are usually shrunk before a model sees them." }
  ]
}
```

==A 224 × 224 colour photo has 150,528 numbers==, which is already about 12,500 times more than a 12-token sentence. Keep that number in mind: in the next section, cutting the picture into patches will turn out to keep all 150,528 numbers and only rearrange them. In Lab Part 1 you will build this table in code and check it with `assert`.

### Three real applications, and many more

A multimodal model is defined by what goes in and what comes out. Three applications show the range.

1. **Image captioning.** A picture goes in and a sentence comes out. A school can use it to draft alt text, the short description that a screen reader speaks, so that a blind student can follow the photos in a newsletter. The risk is a fluent sentence that mentions something that is not in the picture, so ==a person should read it before it is published==.
2. **Visual question answering (VQA).** A picture and a question go in and an answer comes out. You photograph a plant and ask “Is this safe to eat?” A model may give a confident answer, but ==the answer must be checked by an expert== before anyone eats anything: a wrong “yes” can hurt someone.
3. **Speech recognition.** Audio goes in and text comes out. Whisper (Radford et al., 2022), which you met in Week 1, turns recorded speech into written words. It can mishear names, places and unusual accents, so important transcripts need a human check.

Other applications fit the same pattern. **Text-to-image** (a sentence in, a new picture out) is the subject of Weeks 12 and 13. **Document understanding** reads text printed in a scanned form, and the older name for the reading step is [[OCR|Optical character recognition: turning the pixels of printed or handwritten text into characters the computer can store as text.]]. A **medical image plus report** system helps a doctor by pairing a scan with a draft note, and a **robot** can follow an instruction such as “bring the red cup” by combining its camera, its sensors and a sentence. For the last two, errors can have serious consequences, so they are always used with human oversight.

Sort the cards below. Each one asks what the input is and what the output is. One card is an edge case on purpose.

```viz
{
  "type": "sorter",
  "title": "What goes in and what comes out?",
  "intro": "Drag or tap each task into the box that matches its input and output.",
  "bins": [
    { "id": "t2t", "label": "text → text" },
    { "id": "i2t", "label": "image → text" },
    { "id": "t2i", "label": "text → image" },
    { "id": "a2t", "label": "audio → text" },
    { "id": "it2t", "label": "image + text → text" }
  ],
  "cards": [
    { "name": "Write alt text for a newsletter photo", "answer": "i2t", "why": "Only the picture goes in and a sentence comes out. This is image captioning." },
    { "name": "Transcribe a recorded class discussion", "answer": "a2t", "why": "Audio samples go in and written words come out. This is speech recognition, as in Whisper." },
    { "name": "Draw a poster from the slogan “Clean water for all”", "answer": "t2i", "why": "A text prompt goes in and a brand-new picture comes out. You will study how in Weeks 12 and 13." },
    { "name": "Answer “How many chairs are in this photo?”", "answer": "it2t", "why": "The model needs both the picture and the question, and it answers in words. This is visual question answering." },
    { "name": "Translate a sentence from English to Filipino", "answer": "t2t", "why": "Text goes in and text comes out, so this is not multimodal." },
    { "name": "Summarise a long news article", "answer": "t2t", "why": "Both sides are text. A language model from Week 9 is enough." },
    { "name": "Read the text on a scanned enrolment form (OCR)", "answer": "i2t", "why": "A scan is an image, and the output is the characters on it as text." },
    { "name": "Explain what a bar chart shows", "answer": "it2t", "why": "The chart is an image and the request is text. The answer is text, so a vision-language model must read both." },
    { "name": "Describe the one abnormal region in a chest X-ray (draft for a doctor)", "answer": "i2t", "why": "An image goes in and a draft description comes out. A doctor must check it before it is used." },
    { "name": "Pick the best caption for a photo from a list of four (CLIP)", "answer": "either", "why": "Edge case. The model takes an image and candidate sentences and returns scores, not new text. It is multimodal either way, so image → text or image + text → text are both defensible if you can explain why." }
  ]
}
```

Notice how many boxes involve a picture and how many involve a sentence. Most multimodal applications connect the two, and that is why this week focuses on images and text.

### Why use more than one modality?

The modalities are ==complementary==: each one shows what the other leaves out. A photo of a flooded street shows how high the water reaches, but not the barangay or the time. A text report says “Barangay San Roque, 6 p.m.” but not whether the water is at ankle or waist height. Together they say much more than either alone.

The tabs below put three kinds of model in front of the same question. The outputs are hand-written to show the idea, not produced by a real model.

```viz
{
  "type": "flow",
  "title": "One question, three kinds of model",
  "modes": [
    {
      "name": "Text only",
      "tagline": "It can read the question but not see the picture.",
      "inputLabel": "Question",
      "modelLabel": "language model",
      "outputLabel": "Answer",
      "examples": [
        { "label": "Jeepney", "input": "Which jeepney goes to Divisoria? (the photo of the signboard is not available to the model)", "output": "I cannot tell without seeing the signboard. Jeepneys to Divisoria usually say so on the front." },
        { "label": "Flood", "input": "Is the water above the knee? (the photo is not available)", "output": "I cannot see the water, so I cannot say." }
      ],
      "note": "Hand-written, illustrative outputs. A real language model might guess instead of admitting it cannot see."
    },
    {
      "name": "Image only",
      "tagline": "It sees the picture but cannot take a question.",
      "inputLabel": "Photo",
      "modelLabel": "image classifier",
      "outputLabel": "Label",
      "examples": [
        { "label": "Jeepney", "input": "Photo of a blue jeepney with a signboard", "output": "label: jeepney (0.97)" },
        { "label": "Flood", "input": "Photo of a flooded street with a gauge", "output": "label: flood (0.92)" }
      ],
      "note": "Hand-written, illustrative outputs. A fixed classifier answers only the question it was trained for: it cannot read the signboard or the gauge."
    },
    {
      "name": "Image + text",
      "tagline": "It reads the question and looks at the picture together.",
      "inputLabel": "Photo + question",
      "modelLabel": "vision-language model",
      "outputLabel": "Answer",
      "examples": [
        { "label": "Jeepney", "input": "Photo of a jeepney + “Which jeepney goes to Divisoria?”", "output": "This one: its signboard reads “Divisoria via Recto”." },
        { "label": "Flood", "input": "Photo of a flooded street with a gauge + “Is the water above the knee?”", "output": "The water reaches about the middle of the gauge, which looks like knee height. Please confirm on site." }
      ],
      "note": "Hand-written, illustrative outputs. A real system can misread a sign or a gauge, so a person must check anything that matters."
    }
  ]
}
```

Switch between the tabs for the same question. Only the third one can both look and answer, but notice that it also adds the new risk: it can sound sure while being wrong. You will keep coming back to this tension between power and care.

### Where multimodal models sit in the family

==Multimodal models are not a separate invention.== They are the Transformers of Weeks 7 to 9, extended so that some of the tokens come from pictures. Click each ring to see how the idea nests.

```viz
{
  "type": "nested",
  "title": "Where multimodal models sit",
  "levels": [
    { "name": "Machine learning", "blurb": "Models that learn patterns from data instead of following hand-written rules (Week 1).", "examples": ["linear models", "decision trees", "neural networks"] },
    { "name": "Deep learning", "blurb": "Machine learning with many layers of neurons trained by backpropagation (Weeks 2 and 3).", "examples": ["CNNs for images (Week 4)", "RNNs and LSTMs (Week 7)"] },
    { "name": "Transformers", "blurb": "Deep networks built from attention blocks that read a sequence of vectors (Weeks 7 and 8).", "examples": ["BERT", "GPT-style language models", "T5"] },
    { "name": "Multimodal Transformers", "blurb": "Transformers in which some or all of the input vectors come from other modalities, such as image patches or audio frames.", "examples": ["Vision Transformer (ViT)", "CLIP", "Whisper"] },
    { "name": "Vision-language assistants", "blurb": "Chat models that can take a picture and a question together and answer in text.", "examples": ["LLaVA-style models", "Flamingo-style models"] }
  ]
}
```

The inner rings are special cases of the outer ones. A vision-language assistant is a language model (Week 9) that has been taught to read some extra tokens made from a picture.

### Three ways to combine modalities

How do you actually make a model look and read at once? There are a handful of patterns, and the rest of the lesson builds the first three. Switch tabs to compare them.

```viz
{
  "type": "fusion",
  "title": "How a model can combine modalities",
  "intro": "Switch tabs and read the chips from left to right: what goes in, what happens in the middle, what comes out.",
  "patterns": [
    {
      "name": "Two encoders, one space (CLIP)",
      "inputs": ["image", "text"],
      "stages": ["image encoder", "text encoder", "compare the vectors"],
      "output": "similarity score",
      "how": "Each modality has its own encoder. Both encoders map into the same vector space, and training pulls a picture and its caption close together. You then compare vectors with cosine similarity (Week 6).",
      "examples": "CLIP; image search with a sentence; zero-shot classification.",
      "good": "Simple, fast, and it needs no training on your own labels. Great for search and matching.",
      "bad": "It returns scores, not sentences, so it cannot explain or answer a question. It inherits the biases of its image–caption data."
    },
    {
      "name": "Vision encoder + projector + language model (LLaVA-style)",
      "inputs": ["image", "text prompt"],
      "stages": ["vision encoder", "projector", "language model"],
      "output": "text answer",
      "how": "A vision encoder turns the image into patch features. A small learned layer, the projector, maps them into the language model's embedding space. They are placed in front of the text tokens, and the language model writes the answer as it always does (Liu et al., 2023).",
      "examples": "Chat assistants that answer questions about a photo.",
      "good": "Reuses a strong language model and a strong vision encoder, so only a small piece needs new training.",
      "bad": "Each image costs hundreds of tokens of context, and the model can describe things that are not there (hallucination)."
    },
    {
      "name": "Cross-attention inside a language model (Flamingo-style)",
      "inputs": ["image", "text"],
      "stages": ["vision encoder", "cross-attention layers", "language model"],
      "output": "text answer",
      "how": "New layers are inserted into the language model. In them the text tokens are the queries and the image features are the keys and values, so the text can look at the picture whenever it needs to (Alayrac et al., 2022).",
      "examples": "Few-shot captioning and question answering over images and interleaved text.",
      "good": "The image does not use up context tokens, and the language model can stay frozen.",
      "bad": "More new parts to train, and the design is more complex than a projector."
    },
    {
      "name": "Generate an image from text (diffusion)",
      "inputs": ["text prompt"],
      "stages": ["text encoder", "denoising network", "decoder"],
      "output": "new image",
      "how": "A text encoder turns the prompt into a vector, and a network turns random noise into a picture step by step, guided by that vector. This is the subject of Weeks 12 and 13.",
      "examples": "Stable Diffusion and similar text-to-image systems.",
      "good": "Creates new pictures from a sentence.",
      "bad": "May copy styles or biases from its training data and can produce convincing fakes."
    }
  ],
  "note": "A simplified overview. Real systems add many details, and new designs appear every year."
}
```

Notice that the first three differ in where the two modalities meet: at the end (compare vectors), at the start of a language model (the projector places the picture tokens in front), or in the middle (cross-attention). Sections 4 and 5 teach the first two in detail, and the lab's Part 6 builds a miniature version of the first.

### Early fusion and late fusion

There is an older and simpler way to describe the choice. In [[early fusion|Combining the modalities near the input, for example by joining their features into one vector before a single model makes the decision.]] you glue the features of the modalities together first and let one model learn from the combination. In [[late fusion|Letting a separate model handle each modality and combining only their decisions, for example by multiplying or averaging their probabilities.]] each modality gets its own model and you combine the decisions at the end. Neither is always better. ==Early fusion lets the model discover how the clues interact==, and late fusion is simple and easy to debug when one modality is missing.

Here is a tiny example from Lab Part 1. A picture shows one coloured shape, and the aim is to name both its colour and its shape: nine possible captions (3 colours × 3 shapes). A colour model alone cannot tell a red circle from a red square, and a shape model alone cannot tell red from blue. Suppose the colour model says red 0.7, green 0.2, blue 0.1 and the shape model says circle 0.5, square 0.3, cross 0.2. Step through how late fusion combines them.

```viz
{
  "type": "pipeline",
  "title": "Late fusion by multiplying decisions",
  "intro": "Press the step buttons to follow the picture from two separate models to one answer.",
  "nodes": [
    { "id": "pic", "label": "Picture", "sub": "one red shape" },
    { "id": "col", "label": "Colour model", "sub": "3 probabilities" },
    { "id": "shp", "label": "Shape model", "sub": "3 probabilities" },
    { "id": "mul", "label": "Multiply", "sub": "3 × 3 = 9 scores" },
    { "id": "ans", "label": "Best caption", "sub": "arg max" }
  ],
  "steps": [
    { "title": "Two views of one picture", "body": "The picture goes to a colour model and to a shape model. Each one is blind to the other's job, so each alone is ambiguous.", "active": ["pic", "col", "shp"], "example": "Colour: red 0.7, green 0.2, blue 0.1. Shape: circle 0.5, square 0.3, cross 0.2." },
    { "title": "Multiply the probabilities", "body": "If the colour and the shape are judged independently, the chance of a pair is the product of the two chances. This gives one score for each of the nine captions.", "active": ["col", "shp", "mul"], "example": "red circle: 0.7 × 0.5 = 0.35. red square: 0.7 × 0.3 = 0.21. green circle: 0.2 × 0.5 = 0.10." },
    { "title": "Pick the highest", "body": "The nine products add up to 1 because each set of three probabilities adds up to 1. The caption with the highest product is the fused answer.", "active": ["mul", "ans"], "example": "Best: red circle at 0.35. Smallest: blue cross at 0.1 × 0.2 = 0.02." }
  ]
}
```

The heat map shows all nine products. The picture is only as good as its two models: ==a late-fusion answer can be no more reliable than its weaker half==.

```viz
{
  "type": "heatmap",
  "title": "Nine caption scores from two small models",
  "corner": "colour ↓ shape →",
  "rows": ["red", "green", "blue"],
  "cols": ["circle", "square", "cross"],
  "showValues": true,
  "rowsSumToOne": false,
  "W": [ [0.35, 0.21, 0.14], [0.10, 0.06, 0.04], [0.05, 0.03, 0.02] ],
  "note": "Computed from the two illustrative probability lists in the text (colour 0.7, 0.2, 0.1 and shape 0.5, 0.3, 0.2); a real colour model and shape model would give their own numbers."
}
```

Look for the single brightest cell: that is the fused caption. In Lab Part 1.4 and 1.5 you will train both an early-fusion and a late-fusion classifier and compare their accuracy.

Counting parameters shows another difference. With five input features (three colour features and two shape features) and nine captions, one early-fusion linear layer has 5 × 9 + 9 = 54 parameters. Two separate late-fusion layers (three colour features to three colours, two shape features to three shapes) have (3 × 3 + 3) + (2 × 3 + 3) = 12 + 9 = 21 parameters. Early fusion is more flexible, and late fusion is smaller and easier to inspect.

### A short history

The ideas of this week arrived within a few years of each other. Select an event to read what it is known for.

```viz
{
  "type": "timeline",
  "title": "Pictures and words meet",
  "events": [
    { "year": "2020", "title": "Vision Transformer (ViT)", "text": "Dosovitskiy et al. showed that a standard Transformer can classify images when the image is cut into 16 × 16 patches, in a paper titled “An image is worth 16×16 words”." },
    { "year": "2021", "title": "CLIP", "text": "Radford et al. trained an image encoder and a text encoder on about 400 million image–text pairs from the internet so that pictures and sentences share one space, which allows zero-shot classification." },
    { "year": "2022", "title": "Flamingo", "text": "Alayrac et al. inserted cross-attention layers into a pretrained language model so that it could answer questions about images from only a few examples." },
    { "year": "2022", "title": "Whisper", "text": "Radford et al. released a speech recognition model trained on a large amount of audio with transcripts, which transcribes speech in many languages." },
    { "year": "2023", "title": "LLaVA", "text": "Liu et al. connected a CLIP vision encoder to a language model with a small projection and trained it on instructions about pictures, showing that a chat model can be taught to see with a modest amount of new training." }
  ]
}
```

Notice the pattern: ==the vision part (ViT, CLIP) came first== and the language-model part (Flamingo, LLaVA) was built on top of it. That is also the order of this lesson.

## From pixels to tokens: the Vision Transformer {rail=Vision Transformer}

{lede} The Transformer of Week 8 reads a sequence of vectors, and a photo is not a sequence: it is a grid. A [[Vision Transformer (ViT)|A Transformer for images. It cuts the image into fixed-size square patches, turns each patch into a vector, and reads the sequence of patch vectors like a sentence of tokens.]] solves this with one simple move: ==cut the image into patches and treat every patch as one token==. The title of the original paper says it in five words: “An image is worth 16×16 words” (Dosovitskiy et al., 2020).

### Why not one token per pixel?

The most direct idea would be to give the Transformer one token for every pixel, like one token for every word. Week 7 tells you why this does not work. Self-attention compares every token with every other token, so for n tokens it computes n × n scores. A 224 × 224 image has 50,176 pixels, so one attention head in one layer would compute 50,176 × 50,176 = 2,517,630,976 scores: about 2.5 billion, and for a single image. The memory and the time would be far too large.

Select each part of the formula to read it, then follow the worked example for patches and for pixels.

```viz
{
  "type": "equation",
  "title": "Attention cost grows with the square of the tokens",
  "intro": "Select a part of the formula, then read the three worked examples.",
  "formula": [ { "t": "scores", "id": "scores" }, " = ", { "t": "n", "id": "n" }, " × ", { "t": "n", "id": "n2" } ],
  "parts": [
    { "id": "scores", "name": "scores (per head, per layer)", "meaning": "How many query–key comparisons one attention head makes in one layer for one input." },
    { "id": "n", "name": "n (the first n)", "meaning": "The number of tokens, which are the queries: each token asks one question of the others." },
    { "id": "n2", "name": "n (the second n)", "meaning": "The number of tokens again, which are the keys: every query is compared with every key." }
  ],
  "steps": [
    { "title": "Patch tokens (16 × 16 patches)", "expr": "n = 196 + 1 = 197, so scores = 197 × 197 = 38,809", "note": "196 patches plus 1 CLS token (explained below)." },
    { "title": "Pixel tokens", "expr": "n = 224 × 224 = 50,176, so scores = 50,176 × 50,176 = 2,517,630,976", "note": "More than 2.5 billion scores." },
    { "title": "The ratio", "expr": "2,517,630,976 ÷ 38,809 ≈ 64,872", "note": "Patches are about 64,872 times cheaper for the attention step alone. In Lab Part 2.4 you will compute this in code and plot it." }
  ]
}
```

==Patches shrink the sequence from 50,176 tokens to 197==, and that is the entire reason ViT works at a sensible cost. See it in three tokenisations of the same picture.

```viz
{
  "type": "flow",
  "title": "How big is a token?",
  "modes": [
    {
      "name": "One pixel",
      "tagline": "The smallest possible token: one pixel with its 3 colour numbers.",
      "inputLabel": "Image",
      "modelLabel": "tokeniser",
      "outputLabel": "Sequence",
      "examples": [
        { "label": "224 × 224 photo", "input": "224 × 224 pixels, 3 colours each", "output": "50,176 tokens of 3 numbers; 2,517,630,976 attention scores per head" }
      ],
      "note": "Exact arithmetic. Far too many tokens for plain self-attention, and one pixel says almost nothing alone."
    },
    {
      "name": "One 16 × 16 patch",
      "tagline": "A small square of the picture: enough detail to mean something, few enough tokens to be cheap.",
      "inputLabel": "Image",
      "modelLabel": "tokeniser (patchify)",
      "outputLabel": "Sequence",
      "examples": [
        { "label": "224 × 224 photo", "input": "224 × 224 pixels, cut into 16 × 16 squares", "output": "196 tokens of 768 numbers; 38,809 scores per head with the CLS token" }
      ],
      "note": "Exact arithmetic. This is the choice made by ViT-Base with 16 × 16 patches."
    },
    {
      "name": "One 32 × 32 patch",
      "tagline": "Bigger squares: fewer tokens, but each one hides more detail.",
      "inputLabel": "Image",
      "modelLabel": "tokeniser (patchify)",
      "outputLabel": "Sequence",
      "examples": [
        { "label": "224 × 224 photo", "input": "224 × 224 pixels, cut into 32 × 32 squares", "output": "49 tokens of 3,072 numbers; 2,500 scores per head with the CLS token" }
      ],
      "note": "Exact arithmetic. Cheaper still, but small objects can fall inside one patch and be lost."
    }
  ]
}
```

Switch between the three tabs and compare the token counts. A patch is a middle path between a pixel (too small) and the whole image (too big).

### Why not just use a CNN?

You studied convolutional networks in Week 4, and they are excellent at images: a small filter slides over the picture and detects edges, then shapes. The ViT is not a claim that CNNs are bad. It is an experiment that asks a different question: can the *same* Transformer that reads text also read images, with almost no change? The answer was yes, and that is useful because one architecture, one set of tools and one way of scaling can then serve several modalities. Later in this section you will see the price: a ViT has to learn some things that a CNN gets for free.

### Step 1: patchify

To [[patchify|To cut an image into a grid of equal, non-overlapping squares (patches) and flatten each square into a vector of numbers.]] an image, lay a grid over it like tiles on a floor. Each tile is a [[patch|A small square piece of an image, for example 16 × 16 pixels, that becomes one token for the Transformer.]]. Each patch is then **flattened**: its rows are laid end to end so that a P × P × C block becomes a single vector of P · P · C numbers (P is the patch side and C the number of channels).

Try it on a small picture. Change the patch size and watch how the token count moves.

```viz
{
  "type": "patch",
  "title": "Cut a tiny picture into patch tokens",
  "image": [
    [0,9,9,9,9,9,9,0],
    [0,0,0,0,0,9,9,0],
    [0,0,0,0,9,9,0,0],
    [0,0,0,9,9,0,0,0],
    [0,0,9,9,0,0,0,0],
    [0,0,9,9,0,0,0,0],
    [0,0,9,9,0,0,0,0],
    [0,0,0,0,0,0,0,0]
  ],
  "channels": 1,
  "patchSizes": [2, 4, 8]
}
```

With 2 × 2 patches the 8 × 8 picture gives 16 tokens of 4 numbers. With 4 × 4 patches it gives 4 tokens of 16 numbers, and with 8 × 8 patches it gives a single token that holds everything. In every case tokens times numbers per token equals 64: ==cutting into patches loses nothing, it only reorganises the numbers==. In Lab Part 2.1 you will write `patchify` yourself with `reshape` and `transpose`, and test it on a 4 × 4 picture of the numbers 0 to 15.

Now the two formulas that tell you the size of the result. The first counts the patches.

```viz
{
  "type": "equation",
  "title": "How many patches?",
  "intro": "Select a part to read it, then follow the three worked examples for a 224 × 224 image.",
  "formula": [ { "t": "N", "id": "N" }, " = ", "(", { "t": "H", "id": "H" }, " ÷ ", { "t": "P", "id": "P" }, ") × (", { "t": "W", "id": "W" }, " ÷ ", { "t": "P", "id": "P2" }, ") = ", { "t": "H·W", "id": "HW" }, " ÷ ", { "t": "P²", "id": "P3" } ],
  "parts": [
    { "id": "N", "name": "N (number of patches)", "meaning": "How many patch tokens the image becomes, before the CLS token is added." },
    { "id": "H", "name": "H (image height)", "meaning": "The number of pixel rows in the image." },
    { "id": "P", "name": "P (patch side)", "meaning": "The side of one square patch in pixels, for example 16. It must divide H exactly." },
    { "id": "W", "name": "W (image width)", "meaning": "The number of pixel columns in the image." },
    { "id": "P2", "name": "P (patch side, again)", "meaning": "The same patch side, applied to the width." },
    { "id": "HW", "name": "H·W (pixels)", "meaning": "The total number of pixels in one channel of the image." },
    { "id": "P3", "name": "P² (pixels per patch)", "meaning": "How many pixels sit in one patch in one channel." }
  ],
  "steps": [
    { "title": "16 × 16 patches", "expr": "224 ÷ 16 = 14, so N = 14 × 14 = 196", "note": "A 14 by 14 grid of patches." },
    { "title": "32 × 32 patches", "expr": "224 ÷ 32 = 7, so N = 7 × 7 = 49", "note": "Larger patches give fewer tokens." },
    { "title": "8 × 8 patches", "expr": "224 ÷ 8 = 28, so N = 28 × 28 = 784", "note": "Smaller patches give four times as many tokens as 16 × 16." }
  ]
}
```

==Halve the patch side and N becomes four times larger==, because the grid grows in both directions. The second formula tells you how long each flattened patch is.

```viz
{
  "type": "equation",
  "title": "How long is one patch vector?",
  "intro": "Select a part to read it, then follow the worked examples.",
  "formula": [ { "t": "length", "id": "len" }, " = ", { "t": "P", "id": "p" }, " × ", { "t": "P", "id": "p2" }, " × ", { "t": "C", "id": "c" } ],
  "parts": [
    { "id": "len", "name": "length (numbers in one patch)", "meaning": "How many numbers one flattened patch contains." },
    { "id": "p", "name": "P (patch rows)", "meaning": "The number of pixel rows in the patch." },
    { "id": "p2", "name": "P (patch columns)", "meaning": "The number of pixel columns in the patch." },
    { "id": "c", "name": "C (channels)", "meaning": "3 for a colour image, 1 for a grey one." }
  ],
  "steps": [
    { "title": "16 × 16 colour patch", "expr": "16 × 16 × 3 = 768", "note": "Check: 196 patches × 768 numbers = 150,528 = 3 × 224 × 224, so nothing was lost." },
    { "title": "32 × 32 colour patch", "expr": "32 × 32 × 3 = 3,072", "note": "49 patches × 3,072 numbers = 150,528 again." },
    { "title": "8 × 8 colour patch", "expr": "8 × 8 × 3 = 192", "note": "784 patches × 192 numbers = 150,528 again." }
  ]
}
```

Let the calculator do the rest. Slide the image side and the patch size, switch between colour and grey, and compare the patch grid, the token count with the CLS token, and the attention counts against one-token-per-pixel.

```viz
{ "type": "calc", "kind": "vit_tokens", "title": "Token calculator for a Vision Transformer", "d": 768 }
```

Try 224 and 16, then 224 and 32, then 224 and 8. Your numbers should match this table, which is what the lab asks you to reproduce in `vit_counts` (Lab Part 2.3).

| Patch side P | Patches N | Numbers per patch | Sequence with CLS | Patch-embedding parameters (d = 768) |
|---|---|---|---|---|
| 32 | 49 | 3,072 | 50 | 2,360,064 |
| 16 | 196 | 768 | 197 | 590,592 |
| 8 | 784 | 192 | 785 | 148,224 |

### Step 2: the linear projection

A Transformer works with vectors of a fixed length d, called the [[hidden size|The length of the vector that represents each token inside the Transformer, also called the embedding dimension. ViT-Base uses 768.]] (768 in ViT-Base). A flattened patch has length P · P · C, which is usually not d, so the next step is a **linear projection**: one shared weight matrix E that maps every patch vector to a vector of length d. It is exactly the same operation as the embedding lookup of Week 6, except that a word is looked up in a table while a patch is multiplied by a matrix.

```viz
{
  "type": "equation",
  "title": "The linear projection of a patch",
  "intro": "Select a part to read it, then follow the worked example with a tiny 2 × 2 grey patch.",
  "formula": [ { "t": "z", "id": "z" }, " = ", { "t": "x", "id": "x" }, " · ", { "t": "E", "id": "E" }, " + ", { "t": "b", "id": "b" } ],
  "parts": [
    { "id": "z", "name": "z (patch embedding)", "meaning": "The new vector of length d that represents the patch inside the Transformer." },
    { "id": "x", "name": "x (flattened patch)", "meaning": "The P·P·C pixel numbers of one patch laid out as a row." },
    { "id": "E", "name": "E (projection matrix)", "meaning": "A learned matrix with P·P·C rows and d columns. The same E is used for every patch, so it is shared across the image." },
    { "id": "b", "name": "b (bias)", "meaning": "A learned vector of length d that is added to every patch embedding." }
  ],
  "steps": [
    { "title": "A tiny patch and matrix", "expr": "x = [0, 1, 1, 0] (2 × 2 grey patch, flattened); E has 4 rows and d = 3 columns: rows [0.5, 0, −1], [1, 0.5, 0], [0, 1, 0.5], [−0.5, 0, 1]; b = [0.1, 0, −0.1]", "note": "A teaching example. The numbers in E are made up; a real E is learned." },
    { "title": "Multiply and add", "expr": "z1 = 0×0.5 + 1×1 + 1×0 + 0×(−0.5) + 0.1 = 1.1; z2 = 0 + 0.5 + 1 + 0 + 0 = 1.5; z3 = 0 + 0 + 0.5 + 0 − 0.1 = 0.4", "note": "So z = [1.1, 1.5, 0.4]: a patch of 4 numbers became a vector of 3." },
    { "title": "The same E for a different patch", "expr": "x = [1, 0, 0, 1] gives z = [0.1, 0, −0.1]", "note": "Every patch uses the same E and b. That is why patch embedding costs the same however many patches there are." }
  ]
}
```

Notice the last step: ==one E is shared across all patches==, the same idea as weight sharing in a CNN filter. The number of parameters therefore depends on the patch and on d, but not on how many patches the image has.

```viz
{
  "type": "equation",
  "title": "Parameters of the patch embedding",
  "intro": "Select a part to read it, then follow the worked examples for ViT-Base.",
  "formula": [ { "t": "params", "id": "params" }, " = ", { "t": "P·P·C", "id": "in" }, " × ", { "t": "d", "id": "d" }, " + ", { "t": "d", "id": "bias" } ],
  "parts": [
    { "id": "params", "name": "params", "meaning": "How many learned numbers the patch embedding holds." },
    { "id": "in", "name": "P·P·C (inputs)", "meaning": "The length of a flattened patch, which is the number of rows of E." },
    { "id": "d", "name": "d (hidden size)", "meaning": "The length of each patch embedding, which is the number of columns of E." },
    { "id": "bias", "name": "d (bias terms)", "meaning": "One bias number for each output position." }
  ],
  "steps": [
    { "title": "ViT-Base, 16 × 16 patches", "expr": "768 × 768 + 768 = 589,824 + 768 = 590,592", "note": "The same count as the pipeline figure below and as Lab Part 2.3." },
    { "title": "32 × 32 patches", "expr": "3,072 × 768 + 768 = 2,359,296 + 768 = 2,360,064", "note": "Bigger patches mean a bigger E." },
    { "title": "The tiny lab ViT", "expr": "16 × 32 + 32 = 544 (patch 4 on a 16 × 16 grey picture, d = 32)", "note": "In Lab Part 4 you build this and count it in code." }
  ]
}
```

In Lab Part 2.2 you will see that a convolution with kernel size P and stride P does exactly the same thing: it slides a P × P window with no overlap, multiplies by a learned matrix and gives one vector per window. That is how most ViT code implements patchify plus projection in one line.

### Step 3: the CLS token

A sentence of words can end in a summary. Picture-language models need a place where the whole picture can be summarised too. ViT borrows a trick from BERT (Week 9): it puts one extra, learned vector in front of the patches, called the [[CLS token|A special learned vector placed at the start of the sequence. After the Transformer, its output is used as a summary of the whole input for classification.]] (short for “classification”). It starts as a fixed learned vector, and as it passes through the attention layers it can read from every patch. After the last block, ==the CLS vector is the picture's summary==, and a small classifier on top of it gives the class scores.

With the CLS token the sequence is 196 + 1 = 197 tokens, each of length 768. Some ViTs average all the patch outputs instead of using a CLS token; you will compare the two in Lab Part 5.3.

### Step 4: add positions, then the encoder and head

Before the first block, a learned **position embedding** is added to each token so the model knows where each patch came from. That deserves its own section, and it is next. After that, the 197 tokens pass through a stack of ordinary Transformer encoder blocks from Week 8 (attention and a feed-forward layer, each with a residual connection and layer normalisation). The shape does not change from block to block. Finally the CLS vector goes into a small [[MLP head|A small network, here one or two linear layers, that turns the summary vector into one score for each class.]].

### The whole ViT as one pipeline

Step through a 224 × 224 colour image with 16 × 16 patches. Every shape in the figure is computed.

```viz
{
  "type": "pipeline",
  "title": "A Vision Transformer, step by step",
  "intro": "Press the step buttons. Every step shows the tensor shape for ViT-Base on a 224 × 224 colour image.",
  "nodes": [
    { "id": "img", "label": "Image", "sub": "(3, 224, 224)" },
    { "id": "patch", "label": "Patchify", "sub": "(196, 768)" },
    { "id": "proj", "label": "Linear projection", "sub": "(196, 768)" },
    { "id": "cls", "label": "Add CLS", "sub": "(197, 768)" },
    { "id": "pos", "label": "Add positions", "sub": "(197, 768)" },
    { "id": "enc", "label": "12 encoder blocks", "sub": "(197, 768)" },
    { "id": "head", "label": "CLS → head", "sub": "(768) → classes" }
  ],
  "steps": [
    { "title": "Start with the image", "body": "The input is a tensor of pixel values: 3 colour numbers at each of 224 × 224 positions.", "active": ["img"], "example": "3 × 224 × 224 = 150,528 numbers." },
    { "title": "Cut into patches and flatten", "body": "Cut the image into non-overlapping 16 × 16 squares and flatten each one into a list of numbers. Nothing is learned in this step.", "active": ["img", "patch"], "example": "14 × 14 = 196 patches, each with 16 × 16 × 3 = 768 numbers. Result: (196, 768)." },
    { "title": "Linear projection", "body": "One shared linear layer maps each flattened patch to a vector of length d. In ViT-Base d is 768, so here the shape stays the same.", "active": ["patch", "proj"], "example": "768 × 768 + 768 = 590,592 parameters. Result: (196, 768)." },
    { "title": "Put the CLS token in front", "body": "A learned CLS vector is prepended to the sequence, so there is now one extra token.", "active": ["proj", "cls"], "example": "196 patches + 1 CLS = 197 tokens. Result: (197, 768). The CLS token has 768 parameters." },
    { "title": "Add position embeddings", "body": "A learned table with one vector for each of the 197 positions is added to the tokens, element by element, so each token carries where it came from.", "active": ["cls", "pos"], "example": "197 × 768 = 151,296 parameters. The shape stays (197, 768)." },
    { "title": "Transformer encoder", "body": "The tokens pass through 12 Transformer blocks of Week 8. Every patch can attend to every other patch from the very first block. The shape does not change.", "active": ["pos", "enc"], "example": "Attention scores per head: 197 × 197 = 38,809. Parameters: 12 × 7,087,872 = 85,054,464." },
    { "title": "Read out the CLS token", "body": "After the last block, the CLS vector summarises the whole image. A linear head turns it into one score for each class.", "active": ["enc", "head"], "example": "A 1000-class head: 768 × 1000 + 1000 = 769,000 parameters. For 10 classes: 7,690." }
  ]
}
```

Notice that the encoder is exactly the Transformer you already know: ==only the front end changed==, from word embeddings to patch embeddings. In Lab Part 4.2 you will build this entire pipeline in PyTorch at a small scale and print this same shape trace.

### The patch-size trade-off

The patch side P is the main design choice of the front end. Look at a second picture: a 16 × 16 drawing of a jeepney with its roof, windows, body and wheels. Try all three patch sizes.

```viz
{
  "type": "patch",
  "title": "A jeepney cut into tokens",
  "image": [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0],
    [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0],
    [0,6,6,6,1,1,6,1,1,6,1,1,6,6,6,0],
    [0,6,6,6,1,1,6,1,1,6,1,1,6,6,6,0],
    [0,6,6,6,1,1,6,1,1,6,1,1,6,6,6,0],
    [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0],
    [0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,0],
    [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0],
    [0,0,0,9,9,0,0,0,0,0,0,9,9,0,0,0],
    [0,0,0,9,9,0,0,0,0,0,0,9,9,0,0,0],
    [0,0,0,9,9,0,0,0,0,0,0,9,9,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  ],
  "channels": 1,
  "patchSizes": [2, 4, 8]
}
```

At 2 × 2 patches there are 64 tokens, and many of them are all-blank sky: tokens spent on nothing. At 8 × 8 patches there are only 4 tokens and each one mixes windows, body and wheels together. Smaller patches mean more detail but a longer sequence, and the cost of attention grows with the square of that length. Here is the same trade-off for a real 224 × 224 photo, with attention scores per head computed as n × n with the CLS token.

| Patch side P | Tokens n (with CLS) | Scores per head n × n | Compared with P = 16 |
|---|---|---|---|
| 8 | 785 | 616,225 | about 15.9 times more |
| 16 | 197 | 38,809 | 1 (the reference) |
| 32 | 50 | 2,500 | about 0.064 times, so roughly 15.5 times fewer |

==Halving P multiplies the tokens by 4 and the scores by about 16.== In Lab Part 2.5 you will see the other side of the trade-off on small pictures: with tiny patches, many tokens are empty. Try the calculator again with the other settings.

```viz
{ "type": "calc", "kind": "vit_tokens", "title": "Patch size trade-off: try 384 × 384 images too", "d": 384 }
```

Set the image side to 384 and the patch to 16: you should read 24 × 24 = 576 patches and a sequence of 577. Bigger pictures mean many more tokens, which is one reason image resolution is an important cost in vision models. You will meet 576 patches again in Section 5.

### Counting the parameters of ViT-Base

A parameter count lets you check that you understand every part. ViT-Base uses a hidden size d = 768, 12 blocks, 12 heads (each of size 768 ÷ 12 = 64) and a feed-forward width d_ff = 3,072. Each block is counted exactly as in Week 8, with biases: attention 4 · 768² + 4 · 768 = 2,362,368; feed-forward 2 · 768 · 3,072 + 3,072 + 768 = 4,722,432; two layer norms 4 · 768 = 3,072. That adds up to 7,087,872 per block.

```viz
{
  "type": "equation",
  "title": "Parameters of ViT-Base (16 × 16 patches)",
  "intro": "Select a part to read it, then follow the worked example that adds every piece.",
  "formula": [ { "t": "total", "id": "total" }, " = ", { "t": "L × block", "id": "blocks" }, " + ", { "t": "patch embedding", "id": "pe" }, " + ", { "t": "CLS", "id": "cls" }, " + ", { "t": "positions", "id": "pos" }, " + ", { "t": "final norm", "id": "norm" } ],
  "parts": [
    { "id": "total", "name": "total", "meaning": "All learned numbers of the ViT body, not counting the final classification head." },
    { "id": "blocks", "name": "L × block", "meaning": "The number of encoder blocks times the parameters of one block (attention, feed-forward and two layer norms)." },
    { "id": "pe", "name": "patch embedding", "meaning": "The matrix E and the bias of the linear projection." },
    { "id": "cls", "name": "CLS", "meaning": "The learned CLS vector, one number per hidden dimension." },
    { "id": "pos", "name": "positions", "meaning": "The learned position table with one vector for each of the 197 positions." },
    { "id": "norm", "name": "final norm", "meaning": "The layer norm applied to the output of the last block: a scale and a shift for each of the 768 dimensions." }
  ],
  "steps": [
    { "title": "The 12 blocks", "expr": "12 × 7,087,872 = 85,054,464", "note": "Each block: attention 2,362,368 + feed-forward 4,722,432 + layer norms 3,072." },
    { "title": "The front end", "expr": "590,592 (patch embedding) + 768 (CLS) + 151,296 (197 × 768 positions) = 742,656", "note": "Small compared with the blocks: almost all parameters live in the Transformer." },
    { "title": "The body", "expr": "85,054,464 + 742,656 + 1,536 (final norm 2 × 768) = 85,798,656", "note": "About 86 million parameters." },
    { "title": "With a 1000-class head", "expr": "85,798,656 + 769,000 = 86,567,656", "note": "ViT-Base is usually quoted as about 86 million parameters. In Lab Part 4.3 you will apply the same formula to your tiny ViT and to these ViT-Base numbers." }
  ]
}
```

==More than 99% of the parameters sit inside the 12 blocks==, exactly as in a text Transformer. The picture-specific parts are a tiny fraction.

### Do the patches talk to each other?

Inside the encoder every patch can attend to every other patch, as every word can in a text model. The attention figure from Week 7 works the same way for patch tokens. Here a signboard patch is the query and four other patches of the jeepney photo are the keys. The vectors are small and made up, so read the figure as a picture of the idea.

```viz
{
  "type": "attention",
  "title": "A signboard patch looks at other patches",
  "intro": "Pick a query and watch which patches it attends to. The numbers are computed from the made-up vectors.",
  "scale": true,
  "tokens": ["wheel", "window", "signboard", "roof", "road"],
  "keys": [[1, 0], [0, 1], [0.5, 1], [0.2, 0.8], [1, 0.2]],
  "values": [[0.9, 0.1], [0.2, 0.9], [0.1, 1], [0.3, 0.7], [0.8, 0.2]],
  "queries": [ { "label": "signboard", "q": [0.4, 1.2] }, { "label": "wheel", "q": [1.2, 0.1] } ],
  "note": "Illustrative vectors authored for teaching, not from a trained model. In a real ViT every patch is also a query, and there are many heads."
}
```

Notice that the signboard query puts its largest weight on patches that are similar to it (the signboard itself with 0.269, then the window with 0.233 and the roof with 0.208) and the least on the wheel (0.133). A real ViT learns which patches to look at, and unlike a CNN it can link two patches that are far apart in the very first layer.

### What a ViT does not have: inductive bias

A CNN comes with two assumptions built into its design: ==nearby pixels belong together (locality) and the same filter works everywhere== in the picture (weight sharing). These built-in assumptions are called [[inductive biases|Assumptions built into a model's design that help it learn from less data, for example a CNN's belief that nearby pixels are related.]]. A ViT has almost none of them: any patch may attend to any other, and it must learn from data which relationships matter. ==With less built-in structure, a ViT needs more data, or pre-training on a large dataset, to match a CNN==. The UvA Tutorial 15 notebook shows this: trained from scratch on CIFAR10, a ViT reaches about 75% where CNNs reach about 90%. Lab Part 4.5 and Part 5.5 let you see the same effect at small scale, honestly: on a few hundred little pictures, a tiny ViT may not beat a tiny CNN, and that is a result, not a failure.

Place each property below with the model it describes.

```viz
{
  "type": "sorter",
  "title": "CNN, ViT, or both?",
  "intro": "Sort each property. Remember: a ViT shares the projection and the Transformer weights over all patches, but it does not slide a filter.",
  "bins": [
    { "id": "cnn", "label": "CNN" },
    { "id": "vit", "label": "ViT" },
    { "id": "both", "label": "Both" }
  ],
  "cards": [
    { "name": "Slides a small filter across the picture", "answer": "cnn", "why": "This is convolution, the basic operation of a CNN. A ViT does not slide a filter; it cuts patches." },
    { "name": "Cuts the image into patches and flattens each one", "answer": "vit", "why": "Patchify is the first step of a ViT. A CNN keeps the grid shape." },
    { "name": "Needs position embeddings to know where a patch came from", "answer": "vit", "why": "A CNN's filters already know the grid layout. A ViT's attention treats tokens as an unordered set, so positions must be added." },
    { "name": "Builds in the assumption that nearby pixels are related", "answer": "cnn", "why": "Locality is a built-in inductive bias of convolution." },
    { "name": "Every patch can attend to every other patch in the first layer", "answer": "vit", "why": "Self-attention connects all tokens at once. A CNN sees only a small neighbourhood at first and grows its view layer by layer." },
    { "name": "Trained by backpropagation with gradient descent", "answer": "both", "why": "Every model in this course is trained this way (Weeks 2 and 3)." },
    { "name": "Usually needs lots of data or pre-training when trained from scratch", "answer": "vit", "why": "With fewer built-in assumptions, a ViT has more to learn. A CNN is more data-efficient on small datasets." },
    { "name": "Ends with a layer that outputs one score for each class", "answer": "both", "why": "Both finish with a classification head, a linear layer over a feature vector." }
  ]
}
```

Check any card you got wrong against the reason shown, because the CNN versus ViT comparison is a favourite exam question. One last look at where the ViT sits in the story of image models.

```viz
{
  "type": "timeline",
  "title": "From small filters to patch tokens",
  "events": [
    { "year": "1998", "title": "LeNet", "text": "LeCun et al. showed that a small convolutional network can read handwritten digits (Week 4)." },
    { "year": "2012", "title": "AlexNet", "text": "Krizhevsky et al. won an image-recognition contest with a deep CNN trained on GPUs, which started the deep-learning era for images." },
    { "year": "2015", "title": "ResNet", "text": "He et al. added residual connections so that CNNs with many more layers could be trained." },
    { "year": "2017", "title": "The Transformer", "text": "Vaswani et al. introduced attention-only sequence models for language, the architecture of Weeks 7 and 8." },
    { "year": "2020", "title": "Vision Transformer", "text": "Dosovitskiy et al. applied a standard Transformer to a sequence of image patches and showed it can match strong CNNs when pre-trained on large datasets." }
  ]
}
```

## Position embeddings for patches {rail=Positions}

{lede} Imagine tipping a jigsaw puzzle out of its box. Every piece is still there, but the picture is gone, because ==a pile of pieces has no layout==. A Transformer's self-attention treats its tokens like that pile. If you give it the patches of a photo without telling it where each one came from, it cannot tell a face from the same face with its eyes, nose and mouth jumbled. [[Position embeddings|Learned or computed vectors, one for each position, that are added to the token vectors so that the model can tell where each token sits in the sequence or the image.]] put the layout back.

### The problem: attention does not know the order

In Week 8 you met the same problem for words: “the dog chased the cat” and “the cat chased the dog” contain the same words, and plain self-attention cannot tell them apart without positional encoding. For images the effect is just as strong. Self-attention is [[permutation-equivariant|If you shuffle the input tokens, the output tokens are the same vectors shuffled in the same way. The layer has no sense of order.]]: ==shuffle the patches in, and you get the same outputs shuffled out== Select each part of the statement to read it.

```viz
{
  "type": "equation",
  "title": "Self-attention is permutation-equivariant",
  "intro": "Select a part of the statement to read it, then follow the worked example with four patches.",
  "formula": [ { "t": "f", "id": "f" }, "(", { "t": "π", "id": "pi" }, "(", { "t": "X", "id": "X" }, ")) = ", { "t": "π", "id": "pi2" }, "(", { "t": "f", "id": "f2" }, "(", { "t": "X", "id": "X2" }, "))" ],
  "parts": [
    { "id": "f", "name": "f (a self-attention layer)", "meaning": "The layer without any position information: queries, keys and values are all made from the token vectors alone." },
    { "id": "pi", "name": "π (a shuffle)", "meaning": "Any way of re-ordering the tokens, for example swapping the first and the third." },
    { "id": "X", "name": "X (the tokens)", "meaning": "The matrix of token vectors, one row for each patch." },
    { "id": "pi2", "name": "π (the same shuffle)", "meaning": "The same re-ordering, applied to the outputs instead of the inputs." },
    { "id": "f2", "name": "f (the same layer)", "meaning": "The same attention layer, this time given the original, unshuffled tokens." },
    { "id": "X2", "name": "X (the original tokens)", "meaning": "The same matrix, in its original order." }
  ],
  "steps": [
    { "title": "Four patch tokens", "expr": "A = [1, 0], B = [0, 1], C = [1, 1], D = [0.5, −0.5]", "note": "Made-up 2-number patch vectors. Attention uses A, B, C, D as queries, keys and values." },
    { "title": "Output for patch A in the original order A, B, C, D", "expr": "weights on A, B, C, D = [0.313, 0.154, 0.313, 0.220]; output = [0.736, 0.357]", "note": "The weights are softmax of the dot products divided by √2, as in Week 7." },
    { "title": "Shuffle to C, A, D, B and read patch A again", "expr": "weights on C, A, D, B = [0.313, 0.313, 0.220, 0.154]; output = [0.736, 0.357]", "note": "Patch A gets the same output. The weights are reordered with the tokens, but the output vector is identical. This is the proof that attention alone cannot see order." }
  ]
}
```

The same holds for all four patches, as the heat maps below show. The first tab is the attention matrix in the original order. The second tab shuffles the patches, runs attention, and then sorts the rows and columns back to A, B, C, D: you get exactly the same matrix. The third tab does the same after adding a position vector to each slot: now the matrix changes.

```viz
{
  "type": "heatmap",
  "title": "Attention weights among four patches, with and without positions",
  "corner": "query ↓ key →",
  "rows": ["A", "B", "C", "D"],
  "cols": ["A", "B", "C", "D"],
  "showValues": true,
  "rowsSumToOne": true,
  "tabs": [
    { "name": "Original order, no positions", "caption": "Attention among A, B, C, D in the order A, B, C, D.", "W": [ [0.313, 0.154, 0.313, 0.220], [0.174, 0.352, 0.352, 0.122], [0.221, 0.221, 0.449, 0.109], [0.313, 0.154, 0.220, 0.313] ] },
    { "name": "Shuffled, no positions", "caption": "The patches were fed in the order C, A, D, B; rows and columns are sorted back to A, B, C, D. Identical to the first tab.", "W": [ [0.313, 0.154, 0.313, 0.220], [0.174, 0.352, 0.352, 0.122], [0.221, 0.221, 0.449, 0.109], [0.313, 0.154, 0.220, 0.313] ] },
    { "name": "Shuffled, with positions", "caption": "The same shuffled patches, but each slot now carries its own position vector. The matrix changes: the model can tell that the patches moved.", "W": [ [0.559, 0.061, 0.276, 0.104], [0.156, 0.304, 0.317, 0.223], [0.099, 0.045, 0.829, 0.026], [0.286, 0.240, 0.201, 0.274] ] }
  ],
  "note": "Computed from the four made-up patch vectors in the equation figure above and four made-up position vectors; values rounded to three decimals."
}
```

Switch between the first two tabs: the numbers do not change at all. Then open the third: they do. That is the whole job of position embeddings, and in Lab Part 3.1 and 3.4 you will run the same experiment in NumPy with 6 patches and assert it.

### See it on a picture

Now with a picture, which is easier to feel. Select a shuffle and watch what the model would receive. Turn the position embeddings on and off. The drawing has a plus sign in the left half and a square in the right half.

```viz
{
  "type": "patchshuffle",
  "title": "Shuffle the patches: plus sign left, square right",
  "intro": "Press Shuffle, then switch the position embeddings on and off and read the sentence under the picture.",
  "image": [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0],
    [0,0,9,9,9,9,9,9,9,0,0,0,0,0,0,0],
    [0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,9,9,9,9,9,0,0],
    [0,0,0,0,0,0,0,0,0,9,4,4,4,9,0,0],
    [0,0,0,0,0,0,0,0,0,9,4,4,4,9,0,0],
    [0,0,0,0,0,0,0,0,0,9,4,4,4,9,0,0],
    [0,0,0,0,0,0,0,0,0,9,9,9,9,9,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  ],
  "patch": 4,
  "seed": 4,
  "note": "A 16 × 16 teaching picture cut into 4 × 4 patches (16 tokens). In Lab Part 3.2 and Part 5.1 you will use a similar task, “is the plus sign in the left half?”."
}
```

==Without positions the model receives the same 16 patches whichever way they are arranged==, so if the shuffle moves the plus sign into the right half, the model cannot know. With positions, each token also carries its location, and the picture can be told apart from its shuffled copy. Try a second picture, a jeepney, to see that the story is not specific to one drawing.

```viz
{
  "type": "patchshuffle",
  "title": "Shuffle the patches: a jeepney",
  "intro": "Shuffle the jeepney. Can you still tell where the wheels and the windows were?",
  "image": [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,0],
    [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0],
    [0,6,6,6,1,1,6,1,1,6,1,1,6,6,6,0],
    [0,6,6,6,1,1,6,1,1,6,1,1,6,6,6,0],
    [0,6,6,6,1,1,6,1,1,6,1,1,6,6,6,0],
    [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0],
    [0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,0],
    [0,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0],
    [0,0,0,9,9,0,0,0,0,0,0,9,9,0,0,0],
    [0,0,0,9,9,0,0,0,0,0,0,9,9,0,0,0],
    [0,0,0,9,9,0,0,0,0,0,0,9,9,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  ],
  "patch": 4,
  "seed": 7,
  "note": "A 16 × 16 teaching picture cut into 4 × 4 patches. The shuffle is fixed for each seed so the figure is the same every time."
}
```

A person looks at the shuffled jeepney and sees nonsense, because we use position without thinking. A ViT without position embeddings is in the same position as someone looking at the shuffled puzzle with the box lid hidden.

### The fix: add a vector for each position

The remedy is simple. Make a table with one vector per patch position, each as long as a patch embedding, and **add** the right row to each patch embedding. The patch embedding says *what* the patch looks like, and the position vector says *where* it is. ==The two are added, not glued side by side==, so the length d stays the same and nothing else in the Transformer has to change.

```viz
{
  "type": "equation",
  "title": "Adding the position of a patch",
  "intro": "Select a part to read it, then follow the worked example, which continues the tiny example from Section 2.",
  "formula": [ { "t": "z", "id": "z" }, { "t": "i", "id": "zi" }, " = ", { "t": "x", "id": "x" }, { "t": "i", "id": "xi" }, " · ", { "t": "E", "id": "E" }, " + ", { "t": "p", "id": "p" }, { "t": "i", "id": "pi" } ],
  "parts": [
    { "id": "z", "name": "z (token with position)", "meaning": "The vector that actually enters the first Transformer block for patch i." },
    { "id": "zi", "name": "i (the position index)", "meaning": "Which patch we mean: 1 for the first patch, 2 for the second, and so on." },
    { "id": "x", "name": "x (flattened patch)", "meaning": "The pixel numbers of patch i, laid out as a row." },
    { "id": "xi", "name": "i (the patch index)", "meaning": "The same index again, because every patch has its own x." },
    { "id": "E", "name": "E (shared projection)", "meaning": "The one learned projection matrix, the same for all patches. The bias b is folded in here to keep the formula short." },
    { "id": "p", "name": "p (position embedding)", "meaning": "A learned vector, one for each position. Position 1 always gets the same p, whatever the picture is." },
    { "id": "pi", "name": "i (the position index again)", "meaning": "Which row of the position table to add." }
  ],
  "steps": [
    { "title": "Patch at position 1", "expr": "x1·E + b = [1.1, 1.5, 0.4]; p1 = [0.2, −0.1, 0.3]; z1 = [1.3, 1.4, 0.7]", "note": "The same patch and projection as in Section 2, plus a made-up position vector." },
    { "title": "Patch at position 2", "expr": "x2·E + b = [0.1, 0, −0.1]; p2 = [−0.2, 0.1, 0]; z2 = [−0.1, 0.1, −0.1]", "note": "A different patch with its own position vector." },
    { "title": "Swap the places of the two patches", "expr": "the first patch now sits at position 2: z = [1.1 − 0.2, 1.5 + 0.1, 0.4 + 0] = [0.9, 1.6, 0.4], not [1.3, 1.4, 0.7]", "note": "The same patch gets a different vector when it moves, so the model can tell that something moved." }
  ]
}
```

Check the last step: the same patch gives [1.3, 1.4, 0.7] in position 1 and [0.9, 1.6, 0.4] in position 2. Every token now carries its location, so ==the attention scores depend on where tokens are== and the symmetry from the first figure is broken.

### Two ways to build the table

**Learned 1D positions.** The original ViT simply makes the table a set of free parameters, one row for each position in reading order (left to right, top to bottom), and lets training decide the values. For ViT-Base the table has 197 rows (196 patches plus the CLS token) and 768 columns: 197 × 768 = 151,296 learned numbers. Treating the patches as a one-dimensional line, with no built-in notion of rows and columns, sounds crude, but the ViT authors tried more elaborate two-dimensional ways of encoding position and report only small differences in their experiments. For a model with plenty of data, plain learned positions work well. The learned embeddings still discover the grid on their own: they are reported to be more similar for neighbouring patches than for distant ones.

**2D sinusoidal positions.** You can also compute the table with the sine and cosine formula of Week 8, with no learning at all. For an image you apply it twice: one half of the dimensions encodes the row of the patch and the other half encodes its column.

```viz
{
  "type": "equation",
  "title": "A 2D sinusoidal position vector",
  "intro": "Select a part to read it, then follow the worked example with d = 4.",
  "formula": [ { "t": "p(row, col)", "id": "p" }, " = [ ", { "t": "PE(row)", "id": "r" }, " ,  ", { "t": "PE(col)", "id": "c" }, " ]" ],
  "parts": [
    { "id": "p", "name": "p(row, col)", "meaning": "The position vector for the patch in a given row and column of the patch grid. Its length is d." },
    { "id": "r", "name": "PE(row)", "meaning": "The Week 8 sinusoidal encoding of the row number, using d/2 dimensions: pairs of sin(pos / 10000^(2k/(d/2))) and cos of the same angle." },
    { "id": "c", "name": "PE(col)", "meaning": "The same encoding of the column number, using the other d/2 dimensions." }
  ],
  "steps": [
    { "title": "With d = 4 each half has 2 numbers", "expr": "PE(pos) = [sin(pos), cos(pos)] because with 2 numbers only k = 0 is used and the angle is just pos", "note": "A tiny size on purpose; real models use a much larger d." },
    { "title": "Patch (row 0, col 0) and patch (row 0, col 1)", "expr": "p(0, 0) = [0, 1, 0, 1]; p(0, 1) = [0, 1, 0.841, 0.540]", "note": "Same row, so the first half is the same; different column, so the second half differs." },
    { "title": "Patch (row 1, col 0) and patch (row 2, col 3)", "expr": "p(1, 0) = [0.841, 0.540, 0, 1]; p(2, 3) = [0.909, −0.416, 0.141, −0.990]", "note": "All the patches of a grid get different vectors. In Lab Part 3.3 you assert that all 16 vectors of a 4 × 4 grid are different." }
  ]
}
```

The nice property of the sinusoidal version is that ==it already knows the grid==: positions in the same row or column share half of their vector. The learned version has to discover this from data. To see it, the heat map below shows how similar every position of a 4 × 4 grid is to the top-left position (row 0, column 0), measured with cosine similarity between 8-number sinusoidal vectors (4 numbers for the row and 4 for the column).

```viz
{
  "type": "heatmap",
  "title": "How similar is each position to the top-left patch?",
  "corner": "row ↓ column →",
  "rows": ["row 0", "row 1", "row 2", "row 3"],
  "cols": ["col 0", "col 1", "col 2", "col 3"],
  "showValues": true,
  "rowsSumToOne": false,
  "W": [ [1.00, 0.89, 0.65, 0.50], [0.89, 0.77, 0.53, 0.39], [0.65, 0.53, 0.29, 0.15], [0.50, 0.39, 0.15, 0.00] ],
  "note": "Computed: cosine similarity between the 2D sinusoidal vector of each position and that of (row 0, col 0), with d = 8, rounded to two decimals. Learned position embeddings are reported to show a similar pattern, with neighbours more alike than distant patches, but these numbers are for the sinusoidal formula only."
}
```

The top-left cell is 1 because a vector is identical to itself. The numbers fall as you move away, so neighbouring patches (0.89) have more similar positions than far-apart ones (0.00 at the opposite corner). In Lab Part 3.5 you will make this plot yourself for an 8 × 8 grid. Two other families exist and are only named here: *relative* position encodings, which describe the distance between two tokens instead of the absolute place, and *rotary* embeddings, used in many modern language models.

The sinusoidal pieces are just the Week 8 curves. Here is the one-dimensional encoding itself, drawn for 24 positions and 16 numbers per position: for the 2D version you would use one such block for the row and another for the column.

```viz
{ "type": "posenc", "title": "The Week 8 sinusoidal encoding, the building block of the 2D version", "dModel": 16, "positions": 24 }
```

Each row is one position and each column one of the 16 numbers. Look at how the left columns change quickly from row to row (high frequency) and the right columns change slowly, so that no two rows are the same.

### When the image gets bigger

A model trained on 224 × 224 images with 16 × 16 patches has a table of 196 patch positions (plus one for the CLS token). Give it a 384 × 384 image and there are 24 × 24 = 576 patches: the table has no rows for most of them. The common practice is to **interpolate** the table (resize it smoothly, like stretching a small picture into a bigger one): arrange the 196 learned vectors in their 14 × 14 grid, stretch the grid to 24 × 24 with smooth (bilinear) resizing, then flatten it again. It is a trick that usually works, though accuracy often needs a little fine-tuning to recover. Step through it.

```viz
{
  "type": "pipeline",
  "title": "Stretching a position table to a new image size",
  "intro": "Press the step buttons to follow the table from 14 × 14 to 24 × 24.",
  "nodes": [
    { "id": "table", "label": "Learned table", "sub": "197 × 768" },
    { "id": "grid", "label": "Reshape to grid", "sub": "14 × 14 × 768" },
    { "id": "resize", "label": "Resize", "sub": "24 × 24 × 768" },
    { "id": "flat", "label": "Flatten + CLS", "sub": "577 × 768" }
  ],
  "steps": [
    { "title": "Start with the trained table", "body": "The model was trained on 224 × 224 images, so its table has one learned vector for the CLS token and one for each of the 196 patch positions.", "active": ["table"], "example": "197 × 768 = 151,296 numbers." },
    { "title": "Put the 196 patch vectors back in their grid", "body": "Set the CLS row aside. Lay the other 196 rows out on the 14 × 14 grid of the picture, so that neighbours in the picture are neighbours in the grid.", "active": ["table", "grid"], "example": "196 = 14 × 14, each cell holds a vector of 768 numbers." },
    { "title": "Resize the grid like an image", "body": "Treat each of the 768 dimensions as a small image and enlarge it smoothly (bilinear interpolation) to the new patch grid. In PyTorch this is F.interpolate.", "active": ["grid", "resize"], "example": "A 384 × 384 image with 16 × 16 patches has 384 ÷ 16 = 24 patches per side, so the grid becomes 24 × 24 = 576 cells." },
    { "title": "Flatten and add the CLS row back", "body": "Flatten the new grid into a sequence of 576 vectors and put the CLS position in front. The model can now read the larger image.", "active": ["resize", "flat"], "example": "(576 + 1) × 768 = 443,136 numbers in the table, up from 151,296." }
  ]
}
```

Notice that ==no new knowledge is created: the new vectors are blends of the old ones==. That is why accuracy at the new size can drop a little until the model is fine-tuned. In Lab Part 3.6 you will implement this, print the shape before and after, and count the parameters.

### Check yourself: does the task need positions?

Not every task needs layout. Decide for each statement below.

```viz
{
  "type": "sorter",
  "title": "Does it need positions?",
  "intro": "Think about whether the answer changes when the pieces are shuffled.",
  "bins": [
    { "id": "needs", "label": "needs positions" },
    { "id": "no", "label": "does not need positions" }
  ],
  "cards": [
    { "name": "A bag-of-words model that decides if a review is positive by counting words", "answer": "no", "why": "Counting words ignores order by design. Positions would add nothing for such a model." },
    { "name": "A ViT that answers “is the plus sign in the left half?”", "answer": "needs", "why": "The answer depends on where the plus sign is. The patch contents alone cannot say." },
    { "name": "Deciding whether a picture contains any red patch at all", "answer": "no", "why": "A single patch with red is enough, wherever it is, so the set of patches is enough." },
    { "name": "Telling a face from the same face with its eyes and mouth jumbled", "answer": "needs", "why": "The patches are identical in both pictures. Only their arrangement differs." },
    { "name": "Telling “the dog chased the cat” from “the cat chased the dog”", "answer": "needs", "why": "The same words appear in both sentences. Only the order tells who chased whom." },
    { "name": "Checking whether a text mentions the word “flood” anywhere", "answer": "no", "why": "Presence of a word does not depend on its position." },
    { "name": "Reading the time from a clock face", "answer": "needs", "why": "The meaning is in where the hands point, so the layout matters." },
    { "name": "Counting how many black patches a drawing has", "answer": "no", "why": "A count is the same for any arrangement of the patches." }
  ]
}
```

A short rule: ==if shuffling the pieces cannot change the right answer, positions are unnecessary, and if it can, they are essential.== In Lab Part 5.1 and Part 5.2 you will train a tiny ViT with and without positions on exactly this kind of task and measure the gap.

## CLIP: images and words in one space {rail=CLIP}

{lede} Imagine a big box of postcards, each with a photo on the front and a caption on the back, and then somebody shuffles the photos and the captions into two separate piles. You could still learn to pair them up: a photo of a jeepney goes with “a jeepney on a busy street”, not with “a carabao in a rice field”. CLIP (Radford et al., 2021) does this at enormous scale. It trains an image encoder and a text encoder together so that ==a picture and its caption land close together in one shared space, and mismatched ones land far apart==. This idea is called [[contrastive learning|Training a model by pulling matching pairs together and pushing non-matching pairs apart in the embedding space.]].

### Two encoders and one shared space

CLIP has two networks. The **image encoder** (a ViT, or in some versions a CNN) turns a picture into one vector. The **text encoder** (a Transformer, Week 8) turns a sentence into one vector of the same length. The two vectors live in the same [[shared embedding space|One vector space in which the vectors of different modalities, such as images and sentences, can be compared directly because the model was trained to place matching items close together.]]. The training data are about 400 million image–text pairs gathered from the internet, which is why one model can recognise so many different things.

```viz
{
  "type": "pipeline",
  "title": "How CLIP is trained on one batch",
  "intro": "Press the step buttons to follow a batch of N image–caption pairs through the two encoders to the loss.",
  "nodes": [
    { "id": "batch", "label": "Batch", "sub": "N images + N captions" },
    { "id": "ie", "label": "Image encoder", "sub": "(N, 512)" },
    { "id": "te", "label": "Text encoder", "sub": "(N, 512)" },
    { "id": "norm", "label": "Normalise", "sub": "length 1" },
    { "id": "mat", "label": "N × N cosines", "sub": "image vs text" },
    { "id": "loss", "label": "Contrastive loss", "sub": "diagonal wins" }
  ],
  "steps": [
    { "title": "Pick a batch of true pairs", "body": "Take N pictures, each with its own caption. Picture i belongs with caption i and with no other caption in the batch.", "active": ["batch"], "example": "N = 4: a jeepney, a tricycle, a motorbike and a carabao, with their four captions." },
    { "title": "Encode both sides", "body": "The image encoder gives one vector per picture and the text encoder gives one vector per caption, both of the same length. 512 is the length in one common CLIP model; it is only an example here.", "active": ["batch", "ie", "te"], "example": "Image vectors (4, 512) and text vectors (4, 512)." },
    { "title": "Scale every vector to length 1", "body": "Dividing each vector by its length means that a dot product between two vectors is exactly their cosine similarity (Week 6).", "active": ["ie", "te", "norm"], "example": "A vector [3, 4] has length 5, so it becomes [0.6, 0.8]." },
    { "title": "Compare every picture with every caption", "body": "All N × N cosines make a table. The N true pairs sit on the diagonal. The other N × N − N cells are wrong pairs.", "active": ["norm", "mat"], "example": "4 × 4 = 16 cosines: 4 diagonal cells and 12 off-diagonal cells." },
    { "title": "Reward the diagonal", "body": "The loss raises the diagonal cosines and lowers all the others, row by row (which caption fits this picture?) and column by column (which picture fits this caption?).", "active": ["mat", "loss"], "example": "Each picture has N − 1 = 3 wrong captions to push away, and each caption has 3 wrong pictures." }
  ]
}
```

Notice that ==nobody labelled “jeepney” or “carabao” by hand==: the only supervision is that a picture and its caption belong together. That is why the internet could be the training set. In Lab Part 6.3 you will train a miniature CLIP with a small CNN and a two-word text encoder.

### Cosine similarity: which way do they point?

To compare a picture and a sentence, CLIP uses the cosine similarity from Week 6. It measures the angle between two vectors: 1 means they point the same way, 0 means they are unrelated, and −1 means they point in opposite directions.

```viz
{
  "type": "equation",
  "title": "Cosine similarity",
  "intro": "Select a part to read it, then follow the worked example.",
  "formula": [ { "t": "cos(a, b)", "id": "cos" }, " = ", { "t": "a·b", "id": "dot" }, " ÷ (", { "t": "‖a‖", "id": "na" }, " × ", { "t": "‖b‖", "id": "nb" }, ")" ],
  "parts": [
    { "id": "cos", "name": "cos(a, b)", "meaning": "The cosine of the angle between vectors a and b, a number between −1 and 1." },
    { "id": "dot", "name": "a·b (dot product)", "meaning": "Multiply the matching numbers of a and b and add them up." },
    { "id": "na", "name": "‖a‖ (length of a)", "meaning": "The square root of the sum of the squares of the numbers in a." },
    { "id": "nb", "name": "‖b‖ (length of b)", "meaning": "The same for b. Dividing by both lengths removes the effect of how long the vectors are, so only the direction counts." }
  ],
  "steps": [
    { "title": "Two vectors", "expr": "a = [3, 4], b = [4, 3]", "note": "Think of a as an image vector and b as a text vector, in a two-number toy space." },
    { "title": "Dot product and lengths", "expr": "a·b = 3×4 + 4×3 = 24; ‖a‖ = √(9 + 16) = 5; ‖b‖ = √(16 + 9) = 5", "note": "Both vectors have length 5." },
    { "title": "Divide", "expr": "cos(a, b) = 24 ÷ (5 × 5) = 0.96", "note": "Close to 1: the two vectors point in almost the same direction." }
  ]
}
```

Now try it on made-up word vectors with three numbers each: how vehicle-like, how animal-like and how fruit-like a thing is. Pick two items from each drop-down.

```viz
{
  "type": "vecsim",
  "title": "Cosine similarity of four items",
  "intro": "Choose two items and read the dot product, the lengths and the cosine.",
  "dims": ["vehicle", "animal", "fruit"],
  "words": [
    { "label": "jeepney", "v": [0.9, 0.05, 0.05] },
    { "label": "tricycle", "v": [0.8, 0.1, 0.05] },
    { "label": "carabao", "v": [0.05, 0.95, 0.1] },
    { "label": "mango", "v": [0, 0.05, 0.95] }
  ],
  "a": 0,
  "b": 1,
  "note": "The vectors are illustrative, authored for teaching. Real CLIP vectors have hundreds of numbers and no human-readable labels for the dimensions.",
  "checks": [
    { "a": "jeepney", "b": "tricycle", "cos": 0.998 },
    { "a": "jeepney", "b": "mango", "cos": 0.058 },
    { "a": "carabao", "b": "mango", "cos": 0.157 }
  ]
}
```

A jeepney and a tricycle point in nearly the same direction (cosine about 0.998) and a jeepney and a mango are almost unrelated (about 0.058). CLIP applies exactly this idea to a picture vector and a sentence vector.

### The similarity matrix of a batch

For a batch of N pairs, compute the cosine of every image with every text. The result is an N × N table, and by construction the true pairs are on the diagonal. Here are three pictures and three sentences. Select a picture to see its zero-shot probabilities, and slide the logit scale to see how sharp or soft the probabilities become (the logit scale is explained just below).

```viz
{
  "type": "clipsim",
  "title": "Three pictures, three sentences",
  "intro": "Select an image row, then move the logit-scale slider and watch the probabilities and the loss change.",
  "scale": 10,
  "images": [
    { "label": "photo of a jeepney", "v": [0.9, 0.1, 0.1] },
    { "label": "photo of a carabao", "v": [0.1, 0.9, 0.2] },
    { "label": "photo of a mango", "v": [0.1, 0.2, 0.9] }
  ],
  "texts": [
    { "label": "a photo of a jeepney", "v": [1, 0, 0.1] },
    { "label": "a photo of a carabao", "v": [0.1, 1, 0] },
    { "label": "a photo of a mango", "v": [0, 0.1, 1] }
  ],
  "note": "The vectors are illustrative, authored for teaching, not from a trained model. Real CLIP vectors have hundreds of numbers.",
  "checks": [
    { "scale": 10, "cos": [[0, 0, 0.994], [1, 1, 0.976], [2, 2, 0.987], [0, 1, 0.208]], "prob": [[0, 0, 0.999], [1, 1, 0.999]], "loss": 0.001 },
    { "scale": 1, "prob": [[0, 0, 0.534], [1, 1, 0.515]], "loss": 0.649 }
  ]
}
```

Look at the diagonal: every picture is closest to its own sentence, with a cosine of about 0.98 or 0.99, and the off-diagonal cosines are about 0.1 to 0.3. At a logit scale of 10 the right sentence gets a probability of about 0.999. At a scale of 1 it falls to about 0.53, and the loss rises from about 0.001 to about 0.649. Why the scale matters is the next topic.

### The contrastive loss

The loss turns the table into a number that training can reduce. It has two steps, and each is a classification problem you already know. First scale the cosines to make [[logits|The raw scores that go into a softmax. Here each logit is a cosine multiplied by the logit scale.]]. Second, apply softmax (Week 1) along each row to ask “which sentence fits this picture?”, and along each column to ask “which picture fits this sentence?”. The loss is the average of −ln of the probability that the diagonal gets.

```viz
{
  "type": "equation",
  "title": "The logits of an image–text pair",
  "intro": "Select a part to read it, then follow the worked example. The scale is the logit scale that the slider in the figure above changes.",
  "formula": [ { "t": "s", "id": "s" }, { "t": "ij", "id": "ij" }, " = ", { "t": "scale", "id": "scale" }, " × ", { "t": "cos(imageᵢ, textⱼ)", "id": "cos" } ],
  "parts": [
    { "id": "s", "name": "s (logit)", "meaning": "The score that goes into the softmax for picture i and sentence j." },
    { "id": "ij", "name": "ij (row i, column j)", "meaning": "Row i is a picture and column j is a sentence. The diagonal i = j holds the true pairs." },
    { "id": "scale", "name": "scale (logit scale)", "meaning": "A number that stretches the cosines before the softmax. It equals 1 divided by the temperature. A small temperature (a large scale) makes the softmax sharp." },
    { "id": "cos", "name": "cos(imageᵢ, textⱼ)", "meaning": "The cosine similarity of the vector of picture i and the vector of sentence j." }
  ],
  "steps": [
    { "title": "Two pictures and two sentences", "expr": "images [1, 0] and [0.6, 0.8]; texts [0.8, 0.6] and [0, 1]; all four vectors have length 1", "note": "The 2 × 2 toy used in the next figure. Pair 1 is (image 1, text 1) and pair 2 is (image 2, text 2)." },
    { "title": "Cosines", "expr": "cos matrix = [ [0.8, 0], [0.96, 0.8] ]", "note": "For example image 2 with text 1: 0.6×0.8 + 0.8×0.6 = 0.96. Image 2 is closer to the wrong text (0.96) than to its own (0.8)." },
    { "title": "Logits with scale 10", "expr": "s = [ [8, 0], [9.6, 8] ]", "note": "Every cosine is multiplied by 10." }
  ]
}
```

Now the loss itself and its worked example, with the same toy numbers.

```viz
{
  "type": "equation",
  "title": "The symmetric contrastive loss",
  "intro": "Select a part to read it, then follow the worked 2 × 2 example. This is the idea of the loss of CLIP, which belongs to the family of InfoNCE losses.",
  "formula": [ { "t": "L", "id": "L" }, " = −(1 / 2", { "t": "N", "id": "N" }, ")  ", { "t": "Σᵢ", "id": "sum" }, " [ ", { "t": "ln pᵢᵢ (image → text)", "id": "pi" }, "  +  ", { "t": "ln pᵢᵢ (text → image)", "id": "pt" }, " ]" ],
  "parts": [
    { "id": "L", "name": "L (the loss)", "meaning": "One number for the whole batch. Lower is better: it is small when the diagonal wins everywhere." },
    { "id": "N", "name": "N (batch size)", "meaning": "The number of pairs in the batch. The 2N in the denominator averages the two directions over all pairs." },
    { "id": "sum", "name": "Σᵢ (sum over pairs)", "meaning": "Add the terms for i = 1 up to N, one for each true pair." },
    { "id": "pi", "name": "ln pᵢᵢ (image → text)", "meaning": "The natural log of the probability that the softmax over row i gives to the true text. It is 0 when that probability is 1 and large and negative when it is small." },
    { "id": "pt", "name": "ln pᵢᵢ (text → image)", "meaning": "The same for column i: the probability the softmax over the pictures gives to the true picture." }
  ],
  "steps": [
    { "title": "Row softmax (image → text)", "expr": "row 1: logits [8, 0] give p11 = 0.9997; row 2: logits [9.6, 8] give p22 = 1 ÷ (1 + e^1.6) = 0.168", "note": "e^1.6 ≈ 4.953. Image 2 only gives 0.168 to its own text." },
    { "title": "Column softmax (text → image)", "expr": "col 1: logits [8, 9.6] give p11 = 0.168; col 2: logits [0, 8] give p22 = 0.9997", "note": "Text 1 gives only 0.168 to its own image because image 2 matches it better." },
    { "title": "Average of −ln", "expr": "image → text: (0.0003 + 1.784) ÷ 2 = 0.892; text → image: (1.784 + 0.0003) ÷ 2 = 0.892; L = 0.892", "note": "A loss of about 0.892 flags the confusion between the two pairs. A perfect diagonal would give a value near 0. In Lab Part 6.1 you compute exactly this with NumPy and check it against F.cross_entropy." }
  ]
}
```

The loss is ==symmetric== because it looks at the table twice, by rows and by columns. If the image says “my best caption is the right one” but the caption says “my best picture is somebody else”, the loss notices.

### Temperature and the logit scale

Why multiply by a scale at all? Cosines live between −1 and 1, and a softmax of numbers that close together is almost flat: even a perfect pair gets only a modest probability. Multiplying by a large scale spreads the logits out so the right answer can win clearly. The scale is the inverse of a *temperature*, the same dial as in Week 1: ==a small temperature, or a large scale, gives a sharp softmax== and a large one gives a soft softmax. CLIP does not fix the scale by hand but learns it, and in the released model it ends up near 100 (you will measure it in Lab Part 6.5).

Look at a harder batch of four, which contains two look-alike items: a tricycle and a motorbike. Move the logit-scale slider from 1 to 100, and watch the loss and the probabilities.

```viz
{
  "type": "clipsim",
  "title": "Four pictures with two look-alikes",
  "intro": "Select the tricycle picture. Is its probability split between the two look-alike sentences? Then change the logit scale.",
  "scale": 10,
  "images": [
    { "label": "photo of a jeepney", "v": [0.9, 0.2, 0.1, 0.1] },
    { "label": "photo of a tricycle", "v": [0.2, 0.9, 0.7, 0.1] },
    { "label": "photo of a motorbike", "v": [0.1, 0.7, 0.9, 0.1] },
    { "label": "photo of a carabao", "v": [0.1, 0.1, 0.1, 0.9] }
  ],
  "texts": [
    { "label": "a photo of a jeepney", "v": [1, 0.1, 0, 0.1] },
    { "label": "a photo of a tricycle", "v": [0.1, 1, 0.5, 0] },
    { "label": "a photo of a motorbike", "v": [0, 0.5, 1, 0.1] },
    { "label": "a photo of a carabao", "v": [0, 0.1, 0.1, 1] }
  ],
  "note": "The vectors are illustrative, authored for teaching, not from a trained model.",
  "checks": [
    { "scale": 10, "cos": [[1, 1, 0.974], [1, 2, 0.889], [2, 1, 0.899], [3, 3, 0.994]], "prob": [[1, 1, 0.699], [1, 2, 0.301], [3, 3, 0.999]], "loss": 0.185 },
    { "scale": 1, "prob": [[1, 1, 0.347]], "loss": 0.958 },
    { "scale": 100, "prob": [[1, 1, 1.0]], "loss": 0.0002 }
  ]
}
```

At scale 10 the tricycle photo gets 0.699 for its own sentence and 0.301 for “motorbike”: the model hedges between the two look-alikes, and the loss is about 0.185. At scale 1 the loss is high, about 0.958, because the probabilities are nearly flat. At scale 100 the loss is almost 0 on this batch, but notice what that means: a sharp softmax is confident even when the cosines are only slightly different, so a wrong pair would be punished very hard. The scale is a balance, and Lab Part 6.2 plots it for five values.

### Why big batches help

A batch of N pairs gives each picture N − 1 wrong sentences to be told apart from. With N = 4 that is 3 wrong sentences, but with N = 32,768, a batch size that has been reported for CLIP, each picture is compared with over 32,000 wrong sentences at once. ==More wrong answers make the task harder==, and a harder task forces the encoders to learn finer differences than “jeepney or mango”. The wrong pairs inside the same batch are called [[negatives|The non-matching pairs in a contrastive batch. They tell the model what the true pair should not look like.]]; the more negatives per positive, the more the model must learn to tell similar things apart.

Sort the cells of a 4 × 4 batch into the two kinds.

```viz
{
  "type": "sorter",
  "title": "True pair or wrong pair?",
  "intro": "In a batch, picture k belongs with caption k. Decide for each cell whether it is a true pair (the diagonal) or a wrong pair.",
  "bins": [
    { "id": "pos", "label": "true pair (diagonal)" },
    { "id": "neg", "label": "wrong pair (off-diagonal)" }
  ],
  "cards": [
    { "name": "picture 1 with caption 1", "answer": "pos", "why": "Row 1, column 1: this is the picture's own caption, so it is on the diagonal." },
    { "name": "picture 2 with caption 4", "answer": "neg", "why": "Row 2, column 4 is off the diagonal: the caption belongs to picture 4." },
    { "name": "picture 3 with caption 3", "answer": "pos", "why": "Row 3, column 3 is on the diagonal." },
    { "name": "picture 4 with caption 1", "answer": "neg", "why": "Row 4, column 1 is off the diagonal." },
    { "name": "picture 4 with caption 4", "answer": "pos", "why": "Row 4, column 4 is on the diagonal." },
    { "name": "picture 2 with caption 3", "answer": "neg", "why": "Row 2, column 3. Even if the two items look alike, it is a wrong pair in this batch, which is why look-alikes make good negatives." },
    { "name": "picture 1 with caption 3", "answer": "neg", "why": "Row 1, column 3 is off the diagonal." },
    { "name": "picture 2 with caption 2", "answer": "pos", "why": "Row 2, column 2 is on the diagonal." }
  ]
}
```

==In a batch of 4 there are 4 true pairs and 12 wrong pairs.== In general there are N true pairs and N × N − N = N(N − 1) wrong ones, so the wrong pairs grow with the square of the batch size.

### Zero-shot classification

Here is the payoff. Once the two encoders are trained, you can classify a picture into categories CLIP was never trained to classify, without training on any labelled example of those categories. This is [[zero-shot classification|Classifying an input into classes by comparing it with text descriptions of the classes, without training on labelled examples of those classes.]]. The recipe has four steps. Write each class as a sentence with a [[prompt template|A sentence pattern with a gap for the class name, such as “a photo of a {label}”, used to turn a class name into text the model can encode.]]; encode the sentences; encode the picture; and pick the class whose sentence has the highest cosine with the picture.

```viz
{
  "type": "pipeline",
  "title": "Zero-shot classification with CLIP",
  "intro": "Press the step buttons. The classes are decided at the moment you write the prompts: no retraining is involved.",
  "nodes": [
    { "id": "classes", "label": "Class names", "sub": "jeepney, tricycle…" },
    { "id": "prompts", "label": "Prompts", "sub": "a photo of a …" },
    { "id": "te", "label": "Text encoder", "sub": "one vector per class" },
    { "id": "ie", "label": "Image encoder", "sub": "one vector" },
    { "id": "cos", "label": "Cosines", "sub": "image vs each class" },
    { "id": "ans", "label": "Answer", "sub": "highest wins" }
  ],
  "steps": [
    { "title": "Choose your classes", "body": "Write down the labels you care about. They can be any words: nothing was trained on these particular classes.", "active": ["classes"], "example": "jeepney, tricycle, motorbike, carabao." },
    { "title": "Turn each label into a sentence", "body": "Put each label into a template so that it looks like the captions CLIP saw during training.", "active": ["classes", "prompts"], "example": "“a photo of a jeepney”, “a photo of a tricycle”, “a photo of a motorbike”, “a photo of a carabao”." },
    { "title": "Encode the prompts", "body": "The text encoder turns each sentence into a vector in the shared space. These class vectors can be computed once and reused for every picture.", "active": ["prompts", "te"], "example": "4 class vectors." },
    { "title": "Encode the picture and compare", "body": "The image encoder gives the picture's vector, and the cosine with each class vector says how well each sentence fits.", "active": ["ie", "cos"], "example": "Tricycle photo: cosines [0.256, 0.974, 0.889, 0.222]." },
    { "title": "Pick the highest, or read the probabilities", "body": "The class with the largest cosine is the answer. To get probabilities, multiply the cosines by the logit scale and apply softmax over the classes.", "active": ["cos", "ans"], "example": "Scale 10: logits [2.56, 9.74, 8.89, 2.22] give probabilities [0.0005, 0.699, 0.301, 0.0004]: tricycle wins." }
  ]
}
```

==The classes appear only in the prompts==, so changing the categories means editing a list of words. Here is the rule in symbols with the worked example.

```viz
{
  "type": "equation",
  "title": "The zero-shot rule",
  "intro": "Select a part to read it, then follow the worked example for the tricycle photo.",
  "formula": [ { "t": "ŷ", "id": "y" }, " = ", { "t": "arg max", "id": "arg" }, { "t": "k", "id": "k" }, " ", { "t": "cos(image, prompt k)", "id": "cos" }, ",   ", { "t": "P(k)", "id": "pk" }, " = softmax(", { "t": "scale × cos", "id": "sc" }, ")" ],
  "parts": [
    { "id": "y", "name": "ŷ (predicted class)", "meaning": "The class that the model chooses for the picture." },
    { "id": "arg", "name": "arg max", "meaning": "Choose the argument (here the class) that gives the largest value." },
    { "id": "k", "name": "k (class index)", "meaning": "Runs over the list of classes you wrote prompts for." },
    { "id": "cos", "name": "cos(image, prompt k)", "meaning": "How well the picture matches the sentence for class k." },
    { "id": "pk", "name": "P(k)", "meaning": "The probability CLIP gives to class k, which is only a comparison among your listed classes." },
    { "id": "sc", "name": "scale × cos", "meaning": "The logits: cosines multiplied by the logit scale before the softmax." }
  ],
  "steps": [
    { "title": "Cosines with the four prompts", "expr": "cos = [0.256, 0.974, 0.889, 0.222] (jeepney, tricycle, motorbike, carabao)", "note": "Illustrative vectors from the four-picture figure above." },
    { "title": "Pick the largest", "expr": "ŷ = tricycle because 0.974 is the biggest", "note": "No labelled tricycle photos were needed: only a prompt." },
    { "title": "Probabilities at scale 10", "expr": "logits [2.56, 9.74, 8.89, 2.22]; e^logit ≈ [12.9, 16,943, 7,289, 9.2]; sum ≈ 24,254; P = [0.0005, 0.699, 0.301, 0.0004]", "note": "The probabilities add up to 1 across your prompts only: an image of something you did not list still gets spread over the classes you did list." }
  ]
}
```

Read the last note again: ==CLIP can only choose among the prompts you give it==. If a picture shows a mango and your list is jeepney, tricycle, motorbike and carabao, CLIP will still pick the least bad of the four. In Lab Part 6.6 you will test a red circle against “a photo of a dog” and see this for yourself.

### The wording of the prompt matters

The same picture can get different probabilities from different templates, because the text encoder reads the whole sentence. A bare word can be ambiguous, and a longer sentence gives it a context. Compare two prompts for the same image of a bird.

```viz
{
  "type": "flow",
  "title": "Same picture, two prompt templates",
  "modes": [
    {
      "name": "Bare label",
      "tagline": "The prompt is just the class name.",
      "inputLabel": "Prompts",
      "modelLabel": "CLIP text encoder + image encoder",
      "outputLabel": "Probabilities",
      "examples": [
        { "label": "Photo of a bird that eats fish", "input": "“crane” versus “bulb” versus “fan”", "output": "crane 0.46, bulb 0.31, fan 0.23 (the word “crane” can also mean a lifting machine)" }
      ],
      "note": "Hand-written, illustrative numbers to show the idea, not measured from a real model."
    },
    {
      "name": "Descriptive template",
      "tagline": "The prompt says what kind of thing it is.",
      "inputLabel": "Prompts",
      "modelLabel": "CLIP text encoder + image encoder",
      "outputLabel": "Probabilities",
      "examples": [
        { "label": "Same photo", "input": "“a photo of a crane, a large bird” versus “a photo of a light bulb” versus “a photo of a fan”", "output": "crane 0.89, bulb 0.06, fan 0.05" }
      ],
      "note": "Hand-written, illustrative numbers to show the idea, not measured from a real model. In Lab Part 6.5 you measure real differences between three templates."
    }
  ]
}
```

Try the other tab for the same photo. ==A little context in the prompt often helps==, but the effect is not guaranteed, so test your templates on a few pictures before you trust them.

### A picture of the shared space

Here is the idea as a map. Two-number coordinates are far too few for a real model, but they show what “near” means. The text points are sentences, and the photos are pictures. The nearest text point to a photo is its zero-shot answer. Select a photo point to see its nearest neighbours.

```viz
{
  "type": "scatter",
  "title": "A shared space for pictures and sentences",
  "metric": "cosine",
  "points": [
    { "label": "text: jeepney", "x": 4, "y": 1, "group": "text" },
    { "label": "text: carabao", "x": -1, "y": 4, "group": "text" },
    { "label": "text: mango", "x": 2, "y": -4, "group": "text" },
    { "label": "text: tricycle", "x": 3, "y": 3.2, "group": "text" },
    { "label": "photo: new jeepney", "x": 4.3, "y": 1.8, "group": "image" },
    { "label": "photo: new carabao", "x": -1.5, "y": 3.6, "group": "image" },
    { "label": "photo: new mango", "x": 1.5, "y": -3, "group": "image" }
  ],
  "note": "The coordinates are illustrative, authored for teaching. Real CLIP vectors have hundreds of numbers and cannot be drawn. By cosine similarity the new jeepney photo is closest to “text: jeepney” (0.989), then to “text: tricycle” (0.913)."
}
```

Notice that the new jeepney photo sits between “jeepney” and “tricycle”, because both are vehicles, and that the carabao photo is far from the mango. ==Distances in this space carry meaning==, which is the same promise as the word embeddings of Week 6, now across two modalities.

### What can go wrong

CLIP is powerful and it is easy to over-trust. Five honest limits.

- **It inherits its data.** The training pairs are captions written by people on the internet. They reflect what people choose to photograph and caption, in which languages and from which places, and they contain stereotypes. The model can pick up those patterns, so it can describe some groups or places more accurately than others.
- **Fine detail and counting can fail.** CLIP can separate “a dog” from “a car” easily, but it can struggle with fine distinctions (two similar bird species), with counting (three circles versus five) and with the position of things. Lab Part 6.6 asks you to test counting and report honestly what you see.
- **Reading text in an image is unreliable.** It sometimes matches words that appear in the picture, and sometimes it does not, so treat it with care.
- **Spurious cues.** A model can match on a background or a watermark rather than on the object, so it can be right for the wrong reason.
- **Privacy and consent.** Many training images were collected from the web without the people in them being asked. A photo can show a face, a school uniform or a house number. Before you use or collect pictures of people, ask whether they agreed and what could happen if the data leaked, and never use such a model to decide something serious about a person.

None of these limits means that CLIP is useless. They mean that ==a person must check its answers== and test it on the kind of data it will meet in your own setting.

### A cousin: SimCLR

You do not need captions to use a contrastive loss. SimCLR (Chen et al., 2020) takes one picture, makes two randomly altered copies of it (cropped, recoloured, blurred) and trains an encoder to ==put the two copies close together and other pictures far away==. The loss is the same shape as CLIP's, with cosine similarity and a temperature, but the pair is two views of one image instead of an image and its caption. UvA Tutorial 17 builds it on a small image dataset, and you can read about it under “Go further” in the lab.

## Plugging vision into language models {rail=Vision + LLM}

{lede} A chat model from Week 9 only knows how to read a sequence of token embeddings. To let it “see”, you do not rebuild it: you turn the picture into something that looks like more tokens and put them where the model can read them. In a LLaVA-style design (Liu et al., 2023) the recipe is ==encode the picture into patch features, project them into the language model's embedding space, and place them in front of the text tokens==. A Flamingo-style design (Alayrac et al., 2022) instead lets the language model look at the picture through new cross-attention layers.

### Option 1: projector and prefix tokens

Follow one picture. First a vision encoder (a ViT or the image encoder of CLIP, Sections 2 and 4) turns the picture into one feature vector per patch. These vectors live in the vision encoder's space, whose length may differ from the language model's hidden size. So a small learned layer, the [[projector|A small learned layer, often one or two linear layers, that maps the vision encoder's feature vectors into the embedding space of the language model so that they can be read as tokens.]], maps each feature vector to a vector of the language model's size. The projected vectors are then placed in front of the embeddings of the text prompt, and the language model reads the whole sequence and writes its answer one token at a time (Week 9).

```viz
{
  "type": "equation",
  "title": "The projector",
  "intro": "Select a part to read it, then follow the worked example. The sizes are an example configuration, not any specific model.",
  "formula": [ { "t": "T", "id": "T" }, " = ", { "t": "F", "id": "F" }, " · ", { "t": "W", "id": "W" }, " + ", { "t": "b", "id": "b" } ],
  "parts": [
    { "id": "T", "name": "T (image tokens for the language model)", "meaning": "A matrix with one row per patch. Each row has the language model's hidden size, so the language model can treat it like a token embedding." },
    { "id": "F", "name": "F (vision features)", "meaning": "A matrix with one row per patch, produced by the vision encoder. Its width is the vision encoder's hidden size." },
    { "id": "W", "name": "W (projector weights)", "meaning": "A learned matrix that maps the vision width to the language-model width. In its simplest form, it is a single linear layer." },
    { "id": "b", "name": "b (bias)", "meaning": "A learned vector added to every row." }
  ],
  "steps": [
    { "title": "Patch features from the vision encoder", "expr": "a 336 × 336 photo with 14 × 14 patches gives (336 ÷ 14)² = 24 × 24 = 576 patches; with d_vision = 1024, F has shape (576, 1024)", "note": "An example configuration. We compute 336 ÷ 14 = 24 and 24 × 24 = 576." },
    { "title": "Project into the language-model width", "expr": "W has shape (1024, 4096); T = F·W + b has shape (576, 4096)", "note": "Each of the 576 patches becomes a token of the language model's width, d_model = 4096." },
    { "title": "Count the projector's parameters", "expr": "one linear layer: 1024 × 4096 + 4096 = 4,198,400; a two-layer version with a 4096-wide hidden layer adds 4096 × 4096 + 4096 = 16,781,312, for 20,979,712 in all", "note": "Tiny next to a language model with billions of parameters, which is why only this part may need new training." }
  ]
}
```

==A projector is small compared with the models it connects==, because it only has to translate between two spaces that already contain good representations. Step through the whole path with shapes. The configuration is an example, chosen so that the numbers are easy to follow, and it is not the setting of any particular model.

```viz
{
  "type": "pipeline",
  "title": "From a photo to an answer: projector and prefix tokens",
  "intro": "Press the step buttons. Every shape is computed from the example configuration in the text.",
  "nodes": [
    { "id": "photo", "label": "Photo", "sub": "336 × 336 × 3" },
    { "id": "venc", "label": "Vision encoder", "sub": "(576, 1024)" },
    { "id": "proj", "label": "Projector", "sub": "(576, 4096)" },
    { "id": "text", "label": "Text prompt", "sub": "(32, 4096)" },
    { "id": "seq", "label": "Joined sequence", "sub": "(608, 4096)" },
    { "id": "llm", "label": "Language model", "sub": "writes the answer" }
  ],
  "steps": [
    { "title": "A photo and a question arrive", "body": "The photo is resized to the size the vision encoder expects. The question is tokenised and embedded as in Week 6.", "active": ["photo", "text"], "example": "Question: “What is written on the signboard?” turned into 32 prompt tokens, including any instructions." },
    { "title": "The vision encoder makes patch features", "body": "The ViT cuts the photo into patches and runs its Transformer. The output is one feature vector per patch.", "active": ["photo", "venc"], "example": "336 ÷ 14 = 24 patches per side, so 24 × 24 = 576 patches; d_vision = 1024; shape (576, 1024)." },
    { "title": "The projector translates", "body": "A learned layer maps the 1024-number features to the language model's 4096-number embeddings.", "active": ["venc", "proj"], "example": "Shape (576, 1024) becomes (576, 4096)." },
    { "title": "Join the image tokens and the text tokens", "body": "The image tokens are placed in front of the prompt tokens, and the language model sees one long sequence.", "active": ["proj", "text", "seq"], "example": "576 image tokens + 32 text tokens = 608 tokens." },
    { "title": "The language model answers", "body": "As in Week 9, the decoder predicts the next token again and again. Because it attends to the image tokens, the answer can depend on the picture.", "active": ["seq", "llm"], "example": "The image tokens are only the context: no image is created, and the answer is plain text." }
  ]
}
```

Notice that the language model is not told “here is a picture”. It simply ==finds 576 extra vectors in front of its prompt and learns to use them== That is why the training method matters, and it comes up next.

### The cost: image tokens use up the context window

Every image token takes space in the context window of Week 9 and adds to the attention cost. Select each part to read it, and follow the numbers for the example configuration.

```viz
{
  "type": "equation",
  "title": "How much of the context an image uses",
  "intro": "Select a part to read it, then follow the worked example with a 4,096-token context.",
  "formula": [ { "t": "n", "id": "n" }, " = ", { "t": "n_img", "id": "ni" }, " + ", { "t": "n_text", "id": "nt" }, ",   ", { "t": "share", "id": "share" }, " = ", { "t": "n_img", "id": "ni2" }, " ÷ ", { "t": "context", "id": "ctx" } ],
  "parts": [
    { "id": "n", "name": "n (sequence length)", "meaning": "The total number of tokens the language model reads." },
    { "id": "ni", "name": "n_img (image tokens)", "meaning": "How many projected patch tokens the picture adds. It is the number of patches." },
    { "id": "nt", "name": "n_text (text tokens)", "meaning": "How many tokens the prompt has." },
    { "id": "share", "name": "share", "meaning": "The fraction of the context window that the picture occupies." },
    { "id": "ni2", "name": "n_img (again)", "meaning": "The number of image tokens." },
    { "id": "ctx", "name": "context", "meaning": "The size of the context window, the most tokens the model can read at once." }
  ],
  "steps": [
    { "title": "Sequence length", "expr": "n = 576 + 32 = 608", "note": "One picture and a short prompt." },
    { "title": "Share of a 4,096-token context", "expr": "share = 576 ÷ 4096 = 0.1406, about 14%", "note": "One picture uses about 14% of the context. Two pictures would use 1152 ÷ 4096 = 28.1%." },
    { "title": "Attention scores per head", "expr": "608 × 608 = 369,664, against 32 × 32 = 1,024 for the text alone: about 361 times as many", "note": "Adding one picture makes each attention layer of the language model much more costly, which is why image resolution matters." }
  ]
}
```

==One picture takes roughly 14% of this context and increases the attention work by a factor of about 361.== This is why vision-language chat systems limit the number or the resolution of the images you can send.

### Option 2: cross-attention

There is another way to give a language model access to a picture without using up its context. Insert extra attention layers into the language model in which the **queries** come from the text tokens and the **keys and values** come from the image features. This is [[cross-attention|Attention in which the queries come from one sequence and the keys and values come from another. Here the text asks questions and the image features answer them.]]: it is the attention of Week 7 with its two inputs coming from different places. In Flamingo (Alayrac et al., 2022), these layers were added to a pretrained language model whose original layers stayed [[frozen|Kept fixed during training: their numbers are not updated, only the new parts learn.]]. Because the picture is read through the keys and values, ==it does not take any room in the sequence of text tokens==.

Compare the designs side by side, now including a plain “describe then ask” chain.

```viz
{
  "type": "fusion",
  "title": "Three ways to let a language model use a picture",
  "intro": "Switch tabs and compare where the picture enters the language model.",
  "patterns": [
    {
      "name": "Projector and prefix tokens",
      "inputs": ["image", "text prompt"],
      "stages": ["vision encoder", "projector", "language model reads all tokens"],
      "output": "text answer",
      "how": "Patch features are projected to the language model's width and placed in front of the prompt as extra tokens.",
      "examples": "LLaVA-style chat models.",
      "good": "Simple. Only the projector must be learned from scratch, so a small amount of training data can already work.",
      "bad": "Each picture uses hundreds of tokens of context, which makes long prompts and many images expensive."
    },
    {
      "name": "Cross-attention layers",
      "inputs": ["image", "text"],
      "stages": ["vision encoder", "cross-attention inside the language model", "language model"],
      "output": "text answer",
      "how": "Text tokens ask, via cross-attention, and the image features answer; the language model's own layers can be frozen.",
      "examples": "Flamingo-style models.",
      "good": "The picture does not use space in the token sequence, and the language model can keep what it already knows.",
      "bad": "New layers must be added and trained, so the design is more complex than a projector."
    },
    {
      "name": "Describe, then ask (a chain)",
      "inputs": ["image"],
      "stages": ["captioning model writes text", "language model reads the text", "language model answers"],
      "output": "text answer",
      "how": "A first model turns the picture into a caption or a list of facts. A text-only language model then answers from that text. It is late fusion at the level of words.",
      "examples": "Simple pipelines built from separate models.",
      "good": "Easy to build and to inspect, because you can read the middle text.",
      "bad": "Whatever the caption leaves out is lost for ever, and a wrong caption sends the language model in the wrong direction."
    }
  ],
  "note": "A simplified overview. Real systems mix and refine these ideas."
}
```

The riskiest weakness of the chain is easy to see: ==the language model never sees the picture==. The first two options at least give the model the picture itself.

### How such a model is trained

A good way to see the training of a LLaVA-style model is in two stages. In the first, the vision encoder and the language model are left as they are, and only the projector is trained on pairs of pictures and captions, so that the picture tokens end up in a place where the language model understands them: the projector is being *aligned*. In the second, the model is trained on examples of the kind of conversation you want, such as a picture with a question and a good answer; this is [[instruction tuning|Training a model on examples of instructions and good responses so that it follows requests instead of only continuing text. Here the examples include pictures.]] (Week 9). Training only the small part first keeps the cost low.

A model that has been trained this way learns to turn the extra tokens into descriptions, answers and reasoning about the picture. But it may also learn ==to produce text that sounds right without being tied to what is in the picture==

### When the answer is not in the picture: hallucination

Vision-language models can [[hallucinate|To produce fluent output that is not supported by the input, such as a caption that mentions an object that is not in the picture.]]. A caption may mention a person who is not there, read the wrong digit on a gauge, or state a confident number of chairs when it cannot count. The sentence reads well, which is ==exactly what makes the error hard to notice== Three habits reduce the harm: (1) ask for evidence (“Where in the picture do you see that?”); (2) check anything important with the original picture or with a person who can look at it; and (3) never let a model alone decide something that affects safety, health or money.

Pictures also carry private information: faces, school uniforms, house numbers and the places where photos were taken. Before sending a picture to a service, ask whether the people in it agreed, whether a name, a face or a location would be exposed, and whether a service that keeps the picture is acceptable. For photos of students in particular, follow your school's rules about consent.

How do we know whether such a system works? Two common ways are named here. *Visual question answering benchmarks* score the answers against reference answers, and captions are judged by *humans* or by automatic metrics. Both are rough, and a high score on a benchmark does not promise good behaviour on your own photos, so build a small test set from your own setting.

### Which part does what?

Sort these components by what they use.

```viz
{
  "type": "sorter",
  "title": "Vision encoder, language model, or both?",
  "intro": "Decide which part of a LLaVA-style assistant each item belongs to.",
  "bins": [
    { "id": "vis", "label": "uses a vision encoder" },
    { "id": "llm", "label": "uses a language model" },
    { "id": "both", "label": "connects or involves both" }
  ],
  "cards": [
    { "name": "Cutting the photo into 14 × 14 patches", "answer": "vis", "why": "Patchifying is the front end of the vision encoder, a ViT." },
    { "name": "Position embeddings for the image patches", "answer": "vis", "why": "They are added inside the vision encoder so that it knows where each patch came from." },
    { "name": "Predicting the next word of the answer", "answer": "llm", "why": "Next-token prediction is the job of the language model (Week 9)." },
    { "name": "A tokeniser that splits the question into pieces", "answer": "llm", "why": "Text tokenisation belongs to the language-model side (Week 6)." },
    { "name": "The projector layer", "answer": "both", "why": "It takes the vision encoder's features and produces vectors in the language model's space, so it connects the two." },
    { "name": "Cross-attention layers inserted into the language model", "answer": "both", "why": "The queries come from the language model and the keys and values from the image features." },
    { "name": "Image tokens sitting in the context window beside the prompt", "answer": "both", "why": "They came from the vision encoder and are read by the language model." },
    { "name": "The CLS token of the ViT", "answer": "vis", "why": "It is a learned vector inside the vision encoder that summarises the picture." }
  ]
}
```

A projector or a cross-attention layer is the “glue”; most of the intelligence stays in the two big parts that were trained separately. One last comparison: ask “What is in this picture?” of a text-only model and a vision-language model.

```viz
{
  "type": "flow",
  "title": "“What is in this picture?”",
  "modes": [
    {
      "name": "Text-only model",
      "tagline": "It only sees the words of the question.",
      "inputLabel": "Question only",
      "modelLabel": "language model",
      "outputLabel": "Answer",
      "examples": [
        { "label": "Flood photo", "input": "What is in this picture? (no picture is given)", "output": "I cannot see any picture. Please describe it or attach it." }
      ],
      "note": "Hand-written, illustrative output. A real text-only model might invent an answer instead, which is a form of hallucination."
    },
    {
      "name": "Vision-language model",
      "tagline": "It reads the picture tokens and the question together.",
      "inputLabel": "Picture + question",
      "modelLabel": "vision encoder + projector + language model",
      "outputLabel": "Answer",
      "examples": [
        { "label": "Flood photo", "input": "Photo of a flooded street with a gauge + “What is in this picture?”", "output": "A street covered with brown water. A measuring gauge stands at the side. A few people are walking at the far end. I may have missed or misread details." }
      ],
      "note": "Hand-written, illustrative output. Check any detail that matters against the picture itself."
    }
  ]
}
```

Switch the tab and compare what each model is able to say, and what each model might get wrong. A vision-language model has more to say, which also means more things it can get wrong.

## Design a multimodal system {rail=Design}

{lede} Knowing the parts is only half the skill; the other half is deciding which parts a real problem needs and what could go wrong. ==Design starts from the problem, not from the model==: first decide what is going in and coming out, then choose how to combine the modalities, then plan the data, the checks and the way failures are handled. The last part of the lab (Part 7) asks you to do this for a problem from your own school or community.

### Five questions to answer in order

1. **Which modalities and which task?** Write the inputs and the outputs (for example “photo and a short text report in, a risk level out”). If a single modality is enough, use it: it is cheaper and simpler.
2. **Which way of combining them?** A shared space like CLIP's (search and matching), a projector into a language model (questions and descriptions), cross-attention (long text with pictures), or late fusion of separate models (decisions combined at the end).
3. **Which data?** Who took the pictures and wrote the texts? Did the people shown agree? Does the data cover the lighting, the dialects, the devices and the places where the system will really be used? ==Gaps in the data become gaps in the behaviour.==
4. **Which evaluation?** Choose a metric for each modality (for example word error rate for speech, accuracy for labels, a human rating for captions) and a human check. Look at results *per group* (for example per school, per device or per kind of photo), because an average can hide a group that is served badly.
5. **Which failure handling?** Decide in advance what happens when the model is unsure or wrong: a confidence threshold below which the case goes to a person, a way to report mistakes, and a record of what the system did.

### Three mini-designs

Here are three designs for a school or a community. They are examples to learn from, not finished products.

| Problem | Modalities (in → out) | Fusion pattern | Data | Metric | Main risk |
|---|---|---|---|---|---|
| Alt text for photos in a school newsletter, for blind students | photo → short text | projector into a language model (a captioning model) | newsletter photos with permission slips; captions written by volunteers as references | human rating of each caption for accuracy and usefulness, plus a check that no invented names appear | wrong or invented details, or naming a student without consent |
| Flood level from a gauge photo plus a text report | photo + text → low, medium or high | late or early fusion of image features and text features | past gauge photos with the reading recorded by a barangay officer and the text reports | recall for the high class (how many dangerous cases are caught), accuracy per camera and per lighting condition | missing a high flood, or being wrong at night or in heavy rain |
| Voice and text helpdesk for enrolment questions | audio → text → answer | a chain: speech recognition, then a language model | voice messages in Filipino, English and mixed speech, with consent; the school's own FAQ | word error rate for the speech step; correctness of answers rated by staff | mishearing names or numbers; accents served worse than others |

Notice that each risk is specific to the setting. In the second design, the cost of a missed flood is much greater than the cost of a false alarm, which changes the metric you should watch. That is why the choice of metric is a human decision.

```viz
{
  "type": "equation",
  "title": "Recall: how many dangerous cases are caught?",
  "intro": "Select a part to read it, then follow the worked example for a flood classifier. The counts are made up for teaching.",
  "formula": [ { "t": "recall", "id": "recall" }, " = ", { "t": "TP", "id": "tp" }, " ÷ (", { "t": "TP", "id": "tp2" }, " + ", { "t": "FN", "id": "fn" }, ")" ],
  "parts": [
    { "id": "recall", "name": "recall", "meaning": "Of all the cases that really are in the class, the share that the system catches. 1 is perfect and 0 means it caught none." },
    { "id": "tp", "name": "TP (true positives)", "meaning": "Cases that really are high floods and that the system labelled high." },
    { "id": "tp2", "name": "TP (again)", "meaning": "The same count, in the denominator together with the misses." },
    { "id": "fn", "name": "FN (false negatives)", "meaning": "Cases that really are high floods but that the system labelled something lower: the dangerous misses." }
  ],
  "steps": [
    { "title": "Count on a test set", "expr": "20 photos really show a high flood; the system labels 17 of them high and 3 of them medium", "note": "Made-up counts. TP = 17 and FN = 3." },
    { "title": "Compute the recall", "expr": "recall = 17 ÷ (17 + 3) = 17 ÷ 20 = 0.85", "note": "The system catches 85% of the high floods and misses 15%." },
    { "title": "Decide what is acceptable", "expr": "3 missed floods out of 20 is 15%: probably too many for a warning system", "note": "A warning system should be tested for its misses first. A human officer must stay in the loop, and a low-confidence case goes to the officer." }
  ]
}
```

==Choose the metric for the harm that costs most.== For the flood design that is recall for the high class; for the newsletter captions it is the human rating. Next, follow the flood design from photo to decision.

```viz
{
  "type": "pipeline",
  "title": "A flood-level classifier with a human in the loop",
  "intro": "Press the step buttons. The sizes are an example design: 224 × 224 photo, 16 × 16 patches and 768-number summary vectors.",
  "nodes": [
    { "id": "photo", "label": "Gauge photo", "sub": "(3, 224, 224)" },
    { "id": "report", "label": "Text report", "sub": "tokens" },
    { "id": "venc", "label": "Image encoder", "sub": "CLS vector (768)" },
    { "id": "tenc", "label": "Text encoder", "sub": "vector (768)" },
    { "id": "fuse", "label": "Join + classify", "sub": "1536 → 3" },
    { "id": "human", "label": "Human check", "sub": "if unsure" }
  ],
  "steps": [
    { "title": "Collect both inputs", "body": "A camera at the gauge gives a photo, and a barangay volunteer types a short report. Both are stored with the time.", "active": ["photo", "report"], "example": "Report: “Water is rising near the school gate.”" },
    { "title": "Encode each modality", "body": "A ViT turns the photo into one summary vector (the CLS vector of Section 2). A text encoder turns the report into a vector of the same size.", "active": ["photo", "report", "venc", "tenc"], "example": "196 patches + 1 CLS token go through the ViT; the output we keep is 768 numbers. The text vector has 768 numbers as well." },
    { "title": "Join and classify (early fusion)", "body": "Glue the two vectors end to end into one vector of 1,536 numbers and feed it to a small linear layer that gives three scores: low, medium, high.", "active": ["venc", "tenc", "fuse"], "example": "768 + 768 = 1,536 inputs; a linear layer to 3 classes has 1,536 × 3 + 3 = 4,611 parameters." },
    { "title": "Send unsure cases to a person", "body": "If the highest probability is below a threshold, or the class is high, the case goes to an officer who looks at the original photo. The model never decides alone.", "active": ["fuse", "human"], "example": "For example: any case with the highest probability under 0.70, and every case predicted high, is shown to an officer." }
  ]
}
```

Notice the position of the human: not at the end of every case, but ==exactly where the model is unsure and where a mistake costs most==. Planning the size of the input is part of the design too, because the number of image tokens controls the cost.

```viz
{
  "type": "equation",
  "title": "Token budget of an image",
  "intro": "Select a part to read it, then follow the worked example. You will repeat this calculation for your own design in Lab Part 7.2.",
  "formula": [ { "t": "n", "id": "n" }, " = (", { "t": "H", "id": "H" }, " ÷ ", { "t": "P", "id": "P" }, ")² + ", { "t": "n_prompt", "id": "np" }, ",   ", { "t": "scores", "id": "sc" }, " = ", { "t": "n²", "id": "n2" } ],
  "parts": [
    { "id": "n", "name": "n (sequence length)", "meaning": "The number of image tokens plus the number of prompt tokens that a model with prefix image tokens has to read." },
    { "id": "H", "name": "H (image side)", "meaning": "The side of the square image in pixels." },
    { "id": "P", "name": "P (patch side)", "meaning": "The side of a patch in pixels." },
    { "id": "np", "name": "n_prompt (prompt tokens)", "meaning": "The tokens of the text prompt that go with the image." },
    { "id": "sc", "name": "scores", "meaning": "The attention scores per head in one layer." },
    { "id": "n2", "name": "n² (squared)", "meaning": "n times n, because every token is compared with every token." }
  ],
  "steps": [
    { "title": "224 × 224 photo, 16 × 16 patches, a 64-token prompt", "expr": "n = (224 ÷ 16)² + 64 = 196 + 64 = 260; scores = 260² = 67,600", "note": "A modest budget." },
    { "title": "The same photo with 8 × 8 patches", "expr": "n = (224 ÷ 8)² + 64 = 784 + 64 = 848; scores = 848² = 719,104", "note": "Four times as many image tokens." },
    { "title": "Compare", "expr": "719,104 ÷ 67,600 ≈ 10.6", "note": "Halving the patch side makes the attention about 10.6 times as expensive here (not 16, because the prompt tokens are the same in both)." }
  ]
}
```

Try the sorter below to practise the first design question: is a single modality enough?

```viz
{
  "type": "sorter",
  "title": "Multimodal, or is one modality enough?",
  "intro": "Decide whether the problem really needs more than one kind of data.",
  "bins": [
    { "id": "multi", "label": "good fit for a multimodal model" },
    { "id": "single", "label": "a single modality is enough" }
  ],
  "cards": [
    { "name": "Write alt text for photos in the school newsletter", "answer": "multi", "why": "A picture goes in and text comes out, so two modalities are involved." },
    { "name": "Predict tomorrow's rainfall from the last 30 days of rainfall numbers", "answer": "single", "why": "The input is one list of numbers over time. A time-series model is enough." },
    { "name": "Answer questions about a chart in a screenshot", "answer": "multi", "why": "The model needs to read an image and a question together." },
    { "name": "Sort student survey comments into positive and negative", "answer": "single", "why": "Both the data and the labels are text. A language model or classifier is enough." },
    { "name": "Check whether a flood photo matches the text report that came with it", "answer": "multi", "why": "The decision needs both the picture and the sentence, and a shared space like CLIP's fits the job." },
    { "name": "Spell-check an essay", "answer": "single", "why": "Text only. No picture or sound is involved." },
    { "name": "Transcribe voice messages sent to the school office", "answer": "multi", "why": "Audio goes in and text comes out. Speech recognition links two modalities." },
    { "name": "Translate a sentence from English to Filipino", "answer": "single", "why": "Text to text: a language model alone can do it." }
  ]
}
```

If a card surprised you, notice what rule it hides: ==use extra modalities only when the answer needs information that one modality cannot carry==

## Key takeaways

- A ==multimodal model takes in, produces or links more than one kind of data==, such as text, images and audio. Three real applications are image captioning (image → text), visual question answering (image + text → text) and speech recognition (audio → text); every one needs a human check when the answer matters.
- A ViT ==cuts an image into P × P patches, flattens each, projects it with one shared matrix, adds a CLS token and position embeddings== and sends the 197 tokens (for a 224 × 224 image with 16 × 16 patches) through an ordinary Transformer. This cuts the attention cost from about 2.5 billion scores to 38,809 per head.
- Self-attention cannot see order, so ==a bag of patches needs position embeddings==, vectors added to the patch embeddings, to tell where each patch came from.
- CLIP trains an image encoder and a text encoder with a ==contrastive loss so that a picture and its caption share one space==. Cosine similarity then gives zero-shot classification: write the classes as prompts and pick the closest one. It can only choose among the prompts you give it.
- Vision can be plugged into a language model by ==projecting patch features into its embedding space as extra tokens== (or by cross-attention). The costs are context tokens and hallucination, so design with the data, the checks and the human in mind.

<!-- LAB-START -->
## Lab: Imagine a model that sees and reads {rail=Laboratory}

This lab is called *Imagine a model that sees and reads*. Week 11 asks how a Transformer, which reads sequences of tokens, can look at a picture, and how a picture and a sentence can end up in the same space. You answer by building the story from the bottom, with pictures that you draw yourself. *What if each modality alone is ambiguous?* You build a toy problem in which the colour alone and the shape alone each leave you guessing, and fuse them. *What if we cut a photograph into a jigsaw of small squares and read them one by one?* You patchify pictures by hand, count the tokens of a real Vision Transformer, and prove that a bag of patches has no layout, so that position embeddings are needed. Then you build a tiny Vision Transformer of your own and run ablations on positions, pooling and patch size, train a miniature CLIP from scratch with the contrastive loss, and try the real CLIP on clean pictures. You finish by designing a multimodal system for your school or community and reflecting on its risks. ==The goal is to replace the mystery of "AI that sees and reads" by numbers that you counted and measured yourself.== Every part begins with an **Imagine first** moment: you write a prediction before you run any code and compare it with what really happens. Each part points back to a section of the lesson. Only Part 6 downloads anything (the real CLIP, about 600 MB, from the offline cache on the Spark); everything else is drawn with NumPy, runs offline and finishes in seconds. This page contains the setup and the first three parts (modalities and fusion, patches, positions); the other four parts, the deliverables and the marking scheme follow in the same lab.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cells, run a smoke test | 10 |
| 1 | What is multimodal? Modalities, fusion and complementarity | 25 |
| 2 | Patchify: from pixels to tokens | 20 |
| 3 | Why position embeddings? | 25 |
| 4 | Build a tiny Vision Transformer | 30 |
| 5 | ViT experiments: positions, pooling, patch size | 25 |
| 6 | Contrastive learning and CLIP: from scratch and the real one | 30 |
| 7 | Design a multimodal system, and reflect | 15 |

The total is 180 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week11` and a notebook `lab11.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`Xc`, `patchify`, `vit_counts`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cells and then the blocks again from the top (every part takes seconds, so this is quick).
2. Do not install packages: PyTorch, NumPy, Matplotlib, Pillow and `transformers` are already installed. Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook. The only model you need is the real CLIP, `openai/clip-vit-base-patch32`, in Part 6. It comes from the **offline Hugging Face cache** on the Spark: confirm with your instructor that the cache is present and set up for you. If the model cannot be found, the loading cell stops with a clear message instead of a long error. In Colab the same model is downloaded once (about 600 MB). No other download is ever needed: the three picture data sets are drawn with NumPy and PIL.
3. A CPU is enough. Every block of Parts 1 to 3 finishes in a few seconds; the slowest blocks of the later parts (training a small Vision Transformer, loading CLIP) take up to a minute or two. The GPU of the Spark is shared by the whole class and gives no advantage on models and pictures this small, so leave it alone. The helper cell prints which device it found.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste these four blocks into four cells and run them. The first fixes the random seed (so your numbers match the ones printed in this lab), defines the empty `RESULTS` dictionary and the `check(...)` function. The second defines `make_shapes`, the five line shapes of Week 4, the parameter counter and a trainer for any PyTorch classifier. The third defines the two new data sets: `make_colored_shapes` (a red, green or blue square, circle or cross with a caption such as "red circle") and `make_left_right` (a plus sign and a distracting square at random places; the label says whether the plus sign is in the left half). The fourth defines `show_images`, `draw_shape_pil` (a clean 224 by 224 picture of a coloured shape for CLIP) and `load_clip`:

```python
import os, gc, math, random, re, time, warnings
from collections import Counter
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
```

```python
SHAPE_NAMES = ["horizontal bar", "vertical bar", "cross", "square", "diagonal"]

def _draw(kind, size, r, shift):
    P = size                                   # a big blank canvas, so a shifted shape may leave the picture without an error
    img = np.zeros((size + 2 * P, size + 2 * P), np.float32)
    c = P + size // 2
    dy, dx = (r.randint(-shift, shift + 1, 2) if shift else (0, 0))
    h = size // 4                      # half-length of a shape
    if kind == 0:   img[c + dy, c - h + dx:c + h + 1 + dx] = 1                          # horizontal bar
    elif kind == 1: img[c - h + dy:c + h + 1 + dy, c + dx] = 1                          # vertical bar
    elif kind == 2:
        img[c + dy, c - h + dx:c + h + 1 + dx] = 1; img[c - h + dy:c + h + 1 + dy, c + dx] = 1   # cross
    elif kind == 3:
        y0, y1, x0, x1 = c - h + dy, c + h + dy, c - h + dx, c + h + dx
        img[y0, x0:x1 + 1] = 1; img[y1, x0:x1 + 1] = 1; img[y0:y1 + 1, x0] = 1; img[y0:y1 + 1, x1] = 1       # square outline
    else:
        for t in range(-h, h + 1):
            img[c + t + dy, c + t + dx] = 1                                             # diagonal
    return img[P:P + size, P:P + size]

def make_shapes(n=300, size=16, noise=0.15, shift=0, seed=0):
    """n pictures of five shape kinds (labels 0..4, see SHAPE_NAMES). shift = how many pixels the shape may move from the centre in each direction;
    noise = standard deviation of Gaussian noise. Returns X float32 of shape (n, 1, size, size) and y int64 of shape (n,)."""
    r = np.random.RandomState(seed)
    y = np.arange(n) % 5
    r.shuffle(y)
    X = np.stack([_draw(int(k), size, r, shift) for k in y])
    X = X + r.normal(0, noise, X.shape).astype(np.float32)
    return X[:, None].astype(np.float32), y.astype(np.int64)

def count_params(model):
    """Number of trainable numbers (weights and biases) in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_classifier(model, Xtr, ytr, Xte, yte, epochs=30, lr=0.01, opt="adam", batch=64, seed=0, verbose=False):
    """Train with cross-entropy in mini-batches. Returns a dict with per-epoch lists: loss, train_acc, test_acc (and the final model is trained in place)."""
    g = torch.Generator().manual_seed(seed)
    Xtr, ytr = torch.as_tensor(Xtr), torch.as_tensor(ytr); Xte, yte = torch.as_tensor(Xte), torch.as_tensor(yte)
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

```python
COLOURS = ["red", "green", "blue"]; SHAPES3 = ["square", "circle", "cross"]
_RGB = {"red": (1.0, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.25, 1.0)}

def _mask(shape, size, cy, cx, r):
    yy, xx = np.mgrid[0:size, 0:size]
    if shape == "square": return (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    if shape == "circle": return (yy - cy) ** 2 + (xx - cx) ** 2 <= (r + 0.5) ** 2
    return ((np.abs(yy - cy) <= 1) & (np.abs(xx - cx) <= r)) | ((np.abs(xx - cx) <= 1) & (np.abs(yy - cy) <= r))   # cross

def make_colored_shapes(n=300, size=16, noise=0.05, seed=0, jitter=2):
    """n colour pictures (n, 3, size, size) of a red/green/blue square/circle/cross near the centre on a white background.
    Returns X float32, colour index (0..2, see COLOURS), shape index (0..2, see SHAPES3) and captions like 'red circle'."""
    r = np.random.RandomState(seed)
    col = np.arange(n) % 3; r.shuffle(col)
    shp = (np.arange(n) // 3) % 3; r.shuffle(shp)
    X = np.ones((n, 3, size, size), np.float32)
    for i in range(n):
        cy, cx = size // 2 + r.randint(-jitter, jitter + 1, 2)
        m = _mask(SHAPES3[shp[i] ], size, cy, cx, 4)
        for ch in range(3): X[i, ch][m] = _RGB[COLOURS[col[i] ]][ch]
    X = np.clip(X + r.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return X, col.astype(np.int64), shp.astype(np.int64), [f"{COLOURS[c]} {SHAPES3[s]}" for c, s in zip(col, shp)]

def make_left_right(n=400, size=16, noise=0.1, seed=0):
    """n greyscale pictures (n, 1, size, size). Each has ONE plus sign (the target) and ONE filled square (a distractor that tells you nothing) at random places.
    Label 1 if the plus sign is in the LEFT half of the picture, 0 if it is in the right half. Returns X float32 and y int64.
    The answer depends only on WHERE the plus is, not on what the patches look like: a model that ignores positions can only guess."""
    r = np.random.RandomState(seed)
    X = np.zeros((n, 1, size, size), np.float32); y = np.zeros(n, np.int64)
    for i in range(n):
        left = r.rand() < 0.5
        px = r.randint(2, size // 2 - 2) if left else r.randint(size // 2 + 2, size - 2)
        py = r.randint(2, size - 2)
        while True:
            qy, qx = r.randint(2, size - 2, 2)
            if abs(py - qy) > 3 or abs(px - qx) > 3: break
        X[i, 0][_mask("cross", size, py, px, 2)] = 1.0   # a plus with arms 2 pixels long (a 5 x 5 plus), clearly different from the 3 x 3 square
        X[i, 0][_mask("square", size, qy, qx, 1)] = 1.0
        y[i] = 1 if left else 0
    X = np.clip(X + r.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    return X, y
```

```python
def show_images(X, titles=None, n=10, cols=5):
    """Show the first n pictures of X: shape (N, H, W), (N, 1, H, W) or (N, 3, H, W)."""
    X = np.asarray(X); n = min(n, len(X)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.8 * rows))
    for a in np.ravel(axes): a.axis("off")
    for i, a in enumerate(np.ravel(axes)[:n]):
        im = X[i]
        a.imshow(np.transpose(im, (1, 2, 0)) if im.ndim == 3 and im.shape[0] == 3 else im.reshape(im.shape[-2:]), cmap="gray_r", vmin=0, vmax=1)
        if titles is not None: a.set_title(str(titles[i]), fontsize=8)
    plt.tight_layout(); plt.show()

def draw_shape_pil(colour="red", shape="circle", size=224, bg="white"):
    """A clean size x size PIL picture of one coloured shape (colour: any colour name; shape: circle, square, triangle)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), bg); d = ImageDraw.Draw(img); m, M = size // 4, 3 * size // 4
    if shape == "circle": d.ellipse([m, m, M, M], fill=colour)
    elif shape == "square": d.rectangle([m, m, M, M], fill=colour)
    else: d.polygon([(size // 2, m), (m, M), (M, M)], fill=colour)
    return img

def load_clip(name="openai/clip-vit-base-patch32"):
    """Load the real CLIP model and its processor (about 600 MB, downloaded once; offline cache on the Spark)."""
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)
```

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `make_shapes(n, size, noise, shift, seed)` returns `n` greyscale pictures of five line shapes (`SHAPE_NAMES`) as an array of shape (n, 1, 16, 16), and their labels;
   - `make_colored_shapes(n, size, noise, seed, jitter)` returns colour pictures of shape (n, 3, 16, 16), a colour index (`COLOURS`), a shape index (`SHAPES3`) and a caption such as "red circle" for each;
   - `make_left_right(n, size, noise, seed)` returns greyscale pictures with a plus sign and a distractor, and the label "plus sign in the left half?";
   - `show_images(X, titles, n, cols)` draws the first `n` pictures (grey or colour);
   - `count_params(model)` counts the trainable numbers of a model; `train_classifier(model, Xtr, ytr, Xte, yte, epochs, lr, opt, batch, seed, verbose)` trains any PyTorch classifier that maps a batch of pictures (or feature vectors) to class scores and returns its learning curves (`loss`, `train_acc`, `test_acc`);
   - `draw_shape_pil(colour, shape, size, bg)` draws a clean coloured circle, square or triangle as a PIL picture, and `load_clip(name)` loads the real CLIP model and its processor (used only in Part 6);
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

7. Run this smoke test in the next cell. It checks that the helpers work (without loading any model):

```python
Xs, ys = make_shapes(10, seed=SEED)
Xc0, col0, shp0, caps0 = make_colored_shapes(6, seed=SEED)
Xl0, yl0 = make_left_right(6, seed=SEED)
print("make_shapes:", Xs.shape, ys[:10], "  classes:", SHAPE_NAMES)
print("make_colored_shapes:", Xc0.shape, caps0)
print("make_left_right:", Xl0.shape, "labels", yl0)
tiny = nn.Sequential(nn.Flatten(), nn.Linear(256, 5))
hist = train_classifier(tiny, Xs, ys, Xs, ys, epochs=5, lr=0.01, seed=SEED)
print("parameters of the tiny classifier:", count_params(tiny), " loss in epoch 1 and 5:", round(hist["loss"][0], 3), round(hist["loss"][-1], 3))
show_images(Xc0, caps0, n=6, cols=6)
check("helpers work", Xc0.shape == (6, 3, 16, 16) and count_params(tiny) == 1285 and hist["loss"][-1] < hist["loss"][0], "Run the setup cells first.")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
make_shapes: (10, 1, 16, 16) [3 1 0 0 2 2 4 4 3 1]   classes: ['horizontal bar', 'vertical bar', 'cross', 'square', 'diagonal']
make_colored_shapes: (6, 3, 16, 16) ['red circle', 'green square', 'blue square', 'blue square', 'green circle', 'red circle']
make_left_right: (6, 1, 16, 16) labels [1 1 0 1 1 0]
parameters of the tiny classifier: 1285  loss in epoch 1 and 5: 1.555 0.59
✓ helpers work
```

   You should see `Running on: cpu` after the first cell. The smoke test then prints the shape of each data set, a few labels and captions, a figure with six small coloured pictures, the size of a very small classifier and its loss going down, and a tick.

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cells and the earlier blocks again, in order.
- `RuntimeError: Could not load CLIP`: the offline cache or the internet is not available. On the Spark ask your instructor; in Colab check the connection and run the cell again. Parts 1 to 5 and the toy CLIP of Part 6 do not need the model and can be done meanwhile.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the block ends with `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42`. The numbers of Parts 1 to 3 (counts, shapes, parameters, accuracies on these clean pictures, similarities) are exact and must match. Training curves of the later parts can differ a little between library versions and hardware; the qualitative results are stable.
- A cell seems frozen: stop it (the square button), restart the kernel and run again from the top. If a cell takes more than two minutes, you probably changed a loop count or a data size.

### Part 1. What is multimodal? Modalities, fusion and complementarity (25 min)

*Lesson section: "Multimodal".* A **modality** is a kind of data: pictures, sound, text, video. A *multimodal* model takes in, or gives out, more than one of them. The reason is simple: each modality alone is often ambiguous, and two together can remove the doubt. In this part you do not touch a big model yet. You build the smallest honest example of the idea with pictures you draw yourself: a red or green or blue square, circle or cross. The colour is one piece of information and the shape is another, and a caption such as "red circle" needs both. You will measure how well "colour only", "shape only" and "both together" do, compare two ways of putting them together (early and late fusion), and then think about three real applications and how they can go wrong.

**Imagine first.** The data set of this part has nine kinds of picture (three colours times three shapes) and a classifier must name the nine-way caption, for example "green cross". Suppose one model sees only the colour of the picture, a second model sees only the shape, and a third model sees both. Before you run any code, write three predictions: what test accuracy (a number between 0 and 1) will each of the three reach? Remember that guessing among nine classes gives about 0.11. You will compare your answers with Step 1.4.

```python
# Imagine first: the nine-way caption task (colour x shape)
my_colour_only = None   # your guess for the test accuracy of a model that sees only the colour (a number between 0 and 1)
my_shape_only = None    # ... of a model that sees only the shape
my_both = None          # ... of a model that sees both
```

1. **Step 1.1. Modalities are tensors.** Every modality reaches a computer as numbers, and what changes from one to the other is the *shape* of the array and the *amount* of numbers. A colour picture is a 3-D array (channels, height, width); a sound is a long 1-D array of samples; a sentence is a short list of integer token ids that a lookup table turns into vectors. The block below builds one dummy example of each (filled with zeros: only the shapes matter) and prints a small table of shape and size. The fourth row is a real 16 by 16 greyscale picture from `make_shapes`, the data of Week 4.

```python
Xg, yg = make_shapes(4, size=16, seed=SEED)
modalities = [
    ("RGB picture, 224 x 224",          np.zeros((3, 224, 224)), "channels x height x width"),
    ("audio, 1 second at 16 kHz",       np.zeros(16000),         "one number per sample"),
    ("sentence of 12 tokens",           np.arange(12),           "12 integer token ids"),
    ("greyscale picture, 16 x 16",      Xg[0, 0],                "from make_shapes (Week 4)"),
]
print(f"{'modality':30s} {'shape':16s} {'numbers':>9s}   note")
for name, arr, note in modalities:
    print(f"{name:30s} {str(arr.shape):16s} {arr.size:9,d}   {note}")
assert modalities[0][1].size == 150528 and modalities[1][1].size == 16000
assert modalities[2][1].size == 12 and modalities[3][1].size == 256
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
modality                       shape              numbers   note
RGB picture, 224 x 224         (3, 224, 224)      150,528   channels x height x width
audio, 1 second at 16 kHz      (16000,)            16,000   one number per sample
sentence of 12 tokens          (12,)                   12   12 integer token ids
greyscale picture, 16 x 16     (16, 16)               256   from make_shapes (Week 4)
```

   You should see a table with four rows. The 224 by 224 RGB picture has shape (3, 224, 224) and **150,528** numbers; one second of audio at 16 kHz has shape (16000,) and **16,000** numbers; the sentence is only **12** integer ids; and the 16 by 16 greyscale picture of Week 4 has **256** numbers. The silent assertions at the end passed. Notice the range: a single photograph is about ten times larger than a second of sound and thousands of times larger than a sentence, and yet a sentence can carry more meaning. The three modalities have different shapes and very different sizes, and a multimodal model has to bring them to one common format (vectors of the same length, a sequence of tokens).

   Tips and common mistakes:

   - `arr.size` is the number of entries and `arr.shape` the tuple of sizes; `np.zeros` is only a stand-in, because in this table only the shapes and counts matter.
   - A *token id* is not yet a meaningful number: it is an index into an embedding table (Week 7). The 12 ids become a 12 by d array only after that lookup.
   - Numbers are not information: a 150,528-number picture of a blank wall carries far less than a 12-token sentence that tells you something new. Later parts show how models shorten the long modalities into tokens.

   *Exercise 1.1:* compute how many numbers are in (a) 30 seconds of audio at 16 kHz and (b) one 640 by 480 RGB photograph, and store them in `n_audio30` and `n_photo`. Why it matters: the size of the raw input is the first thing a designer of a multimodal system must know: it decides what must be shortened before a Transformer can read it.

```python
n_audio30 = None   # number of numbers in 30 seconds of audio at 16 kHz
n_photo = None     # number of numbers in a 640 x 480 RGB photograph
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if n_audio30 is None or n_photo is None:
    print("Fill in n_audio30 and n_photo above, then run this cell again.")
else:
    check("n_audio30 is right", n_audio30 == 30 * 16000, "Samples per second times seconds.")
    check("n_photo is right", n_photo == 3 * 480 * 640, "Three channels times height times width.")
    RESULTS["p1_counts"] = (int(n_audio30), int(n_photo))
```

   `n_audio30` is 480,000 and `n_photo` is 921,600. Thirty seconds of audio is already 480,000 numbers, and a single phone photograph is almost a million: this is why audio and image models do not read the raw samples one by one but first shorten them into patches or frames.

2. **Step 1.2. Look at the data.** From now on we use `make_colored_shapes`. It draws small 16 by 16 colour pictures with NumPy (nothing is downloaded): a red, green or blue square, circle or cross near the centre of a white background, plus a little noise, and it returns a caption such as "red circle" for each. Always look at your data before you model it: the block shows ten pictures with their captions and counts how many pictures each of the nine possible captions has.

```python
Xc, col, shp, caps = make_colored_shapes(300, seed=SEED)
print("pictures:", Xc.shape, Xc.dtype, " pixel range", round(float(Xc.min()), 2), "to", round(float(Xc.max()), 2))
show_images(Xc, caps, n=10, cols=5)
counts = Counter(caps)
for name, k in sorted(counts.items()):
    print(f"{name:14s} {k}")
print("different captions:", len(counts), " total:", sum(counts.values()))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
pictures: (300, 3, 16, 16) float32  pixel range 0.0 to 1.0
blue circle    30
blue cross     33
blue square    37
green circle   35
green cross    32
green square   33
red circle     34
red cross      34
red square     32
different captions: 9  total: 300
```

   You should see a figure with ten small pictures in two rows of five (a red, green or blue square, circle or cross, each with its caption above), and text like this: the array has shape (300, 3, 16, 16), type float32, and pixel values between 0.0 and 1.0. The nine captions occur between 30 and 37 times each: `blue circle` 30, `blue cross` 33, `blue square` 37, `green circle` 35, `green cross` 32, `green square` 33, `red circle` 34, `red cross` 34 and `red square` 32; 9 different captions and 300 pictures in total. So the classes are well balanced: the most frequent caption, `blue square`, has 37 pictures (12.3 percent) and a model that always answers with it would be right about one time in nine, not better. Every picture is only 16 by 16 pixels, so a square or a circle is a coarse blob; that is enough for this lab.

   *Exercise 1.2:* find the caption that occurs most often in `caps` and how many times, and store them in `top_caption` (a string) and `top_count` (an integer). Why it matters: before you train anything you should know whether the classes are balanced: a model that always answers with the most frequent class sets the baseline that every real model must beat.

```python
top_caption = None   # the most frequent caption (a string such as "red circle")
top_count = None     # how many pictures have it
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if top_caption is None or top_count is None:
    print("Fill in top_caption and top_count above, then run this cell again.")
else:
    t_name, t_count = Counter(caps).most_common(1)[0]
    check("top_caption is right", top_caption == t_name, "Counter(caps).most_common(1) gives a list with one (caption, count) pair.")
    check("top_count is right", top_count == t_count, "The second entry of that pair.")
    check("it beats a uniform share", top_count >= 300 / 9, "The largest class has at least 300 / 9 pictures.")
    RESULTS["p1_top_caption"] = (top_caption, int(top_count))
```

   `top_caption` is `blue square` and `top_count` is 37. The nine classes have between 30 and 37 pictures each: balanced.

3. **Step 1.3. Two hand-made features: colour and shape.** We now describe each picture by a few numbers, the way people did before deep learning. The **colour feature** is the mean red, green and blue value of the pixels that are *not* near-white (the ink pixels). The **shape feature** is made of the number of ink pixels, the width-to-height ratio of the bounding box of the ink, and the *fill* (ink pixels divided by the area of the bounding box). The first one looks only at colours and the others look only at the geometry: two "modalities" of the same picture. The functions below compute them and the block prints three examples.

```python
def ink_mask(X, thr=0.6):
    """True where a pixel is not near-white: its smallest channel is below thr. Shape (n, H, W)."""
    return X.min(axis=1) < thr

def colour_features(X):
    """Mean R, G, B of the ink pixels of each picture: shape (n, 3)."""
    m = ink_mask(X)
    return np.array([ [X[i, ch][m[i] ].mean() for ch in range(3)] for i in range(len(X))], np.float32)

def shape_features(X):
    """Per picture: number of ink pixels, width / height of the bounding box, fill = ink / box area: shape (n, 3)."""
    out = []
    for mi in ink_mask(X):
        ys, xs = np.nonzero(mi)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        out.append([mi.sum(), w / h, mi.sum() / (w * h)])
    return np.array(out, np.float32)

colour_f, shape_f = colour_features(Xc), shape_features(Xc)
for i in range(3):
    print(f"{caps[i]:13s} colour (R, G, B) = {np.round(colour_f[i], 2)}   shape (pixels, ratio, fill) = {np.round(shape_f[i], 2)}")
print("all width/height ratios:", np.unique(shape_f[:, 1]))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
blue circle   colour (R, G, B) = [0.15 0.25 0.98]   shape (pixels, ratio, fill) = [69.    1.    0.85]
blue square   colour (R, G, B) = [0.15 0.25 0.98]   shape (pixels, ratio, fill) = [81.  1.  1.]
blue square   colour (R, G, B) = [0.15 0.26 0.98]   shape (pixels, ratio, fill) = [81.  1.  1.]
all width/height ratios: [1.]
```

   You should see three lines, one for each of the first three pictures. The first is a `blue circle`: colour about (0.15, 0.25, 0.98), that is, no red, a little green and a lot of blue; shape (69 pixels, ratio 1.0, fill 0.85). The next two are `blue square` pictures with the same colour and shape (81 pixels, ratio 1.0, fill 1.0). The last line prints all the width-to-height ratios of the 300 pictures: there is only **one** value, `[1.]`. All three shapes fit a 9 by 9 box, so the ratio tells nothing, and a feature that never changes can teach nothing: that is why we also computed the pixel count and the fill. The colour features separate red, green and blue clearly (the tallest value is the colour of the ink); the shape features separate the three shapes (about 81, 69 and 45 ink pixels for a square, a circle and a cross), but neither says anything about the other task.

   Tips and common mistakes:

   - The threshold 0.6 works because the background is near 1.0 in every channel and each shape has at least one channel below 0.3. The noise (standard deviation 0.05) never crosses it.
   - A feature that is constant (the ratio here) carries no information. Real feature engineering means finding out which numbers do vary with the thing you want to know.

   *Exercise 1.3:* compute the mean number of ink pixels (first column of `shape_f`) for each of the three shapes and store them in the dictionary `mean_pixels`, with the keys `"square"`, `"circle"` and `"cross"`, rounded to one decimal. Which shape has the most ink and which the least? Why it matters: this is how a single geometric number separates a task: here the pixel count alone will tell the three shapes apart, but it says nothing about the colour.

```python
mean_pixels = {"square": None, "circle": None, "cross": None}   # mean number of ink pixels per shape, rounded to 1 decimal
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if any(v is None for v in mean_pixels.values()):
    print("Fill in mean_pixels above, then run this cell again.")
else:
    truth = {s: round(float(shape_f[shp == k, 0].mean()), 1) for k, s in enumerate(SHAPES3)}
    check("the three means are right", all(abs(mean_pixels[s] - truth[s]) < 0.11 for s in SHAPES3), "Select the rows with shp equal to the shape index, then average column 0.")
    check("the order is right (square > circle > cross)", mean_pixels["square"] > mean_pixels["circle"] > mean_pixels["cross"], "A cross is two thin bars, a square is filled.")
    RESULTS["p1_mean_pixels"] = dict(mean_pixels)
```

   `mean_pixels` is `{'square': 81.0, 'circle': 69.0, 'cross': 45.0}`. The square has the most ink (a filled 9 by 9 block), the cross the least. The pixel count alone is enough to separate the three shapes, but gives no information about colour.

4. **Step 1.4. Two ambiguous tasks, one fused answer (early fusion).** The goal is the nine-way caption (colour times shape), with the label `y9 = colour * 3 + shape`. We train three small classifiers, each a single `nn.Linear` layer (logistic regression) trained with the `train_classifier` helper: (a) on the colour features only, (b) on the shape features only, (c) on both feature sets glued together side by side. Gluing the inputs *before* the model is called **early fusion**. We generate 900 new pictures, use 600 for training and 300 for testing, and standardise every feature (subtract the training mean, divide by the training standard deviation) so that the pixel counts, which are large, do not drown the small numbers.

```python
Xa, ca, sa, _ = make_colored_shapes(900, seed=SEED + 1)
Fc_all, Fs_all, y9 = colour_features(Xa), shape_features(Xa), ca * 3 + sa
tr, te = np.arange(900) < 600, np.arange(900) >= 600

def prep(Fm):
    """Standardise with the training mean and standard deviation (a constant column becomes zeros)."""
    mu, sd = Fm[tr].mean(0), Fm[tr].std(0)
    return ((Fm - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)

feats = {"colour only": prep(Fc_all), "shape only": prep(Fs_all)}
feats["both (early fusion)"] = np.concatenate([feats["colour only"], feats["shape only"] ], axis=1)

acc9 = {}
for name, Fm in feats.items():
    torch.manual_seed(SEED)
    h = train_classifier(nn.Linear(Fm.shape[1], 9), Fm[tr], y9[tr], Fm[te], y9[te], epochs=60, lr=0.05, seed=SEED)
    acc9[name] = h["test_acc"][-1]
    print(f"{name:22s} features: {Fm.shape[1]}   train {h['train_acc'][-1]:.3f}   test {acc9[name]:.3f}")
print("chance for nine classes:", round(1 / 9, 3), "  my guesses:", my_colour_only, my_shape_only, my_both)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
colour only            features: 3   train 0.363   test 0.343
shape only             features: 3   train 0.358   test 0.343
both (early fusion)    features: 6   train 1.000   test 1.000
chance for nine classes: 0.111   my guesses: None None None
```

   You should see three lines (the exact decimals are stable here because the pictures are drawn with a fixed seed):

   - colour only (3 features): train 0.363, test **0.343**;
   - shape only (3 features): train 0.358, test **0.343**;
   - both, early fusion (6 features): train 1.000, test **1.000**;

   and the line "chance for nine classes: 0.111" followed by your own guesses. Each single modality reaches only about one third: the colour tells you which of the three colours it is, but then three shapes are still possible, so the best you can do is to pick one of them and be right one time in three. The two modalities together are **perfect** on these clean pictures. This is an easy toy problem, so do not expect 1.000 on real data; the point is the pattern: 0.34 and 0.34 become 1.00. Compare with your guesses of the Imagine-first cell: here the two single modalities are equally ambiguous, and neither can be better than one third.

   **Why this happens.** The colour features know *which colour* but nothing about the shape, so among the nine captions they can only narrow the answer down to three candidates; the best such classifier is right one time in three. The shape features are the mirror image. Each modality alone is *ambiguous* for this task, and the two are *complementary*: what one lacks, the other has. Together they identify the caption. This is the same reason why a photo and its caption, or a voice and a face, can be worth more together than the sum of their parts.

   *Exercise 1.4:* compute the share of the nine-way test accuracy that the fused model gains over the better single-modality model: `gain = acc9["both (early fusion)"] - max(acc9["colour only"], acc9["shape only"])`, rounded to 3 decimals, and store whether the fused model is more than twice as accurate as the better single one in `more_than_double` (True or False). Why it matters: you will do this comparison for every multimodal system you ever evaluate: the question is not "is the model accurate" but "does the second modality add something the first did not have".

```python
gain = None              # accuracy of the fused model minus the accuracy of the better single-modality model (3 decimals)
more_than_double = None  # True or False
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if gain is None or more_than_double is None:
    print("Fill in gain and more_than_double above, then run this cell again.")
else:
    best_single = max(acc9["colour only"], acc9["shape only"]); fused = acc9["both (early fusion)"]
    check("gain is right", abs(gain - round(fused - best_single, 3)) < 0.0015, "fused accuracy minus the larger of the two single accuracies.")
    check("more_than_double is right", bool(more_than_double) == (fused > 2 * best_single), "Compare fused with 2 times the better single accuracy.")
    RESULTS["p1_gain"] = (float(gain), bool(more_than_double))
```

   `gain` is 0.657 (1.000 minus 0.343) and `more_than_double` is `True`. The fused model is about three times as accurate as either single modality, the clearest sign of complementarity.

5. **Step 1.5. Late fusion: combine the answers, not the inputs.** There is another way to combine two modalities: build one expert for each and merge their *answers*. This is **late fusion**. Here we train a colour classifier (three classes, colour features only) and a shape classifier (three classes, shape features only), turn each output into probabilities with a softmax and multiply them: the probability of the caption "colour c and shape s" is `p_colour[c] * p_shape[s]`, an "AND" of two independent opinions. The answer is the caption with the largest product. The block also shows what happens when one modality is missing (we replace the shape expert's opinion by a uniform one).

```python
def fit_probs(Fm, y, k):
    """Train a linear classifier on the training rows of Fm; return the class probabilities for ALL rows."""
    torch.manual_seed(SEED)
    model = nn.Linear(Fm.shape[1], k)
    train_classifier(model, Fm[tr], y[tr], Fm[te], y[te], epochs=60, lr=0.05, seed=SEED)
    with torch.no_grad():
        return F.softmax(model(torch.as_tensor(Fm)), dim=1).numpy()

p_col = fit_probs(feats["colour only"], ca, 3)         # (900, 3)
p_shp = fit_probs(feats["shape only"], sa, 3)          # (900, 3)
joint = p_col[:, :, None] * p_shp[:, None, :]          # (900, 3, 3): p(colour c) * p(shape s)
pred_late = joint.reshape(900, 9).argmax(1)            # index colour * 3 + shape
late_acc = float((pred_late[te] == y9[te]).mean())
shape_missing = (p_col[:, :, None] * np.full((900, 1, 3), 1 / 3)).reshape(900, 9).argmax(1)
print(f"colour expert alone (3 classes): {(p_col[te].argmax(1) == ca[te]).mean():.3f}")
print(f"shape expert alone  (3 classes): {(p_shp[te].argmax(1) == sa[te]).mean():.3f}")
print(f"late fusion, nine-way:           {late_acc:.3f}   (early fusion: {acc9['both (early fusion)']:.3f})")
print(f"late fusion with the shape expert switched off: {(shape_missing[te] == y9[te]).mean():.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
colour expert alone (3 classes): 1.000
shape expert alone  (3 classes): 1.000
late fusion, nine-way:           1.000   (early fusion: 1.000)
late fusion with the shape expert switched off: 0.333
```

   You should see four lines: the colour expert alone is right on **1.000** of the test pictures when it only has to name the colour (three classes); the shape expert alone, **1.000** for the shape (three classes); late fusion, the product of the two, gets the nine-way caption right on **1.000** of the test pictures, the same as early fusion; and when the shape expert is switched off (replaced by a uniform opinion) the nine-way accuracy falls to **0.333**: without the shape the model can only get the colour right and has to pick one of the three shapes. So on this clean data early and late fusion are equally good, and late fusion has the extra property that it survives a missing modality (it falls back to one third instead of failing).

   **Early against late fusion.** Early fusion lets one model see all the inputs at once, so it can learn *interactions* between modalities (a red circle might look different from a red square in a way that no single feature shows), but it needs data in which all modalities are present together and it breaks if one is missing. Late fusion keeps the experts separate: you can train each on its own data, swap one out, and still answer when one modality is missing, but the experts cannot learn from the interaction. Real systems sit between the two: CLIP (Part 6) fuses at the very end, in a shared space, and a vision-language model fuses early, by turning picture patches into tokens next to the words.

   *Exercise 1.5:* in the block above we switched the *shape* expert off. Do the same for the *colour* expert: replace `p_col` by a uniform opinion (`1/3` for every colour), compute the nine-way test accuracy and store it in `colour_missing_acc` (3 decimals). Why it matters: a robust system should degrade gracefully when one sensor fails; knowing how much it degrades is part of the evaluation.

```python
colour_missing_acc = None   # nine-way test accuracy of late fusion when the colour expert says "I do not know" (3 decimals)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if colour_missing_acc is None:
    print("Fill in colour_missing_acc above, then run this cell again.")
else:
    _pred = (np.full((900, 3, 1), 1 / 3) * p_shp[:, None, :]).reshape(900, 9).argmax(1)
    truth = round(float((_pred[te] == y9[te]).mean()), 3)
    check("colour_missing_acc is right", abs(colour_missing_acc - truth) < 0.0015, "Multiply a uniform colour opinion with p_shp, then take the argmax over the nine captions.")
    check("it is clearly worse than late fusion with both", colour_missing_acc < late_acc - 0.3, "Without colour you can only get the shape right.")
    RESULTS["p1_colour_missing"] = float(colour_missing_acc)
```

   `colour_missing_acc` is 0.357. With no colour information the late-fusion model gets the shape right but has to pick one of three colours, so it is right about one time in three (a little more than 0.333 because the pictures are not exactly balanced).

6. **Step 1.6. Three applications: inputs, outputs, failures, oversight.** This last step is a writing exercise. A multimodal system is useful only if you know what goes in, what comes out, how it is likely to fail, and which human check keeps those failures from doing harm. The dictionary below has one entry per application. For **image captioning**, **visual question answering** (VQA: a picture and a question go in, an answer comes out) and **speech recognition** (sound in, text out) fill four fields: `inputs` (the modalities that go in), `output` (the modality that comes out), `failure` (a likely way to go wrong, in a short sentence) and `oversight` (one human-oversight step). A tip: think about a blind user who relies on the caption, a medical question about an X-ray, and a classroom where many students have regional accents.

```python
def show_apps(apps):
    """Print the application dictionary as a table."""
    print(f"{'application':26s} {'inputs':24s} {'output':8s} failure | oversight")
    for name, d in apps.items():
        print(f"{name:26s} {str(d['inputs']):24s} {str(d['output']):8s} {d['failure']} | {d['oversight']}")

demo = {"machine translation": {"inputs": "text (French)", "output": "text",
                                "failure": "It translates an idiom word by word.",
                                "oversight": "A bilingual person checks official letters."}}
show_apps(demo)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
application                inputs                   output   failure | oversight
machine translation        text (French)            text     It translates an idiom word by word. | A bilingual person checks official letters.
```

   You should see one row: the demonstration entry "machine translation" with inputs "text (French)", output "text", and a failure and oversight sentence. It shows you the level of detail we expect: a short phrase for the modalities and a full sentence for the failure and the human check. In the exercise below you will fill in the three applications of the lesson and print the table of your own.

   *Exercise 1.6:* replace every `None` in the dictionary `apps` by a short text of your own (for `inputs` and `output` a few words, for `failure` and `oversight` one sentence), then print the table again. Why it matters: this is the habit you will need in Part 7: for every system, name the inputs and outputs, then name how it fails and who checks it.

```python
apps = {
    "image captioning":   {"inputs": None, "output": None, "failure": None, "oversight": None},
    "visual question answering": {"inputs": None, "output": None, "failure": None, "oversight": None},
    "speech recognition": {"inputs": None, "output": None, "failure": None, "oversight": None},
}
show_apps(apps)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
filled = all(isinstance(v, str) and len(v.strip()) >= 4 for d in apps.values() for v in d.values())
if not filled:
    print("Fill in every field of apps above with a text, then run this cell again.")
else:
    check("every failure and oversight entry is a real sentence", all(len(d["failure"]) >= 15 and len(d["oversight"]) >= 15 for d in apps.values()), "Write at least a short sentence for failure and oversight.")
    check("no two failures are identical", len({d["failure"] for d in apps.values()}) == 3, "Each application has its own typical failure.")
    RESULTS["p1_apps"] = apps
```

   Your sentences will differ; what matters is that each one is specific. For example: image captioning takes a picture and gives text, it may name objects that are not there, and a person reads the captions of important pictures before they are published; visual question answering takes a picture and a question and gives text, it may answer from the words alone without really looking, and medical or legal answers are checked by a qualified person; speech recognition takes audio and gives text, it may make more errors for some accents and noisy rooms, and transcripts are proofread while the error rate is measured per accent. A good failure names a concrete wrong behaviour (a hallucinated object, an answer from the text alone, an accent that is understood worse) and a good oversight step names *who* checks *what* and *when*.

### Part 2. Patchify: from pixels to tokens (20 min)

*Lesson section: "Vision Transformer".* A Transformer reads a *sequence* of tokens, so to let it look at a picture we must first turn the picture into a sequence. The idea of the Vision Transformer (ViT) is to cut the picture into a grid of small squares, called **patches**, flatten every square into a vector, and treat the vectors as the tokens (the "words") of the picture. In this part you do the cutting yourself, first by hand with NumPy, then with the two standard PyTorch tools (`unfold` and a strided `Conv2d`). Then you count: how many tokens, how many numbers per token, how many parameters in the first layer, and why this cutting is what makes a Transformer affordable on pictures at all. These are the ideas of the lesson's figure on the patch grid.

**Imagine first.** A colour picture of 224 by 224 pixels is cut into square patches. Before you run any code, write two predictions: how many patches (tokens) do you get with patches of 16 by 16 pixels, and how many with patches of 8 by 8 pixels? Halving the patch side is a big change: think about what happens to the number of patches. You will compare your answers with Step 2.3.

```python
# Imagine first: patches of a 224 x 224 picture
my_tokens_p16 = None   # number of patches with 16 x 16 patches (an integer)
my_tokens_p8 = None    # number of patches with 8 x 8 patches (an integer)
```

1. **Step 2.1. Patchify by hand with NumPy.** A picture is an array of shape (C, H, W): C channels, H rows, W columns. To cut it into square patches of side `p` we only have to *reshape* the array: split the rows into blocks of `p` and the columns into blocks of `p`, move the "block" axes to the front, and flatten each block. The function `patchify` below returns an array of shape (N, C*p*p), one row per patch, read from left to right and from top to bottom. `unpatchify` does the reverse. First look at a toy: a 4 by 4 picture with the numbers 0 to 15, cut into four patches of 2 by 2.

```python
def patchify(img, p):
    """(C, H, W) array -> (N, C*p*p): N = (H/p)*(W/p) patches in reading order; each patch is flattened as (channel, row, column)."""
    C, H, W = img.shape
    g = img.reshape(C, H // p, p, W // p, p)         # split the rows and the columns into (block, position inside the block)
    g = g.transpose(1, 3, 0, 2, 4)                    # (block row, block column, C, p, p)
    return g.reshape((H // p) * (W // p), C * p * p)

def unpatchify(patches, p, C, H, W):
    """The inverse of patchify: (N, C*p*p) -> (C, H, W)."""
    g = patches.reshape(H // p, W // p, C, p, p).transpose(2, 0, 3, 1, 4)
    return g.reshape(C, H, W)

toy = np.arange(16).reshape(1, 4, 4)
print("the toy picture:"); print(toy[0])
print("its four 2 x 2 patches, one per row:"); print(patchify(toy, 2))

pic = make_shapes(1, seed=SEED)[0][0]                 # a real 16 x 16 picture, shape (1, 16, 16)
for p in (2, 4, 8):
    P = patchify(pic, p)
    assert np.array_equal(unpatchify(P, p, 1, 16, 16), pic)      # the round trip is exact
    print(f"p = {p}: {P.shape[0]:2d} patches of {P.shape[1]:3d} numbers; round trip exact")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
the toy picture:
[ [ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15] ]
its four 2 x 2 patches, one per row:
[ [ 0  1  4  5]
 [ 2  3  6  7]
 [ 8  9 12 13]
 [10 11 14 15] ]
p = 2: 64 patches of   4 numbers; round trip exact
p = 4: 16 patches of  16 numbers; round trip exact
p = 8:  4 patches of  64 numbers; round trip exact
```

   You should see the toy picture (the numbers 0 to 15 in a 4 by 4 grid) and below it four rows, one per patch: `[0 1 4 5]` is the top-left patch (pixels 0, 1 of the first row and 4, 5 of the second row), `[2 3 6 7]` the top-right, `[8 9 12 13]` the bottom-left and `[10 11 14 15]` the bottom-right. Then three lines for the real picture: with p = 2 you get **64** patches of 4 numbers, with p = 4 **16** patches of 16 numbers, with p = 8 only **4** patches of 64 numbers, and the round trip through `unpatchify` is exact every time (the `assert` would stop the cell otherwise). The number of numbers never changes (64 times 4 = 16 times 16 = 4 times 64 = 256); patchifying only rearranges them into tokens.

   Tips and common mistakes:

   - `reshape` never changes the order of the numbers in memory, it only changes how we look at them. The `transpose` is what moves the numbers into patch order.
   - If H or W is not a multiple of `p`, the reshape raises an error. Real ViTs resize or crop the picture first (224 is divisible by 16, 14, 8 and 32).
   - The order inside a patch (channel, row, column) matches PyTorch's `unfold` and the weights of `Conv2d`, which is what Step 2.2 relies on.

   *Exercise 2.1:* in a 16 by 16 picture cut into patches of 4 by 4, the pixel in row 5 and column 9 (counting from 0) lies in one patch. Store in `patch_id` the index of that patch (counting patches in reading order from 0) and in `inside` the index of the pixel inside the flattened patch (row by row, from 0 to 15). Do it with arithmetic first, then confirm it with `patchify`. Why it matters: an image model must be able to go back and forth between "pixel coordinates" and "token number"; later, attention maps are drawn by exactly this conversion.

```python
patch_id = None   # index of the patch that contains the pixel (row 5, column 9)
inside = None     # index of that pixel inside the flattened 4 x 4 patch
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if patch_id is None or inside is None:
    print("Fill in patch_id and inside above, then run this cell again.")
else:
    ramp = np.arange(256).reshape(1, 16, 16)               # every pixel holds its own index 16 * row + column
    rows = patchify(ramp, 4)
    r_, c_ = np.nonzero(rows == 16 * 5 + 9)
    check("patch_id is right", patch_id == int(r_[0]), "Patch row is row // 4, patch column is column // 4; patch_id = patch row * 4 + patch column.")
    check("inside is right", inside == int(c_[0]), "Row % 4 and column % 4 give the place inside the patch: inside = (row % 4) * 4 + (column % 4).")
    RESULTS["p2_pixel"] = (int(patch_id), int(inside))
```

   `patch_id` is 6 and `inside` is 5. The pixel is in patch row 1 and patch column 2 (the 4 by 4 patches), so patch 1 * 4 + 2 = 6; inside the patch it is at row 1, column 1, so 1 * 4 + 1 = 5.

2. **Step 2.2. With PyTorch: unfold and a strided Conv2d.** Real ViT code does not write loops or reshapes; it uses two PyTorch tools. `F.unfold(x, kernel_size=p, stride=p)` cuts a batch of pictures into patches, exactly like `patchify`. And `nn.Conv2d(C, d, kernel_size=p, stride=p)` slides a window of side `p` with step `p`, so the windows do not overlap: it multiplies every patch by the same matrix and adds a bias, which is the same as "patchify, then a linear layer with d outputs". This is the **standard trick for the patch embedding** of a ViT. The block proves it: it copies the convolution's weights into a plain `nn.Linear` and checks that both give the same numbers.

```python
x = torch.rand(2, 3, 32, 32); p = 8; d = 16
patches_t = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2)               # (batch, N, C*p*p) = (2, 16, 192)
mine = torch.as_tensor(np.stack([patchify(x[b].numpy(), p) for b in range(2)]))
print("unfold gives", tuple(patches_t.shape), "- same numbers as my patchify:", torch.allclose(patches_t, mine))

conv = nn.Conv2d(3, d, kernel_size=p, stride=p)                                  # the patch embedding
lin = nn.Linear(3 * p * p, d)
with torch.no_grad():
    lin.weight.copy_(conv.weight.reshape(d, -1)); lin.bias.copy_(conv.bias)      # copy the conv weights into a linear layer
via_conv = conv(x).flatten(2).transpose(1, 2)                                    # (batch, d, 4, 4) -> (batch, N, d)
via_lin = lin(patches_t)
print("conv output", tuple(via_conv.shape), " linear output", tuple(via_lin.shape), " equal:", torch.allclose(via_conv, via_lin, atol=1e-5))
assert torch.allclose(patches_t, mine) and torch.allclose(via_conv, via_lin, atol=1e-5)
print("parameters of the conv:", count_params(conv), "= 192 * 16 + 16 =", 192 * 16 + 16)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
unfold gives (2, 16, 192) - same numbers as my patchify: True
conv output (2, 16, 16)  linear output (2, 16, 16)  equal: True
parameters of the conv: 3088 = 192 * 16 + 16 = 3088
```

   You should see: `unfold gives (2, 16, 192)` and `True` (PyTorch's `unfold` and your `patchify` agree: 16 patches per picture, 192 = 3 * 8 * 8 numbers each); then the convolution and the linear layer both give a tensor of shape (2, 16, 16) and are `equal: True` (to within 1e-5); finally the conv has **3088** parameters, which is 192 * 16 + 16: one weight per number of the patch and per output, plus one bias per output. So a strided convolution *is* "cut into patches and apply one shared linear layer to every patch". It is used in practice because it is faster than materialising the patches.

   *Exercise 2.2:* ViT-Base cuts a 224 by 224 colour picture with patches of 16 by 16 and embeds each patch in d = 768 dimensions. Compute by formula the number of parameters of `nn.Conv2d(3, 768, kernel_size=16, stride=16)` (weights plus biases) and store it in `conv_params`; then confirm it by building the layer and counting with `count_params`. Why it matters: it is the first layer of the network and you will meet this number again in the lesson's table of ViT-Base parameters (590,592).

```python
conv_params = None   # parameters of nn.Conv2d(3, 768, kernel_size=16, stride=16)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if conv_params is None:
    print("Fill in conv_params above, then run this cell again.")
else:
    real = count_params(nn.Conv2d(3, 768, kernel_size=16, stride=16))
    check("conv_params matches count_params", conv_params == real, "Weights: 768 filters of 3 * 16 * 16 numbers; biases: 768.")
    check("it is the lesson's 590,592", conv_params == 590592, "768 * 768 + 768.")
    RESULTS["p2_conv_params"] = int(conv_params)
```

   `conv_params` is 590,592 (768 * 768 + 768). A patch has 3 * 16 * 16 = 768 numbers and there are 768 outputs, so the weights are 768 * 768 = 589,824, plus 768 biases.

3. **Step 2.3. The counting formulas.** Now we count. For a picture of H by W pixels with C channels, patches of side `p` and embedding size `d`: the number of patches is (H/p) times (W/p); each patch is a vector of p times p times C numbers; the sequence the Transformer reads has one more token, the special **CLS** token, so its length is N + 1; and the patch embedding has (p*p*C)*d weights plus d biases. The function `vit_counts` returns all four numbers, and the block prints a table for five settings, including the lesson's ViT-Base (224, patch 16) and a tiny setting that we use from Part 4 on (16 by 16 pictures, patch 4, one channel, d = 32).

```python
def vit_counts(H, W, p, C, d):
    """Token arithmetic of a ViT: patches, numbers per patch, sequence length (with CLS) and patch-embedding parameters."""
    n = (H // p) * (W // p)
    vec = p * p * C
    return {"patches": n, "patch_len": vec, "seq_len": n + 1, "embed_params": vec * d + d}

configs = [("224, p = 16 (ViT-Base)", 224, 224, 16, 3, 768), ("224, p = 32", 224, 224, 32, 3, 768), ("224, p = 8", 224, 224, 8, 3, 768),
           ("384, p = 16", 384, 384, 16, 3, 768), ("16 grey, p = 4, d = 32", 16, 16, 4, 1, 32)]
print(f"{'picture and patch':26s} {'patches':>8s} {'numbers/patch':>14s} {'sequence':>9s} {'embed params':>13s}")
for name, H, W, p, C, d in configs:
    c = vit_counts(H, W, p, C, d)
    print(f"{name:26s} {c['patches']:8d} {c['patch_len']:14,d} {c['seq_len']:9d} {c['embed_params']:13,d}")
base = vit_counts(224, 224, 16, 3, 768)
assert (base["patches"], base["patch_len"], base["seq_len"], base["embed_params"]) == (196, 768, 197, 590592)
assert vit_counts(224, 224, 32, 3, 768)["patches"] == 49 and vit_counts(224, 224, 32, 3, 768)["patch_len"] == 3072
assert vit_counts(224, 224, 8, 3, 768)["patches"] == 784 and vit_counts(224, 224, 8, 3, 768)["patch_len"] == 192
print("my guesses were:", my_tokens_p16, "and", my_tokens_p8)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
picture and patch           patches  numbers/patch  sequence  embed params
224, p = 16 (ViT-Base)          196            768       197       590,592
224, p = 32                      49          3,072        50     2,360,064
224, p = 8                      784            192       785       148,224
384, p = 16                     576            768       577       590,592
16 grey, p = 4, d = 32           16             16        17           544
my guesses were: None and None
```

   You should see a table with five rows (the last line reminds you of your Imagine-first guesses, and prints `None` if you did not fill them in):

   - 224, p = 16: **196** patches, 768 numbers per patch, sequence **197**, 590,592 embedding parameters (this is the ViT-Base of the lesson);
   - 224, p = 32: **49** patches of 3,072 numbers, sequence 50, 2,360,064 parameters;
   - 224, p = 8: **784** patches of 192 numbers, sequence 785, 148,224 parameters;
   - 384, p = 16: 576 patches, sequence 577, 590,592 parameters;
   - 16 by 16 grey, p = 4, d = 32: 16 patches of 16 numbers, sequence 17, 544 parameters (the model of Part 4).

   The answers to the Imagine-first cell are 196 and 784: halving the patch side gives four times as many patches (14 by 14 against 28 by 28), but the vector of each patch gets four times shorter. Notice that the embedding has the **same** 590,592 parameters for 224 and 384 pixels: the parameter count depends on the patch, not on the size of the picture; only the number of tokens grows.

   *Exercise 2.3:* ViT-Large/14 reads 224 by 224 pictures with patches of 14 by 14 and an embedding size of d = 1024. Use `vit_counts` to compute its numbers and store the result in `large14`. Which of the four numbers surprises you? Why it matters: the notation "ViT-L/14" in papers means exactly this: the model size L and the patch side 14. You should be able to turn it into token counts in your head.

```python
large14 = None   # the dictionary that vit_counts returns for a 224 x 224 colour picture, p = 14, d = 1024
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if large14 is None:
    print("Fill in large14 above, then run this cell again.")
else:
    check("patches are right", large14["patches"] == 256, "224 / 14 = 16 patches per side, 16 * 16 in total.")
    check("patch_len is right", large14["patch_len"] == 14 * 14 * 3, "p * p * 3 numbers per patch.")
    check("seq_len and embed_params are right", large14["seq_len"] == 257 and large14["embed_params"] == 588 * 1024 + 1024, "Do not forget the CLS token and the 1024 biases.")
    RESULTS["p2_large14"] = dict(large14)
```

   `large14` is `{'patches': 256, 'patch_len': 588, 'seq_len': 257, 'embed_params': 603136}`. The surprise is usually that ViT-L/14 has more tokens (256) than ViT-B/16 (196), because the patches are smaller, so a "bigger" model is also more expensive in attention.

4. **Step 2.4. The cost of tokens: why patches?.** Why not use every pixel as a token? Because of **self-attention**: each head computes one score for every pair of tokens, so the number of scores is the *square* of the sequence length. The block counts the scores per head for a 224 by 224 picture twice: with 16 by 16 patches (197 tokens including CLS) and with one token per pixel (50,176 tokens), and then plots the score count against the patch size on a logarithmic axis.

```python
scores_patch = vit_counts(224, 224, 16, 3, 768)["seq_len"] ** 2          # 197 ** 2
scores_pixel = (224 * 224) ** 2                                            # one token per pixel
print(f"patches of 16: {scores_patch:,} scores per head")
print(f"one token per pixel: {scores_pixel:,} scores per head")
print(f"ratio: {scores_pixel / scores_patch:,.0f} times more scores")

sizes = [1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112]
scores = [vit_counts(224, 224, s, 3, 768)["seq_len"] ** 2 if s > 1 else scores_pixel for s in sizes]
plt.figure(figsize=(5, 3.2)); plt.plot(sizes, scores, "o-"); plt.xscale("log", base=2); plt.yscale("log")
plt.xlabel("patch side p (pixels); p = 1 means one token per pixel"); plt.ylabel("attention scores per head"); plt.title("224 x 224 picture"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
patches of 16: 38,809 scores per head
one token per pixel: 2,517,630,976 scores per head
ratio: 64,872 times more scores
```

   You should see: with patches of 16 there are **38,809** attention scores per head (197 times 197); with one token per pixel there are **2,517,630,976** (50,176 times 50,176); the ratio is **64,872 times** more scores. The figure shows the score count against the patch side on log-log axes: a falling line, steep at the left. Every doubling of the patch side divides the number of tokens by four and the number of scores by about sixteen. At p = 1 (one token per pixel) the picture needs about 2.5 billion scores per head and per layer, far too many; at p = 16 only about forty thousand. The patch is not a detail of the architecture: it is what makes a Transformer on pictures affordable.

   *Exercise 2.4:* a 448 by 448 picture with 16 by 16 patches has how many attention scores per head (with the CLS token)? Store it in `scores_448`. By what factor is that larger than for the 224 by 224 picture (`scores_patch`)? Store the factor, rounded to one decimal, in `growth`. Why it matters: doubling the side of the picture quadruples the number of tokens and multiplies the attention cost by about sixteen: this is why high-resolution input is expensive and why models tile or downscale it.

```python
scores_448 = None   # attention scores per head for a 448 x 448 picture, patches of 16, with CLS
growth = None       # scores_448 / scores_patch, rounded to 1 decimal
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if scores_448 is None or growth is None:
    print("Fill in scores_448 and growth above, then run this cell again.")
else:
    check("scores_448 is right", scores_448 == (28 * 28 + 1) ** 2, "(448 / 16) = 28 patches per side; the sequence is 28 * 28 + 1 tokens; square it.")
    check("growth is right", abs(growth - round((28 * 28 + 1) ** 2 / scores_patch, 1)) < 0.06, "scores_448 / scores_patch.")
    RESULTS["p2_scores_448"] = (int(scores_448), float(growth))
```

   `scores_448` is 616,225 (785 times 785) and `growth` is 15.9. The picture has 4 times as many patches, and the number of attention scores grows by almost 16, the square of 4.

5. **Step 2.5. Patch size on real pictures: empty patches.** Small patches give a finer description but cost many tokens, and in our line pictures most tokens are blank. We make 200 *clean* pictures (`noise=0`, so that blank patches are exactly zero), cut them with p = 2, 4 and 8, and measure the mean ink (sum of the pixel values) per patch and the share of patches that are completely empty. The figure shows one picture with its 4 by 4 grid and the empty patches shaded.

```python
Xclean, yclean = make_shapes(200, noise=0.0, seed=SEED)
print(f"{'p':>2s} {'tokens':>7s} {'numbers/patch':>14s} {'mean ink per patch':>19s} {'empty patches':>14s}")
empty_share = {}
for p in (2, 4, 8):
    P = np.stack([patchify(x, p) for x in Xclean])                  # (200, N, p*p)
    empty_share[p] = float((P.max(axis=2) == 0).mean())
    print(f"{p:2d} {P.shape[1]:7d} {P.shape[2]:14d} {P.sum(axis=2).mean():19.3f} {empty_share[p]:14.1%}")

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].imshow(Xclean[0, 0], cmap="gray_r")
P4 = patchify(Xclean[0], 4)
for i in range(16):
    if P4[i].max() == 0: ax[0].add_patch(plt.Rectangle(((i % 4) * 4 - 0.5, (i // 4) * 4 - 0.5), 4, 4, color="tab:blue", alpha=0.25))
ax[0].set_xticks(np.arange(-0.5, 16, 4)); ax[0].set_yticks(np.arange(-0.5, 16, 4)); ax[0].grid(color="k", lw=0.5); ax[0].set_xticklabels([]); ax[0].set_yticklabels([])
ax[0].set_title(f"{SHAPE_NAMES[yclean[0] ]}, empty patches shaded", fontsize=9)
ax[1].bar([str(p) for p in empty_share], [100 * v for v in empty_share.values()]); ax[1].set_xlabel("patch side p"); ax[1].set_ylabel("% of patches that are empty")
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 p  tokens  numbers/patch  mean ink per patch  empty patches
 2      64              4               0.237          87.5%
 4      16             16               0.950          72.5%
 8       4             64               3.800          35.0%
```

   You should see a table with the three patch sizes:

   - p = 2: 64 tokens of 4 numbers, mean ink per patch 0.237, **87.5 percent** of the patches empty;
   - p = 4: 16 tokens of 16 numbers, mean ink 0.950, **72.5 percent** empty;
   - p = 8: 4 tokens of 64 numbers, mean ink 3.800, **35.0 percent** empty.

   The figure has the first picture with its 4 by 4 grid (the empty patches are shaded blue) and a bar chart of the empty share. In these line pictures most of the area is blank, so with small patches most tokens carry nothing, and each of them still takes part in the attention. With big patches there are few empty tokens, but each token mixes a lot of detail into a single vector. That is the trade-off of the patch size: resolution of detail against the number of tokens (compute). Part 5 measures the effect on the accuracy.

   *Exercise 2.5:* with p = 1 every pixel is a patch. Compute the share of exactly-zero pixels in `Xclean` and store it in `empty_p1` (a number between 0 and 1, 3 decimals). Is it larger or smaller than the share for p = 2? Why it matters: a token that carries no ink still costs attention and memory; ViTs for large pictures with big empty areas (documents, sky, X-rays) waste a lot of tokens that way.

```python
empty_p1 = None   # share of pixels equal to 0 in Xclean (3 decimals)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if empty_p1 is None:
    print("Fill in empty_p1 above, then run this cell again.")
else:
    check("empty_p1 is right", abs(empty_p1 - round(float((Xclean == 0).mean()), 3)) < 0.0015, "The mean of the boolean array (Xclean == 0).")
    check("it is larger than the p = 2 share", empty_p1 > empty_share[2], "The smaller the patch, the more patches are empty.")
    RESULTS["p2_empty"] = {1: float(empty_p1), 2: empty_share[2], 4: empty_share[4], 8: empty_share[8]}
```

   `empty_p1` is 0.941: 94.1 percent of the pixels are exactly zero, more than the 87.5 percent of empty 2 by 2 patches. The smaller the patch, the more of them are empty, until at the limit nearly all tokens are blank.

6. **Step 2.6. Imagine and write: a patch is like a word.** The lesson says a ViT patch plays the role of a *word*. Write two short sentences: one that says **what is the same** (what the Transformer does with a patch and with a word) and one that says **where the analogy fails** (think about what a word is made of and what a patch is made of, about the dictionary of words against the endless variety of patches, and about what a neighbouring patch means). There is no code to run: store your two sentences in the dictionary `analogy`.

   *Exercise 2.6:* fill in `analogy["alike"]` and `analogy["differs"]` with one sentence each. Why it matters: writing the analogy and its limits is the best way to remember what a patch embedding is: a learned vector, with no fixed dictionary behind it.

```python
analogy = {"alike": None, "differs": None}   # two sentences of your own
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if any(v is None for v in analogy.values()):
    print("Fill in both sentences of analogy above, then run this cell again.")
else:
    check("both are real sentences", all(isinstance(v, str) and len(v.split()) >= 8 for v in analogy.values()), "Write at least eight words for each.")
    check("they are different", analogy["alike"] != analogy["differs"], "One says what is the same, the other where it fails.")
    RESULTS["p2_analogy"] = dict(analogy)
```

   For example: "Both are turned into a vector by a learned layer, and the Transformer then treats all the vectors in the same way, with attention between them" and "A word comes from a finite dictionary and carries meaning by itself, while a patch is a piece of continuous pixels that can be cut anywhere and means little alone." The key points: a patch and a word are both embedded into a vector by a learned layer and then handled by attention in the same way; but words come from a finite dictionary with meaning of their own, whereas patches are cut from a continuous image, can be cut anywhere, and mean little alone.

### Part 3. Why position embeddings? (25 min)

*Lesson section: "Positions".* Once a picture is a set of patch tokens, a worry appears: self-attention treats its tokens as a *set*, not as a *sequence*. If you swap two tokens, the attention layer does not notice where they came from. For text, Week 8 solved this by adding a position vector to each word. For a picture the problem is worse, because the answer to many questions depends on **where** things are: is the cat left of the dog, is the pointer above or below the dial? In this part you prove the problem with NumPy, see it on a real picture, then fix it with two kinds of position embeddings (learned and sinusoidal 2D), look at what they encode, and learn how to reuse them for a bigger picture. These are the ideas of the lesson's figure on shuffled patches.

**Imagine first.** A self-attention layer receives six patch tokens and returns six output vectors, one per token. Now we shuffle the six tokens and run the same layer again. Before you run any code, write your prediction: will the new outputs be (a) the same six vectors but in the shuffled order, (b) six completely different vectors, or (c) the same six vectors in the original order? Write "reordered", "different" or "same order". You will compare with Step 3.1.

```python
# Imagine first: shuffle the tokens, what does self-attention return?
my_prediction = None   # "reordered", "different" or "same order"
```

1. **Step 3.1. Permutation equivariance, by hand.** Here is a tiny self-attention layer in NumPy, exactly the one of Week 8 with a single head and with the queries, keys and values all equal to `X W` (to keep the matrices small). The six tokens have four numbers each, and both matrices are written out so that you can follow them with a pencil. We apply the layer to `X` and to a shuffled copy `X[perm]` and compare. The word for what you will find is **permutation equivariance**: if the input rows are permuted, the output rows are permuted in the same way.

```python
def softmax_rows(S):
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def self_attention(X, W):
    """Single-head self-attention with Q = K = V = X W. Returns (outputs, attention matrix)."""
    Q = X @ W
    A = softmax_rows(Q @ Q.T / np.sqrt(Q.shape[1]))     # (6, 6): row i says how much token i looks at each token
    return A @ Q, A

X = np.array([ [1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1] ], float)       # six tokens, four numbers each
W = np.array([ [0.5, -0.2, 0.1, 0.3], [0.0, 0.4, -0.3, 0.2], [-0.1, 0.3, 0.5, 0.0], [0.2, 0.1, 0.0, -0.4] ])
perm = [3, 0, 5, 1, 4, 2]                                 # token 3 first, then token 0, ...

out, A = self_attention(X, W)
out_shuf, _ = self_attention(X[perm], W)
print("outputs for the original order (rounded):"); print(out.round(3))
print("outputs for the shuffled order, shuffled back with perm:"); print(out_shuf[np.argsort(perm)].round(3))
print("rows of the original output, reordered, equal the new output:", np.allclose(out[perm], out_shuf))
print("same output row by row (no reordering):", np.allclose(out, out_shuf))
assert np.allclose(out[perm], out_shuf) and not np.allclose(out, out_shuf)
print("my prediction was:", my_prediction)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
outputs for the original order (rounded):
[ [0.326 0.273 0.151 0.052]
 [0.274 0.325 0.156 0.053]
 [0.317 0.289 0.131 0.079]
 [0.283 0.31  0.177 0.026]
 [0.309 0.285 0.177 0.064]
 [0.291 0.314 0.13  0.041] ]
outputs for the shuffled order, shuffled back with perm:
[ [0.326 0.273 0.151 0.052]
 [0.274 0.325 0.156 0.053]
 [0.317 0.289 0.131 0.079]
 [0.283 0.31  0.177 0.026]
 [0.309 0.285 0.177 0.064]
 [0.291 0.314 0.13  0.041] ]
rows of the original output, reordered, equal the new output: True
same output row by row (no reordering): False
my prediction was: None
```

   You should see two printed matrices of 6 rows and 4 columns that are **row-for-row the same numbers** (for example the first row is `0.326 0.273 0.151 0.052`), then `True` for "rows of the original output, reordered, equal the new output" and `False` for "same output row by row". So the answer to the Imagine-first cell is **"reordered"**: the layer returns the same six vectors, in the order in which you gave the tokens. The layer cannot tell which token came first; if the order carries meaning (a sentence, the layout of a picture), the layer is blind to it.

   **Why it happens.** The attention score between two tokens depends only on the two tokens themselves, never on their places. So when the rows of `X` are permuted, the matrix of scores is permuted in rows and columns, the softmax acts row by row, and the weighted sum gives the same vectors in the new order. Nothing in the layer can tell the first token from the last. A ViT that reads patches with no position information therefore sees a *bag* of patches.

   *Exercise 3.1:* choose any permutation of the numbers 0 to 5 as `my_perm` (a list with each number once) and store in `equivariant` whether `out[my_perm]` equals the output of the layer on `X[my_perm]` (`True` or `False`; use `np.allclose`). Why it matters: the claim must hold for every permutation, not just the one in the text; testing a claim on several cases is how you gain trust in it.

```python
my_perm = [5, 4, 3, 2, 1, 0]   # change it if you like: a list with the numbers 0..5 once each
equivariant = None             # True or False
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if equivariant is None:
    print("Fill in equivariant above, then run this cell again.")
else:
    check("my_perm is a real permutation", sorted(my_perm) == list(range(6)), "Each of 0..5 exactly once.")
    check("equivariant is True", bool(equivariant) is True, "Compute np.allclose(out[my_perm], self_attention(X[my_perm], W)[0]).")
    RESULTS["p3_equivariant"] = bool(equivariant)
```

   `equivariant` is `True` for every permutation. The layer only reorders its outputs in the same way as its inputs.

2. **Step 3.2. The picture version: a bag of patches.** Now the same thing on pictures. The task of Part 5, `make_left_right`, is "is the plus sign in the **left half** of the picture?": each picture has one plus sign (the target, arms of 2 pixels) and one filled 3 by 3 square (a distractor). We cut a picture into 16 patches of 4 by 4, shuffle the patches (`shuffled = unpatchify(patchify(pic)[order])`) and look at the result. To know where the plus sign goes, the helper `find_plus` first finds it in the original (ink at the centre and 2 pixels away on the four sides, but not at the four diagonal points, unlike the square), and we shuffle a mask of the plus with the same order of patches; the new label is 1 if the centre of the moved plus pixels is in the left half. The block picks a picture where the shuffle moves the plus sign to the other half, shows both pictures, and checks that the two **bags of patches are identical**: if we sort the patch vectors, the sorted lists are the same, although the labels differ.

```python
XLR, yLR = make_left_right(200, seed=SEED)

def find_plus(img):
    """(row, column) of the centre of the plus sign, or None. The plus has ink at the centre and 2 pixels away on its four sides, but not at the diagonal points (the 3 x 3 square reaches no further than 1 pixel)."""
    a = img[0] > 0.5
    for y in range(2, 14):
        for x in range(2, 14):
            if a[y, x] and a[y - 2, x] and a[y + 2, x] and a[y, x - 2] and a[y, x + 2] and not (a[y - 2, x - 2] or a[y - 2, x + 2] or a[y + 2, x - 2] or a[y + 2, x + 2]):
                return y, x
    return None

def plus_mask(pos):
    """A (1, 16, 16) mask with ones on the pixels of a plus sign centred at pos."""
    y, x = pos; m = np.zeros((1, 16, 16), np.float32)
    m[0, y - 1:y + 2, x - 2:x + 3] = 1; m[0, y - 2:y + 3, x - 1:x + 2] = 1
    return m

def shuffle_by(img, p, order):
    return unpatchify(patchify(img, p)[order], p, 1, 16, 16)

def moved_label(img, p, order):
    """Label (1 = left half) of the plus sign after the patches of the picture are put in the new order."""
    cols = np.nonzero(shuffle_by(plus_mask(find_plus(img)), p, order))[2]
    return int(cols.mean() < 7.5)

def sorted_bag(img, p):
    P = patchify(img, p)
    return P[np.lexsort(P.T)]                                 # sort the patch vectors (rows) in a fixed order

print("find_plus agrees with the true label on", int(sum((find_plus(XLR[i]) is not None) and int(find_plus(XLR[i])[1] < 8) == yLR[i] for i in range(200))), "of 200 pictures")
rng = np.random.RandomState(3)
for i in range(200):
    order = rng.permutation(16)
    if moved_label(XLR[i], 4, order) != yLR[i]:
        break
sh = shuffle_by(XLR[i], 4, order)
print(f"picture {i}: plus at {find_plus(XLR[i])}, label {yLR[i]};  after the shuffle its label is {moved_label(XLR[i], 4, order)}")
print("sorted bags of patch vectors identical:", np.array_equal(sorted_bag(XLR[i], 4), sorted_bag(sh, 4)))
assert np.array_equal(sorted_bag(XLR[i], 4), sorted_bag(sh, 4)) and moved_label(XLR[i], 4, order) != yLR[i]
show_images(np.stack([XLR[i], sh]), [f"original, label {yLR[i]}", f"shuffled, label {moved_label(XLR[i], 4, order)}"], n=2, cols=2)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
find_plus agrees with the true label on 200 of 200 pictures
picture 0: plus at (12, 2), label 1;  after the shuffle its label is 0
sorted bags of patch vectors identical: True
```

   You should see:

   - `find_plus agrees with the true label on 200 of 200 pictures`;
   - picture 0: the plus sign is at row 12, column 2 (left half, label 1); after the shuffle of the 16 patches the plus pixels have moved to the right half, so the label is **0**;
   - `sorted bags of patch vectors identical: True`;

   and a figure with the original and the shuffled picture side by side: two pictures that look different, with the labels 1 and 0. The pieces are the same, the answer is not. A network that is given only the bag of patches, without any information about where each patch came from, would have to give the same answer for both pictures, so it is wrong for at least one of them. This is why a ViT without positions can only guess on the left-or-right task.

```python
def flip_study(X, y, p=4, seed=1):
    """Shuffle the patches of every picture once. Return the share of pictures whose label (where the plus sign went) changed."""
    rng = np.random.RandomState(seed); n_patches = (16 // p) ** 2; flipped = 0
    for img, lab in zip(X, y):
        flipped += int(moved_label(img, p, rng.permutation(n_patches)) != lab)
    return flipped / len(X)

print(f"p = 4: the label changed in {flip_study(XLR, yLR, p=4):.1%} of the 200 shuffled pictures")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
p = 4: the label changed in 50.5% of the 200 shuffled pictures
```

   You should see: `p = 4: the label changed in 50.5% of the 200 shuffled pictures`. A random shuffle sends the plus sign to either half with about the same chance, so for a position-blind model about half of the shuffled copies of a picture would need the opposite answer.

   *Exercise 3.2:* repeat the flip study with patches of 8 by 8 (`p=8`, only four patches) and of 2 by 2 (`p=2`, 64 patches). Store the shares of changed labels in `flip8` and `flip2` (3 decimals). Why it matters: whatever the patch size, a model that only sees the bag of patches cannot tell the original from the shuffled picture, yet in a large share of the cases the right answer is different. The best such a model can do is guess.

```python
flip8 = None   # share of pictures whose label changes when the 8 x 8 patches are shuffled (3 decimals)
flip2 = None   # the same for 2 x 2 patches
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if flip8 is None or flip2 is None:
    print("Fill in flip8 and flip2 above, then run this cell again.")
else:
    check("flip8 is right", abs(flip8 - round(flip_study(XLR, yLR, p=8), 3)) < 0.0015, "Call flip_study(XLR, yLR, p=8).")
    check("flip2 is right", abs(flip2 - round(flip_study(XLR, yLR, p=2), 3)) < 0.0015, "Call flip_study(XLR, yLR, p=2).")
    check("a random shuffle changes the answer in roughly half of the cases", 0.3 < flip8 < 0.7 and 0.3 < flip2 < 0.7, "A shuffle sends the plus to either half with about equal chance.")
    RESULTS["p3_flip"] = (float(flip8), float(flip2))
```

   `flip8` is 0.48 and `flip2` is 0.46. With four large patches or with 64 small ones, about half of the labels change (48 and 46 percent): the same message as before.

3. **Step 3.3. Add positions: learned and sinusoidal 2D.** The fix is to add to every patch embedding a vector that tells *where* the patch was. There are two common ways. A **learned** table: a trainable array of shape (N, d), one row per position, that the network adjusts like any other weight (BERT and the original ViT do this). A **sinusoidal 2D** embedding: no training, built from the formula of Week 8, `sin(pos / 10000^(2i/d))` and `cos(...)`, in which half of the d numbers encode the **row** of the patch and the other half the **column**. The function `sincos_2d(grid, d)` below returns an array of shape (grid*grid, d). We print the vectors for a 4 by 4 grid with d = 8 and check that all 16 are different. (Week 8 interleaves sine and cosine; here the sines of one axis come first and then its cosines: the information is the same.)

```python
def sincos_1d(pos, d):
    """(n,) positions -> (n, d): d/2 sines followed by d/2 cosines with the frequencies of Week 8."""
    k = np.arange(d // 2)
    ang = pos[:, None] / (10000 ** (2 * k / d))[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)

def sincos_2d(grid, d):
    """Position vectors for a grid x grid layout of patches, in reading order: shape (grid * grid, d). First half of d: the row, second half: the column."""
    assert d % 4 == 0
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    return np.concatenate([sincos_1d(rows, d // 2), sincos_1d(cols, d // 2)], axis=1)

pos_learned = nn.Parameter(torch.randn(16, 8) * 0.02)       # the learned version: random at first, trained later
P8 = sincos_2d(4, 8)
print("learned table:", tuple(pos_learned.shape), " trainable numbers:", pos_learned.numel())
print("sinusoidal table:", P8.shape, " one row per patch (reading order); numbers 0-3 encode the row, numbers 4-7 the column")
print(P8.round(2))
distinct = len({tuple(r) for r in P8.round(6)})
print("different position vectors:", distinct, "of 16")
assert P8.shape == (16, 8) and distinct == 16
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
learned table: (16, 8)  trainable numbers: 128
sinusoidal table: (16, 8)  one row per patch (reading order); numbers 0-3 encode the row, numbers 4-7 the column
[ [ 0.    0.    1.    1.    0.    0.    1.    1.  ]
 [ 0.    0.    1.    1.    0.84  0.01  0.54  1.  ]
 [ 0.    0.    1.    1.    0.91  0.02 -0.42  1.  ]
 [ 0.    0.    1.    1.    0.14  0.03 -0.99  1.  ]
 [ 0.84  0.01  0.54  1.    0.    0.    1.    1.  ]
 [ 0.84  0.01  0.54  1.    0.84  0.01  0.54  1.  ]
 [ 0.84  0.01  0.54  1.    0.91  0.02 -0.42  1.  ]
 [ 0.84  0.01  0.54  1.    0.14  0.03 -0.99  1.  ]
 [ 0.91  0.02 -0.42  1.    0.    0.    1.    1.  ]
 [ 0.91  0.02 -0.42  1.    0.84  0.01  0.54  1.  ]
 [ 0.91  0.02 -0.42  1.    0.91  0.02 -0.42  1.  ]
 [ 0.91  0.02 -0.42  1.    0.14  0.03 -0.99  1.  ]
 [ 0.14  0.03 -0.99  1.    0.    0.    1.    1.  ]
 [ 0.14  0.03 -0.99  1.    0.84  0.01  0.54  1.  ]
 [ 0.14  0.03 -0.99  1.    0.91  0.02 -0.42  1.  ]
 [ 0.14  0.03 -0.99  1.    0.14  0.03 -0.99  1.  ] ]
different position vectors: 16 of 16
```

   You should see: the learned table has shape (16, 8) and 128 trainable numbers, the sinusoidal table has shape (16, 8) and is printed as a table of 16 rows: the first four numbers of a row encode the row of the patch (two sines then two cosines) and the last four the column. The first row (the patch at the top-left) is `0 0 1 1 0 0 1 1`: sines of 0 and cosines of 0. The second row (row 0, column 1) is `0 0 1 1 0.84 0.01 0.54 1`: the same row code and a different column code. The last line says **16 of 16** position vectors are different, so every patch has its own label.

   *Exercise 3.3:* build `sincos_2d(14, 768)`, the table for ViT-Base without the CLS token, and store its shape in `pe_shape` (a tuple) and the sum of the numbers of the vector of the first patch (row 0, column 0) in `sum00`. Predict the sum first: what are the sine and the cosine of 0? Why it matters: the first patch is at the origin, so all its sines are 0 and all its cosines are 1: a sanity check that you understand the formula.

```python
pe_shape = None   # shape of sincos_2d(14, 768)
sum00 = None      # sum of the 768 numbers of the position vector of the first patch
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if pe_shape is None or sum00 is None:
    print("Fill in pe_shape and sum00 above, then run this cell again.")
else:
    _pe = sincos_2d(14, 768)
    check("pe_shape is right", tuple(pe_shape) == _pe.shape == (196, 768), "A 14 x 14 grid has 196 positions.")
    check("sum00 is right", abs(sum00 - _pe[0].sum()) < 1e-6, "Half of the numbers are sin(0) = 0 and half are cos(0) = 1.")
    RESULTS["p3_pe"] = (tuple(pe_shape), float(sum00))
```

   `pe_shape` is (196, 768) and `sum00` is 384.0: half of the 768 numbers are cos(0) = 1 and the other half sin(0) = 0.

4. **Step 3.4. Positions break the symmetry.** Repeat Step 3.1, but add a position vector to each token *before* the attention layer. The position vectors belong to the **slots**: slot 0 always gets the vector for position 0, whatever token is put there. So when we shuffle the tokens, the positions do not move with them. If the layer is now to give the same rows in a different order, positions would have to follow the tokens; they do not, and the outputs are no longer permutations of each other. We use `sincos_1d` for six positions with d = 4.

```python
Ppos = sincos_1d(np.arange(6), 4)                      # six position vectors of 4 numbers each
out_p, _ = self_attention(X + Ppos, W)
out_p_shuf, _ = self_attention(X[perm] + Ppos, W)       # tokens shuffled, positions stay in their slots
print("without positions: largest difference between out[perm] and the shuffled output:", np.abs(out[perm] - out_shuf).max())
print("with positions:    largest difference between out[perm] and the shuffled output:", round(float(np.abs(out_p[perm] - out_p_shuf).max()), 3))
print("equal with positions:", np.allclose(out_p[perm], out_p_shuf))
assert not np.allclose(out_p[perm], out_p_shuf)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
without positions: largest difference between out[perm] and the shuffled output: 1.1102230246251565e-16
with positions:    largest difference between out[perm] and the shuffled output: 0.194
equal with positions: False
```

   You should see: without positions the largest difference between `out[perm]` and the shuffled output is about 1e-16 (rounding error: it is zero); with positions it is **0.194**, and `equal with positions: False`. The positions stay in their slots when the tokens move, so after the shuffle each token meets a different position vector and the layer gives different results. The layer is now **not** permutation equivariant, and it can use the place of a token.

   *Exercise 3.4:* store in `max_diff_pos` the largest absolute difference between `out_p[perm]` and `out_p_shuf`, rounded to 3 decimals, and in `breaks` whether it is bigger than 0.01 (`True` or `False`). Why it matters: equivariance is an exact property, so even a small difference means it is gone; but how large the difference is tells you how strongly the positions change the result.

```python
max_diff_pos = None   # largest absolute difference (3 decimals)
breaks = None         # True if it is larger than 0.01
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if max_diff_pos is None or breaks is None:
    print("Fill in max_diff_pos and breaks above, then run this cell again.")
else:
    check("max_diff_pos is right", abs(max_diff_pos - round(float(np.abs(out_p[perm] - out_p_shuf).max()), 3)) < 0.0015, "np.abs(out_p[perm] - out_p_shuf).max()")
    check("breaks is right", bool(breaks) is True, "The positions make the difference clearly larger than 0.01.")
    RESULTS["p3_break"] = float(max_diff_pos)
```

   `max_diff_pos` is 0.194 and `breaks` is `True`. The outputs are clearly not permutations of each other any more.

5. **Step 3.5. Similarity of positions.** What do the 2D sinusoidal vectors say about the geometry of the grid? We take an 8 by 8 grid with d = 64, normalise every position vector to length 1, and compute the cosine similarity between position (0, 0) (the top-left corner) and every other position. The result is an 8 by 8 map that we draw as a heat map with the numbers written in.

```python
Pe = sincos_2d(8, 64)
Pn = Pe / np.linalg.norm(Pe, axis=1, keepdims=True)
sim = (Pn[0] @ Pn.T).reshape(8, 8)                       # cosine similarity of every position with (0, 0)
plt.figure(figsize=(5.2, 4.4)); plt.imshow(sim, cmap="viridis")
for r in range(8):
    for c in range(8): plt.text(c, r, f"{sim[r, c]:.2f}", ha="center", va="center", color="w", fontsize=7)
plt.colorbar(label="cosine similarity with position (0, 0)"); plt.title("2D sinusoidal positions, 8 x 8 grid, d = 64"); plt.tight_layout(); plt.show()
dist = np.add.outer(np.arange(8), np.arange(8))           # Manhattan distance from (0, 0) = row + column
for k in (0, 1, 2, 4, 8, 14): print(f"Manhattan distance {k:2d}: mean similarity {sim[dist == k].mean():.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
Manhattan distance  0: mean similarity 1.000
Manhattan distance  1: mean similarity 0.979
Manhattan distance  2: mean similarity 0.938
Manhattan distance  4: mean similarity 0.862
Manhattan distance  8: mean similarity 0.786
Manhattan distance 14: mean similarity 0.714
```

   You should see a heat map of 8 by 8 numbers that is brightest (1.00) at the top-left corner and fades smoothly towards the opposite corner. The table lists the mean similarity by Manhattan distance (rows plus columns) from the corner: 1.000 at distance 0, **0.979** at distance 1, 0.938 at 2, 0.862 at 4, 0.786 at 8 and **0.714** at 14 (the far corner). So neighbouring positions have similar vectors and far positions are less alike: a smooth geometry that costs no training. The map is not a perfect distance: for example position (0, 7) is more similar (0.857) than the far corner although both are 7 steps away in one direction; high-frequency components make some far positions resemble the corner again.

   *Exercise 3.5:* store in `sim_near` the cosine similarity between position (0, 0) and its neighbour (0, 1), and in `sim_far` the similarity with the opposite corner (7, 7), both rounded to 3 decimals. Is the neighbour more similar? Why it matters: if close positions have similar vectors, the network gets a hint of "closeness" for free; but the hint is not perfect, as the map shows.

```python
sim_near = None   # similarity of position (0, 0) with (0, 1)
sim_far = None    # similarity of position (0, 0) with (7, 7)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if sim_near is None or sim_far is None:
    print("Fill in sim_near and sim_far above, then run this cell again.")
else:
    check("sim_near is right", abs(sim_near - round(float(sim[0, 1]), 3)) < 0.0015, "The entry in row 0, column 1 of sim.")
    check("sim_far is right", abs(sim_far - round(float(sim[7, 7]), 3)) < 0.0015, "The entry in row 7, column 7 of sim.")
    check("the neighbour is more similar than the far corner", sim_near > sim_far, "Check which value is larger.")
    RESULTS["p3_sim"] = (float(sim_near), float(sim_far))
```

   `sim_near` is 0.979 and `sim_far` is 0.714. The neighbour is clearly more similar than the opposite corner.

6. **Step 3.6. Another picture size: interpolate the table.** A learned position table has a fixed number of rows. Suppose we trained a ViT on a 4 by 4 grid of patches and now want to use it on a picture with a 6 by 6 grid (a bigger picture, or smaller patches). The sum of the 36 patch embeddings and the 16 positions fails with a shape error. The common remedy is to treat the table as a small picture with d channels and **resize it with bilinear interpolation** (`F.interpolate`), then fine-tune. The block shows the error, the repair, and the numbers.

```python
d = 32
table = nn.Parameter(torch.randn(1, 16, d) * 0.02)       # a position table "trained" for a 4 x 4 grid
tokens6 = torch.randn(1, 36, d)                          # 36 patch embeddings of a bigger picture (6 x 6 grid)
try:
    tokens6 + table
except RuntimeError as e:
    print("shape error:", str(e)[:75])

def resize_pos(table, old, new):
    """(1, old*old, d) learned positions -> (1, new*new, d) by bilinear interpolation of the old x old grid."""
    dd = table.shape[-1]
    g = table.reshape(1, old, old, dd).permute(0, 3, 1, 2)                       # (1, d, old, old): d channels
    g = F.interpolate(g, size=(new, new), mode="bilinear", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(1, new * new, dd)

new_table = resize_pos(table.detach(), 4, 6)
print("new table:", tuple(new_table.shape), " works now:", tuple((tokens6 + new_table).shape))
print("numbers in the old table:", table.numel(), " in the new table:", new_table.numel())
assert new_table.shape == (1, 36, d)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
shape error: The size of tensor a (36) must match the size of tensor b (16) at non-singl
new table: (1, 36, 32)  works now: (1, 36, 32)
numbers in the old table: 512  in the new table: 1152
```

   You should see: the first sum fails with a shape error (`The size of tensor a (36) must match the size of tensor b (16)...`); the resized table has shape (1, 36, 32) and the sum works, giving a result of shape (1, 36, 32). The old table has 16 * 32 = **512** numbers, the new one 36 * 32 = **1152**. Interpolation does not add trainable parameters by itself: it computes the 1,152 numbers from the 512; they are a good starting point that the network then adjusts when it is trained further (fine-tuned) at the new resolution.

   *Exercise 3.6:* the position table of ViT-Base for 224-pixel pictures has a 14 by 14 grid (ignoring the CLS token). For a 384-pixel picture with the same patches of 16 the grid is 24 by 24. Use `resize_pos` on a table of zeros to get the new table; store its shape in `new_shape` and its number of entries in `new_numbers`. Why it matters: this is what libraries do when you load a ViT trained at 224 and feed it a larger image; knowing that the positions are resized, not re-learned, explains why accuracy at a new size usually needs a short fine-tuning.

```python
new_shape = None     # shape of the resized table
new_numbers = None   # how many numbers it holds
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if new_shape is None or new_numbers is None:
    print("Fill in new_shape and new_numbers above, then run this cell again.")
else:
    check("new_shape is right", tuple(new_shape) == (1, 576, 768), "24 * 24 = 576 positions of 768 numbers.")
    check("new_numbers is right", new_numbers == 576 * 768, "576 * 768 = 442,368 (the old table had 196 * 768 = 150,528).")
    RESULTS["p3_resize"] = (tuple(new_shape), int(new_numbers))
```

   `new_shape` is (1, 576, 768) and `new_numbers` is 442,368 (the old table had 196 * 768 = 150,528). ViT-Base at 384 pixels has 576 patches, almost three times as many tokens as at 224.

### Part 4. Build a tiny Vision Transformer (30 min)

*Lesson section: "Vision Transformer".* In Part 2 you cut pictures into patches and counted tokens, and in Part 3 you saw why a Transformer needs position embeddings. Now you put the pieces together and build a complete Vision Transformer of your own, small enough to train in a few seconds on a CPU. It has five parts: a patch embedding that turns each patch into a vector, a learned CLS token that collects the answer, a table of learned position embeddings, a stack of Transformer encoder layers, and a linear head. ==You will write the patch embedding and the forward pass yourself, count every parameter by formula and with the computer, run two sanity checks that every new model should pass, train the model on the five line shapes of Week 4, compare it honestly with a small CNN, and finally look inside to see where the CLS token looks.== A tiny ViT trained from scratch on a few hundred pictures is a weak learner by design (it has no built-in idea that nearby pixels belong together), so this part reports what you measure, not what a big pretrained ViT would do. All numbers are for `SEED = 42`.

**Imagine first.** Picture the model as a filing office. A picture arrives and is cut into 16 small squares (patches). Each square is copied onto a card with 32 numbers (the patch embedding). One extra blank card (the CLS token) is added to the pile and every card gets a sticker with its place number (the position embedding). Then two rounds of clerks (the encoder layers) pass information between the cards, and in the end only the extra card is read to decide which of 5 shapes the picture shows. Every number printed on a card, a sticker or a clerk's rulebook is a parameter. Before you run any code, guess how many numbers the whole office holds: a few thousand, a few tens of thousands, or hundreds of thousands? Write your guess in the cell below. You compare it with Step 4.3.

```python
# Imagine first: how many numbers does a tiny ViT hold? (d = 32, 2 layers, 4 heads, patch 4, 16 x 16 pictures, 5 classes)
my_guess_params = None   # write a number, for example 5000, 30000 or 300000
```

1. **Step 4.1. From patches to a sequence of tokens: PatchEmbed.** A ViT starts by cutting the picture into patches and mapping every patch with the same linear layer to a vector of d numbers. Writing that as loops is slow, so everybody uses a trick you met in Part 2: a convolution whose kernel is as large as the patch and whose stride equals the patch size looks at each patch exactly once, and its weights are the weights of the linear layer. The output of the convolution is a grid of d-vectors, and two reshaping operations turn the grid into a sequence. The cell below follows the shape of a batch of 8 pictures of 16 x 16 pixels through the three operations.

```python
x = torch.randn(8, 1, 16, 16)                                              # a batch of 8 greyscale pictures
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=4)  # one 4 x 4 window per patch, no overlap
y = conv(x)
print("after the Conv2d       :", tuple(y.shape), " (batch, d, rows, columns of the patch grid)")
y = y.flatten(2)
print("after flatten(2)       :", tuple(y.shape), " (batch, d, N)")
y = y.transpose(1, 2)
print("after transpose(1, 2)  :", tuple(y.shape), " (batch, N, d): a sequence of 16 tokens")
print("parameters of the Conv2d:", count_params(conv), "= 1*4*4*32 weights + 32 biases")
```

   *Exercise 4.1:* Write the class `MyPatchEmbed` yourself: a `Conv2d` with `kernel_size=patch` and `stride=patch`, and a forward method that applies it, flattens the two grid axes and moves the d axis to the end. Why it matters: this class is the entry door of every Vision Transformer. If you can write it, you know exactly what a patch token is.

```python
class MyPatchEmbed(nn.Module):
    def __init__(self, img=16, patch=4, in_ch=1, d=32):
        super().__init__()
        self.n_patches = (img // patch) ** 2
        self.proj = None      # YOUR CODE: a Conv2d from in_ch channels to d channels with kernel_size=patch and stride=patch
    def forward(self, x):
        return None           # YOUR CODE: apply self.proj, flatten the two grid axes (flatten(2)), put d last (transpose(1, 2))
pe = MyPatchEmbed()
if pe.proj is None:
    print("Fill in MyPatchEmbed above, then run this cell again.")
else:
    out = pe(torch.randn(8, 1, 16, 16))
    check("output shape is (8, 16, 32)", out is not None and tuple(out.shape) == (8, 16, 32), "flatten(2) then transpose(1, 2) gives (batch, N, d)")
    check("544 parameters", count_params(pe) == 544, "Conv2d(1, 32, kernel_size=4, stride=4) has 1*4*4*32 weights and 32 biases")
```

   *What it means.* `Conv2d(1, 32, 4, 4)` gives a tensor of shape (batch, 32, 4, 4); `flatten(2)` merges the last two axes into 16; `transpose(1, 2)` swaps the d axis and the token axis. The check also tests the 544 parameters. Below the exercise we define `PatchEmbed`, which is the same class, and use it from now on.

```python
class PatchEmbed(nn.Module):
    """Cut a picture into non-overlapping patches and map each patch to a d-vector: (batch, C, H, W) -> (batch, N, d)."""
    def __init__(self, img=16, patch=4, in_ch=1, d=32):
        super().__init__()
        self.n_patches = (img // patch) ** 2
        self.proj = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

pe = PatchEmbed()
tokens = pe(torch.randn(8, 1, 16, 16))
print("PatchEmbed output:", tuple(tokens.shape), " parameters:", count_params(pe))
assert tuple(tokens.shape) == (8, 16, 32) and count_params(pe) == 544
RESULTS["p4_patch_embed_shape"] = tuple(tokens.shape)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
after the Conv2d       : (8, 32, 4, 4)  (batch, d, rows, columns of the patch grid)
after flatten(2)       : (8, 32, 16)  (batch, d, N)
after transpose(1, 2)  : (8, 16, 32)  (batch, N, d): a sequence of 16 tokens
parameters of the Conv2d: 544 = 1*4*4*32 weights + 32 biases
PatchEmbed output: (8, 16, 32)  parameters: 544
```

   **What you should see.**

   You see the shapes (8, 32, 4, 4) after the convolution (a 4 x 4 grid of patches, 32 numbers per patch), (8, 32, 16) after `flatten(2)` (the 16 grid cells in a row) and (8, 16, 32) after `transpose(1, 2)`: a batch of 8 sequences of 16 tokens with 32 numbers each. The convolution has 544 parameters: 1 x 4 x 4 x 32 = 512 weights plus 32 biases. **This is the whole patch embedding.** Everything after it treats the 16 vectors like the 16 word vectors of a sentence.

   Tips and common mistakes:

   - `stride=patch` is what makes the patches do not overlap. With a stride of 1 you would get one vector per pixel position, and a sequence far too long.
   - The batch axis never changes. When a shape surprises you, print it after every operation, as the cell does.

   Try also (optional):

   - Change the patch size to 8 and the number of output channels to 64. Predict the shape after `transpose(1, 2)` before you run the cell. We get (8, 4, 64).

2. **Step 4.2. The TinyViT: patch tokens, a CLS token, positions, a Transformer encoder, a head.** Now the whole model. The cell below builds each piece by hand and prints the shape after every operation, so you see what the forward pass has to do. The CLS token is one learned vector that is copied in front of every picture's sequence; after the encoder, only its output goes to the classification head, so the encoder is forced to collect everything the head needs in that one vector. The position table has 17 rows, one for each patch and one for the CLS token. For the encoder we use PyTorch's `nn.TransformerEncoder` (the same building block as your Week 8 block, with `batch_first=True` and `norm_first=True`, a feed-forward width of 4d, and a final LayerNorm). Then you write the forward pass of the model in an exercise, and below it we define the full `TinyViT` (with the options for the experiments of Part 5).

```python
d, N = 32, 16
pe = PatchEmbed(16, 4, 1, d)
cls_tok = nn.Parameter(torch.zeros(1, 1, d))                      # one learned vector, copied for every picture
pos_tab = nn.Parameter(torch.randn(1, N + 1, d) * 0.02)           # one learned vector per position (CLS included)
layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
encoder = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(d), enable_nested_tensor=False)
head = nn.Linear(d, 5)

x = torch.randn(8, 1, 16, 16)
t = pe(x);                                  print("1. patch tokens          :", tuple(t.shape))
t = torch.cat([cls_tok.expand(8, -1, -1), t], dim=1); print("2. with the CLS token    :", tuple(t.shape))
t = t + pos_tab;                            print("3. plus positions        :", tuple(t.shape))
h = encoder(t);                             print("4. after the encoder     :", tuple(h.shape))
z = h[:, 0];                                print("5. the CLS output        :", tuple(z.shape))
logits = head(z);                           print("6. class scores          :", tuple(logits.shape))
```

   *Exercise 4.2:* Fill in the `forward` method of `MyTinyViT` in five lines: (1) the patch tokens, (2) the CLS token in front, (3) the positions added, (4) the encoder, (5) the head applied to the CLS output (position 0). Why it matters: it is the whole ViT in five lines. Once you can write them, the recipe of the lesson is no longer a diagram but something you can run.

```python
class MyTinyViT(nn.Module):
    def __init__(self, img=16, patch=4, in_ch=1, d=32, heads=4, layers=2, n_classes=5):
        super().__init__()
        self.patch_embed = PatchEmbed(img, patch, in_ch, d)
        n = self.patch_embed.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
        nn.init.normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)   # a small head: the first guess is about uniform
    def forward(self, x):
        return None   # YOUR CODE, five lines: tokens = patch_embed(x); prepend the CLS token (cat on dim 1); add self.pos; run the encoder; head of the CLS output (position 0)
torch.manual_seed(0); mine = MyTinyViT()
out = mine(torch.randn(8, 1, 16, 16))
if out is None:
    print("Fill in the forward method above, then run this cell again.")
else:
    check("class scores have shape (8, 5)", tuple(out.shape) == (8, 5), "use head(h[:, 0]) on the encoder output h of shape (8, 17, 32)")
    check("the output depends on the picture", float((mine(torch.randn(8, 1, 16, 16)) - out).abs().max()) > 1e-6, "did you use x?")
```

   *What it means.* The check tests the shape of the scores (8, 5) and that they change with the picture. The comparison in the next cell is stricter: it loads our weights into your class and compares the scores. In the notebook the solution below shows only the forward method; paste it into the class.

```python
class TinyViT(nn.Module):
    """A small Vision Transformer. pool: 'cls' (head on the CLS token) or 'mean' (head on the mean of the patch tokens).
    use_pos=False removes the position information; pos_kind: 'learned' table or fixed 'sincos' (see sincos_2d in Step 5.6)."""
    def __init__(self, img=16, patch=4, in_ch=1, d=32, heads=4, layers=2, n_classes=5, use_pos=True, pool="cls", pos_kind="learned"):
        super().__init__()
        self.pool, self.use_pos = pool, use_pos
        self.patch_embed = PatchEmbed(img, patch, in_ch, d)
        n = self.patch_embed.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        if use_pos and pos_kind == "sincos":
            fixed = torch.cat([torch.zeros(1, d), torch.tensor(sincos_2d(img // patch, d), dtype=torch.float32)])
            self.register_buffer("pos", fixed[None])                      # not trained
        elif use_pos:
            self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)      # trained
        else:
            self.pos = None
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
        nn.init.normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)   # a small head: the first guess is about uniform
    def embed(self, x):
        t = torch.cat([self.cls.expand(len(x), -1, -1), self.patch_embed(x)], 1)
        return t + self.pos if self.pos is not None else t
    def forward(self, x):
        h = self.encoder(self.embed(x))
        return self.head(h[:, 0] if self.pool == "cls" else h[:, 1:].mean(1))
    @torch.no_grad()
    def cls_attention(self, x):
        """Attention of the CLS token to each patch in the last layer, averaged over heads: (batch, N)."""
        t = self.embed(x)
        for lay in self.encoder.layers[:-1]: t = lay(t)
        last = self.encoder.layers[-1]; h = last.norm1(t)
        _, w = last.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        return w[:, 0, 1:]

torch.manual_seed(SEED); vit = TinyViT()
xb = torch.randn(8, 1, 16, 16)
print("TinyViT on a batch of 8:", tuple(xb.shape), "->", tuple(vit(xb).shape), " (batch, classes)")
print("tokens per picture: 16 patches + 1 CLS =", vit.embed(xb).shape[1], "; each token has", vit.embed(xb).shape[2], "numbers")
try:                                   # does your forward agree with ours? (copy the weights of ours into yours)
    mine = MyTinyViT(); mine.load_state_dict(vit.state_dict())
    ok = mine(xb) is not None and torch.allclose(mine(xb), vit(xb), atol=1e-5)
    check("your MyTinyViT gives the same scores as TinyViT", ok, "compare your forward with the answer: CLS first, then positions")
except Exception as e:
    print("MyTinyViT is not finished yet (that is fine, TinyViT works anyway):", type(e).__name__)
assert tuple(vit(xb).shape) == (8, 5)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
1. patch tokens          : (8, 16, 32)
2. with the CLS token    : (8, 17, 32)
3. plus positions        : (8, 17, 32)
4. after the encoder     : (8, 17, 32)
5. the CLS output        : (8, 32)
6. class scores          : (8, 5)
TinyViT on a batch of 8: (8, 1, 16, 16) -> (8, 5)  (batch, classes)
tokens per picture: 16 patches + 1 CLS = 17 ; each token has 32 numbers
✓ your MyTinyViT gives the same scores as TinyViT
```

   **What you should see.**

   The six lines print (8, 16, 32) for the patch tokens, (8, 17, 32) after the CLS token is put in front and after the positions are added, (8, 17, 32) after the encoder (a Transformer layer never changes the shape), (8, 32) for the CLS output and (8, 5) for the class scores. After your exercise, the cell under it builds `TinyViT`, prints that a batch of 8 pictures gives scores of shape (8, 5), that every picture has 17 tokens of 32 numbers, and checks that your `MyTinyViT` gives exactly the same scores as ours when it receives our weights (a green tick). **If the tick is there, your forward pass is the same computation as ours.**

   Tips and common mistakes:

   - The CLS token starts at zero and the position table at small random numbers (0.02 times a normal random number); both are trained like all other weights. The head also starts with small weights, so that the first guess is almost uniform (you check this in Step 4.4).
   - `self.cls.expand(len(t), -1, -1)` copies the one CLS vector for every picture in the batch without using extra memory. Without `expand` the shapes do not match.
   - `TinyViT` has the options `use_pos=False` (no position information), `pool="mean"` (average of the patch outputs instead of the CLS output) and `pos_kind="sincos"` (fixed positions). They are for Part 5; with the defaults it is exactly your model.

   Try also (optional):

   - Print `vit.encoder` to see the two layers with their attention, feed-forward and normalisation modules.
   - Replace `h[:, 0]` by `h[:, 5]` in your forward pass. The model still runs, but the head now reads a patch token. What would you expect the training to do?

3. **Step 4.3. Count the parameters, by formula and with count_params.** Every parameter of the model belongs to one of six groups: the patch embedding (the convolution), the CLS vector, the position table, the encoder layers, the final LayerNorm and the head. The cell below writes the count as a formula (the Week 8 counting, with the exact bias terms of PyTorch's layer) and compares it with `count_params`. It also defines `vit_counts(H, W, p, C, d)`, the token arithmetic of Part 2, and then applies the same formula to ViT-Base (d = 768, 12 layers, 224 x 224 pictures, 16 x 16 patches) without building it: a model with 85 million parameters needs no training for us to count it.

```python
def vit_counts(H, W, p, C, d):
    """The token arithmetic of a ViT for an H x W picture with C channels, patch size p and width d."""
    n = (H // p) * (W // p); pd = p * p * C
    return dict(patches=n, patch_dim=pd, seq_len=n + 1, embed_params=pd * d + d)

def vit_param_formula(d, layers, n_patches, patch_dim, n_classes=0, mlp=4):
    """Parameters of the TinyViT layout: patch embedding + CLS + positions + L encoder layers + final LayerNorm + head."""
    per_layer = (4 * d * d + 4 * d) + (2 * mlp * d * d + mlp * d + d) + 4 * d     # attention + MLP + two LayerNorms
    head = d * n_classes + n_classes if n_classes else 0
    parts = dict(patch_embed=patch_dim * d + d, cls=d, positions=(n_patches + 1) * d, layers=layers * per_layer, final_norm=2 * d, head=head)
    return parts, per_layer

parts, per_layer = vit_param_formula(32, 2, 16, 16, n_classes=5)
for k, v in parts.items(): print("%-12s %8d" % (k, v))
print("per encoder layer:", per_layer, "= 12*d*d + 13*d =", 12 * 32 * 32 + 13 * 32)
print("by formula   :", sum(parts.values()))
print("count_params :", count_params(vit))
assert sum(parts.values()) == count_params(vit)
RESULTS["p4_tinyvit_params"] = count_params(vit)
print("your Imagine-first guess:", my_guess_params)
```

```python
# The same formula for the real ViT-Base (d = 768, 12 layers, 224 x 224 pictures, 16 x 16 patches): we only count, we do not build it
c = vit_counts(224, 224, 16, 3, 768)
base, base_layer = vit_param_formula(768, 12, c["patches"], c["patch_dim"])
print("tokens: %d patches of %d numbers, sequence %d; patch embedding %d parameters" % (c["patches"], c["patch_dim"], c["seq_len"], c["embed_params"]))
print("per encoder layer:", base_layer, "  12 layers:", base["layers"])
print("positions:", base["positions"], " CLS:", base["cls"], " total without a head: %d (about %.1f M)" % (sum(base.values()), sum(base.values()) / 1e6))
assert (c["patches"], c["patch_dim"], c["seq_len"], c["embed_params"]) == (196, 768, 197, 590592)
assert base_layer == 7087872 and base["layers"] == 85054464 and base["positions"] == 151296
RESULTS["p4_vit_base"] = sum(base.values())
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
patch_embed       544
cls                32
positions         544
layers          25408
final_norm         64
head              165
per encoder layer: 12704 = 12*d*d + 13*d = 12704
by formula   : 26757
count_params : 26757
your Imagine-first guess: None
tokens: 196 patches of 768 numbers, sequence 197; patch embedding 590592 parameters
per encoder layer: 7087872   12 layers: 85054464
positions: 151296  CLS: 768  total without a head: 85798656 (about 85.8 M)
```

   **What you should see.**

   For the tiny model the formula gives patch embedding 544, CLS 32, positions 544 (17 x 32), the two layers 25,408 (12,704 each), the final LayerNorm 64 and the head 165 (32 x 5 + 5): **26,757 in total, exactly what `count_params` finds**, and the assert passes. A layer has 12 d squared + 13 d numbers: 4 d squared + 4 d for attention (the query, key, value and output matrices with their biases), 8 d squared + 5 d for the feed-forward part (d to 4 d and back, with biases) and 4 d for the two LayerNorms (2 d each). The 13 d are the bias terms of the Week 8 counting, which uses only the squares. Your Imagine-first guess is compared with 26,757 in the first block.

   For ViT-Base the second cell gives 196 patches of 768 numbers, a sequence of 197 and a patch embedding of 590,592 parameters (768 x 768 + 768), exactly the lesson's numbers. One encoder layer has 7,087,872 parameters, twelve have 85,054,464, the position table has 197 x 768 = 151,296 numbers, the CLS token 768 and the final LayerNorm 1,536. The total without a head is **85,798,656, about 85.8 million**. The 85,054,464 of the lesson counts only the twelve blocks; adding the embedding, the CLS token, the positions and the last norm gives the 85.8 million of the published model.

   Tips and common mistakes:

   - Do not forget the final LayerNorm and the biases: leaving them out is the usual reason why your count is a few hundred short.
   - The parameters of the patch embedding (590,592) are tiny next to the blocks (85 million): almost all the numbers of a ViT sit in the encoder layers, as in a language model.

   Try also (optional):

   - Change the head to 1000 classes for ViT-Base with `n_classes=1000`: the head adds 769,000 parameters. How much does it change the total in percent? (Less than 1 percent.)

   *Exercise 4.3:* Count the parameters of a ViT with d = 48, 2 layers, 4 heads, patch 4, 16 x 16 pictures and 2 classes (the model of Part 5) and store the number in `params_by_hand`. Why it matters: counting by hand is the way to know what a model costs before you train it, and it is the check that you understood the architecture.

```python
params_by_hand = None   # YOUR CODE: the total for d = 48, 2 layers, 4 heads, patch 4, 16 x 16 pictures, 2 classes, using vit_param_formula or the pieces
if params_by_hand is None:
    print("Fill in params_by_hand above, then run this cell again.")
else:
    torch.manual_seed(0); m48 = TinyViT(d=48, heads=4, layers=2, n_classes=2)
    check("matches count_params", params_by_hand == count_params(m48), "patch embed 16*48+48, CLS 48, positions 17*48, 2 layers of 12d^2+13d, final norm 2d, head 48*2+2")
```

   *What it means.* The patch embedding has 16 x 48 + 48 = 816, the CLS 48, the positions 17 x 48 = 816, each layer 12 x 48 x 48 + 13 x 48 = 28,272 (two layers 56,544), the final norm 96 and the head 48 x 2 + 2 = 98: together 58,418. Notice that more than 96 percent of the numbers are in the two layers.

4. **Step 4.4. Sanity checks before training: the starting loss and an overfitting test.** Before you train a new model for a long time, check two things that cost seconds. First, a model that knows nothing about 5 equally likely classes should start with a loss of ln(5) = 1.609: if you see 5 or 0.1 at the start, something is wrong with the shapes, the labels or the initialisation. Second, a model must be able to overfit a tiny data set: if it cannot memorise 64 pictures, it will never learn 400. We make 400 training and 200 test pictures with `make_shapes` (centred, five shapes, seeds 1 and 2) and run both checks.

```python
Xs_tr, ys_tr = make_shapes(400, seed=1)       # five centred line shapes, 16 x 16, 400 for training
Xs_te, ys_te = make_shapes(200, seed=2)       # 200 the model never sees
EP = 3 if QUICK else 30
torch.manual_seed(SEED); m = TinyViT()
with torch.no_grad():
    loss0 = float(F.cross_entropy(m(torch.as_tensor(Xs_tr)), torch.as_tensor(ys_tr)))
print("starting loss: %.3f   ln(5) = %.3f" % (loss0, math.log(5)))
assert abs(loss0 - math.log(5)) < 0.15
```

```python
torch.manual_seed(SEED); m = TinyViT()
h64 = train_classifier(m, Xs_tr[:64], ys_tr[:64], Xs_tr[:64], ys_tr[:64], epochs=10 if QUICK else 60, lr=3e-3, batch=16)
print("overfit 64 pictures: train accuracy by epoch 10, 30, 60:", [round(h64["train_acc"][i], 2) for i in (9, 29, 59) if i < len(h64["train_acc"])])
print("final loss %.3f, train accuracy %.2f" % (h64["loss"][-1], h64["train_acc"][-1]))
RESULTS["p4_overfit64"] = round(h64["train_acc"][-1], 2)
assert QUICK or h64["train_acc"][-1] > 0.95
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
starting loss: 1.618   ln(5) = 1.609
overfit 64 pictures: train accuracy by epoch 10, 30, 60: [1.0, 1.0, 1.0]
final loss 0.004, train accuracy 1.00
```

   **What you should see.**

   The starting loss is 1.618 against ln(5) = 1.609: the small head makes the first guess almost uniform. The second cell trains on 64 pictures with batches of 16 and a learning rate of 0.003; the training accuracy is already 1.00 after 10 epochs and the loss is 0.004 after 60. **Both checks pass**, so the architecture can learn and the training loop works. Without the small head initialisation (PyTorch's default) we measured a starting loss of 1.84, which is not wrong but is a worse start.

   Tips and common mistakes:

   - Always run these two checks first when you build a model from scratch. A model that fails the overfitting test has a bug, not a data problem.
   - The overfitting test says nothing about how well the model will do on new pictures; that is the job of the test set in Step 4.5.

   Try also (optional):

   - Break the model on purpose: remove the `self.pos` addition from `embed` in a copy and watch the overfitting test. It still passes, because a bag of patches is enough to tell five centred line shapes apart. This is the point of Part 5, where the task needs the positions.

   *Exercise 4.4:* Store the starting loss of a model that knows nothing about 5 classes in `expected_start` (natural logarithm). Why it matters: you will use this number every time you train a classifier in the course.

```python
expected_start = None   # YOUR CODE: the loss of a model that knows nothing about 5 equally likely classes (natural log)
if expected_start is None:
    print("Fill in expected_start above, then run this cell again.")
else:
    check("ln(5)", abs(expected_start - math.log(5)) < 1e-6, "cross-entropy of a uniform guess over 5 classes is -ln(1/5) = ln(5)")
```

   *What it means.* A uniform guess gives each class the probability 1/5, so the loss is minus the log of 1/5, which is ln(5) = 1.6094.

5. **Step 4.5. Train on make_shapes and compare with a small CNN.** Now the real training: 30 epochs of Adam at a learning rate of 0.003 on the 400 pictures. For a fair comparison we build a small CNN in the style of Week 4 (two convolutions with 32 and 64 channels, each followed by ReLU and max pooling, then a linear layer) with a similar number of parameters, and train it with the same recipe. We print a table and draw the learning curves.

```python
class SmallCNN(nn.Module):
    """A two-convolution CNN (Week 4 style) for pictures of side img."""
    def __init__(self, in_ch=1, c1=32, c2=64, n_classes=5, img=16):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
        self.out = nn.Linear(c2 * (img // 4) ** 2, n_classes)
    def forward(self, x): return self.out(self.f(x))

torch.manual_seed(SEED); vit45 = TinyViT()
torch.manual_seed(SEED); cnn45 = SmallCNN()
t0 = time.time(); h_vit = train_classifier(vit45, Xs_tr, ys_tr, Xs_te, ys_te, epochs=EP, lr=3e-3); t_vit = time.time() - t0
t0 = time.time(); h_cnn = train_classifier(cnn45, Xs_tr, ys_tr, Xs_te, ys_te, epochs=EP, lr=3e-3); t_cnn = time.time() - t0
print("%-10s %9s %10s %10s %9s" % ("model", "params", "train acc", "test acc", "seconds"))
print("%-10s %9d %10.3f %10.3f %9.1f" % ("TinyViT", count_params(vit45), h_vit["train_acc"][-1], h_vit["test_acc"][-1], t_vit))
print("%-10s %9d %10.3f %10.3f %9.1f" % ("SmallCNN", count_params(cnn45), h_cnn["train_acc"][-1], h_cnn["test_acc"][-1], t_cnn))
RESULTS["p4_vit_vs_cnn"] = (round(h_vit["test_acc"][-1], 3), round(h_cnn["test_acc"][-1], 3))
```

```python
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for h, name in [(h_vit, "TinyViT"), (h_cnn, "SmallCNN")]:
    ax[0].plot(h["loss"], label=name); ax[1].plot(h["test_acc"], label=name)
ax[0].set_title("training loss"); ax[1].set_title("test accuracy"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
model         params  train acc   test acc   seconds
TinyViT        26757      1.000      1.000       2.4
SmallCNN       23941      1.000      1.000       2.7
```

   **What you should see.**

   The TinyViT has 26,757 parameters, the CNN 23,941. Both reach a **training accuracy of 1.000 and a test accuracy of 1.000** after 30 epochs; the ViT takes about 2.4 seconds and the CNN about 2.7 (times will differ on your machine). So the honest result is a tie: the five centred line shapes are an easy task, and a small ViT can solve it. The curves show that the ViT's test accuracy passes 0.95 already in the second epoch (Exercise 4.5). **Do not read this as 'ViTs are as good as CNNs'**: Step 5.5 shows what happens when the task is harder or the data smaller.

   Tips and common mistakes:

   - Equal parameter counts do not mean an equal amount of computation, but both models are tiny, so the comparison is fair enough for a lesson.
   - If your numbers are a little different (for example 0.995), your library version is different. The result that matters is that both models solve the task.

   Try also (optional):

   - Train both models with `shift=2` in `make_shapes` (the shapes move up to 2 pixels from the centre) and compare. This is the hint for Step 5.5.

   *Exercise 4.5:* Find the first epoch (counting from 1) at which the TinyViT's test accuracy is 0.95 or more and store it in `best_epoch_vit` (write -1 if it never happens). Why it matters: it tells you how fast the model learns, not just where it ends.

```python
best_epoch_vit = None   # YOUR CODE: the first epoch (counting from 1) at which the TinyViT's test accuracy reaches 0.95 or more (use h_vit["test_acc"]); -1 if it never does
if best_epoch_vit is None:
    print("Fill in best_epoch_vit above, then run this cell again.")
else:
    ref = next((i + 1 for i, a in enumerate(h_vit["test_acc"]) if a >= 0.95), -1)
    check("first epoch with test accuracy of at least 0.95", best_epoch_vit == ref, "loop over h_vit['test_acc'] with enumerate; remember that epochs start at 1")
```

   *What it means.* With `enumerate` over `h_vit["test_acc"]` and `next(...)` you find the first index and add 1. We get epoch 2. The CNN is similar; a ViT does not need many epochs on this easy task.

6. **Step 4.6. Look inside: where does the CLS token look?.** A trained ViT can show what its CLS token looks at. In the last encoder layer the CLS token computes attention weights over all 17 tokens (itself and 16 patches); the weights for the 16 patches, averaged over the 4 heads, tell how much each patch contributes to the answer. We took the layer's attention module and ran it with `need_weights=True` (the method `cls_attention` of `TinyViT`), and draw the 16 weights as a 4 x 4 heat map under a picture of each shape.

```python
vit45.eval()
idx = [int(np.where(ys_te == k)[0][0]) for k in range(5)]            # one test picture of each shape
att = vit45.cls_attention(torch.as_tensor(Xs_te[idx]))               # (5, 16): CLS-to-patch attention, last layer
fig, ax = plt.subplots(2, 5, figsize=(10, 4.2))
for j, i in enumerate(idx):
    ax[0, j].imshow(Xs_te[i, 0], cmap="gray_r"); ax[0, j].set_title(SHAPE_NAMES[ys_te[i] ], fontsize=8); ax[0, j].axis("off")
    ax[1, j].imshow(att[j].reshape(4, 4).numpy(), cmap="viridis"); ax[1, j].set_title("CLS attention", fontsize=8); ax[1, j].axis("off")
plt.tight_layout(); plt.show()
print("each row of attention sums to", [round(float(a), 2) for a in att.sum(1)], "(the rest goes to the CLS token itself)")
print("attention to the 4 central patches (index 5, 6, 9, 10): %.2f of the attention on patches" % float((att[:, [5, 6, 9, 10] ].sum(1) / att.sum(1)).mean()))
RESULTS["p4_centre_attention"] = round(float((att[:, [5, 6, 9, 10] ].sum(1) / att.sum(1)).mean()), 2)
assert att.shape == (5, 16)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
each row of attention sums to [0.89, 0.82, 0.98, 0.95, 0.96] (the rest goes to the CLS token itself)
attention to the 4 central patches (index 5, 6, 9, 10): 0.33 of the attention on patches
```

   **What you should see.**

   You see five pictures (one of each shape) with a small heat map under each. The attention weights of the 16 patches add up to between 0.82 and 0.98 for the five pictures (the rest is the CLS token attending to itself). The four central patches (numbers 5, 6, 9 and 10) receive on average **0.33 of the attention on patches**; if the attention were spread evenly they would receive 4/16 = 0.25. So the model looks a little more at the centre, where the shapes are, but the maps are not sharp: with 400 pictures and 30 epochs the model has found a working rule, not a neat picture of the object. For the first picture (a horizontal bar) the strongest patch is at row 1, column 0 (Exercise 4.6).

   Tips and common mistakes:

   - Attention maps are a hint about what the model uses, not a proof. A patch with low attention can still matter through the values of the other tokens and the earlier layers.
   - Real ViTs trained on millions of pictures produce much sharper maps on objects; those are the pictures in the lesson.

   Try also (optional):

   - Draw the map for the first layer instead of the last (take `self.encoder.layers[0]`). Do the maps look different?
   - Average the attention over 50 pictures of each shape and compare the five average maps.

   *Exercise 4.6:* Find the patch the CLS token attends to most in the first of the five pictures and store its (row, column) in `top_patch`. Why it matters: it connects the numbers of the attention matrix to a place in the picture.

```python
top_patch = None   # YOUR CODE: for the first picture (j = 0) the (row, column) in the 4 x 4 grid of the patch the CLS token attends to most (use att[0].argmax() and divmod(..., 4))
if top_patch is None:
    print("Fill in top_patch above, then run this cell again.")
else:
    ref = divmod(int(att[0].argmax()), 4)
    check("the strongest patch", tuple(top_patch) == ref, "index = row * 4 + column, so divmod(index, 4) gives (row, column)")
```

   *What it means.* The flat index of the patch is `att[0].argmax()`; `divmod(index, 4)` converts it to (row, column) of the 4 x 4 grid. We get (1, 0).

### Part 5. ViT experiments: positions, pooling, patch size (25 min)

*Lesson sections: "Positions" and "Vision Transformer".* Part 3 proved on paper that a Transformer without position information cannot tell where anything is. Now you test it on a real task, and then you vary three design choices of the ViT: pooling, patch size and the amount of training data. The task is `make_left_right`: every picture has a plus sign (the target) and a filled square (a distractor), and the label says whether the plus sign is in the left half. The answer depends only on **where** the plus is. ==You will train a small CNN, a ViT with position embeddings and a ViT without them, show with shuffled patches why the last one is stuck at chance, and then compare CLS and mean pooling, three patch sizes and four data sizes, ending with fixed sine-cosine positions.== All numbers are for `SEED = 42`; a test set of 300 pictures has a standard error of about 0.03 near an accuracy of 0.5, so differences of two or three points are noise.

**Imagine first.** Picture the ViT as a jury that receives the 16 patches of the picture on separate cards, shuffled in a bag, with no numbers on the cards. The jury can describe every card (this one has a vertical line, that one is empty) but nobody knows which card came from the left. The question is: is the plus sign in the left half? Before you run any code, guess the jury's accuracy on 300 test pictures. Is it near 0.5 (pure guessing), near 0.75, or near 1.0? Write your guess in the cell below. You compare it with Step 5.1.

```python
# Imagine first: the accuracy of a ViT WITHOUT position embeddings on "is the plus sign in the left half?" (chance is 0.5)
my_guess_nopos = None   # write a number between 0.5 and 1.0
```

1. **Step 5.1. The left/right task: a CNN, a ViT with positions, a ViT without.** We draw 600 training and 300 test pictures (seeds 1 and 2), look at ten of them, and train three models with the same recipe (40 epochs, Adam, learning rate 0.001): a small CNN with 5,826 parameters, a TinyViT with d = 48, 2 layers and 4 heads with learned positions, and the same TinyViT with `use_pos=False`. Chance is 0.5, because the two classes are almost balanced (48 percent 'left' in the training set).

```python
Xlr_tr, ylr_tr = make_left_right(600, seed=1)     # 600 training pictures
Xlr_te, ylr_te = make_left_right(300, seed=2)     # 300 test pictures
show_images(Xlr_tr, titles=["left" if v else "right" for v in ylr_tr], n=10)
print("share of 'left' in the training set: %.2f   chance accuracy: 0.50" % ylr_tr.mean())
```

```python
EP51 = 4 if QUICK else 40
lr_models, lr_hist = {}, {}
builders = [("small CNN", lambda: SmallCNN(1, 16, 32, 2)),
            ("ViT with positions", lambda: TinyViT(d=48, heads=4, layers=2, n_classes=2)),
            ("ViT without positions", lambda: TinyViT(d=48, heads=4, layers=2, n_classes=2, use_pos=False))]
for name, build in builders:
    torch.manual_seed(SEED); mdl = build(); t0 = time.time()
    lr_hist[name] = train_classifier(mdl, Xlr_tr, ylr_tr, Xlr_te, ylr_te, epochs=EP51, lr=1e-3); lr_models[name] = mdl
    print("%-22s params %6d  train %.3f  test %.3f  (%.0f s)" % (name, count_params(mdl), lr_hist[name]["train_acc"][-1], lr_hist[name]["test_acc"][-1], time.time() - t0))
RESULTS["p5_left_right"] = {k: round(v["test_acc"][-1], 3) for k, v in lr_hist.items()}
```

```python
fig, ax = plt.subplots(figsize=(5, 3.2))
for name, h in lr_hist.items(): ax.plot(h["test_acc"], label=name)
ax.axhline(0.5, color="k", ls=":", label="chance"); ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("your Imagine-first guess:", my_guess_nopos)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
share of 'left' in the training set: 0.48   chance accuracy: 0.50
small CNN              params   5826  train 1.000  test 1.000  (3 s)
ViT with positions     params  58418  train 1.000  test 0.993  (6 s)
ViT without positions  params  57602  train 0.722  test 0.537  (6 s)
your Imagine-first guess: None
```

   **What you should see.**

   The three lines show: the **small CNN reaches 1.000 on both the training and the test set**; the **ViT with positions reaches 0.993 on the test set** (training 1.000); the **ViT without positions reaches only 0.537 on the test set**, although its training accuracy is 0.722: it memorised part of the noise of the 600 training pictures but found no rule that carries over, and a test accuracy of 0.537 on 300 pictures is within two standard errors of 0.5. Your Imagine-first guess: the answer is near 0.5. The learning curves show the same: the ViT with positions climbs to 0.99 while the one without wanders around 0.5.

   The reason is not that the no-position ViT is a bad model: on `make_shapes` (Step 4.5) the same architecture is perfect. It is that **this task cannot be solved from the bag of patches**: the plus sign and the square look the same in the left and in the right half. Step 5.2 proves it.

   Tips and common mistakes:

   - A test accuracy of 0.50 or 0.54 'without positions' is the expected result; if you see 0.8 something is wrong (the positions were not removed).
   - The ViT is about ten times larger than the CNN in parameters (58,418 against 5,826) and solves the task no better. The CNN has the idea of 'where' built in, because its filters slide over the picture and the flatten layer remembers the place; the ViT has to learn it from the position table.

   Try also (optional):

   - Train the no-position ViT for 100 epochs. Does the test accuracy move away from 0.5?
   - Use `d=32` for both ViTs. Does the ViT with positions still reach 0.99?

   *Exercise 5.1:* Store the difference in test accuracy (last epoch) between the ViT with positions and the ViT without positions in `gap_positions`. Why it matters: it is the number that sums up the experiment, the value of the positions on this task.

```python
gap_positions = None   # YOUR CODE: test accuracy of "ViT with positions" minus test accuracy of "ViT without positions" (last epoch, from lr_hist)
if gap_positions is None:
    print("Fill in gap_positions above, then run this cell again.")
else:
    ref = lr_hist["ViT with positions"]["test_acc"][-1] - lr_hist["ViT without positions"]["test_acc"][-1]
    check("the gap the positions make", abs(gap_positions - ref) < 1e-9, "take [-1] of the test_acc lists of the two entries of lr_hist and subtract")
```

   *What it means.* 0.993 minus 0.537 is 0.457 (the exact value depends on the unrounded accuracies). About 46 points of accuracy come from the position embeddings alone: same architecture, same data, same recipe.

2. **Step 5.2. Why the ViT without positions fails: shuffle the patches.** Here is the proof from Part 3, on the trained models. We define `shuffle_patches_img`, which cuts every test picture into its 16 patches and puts them back in a random order (a different order for every picture), and we feed the shuffled pictures to the ViT without positions and the ViT with positions. If the first model really sees a bag, its answers cannot change. For the shuffled pictures the label can flip (the plus sign may land in the other half), so we also report the accuracy against the original labels.

```python
def shuffle_patches_img(X, patch=4, seed=0):
    """Cut every picture of X (n, C, H, W) into patches and put the patches back in a random order (a different order per picture)."""
    r = np.random.RandomState(seed); n, c, H, W = X.shape; g = H // patch
    P = X.reshape(n, c, g, patch, g, patch).transpose(0, 2, 4, 1, 3, 5).reshape(n, g * g, c, patch, patch)
    P = np.stack([P[i, r.permutation(g * g)] for i in range(n)])
    return np.ascontiguousarray(P.reshape(n, g, g, c, patch, patch).transpose(0, 3, 1, 4, 2, 5).reshape(n, c, H, W))

Xlr_sh = shuffle_patches_img(Xlr_te, 4, seed=0)
show_images(np.concatenate([Xlr_te[:5], Xlr_sh[:5] ]), titles=["original"] * 5 + ["shuffled"] * 5, n=10)
```

```python
def predict(model, X):
    model.eval()
    with torch.no_grad(): return model(torch.as_tensor(X)).argmax(1).numpy()

rows = []
for name in ["ViT without positions", "ViT with positions"]:
    mdl = lr_models[name]; p0, p1 = predict(mdl, Xlr_te), predict(mdl, Xlr_sh)
    rows.append((name, float((p0 == p1).mean()), float((p0 == ylr_te).mean()), float((p1 == ylr_te).mean())))
print("%-24s %18s %14s %14s" % ("model", "agreement with own", "acc original", "acc shuffled"))
for r_ in rows: print("%-24s %18.3f %14.3f %14.3f" % r_)
RESULTS["p5_shuffle_agreement"] = {r_[0]: round(r_[1], 3) for r_ in rows}
assert rows[0][1] > 0.999, "a model without positions cannot see the order of the patches"
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
model                    agreement with own   acc original   acc shuffled
ViT without positions                 1.000          0.537          0.537
ViT with positions                    0.513          0.993          0.507
```

   **What you should see.**

   The table says: the **ViT without positions gives the same answer on 100 percent of the pictures** (agreement 1.000) and its accuracy is 0.537 before and after: shuffling changes nothing, because for this model the shuffled picture is the same bag. The **ViT with positions agrees with itself only on 51.3 percent** of the pictures, and its accuracy falls from 0.993 to 0.507, which is chance: shuffling destroys the information it used. The assert in the cell is the Part 3 theorem: a Transformer encoder with a CLS token and no positions is blind to the order of its tokens. The exercise adds the CNN: it agrees with itself on 51.3 percent of the shuffled pictures (the same share as the ViT with positions), because it also uses the layout.

   Tips and common mistakes:

   - The agreement of the ViT with positions is close to 0.5 and not near 0 because a random shuffle puts the plus sign in the correct half about half of the time.
   - If the agreement of the no-position model is 0.99 and not 1.000, the difference is a rounding effect of float numbers in a few pictures near a decision border; the exact value is not important.

   Try also (optional):

   - Shuffle only within the left half or only within the right half of the picture and re-test the ViT with positions. How much accuracy does it keep?

   *Exercise 5.2:* Compute the share of test pictures for which the small CNN gives the same answer on the original and on the shuffled pictures and store it in `agree_cnn`. Why it matters: it tells you that a CNN is not blind to layout, even though it has no position table.

```python
agree_cnn = None   # YOUR CODE: the share of test pictures for which the small CNN gives the same answer on Xlr_te and on the patch-shuffled Xlr_sh (use predict)
if agree_cnn is None:
    print("Fill in agree_cnn above, then run this cell again.")
else:
    ref = float((predict(lr_models["small CNN"], Xlr_te) == predict(lr_models["small CNN"], Xlr_sh)).mean())
    check("agreement of the CNN", abs(agree_cnn - ref) < 1e-9, "compare predict(model, Xlr_te) with predict(model, Xlr_sh) and take the mean")
```

   *What it means.* Use `predict(cnn, Xlr_te)` and `predict(cnn, Xlr_sh)`, compare them and take the mean. We get 0.513: the CNN's answers change on almost half of the pictures, as they should when the plus sign moves.

3. **Step 5.3. CLS token or mean pooling?.** The classification head reads either the CLS token (what you built) or the average of the 16 patch outputs ('mean pooling', which some ViTs prefer). We train a mean-pooling model on both tasks with the same recipe and put the four results in a table. The helper `fit_vit(task, ...)` trains a TinyViT with the recipes of Step 4.5 (`shapes`) and Step 5.1 (`lr`) and returns the model, the history and the seconds; the CLS entries reuse the models already trained.

```python
def fit_vit(task, pool="cls", patch=4, d=None, use_pos=True, pos_kind="learned", seed=SEED):
    """Train a TinyViT on 'shapes' (30 epochs, lr 3e-3, d 32) or 'lr' (left/right: 40 epochs, lr 1e-3, d 48). Returns (model, history, seconds)."""
    if task == "shapes": Xa, ya, Xb, yb, ep, lr, d0, k = Xs_tr, ys_tr, Xs_te, ys_te, EP, 3e-3, 32, 5
    else:                Xa, ya, Xb, yb, ep, lr, d0, k = Xlr_tr, ylr_tr, Xlr_te, ylr_te, EP51, 1e-3, 48, 2
    torch.manual_seed(seed); mdl = TinyViT(d=d or d0, heads=4, layers=2, n_classes=k, use_pos=use_pos, pool=pool, patch=patch, pos_kind=pos_kind)
    t0 = time.time(); h = train_classifier(mdl, Xa, ya, Xb, yb, epochs=ep, lr=lr)
    return mdl, h, time.time() - t0

pool_tab = {("cls", "shapes"): h_vit["test_acc"][-1], ("cls", "lr"): lr_hist["ViT with positions"]["test_acc"][-1]}   # already trained
for task in ["shapes", "lr"]:
    pool_tab[("mean", task)] = fit_vit(task, pool="mean")[1]["test_acc"][-1]
print("%-12s %12s %18s" % ("pooling", "make_shapes", "left/right"))
for pool in ["cls", "mean"]: print("%-12s %12.3f %18.3f" % (pool, pool_tab[(pool, "shapes")], pool_tab[(pool, "lr")]))
RESULTS["p5_pooling"] = {"%s/%s" % k: round(v, 3) for k, v in pool_tab.items()}
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
pooling       make_shapes         left/right
cls                 1.000              0.993
mean                1.000              0.993
```

   **What you should see.**

   The table: on `make_shapes` both poolings reach **1.000**; on left/right the CLS model reaches **0.993** and the mean-pooling model **0.993**. So there is no difference between the two on these tasks: both are tied (the exercise answer is `tie`). That is the honest finding. Large ViTs trained on millions of pictures show small differences between the two, and the choice there is made by experiments on the big data set, not by a rule.

   Tips and common mistakes:

   - With positions added, mean pooling still sees where each patch is, because the position information is inside each patch token.
   - Do not conclude 'pooling never matters': on harder tasks and larger models the gap can be a point or two in either direction.

   Try also (optional):

   - Repeat the comparison with `d=32` on the left/right task. Is the tie still there?

   *Exercise 5.3:* Decide which pooling is better on the left/right task according to `pool_tab` and store `"cls"`, `"mean"` or `"tie"` in `better_pool_lr`. Why it matters: it forces you to read the table instead of assuming that one of them must win.

```python
better_pool_lr = None   # YOUR CODE: "cls" or "mean", whichever has the higher left/right accuracy in pool_tab (write "tie" if they are equal)
if better_pool_lr is None:
    print("Fill in better_pool_lr above, then run this cell again.")
else:
    a, b = pool_tab[("cls", "lr")], pool_tab[("mean", "lr")]
    ref = "tie" if abs(a - b) < 1e-9 else ("cls" if a > b else "mean")
    check("the better pooling on left/right", better_pool_lr == ref, "compare pool_tab[('cls', 'lr')] with pool_tab[('mean', 'lr')]")
```

   *What it means.* Both entries are 0.993, so the answer is `tie`. With a different seed one of them may win by 0.003: that is a difference of one picture in 300.

4. **Step 5.4. Patch size: 2, 4 or 8?.** Patch size is the main design choice of a ViT: small patches give long sequences and fine detail, large patches give short sequences and less detail. For the 16 x 16 pictures we compare patch 2 (64 tokens), patch 4 (16 tokens) and patch 8 (4 tokens), on both tasks. The table shows the number of tokens, the parameters (the patch embedding and the position table change with p), the seconds per epoch on `make_shapes`, and the two test accuracies; the plot shows accuracy against tokens.

```python
sizes_p = [2, 4, 8]
patch_rows = []
for p in sizes_p:
    row = dict(p=p, tokens=(16 // p) ** 2)
    for task in ["shapes", "lr"]:
        mdl, h, secs = fit_vit(task, patch=p)
        row[task] = h["test_acc"][-1]; row["sec_" + task] = secs / len(h["loss"]); row["params_" + task] = count_params(mdl)
    patch_rows.append(row)
print("%-6s %7s %14s %15s %12s %13s" % ("patch", "tokens", "params(shapes)", "s/epoch(shapes)", "acc shapes", "acc left/right"))
for r_ in patch_rows: print("%-6d %7d %14d %15.2f %12.3f %13.3f" % (r_["p"], r_["tokens"], r_["params_shapes"], r_["sec_shapes"], r_["shapes"], r_["lr"]))
RESULTS["p5_patch_size"] = {r_["p"]: (r_["tokens"], round(r_["shapes"], 3), round(r_["lr"], 3)) for r_ in patch_rows}
```

```python
fig, ax = plt.subplots(figsize=(5, 3.2))
toks = [r_["tokens"] for r_ in patch_rows]
ax.plot(toks, [r_["shapes"] for r_ in patch_rows], "o-", label="make_shapes"); ax.plot(toks, [r_["lr"] for r_ in patch_rows], "s-", label="left/right")
ax.set_xscale("log", base=2); ax.set_xlabel("tokens per picture (patch 8, 4, 2 from left to right)"); ax.set_ylabel("test accuracy"); ax.legend()
for r_ in patch_rows: ax.annotate("p=%d" % r_["p"], (r_["tokens"], r_["lr"]), fontsize=8, xytext=(3, -10), textcoords="offset points")
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
patch   tokens params(shapes) s/epoch(shapes)   acc shapes acc left/right
2           64          27909            0.20        1.000         1.000
4           16          26757            0.08        1.000         0.993
8            4          27909            0.06        1.000         0.860
```

   **What you should see.**

   The table: patch 2 has **64 tokens**, 27,909 parameters, about 0.20 seconds per epoch, accuracy 1.000 on `make_shapes` and **1.000** on left/right. Patch 4 has 16 tokens, 26,757 parameters, about 0.08 seconds per epoch, 1.000 and **0.993**. Patch 8 has only **4 tokens**, 27,909 parameters, about 0.06 seconds per epoch, 1.000 and **0.860**. The 5-shape task does not care (every patch size solves it), but the left/right task drops to 0.86 with 4 tokens: with patches of 8 x 8 pixels a plus sign of 5 x 5 pixels can lie across two patches, a patch holds the plus sign and the square together, and the model's only way to find the plus is through a few big, mixed patches. The number of attention scores per head grows with the square of the sequence: 65 squared against 17 squared is 14.6 times more for patch 2 than for patch 4 (Exercise 5.4), while the time per epoch grew about 2.5 times, because the small model spends part of its time outside attention.

   The trade-off in one sentence: **smaller patches keep more detail but cost much more; larger patches are cheap but can blur what you need.** For real 224 x 224 pictures the lesson's numbers are 49 tokens for patch 32, 196 for patch 16 and 784 for patch 8.

   Tips and common mistakes:

   - The parameters hardly change with the patch size because the encoder layers (the bulk of the model) do not depend on it; only the embedding (p x p x d) and the position table ((N + 1) x d) change.
   - The seconds per epoch depend on your machine. Compare the ratios, not the absolute values.

   Try also (optional):

   - Try patch 16 (one token) on the left/right task. What accuracy would you expect? (A single patch is the whole picture: the model is a plain MLP in disguise.)

   *Exercise 5.4:* Compute how many times more attention scores per head the patch-2 model computes than the patch-4 model (sequence length: tokens plus 1 for the CLS token; scores: the length squared) and store it in `cost_ratio`. Why it matters: it is the reason why people do not use 1 x 1 patches on real pictures.

```python
cost_ratio = None   # YOUR CODE: how many times more attention scores per head does the patch-2 model compute than the patch-4 model? (sequence length = tokens + 1 for the CLS token; scores = length squared)
if cost_ratio is None:
    print("Fill in cost_ratio above, then run this cell again.")
else:
    check("ratio of attention scores", abs(cost_ratio - (65 ** 2) / (17 ** 2)) < 1e-6, "(64 + 1) squared divided by (16 + 1) squared")
```

   *What it means.* (64 + 1) squared is 4,225 and (16 + 1) squared is 289; the ratio is 14.62.

5. **Step 5.5. Data hunger: how many pictures does each model need?.** Lesson: a ViT has less built-in knowledge about pictures than a CNN, so it needs more data. Test it. We make a harder version of `make_shapes` (the shape may sit up to 2 pixels from the centre, noise 0.3), keep 300 test pictures, and train the TinyViT and the CNN on the first 50, 100, 200 and 400 training pictures. To be fair each model takes about the same number of gradient steps, so the small training sets get more epochs.

```python
# a harder version of make_shapes: the shape may sit up to 2 pixels away from the centre and the noise is larger
Xh_tr, yh_tr = make_shapes(400, shift=2, noise=0.3, seed=1)
Xh_te, yh_te = make_shapes(300, shift=2, noise=0.3, seed=2)
sizes_n = [50, 100, 200, 400]
curve = {"TinyViT": [], "SmallCNN": []}
for n in sizes_n:
    ep = 3 if QUICK else int(round(30 * math.ceil(400 / 64) / math.ceil(n / 64)))     # the same number of gradient steps for every n
    for name, build in [("TinyViT", lambda: TinyViT()), ("SmallCNN", lambda: SmallCNN())]:
        torch.manual_seed(SEED); mdl = build()
        h = train_classifier(mdl, Xh_tr[:n], yh_tr[:n], Xh_te, yh_te, epochs=ep, lr=3e-3)
        curve[name].append(h["test_acc"][-1])
print("%-9s" % "pictures", *["%8d" % n for n in sizes_n])
for name, v in curve.items(): print("%-9s" % name, *["%8.3f" % a for a in v])
RESULTS["p5_data_hunger"] = {k: [round(a, 3) for a in v] for k, v in curve.items()}
```

```python
fig, ax = plt.subplots(figsize=(5, 3.2))
for name, v in curve.items(): ax.plot(sizes_n, v, "o-", label=name)
ax.set_xscale("log"); ax.set_xlabel("training pictures"); ax.set_ylabel("test accuracy"); ax.legend(); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
pictures        50      100      200      400
TinyViT      0.430    0.623    0.690    0.880
SmallCNN     0.887    0.957    0.980    0.993
```

   **What you should see.**

   The table: with 50 pictures the **ViT reaches 0.430 and the CNN 0.887**; with 100 pictures 0.623 against 0.957; with 200 0.690 against 0.980; with 400 pictures **0.880 against 0.993**. The CNN is better at every size, and the gap is 0.457 at 50 pictures and 0.113 at 400: the CNN needs a few dozen pictures to be almost perfect, the ViT still climbs at 400. This is the 'lack of inductive bias' of the lesson: a convolution already assumes that the same pattern may appear anywhere in the picture (translation) and that nearby pixels belong together, so the CNN does not need to learn this from data. A ViT must find it by itself, and the shifted shapes make this expensive.

   On the easy, centred data of Step 4.5 you saw no difference. **The comparison depends on the task and the data size.** Large pretrained ViTs beat CNNs because they are trained on hundreds of millions of pictures; a ViT trained from scratch on a few hundred pictures should not be expected to.

   Tips and common mistakes:

   - Each number is a single run. With a different seed the ViT may move by a few points, but the ordering of the two curves is stable.
   - More epochs for the small sets help the ViT only a little; the problem is the data, not the optimiser.

   Try also (optional):

   - Add 800 and 1,600 training pictures (make a larger set with `make_shapes`). Does the ViT catch up?

   *Exercise 5.5:* Store the difference between the CNN's and the ViT's accuracy at 50 training pictures in `gap_at_50`. Why it matters: it quantifies the data hunger in one number.

```python
gap_at_50 = None   # YOUR CODE: SmallCNN accuracy minus TinyViT accuracy with 50 training pictures (index 0 of the lists in curve)
if gap_at_50 is None:
    print("Fill in gap_at_50 above, then run this cell again.")
else:
    check("the gap at 50 pictures", abs(gap_at_50 - (curve["SmallCNN"][0] - curve["TinyViT"][0])) < 1e-9, "curve['SmallCNN'][0] - curve['TinyViT'][0]")
```

   *What it means.* `curve["SmallCNN"][0]` is 0.887 and `curve["TinyViT"][0]` is 0.430: the gap is 0.457.

6. **Step 5.6. Where else does 'where' matter? Fixed sine-cosine positions instead of a learned table.** So far the position table was learned. A popular alternative is a fixed table made from the sine-cosine formula of Week 8, extended to two dimensions: half of the d numbers encode the row, half the column. It has no parameters, and it can be computed for any grid size. The cell defines `sincos_2d(grid, d)`, checks the table for d = 8 on a 4 x 4 grid, and trains a ViT with `pos_kind="sincos"` on the left/right task. Then you write down three answers about where else 'where' matters.

```python
def sincos_2d(grid, d):
    """Fixed 2D sine-cosine positions for a grid x grid patch grid: (grid*grid, d). Half of the d numbers encode the row, half the column."""
    def one_d(pos, dim):                                            # the Week 8 formula for a vector of positions
        i = np.arange(dim // 2); ang = pos[:, None] / (10000 ** (2 * i / dim))[None, :]
        out = np.zeros((len(pos), dim)); out[:, 0::2] = np.sin(ang); out[:, 1::2] = np.cos(ang)
        return out
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    return np.concatenate([one_d(rows, d // 2), one_d(cols, d // 2)], 1)

P = sincos_2d(4, 8)                                         # d = 8 on a 4 x 4 grid: shape (16, 8)
print("shape:", P.shape, "  all 16 position vectors different:", len({tuple(np.round(r_, 6)) for r_ in P}) == 16)
mdl_sc, h_sc, _ = fit_vit("lr", pos_kind="sincos")
learned = lr_hist["ViT with positions"]["test_acc"][-1]
print("left/right accuracy: learned positions %.3f, fixed sine-cosine positions %.3f, no positions %.3f" % (learned, h_sc["test_acc"][-1], lr_hist["ViT without positions"]["test_acc"][-1]))
RESULTS["p5_sincos"] = round(h_sc["test_acc"][-1], 3)
assert P.shape == (16, 8)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
shape: (16, 8)   all 16 position vectors different: True
left/right accuracy: learned positions 0.993, fixed sine-cosine positions 1.000, no positions 0.537
```

   **What you should see.**

   The table has shape (16, 8) and all 16 position vectors are different. The ViT with fixed sine-cosine positions reaches **1.000** on the left/right task, against 0.993 with learned positions and 0.537 with no positions. So a position table that is not learned at all is enough here: **what the model needs is any code that makes the places different, not necessarily a trained one.** The difference of 0.007 between 1.000 and 0.993 is two pictures in 300, noise.

   Your three written answers go below the cell: for text a position is a single number along a line (a 1D code); for a map or satellite image the position has two coordinates and a real distance, so a 2D code or the geographic coordinates are natural; for an X-ray the same patch of grey can mean a healthy lung in one place and a problem at another, and scans of a different size need a code that can be computed for any grid (sine-cosine) or interpolated (Part 3.6).

   Tips and common mistakes:

   - The sine-cosine table is added to the tokens exactly like the learned one; it is stored as a buffer so that the optimiser leaves it alone.
   - Relative position codes (what matters is the offset between two patches, not the absolute place) are a third family used in many modern models.

   Try also (optional):

   - Count the trainable parameters of the sine-cosine model and of the learned one. Which is smaller and by how much? (The sine-cosine model is smaller by the 17 x 48 = 816 numbers of the table.)

   *Exercise 5.6:* Write three answers (at least 12 words each) in `answers_56`: what position means in text, in a map or satellite image, and in an X-ray, and what you would use instead of a 1D learned table. Why it matters: choosing a position code is a design decision about the data, not a detail.

```python
answers_56 = {
    "text":  "",   # YOUR ANSWER: in a line of text, which direction matters and what position would you give each word or patch?
    "maps":  "",   # YOUR ANSWER: for a map or satellite image, what is different about 'where' (think of coordinates and of a distance)?
    "xray":  "",   # YOUR ANSWER: for an X-ray, why can 'the same patch in another place' mean something else, and what would you use instead of a 1D learned table?
}
empty = [k for k, v in answers_56.items() if len(v.split()) < 12]
if empty:
    print("Write at least 12 words for:", empty)
else:
    check("three answers written", True)
```

   *What it means.* A good answer names the number of dimensions (1 for text, 2 for maps and scans), says whether distances matter, and says what happens if the test picture has a different size. The check only counts words; compare with the sample answers.

### Part 6. Contrastive learning and CLIP (30 min)

*Lesson section: "CLIP".* CLIP learns from pairs of a picture and its caption. It has two encoders, one for pictures and one for text, that map their inputs into the same space, and a loss that pulls the vectors of matching pairs together and pushes the other pairs in the batch apart. After training, a class name written as a sentence can be compared with a picture by the cosine of their vectors: zero-shot classification without a classifier trained on those classes. ==You will compute the contrastive loss by hand, see what the temperature does to it, build and train a miniature CLIP on coloured shapes with two combinations held out, test it zero-shot on combinations it never saw, and then use the real CLIP (openai/clip-vit-base-patch32) on pictures drawn with PIL, including its failures.== The toy model trains in a few seconds; the real CLIP is loaded once (about 600 MB; on the Spark it comes from the offline cache, so ask your instructor if the download fails). All numbers are for `SEED = 42`; the real CLIP outputs were measured with the library versions of the author and can differ in the last digits on your machine.

**Imagine first.** A child has seen red squares, red circles, green squares, green circles, blue squares, blue crosses and a few other combinations, but never a green cross and never a blue circle. Then you ask: here is a picture of a green cross, which of these nine captions fits? A child who knows what 'green' and 'cross' mean separately would get it right. Does a model trained only on pairs of pictures and two-word captions do the same? Before you run any code, guess its accuracy on the two held-out combinations among the 9 captions (chance is 0.11). Write your guess in the cell below. You compare it with Step 6.4.

```python
# Imagine first: a model trained only on image-caption pairs of 7 colour-shape combinations; can it name the 2 combinations it never saw?
my_guess_unseen = None   # write a number between 0 and 1 (chance among the 9 captions is 0.11)
```

1. **Step 6.1. The contrastive loss by hand.** Take three pictures and their three captions, each as a vector of 2 numbers (real CLIP vectors have 512). Image i belongs to text i. The contrastive loss works in four moves: (1) normalise the vectors so that a dot product is a cosine, (2) compute the 3 x 3 matrix of cosines between every picture and every text, (3) multiply by a scale (the temperature trick: scale 10 here) to get the logits, and (4) read each row as a softmax over the texts for one picture and each column as a softmax over the pictures for one text, and take minus the log of the probability of the right partner. The loss is the mean over the rows and over the columns. The third picture is a hard example: its own text is farther away than text 1, so the loss has something to punish.

```python
img = np.array([ [1.0, 0.2], [0.3, 1.0], [0.9, 0.5] ])      # three image vectors (2 numbers each, to keep it small)
txt = np.array([ [0.9, 0.1], [0.2, 0.9], [0.3, 1.0] ])      # their three captions: image i belongs to text i
unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
cos = unit(img) @ unit(txt).T                               # cosine similarity of every image with every text
scale = 10.0
logits = scale * cos
print("cosine matrix (rows images, columns texts):\n", cos.round(3))

def softmax_rows(z):
    e = np.exp(z - z.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
p_i2t = softmax_rows(logits)           # row i: image i against the three texts
p_t2i = softmax_rows(logits.T)         # row j: text j against the three images
print("image -> text probabilities:\n", p_i2t.round(3))
print("text -> image probabilities:\n", p_t2i.round(3))
loss_i2t = -np.log(np.diag(p_i2t)).mean()
loss_t2i = -np.log(np.diag(p_t2i)).mean()
loss = (loss_i2t + loss_t2i) / 2
print("loss image->text %.4f, text->image %.4f, average %.4f" % (loss_i2t, loss_t2i, loss))
```

```python
lt, labels = torch.tensor(logits), torch.arange(3)
ref = (F.cross_entropy(lt, labels) + F.cross_entropy(lt.T, labels)) / 2
print("F.cross_entropy version: %.4f" % float(ref))
assert abs(loss - float(ref)) < 1e-9
RESULTS["p6_loss_by_hand"] = round(float(loss), 4)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
cosine matrix (rows images, columns texts):
 [ [0.996 0.404 0.47 ]
 [0.391 0.997 1.   ]
 [0.922 0.664 0.716] ]
image -> text probabilities:
 [ [0.992 0.003 0.005]
 [0.001 0.493 0.506]
 [0.832 0.063 0.106] ]
text -> image probabilities:
 [ [0.675 0.002 0.323]
 [0.003 0.963 0.034]
 [0.005 0.94  0.055] ]
loss image->text 0.9869, text->image 1.1093, average 1.0481
F.cross_entropy version: 1.0481
```

   **What you should see.**

   The cosine matrix has 0.996, 0.997 and 0.716 on the diagonal. Picture 2 is almost as close to text 3 (1.000) as to its own text 2 (0.997): the two texts are almost the same vector. Picture 3 is closer to text 1 (0.922) than to its own text 3 (0.716). The image-to-text probabilities are 0.992, 0.493 and 0.106 for the right partners (picture 3 gives 0.832 to the wrong text); the text-to-image probabilities are 0.675, 0.963 and 0.055. The loss is **0.9869 in the image-to-text direction, 1.1093 in the text-to-image direction and 1.0481 on average**, and the assert passes: your NumPy loss equals `F.cross_entropy` applied to the logits and to their transpose. A model that knows nothing would score ln(3) = 1.099; this tiny example is only a little better because two of its three pairs are confusing.

   Tips and common mistakes:

   - Subtracting the largest value of each row before `np.exp` does not change the softmax but avoids overflow.
   - The two directions are not the same loss: a row asks 'which text fits this picture', a column asks 'which picture fits this text'. CLIP averages them so that both encoders learn.

   Try also (optional):

   - Make picture 3 closer to its text (change `[0.9, 0.5]` to `[0.4, 1.0]`) and watch the loss go down.

   *Exercise 6.1:* Write `clip_loss_np(logits)`: the symmetric contrastive loss of any square matrix of logits, as the average of the two directions. Why it matters: this function is the training objective of CLIP in a few lines, and you reuse the idea in the training loop of Step 6.3.

```python
def clip_loss_np(logits):
    """The symmetric contrastive loss of a square matrix of logits (row i and column i belong together)."""
    return None   # YOUR CODE: softmax over each row, softmax over each column (the rows of logits.T), minus the log of the diagonal, average of the two directions
out = clip_loss_np(np.array([ [2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0] ]))
if out is None:
    print("Fill in clip_loss_np above, then run this cell again.")
else:
    t = torch.tensor([ [2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0] ], dtype=torch.float64)
    ref = float((F.cross_entropy(t, torch.arange(3)) + F.cross_entropy(t.T, torch.arange(3))) / 2)
    check("agrees with F.cross_entropy", abs(out - ref) < 1e-9, "average the two directions; the second direction uses logits.T")
```

   *What it means.* Each direction is the mean of minus the log of the diagonal of a row softmax; the second direction is the first one applied to `logits.T`. The check compares your function with PyTorch on a new matrix.

2. **Step 6.2. Temperature: the scale decides how sharp the softmax is.** CLIP multiplies the cosines by a scale (the inverse of a temperature) before the softmax. The scale decides how sharp the distribution is. We use the matrix of Step 6.1 with a wrong pair in it, and compute the mean probability of the right partner and the loss for scales 1, 5, 10, 30 and 100.

```python
def clip_loss_ref(logits):
    t = torch.as_tensor(logits); n = len(t)                  # PyTorch's version of the same loss, to check against
    return float((F.cross_entropy(t, torch.arange(n)) + F.cross_entropy(t.T, torch.arange(n))) / 2)

scales = [1, 5, 10, 30, 100]
rows = []
for s in scales:
    z = s * cos
    rows.append((s, float(np.diag(softmax_rows(z)).mean()), clip_loss_ref(z)))
print("%6s %22s %8s" % ("scale", "mean diagonal prob.", "loss"))
for s, pdiag, l in rows: print("%6d %22.3f %8.3f" % (s, pdiag, l))
print("a uniform guess among 3 gives loss ln(3) = %.3f" % math.log(3))
```

```python
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].semilogx([r_[0] for r_ in rows], [r_[1] for r_ in rows], "o-"); ax[0].set_title("mean probability of the right partner"); ax[0].set_xlabel("scale")
ax[1].semilogx([r_[0] for r_ in rows], [r_[2] for r_ in rows], "o-r"); ax[1].set_title("contrastive loss"); ax[1].set_xlabel("scale")
plt.tight_layout(); plt.show()
RESULTS["p6_scale_losses"] = {r_[0]: round(r_[2], 3) for r_ in rows}
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
 scale    mean diagonal prob.     loss
     1                  0.391    0.952
     5                  0.531    0.803
    10                  0.530    1.048
    30                  0.494    2.589
   100                  0.478    8.302
a uniform guess among 3 gives loss ln(3) = 1.099
```

   **What you should see.**

   The table: at scale 1 the mean probability of the right partner is 0.391 and the loss 0.952; at 5 it is 0.531 and 0.803 (the lowest loss); at 10 it is 0.530 and 1.048; at 30 it is 0.494 and **2.589**; at 100 it is 0.478 and **8.302**. A uniform guess would score 1.099. The curve is a U: with a **scale that is too low** (1) every probability is soft and the loss stays near the uniform guess; with a **scale that is too high** the right pairs get near 1 but the one wrong pair (picture 3) is punished hard: the probability of its right partner is almost 0, and minus its log is large (8.3 on average at scale 100). Real CLIP learns its scale during training and ends at 100 (Step 6.5 prints it); our toy model starts at 10 and learns 12.7 (Step 6.3).

   This is a **brittleness** argument for a high scale: it only pays when the pairs are mostly right. In a batch with mislabelled or ambiguous pairs (a caption that fits two pictures), a high scale amplifies their loss.

   Tips and common mistakes:

   - The best scale here (5) depends on the example; it is not a general rule. The learnable `log_scale` finds a good value for the data automatically.
   - Why the log of the scale is learned, not the scale: the log is free to be any real number, the scale stays positive.

   Try also (optional):

   - Fix the third picture (see Step 6.1's Try also) and run the table again: does the best scale change?

   *Exercise 6.2:* Store the scale with the lowest loss in `best_scale`. Why it matters: it makes you read the whole table instead of assuming that bigger is better.

```python
best_scale = None   # YOUR CODE: the scale from `scales` with the lowest loss in `rows`
if best_scale is None:
    print("Fill in best_scale above, then run this cell again.")
else:
    ref = min(rows, key=lambda r_: r_[2])[0]
    check("the scale with the lowest loss", best_scale == ref, "look at the third number of each tuple in rows")
```

   *What it means.* The tuple with the smallest third value is (5, 0.531, 0.803). We get 5.

3. **Step 6.3. A miniature CLIP from scratch, with two combinations held out.** Now we build and train a miniature CLIP. The image encoder is a small CNN that maps a 16 x 16 colour picture to 16 numbers. The text encoder embeds the two words of the caption (a colour word and a shape word, 6 words in total), averages the two embeddings and applies a linear layer to get 16 numbers. Both outputs are normalised to length 1, and a learnable `log_scale` (starting at ln 10) sets the temperature. We train with the symmetric contrastive loss on batches of 32 pairs for 500 steps (Adam, learning rate 0.003). The crucial design choice: **two combinations are removed from the training pictures**, 'green cross' and 'blue circle'. The model sees every colour and every shape, but never these pairs. One more detail: with only seven different captions, a batch of 32 contains the same caption several times, and the plain loss would punish the model for ranking a picture of a red circle next to a different red circle's caption. The helper `clip_loss(logits, ids)` hides these same-caption pairs in the softmax.

```python
Xc, col, shp, caps = make_colored_shapes(1200, seed=1)         # 1,200 pictures with captions such as "red circle"
Xc_te, col_te, shp_te, caps_te = make_colored_shapes(450, seed=2)
print(Counter(caps).most_common(3), "...", len(set(caps)), "different captions")
show_images(Xc, titles=caps, n=10)
HELD = [("green", "cross"), ("blue", "circle")]                  # the two combinations the model must never see in training
held_ids = {COLOURS.index(c) * 3 + SHAPES3.index(s) for c, s in HELD}   # caption index = colour * 3 + shape
combo = col * 3 + shp
print("held-out caption indices:", sorted(held_ids))
```

   *Exercise 6.3:* Build the boolean mask `keep` that is True for the pictures whose combination is not held out (use `np.isin(combo, list(held_ids))` and `~`). Why it matters: a held-out test is only fair if the held-out pictures were really never used for training; this one line decides it.

```python
keep = None   # YOUR CODE: a boolean array over the 1,200 pictures that is True for pictures whose combination is NOT held out (use np.isin(combo, list(held_ids)) and ~)
if keep is None:
    print("Fill in keep above, then run this cell again.")
else:
    check("number of training pictures", int(np.sum(keep)) == int(np.sum(~np.isin(combo, list(held_ids)))), "keep = ~np.isin(combo, list(held_ids))")
    check("no held-out picture is kept", not np.isin(combo[keep], list(held_ids)).any(), "the held-out combinations must be removed")
```

   *What it means.* `np.isin(combo, list(held_ids))` is True for pictures of the held-out combinations; the `~` flips it. The check counts the training pictures (926) and tests that none of the held-out ones are in it. The next cell uses its own `train_mask`, so it works even if yours is unfinished.

```python
train_mask = ~np.isin(combo, list(held_ids))                     # (we use our own copy from here on, so the next cells run even if keep is unfinished)
print("training pictures: %d, held-out pictures set aside: %d" % (train_mask.sum(), (~train_mask).sum()))

class ToyCLIP(nn.Module):
    """Image encoder: small CNN -> 16 numbers. Text encoder: the average of the embeddings of the colour word and the shape word -> linear -> 16 numbers."""
    def __init__(self, dim=16):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(32 * 4 * 4, dim))
        self.words = nn.Embedding(6, 16)                          # 3 colour words and 3 shape words
        self.txt_proj = nn.Linear(16, dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))   # the learnable temperature
    def encode_image(self, x): return F.normalize(self.cnn(x), dim=-1)
    def encode_text(self, c, s): return F.normalize(self.txt_proj((self.words(c) + self.words(3 + s)) / 2), dim=-1)

def clip_loss(logits, ids=None):
    """Symmetric contrastive loss. ids = caption id of each pair: two pairs with the same caption are not 'wrong' partners, so we hide them."""
    n = len(logits)
    if ids is not None:
        same = (ids[:, None] == ids[None, :]) & ~torch.eye(n, dtype=torch.bool)
        logits = logits.masked_fill(same, -1e4)
    labels = torch.arange(n)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

STEPS63 = 60 if QUICK else 500
torch.manual_seed(SEED); toy = ToyCLIP(); opt = torch.optim.Adam(toy.parameters(), lr=3e-3)
rg = np.random.RandomState(0); pool_idx = np.where(train_mask)[0]; losses63 = []
for step in range(STEPS63):
    b = rg.choice(pool_idx, 32, replace=False)                     # a batch of 32 image-caption pairs
    zi = toy.encode_image(torch.tensor(Xc[b])); zt = toy.encode_text(torch.tensor(col[b]), torch.tensor(shp[b]))
    loss = clip_loss(toy.log_scale.exp().clamp(max=100) * zi @ zt.T, ids=torch.tensor(combo[b]))
    opt.zero_grad(); loss.backward(); opt.step(); losses63.append(float(loss))
    if step % 100 == 0 or step == STEPS63 - 1: print("step %3d  loss %.3f  scale %.1f" % (step, losses63[-1], float(toy.log_scale.exp())))
RESULTS["p6_toy_loss"] = (round(losses63[0], 3), round(losses63[-1], 3))
```

```python
plt.figure(figsize=(5, 3)); plt.plot(losses63); plt.xlabel("step"); plt.ylabel("contrastive loss"); plt.title("Toy CLIP training"); plt.tight_layout(); plt.show()
assert losses63[-1] < losses63[0], "the loss should go down"
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
[('red cross', 148), ('blue square', 143), ('blue circle', 140)] ... 9 different captions
held-out caption indices: [5, 7]
training pictures: 926, held-out pictures set aside: 274
step   0  loss 3.924  scale 10.0
step 100  loss 0.012  scale 11.3
step 200  loss 0.004  scale 12.0
step 300  loss 0.002  scale 12.3
step 400  loss 0.001  scale 12.5
step 499  loss 0.000  scale 12.7
```

   **What you should see.**

   The data has 1,200 pictures and 9 caption classes with about 130 pictures each (the three most common are red cross 148, blue square 143, blue circle 140). The held-out caption indices are 5 (green cross) and 7 (blue circle), which leaves **926 training pictures** and 274 held out. The training prints a loss of **3.924 at step 0** (the random model, with 32 pairs of which many look alike) falling to **0.012 at step 100** and **0.000 at step 499**, while the learned scale rises from 10.0 to 12.7. The loss curve falls fast in the first 100 steps and then flattens: for seven captions on clean pictures the model has nothing left to learn. The assert checks that the loss went down.

   Tips and common mistakes:

   - A loss of exactly 0.000 means that the right partner gets a probability of nearly 1 for every pair of the batch; on the clean, nearly noise-free pictures that is plausible, and it does not mean the model is good on pictures it never saw (Step 6.4).
   - `clip_loss(..., ids=None)` is the plain loss. Real CLIP trains on millions of different captions, so repeated captions in a batch are rare, and it does not need the mask.

   Try also (optional):

   - Train for 100 steps only. How much of the final accuracy of Step 6.4 do you already have?
   - Remove the mask (`ids=None`) and compare the final loss. It cannot fall as low, because identical captions are counted as errors.

4. **Step 6.4. Zero-shot with the toy CLIP: seen and unseen combinations.** Zero-shot classification with the toy model: we encode all nine captions (the two unseen ones too), encode the 450 test pictures, and give each picture the caption with the highest cosine. We report the accuracy separately for pictures of the seven seen combinations and the two unseen ones, and what the model says for each unseen caption.

```python
all_c, all_s = np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)       # the nine captions, caption index = colour * 3 + shape
names9 = ["%s %s" % (COLOURS[c], SHAPES3[s]) for c, s in zip(all_c, all_s)]
toy.eval()
with torch.no_grad():
    zt9 = toy.encode_text(torch.tensor(all_c), torch.tensor(all_s))                # 9 text vectors, unseen captions included
    zi_te = toy.encode_image(torch.tensor(Xc_te))
te_pred = (zi_te @ zt9.T).argmax(1).numpy()                                        # the caption with the highest cosine wins
te_true = col_te * 3 + shp_te
unseen = np.isin(te_true, list(held_ids))
acc_seen, acc_unseen = float((te_pred == te_true)[~unseen].mean()), float((te_pred == te_true)[unseen].mean())
print("test pictures: %d seen-combination, %d unseen-combination" % ((~unseen).sum(), unseen.sum()))
print("accuracy on seen combinations  : %.3f" % acc_seen)
print("accuracy on UNSEEN combinations: %.3f   (chance among 9 captions: 0.111)" % acc_unseen)
print("colour right on unseen: %.3f   shape right on unseen: %.3f" % ((all_c[te_pred] == col_te)[unseen].mean(), (all_s[te_pred] == shp_te)[unseen].mean()))
RESULTS["p6_toy_zero_shot"] = (round(acc_seen, 3), round(acc_unseen, 3))
print("your Imagine-first guess:", my_guess_unseen)
```

```python
for t in sorted(held_ids):
    m = te_true == t
    print("true caption %-12s (%3d pictures) -> predicted:" % (names9[t], m.sum()), {names9[k]: int(v) for k, v in Counter(te_pred[m]).most_common()})
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
test pictures: 351 seen-combination, 99 unseen-combination
accuracy on seen combinations  : 1.000
accuracy on UNSEEN combinations: 0.394   (chance among 9 captions: 0.111)
colour right on unseen: 1.000   shape right on unseen: 0.394
your Imagine-first guess: None
true caption green cross  ( 51 pictures) -> predicted: {'green cross': 39, 'green circle': 12}
true caption blue circle  ( 48 pictures) -> predicted: {'blue square': 48}
```

   **What you should see.**

   On the 351 test pictures of **seen** combinations the accuracy is **1.000**; on the 99 pictures of the **unseen** combinations it is **0.394**, against a chance of 0.111. **The colour is right on all 99 pictures (1.000), the shape on only 39 percent.** The breakdown: the 51 pictures of green crosses are named correctly 39 times and called 'green circle' 12 times; the 48 pictures of blue circles are **all** named 'blue square' (48 of 48). So the model does generalise partly: it never confuses the colour, and for green crosses it does well. It fails completely on blue circles. Your Imagine-first guess: not everything or nothing, but a partial yes.

   Why it can generalise: the text encoder builds the caption vector from the colour word and the shape word, which it has seen in other pairs ('green' with squares and circles, 'cross' in red and blue), so it can produce a vector for 'green cross' that it never trained on. Why it can fail: the image encoder has never seen a blue circle, and a small CNN trained on 926 pictures maps it to the nearest picture it knows, a blue square (a circle of radius 4 and a square differ in only a few corner pixels at 16 x 16). The language side was ready, the image side was not.

   Tips and common mistakes:

   - The result depends on which combinations are held out, on the seed and on the CNN. If you hold out other combinations the unseen accuracy changes; try it.
   - 'Compositional generalisation' (understanding a new combination of known parts) is one of the reasons the CLIP idea is attractive, and one of the places where it is known to be unreliable.

   Try also (optional):

   - Hold out 'red square' and 'green circle' instead and re-run Steps 6.3 and 6.4. Is the unseen accuracy higher or lower than 0.394?
   - Print the 9 x 9 cosine matrix of the captions to see which captions the text encoder places next to each other.

   *Exercise 6.4:* Compute the accuracy on the unseen pictures yourself and store it in `acc_unseen_by_hand`: the share of unseen-combination test pictures for which `te_pred` equals `te_true` (use the boolean array `unseen`). Why it matters: it is the number the whole experiment is about.

```python
acc_unseen_by_hand = None   # YOUR CODE: the share of unseen-combination test pictures for which te_pred equals te_true (use the boolean array `unseen`)
if acc_unseen_by_hand is None:
    print("Fill in acc_unseen_by_hand above, then run this cell again.")
else:
    check("accuracy on unseen combinations", abs(acc_unseen_by_hand - acc_unseen) < 1e-9, "(te_pred == te_true)[unseen].mean()")
```

   *What it means.* the mean of the correct predictions (`te_pred` equal to `te_true`) over the pictures selected by `unseen` gives 0.394: 39 of 99 pictures.

5. **Step 6.5. The real CLIP: nine pictures, nine prompts, three templates.** Now the real CLIP. `load_clip()` returns the model and its processor; the first call downloads about 600 MB (on the Spark it reads the offline cache; if loading fails, the cell prints a clear message and the later cells skip). We draw nine clean 224 x 224 pictures with `draw_shape_pil` (red, green and blue; circle, square and triangle), encode them and the nine prompts of each of three templates, compute the 9 x 9 cosine matrix, and read the top-1 accuracy per template. We also print the zero-shot probabilities for one picture, using the model's own `logit_scale`.

```python
try:
    clip_model, clip_proc = load_clip()                       # about 600 MB the first time; the Spark reads its offline cache
    print("CLIP loaded: %.0f M parameters" % (sum(p.numel() for p in clip_model.parameters()) / 1e6))
except Exception as e:
    clip_model = clip_proc = None
    print("Could not load CLIP (%s). Check your internet connection (Colab) or ask your instructor about the offline cache (Spark)." % type(e).__name__)
    print("The cells that need CLIP will print a note and skip; everything before this step still counts.")

def _feat(out): return out if torch.is_tensor(out) else out.pooler_output          # different library versions return a tensor or an object

@torch.no_grad()
def clip_images(imgs): return F.normalize(_feat(clip_model.get_image_features(**clip_proc(images=imgs, return_tensors="pt"))), dim=-1)

@torch.no_grad()
def clip_texts(texts): return F.normalize(_feat(clip_model.get_text_features(**clip_proc(text=texts, return_tensors="pt", padding=True))), dim=-1)
```

```python
RC, RS = ["red", "green", "blue"], ["circle", "square", "triangle"]
pil_imgs = [draw_shape_pil(c, s) for c in RC for s in RS]            # nine clean 224 x 224 pictures, index = colour * 3 + shape
labels9 = ["%s %s" % (c, s) for c in RC for s in RS]
templates = ["a photo of a {} {}", "a {} {}", "a picture of a {} {} on a white background"]
if clip_model is not None:
    zi = clip_images(pil_imgs); acc_tpl = {}
    for t in templates:
        zt = clip_texts([t.format(c, s) for c in RC for s in RS])
        acc_tpl[t] = float(((zi @ zt.T).argmax(1) == torch.arange(9)).float().mean())
        print("%-48s top-1 accuracy %.3f" % ('"' + t + '"', acc_tpl[t]))
    RESULTS["p6_clip_templates"] = {k: round(v, 3) for k, v in acc_tpl.items()}
    print("logit_scale.exp() = %.1f" % float(clip_model.logit_scale.exp()))
else:
    print("CLIP is not loaded: skipped.")
```

```python
if clip_model is not None:
    zt = clip_texts([templates[0].format(c, s) for c in RC for s in RS])
    cosm = (zi @ zt.T).numpy()
    plt.figure(figsize=(5.2, 4.4)); plt.imshow(cosm, cmap="viridis"); plt.colorbar(label="cosine")
    plt.xticks(range(9), labels9, rotation=60, ha="right", fontsize=7); plt.yticks(range(9), labels9, fontsize=7)
    plt.xlabel("prompt"); plt.ylabel("picture"); plt.title("9 x 9 cosine matrix"); plt.tight_layout(); plt.show()
    probs = (clip_model.logit_scale.exp() * zi[0] @ zt.T).softmax(-1)          # zero-shot probabilities of the first picture (red circle)
    print("probabilities for the picture 'red circle':")
    for k in probs.argsort(descending=True)[:4]: print("   %-16s %.3f" % (labels9[int(k)], float(probs[k])))
    print("diagonal cosine (right pair) %.3f on average, off-diagonal %.3f" % (np.diag(cosm).mean(), (cosm.sum() - np.trace(cosm)) / 72))
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
CLIP loaded: 151 M parameters
"a photo of a {} {}"                             top-1 accuracy 0.889
"a {} {}"                                        top-1 accuracy 0.889
"a picture of a {} {} on a white background"     top-1 accuracy 1.000
logit_scale.exp() = 100.0
probabilities for the picture 'red circle':
   red circle       0.781
   red triangle     0.124
   blue circle      0.057
   green circle     0.031
diagonal cosine (right pair) 0.335 on average, off-diagonal 0.260
```

   **What you should see.**

   CLIP-B/32 loads with about **151 million parameters**. The accuracies are **0.889 for 'a photo of a {} {}' (8 of 9 right), 0.889 for 'a {} {}', and 1.000 for 'a picture of a {} {} on a white background'**. The wording of the prompt changes the result: the third template describes what the pictures really look like (a plain background), and CLIP gets all nine. The one error of the first two templates is the red square, which is read as a red triangle. The model's `logit_scale.exp()` is **100.0**, the maximum allowed in training, as in Step 6.2. The probabilities for the red circle are 0.781 for 'red circle', 0.124 for 'red triangle' and 0.057 for 'blue circle': the right answer but far from certain. The heat map shows a bright diagonal (right pairs have a mean cosine of 0.335, wrong pairs 0.260): in CLIP, cosines between a picture and a text are small numbers (a cosine of 0.33 is a good match) and what counts is the order, which is why the scale of 100 is needed.

   Tips and common mistakes:

   - If loading fails, read the printed message: on Colab it is usually the internet connection, on the Spark the offline cache. The other cells of this part still work; only the CLIP cells are skipped.
   - The exact probabilities can differ in the third digit between library versions; the ranking of the nine labels should not.

   Try also (optional):

   - Draw the shapes in other colours that CLIP knows ('orange', 'purple', 'yellow') and see how many of the nine it gets.
   - Try the template 'this is a {} {}' and 'a {} {} shape'.

   *Exercise 6.5:* Write `top1(img_emb, txt_emb)`, which returns for every picture the index of the best-matching text (both inputs are normalised, rows are items). Why it matters: it is the whole zero-shot classifier of CLIP in one line.

```python
def top1(img_emb, txt_emb):
    """Index of the best-matching text for every picture (both are L2-normalised; rows are items)."""
    return None   # YOUR CODE: cosine matrix = img_emb @ txt_emb.T, then argmax over the texts
if clip_model is None:
    print("CLIP is not loaded: skipped.")
elif top1(zi, zt) is None:
    print("Fill in top1 above, then run this cell again.")
else:
    check("same as the cosine argmax", bool((top1(zi, zt) == (zi @ zt.T).argmax(1)).all()), "(img_emb @ txt_emb.T).argmax(1)")
```

   *What it means.* With normalised vectors the cosine matrix is `img_emb @ txt_emb.T`, and the prediction is the arg max over the texts. The check compares with the same expression on the nine pictures; the nine predictions are 0, 2, 2, 3, 4, 5, 6, 7, 8 for the first template: picture 1 (red square) is read as number 2 (red triangle).

6. **Step 6.6. Prompt wording and the limits of CLIP.** A good test of a model is where it fails. Three probes of the real CLIP. First, prompts that include something silly ('a photo of a dog') for a red circle, and a set of prompts in which no prompt is right; CLIP must still pick one. Second, counting: pictures with 2 to 6 black circles and prompts 'a picture of {two, three, four, five, six} black circles'. Third, text in the image: the words STOP and GO drawn with PIL and the prompts 'the word STOP' and 'the word GO'.

```python
def show_probs(img, prompts, top=5):
    p = (clip_model.logit_scale.exp() * clip_images([img]) @ clip_texts(prompts).T).softmax(-1)[0]
    for k in p.argsort(descending=True)[:top]: print("   %-34s %.3f" % (prompts[int(k)], float(p[k])))

if clip_model is not None:
    red_circle = draw_shape_pil("red", "circle")
    print("A red circle, prompts include a dog:")
    show_probs(red_circle, ["a photo of a dog", "a photo of a red circle", "a photo of a blue square", "a photo of a cat"])
    print("A red circle, but NO prompt is right (CLIP must still pick one):")
    show_probs(red_circle, ["a photo of a dog", "a photo of a cat", "a photo of a car"])
else:
    print("CLIP is not loaded: skipped.")
```

```python
def draw_circles(k, size=224):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), "white"); d = ImageDraw.Draw(img)
    spots = [(0.25, 0.3), (0.65, 0.25), (0.45, 0.55), (0.2, 0.75), (0.7, 0.7), (0.5, 0.15), (0.85, 0.45)]
    for x, y in spots[:k]: d.ellipse([x * size - 14, y * size - 14, x * size + 14, y * size + 14], fill="black")
    return img

if clip_model is not None:
    words = ["two", "three", "four", "five", "six"]
    prompts_n = ["a picture of %s black circles" % w for w in words]
    zt_n = clip_texts(prompts_n); hits = 0
    for k in range(2, 7):
        pred = int((clip_images([draw_circles(k)]) @ zt_n.T).argmax()) + 2
        hits += (pred == k); print("picture with %d circles -> CLIP says %d" % (k, pred))
    print("counting: %d of 5 right" % hits)
    RESULTS["p6_counting"] = hits
```

```python
if clip_model is not None:
    from PIL import Image, ImageDraw, ImageFont
    def word_picture(word, size=224):
        img = Image.new("RGB", (size, size), "white")
        try: font = ImageFont.load_default(size=64)
        except TypeError: font = ImageFont.load_default()
        ImageDraw.Draw(img).text((size // 2, size // 2), word, fill="black", anchor="mm", font=font); return img
    for w in ["STOP", "GO"]:
        print('A picture of the word "%s":' % w)
        show_probs(word_picture(w), ["the word STOP", "the word GO", "a photo of a dog"], top=3)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
A red circle, prompts include a dog:
   a photo of a red circle            0.994
   a photo of a blue square           0.003
   a photo of a cat                   0.001
   a photo of a dog                   0.001
A red circle, but NO prompt is right (CLIP must still pick one):
   a photo of a cat                   0.502
   a photo of a dog                   0.272
   a photo of a car                   0.226
picture with 2 circles -> CLIP says 3
picture with 3 circles -> CLIP says 3
picture with 4 circles -> CLIP says 3
picture with 5 circles -> CLIP says 3
picture with 6 circles -> CLIP says 3
counting: 1 of 5 right
A picture of the word "STOP":
   the word STOP                      0.996
   the word GO                        0.003
   a photo of a dog                   0.000
A picture of the word "GO":
   the word GO                        0.989
   the word STOP                      0.010
   a photo of a dog                   0.001
```

   **What you should see.**

   (1) For a red circle, with a dog, a cat, 'red circle' and 'blue square' as options, 'a photo of a red circle' gets **0.994**; the dog gets 0.001. When **no** prompt is right (dog, cat, car), CLIP does not say 'none': it gives cat 0.502, dog 0.272 and car 0.226. A zero-shot classifier **always answers with one of the labels you give it**; the probabilities sum to 1 whatever the picture shows. (2) **Counting fails**: CLIP says 'three' for all five pictures (2, 3, 4, 5, 6 circles) and is right once (for the picture with 3 circles) in 5. Counting needs more than a similarity between a picture and a sentence. (3) **Words in pictures work**: the picture of STOP gets 0.996 for 'the word STOP', the picture of GO gets 0.989 for 'the word GO'. CLIP learned to read short words because captions on the internet often repeat the text that is visible in the picture. This is also why it can be fooled by a label stuck on an object (a typographic attack).

   Tips and common mistakes:

   - The rank order of the labels is what the model offers; do not read 0.502 as 'CLIP is 50 percent sure it is a cat'. The probabilities depend on the labels you chose.
   - For a real system add a 'none of the above' label and a threshold on the cosine, and let a person check what falls below it.

   Try also (optional):

   - Test 'a photo of a red square' against the red circle: is CLIP separating squares from circles at all?
   - Draw the word STOP in red on a green background and test again.

   *Exercise 6.6:* Write 3 to 5 sentences (at least 25 words) in `answer_66`: why can the captions CLIP learned from make it biased or wrong, and what would you check before using it in a school project? Why it matters: the previous probes show failures you can see; biases in the data are failures you have to look for.

```python
answer_66 = ""   # YOUR ANSWER (3 to 5 sentences): why can the web captions CLIP learned from make it biased or wrong, and name one thing a person should check before using it in a school project
if len(answer_66.split()) < 25:
    print("Write at least 25 words in answer_66, then run this cell again.")
else:
    check("answer written", True)
```

   *What it means.* A good answer says that the captions come from the internet, which shows some groups, places and jobs far more often or more stereotypically than others, and that a caption rarely states the obvious; it proposes a test on pictures from your own community with accuracy per group (Step 7.3), not only the average, and a human in the loop.

### Part 7. Design a multimodal system, and reflect (15 min)

*Lesson sections: all of Week 11.* You have patchified pictures, built a ViT, tested positions and trained a small CLIP. In this last part you stop building and start designing: you choose a real problem from your school or community, fill in a design card for a multimodal system, count its token budget with your own `vit_counts`, test how a model can be fair or unfair to groups, print your results, and answer four reflection questions. ==The goal is to show that you can connect the architecture to a real decision: what goes in, how it is fused, how it is checked, and who could be harmed.==

1. **Step 7.1. Imagination design: a multimodal system for your school or community.** Choose a real problem. Examples: alt text for the pictures of the school newsletter; a flood-gauge photo plus a text report from a barangay volunteer; a voice-and-text helpdesk for enrolment questions; recognising jeepney routes from a photo of the signboard. Fill in the design card: what comes in, what comes out, which fusion pattern you would use (a CLIP-style shared space, a projector into a language model, cross-attention, or late fusion, with one sentence of why), where the data comes from and who agreed, how you evaluate each modality plus a human check, what the system does when it is unsure, and one risk. The printed card is part of your submission.

```python
import textwrap
def show_card(design):
    print("=" * 70)
    for k, v in design.items():
        print("%-18s %s" % (k.replace("_", " ").upper() + ":", textwrap.fill(v or "(empty)", 50, subsequent_indent=" " * 19)))
    print("=" * 70)
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
======================================================================
PROBLEM:           Flood-gauge photos and short text reports from
                   barangay volunteers arrive by
                   chat, and the office reads them
                   by hand.
MODALITIES IN:     A photo of the gauge board and a text message in
                   English or Filipino.
MODALITIES OUT:    A water level in centimetres with a confidence,
                   and a one-line text summary for
                   the office.
FUSION:            Projector into a language model: a ViT turns the
                   photo into tokens, a projector
                   maps them into the model's
                   embedding space, and the text
                   report is read together with
                   them.
DATA AND CONSENT:  Photos taken by volunteers who agreed in writing;
                   no faces or house numbers kept;
                   300 labelled photos from last
                   season.
EVALUATION:        Mean error in centimetres on photos, accuracy of
                   the text summary against a
                   person's note, and a human
                   check of every reading above
                   the warning level.
FAILURE HANDLING:  If the confidence is below 0.7 or the photo is
                   dark, the system asks the
                   volunteer for another photo and
                   alerts a person.
RISK:              A wrong low reading in a storm could delay an
                   evacuation, so a person must
                   confirm every high-risk
                   decision.
======================================================================
```

   **What you should see.**

   The cell prints your card with the eight fields in capitals. The check wants at least four words per field and a fusion pattern that names one of the four options. The sample answer (flood-gauge photos with text reports) uses a projector into a language model; yours should say why your choice fits your problem: a shared space such as CLIP is good for search and classification by label, a projector is good when the answer is free text, cross-attention is good when two inputs must be aligned in detail, and late fusion is the simplest and the easiest to check.

   Tips and common mistakes:

   - Be specific: 'accuracy' is not an evaluation, 'mean error in centimetres on 100 photos from last season plus a person who checks every reading above the warning level' is.
   - Consent and privacy are part of the design: photos of children or houses need a plan before the first picture is taken.

   Try also (optional):

   - Write a second card for the same problem with a different fusion pattern and compare the risks.

   *Exercise 7.1:* Fill in the dictionary `design` for your own problem. Why it matters: a model without a design around it is not a system; the card is where the risks become visible before they happen.

```python
design = {
    "problem":          "",   # which real problem of your school or community, in one sentence
    "modalities_in":    "",   # what goes in: pictures, text, voice, ...
    "modalities_out":   "",   # what comes out
    "fusion":           "",   # one of: CLIP-style shared space, projector into a language model, cross-attention, late fusion; and why
    "data_and_consent": "",   # where the data comes from and who agreed
    "evaluation":       "",   # a metric per modality and a human check
    "failure_handling": "",   # what the system does when it is unsure
    "risk":             "",   # one risk and who could be harmed
}
show_card(design)
empty = [k for k, v in design.items() if len(v.split()) < 4]
if empty:
    print("Write at least 4 words for:", empty)
else:
    check("the card is complete", True)
    check("the fusion pattern is one of the four", any(w in design["fusion"].lower() for w in ["clip", "projector", "cross-attention", "late fusion", "shared space"]), "name one of: CLIP-style shared space, projector into a language model, cross-attention, late fusion")
```

   *What it means.* The check prints your card and tests it for completeness. The cell below the exercise in the notebook contains a worked example (`Show a possible answer`).

2. **Step 7.2. A token budget for your design.** A design needs a budget. For an image of your chosen side and patch size, the number of visual tokens is (side / patch) squared; if the image tokens go into a language model together with a prompt of 64 tokens, the sequence has visual plus 64 tokens, and each attention head computes that number squared in scores. We use your `vit_counts` from Step 4.3 with the defaults of a typical visual encoder (336 pixels, patch 14), and compare with a patch half as large.

```python
my_image_px, my_patch, prompt_tokens = 336, 14, 64           # change these to your own design (the image side must be divisible by the patch size)
def budget(px, patch):
    c = vit_counts(px, px, patch, 3, 1024)
    visual = c["patches"]; seq = visual + prompt_tokens
    return dict(patch=patch, visual=visual, patch_dim=c["patch_dim"], seq=seq, scores=seq * seq)
rows7 = [budget(my_image_px, my_patch), budget(my_image_px, my_patch // 2)]
print("%-8s %14s %12s %14s %18s" % ("patch", "visual tokens", "numbers each", "sequence", "scores per head"))
for r_ in rows7: print("%-8d %14d %12d %14d %18d" % (r_["patch"], r_["visual"], r_["patch_dim"], r_["seq"], r_["scores"]))
print("halving the patch size multiplies the attention scores by %.1f" % (rows7[1]["scores"] / rows7[0]["scores"]))
RESULTS["p7_token_budget"] = (rows7[0]["visual"], rows7[0]["seq"], rows7[1]["visual"], rows7[1]["seq"])
assert my_image_px % my_patch == 0 and rows7[0]["seq"] == rows7[0]["visual"] + prompt_tokens
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
patch     visual tokens numbers each       sequence    scores per head
14                  576          588            640             409600
7                  2304          147           2368            5607424
halving the patch size multiplies the attention scores by 13.7
```

   **What you should see.**

   For 336 x 336 pixels with patch 14: **576 visual tokens** (24 x 24) of 588 numbers each (14 x 14 x 3), a sequence of **640** with the prompt and 409,600 attention scores per head. With patch 7: **2,304 tokens** of 147 numbers, a sequence of **2,368** and **5,607,424** scores: halving the patch size multiplies the scores by **13.7** (not 16, because the 64 prompt tokens do not grow). This is why systems that read documents or screenshots, which need small patches to read small text, are expensive, and why they often split the picture into tiles or compress the visual tokens before they reach the language model.

   Tips and common mistakes:

   - Change `my_image_px` and `my_patch` to the values of your design; the assert reminds you that the side must be a multiple of the patch size.
   - The numbers per patch (588 or 147) do not enter the attention cost; they only set the size of the patch embedding.

   Try also (optional):

   - Double the image side instead of halving the patch. How many tokens do you get? (The same as halving the patch: 2,304.)

   *Exercise 7.2:* Store the sequence length of your design (visual tokens plus the prompt tokens) in `seq_len_mine`. Why it matters: it is the number that sets the memory and the time of your system.

```python
seq_len_mine = None   # YOUR CODE: the sequence length of your design: visual tokens for my_image_px and my_patch plus prompt_tokens
if seq_len_mine is None:
    print("Fill in seq_len_mine above, then run this cell again.")
else:
    check("sequence length", seq_len_mine == (my_image_px // my_patch) ** 2 + prompt_tokens, "(side // patch) squared tokens, plus the prompt tokens")
```

   *What it means.* `(my_image_px // my_patch) ** 2 + prompt_tokens`: 24 x 24 + 64 = 640 with the defaults.

3. **Step 7.3. A mini evaluation: accuracy per group and the worst group.** A single accuracy hides who the system fails. You write `group_accuracy(pred, true, groups)`, the accuracy for each group, and apply it to the toy CLIP of Part 6: per colour of the picture, per shape, and for seen against unseen combinations. The worst group is the one a fairness check looks at first. In the exercise you write the function; the cell below it uses a reference version, so the table is printed even if yours is unfinished.

```python
te_cols = [COLOURS[c] for c in col_te]; te_shps = [SHAPES3[s] for s in shp_te]
```

   *Exercise 7.3:* Write `group_accuracy(pred, true, groups)`: for every distinct group name the share of its pictures with `pred` equal to `true`. Why it matters: it is the first tool of any fairness check.

```python
def group_accuracy(pred, true, groups):
    """pred and true: arrays of caption indices; groups: a list with one group name per picture. Returns {group name: accuracy}."""
    return {}   # YOUR CODE: for every distinct group name, the share of its pictures with pred equal to true
out = group_accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), ["a", "a", "b", "b"])
if not out:
    print("Fill in group_accuracy above, then run this cell again.")
else:
    check("small test", out == {"a": 1.0, "b": 0.5}, "for each group take the mean of (pred == true) over the pictures of that group")
```

   *What it means.* Loop over `sorted(set(groups))`, make a boolean mask of the pictures of the group, and take the share of those pictures where `pred` and `true` agree. The check uses a four-picture example: group a is 1.0, group b is 0.5.

```python
def group_accuracy_ref(pred, true, groups):          # our version, so this step works even if yours is unfinished
    return {g: float((pred[np.array([x == g for x in groups])] == true[np.array([x == g for x in groups])]).mean()) for g in sorted(set(groups))}

by_colour = group_accuracy_ref(te_pred, te_true, te_cols)
by_shape = group_accuracy_ref(te_pred, te_true, te_shps)
by_unseen = group_accuracy_ref(te_pred, te_true, ["unseen" if u else "seen" for u in unseen])
print("toy CLIP, accuracy per colour :", {k: round(v, 3) for k, v in by_colour.items()})
print("toy CLIP, accuracy per shape  :", {k: round(v, 3) for k, v in by_shape.items()})
print("toy CLIP, seen against unseen :", {k: round(v, 3) for k, v in by_unseen.items()})
worst = min(by_colour, key=by_colour.get)
print("worst colour group: %s (%.3f), best: %s (%.3f), gap %.3f" % (worst, by_colour[worst], max(by_colour, key=by_colour.get), max(by_colour.values()), max(by_colour.values()) - by_colour[worst]))
RESULTS["p7_worst_group"] = (worst, round(by_colour[worst], 3))
assert abs(by_unseen["unseen"] - acc_unseen) < 1e-9 and abs(by_unseen["seen"] - acc_seen) < 1e-9
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
toy CLIP, accuracy per colour : {'blue': 0.68, 'green': 0.92, 'red': 1.0}
toy CLIP, accuracy per shape  : {'circle': 0.68, 'cross': 0.92, 'square': 1.0}
toy CLIP, seen against unseen : {'seen': 1.0, 'unseen': 0.394}
worst colour group: blue (0.680), best: red (1.000), gap 0.320
```

   **What you should see.**

   The toy CLIP has an accuracy of **1.000 on red pictures, 0.920 on green and 0.680 on blue**; per shape it is 1.000 for squares, 0.920 for crosses and 0.680 for circles; and 1.000 on seen against 0.394 on unseen combinations. The **worst colour group is blue (0.680)**, and the gap to the best (red) is 0.320. The average over all test pictures, about 0.87, would have hidden this. The reason is the one you know from Step 6.4: the blue circle was held out, and the blue pictures contain the unseen combination blue circle, while green contains green cross. In a real system the groups are people, languages, accents, skin tones or neighbourhoods, and the same reasoning applies: report the accuracy for every group that matters, not only the mean. A fairness check for your design would list the groups (for the flood-gauge system: day and night, rain and dry weather, each barangay), measure the error per group on enough examples, set a maximum gap you accept, and say who decides what to do if a group is above it.

   Tips and common mistakes:

   - A group with few pictures has a noisy accuracy; always print how many pictures each group has before you compare.
   - Groups that are not in your data are invisible to this check: the biggest risks are often the people who never took part in the data collection.

   Try also (optional):

   - Print the accuracy for each of the nine combinations. Which two are the worst, and why?

4. **Step 7.4. Your RESULTS summary.** Print everything you stored in `RESULTS` during the lab. It is your summary for the submission: if an entry is missing, the step that stores it has not been run.

```python
for k, v in RESULTS.items(): print("%-26s %s" % (k, v))
print(len(RESULTS), "stored results")
```

   The block prints (your numbers must be the same with `SEED = 42`; times and the library's last digits can differ):

```text
p4_patch_embed_shape       (8, 16, 32)
p4_tinyvit_params          26757
p4_vit_base                85798656
p4_overfit64               1.0
p4_vit_vs_cnn              (1.0, 1.0)
p4_centre_attention        0.33
p5_left_right              {'small CNN': 1.0, 'ViT with positions': 0.993, 'ViT without positions': 0.537}
p5_shuffle_agreement       {'ViT without positions': 1.0, 'ViT with positions': 0.513}
p5_pooling                 {'cls/shapes': 1.0, 'cls/lr': 0.993, 'mean/shapes': 1.0, 'mean/lr': 0.993}
p5_patch_size              {2: (64, 1.0, 1.0), 4: (16, 1.0, 0.993), 8: (4, 1.0, 0.86)}
p5_data_hunger             {'TinyViT': [0.43, 0.623, 0.69, 0.88], 'SmallCNN': [0.887, 0.957, 0.98, 0.993]}
p5_sincos                  1.0
p6_loss_by_hand            1.0481
p6_scale_losses            {1: 0.952, 5: 0.803, 10: 1.048, 30: 2.589, 100: 8.302}
p6_toy_loss                (3.924, 0.0)
p6_toy_zero_shot           (1.0, 0.394)
p6_clip_templates          {'a photo of a {} {}': 0.889, 'a {} {}': 0.889, 'a picture of a {} {} on a white background': 1.0}
p6_counting                1
p7_token_budget            (576, 640, 2304, 2368)
p7_worst_group             ('blue', 0.68)
20 stored results
```

   **What you should see.**

   You see about 20 entries of this part (plus the entries of Parts 1 to 3 if you ran them): the patch embedding shape (8, 16, 32), the 26,757 parameters of the tiny ViT and 85,798,656 for ViT-Base, the overfitting accuracy 1.0, ViT against CNN (1.0, 1.0), the central attention 0.33, the left/right table (CNN 1.0, ViT 0.993, no positions 0.537), the shuffle agreements 1.0 and 0.513, the pooling table, the patch-size table, the data-hunger lists, the sine-cosine accuracy 1.0, the loss by hand 1.0481, the scale losses, the toy CLIP's training loss (3.924 to 0.0) and zero-shot accuracies (1.0 and 0.394), CLIP's template accuracies, the counting score 1, the token budget (576, 640, 2304, 2368) and the worst group (blue, 0.68).

   Tips and common mistakes:

   - Copy the printed block into your submission together with your written answers.

5. **Step 7.5. Four reflection questions.** Four questions, 25 words or more each. Cite a number from your own output where you can.

   **What you should see.**

   The questions are: (1) what is a multimodal model and what are three real applications; (2) how does a ViT turn an image into tokens; (3) why are position embeddings needed; (4) how does CLIP make zero-shot classification possible, and what could go wrong. Good answers use the lab's numbers: 196 tokens of 768 numbers for a 224 x 224 picture with 16 x 16 patches; 0.993 against 0.537 for the ViT with and without positions; 0.394 on unseen combinations for the toy CLIP; and the failures of the real CLIP (it always answers, it cannot count, it repeats the biases of its captions).

   Tips and common mistakes:

   - The check counts only words. Quality is judged by your instructor with the rubric.

   *Exercise 7.5:* Write your four answers in the dictionary `reflection`. Why it matters: explaining it in your own words is the best test that you understood it.

```python
reflection = {
    "q1_multimodal": "",   # What is a multimodal model? Name three real applications.
    "q2_vit_tokens": "",   # How does a ViT turn an image into tokens? Use the numbers of Part 2 or Part 4.
    "q3_positions":  "",   # Why are position embeddings needed? Cite your left/right result of Part 5.
    "q4_clip":       "",   # How does CLIP make zero-shot classification possible, and what could go wrong?
}
short = [k for k, v in reflection.items() if len(v.split()) < 25]
if short:
    print("Write at least 25 words for:", short)
else:
    check("four reflection answers written", True)
    RESULTS["p7_reflection_words"] = {k: len(v.split()) for k, v in reflection.items()}
```

   *What it means.* A good answer is specific, short and uses numbers; see the sample answers.

6. **Step 7.6. Go further.** Two tutorials of the University of Amsterdam Deep Learning course continue where this lab stops. Tutorial 15, Vision Transformers (`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html`), builds a ViT for CIFAR10 from scratch, trains it, compares it with a CNN and explains why the ViT needs far more data, which is what you saw in Step 5.5 on a small scale. Tutorial 17, SimCLR (`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html`), explains contrastive learning on pictures with the same kind of loss as Step 6.1, but with two augmented views of the same picture as the positive pair instead of a picture and a caption. Credit: the tutorials are by Phillip Lippe and are free to read; they are not part of the marked work and no code from them is copied here.

### Deliverables

A short checklist before you submit: every exercise has a written answer under its block; every `None` and every empty string in a starter cell has been replaced; the code blocks ran from the first to the last without an error (the CLIP cells of Part 6 need the model; if it cannot be loaded, say so in your report); and the printed `RESULTS` summary of Step 7.4 is included.

Submit your work the way your instructor asks (the notebook with outputs, or your JupyterLab files with the printed output and your written answers). For Parts 4 to 7 the submission must include:

- **Part 4 (18 points):** `MyPatchEmbed` and the forward pass of `MyTinyViT`, checked (4.1 and 4.2); the shape trace; the parameter count by formula and with `count_params` (26,757), the count for ViT-Base (85,798,656 without a head) and your count for d = 48 (4.3); the starting loss near ln 5 and the overfitting test (4.4); the table of TinyViT against the CNN with an honest reading of the tie, and the learning curves (4.5); and the CLS attention map with the share of attention on the central patches (4.6). Your Imagine-first guess compared with the real count.
- **Part 5 (14 points):** the left/right table with the three models and the gap the positions make (5.1); the shuffle test with the agreement rates and what they prove (5.2); the pooling table and the honest finding of a tie (5.3); the patch-size table (tokens, parameters, seconds, accuracy) with the plot and the 14.6 times cost ratio (5.4); the data-hunger table for the ViT and the CNN with the explanation (5.5); and `sincos_2d`, its accuracy and your three answers on where 'where' matters (5.6). Your Imagine-first guess compared with what you measured.
- **Part 6 (22 points):** `clip_loss_np` and the hand calculation equal to PyTorch (6.1); the table over the scales with your reading of the U shape (6.2); the miniature CLIP, its loss curve and the held-out mask (6.3); the accuracy on seen and unseen combinations, the breakdown of the unseen captions and your explanation (6.4); the real CLIP: accuracy per template, the 9 x 9 heat map, `logit_scale` and `top1` (6.5); and the three probes (silly prompts, counting, words in pictures) with your paragraph on biases (6.6). Your Imagine-first guess compared with the measured 0.394.
- **Part 7 (10 points):** the design card (7.1); the token budget and the comparison with half the patch size (7.2); `group_accuracy`, the table per group and the worst group, with what a fairness check would look like for your design (7.3); the printed `RESULTS` (7.4); and your four reflection answers (7.5).

Together with Parts 1 to 3 the lab is marked out of **100 points**: Part 1 is worth 10, Part 2 is worth 12, Part 3 is worth 14, Part 4 is worth 18, Part 5 is worth 14, Part 6 is worth 22 and Part 7 is worth 10. The rubric is in the section "How Week 11 is assessed".

**Stretch goals.** Train the TinyViT on the left/right task with a 3-layer encoder and see whether the patch-8 result of Step 5.4 improves; hold out other combinations in Step 6.3 and compare the unseen accuracy; add the 'none of the above' label to the CLIP probes of Step 6.6 and set a threshold on the cosine; repeat Step 5.5 with 800 and 1,600 pictures; and write a second design card for the same problem with a different fusion pattern.
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions, one for each idea of the week, with an explanation after each answer. Ten of them ask you to calculate, so keep a calculator or Python at hand.

```quiz
{
  "id": "week-11-check",
  "title": "Week 11 quiz: Multimodal AI",
  "passMark": 70,
  "questions": [
    {
      "q": "Which of these systems is multimodal?",
      "options": [
        "A spam filter that reads the text of an email",
        "A model that takes a photo of a plant and a typed question and answers in text",
        "A model that translates a sentence from English to Filipino",
        "A regression that predicts rainfall from the past 30 days of rainfall numbers"
      ],
      "answer": 1,
      "why": "It links two modalities, an image and text. The other three each use a single kind of data (text, text, numbers)."
    },
    {
      "q": "A 384 × 384 colour image is cut into 32 × 32 patches. How many patch tokens are there (before adding a CLS token)?",
      "options": [
        "12",
        "576",
        "288",
        "144"
      ],
      "answer": 3,
      "why": "384 ÷ 32 = 12 patches per side, and 12 × 12 = 144. The number 12 is only the count per side, and 576 is what 16 × 16 patches would give (24 × 24)."
    },
    {
      "q": "A ViT flattens one 14 × 14 patch of a colour (RGB) image. How many numbers are in the flattened patch?",
      "options": [
        "588",
        "196",
        "42",
        "1,764"
      ],
      "answer": 0,
      "why": "14 × 14 × 3 = 588. A value of 196 ignores the three colour channels, and 42 is 14 × 3."
    },
    {
      "q": "A ViT with hidden size d = 384 uses 16 × 16 colour patches. How many parameters does its linear patch embedding have (weights and bias)?",
      "options": [
        "294,912",
        "589,824",
        "295,296",
        "768"
      ],
      "answer": 2,
      "why": "The weight matrix has 16 × 16 × 3 = 768 rows and 384 columns: 768 × 384 = 294,912, and the bias adds 384, giving 295,296. The first option forgets the bias, and 589,824 is 768 × 768 (for d = 768)."
    },
    {
      "q": "A 224 × 224 colour image is cut into 8 × 8 patches and a CLS token is added. How long is the token sequence?",
      "options": [
        "784",
        "785",
        "786",
        "197"
      ],
      "answer": 1,
      "why": "224 ÷ 8 = 28, so 28 × 28 = 784 patches, plus 1 CLS token gives 785. 197 is the sequence for 16 × 16 patches."
    },
    {
      "q": "A 16 × 16 greyscale picture is cut into 4 × 4 patches and a CLS token is added. How many attention scores does one head compute in one layer?",
      "options": [
        "289",
        "256",
        "65,536",
        "16"
      ],
      "answer": 0,
      "why": "(16 ÷ 4)² = 16 patches, plus 1 CLS = 17 tokens, and 17 × 17 = 289 scores. Using one token per pixel would instead give 256 × 256 = 65,536."
    },
    {
      "q": "Why does a ViT need position embeddings?",
      "options": [
        "They make each patch vector longer so that the model has more features",
        "They stop the patches from overlapping with each other",
        "Self-attention alone returns the same vectors, only reordered, when the patches are shuffled, so it cannot tell where a patch came from",
        "They replace the CLS token as the summary of the image"
      ],
      "answer": 2,
      "why": "Self-attention is permutation-equivariant: shuffle the input and the outputs are shuffled in the same way. Adding a position vector to each patch breaks that symmetry. Position embeddings are added, so the length does not change, and they do not replace the CLS token."
    },
    {
      "q": "Which list gives the stages of a ViT in the right order?",
      "options": [
        "Transformer encoder, patchify, linear projection, CLS head",
        "Linear projection, position embeddings, patchify, encoder, head",
        "CLS head, patchify, encoder, position embeddings",
        "Patchify, linear projection, add CLS token, add position embeddings, Transformer encoder, head on the CLS output"
      ],
      "answer": 3,
      "why": "The image is first cut and flattened, then each patch is projected, then the CLS token is added and positions are added, then the encoder runs, and the head reads the CLS output. The other orders put a step before the data it needs."
    },
    {
      "q": "A ViT-style model has a learned position table for a CLS token and the patches of a 224 × 224 image cut into 16 × 16 patches, with hidden size 384. How many numbers are in the table?",
      "options": [
        "75,264",
        "75,648",
        "151,296",
        "196"
      ],
      "answer": 1,
      "why": "There are 196 + 1 = 197 positions and each has 384 numbers: 197 × 384 = 75,648. The first option forgets the CLS position (196 × 384), and 151,296 is the table for d = 768."
    },
    {
      "q": "An image vector is a = [1, 2] and a text vector is b = [2, 1]. What is their cosine similarity?",
      "options": [
        "0.4",
        "0.6",
        "0.8",
        "1.0"
      ],
      "answer": 2,
      "why": "a·b = 1×2 + 2×1 = 4, and ‖a‖ = ‖b‖ = √5, so the cosine is 4 ÷ 5 = 0.8. A cosine of 1 would need the two vectors to point in exactly the same direction."
    },
    {
      "q": "For one photo, CLIP gives cosine 0.5 with the prompt “a photo of a jeepney” and 0.3 with “a photo of a mango”. With a logit scale of 10 and only these two prompts, what probability does the jeepney prompt get?",
      "options": [
        "0.88",
        "0.55",
        "0.50",
        "0.98"
      ],
      "answer": 0,
      "why": "The logits are 5 and 3, and the probability is 1 ÷ (1 + e^−2) = 0.881. If you forget the scale you get 1 ÷ (1 + e^−0.2) = 0.55, and a probability of 0.50 would mean the two prompts fit equally well."
    },
    {
      "q": "In a batch of 4 image–caption pairs, an untrained model gives every caption the same probability for each image. What is the loss term −ln p for one picture's true caption?",
      "options": [
        "0.25",
        "0.69",
        "4.00",
        "1.39"
      ],
      "answer": 3,
      "why": "The probability is 1 ÷ 4 = 0.25 and −ln 0.25 = ln 4 = 1.386, about 1.39. The value 0.25 is the probability itself, and 0.69 is ln 2, which would be right for a batch of 2."
    },
    {
      "q": "A LLaVA-style model reads a 448 × 448 photo with 14 × 14 patches and a text prompt of 50 tokens, and each patch becomes one token. How long is the sequence the language model reads?",
      "options": [
        "1,074",
        "82",
        "1,024",
        "2,148"
      ],
      "answer": 0,
      "why": "448 ÷ 14 = 32 patches per side, and 32 × 32 = 1,024 image tokens, plus 50 text tokens gives 1,074. The answer 82 adds only 32 + 50, and 1,024 forgets the prompt."
    },
    {
      "q": "Which pairing of a way to combine modalities with its main risk is correct?",
      "options": [
        "CLIP-style shared space: it generates long captions that are always accurate",
        "Late fusion: it can learn every interaction between the modalities without any extra model",
        "Projector with prefix image tokens: each image uses many context tokens, and the model may state things that are not in the picture",
        "Cross-attention: it removes the need for any training data"
      ],
      "answer": 2,
      "why": "Prefix tokens do cost context (for example 576 tokens for one image) and the model can hallucinate. CLIP returns scores, not captions. Late fusion combines decisions and cannot see interactions between the raw inputs. Cross-attention layers still need training."
    },
    {
      "q": "You use CLIP zero-shot with four prompts (jeepney, tricycle, motorbike, carabao) on a photo of a mango. What is the most likely outcome?",
      "options": [
        "CLIP reports that none of the classes fits and returns nothing",
        "CLIP still returns probabilities that add up to 1 over the four prompts, so one wrong class will look like the answer",
        "CLIP creates a new class called mango",
        "CLIP refuses until it has been retrained on mango photos"
      ],
      "answer": 1,
      "why": "Zero-shot probabilities compare only the prompts you supply, so they always add up to 1 across them. A missing class cannot be chosen, which is why you check results and add an option such as “something else”. CLIP does not invent classes or need retraining."
    }
  ]
}
```

## How Week 11 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: What is multimodal? (modalities as tensors, the colour–shape data, unimodal features, early and late fusion, three applications) | 10 | The sizes of the four modalities computed and asserted; the nine caption classes counted; colour and shape features extracted; the three classifiers (colour only, shape only, both) compared in a table with the complementarity explained by the numbers measured; late fusion by the product compared with early fusion; the three applications filled in with a specific failure and a human-oversight step for each | Everything runs and the asserts pass; the comparison of early and late fusion or the applications are brief | Accuracies copied from the lesson rather than measured; one modality used for a task it cannot solve without saying so; applications with no failure or no oversight step |
| Part 2: Patchify (patchify by hand and with PyTorch, token counts, attention cost, patch-size trade-off) | 12 | `patchify` and `unpatchify` with an exact round trip; the Conv2d patch embedding equal to patchify followed by a linear layer; the table from `vit_counts` with 196 / 768 / 197 / 590,592 for 224 pixels and patch 16, 49 / 3,072 for patch 32 and 784 / 192 for patch 8 asserted; the attention scores for patch tokens against pixel tokens with the log plot; the empty-patch counts for p = 2, 4, 8; a two-sentence answer on the word analogy and where it fails | Everything runs and the asserts pass; the cost analysis or the patch-size reading is brief | Patches and pixels confused in the cost; the CLS token forgotten in the sequence length; the analogy asserted with no limit |
| Part 3: Why position embeddings? (permutation equivariance, bags of patches, learned and sine-cosine positions, interpolation) | 14 | Self-attention applied to the tokens and to a shuffled copy with the output rows shuffled in the same way (assert); a patch-shuffled `make_left_right` picture with an identical bag of patches but a possibly different label; the learned table and `sincos_2d` with 16 different vectors; the outputs with positions no longer permutations of each other; the cosine heat map with neighbours more similar than far positions, stated with numbers; the interpolated table with the new shape and the parameter counts | Everything runs and the asserts pass; the explanation of why positions break the symmetry or of interpolation is brief | The shuffled outputs compared without applying the permutation; positions added before the shuffle; a shape error in the interpolation left unexplained |
| Part 4: Build a tiny Vision Transformer (PatchEmbed, TinyViT, parameter count, sanity checks, training against a CNN, CLS attention) | 18 | `MyPatchEmbed` with output (8, 16, 32) and 544 parameters, and a forward pass of `MyTinyViT` that agrees with the reference; the parameter count by formula equal to `count_params` (26,757) with the bias terms explained, and ViT-Base from the same formula (7,087,872 per layer, 85,054,464 for 12 layers, 85,798,656 without a head); a starting loss of about 1.62 (ln 5) and 64 pictures overfitted to 1.00; TinyViT and the CNN both at 1.000 on `make_shapes`, read honestly as a tie on an easy task; a CLS attention map with about 0.33 of the attention on the four central patches against 0.25 if spread evenly; the Imagine-first guess compared with the count | Everything runs and the checks pass; the reading of the comparison with the CNN or of the attention map goes beyond the numbers or is brief | Biases, the final LayerNorm or the CLS position left out of the count; the sanity checks skipped; a ViT declared better than a CNN from an easy task; numbers from a shortened run quoted |
| Part 5: ViT experiments (positions, shuffling, pooling, patch size, data size, sine-cosine positions) | 14 | The left/right table with the CNN (1.000), the ViT with positions (0.993) and the ViT without (0.537, within noise of chance); the shuffle test with agreement 1.000 without positions and 0.513 with, explained by Part 3; CLS and mean pooling reported as a tie (0.993 each); the patch table (64, 16 and 4 tokens; 1.000, 0.993 and 0.860 on left/right) with the 14.6 times cost ratio and the plot; the data-size table with the CNN ahead at every size (0.887 against 0.430 at 50 pictures, 0.993 against 0.880 at 400) explained by inductive bias; sine-cosine positions at 1.000 and three specific answers on where 'where' matters | Everything runs and the asserts pass; the explanation of the shuffle test, the patch trade-off or the data hunger goes beyond the numbers or is brief | A difference of a few pictures called a winner; the no-position result explained as a bad model rather than an impossible task; the data-size experiment run with a different number of steps per size without saying so |
| Part 6: Contrastive learning and CLIP (the loss by hand, temperature, a toy CLIP with held-out pairs, the real CLIP and its limits) | 22 | `clip_loss_np` equal to `F.cross_entropy` (1.0481 for the example); the scale table with its U shape (0.952, 0.803, 1.048, 2.589, 8.302) explained; the toy CLIP trained with the held-out mask and its loss curve; 1.000 on seen and 0.394 on unseen combinations with the finding that the colour is always right and the shape fails for blue circles, and the reason; the real CLIP loaded (or the failure to load reported clearly), the accuracy per template (0.889, 0.889, 1.000), the 9 x 9 heat map and `logit_scale` of 100 as measured; the probes: the answer is always one of the labels given, counting fails, short words in pictures work; a specific paragraph on biases and what to check | Everything runs and the checks pass; the explanation of the temperature, of generalisation or of the biases is general | Real CLIP outputs copied from the lesson rather than measured; the loss written without the second direction; unseen accuracy reported without the seen accuracy or without the held-out check; the probabilities of a zero-shot classifier read as confidence |
| Part 7: Design a multimodal system, and reflect (design card, token budget, accuracy per group, RESULTS, reflection) | 10 | A design card with all eight fields for a real problem, a fusion pattern that is justified, an evaluation with a metric per modality and a human check, and a named risk; the token budget with 576 visual tokens, sequence 640 and the 13.7 times cost of halving the patch (or the same for the chosen values); `group_accuracy` and the table per colour (blue 0.680, green 0.920, red 1.000) with the worst group and a fairness check for the design; the printed `RESULTS`; four reflection answers that cite numbers from the lab | Everything runs and the checks pass; the design card or the fairness check is generic | A card with 'accuracy' as the only evaluation; only the mean accuracy reported; no consent or privacy line; reflection answers that do not use any number from the lab |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave:
1. How does a ViT turn a picture into something that a Transformer can read, and how many tokens does a 224 × 224 image give with 16 × 16 patches, including the CLS token?
2. Why can a ViT not tell a picture from its shuffled copy without position embeddings?
3. In CLIP, what is on the diagonal of the similarity matrix and what does zero-shot classification need from you?

**Homework / self-study**
- Finish the lab exercises and the Part 7 design card and reflection.
- Re-read the `equation` figures of this lesson (patch count, patch length, projection, position embeddings, cosine, contrastive loss) and redo each worked example by hand with a different image size or batch.
- Optional: read UvA Tutorial 15 (Vision Transformers) or Tutorial 17 (SimCLR) from the lab's “Go further” section.
- Preview Week 12: image generation and GANs. Think about how a network could make a brand-new picture from random numbers, with a generator that draws and a discriminator that judges.
