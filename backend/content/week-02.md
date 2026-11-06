> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 2 — Supervised Learning and Neural Networks

**Purpose:** See how a machine ==learns a rule from labelled examples==: a neuron makes a guess, a loss measures the mistake, and training nudges the weights until the mistakes get smaller.

**Term:** Prelim (Weeks 1–5) · **Month:** 1 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a network learn",
  "caption": "Four views of one idea: a single neuron turning numbers into a score, a layer of neurons doing the same thing side by side, the loss falling as training goes on, and the probabilities a trained network gives for a picture. The arithmetic in the first two scenes is exact; the training curve and the digit probabilities are made up for the animation.",
  "scenes": [
    { "name": "One neuron", "kind": "text", "outputLabel": "The neuron's output", "prompt": "inputs x = (2, 1), weights w = (0.5, −1), bias b = 0.25", "output": "z = 0.5×2 + (−1)×1 + 0.25 = 0.25, then sigmoid(0.25) = 0.56" },
    { "name": "A layer", "kind": "text", "outputLabel": "Two neurons at once", "prompt": "3 inputs x = (2, 1, 3) go to 2 neurons that each have their own weights", "output": "z = (0.85, −0.50), and after ReLU the layer outputs a = (0.85, 0)" },
    { "name": "Training", "kind": "loss", "outputLabel": "Loss while training", "prompt": "Show the network 1,000 labelled examples, again and again" },
    { "name": "After training", "kind": "bars", "outputLabel": "Probability for each digit (illustrative)", "prompt": "A new handwritten digit the network has never seen", "bars": [ { "label": "7", "value": 0.86 }, { "label": "1", "value": 0.09 }, { "label": "9", "value": 0.05 } ] }
  ]
}
```

## By the end, you can…

1. ==Explain supervised learning==: examples, features, labels, a model with parameters, a loss, training, and testing on new data.
2. ==Split data into training, validation and test sets== and say why the test set is used only once.
3. ==Compute a neuron's output by hand==: weighted sum, plus bias, then an activation function.
4. ==Describe a layer as φ(Wx + b)== and count the parameters of a small network.
5. Explain why a ==hidden layer and a non-linear activation are needed==, using XOR as the example.
6. Explain ==underfitting, overfitting and generalization==, and recognise each from the training and test scores.

Assessed in the Week 2 lab and quiz. Read each section, then play with the figure under it. The figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. Notes such as “Lab Part 3” tell you which part of the lab lets you try the idea for real. This is the week where you meet your first neural network, so go slowly: every formula comes with a picture and a worked example.

## Supervised learning: data, labels, model {rail=Supervised}

{lede} In [[supervised learning|Learning from examples where every input comes with the correct answer. The model learns a rule that maps inputs to answers, and is then used on inputs it has not seen.]] you show a computer many examples *with the right answer attached*, and it ==learns a rule that maps inputs to answers==. Then you use that rule on new inputs. This is the most common way machines learn today.

### An example from the school canteen

Imagine you run the school canteen and you must decide, every morning, how many rice meals to cook. Cook too few and students go hungry; cook too many and food is wasted. You have been writing notes for a week: whether it rained, how many students were present, and how many meals you actually sold.

| Day | Raining? | Students present | Meals sold |
|---|---|---|---|
| Monday | no | 412 | 380 |
| Tuesday | yes | 398 | 341 |
| Wednesday | no | 405 | 372 |
| Thursday | yes | 390 | 330 |
| Friday | no | 401 | 355 |

(These numbers are made up for teaching.) Each row is one [[example|One item in a data set: one row of the table, such as one day in the canteen notes, one email, or one photo.]]. The first columns, rain and students present, are what you know in the morning. The last column, meals sold, is the answer you wish you had known. A computer can look at many rows like these, find the pattern (“on rainy days fewer students buy rice”) and predict tomorrow’s number before the day starts. That is supervised learning in one picture.

Two more everyday cases. A spam filter learns from thousands of emails that people have already marked “spam” or “not spam”. A digit reader learns from thousands of pictures of handwritten digits, each tagged with the digit it shows (0 to 9). A picture of a digit is 28 × 28 = 784 small brightness numbers, so each example has 784 inputs and one correct answer.

### The words you need

Each part of the story has a name. Read the table once, then use it as a reference. We will keep these words all course.

| Word | Meaning | Canteen example | Spam filter example |
|---|---|---|---|
| Example | One item from the data | One day’s notes | One email |
| Feature | An input number the model sees | Raining (0 or 1), students present | Number of exclamation marks, count of the word “free” |
| Label | The correct answer for that example | Meals sold: 380 | “spam” |
| Model | A function with adjustable numbers inside | A formula that turns features into a guess | A small neural network |
| Parameters | The adjustable numbers inside the model | The weights and bias | The weights and biases |
| Prediction | The model’s answer for an example | “Cook 360 meals” | “92% likely spam” |

A [[feature|One measurable input to the model, such as the number of students present or the brightness of one pixel.]] is what the model is *given*. A [[label|The correct answer attached to a training example, such as the number of meals sold or the word spam. The label is the teacher.]] is what the model should *answer*. A [[model|A function with adjustable numbers inside. Given features it returns a prediction.]] is the machine in the middle, and its adjustable numbers are called [[parameters|The adjustable numbers inside a model, such as weights and biases. Training changes them to make predictions better.]]. The collection of all examples with their labels is the [[training set|The examples, with labels, that the model is allowed to learn from. Its parameters are adjusted using only these.]].

### Two kinds of answer

The label can be a number or a category, and that decides the kind of task. Switch between the two modes below and read the examples.

```viz
{
  "type": "flow",
  "title": "Regression and classification",
  "modes": [
    {
      "name": "Regression",
      "tagline": "The label is a number on a scale. The model predicts a quantity.",
      "inputLabel": "Features",
      "modelLabel": "model with parameters",
      "outputLabel": "Prediction: a number",
      "examples": [
        { "label": "Canteen", "input": "Rainy day, 398 students present", "output": "about 341 meals" },
        { "label": "Jeepney fare", "input": "Distance 8 km, rush hour", "output": "₱ 17 (an estimate)" },
        { "label": "Rainfall", "input": "Humidity, cloud cover, season", "output": "12 mm tomorrow" }
      ],
      "note": "The prediction can be any number, so we judge it by how far it is from the true number."
    },
    {
      "name": "Classification",
      "tagline": "The label is a category from a fixed list. The model predicts which one, usually with a probability for each.",
      "inputLabel": "Features",
      "modelLabel": "model with parameters",
      "outputLabel": "Prediction: a category",
      "examples": [
        { "label": "Spam filter", "input": "Text of an email", "output": "spam (92%)" },
        { "label": "Digit reader", "input": "A 28 × 28 picture", "output": "7 (86%)" },
        { "label": "Flood warning", "input": "Rainfall and river level", "output": "flood likely (73%)" }
      ],
      "note": "Output probabilities add up to 1. We judge the model by how much probability it gave to the correct class."
    }
  ]
}
```

Notice that regression answers “how much?” and classification answers “which one?”. Both are supervised, because both have a correct answer for every training example. If there are no labels at all and the model only looks for structure, such as groups of similar customers, that is *unsupervised* learning, which we do not cover this week.

### The learning loop

How does the model find good parameters? It does not know them at the start. It begins with a guess, checks the guess against the label, and improves, many times over. Step through the loop below.

```viz
{
  "type": "pipeline",
  "title": "The supervised learning loop",
  "intro": "Click the steps in order. Steps 2 to 5 repeat many times; step 6 happens once, at the end.",
  "nodes": [
    { "id": "data", "label": "Labelled data", "sub": "features and labels" },
    { "id": "model", "label": "Model", "sub": "makes a guess" },
    { "id": "loss", "label": "Loss", "sub": "how wrong?" },
    { "id": "adjust", "label": "Adjust", "sub": "change parameters" },
    { "id": "test", "label": "Test", "sub": "new, unseen data" }
  ],
  "steps": [
    { "title": "1. Collect labelled data", "body": "Gather examples with their correct answers, and set some aside for testing later. Good data matters more than a clever model.", "active": ["data"], "example": "Five days of canteen notes, each with the meals sold." },
    { "title": "2. The model guesses", "body": "The model turns the features of an example into a prediction using its current parameters. At the very start the parameters are random, so the guess is poor.", "active": ["data", "model"], "example": "Features: rainy, 398 students. Guess: 400 meals." },
    { "title": "3. Measure the loss", "body": "Compare the prediction with the label and turn the difference into one number, the loss. A small loss means a good guess.", "active": ["model", "loss"], "example": "Real answer 341. The guess was 59 too high, so the loss is large." },
    { "title": "4. Adjust the parameters", "body": "Change the parameters a little in the direction that makes the loss smaller. This is what we call training.", "active": ["loss", "adjust"], "example": "The weight for rain is made more negative, because rain should lower the number." },
    { "title": "5. Repeat", "body": "Go back to step 2 with the improved model. Over thousands of repeats the loss falls and the guesses improve.", "active": ["adjust", "model", "loss"], "example": "After many rounds the guesses are within about 10 meals." },
    { "title": "6. Test on new data", "body": "Finally, try the model on examples it never learned from, and measure how well it does. This tells you whether it has really learned the pattern or just memorised the training set.", "active": ["model", "test"], "example": "Next week’s notes: how close were the predictions?" }
  ]
}
```

Notice that the loop never needs a human to say “change this weight”. The loss tells the machine how wrong it is, and training does the rest. The next sections open each box: the loss (box 3), the model (box 2) and the adjusting (box 4). The test (box 6) comes last.

### How this differs from Week 1

In Week 1, a generative model also learned from data, and its loss was also a number that training made smaller. The difference is the *teacher*. Here every example has a ==correct answer attached, and the label is the teacher==. A spam filter does not make up new emails; it says whether an email is spam. (A language model can be trained with no human labels, because the “answer” for each position is simply the next word in the text. That is a clever way to get labels for free, and it is why the two weeks fit together.)

### Where labels come from, and why they matter

Labels do not appear by magic. A person (a teacher, a nurse, a customer pressing “report spam”) has to provide them, and that costs time. The model can only be as good as its labels. If labels are wrong, the model learns the mistakes. If the labellers were careless or unfair to one group, the model learns the unfairness too. ==A model learns exactly what the labels teach it, including their mistakes.== Keep this in mind; we return to it in the last section.

### Try it: name the parts

Put each card in the bin where it belongs. Two scenarios are mixed together: predicting a flat’s price and filtering spam.

```viz
{
  "type": "sorter",
  "title": "Feature, label, model or loss?",
  "intro": "Decide what role each item plays in a supervised learning problem. You get an explanation either way. One card is an edge case on purpose.",
  "bins": [
    { "id": "feature", "label": "Feature" },
    { "id": "label", "label": "Label" },
    { "id": "model", "label": "Model" },
    { "id": "loss", "label": "Loss" }
  ],
  "cards": [
    { "name": "Floor area of a flat in square metres", "answer": "feature", "why": "It is an input the model is given about each flat." },
    { "name": "Number of bedrooms", "answer": "feature", "why": "Another input number describing the flat. Features are known before the model answers." },
    { "name": "Number of exclamation marks in an email", "answer": "feature", "why": "A number measured from the email, given to the spam model as input." },
    { "name": "The price the flat really sold for", "answer": "label", "why": "It is the correct answer for that example, the thing the model tries to predict." },
    { "name": "The tag “spam” a person put on an email", "answer": "label", "why": "A human-provided correct answer. This is the teacher in supervised learning." },
    { "name": "price = w × area + b", "answer": "model", "why": "A formula with adjustable numbers w and b that turns features into a prediction." },
    { "name": "A small neural network that outputs P(spam)", "answer": "model", "why": "A model with many adjustable weights and biases." },
    { "name": "The weights w and b inside the price formula", "answer": "model", "why": "Edge case. They are the parameters of the model, so they belong with the model. Training changes them; they are not data." },
    { "name": "Average squared gap between predicted and real prices", "answer": "loss", "why": "A single number scoring the mistakes. This is mean squared error." },
    { "name": "−ln of the probability given to the true class", "answer": "loss", "why": "Cross-entropy, the usual loss for classification. Small when the model is confident and right." }
  ]
}
```

Check yourself on the bins you missed. The key split is simple: features go *in*, labels are the *right answer*, the model is the *machine with dials*, and the loss is the *score of its mistakes*. Lab Part 1 asks you to be the model yourself, with a table of mangoes, before any neural network appears.

## Loss: how wrong is the model? {rail=Loss}

{lede} A [[loss function|A rule that turns the model’s mistakes on the training data into one number. Smaller is better, and training tries to make it as small as possible.]] is ==a score for the model’s mistakes==: zero would mean perfect, and bigger means worse. Without a score the machine has no way to know whether a change to its parameters helped. Everything in training is about making this one number smaller.

### Why not just count the errors?

You could count how many predictions were wrong. That number is called the error rate, and it is a fine thing to *report*. But it is a poor thing to *train on*. If a model moves from “very wrong” to “nearly right” its error count may not change at all, so the machine gets no signal that it is going the right way. A good loss changes smoothly: a little better gives a little lower score. That smooth score is what lets the machine find the direction to improve.

### Regression: mean squared error

For numbers, the usual loss is the [[mean squared error (MSE)|The average of the squared differences between the predicted numbers and the true numbers. It is zero for perfect predictions and grows quickly for large mistakes.]]. For each example, take the prediction minus the true value, square it, then average over all examples. Select each part of the formula, then step through the worked example, which uses four flats (prices in millions of pesos, made up for teaching).

```viz
{
  "type": "equation",
  "title": "Mean squared error",
  "intro": "Select each part of the formula to see what it means, then read the worked example underneath.",
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
    { "id": "L", "name": "L (the loss)", "meaning": "One number that scores all the mistakes together. Smaller is better; 0 is perfect." },
    { "id": "n", "name": "1/n (the mean)", "meaning": "n is the number of examples. Multiplying by 1/n averages the squared errors, so the loss does not depend on how many examples you have." },
    { "id": "sum", "name": "Σ (sum)", "meaning": "Greek capital sigma. It means add up the quantity that follows, once for every example i = 1 to n." },
    { "id": "yhat", "name": "ŷᵢ (the prediction)", "meaning": "Read “y-hat”. The model’s prediction for example i. The hat means “estimated, not true”." },
    { "id": "y", "name": "yᵢ (the label)", "meaning": "The true answer for example i, taken from the training data." },
    { "id": "sq", "name": "² (the square)", "meaning": "Squaring makes every error positive, so over- and under-shooting do not cancel, and makes big errors count far more than small ones." }
  ],
  "steps": [
    { "title": "Errors (prediction − true)", "expr": "true prices 3, 5, 2, 4 and predictions 2.5, 5.5, 4, 3.5 give errors −0.5, +0.5, +2, −0.5", "note": "The flat that was guessed 2 million too high is the big miss." },
    { "title": "Square each error", "expr": "0.25, 0.25, 4, 0.25", "note": "The error of 2 becomes 4, sixteen times the size of the error of 0.5 once squared (0.25 × 16 = 4)." },
    { "title": "Add them up", "expr": "0.25 + 0.25 + 4 + 0.25 = 4.75", "note": "That is the Σ part." },
    { "title": "Divide by n = 4", "expr": "L = 4.75 / 4 = 1.1875", "note": "The unit is (million pesos) squared. Its square root, 1.09 million, is back in pesos and says the typical miss is about 1.1 million." }
  ],
  "note": "The flat prices are made-up teaching numbers."
}
```

Notice how one big miss (the flat off by 2) supplies 4 of the total 4.75. That is what we want: a model should be made to care most about its worst mistakes.

### Why we square the error

Squaring does two jobs. It removes the sign, so +2 and −2 are both bad instead of cancelling out. And it makes large errors count much more than small ones. Compare the squared error with the plain absolute error below. Drag across the plot and read both curves at an error of 2.

```viz
{
  "type": "curve",
  "title": "Squared error vs absolute error",
  "xRange": [-3, 3],
  "yRange": [0, 9],
  "series": [ { "fn": "squared", "label": "Squared error" }, { "fn": "abs", "label": "Absolute error" } ],
  "alt": "Two curves of the penalty against the error from minus 3 to 3. The squared error is a smooth U shape that reaches 9 at the edges. The absolute error is a V shape that reaches 3 at the edges.",
  "note": "At an error of 2 the squared penalty is 4 but the absolute penalty is only 2. At an error of 0.5 the squared penalty (0.25) is smaller than the absolute one (0.5)."
}
```

Squared error is gentle for tiny mistakes and harsh for big ones, and its curve is smooth at the bottom, which suits the downhill method you will meet in the Training section. In Lab Part 2 you will add one outlier to a data set and watch how the best answer moves under each loss.

### Classification: cross-entropy

For categories the model outputs a probability for each class, and the loss looks only at the probability it gave to the *true* class. Call that probability p. The [[cross-entropy loss|The usual loss for classification: minus the natural logarithm of the probability the model gave to the true class. It is small when that probability is high and very large when it is low.]] is L = −ln p. The “ln” is the natural logarithm, the button marked ln on a calculator.

```viz
{
  "type": "equation",
  "title": "Cross-entropy loss: minus ln p",
  "intro": "Select each part, then step through four different probabilities for the true class.",
  "formula": [
    { "t": "L", "id": "L" },
    " = −",
    { "t": "ln", "id": "ln" },
    " ( ",
    { "t": "p", "id": "p" },
    " )"
  ],
  "parts": [
    { "id": "L", "name": "L (the loss for one example)", "meaning": "How much this single prediction costs. For a whole data set you average L over the examples." },
    { "id": "ln", "name": "ln (natural logarithm)", "meaning": "A function that turns small probabilities into large numbers. ln 1 = 0, and ln gets more and more negative as its input approaches 0. The minus sign in front makes the loss positive." },
    { "id": "p", "name": "p (probability of the true class)", "meaning": "The probability the model gave to the correct answer. If the email is really spam and the model said 0.9, then p = 0.9. The probabilities given to the wrong classes do not appear." }
  ],
  "steps": [
    { "title": "Confident and right: p = 0.9", "expr": "L = −ln 0.9 = 0.105", "note": "A small cost. The model was nearly sure of the right answer." },
    { "title": "Unsure: p = 0.5", "expr": "L = −ln 0.5 = 0.693", "note": "A coin-flip guess between two classes costs about 0.69." },
    { "title": "Confident and wrong: p = 0.1", "expr": "L = −ln 0.1 = 2.303", "note": "More than twenty times the first case. The model gave the right answer only a 10% chance." },
    { "title": "Very confident and wrong: p = 0.01", "expr": "L = −ln 0.01 = 4.605", "note": "The cost keeps growing as p heads to 0. Confident wrong answers are punished hardest." }
  ],
  "note": "The probabilities are example values; the losses are exact."
}
```

Now look at the whole curve of −ln p, for p from almost 0 to 1.

```viz
{
  "type": "curve",
  "title": "The cross-entropy curve: −ln p",
  "xRange": [0.01, 1],
  "yRange": [0, 5],
  "series": [ { "fn": "neg_log", "label": "Loss −ln p" } ],
  "alt": "A curve that starts very high, about 4.6, when the probability p is 0.01, falls quickly to about 0.69 at p of 0.5 and reaches 0 at p of 1.",
  "note": "The horizontal axis is p, the probability the model gave to the correct class. The curve is exactly −ln p."
}
```

Notice the shape: it is steep on the left and flat on the right. Going from p = 0.01 to p = 0.1 saves you 2.3 of loss, but going from 0.9 to 1.0 saves only 0.1. This is why the model works hardest to fix the examples it is confidently wrong about.

For a whole data set you average: if the true-class probabilities for three examples were 0.9, 0.5 and 0.1, the losses are 0.105, 0.693 and 2.303, and the mean is (0.105 + 0.693 + 2.303) / 3 = 1.034. You have already met this in Week 1: a language model’s loss on the next word is exactly −ln of the probability it gave to the word that really came next. ==Cross-entropy is the loss behind both text generation and classification==, and the “probabilities” come from a function called softmax that you will compute in the Layers section. Lab Part 2 has you code both losses and compare them with PyTorch.

## One artificial neuron {rail=Neuron}

{lede} An [[artificial neuron|A tiny calculator that multiplies each input by a weight, adds the results and a bias, and then passes the total through an activation function to get one output.]] is ==the smallest building block of a neural network==. It takes some numbers in, weighs them, adds a bias, and decides how strongly to “fire”. A network is just many of these joined together, so understanding one neuron well is most of the work.

### A decision a jeepney driver makes

Think of a jeepney driver deciding whether to stop at a corner. Two things matter: how many people are waiting (x₁), and whether it is raining (x₂). Waiting passengers push him to stop, so that input matters a lot and in the “stop” direction. Rain pushes him to stop too, but less. Also, if the jeepney is already full, he is less willing to stop whatever he sees.

That is a neuron. The inputs are the facts. Each input has a [[weight|A number that says how much one input matters to a neuron. A positive weight pushes the neuron to fire; a negative weight pushes it not to fire.]], meaning how much he cares about that fact (and in which direction). There is also a [[bias|A number added to the weighted sum, independent of the inputs. It sets how easily the neuron fires; it shifts the threshold.]], his general readiness to stop, whatever he sees. He adds it all up and then applies an [[activation function|A function applied to the neuron’s weighted sum to produce its output. It decides how strongly the neuron fires and is what makes networks non-linear.]]: a gate that turns the total into a decision. Real neurons in the brain collect signals and fire when enough arrive, which inspired the idea, but an artificial neuron is only arithmetic.

### The formula, with a picture

A neuron does two steps. First it computes z, the weighted sum plus bias. Then it passes z through the activation φ (Greek letter phi) to give its output a. Select each symbol, then read the worked example, which is the same one you will code in Lab Part 3.

```viz
{
  "type": "equation",
  "title": "One neuron: z = Σ wx + b, then a = φ(z)",
  "intro": "Select each part of the formula to see what it means, then step through the worked example.",
  "formula": [
    { "t": "a", "id": "a" },
    " = ",
    { "t": "φ", "id": "phi" },
    " ( ",
    { "t": "z", "id": "z" },
    " )      where      ",
    { "t": "z", "id": "z" },
    " = ",
    { "t": "Σ", "id": "sigma" },
    " ",
    { "t": "w", "id": "w" },
    "ᵢ",
    { "t": "x", "id": "x" },
    "ᵢ  +  ",
    { "t": "b", "id": "b" }
  ],
  "parts": [
    { "id": "x", "name": "xᵢ (inputs)", "meaning": "The numbers coming in: features, or outputs of an earlier layer. The small i counts them: x₁, x₂, …" },
    { "id": "w", "name": "wᵢ (weights)", "meaning": "One weight per input. A big positive weight means this input strongly excites the neuron; a negative weight inhibits it; a weight near 0 means the input is ignored." },
    { "id": "b", "name": "b (bias)", "meaning": "A single number added to the sum. It moves the threshold: a high bias makes the neuron fire easily, a low one makes it hard to fire." },
    { "id": "sigma", "name": "Σ (sum)", "meaning": "Add up the products w₁x₁ + w₂x₂ + … over all inputs." },
    { "id": "z", "name": "z (pre-activation)", "meaning": "The raw total before the gate: the weighted sum plus the bias. It can be any number, positive or negative." },
    { "id": "phi", "name": "φ (activation function)", "meaning": "A gate that turns z into the neuron’s output. Common choices are step, sigmoid, tanh and ReLU, shown below." },
    { "id": "a", "name": "a (activation, the output)", "meaning": "What the neuron passes on. Its range depends on φ: sigmoid gives a number from 0 to 1." }
  ],
  "steps": [
    { "title": "Multiply each input by its weight", "expr": "x = (2, 1), w = (0.5, −1):  0.5 × 2 = 1,  (−1) × 1 = −1", "note": "The second input has a negative weight, so it pulls the total down." },
    { "title": "Add them and the bias", "expr": "z = 1 + (−1) + 0.25 = 0.25", "note": "The bias b = 0.25 nudges the total up a little." },
    { "title": "Apply a step gate", "expr": "step(0.25) = 1, because z > 0", "note": "A step gate fires fully (1) when z is above 0, otherwise stays at 0." },
    { "title": "Apply sigmoid instead", "expr": "σ(0.25) = 1 / (1 + e^(−0.25)) = 0.562", "note": "Sigmoid gives a smooth value between 0 and 1: here a little more than half-on." },
    { "title": "Or ReLU, or tanh", "expr": "ReLU(0.25) = 0.25,  tanh(0.25) = 0.245", "note": "Same z, different gates, different outputs." }
  ],
  "note": "The inputs and weights are made up; the arithmetic is exact."
}
```

Notice that the neuron has only *3 numbers it can learn here*: two weights and one bias. Training will change exactly these. Everything else is arithmetic you can do on a calculator.

### Four common activation functions

The activation is the gate at the end. Here are four, with their value at five test points (computed, rounded to three decimals).

| z | −2 | −1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| Step (1 if z > 0) | 0 | 0 | 0 | 1 | 1 |
| Sigmoid σ(z) = 1/(1 + e^(−z)) | 0.119 | 0.269 | 0.500 | 0.731 | 0.881 |
| Tanh | −0.964 | −0.762 | 0.000 | 0.762 | 0.964 |
| ReLU = max(0, z) | 0 | 0 | 0 | 1 | 2 |

The step is the original idea: fully on or fully off, but with no smooth in-between, which makes it hard to train. [[Sigmoid|An S-shaped activation, 1/(1+e^(−z)), that squeezes any number into the range 0 to 1. Often used when the output is a probability.]] is a smooth version of the step. [[Tanh|An S-shaped activation that squeezes any number into the range −1 to 1 and is centred at 0.]] is similar but runs from −1 to 1. [[ReLU|The rectified linear unit, max(0, z). It outputs 0 for negative z and passes positive z through unchanged. It is the most popular activation inside deep networks.]] simply cuts off the negatives and passes positives through. Plot three of them together and slide across the horizontal axis.

```viz
{
  "type": "curve",
  "title": "Sigmoid, tanh and ReLU",
  "xRange": [-5, 5],
  "yRange": [-1.2, 2],
  "series": [ { "fn": "sigmoid", "label": "Sigmoid" }, { "fn": "tanh", "label": "Tanh" }, { "fn": "relu", "label": "ReLU" } ],
  "alt": "Three curves from minus 5 to 5. Sigmoid rises from 0 to 1 through 0.5 at zero. Tanh rises from minus 1 to 1 through 0 at zero. ReLU is flat at 0 for negative inputs and then rises as a straight line with slope 1.",
  "note": "The step function is not drawn: it is a sudden jump from 0 to 1 at z = 0."
}
```

Notice that sigmoid and tanh flatten out at both ends (a big change in z hardly changes the output), while ReLU keeps growing for positive z. Each shape suits a different job: sigmoid for “a probability of yes”, ReLU for the middle of a deep network. Lab Part 3 plots all four on the same axes.

### What weights and bias do

==The weights decide what the neuron looks at; the bias decides how easily it fires.== Weights have a sign and a size. A positive weight excites, a negative weight inhibits, and the bigger the absolute value, the more that input matters. A weight of 0 means the input is ignored.

The bias shifts the threshold. Take one input, weight w = 2, and a step gate. The neuron fires when 2x + b > 0, which means x > −b/2.

| Bias b | Fires when | Reading |
|---|---|---|
| 0 | x > 0 | fires for any positive input |
| −4 | x > 2 | harder to fire: needs x above 2 |
| +2 | x > −1 | easier to fire: even a slightly negative x is enough |

A negative bias makes the neuron picky and a positive bias makes it eager. In Lab Part 3 you will move a decision line by changing b and see exactly this.

### A neuron draws a straight line

With two inputs, the neuron fires when w₁x₁ + w₂x₂ + b > 0. The edge between “fires” and “does not fire” is where this equals 0, and that is a *straight line* on the page (the decision boundary). The weights tilt the line and the bias slides it. Try it: drag the sliders in the figure and see how many points you can place on the correct side. The AND tab shows a hand-built answer; the separable tab can be separated; the XOR tab cannot.

```viz
{
  "type": "neuron",
  "title": "One neuron, one straight line",
  "intro": "Move w₁, w₂ and b to place the line. Circles should be on the shaded side and squares on the other. Then switch to the XOR tab.",
  "w": [1, 1],
  "b": 0,
  "range": [-3, 3],
  "datasets": [
    { "name": "Separable", "points": [ { "x": 1.5, "y": 1, "c": 1 }, { "x": 1, "y": 2, "c": 1 }, { "x": 2, "y": 0.5, "c": 1 }, { "x": 0.5, "y": 1.5, "c": 1 }, { "x": 2, "y": -0.5, "c": 1 }, { "x": -1.5, "y": -1, "c": 0 }, { "x": -1, "y": -2, "c": 0 }, { "x": -2, "y": 0.5, "c": 0 }, { "x": -0.5, "y": -1.5, "c": 0 }, { "x": 0.5, "y": -2, "c": 0 }, { "x": -1.8, "y": 0.2, "c": 0 } ] },
    { "name": "AND", "points": [ { "x": 0, "y": 0, "c": 0 }, { "x": 0, "y": 1, "c": 0 }, { "x": 1, "y": 0, "c": 0 }, { "x": 1, "y": 1, "c": 1 } ] },
    { "name": "XOR", "points": [ { "x": 1.5, "y": -1.5, "c": 1 }, { "x": -1.5, "y": 1.5, "c": 1 }, { "x": 1.5, "y": 1.5, "c": 0 }, { "x": -1.5, "y": -1.5, "c": 0 } ] }
  ],
  "note": "For AND, try w = (1, 1) and b = −1.5. On the XOR tab, no line gets more than 3 of the 4 points right."
}
```

The AND rule (“both inputs on”) works with w = (1, 1), b = −1.5: only (1, 1) gives 1 + 1 − 1.5 = 0.5 > 0. OR (“at least one on”) works with the same weights and b = −0.5. But for XOR (“exactly one on”), try as you might, one point is always on the wrong side. Hold that thought: it is the puzzle of the next two sections.

### Check: can one neuron do it?

Sort each task by whether one neuron (one straight line) can do it.

```viz
{
  "type": "sorter",
  "title": "Is one straight line enough?",
  "intro": "Think about each task as points on a plane. Can a single line put the 1s on one side and the 0s on the other?",
  "bins": [
    { "id": "yes", "label": "One neuron is enough" },
    { "id": "no", "label": "Needs more than one" }
  ],
  "cards": [
    { "name": "AND: 1 only if both inputs are 1", "answer": "yes", "why": "Only the point (1, 1) is on the 1 side, and a line can cut that corner off." },
    { "name": "OR: 1 if at least one input is 1", "answer": "yes", "why": "Only (0, 0) is on the 0 side, so a line can cut that one corner." },
    { "name": "NAND: the opposite of AND", "answer": "yes", "why": "Flip the sign of every weight and bias of the AND neuron." },
    { "name": "Is x₁ bigger than x₂?", "answer": "yes", "why": "The boundary x₁ = x₂ is a straight line, so the weights (1, −1) and bias 0 work." },
    { "name": "XOR: 1 if exactly one input is 1", "answer": "no", "why": "The 1s sit at opposite corners. Any line that has one on its 1 side also catches one of the 0s, or misses the other 1." },
    { "name": "XNOR: 1 if the inputs are equal", "answer": "no", "why": "The mirror image of XOR: the 1s are at opposite corners, so no line works." },
    { "name": "Is the point inside a circle?", "answer": "no", "why": "A circle is a curved boundary. One line cannot enclose a region." },
    { "name": "Two interlocking half-moons", "answer": "no", "why": "The classes curve around each other, so a straight line misclassifies some points on each side." }
  ]
}
```

## Layers and stacking {rail=Layers}

{lede} A [[layer|A group of neurons that all receive the same inputs side by side. Each neuron has its own weights and bias, and together they produce a vector of outputs.]] is ==many neurons side by side, all looking at the same inputs, each with its own weights==. Stack layers one after another, with each layer feeding the next, and you have a neural network.

### From one neuron to a layer

One neuron looks for one pattern. To look for several patterns at once, use several neurons. Suppose a flood-warning system gets three numbers: rainfall, river level and soil wetness. One neuron could check “heavy rain and high river”. Another could check “wet soil and high river”. Both read the same three inputs but weigh them differently. Their outputs together make the layer’s output.

Writing each neuron out is slow, so mathematicians pack all the weights into a table called a matrix W, with one row per neuron. If there are three inputs and two neurons, W has 2 rows and 3 columns. The inputs form a column x and the biases a column b, and the whole layer is computed at once. The computer is very fast at this; it is what the GPU is for.

### The layer formula, with every number

Select each part, then read the worked example: three inputs, two neurons, ReLU gate (this is the example from the hook, and Lab Part 4 has you code it).

```viz
{
  "type": "equation",
  "title": "A layer: a = φ(Wx + b)",
  "intro": "Select each part to see its shape, then step through a full 3-input, 2-neuron example.",
  "formula": [
    { "t": "a", "id": "a" },
    " = ",
    { "t": "φ", "id": "phi" },
    " ( ",
    { "t": "W", "id": "W" },
    { "t": "x", "id": "x" },
    " + ",
    { "t": "b", "id": "b" },
    " )"
  ],
  "parts": [
    { "id": "x", "name": "x (input vector)", "meaning": "A column of the n numbers going in. Shape: n × 1. Here n = 3." },
    { "id": "W", "name": "W (weight matrix)", "meaning": "One row per neuron, one column per input. Shape: out × in. Row j holds the weights of neuron j. Here it is 2 × 3, so six weights." },
    { "id": "b", "name": "b (bias vector)", "meaning": "One bias per neuron. Shape: out × 1. Here 2 numbers." },
    { "id": "phi", "name": "φ (activation)", "meaning": "Applied to each entry of Wx + b separately, using the same gate for the whole layer." },
    { "id": "a", "name": "a (layer output)", "meaning": "One output per neuron: a column of shape out × 1. It becomes the input of the next layer." }
  ],
  "steps": [
    { "title": "Set up the numbers", "expr": "x = (2, 1, 3);  neuron 1 weights (0.5, −1, 0.2), bias 0.25;  neuron 2 weights (−0.3, 0.8, 0.1), bias −1.0", "note": "W is 2 × 3: each row is one neuron’s weights." },
    { "title": "Neuron 1: z₁", "expr": "z₁ = 0.5×2 + (−1)×1 + 0.2×3 + 0.25 = 1 − 1 + 0.6 + 0.25 = 0.85", "note": "This is the same recipe as the single neuron." },
    { "title": "Neuron 2: z₂", "expr": "z₂ = (−0.3)×2 + 0.8×1 + 0.1×3 + (−1.0) = −0.6 + 0.8 + 0.3 − 1.0 = −0.50", "note": "Its bias is strongly negative, so the total is below 0." },
    { "title": "Apply ReLU to each", "expr": "a = (max(0, 0.85), max(0, −0.50)) = (0.85, 0)", "note": "Neuron 2 is silent; neuron 1 passes 0.85 to the next layer." },
    { "title": "Count the parameters", "expr": "weights 2 × 3 = 6, biases 2, total 8", "note": "Always: inputs × outputs weights, plus one bias per output." }
  ],
  "note": "The inputs and weights are made up; the arithmetic is exact."
}
```

Notice that the matrix does nothing new: it is the single-neuron recipe repeated for each row. Check the shapes whenever you build a layer: W must be (neurons × inputs), or the multiplication Wx cannot be done.

### Counting parameters

The number of parameters tells you how big and how expensive a network is. For one layer with *in* inputs and *out* neurons: *in × out* weights plus *out* biases. Add up the layers. Here is a tiny network with layers of sizes [2, 3, 1] (2 inputs, 3 hidden neurons, 1 output):

| Layer | Shape | Weights | Biases | Parameters |
|---|---|---|---|---|
| 1 | 2 → 3 | 2 × 3 = 6 | 3 | 9 |
| 2 | 3 → 1 | 3 × 1 = 3 | 1 | 4 |
| **Total** | | | | **13** |

Now scale up. A network that reads a 28 × 28 digit picture (784 inputs), has a hidden layer of 128 neurons and gives 10 outputs, one per digit, has 784 × 128 + 128 = 100,480 parameters in the first layer and 128 × 10 + 10 = 1,290 in the second, for 101,770 in total. Open the figure below and move the sliders to see how fast the count grows.

```viz
{
  "type": "mlp",
  "title": "Counting parameters",
  "intro": "Drag the width of the hidden layer and watch the total change. Start at 128 and confirm the total is 101,770.",
  "layers": [784, 128, 10],
  "note": "Weights are in × out per layer, plus one bias per output neuron. This is only the count: it does not say how good the network is."
}
```

Notice that nearly all the parameters sit in the first layer, because it connects every one of 784 pixels to every hidden neuron. Doubling the hidden width to 256 roughly doubles the total. In Lab Part 4 you count parameters in code and check against PyTorch.

### Input, hidden and output layers

A network has three kinds of layer. The *input layer* is just the numbers you feed in (no weights of its own). The *hidden layers* are everything in the middle: their outputs are not shown to the outside, which is why they are called hidden. The *output layer* gives the final answer. A network with several hidden layers is called a deep network, and that is where the phrase “deep learning” comes from: ==“deep” just means “many layers”==.

### A network you can click

The network below takes two inputs (rainfall and river level, each scaled from 0 to 1), passes them through a hidden layer of three ReLU neurons, and gives a flood risk between 0 and 1 from a sigmoid output. Click any circle to see its z and a, then change the input presets.

```viz
{
  "type": "network",
  "title": "A small flood-risk network [2, 3, 1]",
  "intro": "Click each circle and read the numbers z and a. Then choose a different input preset and watch the values change layer by layer.",
  "layers": [2, 3, 1],
  "weights": [ [ [1.5, 0.5], [-1, 1], [0.5, -0.5] ], [ [1.2, 0.8, -0.6] ] ],
  "biases": [ [-0.5, 0, 0.2], [-0.7] ],
  "inputs": [1, 0.5],
  "activation": "relu",
  "outputActivation": "sigmoid",
  "inputLabels": ["rain", "river"],
  "outputLabels": ["flood risk"],
  "presets": [
    { "name": "heavy rain, mid river", "inputs": [1, 0.5] },
    { "name": "dry", "inputs": [0, 0] },
    { "name": "heavy rain, high river", "inputs": [1, 1] },
    { "name": "light rain, high river", "inputs": [0.2, 0.9] }
  ],
  "editable": true,
  "note": "The weights are authored for teaching, not learned from real flood data.",
  "checks": [
    { "inputs": [1, 0.5], "output": [0.6295] },
    { "inputs": [0, 0], "output": [0.3058] },
    { "inputs": [1, 1], "output": [0.7271] },
    { "inputs": [0.2, 0.9], "output": [0.5399] }
  ]
}
```

With the first preset (rain 1, river 0.5) the hidden neurons get z = (1.25, −0.5, 0.45). ReLU keeps 1.25 and 0.45 and turns −0.5 into 0, so the hidden layer outputs (1.25, 0, 0.45). The output neuron then computes z = 1.2×1.25 + 0.8×0 + (−0.6)×0.45 − 0.7 = 0.53, and sigmoid(0.53) = 0.63. So the network says 63% flood risk. Try to find an input that makes the first hidden neuron silent.

### What the last layer should be

The hidden layers can use ReLU or tanh. The output layer depends on the job, because the answer must be in the right form.

| Job | Output layer | Why |
|---|---|---|
| Predict a number (price, temperature) | Linear: no gate, just z | Any real number is allowed |
| Yes or no (spam, flood) | One neuron with sigmoid | A single probability from 0 to 1 |
| One of many classes (digits 0–9) | One neuron per class with softmax | A probability for each class, adding to 1 |

[[Softmax|A function that turns a list of scores into probabilities: each score is exponentiated and divided by the sum, so the results are positive and add up to 1.]] takes the raw scores (called logits) from the last layer and makes them into probabilities. Select the parts and work through the three-class example.

```viz
{
  "type": "equation",
  "title": "Softmax: scores to probabilities",
  "intro": "Select each part, then follow the example with three raw scores.",
  "formula": [
    { "t": "pᵢ", "id": "p" },
    " = ",
    { "t": "e^zᵢ", "id": "num" },
    " / ",
    { "t": "Σⱼ e^zⱼ", "id": "den" }
  ],
  "parts": [
    { "id": "p", "name": "pᵢ (probability of class i)", "meaning": "The result: a number between 0 and 1 for class i. All the pᵢ together add up to 1." },
    { "id": "num", "name": "e^zᵢ (exponential of the score)", "meaning": "Makes every score positive and makes bigger scores count disproportionately more. e is about 2.718." },
    { "id": "den", "name": "Σⱼ e^zⱼ (the total)", "meaning": "Add e^z over all classes. Dividing by it scales the numbers so they sum to 1." }
  ],
  "steps": [
    { "title": "Raw scores (logits)", "expr": "z = (2.0, 1.0, 0.1) for classes A, B, C", "note": "These are just numbers out of the last layer; they are not probabilities yet." },
    { "title": "Exponentiate", "expr": "e² = 7.389,  e¹ = 2.718,  e^0.1 = 1.105", "note": "All positive now." },
    { "title": "Add them up", "expr": "7.389 + 2.718 + 1.105 = 11.212", "note": "The denominator." },
    { "title": "Divide", "expr": "p = (0.659, 0.242, 0.099)", "note": "They add up to 1. Class A wins with 65.9%. The loss, if A were the true class, is −ln 0.659 = 0.417." }
  ],
  "note": "The scores are example values; the arithmetic is exact."
}
```

This is exactly the softmax you used in Week 1 to turn a language model’s scores into next-word probabilities (there with a temperature). Here a digit reader uses it to turn ten scores into ten probabilities, and the loss −ln p looks only at the true digit’s probability.

## Why we need hidden layers and non-linearity {rail=Hidden layers}

{lede} A single neuron can only draw a straight line, but real problems need curves. ==A hidden layer with a non-linear activation lets a network bend the boundary== so it can solve problems that no single neuron can. The classic example is XOR.

### The XOR puzzle

XOR (“exclusive or”) is the rule: output 1 when *exactly one* of the two inputs is 1. The four cases are (0, 0) → 0, (0, 1) → 1, (1, 0) → 1, (1, 1) → 0. Plot them on a plane: the two 1s sit at opposite corners and the two 0s at the other corners. No straight line can put the 1s on one side and the 0s on the other. In 1969 Minsky and Papert, in their book *Perceptrons*, pointed out this limit of single-layer perceptrons, and it is often told as a story of why interest in neural networks cooled for a time. ==The limit is real for one neuron; the cure is a hidden layer.==

```viz
{
  "type": "timeline",
  "title": "A short history of the neuron",
  "events": [
    { "year": "1958", "title": "The perceptron", "text": "Frank Rosenblatt introduces the perceptron, a single artificial neuron that can learn weights from examples." },
    { "year": "1969", "title": "The XOR limit", "text": "Minsky and Papert’s book Perceptrons stresses that a single-layer perceptron cannot compute XOR. The result is a milestone often linked to a slowdown in neural network research." },
    { "year": "1986", "title": "Backpropagation popularised", "text": "Rumelhart, Hinton and Williams show how to train networks with hidden layers by computing gradients layer by layer (backpropagation)." },
    { "year": "1998", "title": "LeNet reads digits", "text": "LeCun and colleagues train a network to read handwritten digits, an early practical success." },
    { "year": "2012", "title": "AlexNet", "text": "Krizhevsky and colleagues win an image-recognition contest with a deep network, and deep learning takes off." }
  ]
}
```

### A hand-built XOR network

You do not need training to see the fix. Here is a network with two hidden neurons, built by hand. Hidden neuron 1 is an OR gate: it fires if *at least one* input is 1 (weights 20 and 20, bias −10). Hidden neuron 2 is an AND gate: it fires only if *both* are 1 (weights 20 and 20, bias −30). The output neuron says “OR but not AND” (weight 20 from OR, weight −20 from AND, bias −10). Big weights make the sigmoids behave almost like on/off switches. Use the presets to try all four inputs and click the hidden circles.

```viz
{
  "type": "network",
  "title": "XOR by hand: OR, AND, then OR-but-not-AND",
  "intro": "Choose each of the four inputs. Click both hidden circles and the output, and read z and a.",
  "layers": [2, 2, 1],
  "weights": [ [ [20, 20], [20, 20] ], [ [20, -20] ] ],
  "biases": [ [-10, -30], [-10] ],
  "inputs": [1, 0],
  "activation": "sigmoid",
  "outputActivation": "sigmoid",
  "inputLabels": ["x1", "x2"],
  "outputLabels": ["XOR"],
  "presets": [
    { "name": "0, 0", "inputs": [0, 0] },
    { "name": "0, 1", "inputs": [0, 1] },
    { "name": "1, 0", "inputs": [1, 0] },
    { "name": "1, 1", "inputs": [1, 1] }
  ],
  "editable": true,
  "note": "Weights authored by hand for teaching; no training was used. Outputs are very close to 0 or 1, not exactly equal.",
  "checks": [
    { "inputs": [0, 0], "output": [0] },
    { "inputs": [0, 1], "output": [1] },
    { "inputs": [1, 0], "output": [1] },
    { "inputs": [1, 1], "output": [0] }
  ]
}
```

Here is what the hidden layer does, rounded to the nearest whole number:

| Input (x₁, x₂) | OR neuron | AND neuron | Output (OR and not AND) |
|---|---|---|---|
| (0, 0) | 0 | 0 | 0 |
| (0, 1) | 1 | 0 | 1 |
| (1, 0) | 1 | 0 | 1 |
| (1, 1) | 1 | 1 | 0 |

The hidden layer has *re-coded* the problem. In the new (OR, AND) coordinates, the two 1s are at (1, 0) and the 0s at (0, 0) and (1, 1): these can be separated by a straight line, which the output neuron draws. ==A hidden layer re-describes the inputs so that the next layer finds the problem easy.== That is the secret of deep learning: each layer builds a better description for the next. You will build this very network in Lab Part 4 and plot the hidden space.

### Without a non-linear gate, depth is useless

It is not enough to just stack layers. Suppose every layer just multiplies by a matrix and adds nothing else (a linear activation). Then two layers W₁ and W₂ give W₂(W₁x), and by the rule for matrix multiplication that equals (W₂W₁)x, which is one single matrix. ==Stacked linear layers collapse into one linear layer==, and so they can still draw only a straight line. The non-linear gate between layers (sigmoid, tanh, ReLU) is what stops the collapse.

```viz
{
  "type": "equation",
  "title": "Two linear layers collapse into one",
  "intro": "Select each part, then check the worked example with real numbers.",
  "formula": [
    { "t": "W₂", "id": "W2" },
    " ( ",
    { "t": "W₁", "id": "W1" },
    { "t": "x", "id": "x" },
    " )  =  ( ",
    { "t": "W₂W₁", "id": "W" },
    " ) ",
    { "t": "x", "id": "x" }
  ],
  "parts": [
    { "id": "x", "name": "x (the input)", "meaning": "A column of numbers going into the first layer." },
    { "id": "W1", "name": "W₁ (first layer)", "meaning": "The weight matrix of layer 1. With no activation, layer 1 just computes W₁x." },
    { "id": "W2", "name": "W₂ (second layer)", "meaning": "The weight matrix of layer 2, applied to the output of layer 1." },
    { "id": "W", "name": "W₂W₁ (one combined layer)", "meaning": "Multiply the two matrices once. One layer with this matrix does exactly the same job as the two." }
  ],
  "steps": [
    { "title": "Pick small matrices", "expr": "W₁ has rows (1, 2) and (0, 1);  W₂ has rows (2, 0) and (1, 1);  x = (1, 2)", "note": "Two layers, no activation." },
    { "title": "Go through both layers", "expr": "W₁x = (1×1 + 2×2, 0×1 + 1×2) = (5, 2);  W₂(5, 2) = (2×5 + 0×2, 1×5 + 1×2) = (10, 7)", "note": "The result of the two-layer network." },
    { "title": "Combine the matrices first", "expr": "W₂W₁ has rows (2, 4) and (1, 3);  (W₂W₁)x = (2×1 + 4×2, 1×1 + 3×2) = (10, 7)", "note": "The same answer from one single layer. Depth added nothing." }
  ],
  "note": "The matrices are made up; the arithmetic is exact."
}
```

Lab Part 4.3 asks you to prove this numerically for random matrices.

### Watch a network learn

Now see a network find these weights by itself. The figure below is a small network, 2 inputs, hidden tanh neurons and 1 sigmoid output, that trains in your browser. It starts with one hidden neuron on the XOR data. Press **Train** and watch the coloured background (the model’s prediction everywhere) and the loss line. Circles are class 1, squares are class 0.

```viz
{
  "type": "trainer",
  "title": "Train a tiny network",
  "intro": "Pick a data set, the number of hidden units and a learning rate, then press Train. Reset before changing the hidden units.",
  "dataset": "xor",
  "hidden": 1,
  "lr": 0.5,
  "note": "Measured: after 1000 epochs at a learning rate of 0.5, XOR reaches only 75% accuracy with 1 hidden unit and 100% with 2 or more. Circle and moons reach 100% with 4 hidden units.",
  "checks": [
    { "dataset": "xor", "hidden": 1, "lr": 0.5, "epochs": 1000, "maxAcc": 0.8 },
    { "dataset": "xor", "hidden": 2, "lr": 0.5, "epochs": 1000, "minAcc": 0.95 },
    { "dataset": "circle", "hidden": 4, "lr": 0.5, "epochs": 1000, "minAcc": 0.95 },
    { "dataset": "moons", "hidden": 4, "lr": 0.5, "epochs": 1000, "minAcc": 0.95 }
  ]
}
```

Try this guided experiment. Each run is about 1000 epochs, which takes a few seconds.

1. Keep XOR and **1 hidden unit**. Train. Accuracy stalls at 75%: the network can get three of the four corners right but never all four, just like a single line.
2. Press Reset, set **2 hidden units**, and train. Accuracy reaches 100% and the background bends into two bands. Two hidden units are enough to bend the boundary.
3. Switch to **circle** and try 1, 2 and then 4 hidden units. With 1 the accuracy is about 73%, with 2 about 85%, and with 4 it reaches 100%. The more hidden units, the more complex a shape the network can draw.
4. Switch to **moons** and use 4 hidden units: 100%.
5. Go back to XOR with 4 hidden units and a **very small learning rate** (the slider’s lowest value, 0.05). After 300 epochs accuracy is only about 67% and it needs about 1000 epochs to reach 100%. A larger rate learns faster: at 0.2 it is already at 98% after 100 epochs. (More about the learning rate in the Training section.)

A short hint about why this works so widely: a network with one hidden layer that is wide enough can, *in principle*, approximate any smooth function to any accuracy you like. This is called the universal approximation idea. It says what is possible, not how easy it is to find the weights or how much data you need, but it explains why hidden layers are so powerful.

## Learning the weights {rail=Training}

{lede} So far we have chosen weights by hand. Real networks have thousands to billions of weights, so we need a method that finds good values automatically. The method is ==guess, measure the loss, and nudge every weight a little in the direction that lowers the loss==, repeated many times. It is called gradient descent.

### The three-line recipe

Training any neural network follows the same three lines, over and over:

1. **Guess.** Run the inputs through the network with its current weights to get predictions.
2. **Measure.** Compute the loss: how wrong were the predictions?
3. **Nudge.** Change each weight by a small amount in the direction that makes the loss go down.

Imagine you are on a foggy hillside in Baguio and want to reach the valley, but you can only see your feet. You feel which way the ground slopes down and take a small step that way. Repeat. You do not need a map; you only need the slope where you stand. The loss is the height of the ground, the weights are your position, and training is walking downhill. The slope of the loss with respect to a weight is called the [[gradient|How fast the loss changes when one weight changes. The gradient points uphill, so training moves the weight the opposite way.]].

### One weight, one hill

The simplest case has one weight. The loss then is a curve over that single weight, and the best weight is the bottom of the valley. In the figure the loss is (w − 3)², so the valley bottom is at w = 3. Press play or step, and move the learning-rate slider.

```viz
{
  "type": "descent",
  "title": "Rolling down a one-weight loss",
  "fn": "quad",
  "lr": 0.1,
  "start": -0.5
}
```

The step size is controlled by the [[learning rate|A small number, written η, that scales each weight update. Too small and learning crawls; too large and the weight overshoots the valley and may never settle.]], written η (Greek letter eta). Try three experiments:

- **Too small** (for example 0.02): the point creeps down the hill. It gets there, but only after very many steps.
- **About right** (for example 0.1 to 0.5): it slides into the valley in a handful of steps.
- **Too large** (1 or more): the point overshoots to the other side, bounces back and forth, and above 1 it climbs *away* from the valley and the loss grows.

For this particular curve, a rate below 1 converges, exactly 1 bounces forever, and above 1 diverges. Other problems have other limits, which is why choosing η is a real skill. Lab Part 5 has you find the limit for a different curve.

### The update rule, with a fully worked step

Each nudge follows one rule: new weight equals old weight minus η times the gradient. The minus sign is there because the gradient points uphill and we want to go downhill. Select each symbol, then step through a real example: a neuron with one weight and no bias, ŷ = w × x, and one training example, x = 2 with label y = 4. The ideal weight is therefore 2. We start at w = 0.5 and use η = 0.1.

```viz
{
  "type": "equation",
  "title": "The update rule: w ← w − η × gradient",
  "intro": "Select each part, then step through one full training step on a one-weight model.",
  "formula": [
    { "t": "w", "id": "w" },
    " ← ",
    { "t": "w", "id": "w" },
    " − ",
    { "t": "η", "id": "eta" },
    " × ",
    { "t": "∂L/∂w", "id": "grad" }
  ],
  "parts": [
    { "id": "w", "name": "w (the weight)", "meaning": "The parameter being trained. The arrow ← means “replace the old value with the new value on the right”." },
    { "id": "eta", "name": "η (learning rate)", "meaning": "Eta. A small positive number that sets the size of each step. It is chosen by you, not learned." },
    { "id": "grad", "name": "∂L/∂w (the gradient)", "meaning": "How much the loss L changes when w changes by a tiny amount. Positive means the loss rises as w rises, so w should go down; negative means the opposite." }
  ],
  "steps": [
    { "title": "1. Guess", "expr": "ŷ = w × x = 0.5 × 2 = 1", "note": "The model predicts 1, but the label is 4." },
    { "title": "2. Measure the loss", "expr": "L = (ŷ − y)² = (1 − 4)² = 9", "note": "A big loss, because the prediction is far too low." },
    { "title": "3. Find the gradient", "expr": "∂L/∂w = 2 (ŷ − y) x = 2 × (−3) × 2 = −12", "note": "Negative: the loss falls if w goes up. The chain rule gives 2(ŷ − y) for the square and x for how ŷ depends on w." },
    { "title": "4. Update", "expr": "w ← 0.5 − 0.1 × (−12) = 0.5 + 1.2 = 1.7", "note": "The weight moved up from 0.5 to 1.7, toward the ideal 2." },
    { "title": "5. Check the new loss", "expr": "ŷ = 1.7 × 2 = 3.4;  L = (3.4 − 4)² = 0.36", "note": "The loss fell from 9 to 0.36 in a single step. A second step gives w = 1.94, and a third 1.988." }
  ],
  "note": "The numbers are made up for teaching; the arithmetic is exact."
}
```

Notice how the step shrinks by itself: the gradient was −12 at first, then −2.4, then −0.48. Near the bottom the slope is gentle, so the steps become small. For this problem each step multiplies the distance to the ideal weight by (1 − 8η). With η = 0.1 that factor is 0.2, so the distance 1.5 becomes 0.3, then 0.06. The training is stable only while the factor stays between −1 and 1, so η must be below 2/8 = 0.25 for this example. Lab Part 5.1 reproduces this worked step and Part 5.2 finds the limit by experiment.

### Epochs, batches and backpropagation

Real training has a few more words. One full pass through the training set is an [[epoch|One complete pass through all the training examples. Training usually runs for many epochs.]]. With big data sets it is slow to compute the loss on every example for each nudge, so we often use a small random [[batch|A small group of training examples used together to compute one loss and one weight update.]] of examples per update. The idea does not change.

A network has many weights, in many layers. To nudge each one we need its gradient. [[Backpropagation|The algorithm that computes the gradient of the loss with respect to every weight, working backward from the output through the layers using the chain rule.]] is the bookkeeping that computes all those gradients efficiently, working backwards from the output. You will not do it by hand here: in PyTorch, one call, `loss.backward()`, does it for you. (Lab Part 5.4 shows a hand-written version you can read and run, but you do not need to memorise it.)

### The same recipe in PyTorch

Here is the whole idea in a few lines of PyTorch. This is for reading now; you will run it in Lab Part 6. It trains the small network from the trainer figure.

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(1000):
    p = model(X)              # 1. guess: X has shape (N, 2)
    loss = loss_fn(p, y)      # 2. measure: y has shape (N, 1), values 0 or 1
    opt.zero_grad()           # clear the old gradients
    loss.backward()           # backpropagation: gradient of loss for every weight
    opt.step()                # 3. nudge: w <- w - lr * gradient
```

Read it line by line:

- `nn.Sequential(...)` stacks layers in order. `nn.Linear(2, 4)` is a layer with 2 inputs and 4 neurons (2 × 4 + 4 = 12 parameters), `nn.Tanh()` is the gate, `nn.Linear(4, 1)` is the output layer (4 + 1 = 5 parameters), and `nn.Sigmoid()` turns the score into a probability. The whole model has 17 parameters.
- `nn.BCELoss()` is binary cross-entropy: the −ln p loss for a yes/no label. `torch.optim.SGD` is plain gradient descent with learning rate 0.5.
- In the loop, `model(X)` is the guess and `loss_fn(p, y)` is the measure.
- `opt.zero_grad()` clears gradients left from the previous step, because PyTorch adds new gradients to old ones.
- `loss.backward()` computes every gradient; `opt.step()` applies the update rule to every weight.

Those three nudging lines are the entire learning algorithm. A network that reads digits or writes text uses exactly the same loop with more layers and more data.

## Testing and generalization {rail=Generalization}

{lede} The aim of training is not to do well on the examples the model has seen, but ==to do well on examples it has never seen==. That ability is called [[generalization|How well a trained model performs on new data it did not learn from, as opposed to the training data.]], and testing is how we measure it.

### Memorising versus understanding

Two students prepare for an exam. One memorises the answers to every past paper. The other understands the topic. On a question copied from a past paper, both score full marks. On a new question, the first student is stuck. A model can do the same: it can memorise its training set perfectly and still fail on anything new. This is why ==training accuracy alone tells you very little==. We must test on data the model did not train on.

### Train, validation and test

We therefore split the data into three parts.

- The **training set** is used to adjust the weights.
- The [[validation set|Data held out from training and used to compare models or settings, such as the number of hidden units or when to stop training.]] is used to make choices: how many hidden units, which learning rate, when to stop.
- The [[test set|Data kept completely untouched until the very end. It gives one honest estimate of how the finished model will do on new data.]] is used once, at the very end, for one honest score.

==The test set must be touched only once.== If you keep checking the test score and changing the model to improve it, you are tuning to the test set, and its score will look better than the model really is. The validation set exists so the test set can stay clean. Try the sliders: with 1,000 examples, a 70% training share and 15% validation gives 700, 150 and 150.

```viz
{
  "type": "calc",
  "kind": "split",
  "title": "Splitting a data set",
  "intro": "Set the data set to 1,000 examples, training to 70% and validation to 15%. Read the three counts. Then try 5,000 examples and a different split."
}
```

Check the arithmetic: 1,000 × 0.70 = 700 for training, 1,000 × 0.15 = 150 for validation, and the remaining 1,000 − 700 − 150 = 150 for test. In Lab Part 6.5 you split data by indices and check that no example appears in two parts.

### Underfitting, a good fit and overfitting

Now let us see generalization happen. In the figure, the dots are noisy measurements of a smooth curve. The filled dots are training data and the hollow ones are test data. The slider chooses the *degree* of a polynomial, a curve with more and more bends as the degree rises. The curve is fitted to the training points only; the figure then shows the error on both sets.

```viz
{
  "type": "fit",
  "title": "Underfitting and overfitting",
  "intro": "Move the degree slider from 0 up to 9. Watch the curve, and read the training error and the test error.",
  "degree": 1,
  "maxDegree": 9,
  "train": [ { "x": -0.95, "y": -0.82 }, { "x": -0.68, "y": -0.76 }, { "x": -0.4, "y": -1.16 }, { "x": -0.12, "y": 0.07 }, { "x": 0.18, "y": 0.26 }, { "x": 0.46, "y": 0.85 }, { "x": 0.74, "y": 0.76 }, { "x": 0.96, "y": 0.83 } ],
  "test": [ { "x": -0.82, "y": -0.98 }, { "x": -0.54, "y": -0.97 }, { "x": -0.26, "y": -0.62 }, { "x": 0.03, "y": -0.09 }, { "x": 0.32, "y": 0.61 }, { "x": 0.6, "y": 0.91 }, { "x": 0.86, "y": 0.86 } ],
  "note": "The data are authored for teaching: a smooth curve plus noise. Errors are mean squared errors."
}
```

Here is what you should see (errors computed from this data):

| Degree | Training error | Test error | What it means |
|---|---|---|---|
| 0 (a flat line) | 0.582 | 0.605 | Underfits: too simple, both errors high |
| 1 (a slanted line) | 0.098 | 0.057 | Still too simple to follow the curve |
| 3 | 0.047 | 0.005 | A good fit: the lowest test error |
| 7 | 0.000 | 0.417 | Overfits: it passes through every training point, but swings wildly in between |

At degree 7 the training error is zero, because 8 training points can be matched exactly by a curve of that flexibility. But the test error has jumped to 0.417, about eighty times the best value. That is [[overfitting|When a model fits the training data, including its noise, so closely that it does badly on new data. Training error is low but test error is high.]]: the model learned the noise, not the pattern. At the other extreme, [[underfitting|When a model is too simple to capture the pattern in the data, so it does badly on both the training data and new data.]] means the model is too simple and both errors stay high. The sweet spot is in the middle. (With only 7 test points, the test error can sometimes come out lower than the training error, as at degree 1, so do not over-read single numbers.)

### Learning curves

You can spot the same story while a network trains. Early on, both the training loss and the validation loss fall. Then the model starts to memorise: the training loss keeps falling, but the validation loss levels off and begins to *rise*. The point where the validation loss is lowest is the moment to stop. For instance (illustrative numbers):

| Epoch | Training loss | Validation loss | Reading |
|---|---|---|---|
| 10 | 0.60 | 0.62 | Both falling: learning |
| 50 | 0.25 | 0.30 | Best validation score so far |
| 100 | 0.10 | 0.35 | Validation rising: overfitting begins |
| 200 | 0.02 | 0.55 | Memorising the training set |

A model that has a big gap between training and validation scores is overfitting. In Lab Part 6.3 you produce such a table yourself and find the best stopping epoch.

### Remedies

When a model overfits, there are four standard fixes. **More data**: the more examples, the harder it is to memorise them all. **A simpler model**: fewer hidden units or fewer layers leave less room for memorising. **Regularization**: add a small penalty on large weights so the model prefers smooth, simple solutions. **Early stopping**: stop at the epoch where validation loss is lowest. When a model underfits, do the opposite: use a bigger model, train longer, or give it better features. Lab Part 6.4 compares three of these remedies.

### Two traps: leakage and biased labels

==Data leakage is when information from the test set sneaks into training.== A typical case: the same photo, or two photos of the same patient, appears in both the training set and the test set. The model then looks excellent at test time, but only because it has effectively seen the answers. Split by person or by source before anything else. Another case is scaling or choosing features using the whole data set, test rows included.

The second trap is in the labels. Suppose a school builds a model to predict which students will fail from past teachers’ marks. If the past marks reflect bias, such as harsher marking for students from one barangay, the model will learn to copy that bias, and a high test score will not reveal it, because the test labels carry the same bias. ==A high test accuracy only proves the model agrees with the labels, not that the labels are fair.== Always ask who made the labels and who might be treated worse.

### Check: which fit is it?

Read the scenario and decide whether the model is underfitting, a good fit, or overfitting.

```viz
{
  "type": "sorter",
  "title": "Underfit, good fit or overfit?",
  "intro": "Use the scores to diagnose each model. You get an explanation either way.",
  "bins": [
    { "id": "underfit", "label": "Underfit" },
    { "id": "good", "label": "Good fit" },
    { "id": "overfit", "label": "Overfit" }
  ],
  "cards": [
    { "name": "Training accuracy 60%, test accuracy 58%", "answer": "underfit", "why": "Both scores are low and close together. The model is too simple to learn the pattern." },
    { "name": "Training accuracy 98%, test accuracy 71%", "answer": "overfit", "why": "A large gap: it does far better on data it has seen than on new data." },
    { "name": "Training accuracy 91%, test accuracy 89%", "answer": "good", "why": "High scores, small gap. It generalizes." },
    { "name": "Training loss still falling, validation loss rising for 20 epochs", "answer": "overfit", "why": "The model is now memorising the training set instead of learning the pattern." },
    { "name": "A straight line through data that clearly curves; error high on both sets", "answer": "underfit", "why": "The model cannot bend enough to follow the data." },
    { "name": "A degree-9 curve through 10 points with zero training error and a huge test error", "answer": "overfit", "why": "It has memorised every point, including the noise." },
    { "name": "Training 84%, validation 83%, test 83%", "answer": "good", "why": "All three scores agree, so the model is not tuned to any one set." },
    { "name": "Predicts the same class for every example, so both accuracies equal that class’s share of the data", "answer": "underfit", "why": "It has learned nothing about the inputs. Both scores are equally poor." }
  ]
}
```

## Key takeaways

- **Supervised learning is learning from labelled examples.** Each example has features and a ==label that is the correct answer==. A model with parameters makes a prediction, a loss scores it, and training adjusts the parameters to shrink the loss.
- **The loss turns mistakes into one number.** ==Mean squared error is used for numbers and cross-entropy (−ln p) for categories==. Both punish big, confident mistakes hardest.
- **A neuron computes a = φ(Σ wx + b).** The weights say what it looks at, the bias says how easily it fires, and ==a layer is many neurons written as φ(Wx + b)==. A layer with in inputs and out neurons has in × out + out parameters.
- **One neuron draws only a straight line,** so it cannot solve XOR. ==A hidden layer with a non-linear activation bends the boundary==; without the non-linearity, stacked layers collapse into one.
- **Training is repeated gradient descent: w ← w − η × gradient.** Then ==test on data the model has never seen==: use train, validation and test sets, touch the test set once, and watch for underfitting, overfitting and leaking data.

<!-- LAB-START -->
## Lab: Imagine a neural network, then build one {rail=Laboratory}

This lab is called *Imagine a neural network, then build one*: you first picture a network as a machine of dials, as a team of voters and as a landscape of mistakes, and then ==build one by hand, in NumPy and in PyTorch, watch it learn, watch it fail (too small, too big, wrong learning rate) and finally design a small classifier of your own==. It has seven parts, and every part begins with an **Imagine first** moment: you write a prediction or describe a picture in words before you run any code and compare it with what really happens. Parts 1 to 3 are in this section; they use only NumPy and a few lines of PyTorch on tiny data sets that the helper cell generates, so nothing is downloaded and everything runs in seconds on a CPU. Each part points back to a section of the lesson. Time: about 3 hours.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cell, run a smoke test | 10 |
| 1 | Supervised learning by hand: be the model | 25 |
| 2 | Loss: measure the mistake | 20 |
| 3 | One neuron, built by hand | 25 |
| 4 | Layers and stacking: a network from scratch | 30 |
| 5 | Learning: gradient descent you can watch | 30 |
| 6 | The same thing in PyTorch, then break it | 25 |
| 7 | Design your own classifier and reflect | 25 |

The total is 190 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week02` and a notebook `lab2.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`mse`, `neuron`, `train_neuron`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cell and then the blocks again from the top.
2. Do not install packages: PyTorch, NumPy and Matplotlib are already installed and there is no internet. Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook. This lab needs **no downloads at all**: every data set is generated by the helper cell or written into the notebook.
3. A CPU is enough. Every block below finishes in seconds (the longest loops train a tiny network for a few hundred epochs). The GPU of the Spark is shared by the whole class and gives no advantage on data this small, so leave it alone. The helper cell prints which device it found.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste this into the first cell and run it. It fixes the random seed (so your numbers match the ones printed in this lab), defines the empty `RESULTS` dictionary and the `check(...)` function, and defines the helpers that generate and draw the small data sets used in every part:

```python
import os, gc, math, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
QUICK = False                       # set True to shorten long loops while you debug
print("Running on:", device)

RESULTS = {}                        # keep your key numbers here; Part 7 prints them together

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

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
    """Colour the plane by predict(points) (probabilities 0..1) and draw the labelled points."""
    ax = ax or plt.gca()
    g = np.linspace(-lim, lim, 120)
    gx, gy = np.meshgrid(g, g)
    P = np.asarray(predict(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32))).reshape(gx.shape)
    ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 11), cmap="RdYlGn", alpha=0.55)
    ax.contour(gx, gy, P, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="s", c="tab:blue", edgecolor="k", s=28, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="o", c="tab:orange", edgecolor="k", s=28, label="class 1")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_title(title)

def plot_network_diagram(layers, ax=None, title=""):
    """Circles for units, lines for weights; layers = e.g. [2, 4, 1]. Returns the number of parameters."""
    ax = ax or plt.gca()
    for l, n in enumerate(layers):
        for j in range(n):
            y = (j - (n - 1) / 2) * 1.2
            if l:
                for i in range(layers[l - 1]):
                    ax.plot([l - 1, l], [(i - (layers[l - 1] - 1) / 2) * 1.2, y], c="gray", lw=0.7, zorder=1)
            ax.scatter([l], [y], s=450, c="white", edgecolor="k", zorder=2)
    ax.set_axis_off(); ax.set_title(title or " → ".join(map(str, layers)))
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))
```

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `make_xor(n, noise, seed)`, `make_moons(...)` and `make_circles(...)` return `X` (an `n × 2` float array of points) and `y` (0 or 1 labels): XOR is four clusters at the corners of a square, moons are two interleaved half-circles, circles are a small disc inside a ring;
   - `plot_decision(predict, X, y, ax, title, lim)` colours the plane by a model's predicted probability (green for class 1, red for class 0, a black line where the model says 0.5) and draws the labelled points; `predict` is any function that takes an `(N, 2)` float array and returns `N` probabilities;
   - `plot_network_diagram(layers, ax, title)` draws circles for units and lines for weights for a list such as `[2, 4, 1]`, and returns the number of parameters;
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

7. Run this smoke test in the next cell. It checks that the helpers work and that your Matplotlib shows figures in the notebook:

```python
X_demo, y_demo = make_xor(8, seed=0)
print("make_xor ->", X_demo.shape, X_demo.dtype, "| labels:", y_demo.astype(int).tolist())
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, (name, maker) in zip(axes, [("xor", make_xor), ("moons", make_moons), ("circles", make_circles)]):
    X_demo, y_demo = maker(200)
    plot_decision(lambda P: np.full(len(P), 0.5), X_demo, y_demo, ax=ax, title=name)   # no model yet: just the points
plt.tight_layout(); plt.show()
plt.figure(figsize=(4, 3)); print("parameters of [2, 4, 1]:", plot_network_diagram([2, 4, 1], title="a 2 - 4 - 1 network")); plt.show()
```

   You should see `Running on: cpu` (or `cuda`) after the first cell, then `make_xor -> (8, 2) float32 | labels: [...]` with eight 0/1 labels, three scatter plots (the XOR points in four corner clusters, the two moons, the disc inside the ring, each with a flat orange-yellow background because the "model" says 0.5 everywhere), and `parameters of [2, 4, 1]: 17`. That 17 is `2·4 + 4` for the hidden layer plus `4·1 + 1` for the output, and you will derive it yourself in Part 4. If you do not see the plots, run `%matplotlib inline` in a cell first.

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cell and the earlier blocks again, in order.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the last line of the block is `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42` and that you did not run a block twice with a loop that changes a shared variable. Numbers that depend on random draws are stated only for a fresh run.
- A cell seems frozen: all blocks in Parts 1 to 3 finish in a couple of seconds on a CPU. If one runs longer, stop it (the square button), restart the kernel and run again from the top.

### Part 1. Supervised learning by hand: be the model (25 min)

*Lesson section: "Supervised learning: data, labels, model".* Supervised learning means learning from examples that come with the correct answer. In this part you are the model: you look at a small table of mangoes, each with a known label (ripe or not ripe), invent a rule, and measure how often the rule is right. Then you turn the rule into a model with three dials (parameters) and find good dial settings by plain random search, the dumbest possible way to train. Along the way you meet the words from the lesson section "Supervised learning": example, feature, label, model, parameter, prediction, training set and test set.

**Imagine first.** You will invent a rule that tells ripe mangoes from unripe ones using only two numbers, the weight and the colour score. Before you look at the data, predict how often your best rule will be right on mangoes it has never seen: write a number between 0 and 1 (0.5 is a coin flip, 1.0 is perfect). You will compare it with the real test accuracy at the end of the part.

```python
# Imagine first: how often (0 to 1) will your best rule be right on mangoes it has never seen?
my_prediction = None   # replace None with a number between 0 and 1, for example my_prediction = 0.8
```

1. **Step 1.1. Read a labelled table and split it.** Below is a tiny data set written by hand: 24 mangoes from a school garden. Every row is one **example**. It has two **features** (numbers the model may look at: the weight in grams and a colour score from 1 for green to 10 for golden) and one **label** (the correct answer: 1 = ripe, 0 = not ripe yet). The block prints the table and splits it in two by slicing: the first 18 rows are the **training set** (the model may learn from them) and the last 6 rows are the **test set** (kept hidden until the end, to see whether the rule works on new mangoes). The rows are already in mixed order, so both sets contain ripe and unripe fruit.

```python
# Each row: weight in grams, colour score (1 = green ... 10 = golden), label (1 = ripe, 0 = not ripe yet)
mangoes = [
    (210, 2.0, 0), (320, 8.5, 1), (250, 3.5, 0), (290, 7.0, 1), (180, 1.5, 0), (340, 9.0, 1),
    (300, 4.0, 0), (260, 7.5, 1), (220, 5.0, 0), (310, 6.5, 1), (200, 3.0, 0), (350, 8.0, 1),
    (270, 2.5, 0), (240, 6.0, 1), (230, 4.5, 0), (330, 7.2, 1), (190, 6.5, 0), (280, 5.5, 1),
    (160, 2.2, 0), (300, 8.8, 1), (255, 5.8, 0), (225, 5.2, 1), (310, 3.8, 0), (270, 6.8, 1),
]

print(f"{'#':>2} {'weight_g':>9} {'colour':>7} {'ripe':>5}")
for i, (weight, colour, ripe) in enumerate(mangoes):
    print(f"{i:>2} {weight:>9} {colour:>7.1f} {ripe:>5}")

train, test = mangoes[:18], mangoes[18:]           # the split, by slicing: 18 for training, 6 for testing
print("\nexamples:", len(mangoes), "| train:", len(train), "| test:", len(test))
print("ripe in train:", sum(r for _, _, r in train), "of", len(train), "| ripe in test:", sum(r for _, _, r in test), "of", len(test))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 #  weight_g  colour  ripe
 0       210     2.0     0
 1       320     8.5     1
 2       250     3.5     0
 3       290     7.0     1
 4       180     1.5     0
 5       340     9.0     1
 6       300     4.0     0
 7       260     7.5     1
 8       220     5.0     0
 9       310     6.5     1
10       200     3.0     0
11       350     8.0     1
12       270     2.5     0
13       240     6.0     1
14       230     4.5     0
15       330     7.2     1
16       190     6.5     0
17       280     5.5     1
18       160     2.2     0
19       300     8.8     1
20       255     5.8     0
21       225     5.2     1
22       310     3.8     0
23       270     6.8     1

examples: 24 | train: 18 | test: 6
ripe in train: 9 of 18 | ripe in test: 3 of 6
```

   You should see 24 numbered rows, then `examples: 24 | train: 18 | test: 6`, then `ripe in train: 9 of 18 | ripe in test: 3 of 6`. Half of each set is ripe, so a model that always answers "ripe" would be right exactly half of the time: 0.5 is the accuracy you must beat.

   Tips and common mistakes:

   - The label is the answer, not an input. If you ever put `ripe` among the features, the model would simply read the answer off, which is cheating.
   - Slicing `mangoes[:18]` takes the first 18 rows and `mangoes[18:]` the rest. The numbers in a slice never overlap, which is what keeps the test set honest.
   - If the table looks wrong, check that every row has three values: a weight, a colour score and a label.

   Try also (optional):

   - Print `mangoes[3]` and say aloud what each of its three numbers is.
   - Count by hand how many unripe mangoes have a colour score above 6 (the answer is 1, number 16 in the table). That is a hint of the hard cases to come.

   *Exercise 1.1:* write down, in a markdown cell, the **features** (two names), the **label** (its name and what 1 and 0 mean), one whole **example**, and the number of examples in each set. Why it matters: every supervised-learning project starts with exactly this vocabulary, whether the data are 24 mangoes or 24 million photos. Then fill in the three numbers below.

```python
n_features = None   # how many features does each example have?
n_train = None      # how many examples are in the training set?
n_test = None       # how many examples are in the test set?
```

2. **Step 1.2. Be the model: write your own rule.** Now do what a model does. A **rule** is a function that takes the features of one mango and returns a prediction, 1 or 0. The block below holds a first rule that looks only at the weight ("heavy means ripe"), plus two helpers that score any rule: **accuracy** (the fraction of examples the rule gets right, found with a loop) and **MSE**, the mean squared error between your 0/1 predictions and the labels (`mean((prediction − label)²)`). For 0/1 answers a wrong guess costs exactly 1, so MSE here equals the fraction of mistakes.

```python
def weight_rule(weight, colour):
    """First attempt: heavy mangoes are ripe."""
    return 1 if weight > 270 else 0

def accuracy(rule, rows):
    """Fraction of rows where the rule's answer equals the label."""
    right = 0
    for weight, colour, ripe in rows:
        if rule(weight, colour) == ripe:
            right += 1
    return right / len(rows)

def rule_mse(rule, rows):
    """Mean squared error of 0/1 predictions against the labels."""
    return sum((rule(w, c) - r) ** 2 for w, c, r in rows) / len(rows)

print("weight rule  train accuracy:", round(accuracy(weight_rule, train), 3), "| test accuracy:", round(accuracy(weight_rule, test), 3))
print("weight rule  train MSE     :", round(rule_mse(weight_rule, train), 3))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
weight rule  train accuracy: 0.833 | test accuracy: 0.5
weight rule  train MSE     : 0.167
```

   You should see a train accuracy of 0.833 (15 of 18 right), a test accuracy of only 0.5 (3 of 6) and a train MSE of 0.167. The weight alone is not enough: some heavy mangoes are still green, and on the six new mangoes the rule is no better than a coin flip. Keep that gap between train and test in mind.

   Tips and common mistakes:

   - `accuracy` counts how many predictions equal the label and divides by the number of rows. A rule that always answers 1 scores 0.5 here.
   - Do not tune your rule on the test mangoes. If you look at them while you design the rule, the test accuracy stops being an honest estimate.
   - If `my_rule` raises an error, check that it returns 1 or 0 in every branch, including the case where no `if` matches.

   Try also (optional):

   - Write a rule that uses a line instead of one threshold, for example `x2 + (x1 - 250) / 100 > 5.5`, and compare its train and test accuracy with your first rule.
   - Make a rule that is wrong on purpose (always 1) and compute its MSE. For 0/1 answers it equals the fraction of mistakes, here 0.5.

   *Exercise 1.2:* write your own function `my_rule(x1, x2)` that beats the weight rule. Use the weight (`x1`), the colour (`x2`), or both, as a threshold or a line. Then run the check, which scores your rule on the training set. Write down in a markdown cell **what you changed** and why you thought it would help (for example "colour looks more important than weight, so I used colour"). Do not look at the test mangoes while you design the rule; that is the whole point of a test set.

```python
def my_rule(x1, x2):
    """x1 = weight in grams, x2 = colour score. Return 1 (ripe) or 0 (not ripe yet). Change the rule!"""
    return 1 if x2 > 5.5 else 0

my_train_acc = accuracy(my_rule, train)
print("my_rule  train accuracy:", round(my_train_acc, 3), "| train MSE:", round(rule_mse(my_rule, train), 3))
```

3. **Step 1.3. A model with parameters.** A hand-written rule has numbers hidden inside it (270 g, 5.5). A **model** makes those numbers explicit so that a computer can search for them. Ours is `predict = 1 if w1·x1 + w2·x2 + b > 0`, with three **parameters**: two weights and a bias. First the features are rescaled so that both are of similar size (`(weight − 250) / 100` and `colour − 5`), otherwise the weight in grams would swamp the colour. The block tries five parameter settings by hand and prints the training accuracy of each. This is the picture of "training": a search for good values of (w1, w2, b).

```python
def scale(rows):
    """Turn rows into a feature matrix X (rescaled) and a label vector y."""
    X = np.array([ [(w - 250) / 100, c - 5.0] for w, c, _ in rows])
    y = np.array([r for _, _, r in rows])
    return X, y

X_train, y_train = scale(train)
X_test, y_test = scale(test)

def predict(X, w1, w2, b):
    """1 where w1*x1 + w2*x2 + b > 0, else 0."""
    return ((w1 * X[:, 0] + w2 * X[:, 1] + b) > 0).astype(int)

def acc(X, y, w1, w2, b):
    return float((predict(X, w1, w2, b) == y).mean())

settings = [(0, 0, 0), (1, 0, 0), (0, 1, 1.5), (0, 1, 0), (0.5, 1, -0.3)]
print(f"{'w1':>5} {'w2':>5} {'b':>5}   train accuracy")
for w1, w2, b in settings:
    print(f"{w1:>5} {w2:>5} {b:>5}   {acc(X_train, y_train, w1, w2, b):.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
   w1    w2     b   train accuracy
    0     0     0   0.500
    1     0     0   0.833
    0     1   1.5   0.778
    0     1     0   0.944
  0.5     1  -0.3   0.944
```

   You should see train accuracies of 0.500 for (0, 0, 0) (the sum is 0 for every mango, so everything is predicted 0, which is wrong for the 9 ripe mangoes), 0.833 for the weight-only setting (1, 0, 0), 0.778 for (0, 1, 1.5) (colour only, but the bias 1.5 pushes the threshold too low, so many unripe mangoes are called ripe), 0.944 for the colour-only setting (0, 1, 0) and 0.944 for (0.5, 1, −0.3). The bias matters: the same weights with a bad bias lose accuracy. Three numbers and a threshold already make a model.

   Tips and common mistakes:

   - Without the rescaling a weight of 250 grams would swamp a colour of 5, and every dial setting would look alike. Rescaling features to similar sizes is a habit in machine learning.
   - Accuracy is a step function of the dials: it stays the same until a point crosses the line, then jumps. That makes it hard to improve by tiny nudges, which is one reason training uses a smoother loss (Part 2).
   - The settings (0, 0, 0) always give z = 0, and `z > 0` is false, so everything is predicted 0.

   Try also (optional):

   - Try `(-1, 1, 0)` and `(1, -1, 0)`. One of them is bad on purpose: the second gives a train accuracy of 0.111, because it flips the answer. A model that is always wrong is just as informative as one that is always right.
   - Try a very large bias such as `b = 5` and explain, using the sign of `z`, why the model now answers the same for every mango.

   *Exercise 1.3:* find your own setting of (w1, w2, b) with a training accuracy of at least 0.85 by changing the dials by hand, and write down in a markdown cell what each dial did when you turned it (the weight `w1` makes the weight feature count more, the bias `b` moves the threshold). Why it matters: ==training a neural network is this same search, only with thousands of dials and a smarter way to turn them==.

```python
w1, w2, b = 0.5, 1.0, -0.3     # change these three numbers and run again
my_acc = acc(X_train, y_train, w1, w2, b)
print("my setting", (w1, w2, b), "-> train accuracy", round(my_acc, 3))
```

4. **Step 1.4. Random search: the dumbest way to train.** If you cannot decide the dials by hand, let the computer try many. **Random search** draws 2000 random triples (w1, w2, b), scores each one on the training set, and keeps the best. It is very crude (it does not learn from earlier tries), but it shows the idea of training without any maths. The key honest step: the winner is chosen using the **training** set only, and only then do we look at the test set once. The second block draws the data and the line `w1·x1 + w2·x2 + b = 0` that the winner found.

```python
def random_search(X, y, n_tries=2000, seed=SEED):
    """Try n_tries random (w1, w2, b) with values in -3..3; return the best one and its accuracy on (X, y)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-3, 3, size=(n_tries, 3))                       # one row per try: w1, w2, b
    guesses = (X @ W[:, :2].T + W[:, 2]) > 0                         # shape (n_examples, n_tries)
    accs = (guesses == y[:, None]).mean(axis=0)                     # accuracy of every try
    best = int(np.argmax(accs))
    return W[best], float(accs[best])

(best_w1, best_w2, best_b), best_train = random_search(X_train, y_train, 2000)
best_test = acc(X_test, y_test, best_w1, best_w2, best_b)
print(f"best of 2000 tries: w1={best_w1:.2f}, w2={best_w2:.2f}, b={best_b:.2f}")
print(f"train accuracy {best_train:.3f} | test accuracy {best_test:.3f}")
RESULTS["part1_random_search"] = {"train": round(best_train, 3), "test": round(best_test, 3)}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
best of 2000 tries: w1=2.23, w2=0.81, b=-0.43
train accuracy 1.000 | test accuracy 0.667
```

   You should see `best of 2000 tries: w1=2.23, w2=0.81, b=-0.43` and `train accuracy 1.000 | test accuracy 0.667`. The search found a line that gets all 18 training mangoes right, but it is right on only 4 of the 6 test mangoes. The winner trusts the weight (w1 = 2.23) much more than the colour (w2 = 0.81): it has fitted the quirks of the 18 training mangoes. That gap is the first sight of **overfitting**, which returns in Part 6. With only 6 test mangoes the number is noisy; do not read too much into 0.667 versus 0.833.

   Tips and common mistakes:

   - The same seed gives the same 2000 tries, so your numbers are repeatable. Change `seed=SEED` to another number to see a different search.
   - `X @ W[:, :2].T` computes all 2000 `w1·x1 + w2·x2` sums at once. If a line like this confuses you, print the shapes: `(18, 2) @ (2, 2000) = (18, 2000)`.
   - A perfect train accuracy of 1.000 is not a success story by itself. Always look at the test accuracy next to it.

   Try also (optional):

   - Change the range `-3..3` to `-30..30`. Does the best accuracy change? Think about which part of (w1, w2, b) decides where the line is: only the direction and the ratio of the dials, not their size.
   - Run the search with `n_tries = 100000`. With `SEED = 42` the train accuracy stays 1.000 and the test accuracy stays 0.667: more search on the training set does not buy a better test accuracy.

   *Exercise 1.4:* run the search with fewer and with more tries, and **repeat each 20 times with different seeds** (the average is more trustworthy than one run). Store the average best-train accuracy for 10 tries in `avg10` and for 2000 tries in `avg2000`. Then write two sentences: does more search always help, and what is the risk of choosing the winner on the training set only? Why it matters: this is the smallest experiment that shows the shape of every later training curve: more search or more training gives better scores on the data you search on.

```python
avg10 = None      # average best-train accuracy of random_search with 10 tries (20 seeds)
avg2000 = None    # the same with 2000 tries
```

5. **Step 1.5. Draw the line, and look at what the model gets wrong.** Draw the 18 training mangoes (circles: ripe, squares: not ripe), the 6 test mangoes (with a black edge) and the line the random search found. Then list the mangoes the model gets wrong. Looking at the mistakes is how a data scientist learns what is hard about a problem.

```python
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for X, y, edge, name in [(X_train, y_train, "none", "train"), (X_test, y_test, "k", "test")]:
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="s", c="tab:blue", edgecolor=edge, s=60, label=f"{name}: not ripe")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="o", c="tab:orange", edgecolor=edge, s=60, label=f"{name}: ripe")
xs = np.linspace(-1.5, 1.5, 50)
ax.plot(xs, -(best_w1 * xs + best_b) / best_w2, "k-", lw=1.5, label="learned line")   # where w1*x1 + w2*x2 + b = 0
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-5, 5)
ax.set_xlabel("weight (rescaled)"); ax.set_ylabel("colour (rescaled)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

print("Mistakes of the learned line (weight_g, colour, true label):")
for (w, c, r), pred in zip(mangoes, predict(np.vstack([X_train, X_test]), best_w1, best_w2, best_b)):
    if pred != r:
        print("  ", (w, c, r), "predicted", pred)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
Mistakes of the learned line (weight_g, colour, true label):
   (255, 5.8, 0) predicted 1
   (225, 5.2, 1) predicted 0
```

   You should see the plot, with the line tilted so that heavy mangoes lie on the ripe side, and two mistakes, both of them test mangoes: `(255, 5.8, 0) predicted 1` (a medium-weight mango with a fairly high colour score that is still not ripe) and `(225, 5.2, 1) predicted 0` (a light mango that is ripe even though its colour is only medium). Both are borderline cases that a person might also label differently, which leads to the next exercise.

   Tips and common mistakes:

   - If the plot is empty, run the block of 1.3 and 1.4 first: it needs `X_train`, `X_test` and the dials `best_w1`, `best_w2`, `best_b`.
   - The line is drawn where `w1·x1 + w2·x2 + b = 0`, so it separates the plane into a ripe side and an unripe side.
   - Your `label_note` can be short, but it should answer all three questions (a), (b) and (c).

   Try also (optional):

   - Pick one mistake and write what extra feature (sweetness, smell, firmness) could have fixed it.
   - Ask a classmate to label the six test mangoes by looking at the weight and colour only, and compare with the real labels: do people disagree on the borderline ones?

   *Exercise 1.5:* in a markdown cell write four or five sentences: (a) what makes a **label** good or bad (who decided that a mango is "ripe", and by what test: the farmer's thumb, a lab sugar meter, a taste test?); (b) one way labels could be wrong or **biased** in the real world (for example a spam filter trained on labels from a single person, or a hospital model whose labels come only from patients who could afford a test); (c) what the model would learn from wrong labels. Save your answer in `label_note` and check that it has at least 25 words. Why it matters: a supervised model can only be as good as its labels.

```python
label_note = """
(write your sentences here)
"""
```

6. **Imagine first, revisited.** Run the next block to compare your prediction with the facts.

```python
# Imagine first, revisited
RESULTS["part1_test_acc"] = round(best_test, 3)
if my_prediction is None:
    print("You did not write a prediction at the top of Part 1; the real test accuracy was", round(best_test, 3))
else:
    print(f"You predicted {my_prediction}. The random-search model scored {best_test:.3f} on the test mangoes.")
assert 0.0 <= best_test <= 1.0
```

   You should see your prediction next to `0.667`, the test accuracy of the random-search model (4 of the 6 test mangoes right). If you guessed 0.9 or more, you met your first lesson of the course: scores on the training set flatter a model, and a small test set is a noisy judge. Write one sentence about it.

   Questions to answer in a markdown cell before you move on:

   - Which of your two features carries more information about ripeness, and how did you decide?
   - Why must the test set stay hidden while you design the rule or search the dials?
   - What is the difference between a *feature*, a *label* and a *parameter*?
   - Why is a train accuracy of 1.000 together with a test accuracy of 0.667 a warning?
   - How would you check whether a label such as "ripe" is fair and consistent?

   A short write-up template you can paste into a markdown cell:

```text
Part 1 write-up
- Features: ...    Label: ...    Examples: 24 (18 train, 6 test)
- My rule: ...     train accuracy ... , MSE ...
- My dials (w1, w2, b): ...     Random search: train ... , test ...
- What went wrong on the test set and why: ...
- One way labels can be wrong or biased: ...
```

**Before you go on to the next part**, make sure you have:

- your answer to Exercise 1.1 (read a labelled table and split it), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 1.2 (be the model: write your own rule), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 1.3 (a model with parameters), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 1.4 (random search: the dumbest way to train), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 1.5 (draw the line, and look at what the model gets wrong), written in a markdown cell or stored in the variables named in the exercise;
- your Imagine-first prediction for Part 1, and one sentence on how far it was from the facts.

### Part 2. Loss: measure the mistake (20 min)

*Lesson section: "Loss: how wrong is the model?".* To improve a model you first need a number that says how bad its mistakes are. That number is the **loss**: small is good, zero is perfect. In this part you write the two losses of the lesson yourself, mean squared error for numbers and cross-entropy (`−ln p`) for categories, check them against PyTorch, and then look at the loss as a landscape: a curve with a lowest point that training will try to reach. Finally you see how the choice of loss changes which model counts as best.

**Imagine first.** Two models predict a number for 10 examples. Model A is wrong by 0.5 on every one of the 10 examples. Model B is exactly right on 9 examples and wrong by 3 on the last one. Which model has the larger mean squared error? Write `"A"` or `"B"`, and keep your reason in one sentence. You will check it in step 2.1.

```python
# Imagine first: which model has the larger mean squared error, "A" (0.5 off on all 10) or "B" (3 off on just one)?
my_pick = None   # replace None with "A" or "B"
```

1. **Step 2.1. Write mse and bce yourself.** **Mean squared error** (MSE) is the average of the squared differences between the true values and the predictions: `mean((y − ŷ)²)`. **Binary cross-entropy** (BCE) is for yes/no labels: the model gives the probability `p` that the label is 1, and the loss for one example is `−ln` of the probability the model gave to the **true** class (`−ln p` when the label is 1, `−ln(1 − p)` when it is 0), averaged over the examples. The block implements both in NumPy, tries them on a small worked example, and compares them with PyTorch's `F.mse_loss` and `F.binary_cross_entropy`. The `assert` lines stop with an error if your numbers and PyTorch's disagree.

```python
import torch.nn.functional as F

def mse(y_true, y_pred):
    """Mean squared error: the average of (y - y_hat)^2."""
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def bce(y_true, p, eps=1e-12):
    """Binary cross-entropy: the average of -ln(probability given to the TRUE class)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)      # keep p away from exactly 0 or 1
    p_true = np.where(y_true == 1, p, 1 - p)                   # the probability the model gave to the right answer
    return float(np.mean(-np.log(p_true)))

# Worked example 1: predicting numbers (say, rice meals sold)
y = [3, 5, 2, 7]
y_hat = [2.5, 5.5, 2, 9]
print("MSE:", mse(y, y_hat), "(squared errors 0.25, 0.25, 0, 4 -> mean 1.125)")

# Worked example 2: predicting yes/no (is the mango ripe?)
labels = [1, 0, 1, 1]
probs = [0.9, 0.2, 0.6, 0.99]
print("per-example -ln p:", [round(float(-np.log(p if l == 1 else 1 - p)), 4) for l, p in zip(labels, probs)])
print("BCE:", round(bce(labels, probs), 4))

# Compare with PyTorch: the two must agree
t_mse = F.mse_loss(torch.tensor(y_hat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).item()
t_bce = F.binary_cross_entropy(torch.tensor(probs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)).item()
print("PyTorch MSE:", round(t_mse, 4), "| PyTorch BCE:", round(t_bce, 4))
assert abs(mse(y, y_hat) - t_mse) < 1e-5 and abs(bce(labels, probs) - t_bce) < 1e-5

# Imagine first, revisited: A is 0.5 off on all 10, B is 3 off on one of 10
mse_A = mse(np.zeros(10), np.full(10, 0.5))
mse_B = mse(np.zeros(10), np.array([0] * 9 + [3]))
print("Model A MSE:", mse_A, "| Model B MSE:", mse_B)
RESULTS["part2_mse_A_B"] = (mse_A, mse_B)
print("Your pick was", my_pick, "- the larger MSE belongs to", "A" if mse_A > mse_B else "B")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
MSE: 1.125 (squared errors 0.25, 0.25, 0, 4 -> mean 1.125)
per-example -ln p: [0.1054, 0.2231, 0.5108, 0.0101]
BCE: 0.2123
PyTorch MSE: 1.125 | PyTorch BCE: 0.2123
Model A MSE: 0.25 | Model B MSE: 0.9
Your pick was None - the larger MSE belongs to B
```

   You should see `MSE: 1.125`, per-example `-ln p` values of `[0.1054, 0.2231, 0.5108, 0.0101]` and `BCE: 0.2123`, and PyTorch printing `1.125` and `0.2123` too, so the assert passes. Then `Model A MSE: 0.25 | Model B MSE: 0.9`: B has the larger MSE, because squaring makes one big miss (3² = 9) cost far more than ten small ones (ten times 0.5² = 0.25 in total, divided by 10). If you picked A you were in good company: ten mistakes feels worse than one, but the loss does not see it that way.

   Tips and common mistakes:

   - `np.clip(p, eps, 1 - eps)` stops `log(0)`, which would give infinity. Real libraries do the same.
   - `np.where(y_true == 1, p, 1 - p)` picks the probability of the **true** class for each example: this is the heart of cross-entropy.
   - PyTorch's `F.binary_cross_entropy` expects probabilities between 0 and 1 (after a sigmoid); in Part 6 you will use `BCEWithLogitsLoss`, which expects raw scores instead.

   Try also (optional):

   - Change `probs` so that the model is sure and right (0.99, 0.01, 0.99, 0.99) and watch the BCE fall to about 0.01.
   - Make one probability exactly 0 for a true 1. What does `bce` return, and why does the clip matter?

   *Exercise 2.1:* compute the BCE of two models on four mangoes with the true labels `[1, 1, 0, 0]`. The **hedging** model always says `p = 0.5`. The **bold** model says `p = 0.9` for every mango. Store the two losses in `loss_hedge` and `loss_bold`. Then write one sentence on why the bold model, which is right about two mangoes and wrong about two, has the larger loss. Why it matters: this is what "confidently wrong is punished hard" looks like in numbers, and it is why classifiers are trained with cross-entropy.

```python
true_labels = [1, 1, 0, 0]
loss_hedge = None   # BCE when the model says p = 0.5 for all four
loss_bold = None    # BCE when the model says p = 0.9 for all four
```

2. **Step 2.2. The −ln p table and curve.** Why does cross-entropy punish confident mistakes so hard? Look at the function `−ln p`, where `p` is the probability the model gave to the **true** class. The block prints its value for five probabilities and plots the curve from 0.01 to 1. A model that is right and sure (p near 1) pays almost nothing; a model that gave the truth a tiny probability pays a lot.

```python
print(f"{'p (given to the truth)':>24}   -ln p")
for p in (0.99, 0.9, 0.5, 0.1, 0.01):
    print(f"{p:>24}   {-math.log(p):.3f}")

ps = np.linspace(0.01, 1.0, 200)
plt.figure(figsize=(6, 3.5))
plt.plot(ps, -np.log(ps), c="tab:red")
for p in (0.9, 0.5, 0.1, 0.01):
    plt.scatter([p], [-math.log(p)], c="k", zorder=3)
    plt.annotate(f"p={p}: {-math.log(p):.2f}", (p, -math.log(p)), textcoords="offset points", xytext=(8, 4), fontsize=8)
plt.xlabel("p, the probability the model gave to the true class"); plt.ylabel("loss = -ln p")
plt.title("The cross-entropy loss of one example"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
  p (given to the truth)   -ln p
                    0.99   0.010
                     0.9   0.105
                     0.5   0.693
                     0.1   2.303
                    0.01   4.605
```

   You should see the table `0.99 -> 0.010`, `0.9 -> 0.105`, `0.5 -> 0.693`, `0.1 -> 2.303`, `0.01 -> 4.605`, and a curve that is nearly flat near p = 1 and shoots up as p moves toward 0. Cutting p from 0.5 to 0.1 more than triples the loss (0.693 to 2.303), and from 0.1 to 0.01 it doubles again. As p → 0 the loss grows without limit.

   Tips and common mistakes:

   - Natural logarithm: `math.log` and `np.log` are `ln`, the logarithm to base e. Do not use `log10`.
   - The x axis starts at 0.01 because `-ln 0` is infinite and cannot be drawn.
   - In the lesson the same table appears as the equation figure for cross-entropy: compare your printed numbers with it.

   Try also (optional):

   - Add p = 0.001 to the table. What loss do you get (6.908)?
   - Plot `-np.log2(ps)` on the same axes. It is the same curve, 1.44 times taller: bits instead of nats.

   *Exercise 2.2:* in a markdown cell write two or three sentences: what happens to `−ln p` as p goes to 0, and why does this punish a model that is **confident and wrong** much more than one that is unsure? (Link it to Week 1: GPT-2 was trained with exactly this loss on the next word.) Then compute the probability that would give a loss of exactly 5 and store it in `p_for_5`.

```python
p_for_5 = None   # the probability p for which -ln p = 5 (hint: undo the logarithm with math.exp)
```

3. **Step 2.3. The loss landscape of a one-weight model.** Take the simplest model there is: `ŷ = w·x`, one dial `w`. The five points below follow the line `y = 2x` with a little noise (so the true slope is about 2). For 81 values of `w` from −2 to 6 the block computes the MSE and plots the result against `w`. The curve is a parabola, a bowl: this is the picture of "downhill" in the lesson. The best `w` is the bottom of the bowl, found here with `argmin`. Training a network is walking down such a bowl, only with thousands of dials at once.

```python
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])             # about 2 * x, with a little noise

ws = np.linspace(-2, 6, 81)                          # 81 values of the dial, spacing 0.1
losses = np.array([mse(y, w * x) for w in ws])       # the loss for each value

i_best = int(np.argmin(losses))
print(f"lowest loss on the grid: w = {ws[i_best]:.1f}, MSE = {losses[i_best]:.3f}")
print(f"loss at w = -2: {losses[0]:.1f} | at w = 6: {losses[-1]:.1f}")

plt.figure(figsize=(6, 3.5))
plt.plot(ws, losses, c="tab:blue")
plt.scatter([ws[i_best] ], [losses[i_best] ], c="tab:red", zorder=3, label=f"minimum at w = {ws[i_best]:.1f}")
plt.xlabel("w (the dial)"); plt.ylabel("MSE"); plt.title("The loss landscape of y = w * x")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
RESULTS["part2_best_w"] = round(float(ws[i_best]), 2)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
lowest loss on the grid: w = 2.0, MSE = 0.022
loss at w = -2: 176.3 | at w = 6: 175.7
```

   You should see `lowest loss on the grid: w = 2.0, MSE = 0.022`, the loss at both ends of the grid huge and almost equal (`176.3` at w = −2 and `175.7` at w = 6, because both are 4 away from the minimum), and a smooth, symmetric bowl with its bottom at w = 2. Anywhere you stand on the bowl, the direction of the slope tells you which way is downhill.

   Tips and common mistakes:

   - `ws[i_best]` is the grid value with the lowest loss. The exact best slope is 2.0036; the grid has a spacing of 0.1, so it finds 2.0.
   - The bowl is a parabola because MSE of a line through the origin is a quadratic function of `w`.
   - If the plot is empty, check that `x` and `y` were not overwritten: this block defines both.

   Try also (optional):

   - Change the data `y` to follow `3 * x` and see the minimum move to 3.
   - Plot the **slope** of the bowl (the difference of neighbouring losses divided by 0.1) and see that it is zero at the bottom: that slope is what the gradient measures in Part 5.

   *Exercise 2.3:* compute the MSE for a model that always predicts 0 (`w = 0`) and for the model `w = 4`, store them in `loss_w0` and `loss_w4`, and say in one sentence why the two numbers are almost the same (both dials are 2 away from the best value, on opposite sides of the bowl). Why it matters: the loss at a particular value of `w` is a height on the landscape; the gradient (Part 5) is the slope there.

```python
loss_w0 = None   # MSE of the model y_hat = 0 * x
loss_w4 = None   # MSE of the model y_hat = 4 * x
```

4. **Step 2.4. Two losses, two behaviours: an outlier.** An **outlier** is an example that does not fit the pattern (a typing mistake, a broken sensor). Add one to the data (x = 6 with y = 30, where the pattern says 12) and ask which `w` is best for two different losses: MSE (squares the error) and **mean absolute error** (MAE, `mean(|y − ŷ|)`). A finer grid of 801 values of `w` is used so that the answers are accurate to 0.01.

```python
def mae(y_true, y_pred):
    """Mean absolute error: the average of |y - y_hat|."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

def best_w(loss, x, y, grid=np.linspace(-2, 6, 801)):
    """The value of w on the grid with the lowest loss."""
    return float(grid[int(np.argmin([loss(y, w * x) for w in grid]))])

x_out = np.append(x, 6.0)
y_out = np.append(y, 30.0)                           # the outlier: the pattern says about 12

w_mse_clean, w_mae_clean = best_w(mse, x, y), best_w(mae, x, y)
w_mse_out, w_mae_out = best_w(mse, x_out, y_out), best_w(mae, x_out, y_out)
print(f"{'':<12}{'MSE':>8}{'MAE':>8}")
print(f"{'clean data':<12}{w_mse_clean:>8.2f}{w_mae_clean:>8.2f}")
print(f"{'with outlier':<12}{w_mse_out:>8.2f}{w_mae_out:>8.2f}")
RESULTS["part2_outlier_w"] = {"mse": round(w_mse_out, 2), "mae": round(w_mae_out, 2)}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
                 MSE     MAE
clean data      2.00    2.02
with outlier    3.19    2.02
```

   You should see, for the best `w`: clean data 2.00 (MSE) and 2.02 (MAE); with the outlier 3.19 (MSE) but still 2.02 (MAE). One bad point drags the MSE answer from 2.0 to 3.2, a push of about 60 %, while the MAE answer does not move at all (MAE behaves like a median, MSE like a mean). Squaring makes a big error count far more than a small one, so the MSE loss "listens" to the outlier. That is sometimes what you want (large mistakes really are costly) and sometimes not (the point is a typing error).

   Tips and common mistakes:

   - MAE is the average of the absolute errors. It grows in a straight line with the error, MSE in a curve.
   - The finer grid (801 values) is needed so that the MAE minimum is not rounded to a coarse value.
   - An outlier is not always an error: sometimes it is the most important example. Look before you delete.

   Try also (optional):

   - Move the outlier to x = 6, y = 0. Does the MSE answer go below 2? Why?
   - Add two outliers instead of one and see which loss gets confused first.

   *Exercise 2.4:* choose your own outlier height `outlier_y` for x = 6 (anything far from 12), and compute how far each best `w` moves from its clean value: `shift_mse` and `shift_mae`. Then write one sentence in a markdown cell on why squaring matters, and one on a situation where you would prefer MAE.

```python
outlier_y = 30         # the y value of the extra point at x = 6 (the pattern says about 12)
shift_mse = None       # |best w for MSE with the outlier  -  best w for MSE on the clean data|
shift_mae = None       # the same for MAE
```

5. **Imagine first, revisited.** Run the next block to compare your prediction with the facts.

```python
print("Part 2 summary:", {k: v for k, v in RESULTS.items() if k.startswith("part2")})
```

   You should see one line with your Part 2 numbers: the two MSEs of the imagine question, the BCE of the hedging and bold models, the best `w` and the outlier results.

   Questions to answer in a markdown cell before you move on:

   - Why does the loss use the probability of the *true* class and not of the predicted class?
   - What loss would a model get for always saying 0.5, in the yes/no case?
   - Why is the minimum of a bowl-shaped loss easy to find, and what makes a real network's loss harder?
   - When would you choose MAE and when MSE?
   - How does the idea of `-ln p` connect to Week 1's training of GPT-2 on the next word?

   A short write-up template you can paste into a markdown cell:

```text
Part 2 write-up
- Which has the larger MSE, A or B, and why: ...
- bce(hedge) = ... , bce(bold) = ... , because ...
- -ln p as p -> 0: ...
- Best w on the grid: ... , loss there: ...
- Outlier: best w for MSE moved from ... to ... ; for MAE from ... to ... ; because ...
```

**Before you go on to the next part**, make sure you have:

- your answer to Exercise 2.1 (write mse and bce yourself), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 2.2 (the −ln p table and curve), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 2.3 (the loss landscape of a one-weight model), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 2.4 (two losses, two behaviours: an outlier), written in a markdown cell or stored in the variables named in the exercise;
- your Imagine-first prediction for Part 2, and one sentence on how far it was from the facts.

### Part 3. One neuron, built by hand (25 min)

*Lesson section: "One artificial neuron".* An artificial neuron is the smallest working piece of a neural network: it multiplies each input by a weight, adds a bias, and passes the sum through an activation function. In this part you code that recipe in a few lines, move its dials and watch the straight line it draws, set the weights for the logic gates AND and OR by hand, and then discover what one neuron cannot do: XOR. At the end you train a single neuron with PyTorch, so the same pieces appear once by hand and once by library.

**Imagine first.** Picture one neuron looking at two numbers (x1, x2) and answering yes or no. In words, what shape of boundary can it draw between the "yes" region and the "no" region of the plane: a straight line, a curve, a circle? Then predict whether one neuron can learn each of three logic gates (the inputs are 0 or 1): **AND** (yes only if both inputs are 1), **OR** (yes if at least one input is 1) and **XOR** (yes if exactly one input is 1). For each gate write `"yes"` or `"no"`.

```python
# Imagine first: can ONE neuron learn each gate? Write "yes" or "no" for each.
my_prediction = {"AND": None, "OR": None, "XOR": None}
boundary_shape = ""   # in a few words: what shape of boundary can one neuron draw?
```

1. **Step 3.1. Code the neuron and its four activations.** A neuron computes `z = w1·x1 + w2·x2 + b` and then `a = φ(z)`, where φ is the **activation function**. Four common activations are coded below: **step** (1 if z > 0, else 0), **sigmoid** `1 / (1 + e^(−z))` (a smooth step between 0 and 1), **tanh** (a smooth step between −1 and 1) and **ReLU** `max(0, z)`. The block reproduces the worked example of the lesson section "One artificial neuron": inputs `x = (2, 1)`, weights `w = (0.5, −1)`, bias `b = 0.25`. Then it plots the four activations on −5..5 and marks the value of `z` for this example.

```python
def step(z):    return np.where(np.asarray(z, dtype=float) > 0, 1.0, 0.0)
def sigmoid(z): return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))
def tanh(z):    return np.tanh(np.asarray(z, dtype=float))
def relu(z):    return np.maximum(0.0, np.asarray(z, dtype=float))

ACTIVATIONS = {"step": step, "sigmoid": sigmoid, "tanh": tanh, "relu": relu}

def neuron(x, w, b, act="sigmoid"):
    """One neuron: z = w . x + b, then a = act(z). Returns (z, a)."""
    z = float(np.dot(w, x) + b)
    return z, float(ACTIVATIONS[act](z))

x, w, b = (2, 1), (0.5, -1), 0.25
for name in ACTIVATIONS:
    z, a = neuron(x, w, b, name)
    print(f"{name:<8} z = {z:.2f}   a = {a:.3f}")

zs = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharex=True)
for ax, (name, f) in zip(axes, ACTIVATIONS.items()):
    ax.plot(zs, f(zs), c="tab:blue")
    z0 = neuron(x, w, b, name)[0]
    ax.scatter([z0], [f(z0)], c="tab:red", zorder=3)                 # the worked example
    ax.set_title(name); ax.grid(alpha=0.3); ax.set_xlabel("z")
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
step     z = 0.25   a = 1.000
sigmoid  z = 0.25   a = 0.562
tanh     z = 0.25   a = 0.245
relu     z = 0.25   a = 0.250
```

   You should see `z = 0.25` for all four lines (2·0.5 + 1·(−1) + 0.25 = 0.25) and the activations `step 1.000`, `sigmoid 0.562`, `tanh 0.245`, `relu 0.250`. The sigmoid value 0.562 is the one in the lesson. In the plot the red dot sits just right of zero: for step it is already at 1, for sigmoid just above the middle (0.5), for ReLU it equals z.

   Tips and common mistakes:

   - `np.dot(w, x)` multiplies the weights with the inputs and adds them up: 0.5·2 + (−1)·1 = 0. The bias then adds 0.25.
   - The step function uses `> 0`, so exactly z = 0 gives 0. Some books use `>= 0`; both are fine as long as you are consistent.
   - ReLU is not smooth at 0 but is the most common hidden activation today because it is cheap and does not flatten for positive z.

   Try also (optional):

   - Print the four activations for z = −3 and z = +3. Which ones are almost the same at +3? (Sigmoid 0.953, tanh 0.995, step 1, relu 3.)
   - Double the bias to 0.5 and keep the weights (z = 0.5). Which activation does not change at all? (The step: it is already 1.)

   *Exercise 3.1:* compute a second neuron by hand first, then with the code: inputs `x = (1, 3)`, weights `w = (0.4, −0.2)`, bias `b = 0.1`. Write the arithmetic of `z` in a markdown cell, then store `z_mine`, the ReLU output `a_relu` and the sigmoid output `a_sigmoid` (rounded to 3 decimals). Why it matters: every network in this course, up to the biggest models, is built from this one calculation repeated many times.

```python
z_mine = None         # z = w1*x1 + w2*x2 + b for x = (1, 3), w = (0.4, -0.2), b = 0.1
a_relu = None         # relu(z)
a_sigmoid = None      # sigmoid(z), 3 decimals
```

2. **Step 3.2. Weights and bias as dials.** A neuron says "yes" where `w1·x1 + w2·x2 + b > 0`. The edge between yes and no is the line `w1·x1 + w2·x2 + b = 0`. The block draws that line for different dial settings. On the left the weights stay at (1, 1) and the **bias** moves through −2..2; on the right the bias stays 0 and the **weights** change. It also prints the distance from the origin to the line, `|b| / ‖w‖`.

```python
xs = np.linspace(-3, 3, 100)

def line_y(w, b, xs):
    """Solve w1*x + w2*y + b = 0 for y."""
    return -(w[0] * xs + b) / w[1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for b in (-2, -1, 0, 1, 2):
    axes[0].plot(xs, line_y((1, 1), b, xs), label=f"b = {b}")
axes[0].set_title("weights (1, 1): the bias slides the line")
for w in [(1, 1), (1, 0.3), (0.3, 1), (1, -1)]:
    axes[1].plot(xs, line_y(w, 0, xs), label=f"w = {w}")
axes[1].set_title("bias 0: the weights tilt the line")
for ax in axes:
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
    ax.legend(fontsize=7); ax.set_aspect("equal")
plt.tight_layout(); plt.show()

for b in (-2, 0, 2):
    print(f"w = (1, 1), b = {b:>2}: distance from the origin to the line = {abs(b) / np.linalg.norm((1, 1)):.2f}")
print("w = (2, 2), b = 2: distance =", round(abs(2) / np.linalg.norm((2, 2)), 2), "(doubling all dials keeps the line closer)")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
w = (1, 1), b = -2: distance from the origin to the line = 1.41
w = (1, 1), b =  0: distance from the origin to the line = 0.00
w = (1, 1), b =  2: distance from the origin to the line = 1.41
w = (2, 2), b = 2: distance = 0.71 (doubling all dials keeps the line closer)
```

   You should see five parallel lines on the left: they all have slope −1, and the bias only moves the line (b = 0 passes through the origin, b = 2 lies below it). On the right the line turns when the weights change: (1, 0.3) is steep, (0.3, 1) is flat, (1, −1) tilts the other way. The printed distances are `1.41`, `0.00` and `1.41`, and `0.71` for (2, 2) with b = 2: doubling both weights but keeping the bias halves the distance. In short: **the weights set the direction of the line, the bias sets its position**.

   Tips and common mistakes:

   - The line equation is solved for the vertical axis; if `w[1]` is 0 the line is vertical and the formula divides by zero. Avoid `w2 = 0` in `line_y`.
   - Only the **direction** of w and the ratio b/‖w‖ decide where the line is. Multiplying everything by 2 does not move the step neuron's line at all (but it makes the sigmoid sharper).
   - A positive z means "yes": the weight vector points toward the yes side.

   Try also (optional):

   - Draw the line for `w = (2, 1)` and `b = -1` on top of the previous plot and describe its tilt and position in one sentence.
   - Take the neuron `w = (-1, 1), b = 0` from the exercise and draw its line together with a few random points, coloured by the neuron's answer.

   *Exercise 3.2:* choose weights and a bias so that the neuron answers 1 **only for points above the line y = x** (where x2 > x1). Think first: what must `w1` and `w2` look like if the line `w1·x1 + w2·x2 + b = 0` must be `x2 = x1`, and which side is "yes"? Store them in `w_above` (a pair) and `b_above`. The check tests six points.

```python
w_above = None    # a pair (w1, w2)
b_above = None    # the bias
```

3. **Step 3.3. One neuron learns AND and OR.** Logic gates take two inputs that are 0 or 1. **AND** answers 1 only when both inputs are 1; **OR** answers 1 when at least one input is 1. Both can be written as one neuron with a step activation. Choose the weights (1, 1) for both gates; AND needs a threshold between 1 and 2 (bias −1.5), OR a threshold between 0 and 1 (bias −0.5). The block prints the truth tables and checks them.

```python
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

def gate_outputs(w, b):
    """The step-neuron's answer for the four input pairs."""
    return [int(neuron(p, w, b, "step")[1]) for p in inputs]

and_w, and_b = (1, 1), -1.5
or_w, or_b = (1, 1), -0.5

print(f"{'x1':>3} {'x2':>3} | {'AND':>4} {'OR':>4}")
for p, a_out, o_out in zip(inputs, gate_outputs(and_w, and_b), gate_outputs(or_w, or_b)):
    print(f"{p[0]:>3} {p[1]:>3} | {a_out:>4} {o_out:>4}")

check("AND truth table", gate_outputs(and_w, and_b) == [0, 0, 0, 1], "AND is 1 only for (1, 1).")
check("OR truth table", gate_outputs(or_w, or_b) == [0, 1, 1, 1], "OR is 0 only for (0, 0).")
assert gate_outputs(and_w, and_b) == [0, 0, 0, 1] and gate_outputs(or_w, or_b) == [0, 1, 1, 1]
RESULTS["part3_and_or_weights"] = {"AND": (and_w, and_b), "OR": (or_w, or_b)}
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
 x1  x2 |  AND   OR
  0   0 |    0    0
  0   1 |    0    1
  1   0 |    0    1
  1   1 |    1    1
✓ AND truth table
✓ OR truth table
```

   You should see the table with AND `0 0 0 1`, OR `0 1 1 1`, and two ticks. For AND the sum x1 + x2 is 0, 1 or 2 and the bias −1.5 makes z positive only for 2; for OR the bias −0.5 makes z positive as soon as the sum reaches 1. Same weights, different bias: **the bias moves the threshold**.

   Tips and common mistakes:

   - The truth table lists all four input pairs because there are only four. For the 28 × 28 inputs of a digit image there are far too many to list, so we have to learn weights from examples instead.
   - The threshold of AND lies between a sum of 1 (one input) and 2 (both inputs); any bias between −2 and −1 works.
   - The same weights (1, 1) give different gates just by changing the bias: this is the "dial" picture of Part 3 again.

   Try also (optional):

   - Find a bias range for OR by trying `b = -0.1`, `-0.9`, `-1.1`. Which still give OR? (−0.1 and −0.9 do, −1.1 does not.)
   - Build NOT (one input, weight −1, bias 0.5) with a step neuron and print its truth table.

   *Exercise 3.3:* find the weights and bias of a **NAND** gate (the opposite of AND: 0 only for input (1, 1)) with a single step neuron. Hint: flip the signs of the AND neuron. Store them in `nand_w` (a pair) and `nand_b`, and in a markdown cell explain how you flipped the signs.

```python
nand_w = None     # a pair (w1, w2)
nand_b = None     # the bias
```

4. **Step 3.4. One neuron cannot learn XOR.** **XOR** answers 1 when exactly one input is 1. We cannot set its weights by hand, so let the computer try. The block gives a single step neuron 50,000 random sets of dials (w1, w2, b), each from −5 to 5, scores each set on the four-row truth table, and reports the best accuracy for AND, OR and XOR. The second plot draws the four XOR points: no straight line can put the two orange circles on one side and the two blue squares on the other.

```python
def search(X, y, n_tries=50000, seed=SEED, scale=5.0):
    """Try n_tries random (w1, w2, b); return all accuracies and the winning dials."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=(n_tries, 3))
    guesses = (X @ W[:, :2].T + W[:, 2]) > 0
    accs = (guesses == y[:, None]).mean(axis=0)
    return accs, W[int(np.argmax(accs))]

gates_X = np.array(inputs, dtype=float)
gate_y = {"AND": np.array([0, 0, 0, 1]), "OR": np.array([0, 1, 1, 1]), "XOR": np.array([0, 1, 1, 0])}

best_acc = {}
all_accs = {}
for name, y_gate in gate_y.items():
    all_accs[name], dials = search(gates_X, y_gate)
    best_acc[name] = float(all_accs[name].max())
    print(f"{name:<4} best accuracy over 50,000 random neurons: {best_acc[name]:.2f}")
RESULTS["part3_xor_best_acc"] = best_acc["XOR"]

# the four XOR points
yx = gate_y["XOR"]
plt.figure(figsize=(4, 4))
plt.scatter(gates_X[yx == 0, 0], gates_X[yx == 0, 1], marker="s", s=200, c="tab:blue", edgecolor="k", label="0")
plt.scatter(gates_X[yx == 1, 0], gates_X[yx == 1, 1], marker="o", s=200, c="tab:orange", edgecolor="k", label="1")
plt.plot([-0.3, 1.3], [0.8, -0.8], "k--", lw=1, label="x1 + x2 = 0.5: 3 of 4 right")      # the best a line can do
plt.xlim(-0.3, 1.3); plt.ylim(-0.3, 1.3); plt.gca().set_aspect("equal")
plt.title("XOR: no straight line works"); plt.legend(fontsize=7); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
AND  best accuracy over 50,000 random neurons: 1.00
OR   best accuracy over 50,000 random neurons: 1.00
XOR  best accuracy over 50,000 random neurons: 0.75
```

   You should see `AND best accuracy ...: 1.00`, `OR ...: 1.00` and `XOR ...: 0.75`. Out of 50,000 random neurons, XOR never reaches more than 3 of 4 rows right (0.75). It is not bad luck, it is geometry: the two 1s sit on one diagonal and the two 0s on the other, and a straight line cannot separate diagonals. This is the limit of a single neuron, and the reason for hidden layers in Part 4.

   Tips and common mistakes:

   - `all_accs[name]` holds the accuracy of every one of the 50,000 tries, so `(all_accs["OR"] == 1.0).sum()` counts the perfect ones.
   - The accuracy can only be 0.25, 0.5, 0.75 or 1.0 on four rows. For XOR the best value is always 0.75, however many tries you make.
   - Changing `SEED` changes the tries but not the outcome of the experiment.

   Try also (optional):

   - Run the search with 500,000 tries (about a second or two). Does XOR ever reach 1.0?
   - Add a third input x3 = x1·x2 to the XOR points and ask whether a single neuron can now solve it. This is the idea of a hidden feature, which Part 4 builds.

   *Exercise 3.4:* count how many of the 50,000 random neurons got a **perfect** score on XOR and on OR, store the counts in `n_perfect_xor` and `n_perfect_or`, and compare with the predictions you made at the top of Part 3. Write two sentences: which gates could one neuron learn, and why can the same neuron solve OR but not XOR? Why it matters: the number 0 for XOR is not a failure of the search, it is a proof by experiment that the model is too simple.

```python
n_perfect_xor = None   # how many of the 50,000 tries scored 1.0 on XOR?
n_perfect_or = None    # ... on OR?
```

5. **Step 3.5. Train a single neuron with PyTorch.** Now let a neuron find its own weights by training, using the standard PyTorch loop (it is explained line by line in Part 6, so only skim it for now). The neuron is `nn.Linear(2, 1)` followed by a sigmoid, the loss is binary cross-entropy (`nn.BCELoss`) and the optimiser is SGD, which nudges the dials downhill. The data are the two interleaved half-moons from `make_moons`. A single neuron draws one straight line, so it can only do so well on curved data. A second set of moons with a different seed is the test set.

```python
def train_neuron(X, y, epochs=300, lr=0.5, seed=SEED):
    """Train one sigmoid neuron on (X, y). Returns the model, the loss at every epoch and the train accuracy."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())          # z = w.x + b, then sigmoid
    Xt, yt = torch.tensor(X), torch.tensor(y).unsqueeze(1)
    loss_fn = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()                                            # forget the last step's gradients
        loss = loss_fn(model(Xt), yt)                              # measure the mistake
        loss.backward()                                            # work out which way is downhill
        opt.step()                                                 # take a small step that way
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((model(Xt) > 0.5).float() == yt).float().mean().item()
    return model, losses, acc

def as_predictor(model):
    """Wrap a model so that plot_decision can call it with a NumPy array."""
    return lambda P: model(torch.tensor(P)).detach().numpy().ravel()

X_moons, y_moons = make_moons(200, noise=0.15, seed=0)
X_moons_test, y_moons_test = make_moons(200, noise=0.15, seed=1)
neuron_model, neuron_losses, acc_train = train_neuron(X_moons, y_moons)
acc_test = float(((as_predictor(neuron_model)(X_moons_test) > 0.5) == y_moons_test).mean())
print(f"loss: {neuron_losses[0]:.3f} -> {neuron_losses[-1]:.3f} | train accuracy {acc_train:.3f} | test accuracy {acc_test:.3f}")
assert neuron_losses[-1] < neuron_losses[0], "training should lower the loss"
RESULTS["part3_moons_neuron"] = {"train": round(acc_train, 3), "test": round(acc_test, 3)}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(neuron_losses); axes[0].set_title("loss during training"); axes[0].set_xlabel("epoch")
plot_decision(as_predictor(neuron_model), X_moons, y_moons, ax=axes[1], title=f"one neuron on moons (train acc {acc_train:.2f})")
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
loss: 0.702 -> 0.267 | train accuracy 0.875 | test accuracy 0.825
```

   You should see `loss: 0.702 -> 0.267 | train accuracy 0.875 | test accuracy 0.825`. The loss starts near 0.70 (that is `-ln 0.5`: an untrained neuron says "50 %" about everything) and falls steadily. The plot shows a nearly horizontal straight boundary (the trained weights are about (1.28, −4.28) with a bias of 0.13, so the vertical position matters most): it separates the upper moon from the lower one reasonably well, but the two tips of the moons reach across the line, so 12.5 % of the training points and 17.5 % of the test points are on the wrong side. A single neuron cannot bend its line, and that is the limit you hit here.

   Tips and common mistakes:

   - `opt.zero_grad()` first, then forward, then `loss.backward()`, then `opt.step()`: this order is the same in every PyTorch training loop.
   - BCELoss expects probabilities, so the model ends with `nn.Sigmoid()`. If you remove the sigmoid the loss will complain about values outside 0..1.
   - The learning rate 0.5 is large for a network but fine for one neuron on data of this size. You will explore learning rates in Part 5.

   Try also (optional):

   - Change `epochs` to 30 and to 3000 and compare the train accuracy with 300: it is 0.810 after 30 epochs, 0.875 after 300 and 0.880 after 3000. It saturates because the straight line is the limit, not the training.
   - Train on `make_xor` instead of moons. A single neuron reaches only 0.43 on it (about chance); why is a straight line no better than chance on the four corner clusters?

   *Exercise 3.5:* a straight line is perfect when the classes really are separated by a line. Make a **linearly separable** data set yourself with NumPy (two blobs of 100 points: one around (−1, −1), one around (+1, +1), spread 0.5), train the same neuron on it with `train_neuron`, and store the training accuracy in `acc_blobs`. Compare it with the moons accuracy and write two sentences: what does this tell you about when one neuron is enough?

```python
acc_blobs = None   # train accuracy of one neuron on two well-separated blobs
```

6. **Imagine first, revisited.** Run the next block to compare your prediction with the facts.

```python
# Imagine first, revisited: the truth about one neuron
truth = {"AND": "yes", "OR": "yes", "XOR": "no"}
hits = sum(my_prediction.get(g) == t for g, t in truth.items())
RESULTS["part3_gate_predictions_right"] = hits
print(f"You predicted {hits} of the 3 gates correctly. The facts: {truth}")
print("Your boundary shape:", repr(boundary_shape) if boundary_shape else "(you did not write one; the right answer is: a straight line)")
```

   You should see how many of your three gate predictions were right (the facts are AND yes, OR yes, XOR no). Compare your sentence about the boundary shape with what you have seen in 3.2 to 3.5: one neuron always draws a **straight line** (in 2D; a flat plane in 3D), never a curve.

   Questions to answer in a markdown cell before you move on:

   - Write the formula of a neuron in one line. Which parts are learned and which are chosen?
   - What does the bias do to the decision line, and what do the weights do?
   - Why can one neuron do AND and OR but not XOR?
   - What would the 'dials' look like for a neuron with three inputs? (A plane instead of a line.)
   - What did the loss curve in 3.5 tell you about training before you saw the accuracy?

   A short write-up template you can paste into a markdown cell:

```text
Part 3 write-up
- My four neurons' outputs for the worked example: z = ... , sigmoid = ...
- The line of a neuron: the weights ... , the bias ...
- AND: w = ..., b = ...   OR: w = ..., b = ...   NAND: w = ..., b = ...
- Best XOR accuracy over 50,000 random neurons: ... because ...
- One neuron on moons: train ... , test ... ; on blobs: ... because ...
```

**Before you go on to the next part**, make sure you have:

- your answer to Exercise 3.1 (code the neuron and its four activations), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 3.2 (weights and bias as dials), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 3.3 (one neuron learns and and or), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 3.4 (one neuron cannot learn xor), written in a markdown cell or stored in the variables named in the exercise;
- your answer to Exercise 3.5 (train a single neuron with pytorch), written in a markdown cell or stored in the variables named in the exercise;
- your Imagine-first prediction for Part 3, and one sentence on how far it was from the facts.

### Part 4. Layers and stacking: a network from scratch (30 min)

*Lesson sections: "Layers" and "Hidden layers".* In Part 3 one neuron could only draw a straight line. Here you stack neurons side by side into a **layer**, and layers one after another into a **network**. You will see a layer as one matrix multiplication, build by hand a tiny network that solves XOR (the problem a single neuron cannot solve), prove that layers without an activation collapse into one, count a network's parameters, and finally design a digit reader on paper. ==A hidden layer re-codes the inputs so that the next layer needs only a straight line.==

**Imagine first.** Picture a team of **two specialist neurons** that both look at the same two inputs, followed by **one decider neuron** that only listens to the two specialists. No single member can draw anything but a straight line. Can the *team* solve XOR (output 1 when exactly one input is 1)? Write one or two sentences about what each specialist could be good at and what the decider would do with their answers, and write your yes/no prediction. You test it in Step 4.2.

```python
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

def sigmoid(z):
    """The sigmoid activation, for numbers or NumPy arrays."""
    return 1.0 / (1.0 + np.exp(-z))

team_idea = ""            # YOUR TURN: e.g. "One specialist notices 'at least one input is on', the other 'both are on' ..."
team_solves_xor = None    # YOUR TURN: True or False
print("Your idea:", team_idea if team_idea else "(nothing yet: write your sentence above and run this cell again)")
print("Your prediction:", team_solves_xor)
```

1. **Step 4.1. A layer is a matrix.** A **layer** is many neurons side by side, all reading the *same* inputs. Neuron number *i* has its own row of weights and its own bias. Stack the rows on top of each other and you get a **weight matrix** `W` with one row per neuron and one column per input, so the whole layer is one line of maths, `a = φ(Wx + b)`. The block builds a layer of 3 neurons reading 2 inputs. It computes the answer twice, once with a loop over the neurons (the way you built a neuron in Part 3) and once with a single matrix product, and an `assert` stops the cell if they disagree. Run it and read the shapes: they follow the rule *(out × in) times (in) gives (out)*.

```python
x = np.array([2.0, 1.0])                       # 2 inputs
W = np.array([ [ 1.0, -1.0],                    # neuron 0: weights for (x1, x2)
              [ 0.5,  2.0],                    # neuron 1
              [-1.5,  0.5] ])                   # neuron 2   -> shape (3, 2)
b = np.array([0.0, -1.0, 0.5])                 # one bias per neuron

# way 1: neuron by neuron
a_loop = np.array([sigmoid(np.dot(W[i], x) + b[i]) for i in range(3)])
# way 2: the whole layer at once
z = W @ x + b
a_matrix = sigmoid(z)

print("shapes: W", W.shape, " x", x.shape, " b", b.shape, " z", z.shape)
print("z (before the activation):", z)
print("a by loop  :", a_loop.round(3))
print("a by matrix:", a_matrix.round(3))
assert np.allclose(a_loop, a_matrix), "the loop and the matrix product should agree"
RESULTS["p4_layer_output"] = a_matrix.round(3).tolist()
```

   You should see `z = [ 1.  2. -2.]`, and `a = [0.731 0.881 0.119]` both ways. The weight matrix has shape (3, 2), the bias and `z` have 3 numbers.

   Try also (optional):

   - Print `W[1]` and say aloud which input neuron 1 cares about more, and whether its weights excite or inhibit.
   - Replace `sigmoid` by `np.tanh` in the layer and look at the three numbers: how do they differ in range?
   - Make a layer of 5 neurons reading 4 inputs with random numbers (`np.random.randn(5, 4)`) and print the shape of the answer.

   *Exercise 4.1:* change the first row of `W` (call the new row `row0`) and its bias `b0` so that neuron 0 gives **more than 0.9** for the same input `x = (2, 1)`, while neurons 1 and 2 stay as they are. Hint: you need `z > 2.2`, because `sigmoid(2.2)` is about 0.90. Then answer: what is the shape of the weight matrix of a layer with 5 neurons that reads 4 inputs? Why it matters: in every network you will meet, a layer is exactly this matrix, and its shape tells you at a glance how many neurons and inputs it has. Write in a markdown cell the row you chose, the value of `z` it gives and the new `a[0]`. One valid answer is `row0 = [1.5, 0.5]` and `b0 = 0`, giving `z = 3.5` and `a[0] = 0.971`; the 5-neuron, 4-input matrix has shape `(5, 4)`.

```python
row0 = [1.0, -1.0]     # YOUR CODE HERE: two weights for neuron 0
b0 = 0.0               # YOUR CODE HERE: its bias
shape_5x4 = None       # YOUR ANSWER: a tuple (rows, columns)
W_new = W.copy(); W_new[0] = row0
b_new = b.copy(); b_new[0] = b0
a_new = sigmoid(W_new @ x + b_new)
print("new layer output:", a_new.round(3))
check("neuron 0 now fires above 0.9", a_new[0] > 0.9, "you need 2*w1 + w2 + b0 > 2.2")
check("the other two neurons are unchanged", np.allclose(a_new[1:], a_matrix[1:]))
```

2. **Step 4.2. A hand-built network that solves XOR.** Now the team from the Imagine-first box. The network is `[2, 2, 1]`: 2 inputs, 2 hidden neurons, 1 output neuron, all with sigmoid activations, and we choose the weights by hand (they are the weights of the XOR network in the lesson). **Hidden neuron 1 acts like OR** (weights 20, 20, bias −10: it fires when `x1 + x2` is at least about 1). **Hidden neuron 2 acts like AND** (weights 20, 20, bias −30: it fires only when both inputs are 1). **The output neuron says "OR but not AND"** (weights +20 and −20, bias −10). Big weights such as 20 make the sigmoid almost a step, so the values are very close to 0 or 1. The block prints the hidden values and the output for all four inputs, and an `assert` checks the XOR truth table.

```python
W1 = np.array([ [20.0, 20.0],    # hidden 1 = OR
               [20.0, 20.0] ])   # hidden 2 = AND
b1 = np.array([-10.0, -30.0])
W2 = np.array([ [20.0, -20.0] ])  # output = hidden1 AND NOT hidden2   (shape 1 x 2)
b2 = np.array([-10.0])

def forward(x):
    """Return (hidden activations, output) for one input x of 2 numbers."""
    h = sigmoid(W1 @ x + b1)
    out = sigmoid(W2 @ h + b2)
    return h, out[0]

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
H_points = []
print("x1 x2 | hidden1 (OR)  hidden2 (AND) | output")
for x1, x2 in inputs:
    h, out = forward(np.array([x1, x2], dtype=float))
    H_points.append(h)
    print(f" {x1}  {x2} |   {h[0]:.4f}        {h[1]:.4f}      | {out:.4f}")
H_points = np.array(H_points)

xor_table = [round(forward(np.array(p, dtype=float))[1]) for p in inputs]
assert xor_table == [0, 1, 1, 0], "the hand-built network should compute XOR"
RESULTS["p4_xor_truth_table"] = xor_table
```

   You should see the output column read 0.0000, 1.0000, 1.0000, 0.0000: that is XOR. Look at the two hidden columns. For the four inputs the hidden pair is about (0, 0), (1, 0), (1, 0) and (1, 1). The hidden layer has **re-coded** the inputs: the two "1" examples now sit on the same spot, and a single straight line separates them from the two "0" examples. The next block draws the two pictures side by side.

```python
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
cols = ["tab:blue", "tab:orange", "tab:orange", "tab:blue"]           # XOR labels 0,1,1,0
for (x1, x2), c in zip(inputs, cols):
    axes[0].scatter([x1], [x2], c=c, s=120, edgecolor="k"); axes[0].annotate(f"({x1},{x2})", (x1 + 0.04, x2 + 0.04))
axes[0].set_title("Input space: no straight line works"); axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
axes[0].set_xlim(-0.3, 1.4); axes[0].set_ylim(-0.3, 1.4)
for (x1, x2), c, hp in zip(inputs, cols, H_points):
    axes[1].scatter([hp[0] ], [hp[1] ], c=c, s=120, edgecolor="k", alpha=0.8); axes[1].annotate(f"({x1},{x2})", (hp[0] + 0.03, hp[1] + 0.03))
xs = np.linspace(-0.2, 1.3, 50)
axes[1].plot(xs, xs - 0.5, "k--", lw=1)                                # the output neuron's line: h1 - h2 = 0.5
axes[1].set_title("Hidden space: one straight line works"); axes[1].set_xlabel("hidden 1 (OR)"); axes[1].set_ylabel("hidden 2 (AND)")
axes[1].set_xlim(-0.2, 1.3); axes[1].set_ylim(-0.3, 1.3)
plt.tight_layout(); plt.show()
```

   The left picture (input space) shows the four corners with the two classes on opposite diagonals, so no straight line can separate them. The right picture (hidden space) shows the same four examples after the hidden layer; the dashed line `h1 − h2 = 0.5` is the straight line that the output neuron draws.

   Try also (optional):

   - Change the bias of hidden neuron 2 from −30 to −10. What function does it now compute, and does the network still solve XOR?
   - Change the weights 20 to 2. The sigmoid is no longer a sharp step: print the output column and see how close to 0 and 1 it still gets.
   - Print `H_points` and write the four hidden pairs in your notebook, next to the four input pairs.

   *Exercise 4.2:* (a) Keep the same hidden layer and change **only the output neuron** (`W2_x`, `b2_x`) so that the network computes **XNOR**, the opposite of XOR (output 1 when the two inputs are equal). Hint: flip the signs of the two output weights, and choose the bias so that the hidden pair (0, 0) gives a positive `z`. (b) For the input (1, 1), which hidden neuron is on, and what does it do to the output? (c) Were you right in the Imagine-first box that a team of straight-line neurons can solve XOR? Write one sentence in a markdown cell. Why it matters: this is the whole idea of a hidden layer in miniature, and the picture of "re-coding" will return in Part 6. A correct answer is `W2_x = [ [-20, 20] ]` and `b2_x = [10]`; for (1, 1) both hidden neurons are on, and the AND neuron cancels the OR neuron in the output.

```python
W2_x = np.array([ [20.0, -20.0] ])     # YOUR CODE HERE: change the output weights
b2_x = np.array([-10.0])             # YOUR CODE HERE: change the output bias
def forward_x(x):
    h = sigmoid(W1 @ x + b1)
    return sigmoid(W2_x @ h + b2_x)[0]
xnor_table = [round(forward_x(np.array(p, dtype=float))) for p in inputs]
print("your table for (0,0) (0,1) (1,0) (1,1):", xnor_table, "  goal: [1, 0, 0, 1]")
check("the network computes XNOR", xnor_table == [1, 0, 0, 1], "flip the signs of W2 and make the bias positive")
```

3. **Step 4.3. Without an activation, depth gives nothing.** Why does the lesson insist on a *non-linear* activation between layers? Because two linear layers in a row are the same as one: `W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)`. The block builds a layer from 3 inputs to 4 neurons followed by a layer from 4 to 2, both **without** an activation, merges them into one matrix, and compares the answers. Then it puts a `tanh` in the middle and shows that the merge no longer works.

```python
rng = np.random.default_rng(0)
W1r, b1r = rng.normal(size=(4, 3)), rng.normal(size=4)
W2r, b2r = rng.normal(size=(2, 4)), rng.normal(size=2)
xr = rng.normal(size=3)

two_layers = W2r @ (W1r @ xr + b1r) + b2r           # layer 1 then layer 2, no activation
W_one, b_one = W2r @ W1r, W2r @ b1r + b2r           # ONE layer with these merged numbers
one_layer = W_one @ xr + b_one
print("merged matrix shape:", W_one.shape)
print("two layers:", two_layers.round(4), " one layer:", one_layer.round(4))
assert np.allclose(two_layers, one_layer), "two linear layers must equal one"

# with a tanh in the middle the merge no longer works
with_tanh = W2r @ np.tanh(W1r @ xr + b1r) + b2r
print("with tanh :", with_tanh.round(4), "  same as the merged layer?", np.allclose(with_tanh, one_layer))
RESULTS["p4_collapse_merged_shape"] = W_one.shape
```

   You should see two identical output vectors, `[1.961 2.5786]`, and a merged matrix of shape (2, 3). With the `tanh` in the middle the answer changes to `[1.6942 1.1502]`: the bend of the activation is what lets depth add power.

   Try also (optional):

   - Change the sizes to a 3 to 10 to 2 network and check that the merged matrix is still (2, 3).
   - Replace `np.tanh` by `np.maximum(0, ...)` (ReLU) in the middle and test `np.allclose` again. Does ReLU also break the merge? When might it not?

   *Exercise 4.3:* imagine the network `784 → 128 → 10` **with no activation**. (a) What is the shape of the single matrix that does the same job? (b) How many parameters does that one layer have, weights plus biases? (c) The real network has 101,770 parameters; in one sentence, what would the extra parameters buy you if there were no activation? Why it matters: it explains why every network in this course puts a ReLU, tanh or sigmoid after each hidden layer. You should get `(10, 784)` and `7,850` parameters, and the answer to (c) is "nothing: the whole network could only compute a straight-line (linear) map".

```python
merged_shape = None     # YOUR ANSWER: (rows, columns)
merged_params = None    # YOUR ANSWER: an integer
```

4. **Step 4.4. Count the parameters.** A layer with `n_in` inputs and `n_out` neurons has `n_in × n_out` weights and `n_out` biases, so the total for a network is a sum over its layers. The block writes `count_params`, builds the matching PyTorch `nn.Sequential` models, and uses an `assert` to make sure that the formula and PyTorch agree. Memory: every parameter is a 32-bit number, so **4 bytes**.

```python
def count_params(layers):
    """Parameters of a fully connected network with the given layer sizes, e.g. [2, 4, 1]."""
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))

def torch_model(layers):
    """The matching PyTorch model: Linear + ReLU between layers, Linear at the end."""
    mods = []
    for i in range(len(layers) - 1):
        mods.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2: mods.append(nn.ReLU())
    return nn.Sequential(*mods)

print(f"{'layers':<22}{'by formula':>12}{'by PyTorch':>12}{'memory (MB)':>14}")
for layers in ([2, 4, 1], [784, 128, 10], [784, 512, 512, 10]):
    mine = count_params(layers)
    theirs = sum(p.numel() for p in torch_model(layers).parameters())
    assert mine == theirs, f"{layers}: formula says {mine}, PyTorch says {theirs}"
    print(f"{str(layers):<22}{mine:>12,}{theirs:>12,}{mine * 4 / 1e6:>14.4f}")
RESULTS["p4_params_784_128_10"] = count_params([784, 128, 10])
```

   You should see 17 parameters for `[2, 4, 1]`, 101,770 for `[784, 128, 10]` and 669,706 for `[784, 512, 512, 10]`, which is 0.0001 MB, 0.4071 MB and 2.6788 MB. Notice where the parameters live: in `[784, 128, 10]` the first layer alone holds 784 × 128 + 128 = 100,480 of the 101,770.

   Try also (optional):

   - Make one layer 10 times wider, `[784, 1280, 10]`, and see by how much the count grows. Now make the network 10 layers of 128 and compare. Which kind of growth is more expensive?
   - Compute the memory of a network with 7 billion parameters at 4 bytes each (28 GB) and at 2 bytes each (14 GB), the size of a small language model.

   *Exercise 4.4:* (a) Use `count_params` for `[784, 1024, 1024, 10]` and compute the memory in MB. (b) Which single layer holds the most parameters? (c) How many times more parameters than `[784, 128, 10]` does it have? Why it matters: parameter counts decide memory and speed, and they will return in Week 3 and when you meet large language models. Write the three answers in a markdown cell. You should get 1,863,690 parameters (about 7.45 MB), the middle 1024 × 1024 layer (1,049,600 parameters), and about 18 times more.

```python
params_big = None      # YOUR CODE HERE: use count_params
mb_big = None          # YOUR CODE HERE
biggest_layer = None   # YOUR ANSWER: 0, 1 or 2
times_more = None      # YOUR ANSWER: an integer
```

5. **Step 4.5. Draw it, then design one.** `plot_network_diagram` draws a circle for every unit and a line for every weight, and returns the parameter count. The block draws three architectures and uses an `assert` to confirm that the returned count equals your `count_params`. Notice how quickly the lines multiply as the layers get wider.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, layers in zip(axes, ([2, 3, 1], [2, 4, 4, 1], [4, 8, 3])):
    n = plot_network_diagram(layers, ax=ax, title=f"{layers}\n{count_params(layers)} parameters")
    assert n == count_params(layers)
    ax.text(0, -(max(layers) / 2 + 0.6) * 1.2, "input", ha="center"); ax.text(len(layers) - 1, -(max(layers) / 2 + 0.6) * 1.2, "output", ha="center")
    if len(layers) > 2: ax.text((len(layers) - 1) / 2, -(max(layers) / 2 + 0.6) * 1.2, "hidden", ha="center")
plt.tight_layout(); plt.show()
```

   Try also (optional):

   - Draw `[784, 128, 10]` with `plot_network_diagram` capped at 8 circles per layer and write on paper how many circles the real picture would have.
   - Count the parameters of a network with 3 inputs, 2 hidden layers of 5 neurons and 1 output by hand, then check with `count_params([3, 5, 5, 1])` (you should get 56).

   *Exercise 4.5:* a digit picture is 28 × 28 = 784 numbers and the answer is one of 10 digits. Design a network on paper first: write the layer sizes, for example `[784, 128, 64, 10]`, and then compute its parameters and memory with the code below. In a markdown cell (3 to 4 sentences) say what you expect each layer to do (for instance "the first hidden layer might detect small strokes and edges, the second might combine them into loops and bars"), why the output layer has exactly 10 neurons (one score per digit, turned into probabilities by softmax), and why a bigger network is not automatically better (more memory and time, and more risk of overfitting, which you meet in Part 6). Why it matters: choosing layer sizes is the first design decision of every project, and you must be able to justify it with numbers. For `[784, 128, 64, 10]` you should get 109,386 parameters, about 0.44 MB.

```python
my_digit_net = [784, 64, 10]     # YOUR CODE HERE: change the hidden sizes, add a layer if you like
my_params = count_params(my_digit_net)
print(f"{my_digit_net}: {my_params:,} parameters = {my_params * 4 / 1e6:.2f} MB")
check("starts with 784 and ends with 10", my_digit_net[0] == 784 and my_digit_net[-1] == 10, "one input per pixel, one output per digit")
RESULTS["p4_my_digit_net"] = my_digit_net
RESULTS["p4_my_digit_params"] = my_params
```

   **Before you move on (Part 4).** Check that you can say each of these in your own words: a layer is `φ(Wx + b)` and `W` has shape (out, in); the hand-built network solved XOR because its hidden layer re-coded the inputs; two linear layers without an activation equal one layer; and the parameters of a layer are `in × out + out`.

6. **Imagine first, revisited.** Run the next block to compare your Imagine-first prediction with the facts.

```python
print("Your prediction (team can solve XOR):", team_solves_xor)
print("The hand-built [2,2,1] network computes XOR:", xor_table == [0, 1, 1, 0])
print("Output of the four inputs:", xor_table)
RESULTS["p4_team_prediction"] = team_solves_xor
```

   In a markdown cell write what each of your two specialists ended up doing (compare with OR and AND), and what the decider did.

### Part 5. Learning: gradient descent you can watch (30 min)

*Lesson section: "Training".* A new network has random weights and is useless. **Training** finds good weights by repeating three steps: *guess, measure the loss, nudge every weight a little downhill.* You will take one step by hand and then twenty, run a learning-rate experiment that shows four different behaviours, check a hand-written gradient against PyTorch, and then train a whole `[2, 4, 1]` network in pure NumPy while you watch its decision boundary bend into shape. ==The learning rate is the size of the step: too small and you crawl, too large and you overshoot.==

**Imagine first.** Training walks downhill on the loss landscape, and the learning rate `η` is the size of each step. Suppose `η` is much too large. Sketch in words what the loss curve (loss on the vertical axis, step number on the horizontal axis) would look like, choose one of `"smooth fall"`, `"crawls"`, `"bounces then settles"` or `"explodes"`, and give a reason. You test it in Step 5.2.

```python
lr_prediction = "smooth fall"   # YOUR TURN: change to "crawls", "bounces then settles" or "explodes" if you disagree
lr_reason = ""                  # YOUR TURN: one sentence, e.g. "A huge step jumps over the valley and lands higher up than before."
print("Prediction for a huge learning rate:", lr_prediction, "|", lr_reason if lr_reason else "(add a reason and run again)")
```

1. **Step 5.1. One weight, one step by hand.** The smallest model is `ŷ = w · x`. One training example has `x = 2` and true answer `y = 4`, so the perfect weight is `w = 2`. We start at `w = 0.5`. The loss is the squared error `L = (w·x − y)²`, its slope with respect to `w` is `dL/dw = 2 (ŷ − y) x`, and the update rule is `w ← w − η · dL/dw`. With `η = 0.1`: the prediction is `ŷ = 1`, the loss is 9, the gradient is `2 (1 − 4) · 2 = −12`, and the new weight is `0.5 − 0.1 · (−12) = 1.7`. The slope was negative (the loss falls as `w` grows), so the update *increased* `w`. The block does this step, then walks twenty steps and draws the loss curve and the walk down the parabola.

```python
x_t, y_t = 2.0, 4.0

def loss_1w(w):
    return (w * x_t - y_t) ** 2

def grad_1w(w):
    return 2 * (w * x_t - y_t) * x_t

w, eta = 0.5, 0.1
print(f"start: w={w}  prediction={w * x_t}  loss={loss_1w(w)}  gradient={grad_1w(w)}")
w = w - eta * grad_1w(w)
print(f"after ONE step: w={w:.4f}  loss={loss_1w(w):.4f}")
assert abs(w - 1.7) < 1e-9, "the lesson's worked step gives 1.7"

# now twenty steps, remembering where we walked
w, path = 0.5, [0.5]
for step in range(20):
    w = w - eta * grad_1w(w)
    path.append(w)
print("w after steps 0, 1, 2, 3, 5, 10, 20:", [round(path[i], 4) for i in (0, 1, 2, 3, 5, 10, 20)])
RESULTS["p5_w_after_20"] = round(path[-1], 6)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot([loss_1w(p) for p in path], "o-"); axes[0].set_yscale("log")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss (log scale)"); axes[0].set_title("loss per step")
grid = np.linspace(-0.5, 4.5, 200)
axes[1].plot(grid, loss_1w(grid), "k-")
axes[1].plot(path[:8], [loss_1w(p) for p in path[:8] ], "o-", c="tab:red")
axes[1].set_xlabel("w"); axes[1].set_ylabel("loss"); axes[1].set_title("the first 7 steps walk down the parabola")
plt.tight_layout(); plt.show()
```

   You should see: after one step `w = 1.7` and the loss has fallen from 9 to 0.36. The error `w − 2` is multiplied by 0.2 at every step (because `1 − η · 8 = 0.2`), so `w` is 2.0000 after 20 steps and the loss is practically zero. Early steps are big (the slope is steep), later steps tiny: gradient descent slows down by itself as it nears the bottom. The left picture is a straight line on a log scale, the right shows red dots hopping down the parabola.

   Try also (optional):

   - Start from `w = 5.0` instead of `0.5`. Which direction does the first step go, and why? (The gradient is positive there, so `w` decreases.)
   - Print the gradient at every step. Does it get bigger or smaller as `w` nears 2, and what does that do to the size of the steps?

   *Exercise 5.1:* (a) Do one step by hand for `x = 3`, `y = 6`, `w = 1.0` and `η = 0.05`: write the prediction, the gradient and the new weight, then check with Python. (b) With the problem above (`η = 0.1`, start `w = 0.5`), write a loop with a counter that finds how many steps are needed before the loss is below 0.001. Why it matters: every training run in this course is this step repeated millions of times, and doing it by hand once makes `loss.backward()` and `optimizer.step()` less mysterious. You should get `w = 1.9` for (a) (prediction 3, gradient −18), and 3 steps for (b) (the losses are 9, 0.36, 0.0144, 0.000576).

```python
w_next = None     # YOUR ANSWER: the new weight after one step
n_steps = None    # YOUR CODE HERE: loop with a counter until loss_1w(w) < 0.001
```

2. **Step 5.2. The learning-rate experiment.** Same problem, same start, six learning rates in a row. For this problem the error `w − 2` is multiplied at every step by `1 − 8η` (8 is the curvature `2x²`), and that single number explains everything. If `|1 − 8η|` is below 1 the error shrinks: `w` converges, smoothly when `1 − 8η` is positive and hopping from side to side when it is negative. If it is exactly 1 in size (`η = 0.25`) the error never shrinks and `w` jumps between two values for ever. If it is above 1 (`η > 0.25`) each step overshoots by more than the last: the loss **explodes**. The stability limit is `η = 2 / (2x²) = 1/x² = 0.25`. The block plots five learning rates on one chart and prints the factor and the final loss for each.

```python
def run_gd(eta, steps=20, w0=0.5):
    """Return the loss at every step of gradient descent on the one-weight problem."""
    w, losses = w0, [loss_1w(w0)]
    for _ in range(steps):
        w = w - eta * grad_1w(w)
        losses.append(loss_1w(w))
    return losses

etas = [0.001, 0.05, 0.2, 0.25, 0.3]
notes = {0.001: "crawls", 0.05: "converges", 0.2: "bounces, then settles", 0.25: "bounces forever", 0.3: "explodes"}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for eta in etas:
    L = run_gd(eta)
    ax.plot(L, "o-", ms=3, label=f"η = {eta}  ({notes[eta]})")
    print(f"η = {eta:<6} factor 1-8η = {1 - 8 * eta:+.2f}   loss after 20 steps = {L[-1]:.3g}")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss (log scale)"); ax.set_title("Four behaviours of gradient descent")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

limit = 2 / (2 * x_t ** 2)
print("stability limit 2 / (2 x^2) =", limit)
assert abs(limit - 0.25) < 1e-12
assert run_gd(0.3)[-1] > run_gd(0.3)[0], "η = 0.3 should blow up"
assert run_gd(0.05)[-1] < 1e-6, "η = 0.05 should converge"
RESULTS["p5_stability_limit"] = limit
```

   You should see: `η = 0.001` crawls (loss still 6.53 after 20 steps); `η = 0.05` (factor +0.60) and `η = 0.2` (factor −0.60, hopping from side to side) both end at about 1.2e-08; `η = 0.25` stays at loss 9 for ever; and `η = 0.3` (factor −1.40) has reached 6.3e+06 by step 20 and keeps growing. The assertions confirm that 0.3 blows up and 0.05 converges.

   Try also (optional):

   - Run `run_gd(0.25)` and print the first six values of `w` instead of the loss: you should see 0.5, 3.5, 0.5, 3.5 ... bouncing between two values.
   - Try `eta = 0.12` and `eta = 0.249`. Both converge, but one rings like a bell and the other creeps. Which is which, and what is their factor `1 − 8η`?
   - Plot the *same* five learning rates but for `x = 1`. Where is the new limit?

   *Exercise 5.2:* (a) Suppose the training example is `x = 3`, `y = 6`. Using the idea above, what is the largest learning rate that still converges? (b) Test it: write a function like `run_gd` for that problem and run it just below and just above your limit. (c) In one sentence: why must the learning rate depend on the size of the inputs? Why it matters: it explains why people scale their inputs before training, and why a learning rate that worked in one project can explode in another. You should get `2 / (2 · 3²) = 0.111`; just below (×0.95) the final loss is about 0.04 after 30 steps and still shrinking, just above (×1.05) it is about 6,000 and growing.

```python
limit_x3 = None            # YOUR ANSWER: a number
def run_gd_xy(eta, x, y, steps=30, w0=0.5):
    w = w0
    for _ in range(steps):
        w = w - eta * 2 * (w * x - y) * x
    return (w * x - y) ** 2          # the final loss
```

3. **Step 5.3. Two weights and a bias, checked against `autograd`.** A real neuron has several weights. For one sigmoid neuron with prediction `p = σ(w1 x1 + w2 x2 + b)` and cross-entropy loss, calculus gives a simple result, averaged over the examples: `dL/dw1 = (p − y) · x1`, `dL/dw2 = (p − y) · x2`, `dL/db = (p − y)`. In words: *the error times the input.* The block computes this by hand on six points and compares with PyTorch's automatic gradients (`loss.backward()`); an `assert` stops the cell if they differ.

```python
# six labelled points: the label is 1 when x1 + x2 is large
Xs = np.array([ [0.0, 0.5], [1.0, 0.0], [0.5, 1.0], [2.0, 1.5], [1.5, 2.0], [2.5, 2.0] ])
ys = np.array([0, 0, 0, 1, 1, 1], dtype=float)

def loss_and_grads(w, b, X, y):
    """Mean cross-entropy and the gradients, written by hand."""
    p = sigmoid(X @ w + b)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    err = p - y                          # (p - y) for every example
    return loss, X.T @ err / len(y), err.mean()

w0, b0 = np.array([0.3, -0.2]), 0.1
loss0, gw, gb = loss_and_grads(w0, b0, Xs, ys)

# the same numbers in PyTorch
wt = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
bt = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
pt = torch.sigmoid(torch.tensor(Xs) @ wt + bt)
lt = nn.functional.binary_cross_entropy(pt, torch.tensor(ys))
lt.backward()

print("loss  by hand:", round(loss0, 6), "  PyTorch:", round(lt.item(), 6))
print("dL/dw by hand:", gw.round(6), "  PyTorch:", wt.grad.numpy().round(6))
print("dL/db by hand:", round(gb, 6), "  PyTorch:", round(bt.grad.item(), 6))
assert np.allclose(gw, wt.grad.numpy()) and abs(gb - bt.grad.item()) < 1e-9, "hand gradients must match autograd"
RESULTS["p5_grad_check"] = "hand-written gradients match autograd"
```

   You should see a loss of 0.658641 in both columns, `dL/dw = [-0.269158 -0.257222]` and `dL/db = 0.059657`, identical for hand and PyTorch. This is the idea of backpropagation in miniature: the maths has a pattern, so a computer can produce every gradient for you.

   Try also (optional):

   - Change one label in `ys` from 1 to 0 so that the data can no longer be separated, and look at how the final loss changes.
   - Add a third input to the neuron (a third column of `Xs` and a third weight) and extend `loss_and_grads`; the check against autograd should still pass.

   *Exercise 5.3:* the next block trains the neuron for 200 steps with four learning rates. Run it, read the final losses, and choose the learning rate you trust most (it must be one of the tried values). In a markdown cell explain why you did not pick the smallest or the largest. Why it matters: choosing a learning rate by looking at the loss is what practitioners do every day. You should see final losses of about 0.535, 0.222, 0.032 and 0.0026 for 0.01, 0.1, 1 and 10. This data set is tiny and its inputs are small, so even 10 is stable here; with bigger inputs it would explode (Step 5.2).

```python
def train_neuron(lr, steps=200):
    w, b = np.array([0.3, -0.2]), 0.1
    for _ in range(steps):
        _, gw_, gb_ = loss_and_grads(w, b, Xs, ys)
        w, b = w - lr * gw_, b - lr * gb_
    return w, b, loss_and_grads(w, b, Xs, ys)[0]

lrs = [0.01, 0.1, 1.0, 10.0]
final = {}
for lr in lrs:
    w_, b_, L_ = train_neuron(lr)
    final[lr] = L_
    print(f"lr = {lr:<6} final loss = {L_:.4f}   w = {w_.round(2)}  b = {b_:.2f}")

best_lr = None     # YOUR ANSWER: one of the values in lrs
best_why = None    # YOUR ANSWER: one sentence
```

4. **Step 5.4. A whole network in pure NumPy.** Now the real thing: the `[2, 4, 1]` network (2 inputs, 4 `tanh` hidden units, 1 `sigmoid` output) learns the XOR data set. Everything is plain NumPy, including the **backward pass** that produces the gradient of every weight. The function `backward` is short and heavily commented: it is the same "error times input" idea as in Step 5.3, passed backwards one layer at a time. ==You do not need to memorize backpropagation. PyTorch does it for you in Part 6 with one call, `loss.backward()`.== The last lines of the block prove that the hand-written `backward` is right by comparing it with PyTorch on the same numbers.

```python
def init_net(hidden, seed=0):
    """Small random weights. Rows of W1 = inputs, columns = hidden units."""
    r = np.random.RandomState(seed)
    return {"W1": r.normal(0, 1, (2, hidden)), "b1": np.zeros(hidden),
            "W2": r.normal(0, 1, (hidden, 1)), "b2": np.zeros(1)}

def forward_np(p, X):
    """Forward pass: returns the hidden activations A1 and the output probabilities P."""
    A1 = np.tanh(X @ p["W1"] + p["b1"])                 # hidden layer: tanh(XW + b)
    P = sigmoid(A1 @ p["W2"] + p["b2"]).ravel()         # output layer: sigmoid
    return A1, P

def bce_np(P, y):
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(P) + (1 - y) * np.log(1 - P)))

def backward(p, X, y):
    """Gradient of the mean cross-entropy with respect to every parameter."""
    A1, P = forward_np(p, X)
    d2 = ((P - y) / len(y))[:, None]                    # output error: (p - y), averaged over the examples
    g = {"W2": A1.T @ d2, "b2": d2.sum(0)}              # "error times input": the input of the output layer is A1
    d1 = (d2 @ p["W2"].T) * (1 - A1 ** 2)               # send the error back through W2, times tanh'(z) = 1 - tanh(z)^2
    g["W1"] = X.T @ d1                                  # again "error times input": the input is X
    g["b1"] = d1.sum(0)
    return g

def train_np(X, y, hidden=4, lr=0.5, epochs=1500, seed=0, snap_at=(0, 50, 300, 1500)):
    """Full-batch gradient descent. Returns the final parameters, the loss per epoch and snapshots of the parameters."""
    p = init_net(hidden, seed)
    losses, snaps = [], {}
    for epoch in range(epochs + 1):
        losses.append(bce_np(forward_np(p, X)[1], y))
        if epoch in snap_at:
            snaps[epoch] = {k: v.copy() for k, v in p.items()}
        g = backward(p, X, y)
        for k in p:
            p[k] -= lr * g[k]                           # the update rule: w <- w - lr * gradient
    return p, losses, snaps

# first, prove that backward() is right: compare with PyTorch autograd on the same numbers
Xg, yg = make_xor(20, 0.12, seed=5)
pg = init_net(4, seed=1)
tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in pg.items()}
h = torch.tanh(torch.tensor(Xg, dtype=torch.float64) @ tp["W1"] + tp["b1"])
pr = torch.sigmoid(h @ tp["W2"] + tp["b2"]).squeeze(1)
nn.functional.binary_cross_entropy(pr, torch.tensor(yg, dtype=torch.float64)).backward()
mine = backward(pg, Xg, yg)
for k in pg:
    assert np.allclose(mine[k], tp[k].grad.numpy(), atol=1e-8), f"gradient of {k} is wrong"
print("hand-written backward() matches PyTorch autograd for W1, b1, W2, b2")
```

   Now train on 200 XOR points, take snapshots at epochs 0, 50, 300 and 1500, and draw the four decision-region pictures in a row, plus the loss curve.

```python
Xx, yx = make_xor(200, 0.12, seed=0)
params, losses, snaps = train_np(Xx, yx, hidden=4, lr=0.5)
acc_np = float(((forward_np(params, Xx)[1] > 0.5) == yx).mean())
print(f"loss: epoch 0 = {losses[0]:.3f}, epoch 50 = {losses[50]:.3f}, epoch 300 = {losses[300]:.3f}, epoch 1500 = {losses[1500]:.3f}")
print(f"final training accuracy: {acc_np:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, e in zip(axes, (0, 50, 300, 1500)):
    snap = snaps[e]
    a = float(((forward_np(snap, Xx)[1] > 0.5) == yx).mean())
    plot_decision(lambda Z, s=snap: forward_np(s, Z)[1], Xx, yx, ax=ax, title=f"epoch {e}: loss {losses[e]:.2f}, acc {a:.2f}")
plt.tight_layout(); plt.show()

plt.figure(figsize=(5.5, 3.4)); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("training loss"); plt.show()
assert losses[-1] < losses[0] * 0.5, "the loss should at least halve"
assert acc_np > 0.88, "the network should solve XOR (the data are noisy, so not 100 %)"
RESULTS["p5_numpy_acc"] = round(acc_np, 3)
RESULTS["p5_numpy_final_loss"] = round(losses[-1], 3)
```

   You should see the loss fall from 0.824 (epoch 0) to 0.569 (epoch 50), 0.185 (epoch 300) and 0.136 (epoch 1500). Training accuracy goes from 0.545 at epoch 0 to 0.73 at epoch 50, 0.925 at epoch 300 and **0.935** at epoch 1500. It never reaches 1.0 because `make_xor` adds noise: a few points sit on the wrong side of the corner they belong to, because the labels were set before the noise moved the points.

   Try also (optional):

   - Use `make_circles(200)` instead of `make_xor` and train a `[2, 4, 1]` network. Does the same code work? How does the boundary look at epoch 50?
   - Replace `np.tanh` and its derivative `1 - A1 ** 2` by a sigmoid (derivative `A1 * (1 - A1)`), keeping everything else. Does it still train, and is it faster or slower?
   - Re-run the training with a different `seed` for `init_net`. Do the pictures at epoch 50 look the same? What changes and what stays the same?

   *Exercise 5.4:* change `hidden` and `lr` and fill in a table of final loss and accuracy. Try at least `hidden = 1`, `hidden = 2`, `hidden = 4` and `hidden = 8`, and one learning rate that is too small (`lr = 0.05`). Write in a markdown cell the smallest number of hidden units that reaches an accuracy of at least 0.88, and what you saw with the small learning rate. Why it matters: it shows with your own numbers that a single hidden unit cannot do XOR, and that learning rate and capacity are separate knobs. You should get: 1 unit loss 0.578, accuracy 0.680; 2 units loss 0.371, accuracy 0.815; 4 units loss 0.136, accuracy 0.935; 8 units loss 0.132, accuracy 0.940; and with 4 units and `lr = 0.05` the loss is still 0.274 after 1500 epochs (accuracy 0.900), so training is simply slower.

```python
settings = [(1, 0.5), (2, 0.5), (4, 0.5), (4, 0.05), (8, 0.5)]    # YOUR TURN: add your own pairs
table = {}
for hidden, lr in settings:
    pm, ls, _ = train_np(Xx, yx, hidden=hidden, lr=lr, epochs=1500, snap_at=())
    a = float(((forward_np(pm, Xx)[1] > 0.5) == yx).mean())
    table[(hidden, lr)] = (ls[-1], a)
    print(f"hidden = {hidden:<2} lr = {lr:<5} final loss = {ls[-1]:.3f}   accuracy = {a:.3f}")
smallest_hidden = None   # YOUR ANSWER: an integer
RESULTS["p5_numpy_acc"] = round(acc_np, 3)
```

5. **Step 5.5. Observe and write.** Look again at the four decision-region pictures from Step 5.4 (epochs 0, 50, 300, 1500). In a markdown cell describe in 3 or 4 sentences what the boundary looked like **early** and **late**, what the colours do, and where the network still makes mistakes. Use the words *boundary*, *loss* and *epoch*. Why it matters: reading a decision-region picture is the skill that lets you see what a network has learned, and it is how you will spot underfitting and overfitting in Part 6. A good answer says that at epoch 0 the boundary is a nearly random line that cuts the picture in two (accuracy about 0.55); by epoch 50 it has started to bend; by epoch 300 it follows the four corners; and at epoch 1500 it is a smooth cross, with the remaining mistakes being the noisy points that lie next to it.

```python
boundary_note = None   # YOUR ANSWER: 3 or 4 sentences
RESULTS["p5_boundary_note"] = boundary_note
```

   **Before you move on (Part 5).** You should be able to say: the update rule is `w ← w − η · gradient`; for the one-weight problem the safe learning rate is below `1/x²`; the gradient of the cross-entropy for a sigmoid neuron is the error `(p − y)` times the input; and a network starts with a nearly random boundary that bends into shape as the loss falls.

   Try also (optional): look at the snapshots of the four epochs as numbers, `snaps[50]["W1"]`, and compare them with `snaps[1500]["W1"]`. Which weights changed the most? Large weights mean steep boundaries.

6. **Imagine first, revisited.** Run the next block to compare your learning-rate prediction with the facts.

```python
print("You predicted for a huge learning rate:", lr_prediction)
print("Loss after 20 steps with eta = 0.3:", f"{run_gd(0.3)[-1]:.3g}", "(started at 9)")
RESULTS["p5_lr_prediction"] = lr_prediction
```

   In a markdown cell write whether you were right. A huge learning rate does not give a smooth fall: each step jumps over the valley and lands higher than before, so the loss explodes (or, exactly at the limit, jumps between two values for ever).


### Part 6. The same thing in PyTorch, then break it (25 min)

*Lesson sections: "Training" and "Generalization".* In Part 5 you wrote the backward pass yourself. PyTorch does it for you, so now you can ask the interesting questions: *how big should a network be?* and *what happens when it is too big or trained too long?* You will train a network with the standard five-line loop, sweep the number of hidden units, watch a big network **overfit** a tiny noisy data set, fix it in three ways, and finally split data honestly into train, validation and test. ==A model is only good if it works on data it has never seen.==

**Imagine first.** The `make_circles` data set is a small disc of class 1 inside a ring of class 0. A hidden layer with `tanh` units cuts the plane with one smooth line per unit. How many hidden units does the network need to separate disc from ring well? Choose a number from 1, 2, 4, 8 or 32 and write one sentence of reasoning (think: how many straight cuts does it take to fence in a round shape?). You test it in Step 6.2.

```python
h_guess = 2          # YOUR TURN: 1, 2, 4, 8 or 32
h_reason = ""        # YOUR TURN: one sentence
print("My guess:", h_guess, "hidden units.", h_reason if h_reason else "(add a reason and run again)")
```

1. **Step 6.1. The standard training loop.** Build `nn.Sequential(nn.Linear(2, H), nn.Tanh(), nn.Linear(H, 1))`. The last layer has **no sigmoid**: the loss `BCEWithLogitsLoss` applies the sigmoid inside, which is numerically safer. The optimiser **Adam** is a popular, forgiving version of gradient descent. Every training loop in the world has the same five lines, annotated in the block. A separate test set is made with a different seed (`seed=1`), so the network has never seen those points.

```python
def make_mlp(H, depth=1):
    """2 inputs -> `depth` hidden layers of H tanh units -> 1 output (a logit, not yet a probability)."""
    layers, n_in = [], 2
    for _ in range(depth):
        layers += [nn.Linear(n_in, H), nn.Tanh()]
        n_in = H
    return nn.Sequential(*layers, nn.Linear(n_in, 1))

def accuracy(model, X, y):
    with torch.no_grad():
        return float(((model(torch.tensor(X)).squeeze(1) > 0).float().numpy() == y).mean())

def prob_fn(model):
    """Wrap a model so that plot_decision can call it: (N, 2) array in, probabilities out."""
    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(Z))).squeeze(1).numpy()
    return predict

Xc, yc = make_circles(200, seed=0)           # training data
Xc_test, yc_test = make_circles(400, seed=1) # a separate test set from a different seed
Xt_, yt_ = torch.tensor(Xc), torch.tensor(yc)

torch.manual_seed(SEED)
model = make_mlp(8)
loss_fn = nn.BCEWithLogitsLoss()                         # sigmoid + cross-entropy in one
opt = torch.optim.Adam(model.parameters(), lr=0.05)      # the optimiser: knows the parameters and the learning rate
epochs = 150 if QUICK else 400
history = []
for epoch in range(epochs):
    opt.zero_grad()                                      # 1. forget the gradients from the last step
    logits = model(Xt_).squeeze(1)                       # 2. forward pass: predictions for all 200 points
    loss = loss_fn(logits, yt_)                          # 3. measure the loss
    loss.backward()                                      # 4. backpropagation: fill in every parameter's gradient
    opt.step()                                           # 5. nudge every parameter downhill
    history.append(loss.item())

train_acc, test_acc = accuracy(model, Xc, yc), accuracy(model, Xc_test, yc_test)
print(f"loss: first {history[0]:.3f} -> last {history[-1]:.4f}   train acc {train_acc:.3f}   test acc {test_acc:.3f}")
assert history[-1] < history[0] * 0.5, "the loss should go down a lot"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(history); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss"); axes[0].set_title("loss curve (H = 8)")
plot_decision(prob_fn(model), Xc, yc, ax=axes[1], title=f"circles: train {train_acc:.2f}, test {test_acc:.2f}")
plt.tight_layout(); plt.show()
RESULTS["p6_circles_h8_test_acc"] = round(test_acc, 3)
```

   You should see the loss fall from about 0.715 (a coin toss would be `ln 2 = 0.693`) to about 0.004, training accuracy of 1.000 and test accuracy of about 0.988. The decision region is a roundish blob around the disc. Because the weights start random-ish, your last digits can differ a little. The `assert` checks that the loss at least halves.

   Try also (optional):

   - Replace `Adam` by `torch.optim.SGD(model.parameters(), lr=0.5)` and look at the loss curve. How many epochs does it need for the same loss?
   - Print `model[0].weight.shape` and `model[2].weight.shape`. They should be (8, 2) and (1, 8): the same (out, in) rule as in Part 4.

   *Exercise 6.1:* (a) Write one short phrase for each of the five lines in the loop (`zero_grad`, forward, loss, `backward`, `step`): what does it do and what would go wrong without it? (b) If you removed `opt.zero_grad()`, would training still work? Try it in a copy of the loop. (c) Explain in one sentence why the first loss is close to 0.693. Why it matters: when training goes wrong, the cause is very often one of these five lines in the wrong order or missing. Write your answers in a markdown cell. A good answer: `zero_grad` clears the old gradients, because PyTorch adds new gradients to the old ones; forward computes predictions; loss scores the mistakes; `backward` computes the gradient of every weight; `step` moves every weight a little downhill. Without `zero_grad` the gradients pile up and training goes wrong. At the start the model says about 0.5 for everything, and `−ln 0.5 = ln 2 = 0.693`.

2. **Step 6.2. Capacity sweep: how many hidden units are enough?** The block writes `train_model` (the same five-line loop in a function, which also returns the loss on an evaluation set after every epoch) and trains networks with `H = 1, 2, 4, 8, 32` hidden units on both the circles and the moons data sets. It records train and test accuracy for each and draws a 2 × 5 grid of decision regions.

```python
def train_model(model, Xtr, ytr, Xev, yev, epochs=400, lr=0.05, wd=0.0):
    """Train with Adam. Returns the per-epoch training loss and evaluation loss (on Xev, yev), plus the best state seen."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t, Xev_t, yev_t = (torch.tensor(a) for a in (Xtr, ytr, Xev, yev))
    tr, ev, best = [], [], (1e9, 0, None)
    for epoch in range(epochs):
        opt.zero_grad()
        l = lf(model(Xtr_t).squeeze(1), ytr_t); l.backward(); opt.step()
        tr.append(l.item())
        with torch.no_grad():
            ev.append(lf(model(Xev_t).squeeze(1), yev_t).item())
        if ev[-1] < best[0]:
            best = (ev[-1], epoch, {k: v.clone() for k, v in model.state_dict().items()})
    return tr, ev, best

sweep_epochs = 150 if QUICK else 400
Hs = [1, 2, 4, 8, 32]
sweep = {}
fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
for row, (name, maker) in enumerate([("circles", make_circles), ("moons", make_moons)]):
    Xtr, ytr = maker(200, seed=0)
    Xte, yte = maker(400, seed=1)
    for col, H in enumerate(Hs):
        torch.manual_seed(SEED)
        m = make_mlp(H)
        train_model(m, Xtr, ytr, Xte, yte, epochs=sweep_epochs)
        sweep[(name, H)] = (accuracy(m, Xtr, ytr), accuracy(m, Xte, yte))
        plot_decision(prob_fn(m), Xtr, ytr, ax=axes[row, col], title=f"{name}, H={H}\ntrain {sweep[(name, H)][0]:.2f}  test {sweep[(name, H)][1]:.2f}")
plt.tight_layout(); plt.show()

print(f"{'data':<9}{'H':>4}{'train acc':>11}{'test acc':>10}")
for (name, H), (a_tr, a_te) in sweep.items():
    print(f"{name:<9}{H:>4}{a_tr:>11.3f}{a_te:>10.3f}")
assert sweep[("circles", 32)][1] > sweep[("circles", 1)][1] + 0.1, "more hidden units should clearly help on circles"
RESULTS["p6_sweep_circles_test"] = {H: round(sweep[("circles", H)][1], 3) for H in Hs}
RESULTS["p6_sweep_moons_test"] = {H: round(sweep[("moons", H)][1], 3) for H in Hs}
```

   You should see: on circles, test accuracy climbs from 0.667 (H = 1) to 0.818 (H = 2), 0.978 (H = 4), 0.988 (H = 8) and 0.993 (H = 32); on moons, H = 1 and 2 reach 0.890 and 0.877, and H = 4, 8 and 32 reach 0.990. So **4 hidden units** is the smallest number that works well on both data sets in our runs, and 8 and 32 add very little. One unit draws one smooth cut, two units draw a wedge, four units can fence in a round shape. Your decimals can vary a little with the PyTorch version.

   Try also (optional):

   - Change `make_mlp` to use `nn.ReLU()` instead of `nn.Tanh()` and repeat the sweep. Is the smallest H that works still 4?
   - Train `H = 4` for only 20 epochs and for 2000 epochs. Which pictures change, and which accuracy changes?
   - Add the data set `make_xor` to the sweep. How many hidden units does XOR need in PyTorch?

   *Exercise 6.2:* (a) From the printed table, find for each data set the smallest H whose test accuracy is at least 0.95. (b) Was your Imagine-first guess too small, right or too big? (c) In one or two sentences, explain why a network with H = 1 cannot solve circles however long you train it. Why it matters: choosing the size of a network is a design decision, and the sweep is how you make it with evidence. Write the answers in a markdown cell; store (a) in `smallest_circles` and `smallest_moons`. You should get 4 and 4. Capacity is the kind of shapes a network can draw; too little capacity means underfitting, and no amount of training cures it.

3. **Step 6.3. Overfitting you can see.** Now the opposite problem. We give a **big** network (3 hidden layers of 256 units, 132,609 parameters) a **tiny, noisy** training set: only 30 points of moons with a lot of noise (0.35). We also make a validation set (300 points) and a test set (1000 points) from other seeds, and record the loss on the training set and the validation set after each epoch. The network can only learn from the 30 points; if it learns their *noise*, it does worse on new points.

```python
Xtr3, ytr3 = make_moons(30, noise=0.35, seed=3)
Xva3, yva3 = make_moons(300, noise=0.35, seed=4)
Xte3, yte3 = make_moons(1000, noise=0.35, seed=5)

torch.manual_seed(SEED)
big = make_mlp(256, depth=3)
print("parameters in the big network:", f"{sum(p.numel() for p in big.parameters()):,}")
tr3, va3, best3 = train_model(big, Xtr3, ytr3, Xva3, yva3, epochs=600, lr=0.005)
best_epoch = int(np.argmin(va3))
print(f"lowest validation loss {va3[best_epoch]:.3f} at epoch {best_epoch}; at epoch 599 the validation loss is {va3[-1]:.3f}")
print(f"final accuracies: train {accuracy(big, Xtr3, ytr3):.3f}  validation {accuracy(big, Xva3, yva3):.3f}  test {accuracy(big, Xte3, yte3):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(tr3, label="training loss"); axes[0].plot(va3, label="validation loss")
axes[0].axvline(best_epoch, c="k", ls=":", label=f"best epoch = {best_epoch}")
axes[0].set_ylim(0, 2.5); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("the two curves part ways")
plot_decision(prob_fn(big), Xtr3, ytr3, ax=axes[1], lim=2.5, title="after 600 epochs: a wiggly boundary")
plt.tight_layout(); plt.show()
assert va3[-1] > va3[best_epoch] * 1.5, "validation loss should end clearly above its minimum (overfitting)"
assert tr3[-1] < 0.05, "the big network should memorise the 30 training points"
RESULTS["p6_overfit_best_epoch"] = best_epoch
RESULTS["p6_overfit_test_acc_end"] = round(accuracy(big, Xte3, yte3), 3)
```

   You should see: the training loss sinks towards zero (below 0.001) and training accuracy is 1.000. The validation loss reaches its lowest point, 0.392, very early, at epoch 23, and then climbs steadily to 1.641 at epoch 599, four times higher. The boundary has become a wiggly shape that wraps around single training points. Validation accuracy is 0.767 and test accuracy only 0.739. That gap between 1.00 on training data and 0.74 on new data is **overfitting**. The dotted line marks the *early-stopping point*, the epoch where the validation loss was lowest.

   Try also (optional):

   - Draw the decision regions at epoch 23 and at epoch 599 side by side (train two copies of the network, one for 24 epochs and one for 600). Which boundary would you rather trust?
   - Use `make_moons(30, noise=0.1, seed=3)` instead of noise 0.35. Does the network still overfit? What does that say about noise and overfitting?

   *Exercise 6.3:* (a) Store the epoch of the lowest validation loss in `stop_epoch` (use `np.argmin`) and that loss, rounded to 3 decimals, in `stop_loss`. (b) By how many times is the final validation loss bigger than the lowest? (c) In two sentences explain what the network "learned" after the best epoch, using the words *noise* and *training*. Why it matters: the training loss alone would tell you everything is perfect; only data the model has not seen reveal the problem. You should get epoch 23, loss 0.392 and a factor of 4.2 (it may differ slightly on your machine).

```python
stop_epoch = int(np.argmin(va3))        # YOUR CODE HERE (already filled: check that you understand it)
stop_loss = round(min(va3), 3)
blowup = None                           # YOUR CODE HERE: va3[-1] divided by the lowest loss, one decimal
overfit_note = None                     # YOUR ANSWER: two sentences
print(stop_epoch, stop_loss, blowup)
```

4. **Step 6.4. Fix it: three remedies.** We try three fixes on the same problem and compare the **test accuracy**, averaged over 5 random starting weights so that one lucky start does not mislead us. (1) **More data:** 150 training points instead of 30, same big network. (2) **Smaller network:** only 4 hidden units in one layer (17 parameters) plus a little weight decay. (3) **Early stopping:** keep the big network but use the weights from the epoch with the lowest *validation* loss. The baseline is the big network trained for 600 epochs on the 30 points. The stopping epoch is chosen with the **validation** set, never with the test set.

```python
def one_run(seed, n_train=30, H=256, depth=3, wd=0.0, epochs=600, lr=0.005, early=False):
    """Train one network (random start `seed`) and return its test accuracy. The 30-point training set is the one from 6.3."""
    Xtr, ytr = make_moons(n_train, noise=0.35, seed=3)       # n_train = 30 gives exactly the set of step 6.3
    torch.manual_seed(seed)
    m = make_mlp(H, depth)
    _, _, best = train_model(m, Xtr, ytr, Xva3, yva3, epochs=epochs, lr=lr, wd=wd)
    if early:
        m.load_state_dict(best[2])            # go back to the weights of the best validation epoch
    return accuracy(m, Xte3, yte3)

n_seeds = 2 if QUICK else 5
recipes = {
    "baseline: 30 points, big network": dict(),
    "1. more data: 150 points":         dict(n_train=150),
    "2. smaller net: H=4 + weight decay": dict(H=4, depth=1, wd=0.01, lr=0.02),
    "3. early stopping (validation)":   dict(early=True),
}
remedy = {}
for name, kw in recipes.items():
    accs = [one_run(s, **kw) for s in range(n_seeds)]
    remedy[name] = float(np.mean(accs))
    print(f"{name:<38} mean test accuracy = {remedy[name]:.3f}   (runs: {', '.join(f'{a:.2f}' for a in accs)})")
RESULTS["p6_remedies"] = {k: round(v, 3) for k, v in remedy.items()}
base = remedy["baseline: 30 points, big network"]
assert max(remedy.values()) > base + 0.02, "at least one remedy should beat the baseline"
```

   You should see: baseline 0.754; more data 0.855; smaller network with weight decay 0.822; early stopping 0.778. The best possible accuracy on this noisy data set is only about 0.87 to 0.89, because the two moons overlap. Early stopping helps least here: by the best epoch the network has seen the 30 points only briefly, and 30 noisy points simply do not contain much information. These are averages over 5 random starts on the same 30 points; another training set gives different values, and single runs jump by a few percent.

   Try also (optional):

   - Combine two remedies, for example early stopping on 150 points. Is the sum better than the best single remedy?
   - Change `wd=0.01` to `0.1` and to `0.0` for the small network and look at the mean test accuracy.

   *Exercise 6.4:* name the remedy with the highest mean test accuracy, and then, in two sentences, say which remedy you would choose if you could **not** collect more data, and what each of the other two costs. Why it matters: in real projects you choose among imperfect fixes, and every fix has a price. A good answer: more data wins here, but it costs time and money to collect and label; without more data I would choose a smaller network or early stopping, because both are free, but early stopping needs a validation set and a smaller network can underfit if it is made too small.

5. **Step 6.5. A fair test.** Why do we need a **validation** set *and* a **test** set? The validation set is used while building (to choose epochs, hidden units, learning rate). The more choices we make by looking at it, the more it quietly becomes part of training. The **test set** is locked away and used once, at the very end, for an honest score. The block makes a 60 / 20 / 20 split using shuffled indices and a leakage check that no example is in two sets, then smuggles 5 training examples into the validation set on purpose to show the check working.

```python
Xall, yall = make_moons(500, noise=0.2, seed=11)
n = len(Xall)
perm = np.random.RandomState(SEED).permutation(n)             # shuffled indices 0..n-1
n_train, n_val = int(0.6 * n), int(0.2 * n)
idx_train, idx_val, idx_test = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
print("sizes:", len(idx_train), len(idx_val), len(idx_test))

def leakage(a, b):
    """The indices that appear in both sets (should be empty)."""
    return set(a.tolist()) & set(b.tolist())
assert not leakage(idx_train, idx_val) and not leakage(idx_train, idx_test) and not leakage(idx_val, idx_test), "overlap found!"
assert len(idx_train) + len(idx_val) + len(idx_test) == n
print("no overlap: every example is in exactly one set")

# a leak on purpose
bad_val = np.concatenate([idx_val, idx_train[:5] ])
print("5 train examples smuggled into validation -> overlap size:", len(leakage(idx_train, bad_val)))
```

   You should see sizes `300 100 100`, the message "no overlap", and an overlap of size 5 for the deliberate leak.

   Try also (optional):

   - Use `sklearn`-free stratified splitting: make sure that each of the three sets contains both classes (count `yall[idx_train].mean()`, and the same for the others). Why could a split with only one class in the test set be misleading?
   - Shuffle *before* splitting versus splitting the original order: for time-ordered data (rain by date) which one is right, and why?

   *Exercise 6.5:* (a) Split a data set of `n = 1000` examples 70 / 15 / 15 with shuffled indices and store the three sizes in `sizes_1000`. (b) Write a function `has_leak(a, b)` that returns `True` when two index arrays share any example; test it on a clean and a leaky split. (c) In two sentences explain why we do not simply use the test set to choose the best number of epochs. Why it matters: leakage is the most common reason a model looks great in the lab and fails in real life. You should get `[700, 150, 150]`; for (c) a good answer says that choosing by the test score tunes the model to the test set, so its score is no longer an honest estimate for new data.

```python
sizes_1000 = None   # YOUR ANSWER: [train, val, test]
def has_leak(a, b):
    return False    # YOUR CODE HERE: return True if a and b share an index
p1000 = np.random.RandomState(0).permutation(1000)
a_, b_, c_ = p1000[:700], p1000[700:850], p1000[850:]
check("clean split has no leak", has_leak(a_, b_) is False and has_leak(a_, c_) is False)
check("leaky split is caught", has_leak(a_, np.concatenate([b_, a_[:3] ])) is True, "use set(a) & set(b)")
```

   **Before you move on (Part 6).** You should be able to say: the five lines of the training loop in order; that too few hidden units underfit (circles with H = 1) and a huge network trained too long overfits (30 points, 600 epochs); that early stopping, more data and a smaller network are three remedies; and that the test set is touched once, at the end.

6. **Imagine first, revisited.** Run the next block to compare your hidden-unit guess with the table.

```python
ok = [H for H in Hs if sweep[("circles", H)][1] >= 0.95]
print(f"You guessed {h_guess} hidden units. The smallest that reaches 0.95 test accuracy on circles is {min(ok) if ok else None}.")
RESULTS["p6_h_guess"] = h_guess
RESULTS["p6_sweep_circles_test"] = {H: round(sweep[("circles", H)][1], 3) for H in Hs}
RESULTS["p6_overfit_test_acc_end"] = round(accuracy(big, Xte3, yte3), 3)
RESULTS["p6_remedies"] = {k: round(v, 3) for k, v in remedy.items()}
```

   In a markdown cell write one sentence: were you too small, right or too big, and what surprised you?

### Part 7. Design your own classifier and reflect (25 min)

*Lesson sections: all of them.* Now you are the designer. You choose a real yes/no problem from your school or community, write the **spec** of a classifier for it, build a small synthetic version, train it with PyTorch, stress-test it with bad labels, collect your results and reflect. ==A classifier is only as good as its labels and the test that proves it.==

**Imagine first.** Which yes/no problem around you would a classifier help with? Some ideas: spam or not spam text messages; ripe or unripe mangoes; a flooded or clear road; a jeepney that is crowded or not; whether the canteen will sell out of rice meals today. Keep your choice in mind for Step 7.1.

1. **Step 7.1. Imagination design: the spec.** Fill in the spec before you build anything. It must name at least two numeric features (numbers a computer could measure), one label, where the data would come from, how many examples you would need, the network shape (use `count_params`), the loss, how you would test it, and one fairness or data risk. A complete example (flooded or clear roads) is filled in: replace every entry with your own idea.

```python
spec = {
    "problem": "Is this road flooded or clear? (yes = flooded)",                                      # YOUR TURN: replace
    "features": ["rainfall in the last 3 hours (mm)", "water level of the nearest creek (cm)"],       # at least 2 numbers
    "label": "1 = flooded, 0 = clear (decided by a barangay volunteer who checks the road)",
    "data_source": "Volunteers record the two numbers and the road state every 30 minutes during rain for one rainy season.",
    "n_examples": 600,
    "network_shape": [2, 8, 1],
    "loss": "binary cross-entropy (BCE), because the answer is yes/no",
    "how_to_test": "Split by date, not at random: train on June to August, validate on September, test on October, so the test days are really new.",
    "risk": "Only roads near the creek were measured, so the model may fail on roads in other places, and a sensor failure during a storm would produce wrong labels.",
}
print(f"Network {spec['network_shape']} has {count_params(spec['network_shape'])} parameters.")
for k, v in spec.items():
    print(f"  {k}: {v}")
```

   Write your spec in plain words first, on paper, before you touch the code: a classifier you cannot describe in a paragraph is not ready to be coded. The helper `count_params` from Part 4 tells you the size of the network you wrote down.

   *Exercise 7.1:* replace all nine entries with your own problem. Then in a markdown cell answer three questions: what could go wrong with your labels (who decides and could they be wrong or biased?), why your `how_to_test` plan keeps the test examples really new, and what the risk entry means for a real person who would be affected. Why it matters: most failures of real AI systems start in the spec, not in the code. A strong answer has at least 2 features, a network whose first number equals the number of features and whose last is 1, a loss that fits yes/no (binary cross-entropy), and a risk of at least a sentence that names who is affected. The example gives a network of `[2, 8, 1]` with 33 parameters.

2. **Step 7.2. Build and train a synthetic version.** We have no real flood or jeepney data, so we **invent** a data set with NumPy: two Gaussian blobs, one per class. The numbers `mean0`, `mean1` and `spread` describe your problem (where a typical "clear road" and a typical "flooded road" sit in feature space, and how much they overlap). Make the blobs overlap a little: real data are never perfectly separable. The block makes 200 points, splits them 120 / 40 / 40 with shuffled indices (it asserts that there is no overlap), trains a small PyTorch network and plots the decision regions on the test points. It reuses `make_mlp`, `train_model` and `accuracy` from Part 6.

```python
mean0 = np.array([-0.6, -0.5])    # YOUR TURN: typical feature values for class 0 (e.g. 'clear road')
mean1 = np.array([0.8, 0.7])      # YOUR TURN: typical feature values for class 1 (e.g. 'flooded')
spread = 0.7                      # YOUR TURN: bigger = more overlap = harder problem

def make_my_data(n, seed):
    """n points, half from each blob, as float32 arrays (X: n x 2, y: 0/1)."""
    r = np.random.RandomState(seed)
    m = n // 2
    X = np.concatenate([r.normal(mean0, spread, (m, 2)), r.normal(mean1, spread, (n - m, 2))]).astype(np.float32)
    y = np.concatenate([np.zeros(m), np.ones(n - m)]).astype(np.float32)
    return X, y

X7, y7 = make_my_data(200, seed=7)
perm7 = np.random.RandomState(SEED).permutation(200)
tr_i, va_i, te_i = perm7[:120], perm7[120:160], perm7[160:]
Xtr7, ytr7, Xva7, yva7, Xte7, yte7 = X7[tr_i], y7[tr_i], X7[va_i], y7[va_i], X7[te_i], y7[te_i]
assert not (set(tr_i) & set(va_i)) and not (set(tr_i) & set(te_i)) and not (set(va_i) & set(te_i))

torch.manual_seed(SEED)
net7 = make_mlp(8)
tr7, va7, _ = train_model(net7, Xtr7, ytr7, Xva7, yva7, epochs=200 if QUICK else 300, lr=0.02)
accs7 = {"train": accuracy(net7, Xtr7, ytr7), "validation": accuracy(net7, Xva7, yva7), "test": accuracy(net7, Xte7, yte7)}
print({k: round(v, 3) for k, v in accs7.items()})

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
axes[0].plot(tr7, label="train loss"); axes[0].plot(va7, label="validation loss"); axes[0].legend(); axes[0].set_xlabel("epoch")
plot_decision(prob_fn(net7), Xte7, yte7, ax=axes[1], lim=2.5, title=f"test points: accuracy {accs7['test']:.2f}")
plt.tight_layout(); plt.show()
RESULTS["p7_accuracies"] = {k: round(v, 3) for k, v in accs7.items()}
```

   You should see, with the default blobs: training accuracy 0.917, validation 0.825 and test 0.950. These three numbers disagree by more than you might expect because the validation and test sets have only 40 points each, and one wrong point moves the score by 2.5 percentage points. The boundary is a smooth diagonal line between the two clouds with a few points on the wrong side: that overlap is the part no model can fix.

   Try also (optional):

   - Move `mean1` so that the blobs overlap almost completely, for instance `[0.2, 0.1]`. What is the best test accuracy you can get, and does a bigger network help?
   - Give the classes different sizes (for example 180 points of class 0 and 20 of class 1) and compute the accuracy of a model that always answers 0.

   *Exercise 7.2:* (a) Change `spread` to 0.4 and 1.2 (and keep 0.7), train again on 400 fresh test points, and record the three pairs (spread, test accuracy) in a dictionary `spread_table`. (b) In two sentences explain why the test accuracy is lower than the training accuracy, and why a single 40-point test set is a weak measure. Why it matters: the amount of overlap in your data sets a ceiling on accuracy that no network can pass. You should get about 0.985 for spread 0.4, 0.875 for spread 0.7 and 0.745 for spread 1.2. Hint for (a): write a function `accuracy_for_spread(s)` that sets the global `spread`, builds the data with `make_my_data`, trains a `make_mlp(8)` for 300 epochs with `lr=0.02` and returns the accuracy on `make_my_data(400, seed=8)`.

3. **Step 7.3. Stress test: label noise.** Real labels are made by people or sensors, and some are wrong. We **flip 10 % of the training labels** (12 of 120) on purpose, train a larger network (`make_mlp(32, depth=2)`) that is able to memorise them, and measure the test accuracy on a clean test set of 1000 points. To be fair we repeat with 5 seeds and average.

```python
Xbig, ybig = make_my_data(1000, seed=99)          # clean test set for the stress test

def noisy_vs_clean(seed, flip=0.10):
    r = np.random.RandomState(seed)
    Xa, ya = make_my_data(120, seed=seed + 20)
    ynoisy = ya.copy()
    k = int(round(flip * len(ya)))
    flipped = r.choice(len(ya), k, replace=False)
    ynoisy[flipped] = 1 - ynoisy[flipped]
    out = []
    for labels in (ya, ynoisy):
        torch.manual_seed(seed)
        m = make_mlp(32, depth=2)
        train_model(m, Xa, labels, Xbig, ybig, epochs=300, lr=0.01)
        out.append(accuracy(m, Xbig, ybig))
    return out

runs = [noisy_vs_clean(s) for s in range(3 if QUICK else 5)]
clean_acc = float(np.mean([r[0] for r in runs])); noisy_acc = float(np.mean([r[1] for r in runs]))
print("per seed (clean, noisy):", [(round(a, 3), round(b, 3)) for a, b in runs])
print(f"mean test accuracy with clean labels: {clean_acc:.3f}   with 10 % flipped labels: {noisy_acc:.3f}   drop: {clean_acc - noisy_acc:.3f}")
assert clean_acc > 0.8, "the clean model should work"
RESULTS["p7_clean_acc"] = round(clean_acc, 3)
RESULTS["p7_noisy_acc"] = round(noisy_acc, 3)
```

   You should see: mean test accuracy with clean labels 0.877 and with 10 % flipped labels 0.810, a drop of about 0.067; every one of the 5 seeds lost between 0.06 and 0.08. The network is large enough to memorise the wrong labels, so it learns some of them.

   Try also (optional):

   - Flip the labels of only the points that are *close to the boundary* instead of random points. Is the damage bigger or smaller than for random flips?
   - Train the same noisy data with `make_mlp(4)` (a small network). Does a small network resist label noise better than the large one?

   *Exercise 7.3:* (a) Compute the average drop (clean minus noisy) rounded to 3 decimals. (b) Run `noisy_vs_clean(seed, flip=0.30)` for 3 seeds and compute the new average drop: is it bigger or smaller? (c) In one sentence say why clean labels matter even more than a fancy network. Why it matters: the model learns exactly what the labels say. You should get a drop of 0.067 for 10 % noise and about 0.24 for 30 % noise (clean 0.88, noisy about 0.65); a good sentence says that wrong labels teach the model wrong rules and no architecture can fix a wrong teacher.

4. **Step 7.4. Collect your results.** Everything you stored in `RESULTS` (in all parts) is printed together. Make sure that you also store your own writing there, for example `RESULTS["p7_spec"] = spec`, `RESULTS["p5_boundary_note"] = boundary_note` and your `reflection`. Skipped parts will show up as missing entries, so go back and run them.

```python
def show_value(v, limit=140):
    s = str(v)
    return s if len(s) <= limit else s[:limit] + " ..."

print("=" * 60)
print("MY WEEK 2 RESULTS")
print("=" * 60)
for key in sorted(RESULTS):
    print(f"{key:<30} {show_value(RESULTS[key])}")
print("=" * 60)
print(f"{len(RESULTS)} items collected.")
mine = [k for k in RESULTS if k[:2] in ("p4", "p5", "p6", "p7")]
assert len(mine) >= 20, "Parts 4 to 7 should have stored at least 20 items; run all their cells"
```

   *Exercise 7.4:* look at the summary and pick the three numbers from this lab that you think are most important (for example the smallest `H` that works, the validation-loss minimum of the overfit run, the accuracy drop with noisy labels). In a markdown cell write one sentence for each saying what it tells you. You should see roughly 45 to 50 entries.

5. **Step 7.5. Reflection.** Answer the four questions below in 2 or 3 sentences each, referring to something you actually saw in this lab (a number, a picture, a table). (1) What surprised you most this week, and why? (2) What does a **hidden layer** actually do? Use the XOR network of Step 4.2 as your example. (3) When would you trust a model's 95 % accuracy, and when would you not? Think about the test set, class balance and who is affected. (4) How do **supervised learning** (this week) and the **generative** idea of Week 1 differ, and what do they share? Why it matters: being able to say in plain words what a model does, and when to doubt it, is the skill that lasts. Hints: for (2) say that the OR and AND neurons turned the four corners into new coordinates where one straight line separates the classes; for (3) a 95 % score only counts if it was measured on a test set the model never saw, if the classes are not very uneven (a model that always says "no" is 95 % right when only 5 % of the cases are "yes"), and if the test data resemble the real use; for (4) both start from data, both use a loss and gradient descent, but supervised learning compares with a correct label while a generative model learns the distribution of the data itself.

```python
reflection = {
    "1_surprised": None,
    "2_hidden_layer": None,
    "3_trust_95": None,
    "4_supervised_vs_generative": None,
}
```

**Before you finish.** Save the notebook with its outputs. If you worked on the shared Spark, remove large variables you do not need (`del big, model, net7; gc.collect()`) and, if you used the GPU, call `torch.cuda.empty_cache()`.

**Before you move on (Part 7).** You should be able to say: what your classifier is for and what it learns from; why the spec comes before the code; how overlap between classes limits accuracy; and why label noise and a tiny test set make numbers untrustworthy.

**Looking ahead.** In Week 3 you meet the tools that make all this practical, the PyTorch ecosystem and the data pipeline that feeds a network. The five-line loop of Step 6.1 is the one you will keep using.

### What you should see

- Step 4.1: shapes (3, 2), (2,), (3,); `z = [1, 2, -2]` and `a = [0.731, 0.881, 0.119]` by loop and by matrix.
- Step 4.2: the output column of the hand-built network is 0, 1, 1, 0 (rounded); the hidden pair for the four inputs is about (0,0), (1,0), (1,0), (1,1). The XNOR network gives 1, 0, 0, 1.
- Step 4.3: both ways give `[1.961 2.5786]` and the merged matrix has shape (2, 3); with `tanh` the answer is `[1.6942 1.1502]`. For `784 → 128 → 10` without an activation the merged layer is (10, 784) with 7,850 parameters.
- Step 4.4: 17, 101,770 and 669,706 parameters (0.0001, 0.4071 and 2.6788 MB); `[784, 1024, 1024, 10]` has 1,863,690 (about 7.45 MB, 18 times more than `[784, 128, 10]`).
- Step 5.1: after one step `w = 1.7` and the loss falls from 9 to 0.36; after 20 steps `w = 2.0000`; the exercise gives `w = 1.9` and 3 steps.
- Step 5.2: the factor `1 − 8η` is +0.99, +0.60, −0.60, −1.00, −1.40 for `η` = 0.001, 0.05, 0.2, 0.25, 0.3; final losses 6.53, 1.2e-08, 1.2e-08, 9 and 6.3e+06; the stability limit is 0.25 (0.111 for `x = 3`).
- Step 5.3: loss 0.658641, gradients `[-0.269158 -0.257222]` and `0.059657`, equal for hand and PyTorch; final losses 0.535, 0.222, 0.032, 0.0026 for learning rates 0.01, 0.1, 1, 10.
- Step 5.4: loss 0.824, 0.569, 0.185, 0.136 at epochs 0, 50, 300, 1500; accuracy 0.545, 0.73, 0.925, 0.935; accuracy 0.680 / 0.815 / 0.935 / 0.940 for 1 / 2 / 4 / 8 hidden units.
- Step 6.1: loss from 0.715 to about 0.004; train accuracy 1.000, test accuracy about 0.988.
- Step 6.2: circles test accuracy 0.667, 0.818, 0.978, 0.988, 0.993 and moons 0.890, 0.877, 0.990, 0.990, 0.990 for H = 1, 2, 4, 8, 32; the smallest H with at least 0.95 is 4 for both.
- Step 6.3: 132,609 parameters; lowest validation loss 0.392 at epoch 23, final validation loss 1.641; train accuracy 1.000, validation 0.767, test 0.739.
- Step 6.4: mean test accuracy 0.754 (baseline), 0.855 (more data), 0.822 (smaller net + weight decay), 0.778 (early stopping).
- Step 6.5: sizes 300 / 100 / 100 and an overlap of 5 for the deliberate leak; 700 / 150 / 150 for 1000 examples.
- Step 7.1: the example spec has `[2, 8, 1]`, 33 parameters, and passes all checks.
- Step 7.2: train 0.917, validation 0.825, test 0.950; test accuracy 0.985, 0.875 and 0.745 for spread 0.4, 0.7 and 1.2.
- Step 7.3: clean 0.877, noisy 0.810, drop about 0.067 for 10 % flipped labels; about 0.24 for 30 %.
- Step 7.4: a printed list of every item in `RESULTS`.

Where your numbers differ in the last decimals, that is normal: PyTorch and NumPy versions and processors give small differences. Where they differ a lot, re-run from the Setup cell so the seed is set again.

### Common mistakes

- Reading `W` the wrong way round. A layer with `n_out` neurons and `n_in` inputs has a weight matrix of shape `(n_out, n_in)`, and `W @ x` needs `x` to have `n_in` numbers.
- Forgetting the bias when counting parameters: it is `n_in × n_out + n_out`, not `n_in × n_out`. `[784, 128, 10]` has 101,770 parameters, not 101,632.
- Using the same variable for the hand-built XOR network's weights and for the random merge test in Step 4.3: the code uses `W1r`, `W2r` for the random ones, so that the XOR network from Step 4.2 keeps working.
- Choosing a learning rate by luck. If the loss grows or turns into `nan`, the learning rate is above the stability limit; divide it by 10 and try again.
- Forgetting `opt.zero_grad()`: PyTorch adds new gradients to the old ones, so the steps become wrong and the loss misbehaves.
- Putting a `sigmoid` in the model **and** using `BCEWithLogitsLoss`: the loss already applies the sigmoid, so the output of the last `nn.Linear` should be used as it is. For plotting, `prob_fn` applies the sigmoid once.
- Judging a model by its training accuracy. The big network in Step 6.3 has 1.000 training accuracy and 0.74 test accuracy.
- Choosing the number of epochs, or any other setting, by looking at the **test** set. Use the validation set; touch the test set once.
- Leaking examples from the training set into the validation or test set (for example by shuffling after you copy the data). Always split by indices and run the overlap check.
- Believing a single 40-point test score. One wrong point moves it by 2.5 percentage points; use a bigger test set, or repeat with several seeds and average.
- Concluding that "10 % wrong labels cost exactly 7 %": the drop depends on the data, the network and the seed. Say "in our run".

### Deliverables

A short checklist before you submit: every exercise has a written answer in a markdown cell under its block; every `None` in a starter cell has been replaced; the notebook runs from the first cell to the last without an error; and the printed `RESULTS` summary of Step 7.4 is the last output.

Submit the notebook with all outputs visible, the way your instructor asks. For Parts 4 to 7 it must include:

- **Part 4 (16 points):** the layer outputs and shapes (4.1), your XNOR network and your answer about the team of specialists (4.2), the collapse test and your answers about the merged layer (4.3), your parameter counts and memory (4.4), and your **digit-reader design** with its parameter count and your explanation (4.5).
- **Part 5 (16 points):** the worked steps (5.1), the learning-rate plot with your written explanation of the four behaviours and the stability limit (5.2), the gradient check and your learning-rate choice (5.3), the NumPy network's loss curve, the four decision-region pictures and your table of hidden units and learning rates (5.4), and your description of early and late boundaries (5.5).
- **Part 6 (15 points):** the PyTorch loop with your annotations (6.1), the capacity sweep and the smallest H that works (6.2), the overfitting curves with the best epoch and your note (6.3), the table of three remedies and your choice (6.4), and the 60/20/20 split with the leakage check (6.5).
- **Part 7 (15 points):** your **spec** (7.1), the synthetic classifier with its train/validation/test accuracies and decision picture (7.2), the label-noise measurements (7.3), the printed `RESULTS` summary (7.4), and your **four reflection answers** (7.5).

Together with Parts 1 to 3 the lab is marked out of **100 points**: Part 1 is worth 12, Part 2 is worth 12, Part 3 is worth 14, Part 4 is worth 16, Part 5 is worth 16, Part 6 is worth 15 and Part 7 is worth 15. The rubric is in the section "How Week 2 is assessed". The graded notebook is the one your instructor specifies.

**Stretch goals.** Train the network of Part 5.4 on `make_circles` and `make_moons` and compare how many hidden units each needs; replace `tanh` with `nn.ReLU()` in `make_mlp` and see whether the sweep of Step 6.2 changes; add a third set of curves to Step 6.3 that uses weight decay `1e-2` and compare the validation minimum; try `torch.optim.SGD` instead of Adam in Step 6.1 and find a learning rate that works; for your own spec in Step 7.1, change the class balance to 90 % / 10 % and show that accuracy alone can hide a useless model (a model that always answers "no" is 90 % right).
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions, with an explanation after each one. Seven of them need a calculation, so have a calculator ready and use the natural logarithm (ln) where a question mentions it.

```quiz
{
  "id": "week-02-check",
  "title": "Week 2 quiz: Supervised Learning and Neural Networks",
  "passMark": 70,
  "questions": [
    {
      "q": "In supervised learning, what is a label?",
      "options": [
        "A number inside the model that training adjusts",
        "The correct answer attached to each training example",
        "The model’s own prediction for a new example",
        "One of the input numbers given to the model"
      ],
      "answer": 1,
      "why": "The label is the correct answer supplied with each example; it is the teacher the loss compares predictions against. Numbers the model adjusts are parameters, and inputs are features."
    },
    {
      "q": "You have 1,200 labelled examples and split them 70% training, 15% validation and 15% test. How many examples are in the test set?",
      "options": [
        "120",
        "150",
        "180",
        "360"
      ],
      "answer": 2,
      "why": "15% of 1,200 is 0.15 × 1,200 = 180. (Training gets 840 and validation 180.) 360 would be 30%, which is validation and test together."
    },
    {
      "q": "Why should the test set be used only once, at the very end?",
      "options": [
        "Repeated use would leak test information into your choices, so its score would look better than the model really is",
        "Using it more than once makes the model train more slowly",
        "The test set is too small to give a score the second time",
        "The model forgets its weights when it is tested twice"
      ],
      "answer": 0,
      "why": "If you keep adjusting the model to improve its test score you are tuning to the test set, and it no longer gives an honest estimate. The validation set is for those choices. Testing does not change the weights or the speed of training."
    },
    {
      "q": "A neuron has inputs x = (1, 3), weights w = (2, −1), bias b = 0.5, and a ReLU activation. What is its output?",
      "options": [
        "−0.5",
        "0.5",
        "1.5",
        "0"
      ],
      "answer": 3,
      "why": "z = 2×1 + (−1)×3 + 0.5 = −0.5. ReLU is max(0, z), and the maximum of 0 and −0.5 is 0. The value −0.5 is z, before the activation."
    },
    {
      "q": "A model predicts (3, 4, 4) for true values (2, 4, 6). What is the mean squared error?",
      "options": [
        "1.00",
        "1.67",
        "5.00",
        "−0.33"
      ],
      "answer": 1,
      "why": "The errors are +1, 0 and −2, so the squares are 1, 0 and 4. Their sum is 5 and the mean is 5/3 = 1.67. 1.00 is the mean absolute error (1+0+2)/3, 5.00 forgets to divide by 3, and −0.33 is the mean of the signed errors, where plus and minus cancel."
    },
    {
      "q": "In a two-class problem the true class is spam and the model gives it probability 0.5. What is the cross-entropy loss −ln p?",
      "options": [
        "0.301",
        "0.500",
        "0.693",
        "2.303"
      ],
      "answer": 2,
      "why": "−ln 0.5 = 0.693. The value 0.301 is the base-10 logarithm, which is not the loss we use, and 2.303 is −ln 0.1, the loss when the model gave the true class only a 10% chance."
    },
    {
      "q": "How many parameters (weights and biases) does a network with layer sizes [4, 5, 3] have?",
      "options": [
        "43",
        "35",
        "60",
        "12"
      ],
      "answer": 0,
      "why": "Layer 1 has 4×5 + 5 = 25 parameters and layer 2 has 5×3 + 3 = 18, so the total is 43. The value 35 forgets the biases (20 + 15), and 60 is 4×5×3, which multiplies instead of adding the layers."
    },
    {
      "q": "A softmax layer outputs probabilities from the scores (2.0, 1.0, 0.1). What probability does the first class get, to two decimal places?",
      "options": [
        "0.33",
        "0.50",
        "0.90",
        "0.66"
      ],
      "answer": 3,
      "why": "e² = 7.389, e¹ = 2.718, e^0.1 = 1.105, which sum to 11.212. The first class gets 7.389 / 11.212 = 0.659, or 0.66. The value 0.33 is what three equal scores would give, and 0.90 is roughly the first two classes together (0.659 + 0.242)."
    },
    {
      "q": "A one-weight model ŷ = w × x has w = 1, x = 3 and label y = 6. The loss is (ŷ − y)² and the learning rate is 0.05. What is w after one gradient descent step?",
      "options": [
        "0.1",
        "1.9",
        "−17",
        "19"
      ],
      "answer": 1,
      "why": "ŷ = 3, so the gradient is 2(ŷ − y)x = 2 × (−3) × 3 = −18. The update is w ← 1 − 0.05 × (−18) = 1 + 0.9 = 1.9. The value 0.1 comes from a sign error, −17 forgets the learning rate, and 19 forgets both the learning rate and the sign."
    },
    {
      "q": "Why can a single neuron not solve XOR?",
      "options": [
        "XOR has too many inputs for one neuron to read",
        "One neuron can only output 0, never 1",
        "Its decision boundary is a straight line, and no line separates the XOR 1s from the 0s",
        "XOR needs a negative bias, which a neuron cannot have"
      ],
      "answer": 2,
      "why": "A neuron fires when w₁x₁ + w₂x₂ + b > 0, which splits the plane with a straight line. The XOR 1s sit at opposite corners, so any line that catches one 1 also catches a 0 or misses the other 1. XOR has just two inputs, and neurons can output 1 and have negative biases."
    },
    {
      "q": "A network has two layers with weight matrices W₁ and W₂ and no activation function between them. What does it compute?",
      "options": [
        "Still just one linear map, (W₂W₁)x, so depth adds nothing",
        "A curved boundary, because it has two layers",
        "Something more powerful than any single layer",
        "Nothing: the output is always 0"
      ],
      "answer": 0,
      "why": "W₂(W₁x) equals (W₂W₁)x, a single matrix, so stacked linear layers collapse into one. A non-linear activation between layers is what lets depth bend the boundary."
    },
    {
      "q": "A layer takes 3 inputs and has 5 neurons. What is the shape of its weight matrix W (rows × columns)?",
      "options": [
        "3 × 5",
        "3 × 3",
        "5 × 5",
        "5 × 3"
      ],
      "answer": 3,
      "why": "W has one row per neuron and one column per input, so it is out × in = 5 × 3. That is 15 weights; with 5 biases the layer has 20 parameters. The shape 3 × 5 would not multiply the 3 × 1 input."
    },
    {
      "q": "A model reaches 99% accuracy on its training set but only 71% on the test set. What is the most likely diagnosis?",
      "options": [
        "Underfitting, because the test score is low",
        "Overfitting: it has memorised the training data and generalizes poorly",
        "A good fit, because the training score is high",
        "The test set is too easy"
      ],
      "answer": 1,
      "why": "A large gap between a very high training score and a lower test score is the sign of overfitting. Underfitting would show low scores on both sets, and a good fit would show similar, high scores."
    },
    {
      "q": "During training the loss on the training data bounces up and down and then grows larger and larger. What is the most likely cause?",
      "options": [
        "The learning rate is too small",
        "There are too few hidden units",
        "The learning rate is too large, so each update overshoots the valley",
        "The model is underfitting"
      ],
      "answer": 2,
      "why": "A step that is too big jumps past the minimum to the other side, and a very large one lands higher up each time, so the loss oscillates or diverges. A learning rate that is too small gives a loss that falls very slowly instead."
    },
    {
      "q": "A neuron’s bias is increased while its weights and inputs stay the same. What happens?",
      "options": [
        "It becomes easier for the neuron to fire, because z is larger for every input",
        "It becomes harder for the neuron to fire",
        "Its output no longer depends on the inputs",
        "Its weights are multiplied by the bias"
      ],
      "answer": 0,
      "why": "The bias is added to the weighted sum, so a larger bias raises z for every input and shifts the threshold lower. The inputs still matter, because the weighted sum is still there. Bias is added, not multiplied."
    }
  ]
}
```

## How Week 2 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: Be the model (labelled table, own rule, parameters, random search) | 12 | Features, label and split named correctly; own rule written and improved with a stated reason; accuracy and MSE computed by loops; random search reported on train and test with the line plotted; specific note on good and biased labels | Steps run and numbers are right; the notes are general | Table, rule or accuracy missing; code not running |
| Part 2: Loss (MSE, cross-entropy, landscape, outlier) | 12 | `mse` and `bce` match the worked numbers and PyTorch; the −ln p curve explained for p → 0; parabola with the minimum marked; outlier effect on MSE and absolute error measured and explained | Functions work; the explanations are brief | Loss functions wrong or missing; no comparison |
| Part 3: One neuron (activations, dials, AND/OR, XOR, training) | 14 | Neuron reproduces 0.562; all four activations plotted; bias and weight effects described; AND/OR truth tables pass; best accuracy near 0.75 on XOR with a clear reason; trained neuron reported | Calculations done; the XOR reason or the dials description is vague | Neuron or truth tables missing; XOR claim wrong |
| Part 4: Layers and stacking (matrix layer, hand-built XOR network, collapse, parameters, design) | 16 | Loop and matrix agree; XOR network and XNOR change correct with the hidden-space idea explained; collapse test passes with a clear reason; parameter counts match PyTorch (17; 101,770; 669,706); digit-reader design counted and justified | Steps run; the explanation of the hidden layer or the design is general | Matrix layer or XOR network missing; counts wrong |
| Part 5: Learning (gradient descent, learning rate, autograd check, NumPy network) | 16 | Worked step gives 1.7 and 20 steps plotted; four learning-rate behaviours shown with the 0.25 limit explained; hand gradients match autograd; the NumPy network reaches about 0.93 with the four-epoch pictures described and a table of hidden units and learning rates | Steps run; the stability limit or the pictures are explained loosely | Training loop missing or loss not decreasing |
| Part 6: PyTorch and breaking it (training loop, capacity, overfitting, remedies, splits) | 15 | Loop annotated line by line; capacity table and smallest H found; overfitting curves with the best epoch found; three remedies compared on test accuracy; 60/20/20 split with a leakage check | Everything runs; reasons for the choices are general | Overfitting not shown; test set used to choose settings |
| Part 7: Design, stress test and reflect (spec, synthetic classifier, label noise, summary) | 15 | Complete spec with at least two features and a concrete risk; classifier trained with train/validation/test accuracies; label-noise drop measured and explained; summary complete; four specific reflection answers that cite results from the lab | Everything present; the risk or the answers are general | Spec or reflection missing; no stress test |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave (your instructor will use them to open Week 3):
1. Name the parts of a supervised learning problem (features, label, model, loss) using the canteen or spam example, and say what the loss is used for.
2. A neuron has inputs (2, 1), weights (0.5, −1) and bias 0.25. Compute z, and say what sigmoid does to it. Then say why a single neuron could not learn XOR.
3. A model scores 98% on its training data and 71% on its test data. What has probably happened, and name one thing you could do about it?

**Homework / self-study**
- Finish the lab, including the written answers and the Part 4 and Part 7 designs.
- Read the Module 1 material for neural networks and write a half-page summary of how a neuron, a layer and training fit together.
- Preview Week 3: look at the sigmoid, tanh and ReLU curves again and think about what happens to a gradient when a curve is almost flat.
