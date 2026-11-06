> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 4 — Convolutional Neural Networks

**Purpose:** See why a network that looks at pictures should ==slide one small shared filter across the whole image== instead of connecting every pixel to every neuron, and learn to compute that operation, and its gradient, by hand.

**Term:** Prelim (Weeks 1–5) · **Month:** 1 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a filter slide over a picture",
  "caption": "A small filter slides over an image and writes one number at each position. The grid in the first scene is real arithmetic. The parameter counts and the shifted pattern in the next two scenes are exact. The bars and the loss curve in the last two scenes are made up to show what training looks like.",
  "scenes": [
    { "name": "Sliding filter", "kind": "conv", "outputLabel": "Edge map", "prompt": "A 3×3 vertical-edge filter moves over a tiny image, one position at a time", "grid": [ [0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9] ] },
    { "name": "Why share", "kind": "text", "outputLabel": "Weights needed to look at one 28×28 picture", "prompt": "A 28×28 picture has 784 numbers. How many parameters to look at it?", "output": "A fully connected layer with 100 units: 784 × 100 + 100 = 78,500. A 3×3 convolution with 8 filters: 3 × 3 × 1 × 8 + 8 = 80." },
    { "name": "Shifted pattern", "kind": "text", "outputLabel": "The same filter, the pattern moved one step", "prompt": "Row A: 0 0 9 9 0 0 0 0. Row B is the same bright bar one pixel to the right. Filter: −1, +1 (right neighbour minus the pixel).", "output": "Row A gives 0 9 0 −9 0 0 0. Row B gives 0 0 9 0 −9 0 0. Same answer, shifted by one: a shared filter finds the bar wherever it sits." },
    { "name": "After training", "kind": "bars", "outputLabel": "Probability for each class (illustrative)", "prompt": "A photo shown to a trained CNN", "bars": [ { "label": "cat", "value": 0.81 }, { "label": "dog", "value": 0.13 }, { "label": "fox", "value": 0.06 } ] },
    { "name": "Training", "kind": "loss", "outputLabel": "Loss while training (illustrative)", "prompt": "Repeat: forward pass, backward pass through every convolution, update the kernels" }
  ]
}
```

## By the end, you can…

1. ==Explain why fully connected layers fit images poorly==, with numbers: too many parameters, no use of the layout, no tolerance to shifts.
2. ==Explain local connectivity and weight sharing== and why together they let a CNN find the same pattern anywhere in a picture.
3. ==Compute a convolution by hand==, including bias, ReLU, stride and padding.
4. ==Compute an output size and a parameter count== with the formulas ⌊(n + 2p − k) / s⌋ + 1 and k·k·C_in·C_out + C_out.
5. ==Compute dL/dK and dL/dx of a convolution by hand==, and say why the kernel gradient is a sum over positions.
6. ==Trace shapes through a small CNN== (convolution, ReLU, pooling, flatten, dense) and name what pooling and stacking add.

Assessed in the Week 4 lab and quiz. Read each section, then play with the figure under it: the figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. Notes such as “Lab Part 4” tell you which part of the lab lets you try the idea for real. This week has two jobs. First, we explain why a network for pictures needs a different kind of layer. Second, we open that layer up and compute it, forwards and backwards, with real numbers. You already know the neuron, the layer, the loss, backpropagation and the chain rule from Weeks 2 and 3, so we will refer back to them instead of repeating them. Every number in the text was computed, not guessed.

## Images are grids of numbers {rail=Images}

{lede} Before a network can “see”, a picture has to become numbers. A [[convolutional neural network (CNN)|A neural network built from convolution layers, which slide small filters with shared weights over an image, usually together with pooling layers and a few dense layers at the end.]] is a network designed for that kind of data: ==it reuses one small filter at every position==, so it needs far fewer parameters and finds a pattern wherever it appears. To see why we need it, we first look at what goes wrong when we use the layers you already know.

### From pixels to numbers

A photo on your phone is a grid of tiny squares. Each square is a [[pixel|One small square of an image. In a greyscale picture it holds one number for its brightness (for example 0 for black and 255 or 1.0 for white); in a colour picture it holds three numbers, for red, green and blue.]]. Think of a school ID photo printed on graph paper, with one number written in every box: dark boxes get small numbers and bright boxes get large ones. That is exactly what the computer stores. Here is a 6 × 6 greyscale picture of a handwritten “7”, with brightness from 0 (black) to 1 (white).

```viz
{
  "type": "heatmap",
  "title": "A tiny picture is just a table of numbers",
  "rows": ["row 1", "row 2", "row 3", "row 4", "row 5", "row 6"],
  "cols": ["col 1", "col 2", "col 3", "col 4", "col 5", "col 6"],
  "corner": "row ↓ col →",
  "showValues": true,
  "W": [
    [0, 0, 0, 0, 0, 0],
    [0.9, 0.9, 0.9, 0.9, 0.9, 0],
    [0, 0, 0, 0, 0.8, 0],
    [0, 0, 0, 0.8, 0, 0],
    [0, 0, 0.8, 0, 0, 0],
    [0, 0.8, 0, 0, 0, 0]
  ],
  "note": "A hand-made 6 × 6 picture for teaching. Real digit pictures such as 28 × 28 work the same way, only with more boxes."
}
```

Notice the bright strokes: a bar across the top and a slanted line going down to the left. Your eye sees a “7” at once, but the computer only has the 36 numbers. A 28 × 28 greyscale picture has 28 × 28 = 784 numbers.

### Colour and the shape of a picture tensor

A colour picture keeps three tables of the same size, one for red, one for green and one for blue. Each table is called a [[channel|One layer of numbers of the same height and width as the picture. A colour image has 3 channels (red, green, blue); the output of a convolution layer with 16 filters has 16 channels.]]. Stack the tables and you get a [[tensor|A grid of numbers with any number of dimensions: a vector has one, a matrix has two, a stack of matrices has three or more.]] with three dimensions. PyTorch writes its shape as (C, H, W): channels, height, width. A 224 × 224 colour photo is (3, 224, 224), which holds 3 × 224 × 224 = 150,528 numbers. A greyscale digit is (1, 28, 28). In training we feed a [[batch|A group of examples processed together in one step. The batch size N is how many pictures are in the group.]] of N pictures at once, so the tensor gets a fourth dimension in front: (N, C, H, W). A batch of 64 colour photos is (64, 3, 224, 224). You will print these shapes in Lab Part 1, and remember them, because most PyTorch errors this week are shape errors.

| Picture | Shape (C, H, W) | Numbers in one picture | In a batch of 64 (N, C, H, W) |
|---|---|---|---|
| Greyscale digit | (1, 28, 28) | 784 | (64, 1, 28, 28) |
| Small colour thumbnail | (3, 32, 32) | 3,072 | (64, 3, 32, 32) |
| Colour photo | (3, 224, 224) | 150,528 | (64, 3, 224, 224) |

### Why fully connected layers fit images poorly

In Week 2 you built the fully connected (dense) layer: every unit is connected to every input. To use it on a picture we unroll the picture into one long list of numbers. There are three problems, and each one is a reason to invent something new.

#### Reason 1: too many parameters

A dense layer with n_in inputs and n_out units has one weight for every pair plus one bias per unit. Select each symbol below and follow the digit example.

```viz
{
  "type": "equation",
  "title": "Parameters of a dense layer: n_in × n_out + n_out",
  "intro": "Select each symbol, then step through the numbers for a 28 × 28 picture feeding 100 units.",
  "formula": [
    { "t": "P", "id": "P" },
    " = ",
    { "t": "n_in", "id": "nin" },
    " × ",
    { "t": "n_out", "id": "nout" },
    " + ",
    { "t": "n_out", "id": "nout" }
  ],
  "parts": [
    { "id": "P", "name": "P (parameters)", "meaning": "The number of adjustable numbers in the layer: all its weights plus all its biases. Training has to find every one of them." },
    { "id": "nin", "name": "n_in (inputs)", "meaning": "How many numbers enter the layer. For a picture unrolled into a list it is height × width × channels." },
    { "id": "nout", "name": "n_out (units)", "meaning": "How many units the layer has. Each unit has n_in weights, plus one bias (the second n_out in the formula)." }
  ],
  "steps": [
    { "title": "Count the inputs", "expr": "n_in = 28 × 28 × 1 = 784", "note": "Unrolling the picture gives a list of 784 numbers." },
    { "title": "Weights", "expr": "784 × 100 = 78,400", "note": "Every one of the 100 units has its own weight for every pixel." },
    { "title": "Add the biases", "expr": "78,400 + 100 = 78,500", "note": "One bias per unit." },
    { "title": "A full small network [784, 100, 10]", "expr": "78,500 + (100 × 10 + 10) = 78,500 + 1,010 = 79,510", "note": "The output layer adds 1,010 more. Almost all the cost is the first layer." },
    { "title": "A real photo", "expr": "n_in = 224 × 224 × 3 = 150,528;  1,000 units: 150,528 × 1,000 + 1,000 = 150,529,000", "note": "One layer with about 150 million parameters, about 600 MB at 4 bytes each." }
  ],
  "note": "The arithmetic is exact. A 600 MB layer is only the first one."
}
```

Notice how fast it grows. Try it yourself with the calculator, then with the network figure. Slide the image side up and watch the dense count explode while the convolution count does not move.

```viz
{
  "type": "calc",
  "title": "Parameters: fully connected or convolution?",
  "kind": "fc_vs_conv",
  "channels": 1,
  "filters": 8,
  "kernel": 3
}
```

At 28 × 28 with 100 units, the dense layer has 78,500 parameters and eight 3 × 3 filters have 3 × 3 × 1 × 8 + 8 = 80, about 980 times fewer. Set the side to 224 and the units to 1,000 and you reach the 150 million of the example above; the convolution stays at 80. ==The parameter count of a convolution does not depend on the size of the picture.== Next, the small digit network as a table.

```viz
{
  "type": "mlp",
  "title": "A dense network for 28 × 28 digits",
  "intro": "Read the table for [784, 100, 10]: 79,510 parameters. Then drag the hidden width and see how the first layer dominates.",
  "layers": [784, 100, 10],
  "note": "Counting parameters only; it does not say how accurate the network is."
}
```

==So many parameters need a lot of data to pin down, take memory and overfit easily.== That is reason 1.

#### Reason 2: the dense layer does not use the layout

Imagine you cut a photo of a barangay map into 784 little squares, drop them into a bag, and then hand them to a friend in the same scrambled order every time. She can still learn “this order means a school, that order means a church”, because each position in the list has its own weight and the scramble is the same for every picture. The same is true of a dense network: ==if you shuffle the pixels with one fixed shuffle for every picture, a dense network can learn the shuffled data about as well as the original==. It does not know which pixels are neighbours; it treats the list as 784 unrelated numbers. A person cannot do that, because the layout carries the meaning: a line is bright pixels next to each other. In Lab Part 1 you shuffle the pixels and measure it. ==A good layer for pictures should use the fact that nearby pixels belong together.==

#### Reason 3: the dense layer is not robust to shifts

Here is the third problem. Suppose a cat’s ear is at pixel 100 in one photo and 5 pixels to the right in the next. For a dense layer these are two different inputs: pixel 100 has its own weights, and pixel 105 has other weights. The network has to learn “ear” separately for every position, and a position it has never seen the ear in is a gap in its knowledge. Compare the two designs in the figure below.

```viz
{
  "type": "flow",
  "title": "The same bright bar, one pixel to the right",
  "modes": [
    {
      "name": "Dense layer",
      "tagline": "One weight per pixel position. Moving the pattern changes which weights are used.",
      "inputLabel": "Picture",
      "modelLabel": "Dense unit (weights tied to positions)",
      "outputLabel": "What it does",
      "examples": [
        { "label": "Bar at pixels 3–4 (counting from 0)", "input": "0 0 0 9 9 0 0 0", "output": "Uses the weights of pixels 3 and 4. If these were trained to say “bar”, the unit responds." },
        { "label": "Bar one step right", "input": "0 0 0 0 9 9 0 0", "output": "Now pixels 4 and 5 are bright. Pixel 5’s weight never saw a bar, so the response can be very different." }
      ],
      "note": "Typical behaviour, not a measured claim for one network. You measure it in Lab Part 1."
    },
    {
      "name": "Convolution",
      "tagline": "The same few weights slide over every position. Moving the pattern moves the answer.",
      "inputLabel": "Picture",
      "modelLabel": "Shared filter [−1, +1]",
      "outputLabel": "What it does",
      "examples": [
        { "label": "Bar at pixels 3–4 (counting from 0)", "input": "0 0 0 9 9 0 0 0", "output": "Output 0 0 9 0 −9 0 0: a positive peak at the left edge, a negative peak at the right edge." },
        { "label": "Bar one step right", "input": "0 0 0 0 9 9 0 0", "output": "Output 0 0 0 9 0 −9 0: exactly the same two peaks, shifted by one position." }
      ],
      "note": "Exact arithmetic for a 1-D filter [−1, +1]: each output is the right neighbour minus the pixel."
    }
  ]
}
```

Notice that the convolution output is the same answer in a new place. ==A pattern learned once is recognised everywhere==, and that is why photographs from a phone, where the subject is never in the same spot twice, are handled so much better by a CNN. Now check that you can tell the two layer types apart.

```viz
{
  "type": "sorter",
  "title": "Dense layer, convolution layer, or both?",
  "intro": "Sort each property into the layer type it describes. Some properties belong to both.",
  "bins": [ { "id": "dense", "label": "Dense layer" }, { "id": "conv", "label": "Convolution layer" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "Uses a small neighbourhood of pixels as its input", "answer": "conv", "why": "Local connectivity: a convolution unit looks at one small patch, for example 3 × 3, not at the whole picture." },
    { "name": "Learns separate weights for every pixel position", "answer": "dense", "why": "A dense unit has one weight per input, so every position gets its own weight." },
    { "name": "Its parameter count grows with the picture size", "answer": "dense", "why": "n_in is height × width × channels, so a bigger picture means more weights. A convolution’s kernel stays the same size." },
    { "name": "Reuses the same weights at every position", "answer": "conv", "why": "Weight sharing: one kernel slides over the whole picture." },
    { "name": "Computes a weighted sum plus a bias, followed by an activation", "answer": "both", "why": "A convolution unit is still a neuron: a weighted sum of its inputs, a bias, then ReLU or another gate." },
    { "name": "Gives about the same accuracy if all pixels are shuffled with one fixed shuffle", "answer": "dense", "why": "The dense layer never knew which pixels were neighbours. A convolution depends on neighbours, so shuffling breaks it." },
    { "name": "Is trained with backpropagation and the chain rule", "answer": "both", "why": "Both are differentiable operations made of multiplications and additions, so Week 3’s rules apply." },
    { "name": "Finds a learned pattern even if the object moves a few pixels", "answer": "conv", "why": "Because the same kernel is applied at every position, the response moves with the pattern." }
  ]
}
```

Notice that the “both” cards remind you that a convolution layer is not magic: it is the neuron you already know, with two restrictions (look locally, share weights) that suit pictures.

### A short history

The idea has a long history. The timeline lists only milestones you can look up easily; click each one.

```viz
{
  "type": "timeline",
  "title": "From the visual cortex to deep CNNs",
  "events": [
    { "year": "1959", "title": "Cells that respond to edges", "text": "Hubel and Wiesel record from single cells in a cat’s visual cortex and find that many respond to an edge of a particular orientation in one small region. This inspired the idea of small, local detectors." },
    { "year": "1998", "title": "LeNet", "text": "LeCun and colleagues train LeNet, a small CNN that reads handwritten digits. It already has the pattern you will study: convolution, pooling, then dense layers." },
    { "year": "2012", "title": "AlexNet", "text": "Krizhevsky, Sutskever and Hinton train AlexNet, a much larger CNN, on graphics processors and win the ImageNet image-recognition contest by a wide margin. The field turns to deep learning." },
    { "year": "2015", "title": "ResNet", "text": "He and colleagues add the residual connections you met in Week 3, and CNNs with over 100 layers become trainable." }
  ]
}
```

Notice that the CNN idea (1998) is much older than its breakthrough (2012). What changed was data, fast hardware and the fixes for deep training from Week 3.

## The convolution operation {rail=Convolution}

{lede} A [[convolution|The operation of sliding a small grid of weights (the kernel) over an image: at each position, multiply the kernel with the patch under it, add up the products, and write one output number.]] is only multiplication and addition, repeated. ==Place the kernel on a patch of the picture, multiply matching numbers, add them up, write the result, then slide one step.== Two ideas make it right for pictures, and both have an everyday picture.

### Two ideas: look locally, share the weights

The first idea is [[local connectivity|Each unit looks only at a small patch of the input, such as 3 × 3 pixels, instead of at every pixel.]]. Imagine reading a page through a small magnifying glass. You see only a few letters at a time, but a few letters are enough to recognise a stroke or a corner. A unit in a convolution layer has a magnifying glass of this kind: it is connected only to the pixels under the glass, so it needs only a few weights (9 for a 3 × 3 patch of a greyscale picture) instead of 784.

The second idea is [[weight sharing|Using exactly the same weights at every position of the image, so a pattern learned in one place is detected everywhere.]]. Imagine a barangay inspector walking every street with the same checklist: “Is there a pothole? Is the drain clear?” She does not write a new checklist for each street. A [[kernel|A small grid of weights (for example 3 × 3) that slides over the input and acts as a pattern detector. Also called a filter.]] is that checklist, and the walk is the sliding. Or think of a rubber stamp: one stamp, pressed on every part of the page. ==The same few weights are used at every position==, so the layer needs only those few numbers however large the picture is, and a pattern is found wherever it appears.

### Step by step: one window at a time

Take a small 4 × 4 picture x and a 2 × 2 kernel K. We use small whole numbers so you can follow every product.

| x | | | |
|---|---|---|---|
| 1 | 2 | 0 | 1 |
| 0 | 1 | 3 | 1 |
| 2 | 1 | 0 | 0 |
| 1 | 0 | 1 | 2 |

K has rows (1, 0) and (0, −1): it adds the top-left pixel of the window and subtracts the bottom-right one. At the first position the window is the top-left 2 × 2 patch, rows (1, 2) and (0, 1). Multiply matching cells and add: 1 × 1 + 2 × 0 + 0 × 0 + 1 × (−1) = 0. Slide one step right: the window is rows (2, 0) and (1, 3), and 2 × 1 + 0 × 0 + 1 × 0 + 3 × (−1) = −1. Keep going over every position and you fill a 3 × 3 output called y. Step through it with the figure and check the first two numbers.

```viz
{
  "type": "conv",
  "title": "Convolution by hand: a 4 × 4 picture and a 2 × 2 kernel",
  "mode": "conv",
  "stride": 1,
  "padding": 0,
  "input": [ [1,2,0,1],[0,1,3,1],[2,1,0,0],[1,0,1,2] ],
  "kernel": [ [1,0],[0,-1] ]
}
```

The output is y with rows (0, −1, −1), (−1, 1, 3) and (2, 0, −2). Notice two things. The output is smaller than the input (3 × 3 from 4 × 4), because the kernel needs room to fit; section 3 explains how to count this exactly. And every output number reuses the same four kernel weights. You will compute these values with loops and with PyTorch in Lab Part 2, and we use this same x and K again in section 5.

### The formula, with a worked example

Now the formula, written so each symbol has a meaning. The output at row i and column j is the sum, over the kernel’s rows a and columns c, of the picture value under the kernel multiplied by the kernel weight, plus a bias.

```viz
{
  "type": "equation",
  "title": "The convolution: y[i, j] = Σₐ Σ_c x[i + a, j + c] · K[a, c] + b",
  "intro": "Select each symbol, then step through the worked example with the 4 × 4 picture and the 2 × 2 kernel above.",
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
    { "id": "y", "name": "y[i, j] (output)", "meaning": "The number written at row i, column j of the output (the feature map). One number per kernel position." },
    { "id": "sum", "name": "Σₐ Σ_c (sum over the window)", "meaning": "Add up the products for every cell of the kernel: a runs over the kernel’s rows and c over its columns. A 2 × 2 kernel has 4 terms, a 3 × 3 kernel has 9." },
    { "id": "x", "name": "x[i + a, j + c] (picture value under the kernel)", "meaning": "The pixel at row i + a, column j + c. When the kernel’s top-left corner is at (i, j), cell (a, c) of the kernel sits on this pixel." },
    { "id": "K", "name": "K[a, c] (kernel weight)", "meaning": "The weight in row a, column c of the kernel. The same weights are used at every (i, j): this is weight sharing." },
    { "id": "b", "name": "b (bias)", "meaning": "One number added to every output of this kernel, as in Week 2. It shifts how easily the detector fires." }
  ],
  "steps": [
    { "title": "Window at (0, 0)", "expr": "y[0,0] = 1×1 + 2×0 + 0×0 + 1×(−1) + 0 = 0", "note": "The top-left 2 × 2 patch is rows (1, 2) and (0, 1). Bias taken as 0 for now." },
    { "title": "Window at (0, 1)", "expr": "y[0,1] = 2×1 + 0×0 + 1×0 + 3×(−1) = −1", "note": "Slide one step right: the patch is rows (2, 0) and (1, 3)." },
    { "title": "Window at (1, 2)", "expr": "y[1,2] = 3×1 + 1×0 + 0×0 + 0×(−1) = 3", "note": "The largest output in this example: a bright 3 sits under the +1 weight." },
    { "title": "Window at (2, 2)", "expr": "y[2,2] = 0×1 + 0×0 + 1×0 + 2×(−1) = −2", "note": "The last window is the bottom-right patch, rows (0, 0) and (1, 2)." },
    { "title": "Add a bias", "expr": "with b = 1:  y[1,2] = 3 + 1 = 4;  y[2,2] = −2 + 1 = −1", "note": "The bias is added to every output of this kernel." }
  ],
  "note": "The numbers are authored for teaching; every product is exact."
}
```

Notice that nothing here is new: ==each output is a weighted sum of a few inputs plus a bias==, the neuron from Week 2, applied at every position with the same weights.

### Bias and ReLU

After the weighted sum and the bias, a convolution layer applies an activation, usually ReLU from Week 3. Take the output above with a bias of 1: it becomes rows (1, 0, 0), (0, 2, 4) and (3, 1, −1). ==ReLU replaces every negative number by zero==, so the layer’s result is rows (1, 0, 0), (0, 2, 4) and (3, 1, 0). The negative responses are thrown away and the positive ones, where the pattern was found, are kept. In a trained network the bias is learned too, and it decides how strong the evidence must be before a detector “fires”.

### Filters are detectors

What does a kernel detect? A kernel is a pattern, and the output is large where the picture under the kernel looks like the pattern. Try this with a 10 × 10 picture of the letter “L”. A famous hand-made kernel is the Sobel kernel, with rows (−1, 0, 1), (−2, 0, 2), (−1, 0, 1): it subtracts the left column of the patch from the right column, so it fires where the picture gets brighter going to the right.

```viz
{
  "type": "filters",
  "title": "Filters as detectors: the letter L",
  "intro": "Click “vertical edges” and look at the feature map: green is positive, blue is negative. Then click “horizontal edges”, “blur” and “sharpen”. Finally click cells in the “your own” kernel to cycle their values and design a detector.",
  "image": [
    "0000000000",
    "0990000000",
    "0990000000",
    "0990000000",
    "0990000000",
    "0990000000",
    "0990000000",
    "0999999900",
    "0999999900",
    "0000000000"
  ],
  "presets": [
    { "name": "vertical edges", "kernel": [ [-1,0,1],[-2,0,2],[-1,0,1] ], "note": "Positive where the picture gets brighter to the right, negative where it gets darker. The stem of the L is flagged on both of its sides, the left edge in green and the right edge in blue." },
    { "name": "horizontal edges", "kernel": [ [-1,-2,-1],[0,0,0],[1,2,1] ], "note": "Positive where the picture gets brighter going down. The top of the stem and the top of the foot are green, the bottom of the foot is blue, and the middle of the stem is zero." },
    { "name": "blur", "kernel": [ [1,1,1],[1,1,1],[1,1,1] ], "note": "Adds up the 9 pixels under the window, so edges spread out and the picture gets softer. This version is the sum, not the average; dividing by 9 gives the average." },
    { "name": "sharpen", "kernel": [ [0,-1,0],[-1,5,-1],[0,-1,0] ], "note": "Boosts the centre pixel and subtracts its four neighbours, so edges stand out more." }
  ],
  "relu": false,
  "padding": 0,
  "note": "The kernels are hand-made for teaching. With these settings the strongest vertical-edge response on the L is 36 (and −36), which is 9 × 4 from a perfect edge."
}
```

What to notice: with the vertical-edge kernel, the long vertical stem lights up while the foot lights up only at its ends. With the horizontal kernel it is the other way round. The long middle of the stem is zero for the horizontal kernel because every window there looks the same above and below. This is the whole idea of a filter: ==each kernel answers one question, “is my pattern here?”, at every position==. The grid of answers is called a [[feature map|The grid of numbers a kernel produces as it slides over the input. It shows where in the picture that kernel’s pattern was found.]]. Switch the ReLU toggle on and see that the blue (negative) responses disappear and only the “found it” responses remain.

### Many filters, many channels

One kernel finds one pattern. A convolution layer therefore uses many kernels, say 16, each producing its own feature map. Stacked together, the 16 maps form an output with 16 channels, which is the input of the next layer. Try a second picture, a house, and notice how different kernels pick out the roof, the walls and the door.

```viz
{
  "type": "filters",
  "title": "Filters as detectors: a house",
  "intro": "Start with “outline” and then try “diagonal”. Turn ReLU on and off. Then build your own kernel that fires on the roof but not on the walls.",
  "image": [
    "0000990000",
    "0009999000",
    "0099999900",
    "0999999990",
    "0999999990",
    "0999999990",
    "0999009990",
    "0999009990",
    "0999009990",
    "0000000000"
  ],
  "presets": [
    { "name": "outline", "kernel": [ [-1,-1,-1],[-1,8,-1],[-1,-1,-1] ], "note": "Compares the centre with its eight neighbours. It is positive (up to 45) on bright pixels along the outside border of the shape and strongly negative (−45) on the dark door pixels next to the bright wall." },
    { "name": "diagonal", "kernel": [ [0,1,2],[-1,0,1],[-2,-1,0] ], "note": "Responds to edges that run from top-left to bottom-right. The right-hand slope of the roof is strongly negative (−36), while the left slope, which runs the other way, gives a much weaker response." },
    { "name": "vertical edges", "kernel": [ [-1,0,1],[-2,0,2],[-1,0,1] ], "note": "Flags the outer walls (green on the left wall, blue on the right wall) and the two sides of the door (blue on its left side, green on its right side)." },
    { "name": "horizontal edges", "kernel": [ [-1,-2,-1],[0,0,0],[1,2,1] ], "note": "Flags the roof (green, dark above and bright below), the top of the door (blue) and the base of the house (blue, where bright wall meets dark ground)." }
  ],
  "relu": true,
  "padding": 0,
  "note": "Hand-made kernels on a hand-made picture. In a trained CNN the kernels are found by training, not written by people."
}
```

Notice that a handful of small kernels already pick out the important parts of the shape. A trained CNN does not wait for people to design them. ==The kernel values are learned==: they start random, and the backpropagation you met in Week 3 nudges them, step after step, towards values that lower the loss. Section 5 shows exactly how the gradient for a kernel is computed.

### Colour pictures: kernels with depth

A colour picture has 3 channels, so a kernel for it has depth 3 as well: a 3 × 3 kernel for an RGB picture is really a 3 × 3 × 3 block of 27 weights. At each position it multiplies all 27 numbers under it, adds them (plus one bias) and writes **one** number. If the layer has 16 such kernels, it writes 16 numbers per position, so the output has 16 channels. That gives the parameter formula, which you should be able to apply to any layer.

```viz
{
  "type": "equation",
  "title": "Parameters of a convolution layer: k · k · C_in · C_out + C_out",
  "intro": "Select each symbol, then step through three layers, including a colour one.",
  "formula": [
    { "t": "P", "id": "P" },
    " = ",
    { "t": "k · k", "id": "kk" },
    " · ",
    { "t": "C_in", "id": "cin" },
    " · ",
    { "t": "C_out", "id": "cout" },
    " + ",
    { "t": "C_out", "id": "cout" }
  ],
  "parts": [
    { "id": "P", "name": "P (parameters)", "meaning": "All the learnable numbers in the layer: kernel weights plus biases." },
    { "id": "kk", "name": "k · k (kernel size)", "meaning": "The height times the width of one kernel, for example 3 × 3 = 9 or 5 × 5 = 25." },
    { "id": "cin", "name": "C_in (input channels)", "meaning": "How many channels the layer receives: 1 for greyscale, 3 for RGB, or the number of filters of the previous layer. Each kernel has this much depth." },
    { "id": "cout", "name": "C_out (filters / output channels)", "meaning": "How many kernels the layer has. Each makes one feature map, so this is also the number of output channels. Each kernel has one bias, which is the last term." }
  ],
  "steps": [
    { "title": "A colour picture, 16 filters of 3 × 3", "expr": "3×3×3×16 + 16 = 432 + 16 = 448", "note": "Each kernel has 27 weights, 16 kernels, and 16 biases." },
    { "title": "A deeper layer: 16 channels in, 32 filters, 3 × 3", "expr": "3×3×16×32 + 32 = 4,608 + 32 = 4,640", "note": "Each kernel now has depth 16, so 144 weights." },
    { "title": "The first layer of LeNet: greyscale, 6 filters, 5 × 5", "expr": "5×5×1×6 + 6 = 150 + 6 = 156", "note": "You meet this network in section 4." },
    { "title": "Does the image size appear?", "expr": "No: the same 156 whatever the picture size", "note": "The formula has no height or width of the picture. This is weight sharing in one line." }
  ],
  "note": "Exact arithmetic. The 4,640 and 156 are layers you will meet again in the lab and in section 4."
}
```

Notice that the picture size is missing from the formula. Compare 448 with the 15 million a dense layer would need for the same 224 × 224 colour picture and only 100 units: 150,528 × 100 + 100 = 15,052,900. In PyTorch the weights of `nn.Conv2d(3, 16, 3)` have the shape (16, 3, 3, 3): 16 kernels, each with depth 3 and size 3 × 3. You print this in Lab Part 2.

### An aside: “convolution” that is not quite convolution

In a maths textbook, convolution first flips the kernel, upside down and left to right, and then slides it. What deep-learning libraries (PyTorch, TensorFlow) compute, and what this course and every figure here use, does not flip the kernel. Strictly this is called [[cross-correlation|The sliding multiply-and-add without flipping the kernel. It is what deep-learning libraries call convolution.]]. ==For a network it makes no difference to what can be learned==, because the kernel is learned anyway: a network that wanted the flipped kernel would simply learn the flipped numbers. Everyone says “convolution”, so we do too, but the formula in this lesson is the no-flip version. You compare both in Lab Part 2.

## Stride, padding and output size {rail=Output size}

{lede} In the 4 × 4 example the output shrank to 3 × 3. How much a picture shrinks, and what you can do about it, depends on two choices: how far the kernel jumps each step, and whether you add a border of zeros first. ==You must be able to predict the output size before you run any code==, because every shape error in PyTorch comes back to this arithmetic.

### Stride and padding

The [[stride|How many pixels the kernel moves between two positions. Stride 1 moves one pixel; stride 2 skips every other position and roughly halves the output size.]] is the size of the kernel’s step. With stride 1 it visits every position; with stride 2 it jumps two pixels at a time, like someone who walks the street by hopping every second paving stone: it is faster and sees less detail. [[Padding|Adding a border of zeros around the picture before the kernel slides, so the kernel can be centred on edge pixels and the output can keep the input’s size.]] is a frame of zeros around the picture, like a mat around a photo. A convolution with no padding is called “valid”, since the kernel always fits inside the picture; a convolution that adds enough padding to keep the size at stride 1 is called “same”. For an odd kernel size k, ==the padding that keeps the size at stride 1 is p = (k − 1) / 2==: 1 for 3 × 3, 2 for 5 × 5, 3 for 7 × 7.

### The output-size formula

Suppose the picture has side n, the kernel has side k, the padding is p on each side, and the stride is s. The padded picture has side n + 2p. The kernel’s left edge can start at positions 0, s, 2s, … as long as it still fits, and the number of positions is the output size. Select each symbol, and check the three worked examples.

```viz
{
  "type": "equation",
  "title": "Output size: ⌊(n + 2p − k) / s⌋ + 1",
  "intro": "Select each symbol, then step through the three examples. The same formula works for the height and for the width.",
  "formula": [
    { "t": "n_out", "id": "nout" },
    " = ⌊(",
    { "t": "n", "id": "n" },
    " + 2",
    { "t": "p", "id": "p" },
    " − ",
    { "t": "k", "id": "k" },
    ") / ",
    { "t": "s", "id": "s" },
    "⌋ + 1"
  ],
  "parts": [
    { "id": "nout", "name": "n_out (output size)", "meaning": "The height (or width) of the output feature map, in positions." },
    { "id": "n", "name": "n (input size)", "meaning": "The height (or width) of the input in pixels, before padding." },
    { "id": "p", "name": "p (padding)", "meaning": "How many rows or columns of zeros are added on each side. 2p is added in total." },
    { "id": "k", "name": "k (kernel size)", "meaning": "The side of the square kernel. The kernel needs k pixels, so n + 2p − k is the space left over for moving." },
    { "id": "s", "name": "s (stride)", "meaning": "The step in pixels. Dividing the spare room by s counts how many full steps fit; the ⌊ ⌋ brackets round down, and the + 1 counts the starting position." }
  ],
  "steps": [
    { "title": "Example 1: no padding, stride 1", "expr": "n = 28, k = 5, p = 0, s = 1:  ⌊(28 + 0 − 5) / 1⌋ + 1 = 23 + 1 = 24", "note": "A 28 × 28 digit becomes 24 × 24. This is the number you predict in the lab’s Imagine first." },
    { "title": "Example 2: same padding", "expr": "n = 28, k = 3, p = 1, s = 1:  ⌊(28 + 2 − 3) / 1⌋ + 1 = 27 + 1 = 28", "note": "Padding 1 with a 3 × 3 kernel keeps the size." },
    { "title": "Example 3: stride 2", "expr": "n = 32, k = 3, p = 0, s = 2:  ⌊(32 − 3) / 2⌋ + 1 = ⌊14.5⌋ + 1 = 14 + 1 = 15", "note": "29 spare pixels, steps of 2: 14 full steps fit and the 15th position is the start. The last pixel column is never reached." },
    { "title": "Why +1?", "expr": "n = 3, k = 3, p = 0, s = 1:  ⌊0 / 1⌋ + 1 = 1", "note": "A kernel the size of the picture fits in exactly one position. Without the + 1 the formula would say zero." }
  ],
  "note": "Exact arithmetic. ⌊x⌋ means x rounded down to a whole number."
}
```

Here is a table of more cases to practise with. Cover the last column and predict it first.

| Input n | Kernel k | Padding p | Stride s | ⌊(n + 2p − k) / s⌋ + 1 | Output |
|---|---|---|---|---|---|
| 28 | 5 | 0 | 1 | ⌊23 / 1⌋ + 1 | 24 |
| 28 | 3 | 1 | 1 | ⌊27 / 1⌋ + 1 | 28 |
| 32 | 3 | 0 | 2 | ⌊29 / 2⌋ + 1 | 15 |
| 28 | 5 | 2 | 2 | ⌊27 / 2⌋ + 1 | 14 |
| 224 | 7 | 3 | 2 | ⌊223 / 2⌋ + 1 | 112 |
| 6 | 3 | 0 | 2 | ⌊3 / 2⌋ + 1 | 2 |

Notice the fourth row: a 5 × 5 kernel with padding 2 would keep the size at stride 1, and stride 2 then halves it to 14. You write this formula as a function in Lab Part 2 and test it against real convolutions.

### Stride 2 in action

Take a 6 × 6 picture with a dark left half and a bright right half, and the vertical-edge kernel with rows (1, 0, −1) three times. At stride 1 the output is 4 × 4 and each row reads (0, −27, −27, 0): the edge is found in the two middle columns. Now use stride 2. By the formula the output is ⌊(6 − 3) / 2⌋ + 1 = 2, so 2 × 2. Run the figure and set the stride to 2.

```viz
{
  "type": "conv",
  "title": "Stride 2: fewer positions, the edge is still found",
  "mode": "conv",
  "stride": 2,
  "padding": 0,
  "input": [ [0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9],[0,0,0,9,9,9] ],
  "kernel": [ [1,0,-1],[1,0,-1],[1,0,-1] ]
}
```

The output is rows (0, −27) and (0, −27). The sign is negative because this picture goes from dark to bright, the opposite of what the kernel looks for, so the response flips. What to notice: the output has a quarter as many numbers as the stride-1 output (4 instead of 16), the work is about four times smaller, and the edge is still found. That is the case for stride: ==it makes a layer cheaper and shrinks the picture, at the price of skipping positions==.

### Padding in action

Now a flat 5 × 5 picture where every pixel is 5, and a 3 × 3 kernel of all ones, which simply adds up the 9 pixels under it. With no padding, every output is 9 × 5 = 45, and the output is 3 × 3. With padding 1 the output is 5 × 5, the same size as the input, but look at its values.

```viz
{
  "type": "conv",
  "title": "Padding 1: the output keeps its size, the border sees zeros",
  "mode": "conv",
  "stride": 1,
  "padding": 1,
  "input": [ [5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5] ],
  "kernel": [ [1,1,1],[1,1,1],[1,1,1] ]
}
```

The middle outputs are 45, the edge outputs are 30, and the four corners are 20. Why? A corner window covers only 4 real pixels (4 × 5 = 20), and the other 5 cells are padded zeros. An edge window covers 6 real pixels (30). ==Padding keeps the size, but border outputs see less of the picture, because zeros are not real data.== That is the price of padding, and the reason to use it only when you want to preserve the size, or when the borders matter.

### Why pad, why stride

Use padding when you want the picture to keep its size layer after layer (so a deep stack does not shrink to nothing: three layers of 5 × 5 without padding take 28 down to 24, then 20, then 16) and when information at the borders matters. Use stride (or pooling, next section) when you want to shrink the picture on purpose, to save computation and to make later layers look at larger parts of the original. ==Parameters do not depend on the picture size, but computation does==: a convolution does one set of multiplications per output position, so a bigger output costs more time even though it needs no extra weights.

### Parameters do not depend on the picture size

Let us use the calculator once more, this time with a colour picture. Set the channels to 3 and the filters to 16, and compare a dense layer with a convolution.

```viz
{
  "type": "calc",
  "title": "Parameters again: a colour picture, 16 filters of 3 × 3",
  "kind": "fc_vs_conv",
  "channels": 3,
  "filters": 16,
  "kernel": 3
}
```

The convolution stays at 3 × 3 × 3 × 16 + 16 = 448 parameters. Slide the image side to 224 and the units to 1,000 and the dense layer reaches 224 × 224 × 3 × 1,000 + 1,000 = 150,529,000. Slide the side down to 32 and the dense layer has 3,073,000, but the convolution is still 448. Notice what this means in practice: a CNN trained on small pictures has exactly the same number of weights as one for large pictures. Only the feature maps, and the compute, get bigger.

## Pooling and stacking {rail=Stacking}

{lede} One convolution layer finds small, simple patterns. To recognise a digit or a face, a network has to combine them, and it also has to shrink the picture along the way so that the final dense layers stay small. ==A CNN stacks blocks of convolution, ReLU and pooling, so that each layer looks at a bigger part of the picture than the one before.== This section explains the missing piece, pooling, and how to follow shapes, parameters and the size of the view through a whole stack.

### Pooling: summarise each neighbourhood

[[Pooling|A layer that shrinks a feature map by replacing each small window with one number: the maximum (max pooling) or the average (average pooling). It has no weights to learn.]] is like asking each group of four houses for one report. If you ask for “the tallest flood mark in this group” you do max pooling; if you ask for “the average flood mark” you do average pooling. In 2 × 2 pooling with stride 2, we split the map into non-overlapping 2 × 2 windows and keep one number from each. Take the 4 × 4 map with rows (1, 3, 2, 0), (4, 2, 1, 5), (0, 1, 3, 2) and (2, 6, 1, 1). The four windows have maxima 4, 5, 6 and 3, so max pooling gives the 2 × 2 map with rows (4, 5) and (6, 3). Their averages are 2.5, 2, 2.25 and 1.75. Click the cells in the figure to see each window, and switch between max and average.

```viz
{
  "type": "pool",
  "title": "Max pooling and average pooling, 2 × 2 with stride 2",
  "intro": "Select an output cell to see which 2 × 2 window it came from and the arithmetic. Switch between the Max and Average tabs and compare the same window.",
  "input": [ [1,3,2,0],[4,2,1,5],[0,1,3,2],[2,6,1,1] ],
  "size": 2,
  "stride": 2,
  "note": "Authored numbers. Pooling has no weights, so there is nothing to train in this layer."
}
```

Notice that the 4 × 4 map has become 2 × 2: a quarter of the numbers remain. The size follows the same idea as the convolution formula, with no padding. Select each symbol and try the examples.

```viz
{
  "type": "equation",
  "title": "Pooled size: ⌊(n − k) / s⌋ + 1",
  "intro": "This is the output-size formula with p = 0. For the usual 2 × 2 pool, k = s = 2, so the size is exactly halved (rounded down).",
  "formula": [
    { "t": "n_out", "id": "nout" },
    " = ⌊(",
    { "t": "n", "id": "n" },
    " − ",
    { "t": "k", "id": "k" },
    ") / ",
    { "t": "s", "id": "s" },
    "⌋ + 1"
  ],
  "parts": [
    { "id": "nout", "name": "n_out (pooled size)", "meaning": "The height or width of the map after pooling." },
    { "id": "n", "name": "n (input size)", "meaning": "The height or width of the feature map entering the pool." },
    { "id": "k", "name": "k (window size)", "meaning": "The side of the pooling window, usually 2." },
    { "id": "s", "name": "s (stride)", "meaning": "How far the window moves. In ordinary pooling it equals k, so windows do not overlap." }
  ],
  "steps": [
    { "title": "The 4 × 4 map above", "expr": "⌊(4 − 2) / 2⌋ + 1 = 1 + 1 = 2", "note": "A 2 × 2 result, as in the figure." },
    { "title": "LeNet, after its first convolution", "expr": "⌊(28 − 2) / 2⌋ + 1 = 13 + 1 = 14", "note": "28 × 28 maps become 14 × 14. The number of channels (6) does not change." },
    { "title": "LeNet, after its second convolution", "expr": "⌊(10 − 2) / 2⌋ + 1 = 4 + 1 = 5", "note": "10 × 10 maps become 5 × 5." },
    { "title": "An odd size", "expr": "⌊(5 − 2) / 2⌋ + 1 = 1 + 1 = 2", "note": "A 5 × 5 map becomes 2 × 2, and the last row and column are ignored. This is why the window and stride in the pool figure must tile the input exactly." }
  ],
  "note": "Exact arithmetic."
}
```

### What pooling gives, and what it throws away

Pooling gives three things: ==smaller maps, a little tolerance to small shifts, and zero parameters==. First, smaller maps: fewer numbers go to the next layer, so the next layer is cheaper. Second, a little tolerance to small shifts: if a bright pixel moves inside its window, for example from the window’s top-right to its top-left, the maximum is still 9, so the output does not change. Third, it costs nothing to store, since it has zero parameters. It also throws something away. It forgets exactly where in the window the maximum was, and with average pooling it blurs weak and strong responses together. And the shift tolerance is partial: if the bright pixel moves across the border between two windows, the pooled output does change. ==Pooling trades exact position for a smaller and steadier summary.== In Lab Part 3 you shift an input by one pixel and count how many pooled values change compared with the feature map.

### Receptive field: how much of the picture a unit sees

A unit in the first layer sees only a 3 × 3 patch. A unit in the second layer sees a patch of first-layer outputs, and each of those saw 3 × 3 pixels, so it sees more of the picture. The [[receptive field|The region of the original input that can affect one unit. In the first layer it is the kernel size; in deeper layers it grows with every layer.]] is the size of the input region behind one unit. It grows when you stack layers, and it grows faster after a stride or a pool. We track two numbers through the stack: the receptive field rf, and the “jump”, the distance in input pixels between two neighbouring units.

```viz
{
  "type": "equation",
  "title": "Receptive field: rf = rf_prev + (k − 1) · jump_prev, then jump = jump_prev · s",
  "intro": "Start with rf = 1 and jump = 1 at the input. Apply the two lines for each layer in turn. Select the symbols and follow the 3-layer example.",
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
    { "t": "jump_prev", "id": "jp" },
    " · ",
    { "t": "s", "id": "s" }
  ],
  "parts": [
    { "id": "rf", "name": "rf (receptive field)", "meaning": "How many input pixels wide the region behind one unit of this layer is." },
    { "id": "rfp", "name": "rf_prev", "meaning": "The receptive field of the layer before. At the input, a pixel sees only itself, so rf = 1." },
    { "id": "k", "name": "k (kernel or window size)", "meaning": "The size of this layer’s kernel (a pool counts too, with its window size). The (k − 1) extra pixels are added to the view." },
    { "id": "jp", "name": "jump_prev", "meaning": "The distance, in input pixels, between neighbouring units of the previous layer. It is 1 at the input and doubles after each stride-2 layer." },
    { "id": "jump", "name": "jump (this layer)", "meaning": "The new distance between neighbouring units: the previous jump times this layer’s stride s." },
    { "id": "s", "name": "s (stride)", "meaning": "This layer’s stride. A 2 × 2 pool has s = 2, so the jump doubles." }
  ],
  "steps": [
    { "title": "Input", "expr": "rf = 1, jump = 1", "note": "One pixel sees itself." },
    { "title": "Layer 1: 3 × 3 convolution, stride 1", "expr": "rf = 1 + (3 − 1)·1 = 3;  jump = 1·1 = 1", "note": "Each unit sees a 3 × 3 patch." },
    { "title": "Layer 2: 2 × 2 max pool, stride 2", "expr": "rf = 3 + (2 − 1)·1 = 4;  jump = 1·2 = 2", "note": "The pool sees a 2 × 2 block of layer-1 units, which cover 4 pixels. Neighbouring units are now 2 pixels apart." },
    { "title": "Layer 3: 3 × 3 convolution, stride 1", "expr": "rf = 4 + (3 − 1)·2 = 8;  jump = 2·1 = 2", "note": "The 3 × 3 patch is of units that are 2 pixels apart, so the view jumps from 4 to 8." },
    { "title": "Two 3 × 3 layers, no pool", "expr": "rf = 1 + 2 = 3 after the first; rf = 3 + 2·1 = 5 after the second", "note": "Two stacked 3 × 3 layers see 5 × 5, like one 5 × 5 layer, but use 2 × 9 = 18 weights per channel pair instead of 25." }
  ],
  "note": "Exact arithmetic. The values 3, 4 and 8 are the ones the cnnstack figure reports for the same three layers."
}
```

Notice the jump in step 4. ==Pooling is a very cheap way to widen the view==: after one pool, every later 3 × 3 kernel covers twice as many pixels. Now watch the whole stack.

### A whole network: LeNet-style

Here is the small network from the history box (LeNet, 1998), in the modern form we use all week: a 28 × 28 greyscale digit goes through a 5 × 5 convolution with 6 filters and padding 2, a 2 × 2 pool, a 5 × 5 convolution with 16 filters, another pool, then it is [[flattened|Unrolled from a stack of feature maps into one long list of numbers, so that a dense layer can read it. A 5 × 5 × 16 stack becomes a list of 400 numbers.]] and passes through dense layers of 120, 84 and 10 units. The table gives each layer’s output shape, parameters and receptive field.

```viz
{
  "type": "cnnstack",
  "title": "A LeNet-style network, layer by layer",
  "intro": "Read the shape, parameters and receptive field of each layer. Then change a kernel size, a padding or a number of filters and watch the shapes after it.",
  "input": [28, 28, 1],
  "layers": [
    { "type": "conv", "k": 5, "filters": 6, "pad": 2 },
    { "type": "pool", "k": 2 },
    { "type": "conv", "k": 5, "filters": 16 },
    { "type": "pool", "k": 2 },
    { "type": "flatten" },
    { "type": "fc", "units": 120 },
    { "type": "fc", "units": 84 },
    { "type": "fc", "units": 10 }
  ],
  "note": "Total 61,706 parameters. The receptive field of the last pool is 16: each unit there sees a 16 × 16 patch of the 28 × 28 digit.",
  "checks": [
    { "layer": 0, "shape": [28, 28, 6], "params": 156, "rf": 5 },
    { "layer": 1, "shape": [14, 14, 6], "params": 0, "rf": 6 },
    { "layer": 2, "shape": [10, 10, 16], "params": 2416, "rf": 14 },
    { "layer": 3, "shape": [5, 5, 16], "params": 0, "rf": 16 },
    { "layer": 5, "params": 48120 },
    { "layer": 6, "params": 10164 },
    { "layer": 7, "params": 850 },
    { "total": 61706 }
  ]
}
```

Try these experiments, and predict each result with the formulas before you look. First, set the padding of the first convolution from 2 to 0. The shapes become 24 × 24 × 6, 12 × 12 × 6, 8 × 8 × 16 and 4 × 4 × 16, the flatten gives 256, the first dense layer needs 256 × 120 + 120 = 30,840 parameters and the total falls to 44,426. Second, set the second convolution’s kernel to 15. The 14 × 14 map is smaller than the kernel, so the formula gives ⌊(14 − 15) / 1⌋ + 1 = 0 positions: the layer cannot work, which is what “too small” means. Third, set the first kernel to 3 with padding 1. The receptive fields of the four layers become 3, 4, 12 and 14 instead of 5, 6, 14 and 16: smaller kernels see less, which is why deep stacks need more layers.

Now the surprising number. The two convolutions have only 156 + 2,416 = 2,572 parameters, about 4% of 61,706, while the first dense layer after the flatten has 48,120, about 78%. ==The convolutions are cheap in parameters but busy in arithmetic==: the first one does 28 × 28 × 5 × 5 × 6 = 117,600 multiply-adds and the second 10 × 10 × 5 × 5 × 6 × 16 = 240,000. A dense layer producing the same 28 × 28 × 6 numbers as the first convolution would need 784 × 4,704 = 3,687,936 weights; the convolution uses 150. Compare with the dense network of section 1, which had 79,510 parameters: the LeNet-style network has fewer, and uses the layout.

### Follow one picture through the pipeline

```viz
{
  "type": "pipeline",
  "title": "One digit through the LeNet-style network",
  "intro": "Press Next to follow a 28 × 28 digit. Watch height and width shrink while the channels grow, then everything is flattened.",
  "nodes": [
    { "id": "in", "label": "Input", "sub": "28×28×1" },
    { "id": "c1", "label": "Conv 5×5, 6 + ReLU", "sub": "28×28×6" },
    { "id": "p1", "label": "Max pool 2×2", "sub": "14×14×6" },
    { "id": "c2", "label": "Conv 5×5, 16 + ReLU", "sub": "10×10×16" },
    { "id": "p2", "label": "Max pool 2×2", "sub": "5×5×16" },
    { "id": "fl", "label": "Flatten", "sub": "400" },
    { "id": "fc", "label": "Dense 120, 84", "sub": "84" },
    { "id": "out", "label": "Dense 10 + softmax", "sub": "10 probabilities" }
  ],
  "steps": [
    { "title": "Input", "body": "A greyscale digit: one channel, 28 × 28 = 784 numbers.", "active": ["in"], "example": "28 × 28 × 1" },
    { "title": "Convolution 1 and ReLU", "body": "Six 5 × 5 kernels with padding 2 keep the size at 28 × 28 and make 6 channels. Parameters: 5×5×1×6 + 6 = 156.", "active": ["in", "c1"], "example": "28 × 28 × 6, 156 parameters" },
    { "title": "Pooling 1", "body": "A 2 × 2 max pool with stride 2 halves height and width: 28 becomes 14. The 6 channels stay. No parameters.", "active": ["c1", "p1"], "example": "14 × 14 × 6" },
    { "title": "Convolution 2 and ReLU", "body": "Sixteen 5 × 5 kernels, each with depth 6, no padding: ⌊(14 − 5) / 1⌋ + 1 = 10. Parameters: 5×5×6×16 + 16 = 2,416.", "active": ["p1", "c2"], "example": "10 × 10 × 16, 2,416 parameters" },
    { "title": "Pooling 2", "body": "10 becomes 5. Each unit now has a receptive field of 16 × 16 pixels, more than half of the digit’s width.", "active": ["c2", "p2"], "example": "5 × 5 × 16" },
    { "title": "Flatten", "body": "Unroll 5 × 5 × 16 into one list: 5 × 5 × 16 = 400 numbers. From here on the layers are dense, and the layout is no longer used.", "active": ["p2", "fl"], "example": "400 numbers" },
    { "title": "Dense layers", "body": "400 to 120 (48,120 parameters), then 120 to 84 (10,164), each followed by ReLU.", "active": ["fl", "fc"], "example": "84 numbers" },
    { "title": "Output and softmax", "body": "84 to 10 class scores (850 parameters), turned into 10 probabilities that sum to 1 by the softmax from Week 2. Total parameters: 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706.", "active": ["fc", "out"], "example": "10 probabilities, 61,706 parameters in all" }
  ]
}
```

Notice the pattern: the picture gets smaller and the channels get more numerous, until the network has a short list of high-level numbers to classify. The sorter below checks that you can tell the three layer types apart by their numbers.

```viz
{
  "type": "sorter",
  "title": "Convolution, pooling or dense?",
  "intro": "Sort each description into the layer type it belongs to.",
  "bins": [ { "id": "conv", "label": "Convolution layer" }, { "id": "pool", "label": "Pooling layer" }, { "id": "dense", "label": "Dense layer" } ],
  "cards": [
    { "name": "Has 2,416 parameters in the LeNet-style network (5×5×6×16 + 16)", "answer": "conv", "why": "A convolution’s count is k·k·C_in·C_out + C_out." },
    { "name": "Has no parameters at all", "answer": "pool", "why": "Pooling only takes a maximum or an average; there is nothing to train." },
    { "name": "Has 48,120 parameters in the LeNet-style network (400×120 + 120)", "answer": "dense", "why": "A dense layer has n_in × n_out weights plus n_out biases, and the flattened input is long." },
    { "name": "Keeps the largest number in each 2 × 2 window", "answer": "pool", "why": "That is the definition of max pooling." },
    { "name": "Every unit depends on every number of the flattened vector", "answer": "dense", "why": "Fully connected means every output is connected to every input, which is also why the layout is lost." },
    { "name": "Reuses each of its few weights at every position, so it does many multiplications", "answer": "conv", "why": "Weight sharing means each kernel weight is used at every output position: 117,600 multiply-adds for 150 weights in the first LeNet layer." },
    { "name": "Shrinks each feature map but keeps the number of channels the same", "answer": "pool", "why": "A pool works on each channel separately, so 6 channels stay 6." },
    { "name": "Needs a flatten step in front of it", "answer": "dense", "why": "A dense layer expects a list of numbers, so the stack of feature maps is unrolled first." }
  ]
}
```

### Hierarchy, equivariance and invariance

Why stack at all? Because the layers seem to build a [[hierarchy of features|A layered description in which each level is built from the level below: edges make textures and parts, parts make objects.]], as the Week 3 history of deep learning suggested. In trained image CNNs the first layers often respond to edges and spots of colour, the middle layers often to textures and parts such as an eye or a wheel, and the last ones to whole objects. ==Often, not always==: what each layer learns is found by training, not designed, and in a small network such as LeNet there may be only a hint of it. In Lab Part 6 you plot what the first layer of your own CNN learned.

Two words describe how a CNN reacts to shifts. A convolution is [[equivariant to shifts|Shift the input and the output shifts by the same amount. The pattern is found in the new place, with the same value.]]: shift the picture, and the feature map shifts with it. The 1-D example from section 1 showed it: the output peaks moved from positions 2 and 4 to 3 and 5 when the bar moved one step. Pooling and the dense layers after it then make the final answer more [[invariant to small shifts|The output stays (almost) unchanged when the input is shifted by a small amount.]]: the class probabilities should stay the same when the digit moves a little. Equivariance in the early layers, plus a bit of invariance from pooling, is the combination you want for “what is in this picture?”. It is not perfect, which is why we also train with shifted examples (section 6).

## Backpropagation through a convolution {rail=Backward}

{lede} In Week 3 you learned that backpropagation is the chain rule applied from the loss backwards: at every operation, ==the gradient going back equals the local derivative times the gradient arriving from the next layer==. A convolution is nothing but many multiplications and additions, so the same rule applies. In this section we work out two gradients that a convolution layer has to produce, and compute both by hand.

### What a convolution layer must compute

Suppose the layer computed y = x ∗ K (we write ∗ for the sliding multiply-and-add). The layers after it have computed how the loss changes with each output: that gradient arrives as a grid G, the same size as y, with G[i, j] = ∂L/∂y[i, j]. (This is the “upstream gradient”, the δ of Week 3, without the activation slope for now.) The layer must produce two things:

1. **∂L/∂K**, how the loss changes with each kernel weight. This is what the optimizer needs to update the kernel.
2. **∂L/∂x**, how the loss changes with each input value. This is passed to the layer before, which treats it as its own G.

Think of the barangay inspector again. If a street reports a mistake (G), her single checklist item was used on every street, so ==the blame for that checklist item is the sum of the blame from every street where it was used==. That is why the kernel’s gradient is a sum over positions. And a single pixel was seen only by the few windows that covered it, so its gradient adds up only those windows’ blame.

### The kernel gradient

Positions are counted from 0, as in Python, so x[0,0] is the top-left pixel. Each weight K[a, c] was multiplied by x[i + a, j + c] to make y[i, j], at every position (i, j). By the chain rule, a change in K[a, c] changes y[i, j] by x[i + a, j + c], and changes the loss by G[i, j] times that. Add it up over all positions.

```viz
{
  "type": "equation",
  "title": "The kernel gradient: ∂L/∂K[a, c] = Σ_{i,j} G[i, j] · x[i + a, j + c]",
  "intro": "Select each symbol, then step through the numbers for the 4 × 4 picture, the 2 × 2 kernel and the upstream gradient G below.",
  "formula": [
    { "t": "∂L/∂K[a, c]", "id": "dK" },
    " = ",
    { "t": "Σ_{i,j}", "id": "sum" },
    " ",
    { "t": "G[i, j]", "id": "G" },
    " · ",
    { "t": "x[i + a, j + c]", "id": "x" }
  ],
  "parts": [
    { "id": "dK", "name": "∂L/∂K[a, c] (kernel gradient)", "meaning": "How much the loss changes per unit change in the weight K[a, c]. The result has the same shape as the kernel, one number per weight." },
    { "id": "sum", "name": "Σ_{i,j} (sum over every output position)", "meaning": "Add the terms for every position (i, j) at which the kernel was placed. A sum is needed because the same weight K[a, c] was used at every position (weight sharing)." },
    { "id": "G", "name": "G[i, j] (upstream gradient)", "meaning": "∂L/∂y[i, j], the gradient that arrives from the layers after this one. Same size as the output y." },
    { "id": "x", "name": "x[i + a, j + c] (the input value that K[a, c] touched)", "meaning": "The pixel that was multiplied by K[a, c] to make y[i, j]. It is the local derivative ∂y[i, j]/∂K[a, c]." }
  ],
  "steps": [
    { "title": "Set up", "expr": "x = rows (1,2,0,1), (0,1,3,1), (2,1,0,0), (1,0,1,2);  K = rows (1,0), (0,−1);  G = rows (1,0,−1), (2,1,0), (0,1,1)", "note": "y = x ∗ K has 3 × 3 outputs, so G is 3 × 3. G is authored for teaching: it could be the gradient from a loss further on." },
    { "title": "dL/dK[0,0]: the 3 × 3 block of x starting at (0,0)", "expr": "1×1 + 0×2 + (−1)×0 + 2×0 + 1×1 + 0×3 + 0×2 + 1×1 + 1×0 = 3", "note": "Each G is multiplied by the pixel that K[0,0] touched at that position, then all nine are added." },
    { "title": "dL/dK[0,1]: the block starting at (0,1)", "expr": "1×2 + 0×0 + (−1)×1 + 2×1 + 1×3 + 0×1 + 0×1 + 1×0 + 1×0 = 6", "note": "Same recipe, with every x shifted one column to the right." },
    { "title": "dL/dK[1,0] and dL/dK[1,1]", "expr": "dL/dK[1,0] = 0 − 0 − 3 + 4 + 1 + 0 + 0 + 0 + 1 = 3;  dL/dK[1,1] = 1 + 0 − 1 + 2 + 0 + 0 + 0 + 1 + 2 = 5", "note": "So dL/dK = rows (3, 6) and (3, 5). All four were checked by nudging each weight by ±0.00001." },
    { "title": "Why a sum", "expr": "K[0,0] was used 9 times (once per output), so its gradient has 9 terms", "note": "A weight used once would have one term; a weight shared over a whole picture collects the blame of the whole picture." },
    { "title": "One update (learning rate 0.1)", "expr": "K ← K − 0.1 × dL/dK = rows (0.7, −0.6), (−0.3, −1.5)", "note": "Gradient descent from Week 2, applied to the kernel. Repeated, this is how the kernel is learned." }
  ],
  "note": "G is made up; the arithmetic is exact and matches finite differences."
}
```

Notice that the kernel gradient is itself a sliding multiply-and-add: it slides G, as if it were a kernel, over x. That is why the backward pass costs the same order of work as the forward pass.

### The input gradient

Now the other direction. A pixel x[p, q] affects only the outputs whose window covered it. For each such output (i, j), the pixel was multiplied by the kernel cell K[p − i, q − j], so its gradient adds G[i, j] times that weight.

```viz
{
  "type": "equation",
  "title": "The input gradient: ∂L/∂x[p, q] = Σ_{i,j} G[i, j] · K[p − i, q − j]",
  "intro": "Select each symbol, then follow the examples. K[a, c] counts as zero when a or c is outside the kernel, so only the outputs that really used the pixel contribute.",
  "formula": [
    { "t": "∂L/∂x[p, q]", "id": "dx" },
    " = ",
    { "t": "Σ_{i,j}", "id": "sum" },
    " ",
    { "t": "G[i, j]", "id": "G" },
    " · ",
    { "t": "K[p − i, q − j]", "id": "K" }
  ],
  "parts": [
    { "id": "dx", "name": "∂L/∂x[p, q] (input gradient)", "meaning": "How much the loss changes if the input value at row p, column q changes. This is what is sent back to the previous layer." },
    { "id": "sum", "name": "Σ_{i,j} (sum over the outputs that used this pixel)", "meaning": "Only windows that covered the pixel contribute. A corner pixel is covered once; a pixel in the middle of a 2 × 2 sliding window is covered four times." },
    { "id": "G", "name": "G[i, j] (upstream gradient)", "meaning": "The gradient at output position (i, j), arriving from the next layer." },
    { "id": "K", "name": "K[p − i, q − j] (the weight that linked them)", "meaning": "At output (i, j) the pixel (p, q) sat under kernel cell (p − i, q − j). That weight is the local derivative ∂y[i, j]/∂x[p, q]." }
  ],
  "steps": [
    { "title": "A corner pixel, x[0,0]", "expr": "only output (0,0) used it, under K[0,0]:  1×1 = 1", "note": "dL/dx[0,0] = 1." },
    { "title": "A middle pixel, x[1,1]", "expr": "G[0,0]·K[1,1] + G[0,1]·K[1,0] + G[1,0]·K[0,1] + G[1,1]·K[0,0] = 1×(−1) + 0×0 + 2×0 + 1×1 = 0", "note": "Four windows covered it. The positive and the negative contributions cancel, so dL/dx[1,1] = 0." },
    { "title": "x[2,1]", "expr": "G[1,0]·K[1,1] + G[1,1]·K[1,0] + G[2,0]·K[0,1] + G[2,1]·K[0,0] = 2×(−1) + 1×0 + 0×0 + 1×1 = −1", "note": "dL/dx[2,1] = −1." },
    { "title": "A pixel on the right edge, x[1,3]", "expr": "G[0,2]·K[1,1] + G[1,2]·K[0,1] = (−1)×(−1) + 0×0 = 1", "note": "Only two windows reached the last column." },
    { "title": "The other corner, x[3,3]", "expr": "G[2,2]·K[1,1] = 1×(−1) = −1", "note": "The whole gradient is rows (1, 0, −1, 0), (2, 0, 0, 1), (0, −1, 0, 0), (0, 0, −1, −1)." }
  ],
  "note": "G is made up; the arithmetic is exact and matches finite differences."
}
```

Notice the difference between the two: ==for the kernel, the sum runs over all positions because the weight is shared; for the input, it runs over only the windows that covered the pixel==. Now try both on the figure. It shows the forward pass, then the gradients.

```viz
{
  "type": "convgrad",
  "title": "Backward pass of a convolution: a 4 × 4 picture and a 2 × 2 kernel",
  "intro": "Select an input cell, for example the one in the second row and second column, and read the four G × K terms that make its gradient. Then select a kernel cell and read the nine G × x terms. Compare with the worked examples above.",
  "input": [ [1,2,0,1],[0,1,3,1],[2,1,0,0],[1,0,1,2] ],
  "kernel": [ [1,0],[0,-1] ],
  "upstream": [ [1,0,-1],[2,1,0],[0,1,1] ],
  "stride": 1,
  "padding": 0,
  "note": "The upstream gradient G is authored for teaching. The values shown were also verified by finite differences.",
  "checks": [
    { "kind": "y", "i": 0, "j": 0, "value": 0 },
    { "kind": "y", "i": 1, "j": 2, "value": 3 },
    { "kind": "dx", "i": 1, "j": 1, "value": 0 },
    { "kind": "dx", "i": 2, "j": 1, "value": -1 },
    { "kind": "dx", "i": 0, "j": 0, "value": 1 },
    { "kind": "dK", "i": 0, "j": 1, "value": 6 },
    { "kind": "dK", "i": 1, "j": 1, "value": 5 }
  ]
}
```

What to notice: each cell of dL/dK is a list of nine products, and each cell of dL/dx is a list of one to four. The cells on the picture’s edge have fewer terms. You compute the same lists with loops in Lab Part 4 and check them with finite differences and with PyTorch.

### With stride and padding

The rules do not change when the kernel steps by 2 or when the picture is padded; only the bookkeeping does. Padded cells are zeros, so a padded position contributes nothing to dL/dK, and any gradient that lands on a padded cell is simply dropped (the padding is not a real input). Here is a 5 × 5 picture, a 3 × 3 kernel, padding 1 and stride 2, so the output is ⌊(5 + 2 − 3) / 2⌋ + 1 = 3, and G is 3 × 3.

```viz
{
  "type": "convgrad",
  "title": "Backward pass with padding 1 and stride 2",
  "intro": "Select the kernel cell in the centre: its nine terms are G × x, and none of the pixels are padding. Then select the kernel cell in the top-left corner and count how many of its nine positions fall on padding. Finally select the input cell in the centre of the picture.",
  "input": [ [1,0,2,1,0],[0,3,1,0,2],[2,1,0,1,1],[1,0,2,3,0],[0,2,1,0,1] ],
  "kernel": [ [1,2,0],[0,-1,1],[2,1,-1] ],
  "upstream": [ [1,-1,0],[2,1,-2],[0,1,1] ],
  "stride": 2,
  "padding": 1,
  "note": "Authored numbers, checked by finite differences. With stride 2 and padding 1 every kernel cell lands on at least some real pixels.",
  "checks": [
    { "kind": "y", "i": 0, "j": 0, "value": -4 },
    { "kind": "y", "i": 1, "j": 2, "value": 9 },
    { "kind": "dx", "i": 1, "j": 1, "value": -2 },
    { "kind": "dx", "i": 2, "j": 2, "value": -1 },
    { "kind": "dx", "i": 1, "j": 0, "value": 5 },
    { "kind": "dK", "i": 0, "j": 0, "value": 6 },
    { "kind": "dK", "i": 1, "j": 1, "value": 3 }
  ]
}
```

Notice that dL/dK[0,0] = 6 has only four terms that land on real pixels: five of its nine positions fell on padded zeros, and the value is G[1,1] × 3 + G[1,2] × 0 + G[2,1] × 0 + G[2,2] × 3 = 3 + 3. And with stride 2, the centre pixel (x[2,2], counting from 0) was covered by only one window, so its gradient is a single product, G[1,1] × K[1,1] = 1 × (−1) = −1. You repeat this with loops in Lab Part 4.

### The input gradient is itself a convolution

Look again at the formula for ∂L/∂x: it sums G[i, j] times a kernel value that depends on p − i and q − j. This is a convolution of G with the kernel, except that now each G value is “spread” over the kernel’s footprint in the input. Written another way: put a border of k − 1 zeros around G, flip the kernel up-down and left-right, and slide it as usual. This operation is called a [[transposed convolution|The operation that spreads each input value over a kernel-sized block of the output and adds the overlaps. For stride 1 it equals the gradient of a convolution with respect to its input.]]. It is the backward pass of the convolution. Use the figure in transpose mode with G as the input and K as the kernel at stride 1.

```viz
{
  "type": "conv",
  "title": "The input gradient as a transposed convolution (stride 1)",
  "mode": "transpose",
  "stride": 1,
  "input": [ [1,0,-1],[2,1,0],[0,1,1] ],
  "kernel": [ [1,0],[0,-1] ]
}
```

The output is a 4 × 4 map with rows (1, 0, −1, 0), (2, 0, 0, 1), (0, −1, 0, 0) and (0, 0, −1, −1): exactly the dL/dx you computed term by term. Check one cell with the flipped-kernel version. The flipped kernel has rows (−1, 0) and (0, 1). The padded G has a 2 × 2 window at the top-left with rows (0, 0) and (0, 1), and 0×(−1) + 0×0 + 0×0 + 1×1 = 1, which is dL/dx[0,0]. At (1,1) the window is rows (1, 0) and (2, 1): 1×(−1) + 0×0 + 2×0 + 1×1 = 0, matching dL/dx[1,1] = 0. In Lab Part 4 you check this against `F.conv_transpose2d`. Transposed convolutions are also used in some networks to enlarge pictures (that is a later topic); here we only need the fact that ==the arithmetic of the input gradient is the same as a transposed convolution==.

### Backward through ReLU

After the convolution comes ReLU. Its local derivative is 1 where the input z was positive and 0 where it was not (Week 3). So the gradient passes unchanged through cells that were “on” and is stopped at cells that were “off”.

```viz
{
  "type": "equation",
  "title": "Backward through ReLU: δ = G ⊙ 1[z > 0]",
  "intro": "Select each symbol, then step through a four-number example.",
  "formula": [
    { "t": "δ", "id": "delta" },
    " = ",
    { "t": "G", "id": "G" },
    " ⊙ ",
    { "t": "1[z > 0]", "id": "gate" }
  ],
  "parts": [
    { "id": "delta", "name": "δ (gradient at the pre-activation z)", "meaning": "The gradient with respect to the number z that went into ReLU. This is what continues backwards to the convolution." },
    { "id": "G", "name": "G (gradient arriving from after the ReLU)", "meaning": "The gradient with respect to the ReLU’s output a = max(0, z)." },
    { "id": "gate", "name": "1[z > 0] (the gate)", "meaning": "1 where z was positive, 0 elsewhere. This is the slope of ReLU, which you plotted in Week 3. (At exactly z = 0 the usual convention is 0.)" }
  ],
  "steps": [
    { "title": "Forward values", "expr": "z = (2, −1, 0.5, −3)  gives  a = (2, 0, 0.5, 0)", "note": "Negative inputs were replaced by 0 on the way forward." },
    { "title": "The gate", "expr": "1[z > 0] = (1, 0, 1, 0)", "note": "It remembers which cells were on." },
    { "title": "Gradient arriving", "expr": "G = (0.4, 0.7, −0.2, 1.0)", "note": "Made-up values from the layer after the ReLU." },
    { "title": "Multiply cell by cell (⊙)", "expr": "δ = (0.4×1, 0.7×0, −0.2×1, 1.0×0) = (0.4, 0, −0.2, 0)", "note": "Cells that were switched off receive no blame, because changing them a little would not have changed the output." }
  ],
  "note": "Authored numbers; the arithmetic is exact."
}
```

Here is the slope in a picture. Select the two curves and find where the gradient is 1 and where it is 0.

```viz
{
  "type": "curve",
  "title": "ReLU and its slope",
  "xRange": [-3, 3],
  "yRange": [-0.5, 3],
  "series": [ { "fn": "relu", "label": "ReLU(z)" }, { "fn": "relu_grad", "label": "slope of ReLU" } ],
  "alt": "ReLU is zero for negative z and equals z for positive z. Its slope is 0 for negative z and 1 for positive z, with a jump at 0.",
  "note": "Drawn from the function definitions."
}
```

Notice that the slope is exactly 1 on the active side. That is the reason ReLU helps against vanishing gradients in Week 3: passing back through an active ReLU does not shrink the gradient.

### Backward through pooling

Max pooling copied one number from each window and ignored the others. So the gradient goes back only to the cell that held the maximum, and every other cell in the window gets 0. Average pooling mixed all four cells equally, so each receives a quarter of the gradient. Use the figure with a new map and check the two backward rules by hand.

```viz
{
  "type": "pool",
  "title": "Pooling backward: where does the gradient go?",
  "intro": "Select each output cell in the Max tab and find the cell that held the maximum. Then switch to Average tab. Suppose the upstream gradient for the four outputs is rows (1, 2) and (3, 4).",
  "input": [ [2,0,1,4],[3,1,0,2],[0,5,2,1],[1,0,3,0] ],
  "size": 2,
  "stride": 2,
  "note": "Authored numbers. Max pooling gives rows (3, 4) and (5, 3); average pooling gives rows (1.5, 1.75) and (1.5, 1.5)."
}
```

With upstream gradient rows (1, 2) and (3, 4), max pooling sends 1 to the cell holding the 3 (row 2, column 1), 2 to the cell holding the 4 (row 1, column 4), 3 to the cell holding the 5 (row 3, column 2) and 4 to the cell holding the 3 in the bottom-right window (row 4, column 3). Every other cell gets 0, so the gradient map has only four non-zero numbers: rows (0, 0, 0, 2), (1, 0, 0, 0), (0, 3, 0, 0) and (0, 0, 4, 0). Average pooling sends one quarter of each gradient to every cell of its window: 0.25 in the top-left window, 0.5 in the top-right, 0.75 in the bottom-left and 1 in the bottom-right. ==Max pooling routes the gradient to the winner; average pooling shares it equally.== (If two cells tie for the maximum, implementations choose one of them; ties are rare for real-valued data.)

### One whole pass: conv, ReLU and pool, forward and back

Put it together on a small example. A 5 × 5 picture goes through a 2 × 2 convolution with kernel rows (1, 0) and (0, −1), ReLU and a 2 × 2 max pool, and the loss is L = 1×P₁₁ − 1×P₁₂ + 2×P₂₁ + 3×P₂₂ on the four pooled values P. Step through the pipeline and keep track of the numbers.

```viz
{
  "type": "pipeline",
  "title": "One forward and one backward pass: conv, ReLU, pool",
  "intro": "Press Next. The first four steps go forward, the last four go backward. Picture x has rows (1,2,0,1,3), (0,1,3,1,0), (2,1,0,0,2), (1,0,1,2,1), (3,1,0,2,0). Kernel K has rows (1,0), (0,−1).",
  "nodes": [
    { "id": "x", "label": "Picture x", "sub": "5×5" },
    { "id": "conv", "label": "Convolution", "sub": "y = x ∗ K" },
    { "id": "relu", "label": "ReLU", "sub": "a = max(0, y)" },
    { "id": "pool", "label": "Max pool 2×2", "sub": "P" },
    { "id": "loss", "label": "Loss L", "sub": "weighted sum of P" }
  ],
  "steps": [
    { "title": "Forward: convolution", "body": "Slide K over x. The output y is 4 × 4 with rows (0, −1, −1, 1), (−1, 1, 3, −1), (2, 0, −2, −1), (0, 0, −1, 2).", "active": ["x", "conv"], "example": "y[1,2] = 3×1 + 1×0 + 0×0 + 0×(−1) = 3" },
    { "title": "Forward: ReLU", "body": "Negative values become 0: rows (0, 0, 0, 1), (0, 1, 3, 0), (2, 0, 0, 0), (0, 0, 0, 2).", "active": ["conv", "relu"], "example": "max(0, −1) = 0" },
    { "title": "Forward: max pool", "body": "Maxima of the four 2 × 2 windows, remembering where they were: P has rows (1, 3) and (2, 2), at positions (1,1), (1,2), (2,0) and (3,3) counting rows and columns from 0.", "active": ["relu", "pool"], "example": "P = rows (1, 3), (2, 2)" },
    { "title": "Forward: the loss", "body": "With the weights 1, −1, 2, 3 on the pooled values, L = 1×1 + (−1)×3 + 2×2 + 3×2 = 8. The weights here play the part of everything after the pool; they are also the upstream gradient dL/dP.", "active": ["pool", "loss"], "example": "L = 1 − 3 + 4 + 6 = 8" },
    { "title": "Backward: through the pool", "body": "dL/dP = rows (1, −1), (2, 3). The gradient goes only to the four winning cells; everything else gets 0: rows (0, 0, 0, 0), (0, 1, −1, 0), (2, 0, 0, 0), (0, 0, 0, 3).", "active": ["loss", "pool"], "example": "−1 goes to the cell that held the 3" },
    { "title": "Backward: through ReLU", "body": "All four winning cells had y greater than 0, so the gate is 1 and the gradient passes unchanged. (A winner cannot be a switched-off cell unless the whole window is zero.)", "active": ["pool", "relu"], "example": "1[y > 0] = 1 at the four winners" },
    { "title": "Backward: kernel gradient", "body": "Apply the kernel-gradient formula with this G over the 4 × 4 output: dL/dK has rows (8, 7) and (9, 0). Check one entry: dL/dK[0,0] = 1×x[1,1] + (−1)×x[1,2] + 2×x[2,0] + 3×x[3,3] = 1 − 3 + 4 + 6 = 8.", "active": ["relu", "conv"], "example": "dL/dK = rows (8, 7), (9, 0)" },
    { "title": "Backward: input gradient", "body": "Apply the input-gradient formula: dL/dx has rows (0,0,0,0,0), (0,1,−1,0,0), (2,0,−1,1,0), (0,−2,0,3,0), (0,0,0,0,−3). It is passed to whatever came before the picture (or discarded, if the picture is the raw data).", "active": ["conv", "x"], "example": "dL/dx[4,4] = 3×(−1) = −3 (counting from 0)" }
  ]
}
```

All the numbers in this pipeline were checked by nudging every weight by ±0.00001, as in a gradient check. Notice how the three rules fit together: the pool picks four cells, the ReLU lets them through, and the convolution adds up their blame onto four weights.

### Which direction? A quick sort

```viz
{
  "type": "sorter",
  "title": "Forward or backward?",
  "intro": "Sort each operation into the pass in which it happens.",
  "bins": [ { "id": "forward", "label": "Forward pass" }, { "id": "backward", "label": "Backward pass" } ],
  "cards": [
    { "name": "Multiply the kernel with each patch and add (y = x ∗ K + b)", "answer": "forward", "why": "This is the convolution itself, which makes the feature map." },
    { "name": "Sum G × x over every position to get dL/dK", "answer": "backward", "why": "This is the kernel-gradient formula; it needs the stored input x." },
    { "name": "Replace negative numbers by zero", "answer": "forward", "why": "ReLU as a function: max(0, z)." },
    { "name": "Let the gradient pass only where z was positive", "answer": "backward", "why": "ReLU’s local derivative gates the gradient on the way back." },
    { "name": "Keep the maximum of each 2 × 2 window", "answer": "forward", "why": "Max pooling picks one number per window." },
    { "name": "Send the gradient only to the position that held the maximum", "answer": "backward", "why": "Only the winner affected the output, so only the winner gets blame." },
    { "name": "Share the gradient equally among the four cells of a window", "answer": "backward", "why": "Average pooling mixed the four cells equally, so each gets one quarter." },
    { "name": "Slide the kernel over the picture with stride s", "answer": "forward", "why": "The sliding happens in the forward pass; the backward pass reuses the same positions." }
  ]
}
```

Notice that the backward pass needs the numbers the forward pass stored (x for dL/dK, the positions of the maxima, the gate of the ReLU), which is the memory cost you met in Week 3.

### What PyTorch does, and how to check it

When you call `loss.backward()` on a model with `nn.Conv2d`, PyTorch’s autograd applies exactly these rules (with fast code) and puts the result in `K.grad` and `x.grad`. You can reproduce the first example in a few lines:

```python
import torch
import torch.nn.functional as F

x = torch.tensor([ [1., 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2] ]).reshape(1, 1, 4, 4)
K = torch.tensor([ [1., 0], [0, -1] ]).reshape(1, 1, 2, 2)
G = torch.tensor([ [1., 0, -1], [2, 1, 0], [0, 1, 1] ]).reshape(1, 1, 3, 3)
x.requires_grad_(True)
K.requires_grad_(True)

y = F.conv2d(x, K)          # forward: shape (1, 1, 3, 3)
loss = (G * y).sum()        # then dL/dy is exactly G
loss.backward()
print(K.grad.reshape(2, 2)) # dL/dK = rows (3, 6), (3, 5)
print(x.grad.reshape(4, 4)) # dL/dx = the 4 x 4 gradient above
```

The loss (G * y).sum() is a trick: because it is a weighted sum of the outputs, its gradient with respect to y is exactly G. The habit from Week 3 still applies: ==when you write a backward pass yourself, check it with finite differences==. Nudge one number up and down by a tiny ε and measure the slope.

```viz
{
  "type": "equation",
  "title": "The gradient check: (L(w + ε) − L(w − ε)) / 2ε",
  "intro": "Select each symbol. Then check dL/dK[0,0] of our example by nudging one kernel weight.",
  "formula": [
    { "t": "∂L/∂w", "id": "g" },
    " ≈ (",
    { "t": "L(w + ε)", "id": "up" },
    " − ",
    { "t": "L(w − ε)", "id": "down" },
    ") / 2",
    { "t": "ε", "id": "eps" }
  ],
  "parts": [
    { "id": "g", "name": "∂L/∂w (numerical gradient)", "meaning": "An estimate of the slope of the loss with respect to one weight, found by measuring instead of by calculus." },
    { "id": "up", "name": "L(w + ε)", "meaning": "The loss after raising just this one number by ε and running the forward pass again." },
    { "id": "down", "name": "L(w − ε)", "meaning": "The loss after lowering the same number by ε." },
    { "id": "eps", "name": "ε (a tiny step)", "meaning": "A small number such as 0.00001. Smaller steps are more accurate until rounding errors take over." }
  ],
  "steps": [
    { "title": "The loss at the start", "expr": "L = Σ G × y = 1×0 + 0×(−1) + (−1)×(−1) + 2×(−1) + 1×1 + 0×3 + 0×2 + 1×0 + 1×(−2) = −2", "note": "Using the y and G of the first example." },
    { "title": "Nudge K[0,0] up and down by 0.001", "expr": "L(K[0,0] + 0.001) = −1.997;  L(K[0,0] − 0.001) = −2.003", "note": "Raising K[0,0] by 0.001 increases y wherever x was bright and changes L by 0.001 × (sum of G × x) = 0.003." },
    { "title": "The slope", "expr": "(−1.997 − (−2.003)) / (2 × 0.001) = 0.006 / 0.002 = 3", "note": "This equals dL/dK[0,0] = 3 from the formula. In the lab you check all 4 kernel and all 16 input numbers." }
  ],
  "note": "Exact here because L is linear in K. For a real network the agreement is to many decimals, not exactly."
}
```

Notice that the check is far too slow to train with: it needs two forward passes for every single number. It is the standard way to catch a mistake in a hand-written backward pass, and you will use it in Lab Part 4.

## Training a CNN {rail=Training}

{lede} Training a CNN uses exactly the loop you already know from Weeks 2 and 3: forward pass, loss, backward pass, update. ==Only the layers change==, and PyTorch supplies the convolution and pooling layers together with their backward rules from section 5. The skill to practise here is reading a model definition and knowing, line by line, what shape and how many parameters each line produces.

### The LeNet-style model in PyTorch

Here is the network from section 4 as a PyTorch model. It is shown, not run; you build and train a similar one in Lab Part 5. Read the comments: the shape is (N, C, H, W) with N the batch size, and the numbers on the right are the parameter counts you verified with the formulas.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 6, 5, padding=2),  # (N, 1, 28, 28) -> (N, 6, 28, 28)    5*5*1*6 + 6    = 156
    nn.ReLU(),
    nn.MaxPool2d(2),                # -> (N, 6, 14, 14)                  no parameters
    nn.Conv2d(6, 16, 5),            # -> (N, 16, 10, 10)                 5*5*6*16 + 16  = 2,416
    nn.ReLU(),
    nn.MaxPool2d(2),                # -> (N, 16, 5, 5)                   no parameters
    nn.Flatten(),                   # -> (N, 400)                        5*5*16 = 400
    nn.Linear(400, 120),            # -> (N, 120)                        400*120 + 120  = 48,120
    nn.ReLU(),
    nn.Linear(120, 84),             # -> (N, 84)                         120*84 + 84    = 10,164
    nn.ReLU(),
    nn.Linear(84, 10),              # -> (N, 10) class scores            84*10 + 10     = 850
)
print(sum(p.numel() for p in model.parameters()))  # 61706
```

Some points in these lines are worth saying out loud. `nn.Conv2d(1, 6, 5, padding=2)` means one input channel, six filters, kernel 5 × 5, so its weight tensor has the shape (6, 1, 5, 5): 150 weights, plus 6 biases. `nn.Flatten()` keeps the batch dimension and unrolls the rest, turning (N, 16, 5, 5) into (N, 400). The last layer has no ReLU, because it produces the raw class scores (the “logits”), which the loss function turns into probabilities. The 61,706 matches the `cnnstack` figure of section 4. The last three layers are a small dense network on 400 inputs, and you can look at it as a table.

```viz
{
  "type": "mlp",
  "title": "The dense head of the LeNet-style network",
  "intro": "Read the table for [400, 120, 84, 10]: 48,120 + 10,164 + 850 = 59,134 parameters, almost all of the 61,706.",
  "layers": [400, 120, 84, 10],
  "note": "Counting parameters only. The convolution layers before the flatten add the remaining 2,572."
}
```

### Training: the loop you know

```python
import torch

loss_fn = nn.CrossEntropyLoss()                      # takes raw scores and integer labels
opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # or plain SGD from Week 2

for epoch in range(10):
    for xb, yb in loader:                            # xb: (64, 1, 28, 28), yb: (64,)
        opt.zero_grad()                              # clear the old gradients
        loss = loss_fn(model(xb), yb)                # forward pass and loss
        loss.backward()                              # backward pass: section 5, for every layer
        opt.step()                                   # update every kernel and weight
```

```viz
{
  "type": "pipeline",
  "title": "One training step of a CNN",
  "intro": "Press Next to follow one batch through a step. It is the same loop as in Week 3, with convolution layers inside.",
  "nodes": [
    { "id": "batch", "label": "Batch", "sub": "(64, 1, 28, 28)" },
    { "id": "fwd", "label": "Forward", "sub": "conv, pool, dense" },
    { "id": "loss", "label": "Loss", "sub": "cross-entropy" },
    { "id": "bwd", "label": "Backward", "sub": "dL/dK, dL/dx per layer" },
    { "id": "upd", "label": "Update", "sub": "K ← K − η·dL/dK" }
  ],
  "steps": [
    { "title": "Take a batch", "body": "Load 64 pictures and their labels. Each picture is scaled to a small range, for example 0 to 1, and the batch has the shape (N, C, H, W).", "active": ["batch"], "example": "xb: (64, 1, 28, 28), yb: (64,)" },
    { "title": "Forward pass", "body": "Every layer transforms its input in turn and stores what the backward pass will need: the pictures, the ReLU gates and the positions of the maxima.", "active": ["batch", "fwd"], "example": "(64, 1, 28, 28) → … → (64, 10)" },
    { "title": "Loss", "body": "Cross-entropy compares the 10 class scores with the true label for each picture and averages over the batch. One number.", "active": ["fwd", "loss"], "example": "loss = 2.31 at the start (about ln 10, a guess among 10 classes)" },
    { "title": "Backward pass", "body": "loss.backward() goes through the layers in reverse: dense layers, then pooling (route to the winners), ReLU (gate), convolution (dL/dK as a sum over positions and dL/dx for the layer before).", "active": ["loss", "bwd"], "example": "K.grad has the same shape as K" },
    { "title": "Update", "body": "The optimizer moves every kernel and weight a little against its gradient. Repeat for the next batch; after many batches the kernels have become useful detectors.", "active": ["bwd", "upd"], "example": "K ← K − 0.001 × K.grad (with Adam, a scaled version of this)" }
  ],
  "note": "The starting loss of about 2.31 is ln 10 = 2.3026, which is what you get from ten equally likely classes; a real run starts near it, but the exact value depends on the initial weights."
}
```

Notice that ==nothing in the loop knows about convolutions==. That is the beauty of autograd: you change the layers, and the same four lines keep working. Two settings matter from Week 3: the learning rate (too large and the loss jumps or goes to NaN, too small and nothing moves) and the batch size (a bigger batch gives a smoother gradient and uses more memory, since activations are stored for every picture). Start from learning rate 0.001 with Adam, or 0.01 to 0.1 with SGD, and a batch size of 32 or 64, and adjust by watching the loss curve.

### Data augmentation

A CNN is not fully shift-proof (pooling helps only a little), and a network can only be good at what it has seen. [[Data augmentation|Making extra training examples by changing the existing ones in ways that do not change the label: shifting, flipping, rotating a little, adding noise. It teaches the network what it should ignore.]] means giving the network each training picture in several slightly changed forms. Shift a digit by two pixels, rotate it by five degrees, add a little noise: it is still the same digit, but a new input. The network learns that these changes do not matter, which is a cheap way of getting more data and a safer way of getting robustness. A caution: ==an augmentation must keep the label true==. Flipping a photo of a cat left to right is fine. Flipping the digit 2 or the letter b makes something that is not the same character, and rotating a 6 by 180 degrees makes a 9. Augment only the training data, never the test data. In Lab Part 5 you train on shifted pictures and measure what it changes.

### The 1 × 1 convolution

A kernel of size 1 × 1 looks at no neighbours at all: at every position it takes the C_in numbers in the channels and mixes them into C_out new numbers, like a tiny dense layer applied at every pixel. It is the cheapest way to change the number of channels. For 16 input channels and 8 output channels it has 1 × 1 × 16 × 8 + 8 = 136 parameters, and for 64 channels reduced to 16 it has 1 × 1 × 64 × 16 + 16 = 1,040. It is a common tool in modern CNNs for squeezing channels before an expensive 3 × 3 convolution.

### The lab’s CNN

In Lab Part 5 you build a small CNN for 16 × 16 pictures of simple shapes and compare it with a dense network. Check the shapes and parameters here first.

```viz
{
  "type": "cnnstack",
  "title": "The lab’s small CNN for 16 × 16 shape pictures",
  "intro": "Read the shapes. Then change the number of filters of the second convolution and watch what happens to the size of the last layer.",
  "input": [16, 16, 1],
  "layers": [
    { "type": "conv", "k": 3, "filters": 8, "pad": 1 },
    { "type": "pool", "k": 2 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "pool", "k": 2 },
    { "type": "flatten" },
    { "type": "fc", "units": 5 }
  ],
  "note": "Total 2,533 parameters: 80 + 1,168 + 1,285. The same task with a dense network [256, 64, 5] would need 16,773.",
  "checks": [
    { "layer": 0, "shape": [16, 16, 8], "params": 80, "rf": 3 },
    { "layer": 2, "shape": [8, 8, 16], "params": 1168, "rf": 8 },
    { "layer": 3, "shape": [4, 4, 16], "params": 0, "rf": 10 },
    { "layer": 5, "params": 1285 },
    { "total": 2533 }
  ]
}
```

This network has 2,533 parameters: 80 + 1,168 + 1,285. A dense network for the same pictures, [256, 64, 5], has 256 × 64 + 64 + 64 × 5 + 5 = 16,773. Change the filters of the second convolution from 16 to 32 and the last layer grows from 4 × 4 × 16 × 5 + 5 = 1,285 to 4 × 4 × 32 × 5 + 5 = 2,565, because the flatten gets longer.

### The same task, two designs

```viz
{
  "type": "flow",
  "title": "Dense network or CNN for the same picture task",
  "modes": [
    {
      "name": "Dense network",
      "tagline": "Unroll the picture, connect every pixel to every unit. Simple, but the layout is not used.",
      "inputLabel": "Task",
      "modelLabel": "[784, 100, 10]",
      "outputLabel": "What typically happens",
      "examples": [
        { "label": "Parameters", "input": "28 × 28 digit", "output": "79,510 parameters; for a 224 × 224 colour photo the first layer alone has 150,528 weights per unit" },
        { "label": "Shifted picture", "input": "The digit moved 3 pixels to the side", "output": "Typically a clear drop in accuracy, because position-specific weights see an unfamiliar input" },
        { "label": "Shuffled pixels", "input": "All pixels shuffled in one fixed way", "output": "Typically about the same accuracy as before" }
      ],
      "note": "Typical behaviour, stated qualitatively. Lab Parts 1 and 5 measure it for the lab’s pictures."
    },
    {
      "name": "CNN",
      "tagline": "Small shared filters, pooling, then a short dense head. The layout is built in.",
      "inputLabel": "Task",
      "modelLabel": "LeNet-style network",
      "outputLabel": "What typically happens",
      "examples": [
        { "label": "Parameters", "input": "28 × 28 digit", "output": "61,706 parameters; convolution weights do not grow with the picture size" },
        { "label": "Shifted picture", "input": "The digit moved 3 pixels to the side", "output": "Typically degrades less, because filters and pooling tolerate small moves; still not perfect" },
        { "label": "Shuffled pixels", "input": "All pixels shuffled in one fixed way", "output": "Typically much worse, because the filters need neighbouring pixels to mean something" }
      ],
      "note": "Typical behaviour, stated qualitatively; real results depend on the data and the training. Lab Part 5 measures it."
    }
  ]
}
```

Notice that the CNN is worse on the shuffled pixels. That is not a flaw: it shows the CNN is using the layout, an assumption that is true for real pictures.

### Famous CNNs

```viz
{
  "type": "timeline",
  "title": "A few landmark CNNs",
  "events": [
    { "year": "1998", "title": "LeNet-5", "text": "LeCun and colleagues read handwritten digits with two convolution and pooling stages followed by dense layers: the pattern you just studied, at about 60 thousand parameters." },
    { "year": "2012", "title": "AlexNet", "text": "A much bigger CNN with five convolution layers and three dense layers, trained on graphics processors. It used ReLU and a trick called dropout, and it won the ImageNet contest by a large margin." },
    { "year": "2014", "title": "VGG", "text": "The VGG networks showed that stacking many small 3 × 3 convolutions, with pooling in between, gives a deep and effective network (versions with 16 and 19 weight layers are well known)." },
    { "year": "2015", "title": "ResNet", "text": "He and colleagues added the residual connection x + F(x) that you met in Week 3. It lets a gradient flow around layers, and CNNs with over 100 layers train well." }
  ]
}
```

Notice that ResNet is the Week 3 fix applied to a CNN: nothing new is needed in the convolution itself. Later in the course you will meet vision transformers, which handle pictures without convolutions; CNNs remain a standard, efficient choice and a good way to understand what “learning features from pixels” means.

### A debugging checklist

When a CNN does not run or does not learn, ==check the shapes first, then the numbers==.

1. **Shape errors.** A message such as “mat1 and mat2 shapes cannot be multiplied (64x400 and 256x120)” means the flatten gave 400 numbers and the next `nn.Linear` expected 256. Recompute the shape layer by layer with the output-size formula (section 3), or push a dummy batch through the model and print each shape.
2. **Forgetting `nn.Flatten()`** before the first `nn.Linear`. The dense layer then receives a four-dimensional tensor.
3. **Wrong channel count.** The `in_channels` of a convolution must equal the channels of its input: 1 for a greyscale picture, 3 for RGB, and the previous layer’s number of filters afterwards. A colour picture sent into a one-channel model gives an error about channels.
4. **Wrong tensor layout or a missing batch dimension.** PyTorch wants (N, C, H, W). A single picture needs `unsqueeze(0)` to become a batch of one, and pictures from some libraries come as (H, W, C) and must be rearranged.
5. **Input scaling.** Pixels from 0 to 255 should be divided by 255 (and often normalised further). Large inputs make early training unstable, as in the exploding-gradient section of Week 3.
6. **Learning does not start.** Look at the loss at step 0 (about ln of the number of classes), then at the gradient norm of each layer, as in Week 3. If the first layer’s gradient is tiny, suspect the activations and the initialization (He for ReLU).

In Lab Part 5 you provoke three of these errors on purpose and write your own five-line version of this checklist.

## Key takeaways

- **Dense layers fit pictures poorly.** A 28 × 28 picture into 100 units already costs 78,500 parameters and a 224 × 224 colour photo into 1,000 units costs 150,529,000; ==a dense layer also ignores the layout and treats a shifted pattern as new input==.
- **Local connectivity and weight sharing fix both problems.** A small kernel looks at a patch and is ==reused at every position, so a pattern is found wherever it appears and the parameter count k·k·C_in·C_out + C_out does not depend on the picture size==.
- **You can compute a convolution and its shape by hand.** Each output is Σ x · K + b, then ReLU, and ==the output size is ⌊(n + 2p − k) / s⌋ + 1==; padding keeps the size, stride and pooling shrink it, and the receptive field grows with rf = rf_prev + (k − 1)·jump_prev.
- **A CNN stacks convolution, ReLU and pooling, then flattens into dense layers.** The LeNet-style network is ==28×28×1 → 28×28×6 → 14×14×6 → 10×10×16 → 5×5×16 → 400 → 120 → 84 → 10 with 61,706 parameters==; pooling gives smaller maps and a little shift tolerance, and throws away exact position.
- **The backward pass is the chain rule again.** ==∂L/∂K[a,c] = Σ G[i,j]·x[i+a, j+c] sums over every position because the kernel is shared, and ∂L/∂x[p,q] = Σ G[i,j]·K[p−i, q−j] sums only over the windows that covered the pixel==; ReLU lets the gradient through where z > 0, and max pooling routes it to the winner.

<!-- LAB-START -->
## Lab: Imagine a CNN, then build one {rail=Laboratory}

This lab is called *Imagine a CNN, then build one*. Week 4 gives the network eyes: you first picture yourself as a tiny inspector who walks over a photograph with a stencil (the idea behind local connectivity and weight sharing), and then ==work with pictures as numbers, compute convolutions and their gradients by hand, design your own filters, build and train a small CNN in PyTorch, and compare it with a dense network on shifted and noisy pictures==. It has seven parts, and every part begins with an **Imagine first** moment: you write a prediction (a number or a sentence) before you run any code and compare it with what really happens. Each part points back to a section of the lesson. All the pictures are drawn by a helper cell (five simple shapes on a small grid), nothing is downloaded, and everything runs in seconds to a couple of minutes on a CPU. Time: about 3 hours. This page contains the setup and the first three parts; the other four parts, the deliverables and the marking scheme follow in the same lab.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cell, run a smoke test | 10 |
| 1 | Pictures are numbers, and why dense layers struggle | 25 |
| 2 | Convolution by hand | 25 |
| 3 | Filters, feature maps and pooling | 25 |
| 4 | The backward pass of a convolution, by hand | 30 |
| 5 | Build and train a CNN | 30 |
| 6 | Look inside the network | 20 |
| 7 | Design your own CNN and reflect | 15 |

The total is 180 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week04` and a notebook `lab4.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`make_shapes`, `out_size`, `mlp`, `SOBEL_X`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cell and then the blocks again from the top.
2. Do not install packages: PyTorch, NumPy and Matplotlib are already installed and there is no internet. Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook. This lab needs **no downloads at all**: every picture is drawn by `make_shapes`.
3. A CPU is enough. Most blocks finish in a second or two; the slowest ones (training the networks in Parts 1, 5 and 7) take from a few seconds to about a minute each. The GPU of the Spark is shared by the whole class and gives no advantage on pictures this small, so leave it alone. The helper cell prints which device it found.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste this into the first cell and run it. It fixes the random seed (so your numbers match the ones printed in this lab), defines the empty `RESULTS` dictionary and the `check(...)` function, and defines the helpers that draw, show, convolve, count and train:

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
RESULTS = {}


def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

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
    r = np.random.RandomState(seed)
    y = np.arange(n) % 5
    r.shuffle(y)
    X = np.stack([_draw(int(k), size, r, shift) for k in y])
    X = X + r.normal(0, noise, X.shape).astype(np.float32)
    return X[:, None].astype(np.float32), y.astype(np.int64)

def show_images(X, y=None, n=10, cols=5, titles=None):
    X = np.asarray(X); n = min(n, len(X)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows))
    for a in np.ravel(axes): a.axis("off")
    for i, a in enumerate(np.ravel(axes)[:n]):
        a.imshow(X[i].reshape(X[i].shape[-2:]), cmap="gray_r", vmin=0, vmax=1)
        if titles is not None: a.set_title(str(titles[i]), fontsize=8)
        elif y is not None: a.set_title(SHAPE_NAMES[int(y[i])], fontsize=8)
    plt.tight_layout(); plt.show()

def conv2d_np(x, k, stride=1, pad=0):
    x = np.pad(np.asarray(x, dtype=float), pad); k = np.asarray(k, dtype=float)
    kh, kw = k.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i * stride:i * stride + kh, j * stride:j * stride + kw] * k)
    return out

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_classifier(model, Xtr, ytr, Xte, yte, epochs=30, lr=0.01, opt="adam", batch=64, seed=0, verbose=False):
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

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `make_shapes(n, size, noise, shift, seed)` returns `X`, an array of `n` pictures of shape `(n, 1, size, size)` (float32, values near 0 for background and near 1 for the shape), and `y`, the labels 0 to 4 (`SHAPE_NAMES`: horizontal bar, vertical bar, cross, square, diagonal); `shift` is how many pixels a shape may move away from the centre in each direction and `noise` is the standard deviation of the random noise added to every pixel;
   - `show_images(X, y, n, cols, titles)` draws the first `n` pictures in a grid, with the class name (or your own titles) above each;
   - `conv2d_np(x, k, stride, pad)` is a convolution of one 2-D array with one kernel written with plain NumPy loops; like every deep-learning library it does not flip the kernel;
   - `count_params(model)` counts the numbers a PyTorch model can learn (weights and biases);
   - `train_classifier(model, Xtr, ytr, Xte, yte, epochs, lr, opt, batch, seed, verbose)` trains with cross-entropy and mini-batches and returns a dictionary with the lists `loss`, `train_acc` and `test_acc` (one value per epoch);
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

7. Run this smoke test in the next cell. It checks that the helpers work and that your Matplotlib shows figures in the notebook:

```python
Xd, yd = make_shapes(10, 16, noise=0.1, seed=0)
print("make_shapes ->", Xd.shape, Xd.dtype, "| labels:", yd.tolist())
show_images(Xd, yd, n=10)
k_demo = np.array([ [1, 0], [0, -1] ])
print("conv2d_np of a 3 x 3 picture with a 2 x 2 kernel has shape", conv2d_np(np.arange(9).reshape(3, 3), k_demo).shape)
tiny = nn.Sequential(nn.Flatten(), nn.Linear(256, 5))
hist = train_classifier(tiny, Xd, yd, Xd, yd, epochs=5, lr=0.05)
print("count_params(tiny) =", count_params(tiny), "| loss", round(hist["loss"][0], 3), "->", round(hist["loss"][-1], 3))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
Running on: cpu
make_shapes -> (10, 1, 16, 16) float32 | labels: [2, 3, 4, 4, 1, 1, 2, 3, 0, 0]
conv2d_np of a 3 x 3 picture with a 2 x 2 kernel has shape (2, 2)
count_params(tiny) = 1285 | loss 1.727 -> 0.036
```

   You should see `Running on: cpu` (or `cuda`) after the first cell, then ten small pictures (two or more of each shape, with a little speckle), the lines above, and the loss falling during the five epochs. The `conv2d_np` line shows that a 3 × 3 picture and a 2 × 2 kernel give a 2 × 2 output. If you do not see the pictures, run `%matplotlib inline` in a cell first.

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cell and the earlier blocks again, in order.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the helper ends with `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42`, and that you did not train a model twice in the same variable. The blocks that train a network set their own seed before building it, but small differences in the last digit can appear between PyTorch versions; the stated ranges and the checks are tolerant, the exact digits are not.
- A cell seems frozen: stop it (the square button), restart the kernel and run again from the top. If a cell takes more than two minutes, you probably changed a loop count or a data size.

### Part 1. Pictures are numbers, and why dense layers struggle (25 min)

*Lesson section: "Images".* A photograph feels like a *thing*: a face, a jeepney, a handwritten 7. To a computer it is a table of numbers, and a network that takes in the table has to decide what to do with it. In this part you make a small set of pictures, look at them as numbers, count how many parameters a dense (fully connected) layer needs for them, and then train a dense network on shapes that sit in the middle of the picture. After that you run three experiments that expose what a dense network does *not* know: it does not know where a shape is allowed to be, and it does not know which pixels are neighbours. These are exactly the two weaknesses that convolution repairs in Part 2.

All pictures come from `make_shapes`, a helper that draws five kinds of shape (a horizontal bar, a vertical bar, a cross, a square outline and a diagonal) on a 16 × 16 grid, can move the shape away from the centre (`shift`) and can add random noise (`noise`). Nothing is downloaded.

**Imagine first.** A dense network will be trained on shapes that are always in the **centre** of a 16 × 16 picture. Then it is tested on shapes that have moved 1, 2, 3 or 4 pixels away from the centre. Before you run anything, write two guesses: the accuracy (between 0 and 1) on the *centred* test pictures, and the accuracy when the shapes may move **2** pixels from the centre. There are five classes, so guessing at random gives 0.20. You will compare your guesses with the measured numbers in Steps 1.3 and 1.4.

```python
# Imagine first: a dense network trained on centred shapes, tested on shapes that moved.
my_acc_centred = None   # replace None with your guess for the accuracy on centred test pictures (0 to 1)
my_acc_shift2 = None    # replace None with your guess for the accuracy when shapes may move 2 pixels
```

1. **Step 1.1. Make the pictures and look at them as numbers.** A greyscale picture is a grid of numbers: 0 means a blank (white) pixel and 1 means a fully inked (dark) pixel. PyTorch keeps a whole *batch* of pictures in one array with four numbers in its shape, **(N, C, H, W)**: N pictures, C channels (1 for grey, 3 for red-green-blue colour), H rows (height) and W columns (width). In the first block you create 300 pictures with a little noise, print the shapes, and draw the first ten. In the second block you print one picture as a table of numbers, so that you see that the "bar" is nothing but a row of values near 1 in a sea of values near 0. In the third block you stack three copies of a grey picture to imitate a colour picture and see how the shape changes.

```python
X, y = make_shapes(300, 16, noise=0.1, seed=1)
print("X.shape:", X.shape, "| dtype:", X.dtype, "| y.shape:", y.shape)
print("one picture, shape (C, H, W):", X[0].shape, "| min %.2f, max %.2f" % (X[0].min(), X[0].max()))
label0 = int(y[0])
print("its label:", label0, "=", SHAPE_NAMES[label0])
print("pictures per class:", np.bincount(y).tolist())
show_images(X, y, n=10)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
X.shape: (300, 1, 16, 16) | dtype: float32 | y.shape: (300,)
one picture, shape (C, H, W): (1, 16, 16) | min -0.27, max 1.18
its label: 4 = diagonal
pictures per class: [60, 60, 60, 60, 60]
```

```python
print("picture 0 as numbers (rounded to one decimal):")
for row in X[0, 0]:
    print(" ".join(f"{v:4.1f}" for v in row))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
picture 0 as numbers (rounded to one decimal):
-0.1  0.1  0.0  0.1 -0.0  0.1  0.1  0.0 -0.0 -0.2 -0.0  0.0 -0.1  0.2 -0.1  0.1
 0.0  0.1  0.1 -0.1 -0.1  0.0  0.2 -0.1 -0.0  0.0  0.0  0.0  0.0  0.1 -0.1  0.1
 0.2  0.1  0.0  0.1  0.1  0.0  0.1 -0.1 -0.2 -0.1 -0.2 -0.2  0.1 -0.2  0.0 -0.0
-0.1  0.1 -0.2 -0.1  0.0  0.0  0.0  0.1 -0.1 -0.1  0.0  0.1  0.0  0.1  0.0  0.2
 0.0  0.1 -0.2 -0.0  1.0 -0.1  0.0 -0.0  0.0  0.1 -0.1  0.0 -0.2 -0.0  0.0 -0.1
 0.1 -0.0 -0.0  0.0 -0.0  1.0  0.2 -0.0 -0.1 -0.1 -0.0  0.1 -0.1  0.1  0.0 -0.1
-0.1 -0.0  0.1  0.2 -0.1  0.0  0.9  0.1  0.1  0.0  0.1  0.0  0.1  0.1  0.1  0.0
-0.1  0.1  0.1 -0.1  0.1 -0.1  0.1  1.0  0.0  0.0  0.1  0.3 -0.0 -0.1  0.1 -0.0
 0.1  0.1 -0.0 -0.1  0.0  0.0 -0.1 -0.1  1.1 -0.1 -0.1 -0.0 -0.1  0.0 -0.1  0.1
 0.0 -0.0 -0.0  0.1  0.0  0.2  0.1  0.0 -0.0  1.2 -0.0 -0.0 -0.1 -0.0  0.1  0.1
 0.1  0.1  0.1 -0.0  0.1 -0.0 -0.1  0.0 -0.0 -0.1  0.9  0.1 -0.0  0.2 -0.1 -0.3
 0.1 -0.0  0.1 -0.0  0.0 -0.1  0.0 -0.0 -0.0 -0.0  0.1  1.0  0.0  0.0 -0.1 -0.1
-0.0  0.0  0.1 -0.1  0.0 -0.2  0.1 -0.0 -0.1 -0.2  0.1  0.0  0.8  0.0  0.0  0.1
-0.0  0.2 -0.1  0.1  0.0 -0.0  0.0 -0.2 -0.0  0.1 -0.1 -0.1 -0.0 -0.0 -0.1 -0.1
 0.0 -0.1  0.1 -0.0  0.1 -0.1  0.0 -0.2 -0.0 -0.0  0.1 -0.1  0.4 -0.1  0.0 -0.1
 0.0  0.0  0.0  0.1 -0.0 -0.1  0.0 -0.0  0.1  0.1  0.1  0.1 -0.0 -0.0 -0.0 -0.2
```

```python
rgb = np.concatenate([X[:8], X[:8], X[:8] ], axis=1)   # three copies of the grey pictures as red, green, blue channels
print("grey batch:", X[:8].shape, "-> colour-like batch:", rgb.shape)
print("numbers in one grey picture:", X[0].size, "| in one colour picture:", rgb[0].size)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
grey batch: (8, 1, 16, 16) -> colour-like batch: (8, 3, 16, 16)
numbers in one grey picture: 256 | in one colour picture: 768
```

   You should see the shapes `X.shape: (300, 1, 16, 16)`, `dtype: float32` and `y.shape: (300,)`, and one picture of shape `(1, 16, 16)` whose values lie between about -0.27 and 1.18 (the noise pushes a few pixels slightly below 0 and above 1). Picture 0 has the label 4 (diagonal); each class has 60 pictures. The ten small pictures show the five shapes in the middle of a grey, speckled background. In the table of numbers you can find the shape: values near 1.0 form a line while the background is made of small numbers like 0.1 or -0.1. The third block prints `(8, 1, 16, 16) -> (8, 3, 16, 16)`: a colour picture has three times as many numbers (256 grey, 768 colour). A real photo, 224 pixels wide and 224 high, with three channels, has 150,528 numbers.

   Tips and common mistakes:

   - The first number of the shape is *how many pictures*, never part of one picture. A single picture is `X[0]`, shape `(1, 16, 16)`; its pixel table is `X[0, 0]`, shape `(16, 16)`.
   - `dtype float32` matters: PyTorch layers expect 32-bit floats, and `make_shapes` already returns them. If you ever see an error about `Double` and `Float`, convert with `.astype(np.float32)`.
   - The noise is random but fixed by `seed=1`, so you and your neighbour see the same pictures.

   Try also (optional):

   - Draw 10 pictures with `noise=0.5` and decide how much noise you still tolerate when you try to name the shape by eye.
   - Print `X[0, 0].round(0)` for a picture with `noise=0` and mark the pixels that are 1.

   *Exercise 1.1:* a school photo app stores a batch of **32** colour pictures that are each **28 × 28** pixels. Write the shape of the batch as a tuple in `shape_32` (in the order N, C, H, W) and the number of numbers in one picture in `n_numbers`. Why it matters: shape errors are the most common bug in this course, and reading a shape fluently saves hours.

```python
shape_32 = None   # a tuple (N, C, H, W)
n_numbers = None  # how many numbers one 28 x 28 colour picture holds
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if shape_32 is None or n_numbers is None:
    print("Fill in shape_32 and n_numbers above, then run this cell again.")
else:
    check("shape_32 is (32, 3, 28, 28)", tuple(shape_32) == (32, 3, 28, 28), "Order is N, C, H, W: 32 pictures, 3 colour channels, 28 rows, 28 columns.")
    check("n_numbers is 3 * 28 * 28 = 2352", n_numbers == 2352, "Multiply channels, rows and columns for ONE picture (not the batch).")
    RESULTS["p1_shape_32"] = shape_32; RESULTS["p1_numbers_one_picture"] = n_numbers
```

2. **Step 1.2. Count the parameters of a dense layer.** A dense layer connects **every** input number to **every** output unit, so a layer from `n_in` inputs to `n_out` units has `n_in · n_out` weights plus `n_out` biases. For a picture, `n_in` is the number of pixels, and that grows fast. The first block counts, by hand and with the helper `count_params`, the dense network `[256, 64, 5]` that you will train in Step 1.3 (a 16 × 16 picture has 256 pixels) and compares it with a convolution layer with eight 3 × 3 filters, which you meet properly in Part 2. The second block repeats the dense calculation for a real photo. Do the arithmetic yourself first, then let the code agree with you.

```python
dense1 = 256 * 64 + 64          # 256 inputs to 64 units: weights + biases
dense2 = 64 * 5 + 5             # 64 units to 5 classes
print("dense 256 -> 64:", dense1, "| dense 64 -> 5:", dense2, "| total:", dense1 + dense2)

mlp_demo = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
print("count_params(mlp_demo) =", count_params(mlp_demo))
assert count_params(mlp_demo) == dense1 + dense2

conv_demo = nn.Conv2d(1, 8, kernel_size=3)       # 8 filters of size 3 x 3 on a 1-channel picture
by_hand = 3 * 3 * 1 * 8 + 8
print("conv 3x3, 8 filters: by hand", by_hand, "| count_params", count_params(conv_demo))
assert count_params(conv_demo) == by_hand
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
dense 256 -> 64: 16448 | dense 64 -> 5: 325 | total: 16773
count_params(mlp_demo) = 16773
conv 3x3, 8 filters: by hand 80 | count_params 80
```

```python
pixels = 224 * 224 * 3
big = pixels * 1000 + 1000
print(f"a 224 x 224 colour picture has {pixels:,} numbers; ONE dense layer to 1,000 units has {big:,} parameters")
assert big == 150_529_000
print("a 3x3 convolution with 64 filters on that picture:", 3 * 3 * 3 * 64 + 64, "parameters, whatever the picture size")
RESULTS["p1_dense_photo_params"] = big
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
a 224 x 224 colour picture has 150,528 numbers; ONE dense layer to 1,000 units has 150,529,000 parameters
a 3x3 convolution with 64 filters on that picture: 1792 parameters, whatever the picture size
```

   You should see `dense 256 -> 64: 16448 | dense 64 -> 5: 325 | total: 16773`, and `count_params(mlp_demo) = 16773`: your hand count and the helper agree. The convolution layer with eight 3 × 3 filters has only **80** parameters (`3 · 3 · 1 · 8 + 8`), about 200 times fewer than the first dense layer alone, although it also looks at the whole picture. The second block prints `150,528` numbers for a 224 × 224 colour picture and **150,529,000** parameters for one dense layer to 1,000 units: a hundred and fifty million numbers to learn, in a single layer, before any useful work. The asserts pass silently. The 3 × 3 convolution with 64 filters needs only 1,792 parameters, and that number does not change if the picture is 28 or 224 pixels wide.

   Tips and common mistakes:

   - Every unit has its own bias, which is why you add `n_out` after the weights.
   - `nn.Flatten()` turns `(N, 1, 16, 16)` into `(N, 256)`. It has no parameters.
   - `count_params` counts only parameters that learn (`requires_grad`). ReLU, Flatten and pooling add zero.

   Try also (optional):

   - Change `256` to `784` and `64` to `100` and check that the first layer has the 78,500 parameters from the lesson.
   - Count the parameters of `nn.Conv2d(3, 16, 3)` by hand: `3 * 3 * 3 * 16 + 16`.

   *Exercise 1.2:* compute the parameters of (a) a dense layer from a 28 × 28 greyscale picture to 100 units, store it in `p_dense_28`, and (b) a convolution layer with six 5 × 5 filters on a greyscale picture, store it in `p_conv_5x5_6`. Why it matters: these two numbers (78,500 and 156) are the first layer of the dense network and of the LeNet-style network in the lesson.

```python
p_dense_28 = None      # 28*28 inputs to 100 units
p_conv_5x5_6 = None    # six 5x5 filters on one channel
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if p_dense_28 is None or p_conv_5x5_6 is None:
    print("Fill in both numbers above, then run this cell again.")
else:
    check("dense 784 -> 100 has 78,500", p_dense_28 == 78500, "784 * 100 weights plus 100 biases.")
    check("conv with six 5x5 filters has 156", p_conv_5x5_6 == 156, "5 * 5 * 1 * 6 weights plus 6 biases.")
    check("count_params agrees", count_params(nn.Conv2d(1, 6, 5)) == p_conv_5x5_6, "Build nn.Conv2d(1, 6, 5) and count it.")
    RESULTS["p1_dense_28_params"] = p_dense_28; RESULTS["p1_conv_5x5_6_params"] = p_conv_5x5_6
```

3. **Step 1.3. Train a dense network on centred shapes.** Now the baseline. You build a small multilayer perceptron (MLP: a network made only of dense layers) with 256 inputs, 64 hidden ReLU units and 5 outputs, and train it on 500 centred shapes (`shift=0`, noise 0.15). The helper `train_classifier` trains with cross-entropy and mini-batches, and returns the loss and the accuracy on the training and the test pictures after every epoch. The test pictures are 300 *new* pictures drawn with a different seed, so the test accuracy tells you whether the network learned the shapes or just remembered the training pictures. The block also defines the function `accuracy(model, X, y)` that you will use all lab long.

```python
EPOCHS = 8 if QUICK else 30
Xtr, ytr = make_shapes(500, 16, noise=0.15, shift=0, seed=1)    # training pictures, centred
Xte, yte = make_shapes(300, 16, noise=0.15, shift=0, seed=2)    # new test pictures, centred

def accuracy(model, X, y):
    """Fraction of pictures the model classifies correctly."""
    model.eval()
    with torch.no_grad():
        return float((model(torch.as_tensor(X)).argmax(1) == torch.as_tensor(y)).float().mean())

def make_mlp(hidden=64):
    return nn.Sequential(nn.Flatten(), nn.Linear(256, hidden), nn.ReLU(), nn.Linear(hidden, 5))

torch.manual_seed(SEED)
mlp = make_mlp(64)
h_mlp = train_classifier(mlp, Xtr, ytr, Xte, yte, epochs=EPOCHS, lr=0.01, seed=0, verbose=True)
print("MLP parameters:", count_params(mlp))
print(f"final train accuracy {h_mlp['train_acc'][-1]:.3f} | test accuracy {h_mlp['test_acc'][-1]:.3f}")
RESULTS["p1_mlp_centred_test_acc"] = h_mlp["test_acc"][-1]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
epoch   0  loss 0.994  train 1.00  test 1.00
epoch   5  loss 0.000  train 1.00  test 1.00
epoch  10  loss 0.000  train 1.00  test 1.00
epoch  15  loss 0.000  train 1.00  test 1.00
epoch  20  loss 0.000  train 1.00  test 1.00
epoch  25  loss 0.000  train 1.00  test 1.00
epoch  29  loss 0.000  train 1.00  test 1.00
MLP parameters: 16773
final train accuracy 1.000 | test accuracy 1.000
```

```python
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(h_mlp["loss"], c="tab:red"); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
ax[1].plot(h_mlp["train_acc"], label="train"); ax[1].plot(h_mlp["test_acc"], label="test")
ax[1].set_ylim(0, 1.05); ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()
print("epochs until the test accuracy first reached 0.95:", next((i + 1 for i, a in enumerate(h_mlp["test_acc"]) if a >= 0.95), None))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
epochs until the test accuracy first reached 0.95: 1
```

   You should see the loss fall from 0.99 at epoch 0 to 0.0001 at the end, and an accuracy of **1.00 on the training pictures and 1.00 on the new test pictures**, with 16,773 parameters. The learning curves show that the network is already at (or very near) perfect accuracy after the first epoch or two: shapes in the centre with this little noise are an easy job. Compare with your first *Imagine first* guess, `my_acc_centred`. Do not be fooled by the 1.00: the network only ever saw shapes in one place. The next step shows what that one-place knowledge is worth.

   Tips and common mistakes:

   - `train_classifier` trains the model *in place* and returns the history; the model itself is `mlp`.
   - Two different seeds appear on purpose: `torch.manual_seed(SEED)` fixes the starting weights and `seed=0` inside `train_classifier` fixes the order of the mini-batches.
   - `QUICK` is False on the Spark; it only shortens the loop when the author tests the notebook.

   Try also (optional):

   - Set `noise=0.5` in both data sets and train again. Which accuracy changes first, train or test?
   - Use `opt="sgd"` and `lr=0.05` and compare how many epochs it takes to reach 0.95.

   *Exercise 1.3:* train a dense network with only **16** hidden units (`make_mlp(16)`) on the same centred data, as in the block above (same epochs, `lr=0.01`), and store its final test accuracy in `acc_small`. Does the small network still solve the centred problem? Why it matters: it tells you how much of the 16,773 parameters this easy problem really needed.

```python
acc_small = None   # final test accuracy of the 16-hidden-unit MLP
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if acc_small is None:
    print("Fill in acc_small above, then run this cell again.")
else:
    check("acc_small is a fraction between 0 and 1", 0 <= acc_small <= 1, "Use accuracy(model, Xte, yte) after training.")
    check("a 16-unit MLP still solves centred shapes (above 0.9)", acc_small > 0.9, "Check that you trained on Xtr, ytr and tested on Xte, yte, with lr=0.01.")
    RESULTS["p1_mlp16_centred_test_acc"] = acc_small
```

4. **Step 1.4. The shift test: move the shapes.** Take the **same trained network** `mlp` from Step 1.3 and test it on new pictures in which the shape may sit up to `s` pixels away from the centre, for s = 0, 1, 2, 3 and 4. The network is not trained again. If it had learned "what a vertical bar *is*", a bar one pixel to the left would still be a bar. If it only learned "which of the 256 pixels light up for each class", a moved bar is a different input and may be misclassified. The block creates a fresh test set for every shift, measures the accuracy, prints a small table and draws a bar chart.

```python
shift_acc_mlp = []
for s in range(5):
    Xs, ys = make_shapes(300, 16, noise=0.15, shift=s, seed=10 + s)
    shift_acc_mlp.append(accuracy(mlp, Xs, ys))
    print(f"shift {s}: test accuracy {shift_acc_mlp[-1]:.3f}")

plt.figure(figsize=(5.5, 3.2))
plt.bar(range(5), shift_acc_mlp, color="tab:red")
plt.axhline(0.2, ls="--", c="gray"); plt.text(4.45, 0.22, "guessing", ha="right", fontsize=8)
plt.xlabel("how far the shape may move from the centre (pixels)"); plt.ylabel("accuracy"); plt.ylim(0, 1.05)
plt.title("MLP trained on centred shapes"); plt.tight_layout(); plt.show()
RESULTS["p1_mlp_shift_acc"] = [round(a, 3) for a in shift_acc_mlp]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
shift 0: test accuracy 1.000
shift 1: test accuracy 0.353
shift 2: test accuracy 0.270
shift 3: test accuracy 0.210
shift 4: test accuracy 0.150
```

```python
Xs2, ys2 = make_shapes(300, 16, noise=0.15, shift=2, seed=12)
pred = mlp(torch.as_tensor(Xs2)).argmax(1).numpy()
true_names = [SHAPE_NAMES[k] for k in ys2]
pred_names = [SHAPE_NAMES[k] for k in pred]
show_images(Xs2[:10], titles=[f"{true_names[i]}\n-> {pred_names[i]}" for i in range(10)])
```

   You should see a collapse: accuracy **1.00** at shift 0, then **0.35** at shift 1, 0.27 at shift 2, 0.21 at shift 3 and 0.15 at shift 4. Shifting the shape by a *single pixel* is enough to throw the network from perfect to about one right answer in three; by shift 3 and 4 it is around or below guessing level (0.20, the dashed line), because bigger shifts also push parts of a shape out of the picture. Compare with your second guess, `my_acc_shift2` (the measured value for shift 2 is 0.27). The second block draws ten shifted test pictures with the true name above and the network's guess under it: look at how many wrong guesses involve a shape that is perfectly easy to recognise by eye. A person would not notice a one-pixel move; this network was never *told* that a moved bar is still a bar, and it has no way of working that out.

   Tips and common mistakes:

   - Shifts are applied to the *test* data only. The network's weights do not change in this step.
   - Below 0.20 is possible because the shape can leave the picture or shrink to a few pixels at the edge; then the network sees something unlike all of its training pictures.
   - Each shift uses a different seed, so the test pictures differ between rows of the table; with 300 pictures an accuracy has a spread of about plus or minus 0.03.

   Try also (optional):

   - Test the same network on `shift=1` pictures with `noise=0.0`. Does the noise matter, or only the move?
   - Print the confusion between classes: which wrong answer does the network give most often?

   *Exercise 1.4:* using the list `shift_acc_mlp`, find the smallest shift `first_bad_shift` at which the accuracy falls **below 0.8** and store it, and store in `drop_1` how much accuracy is lost between shift 0 and shift 1 (as a fraction, for example 0.65). Why it matters: it turns the plot into two numbers you can quote.

```python
first_bad_shift = None   # smallest shift whose accuracy is below 0.8
drop_1 = None            # accuracy at shift 0 minus accuracy at shift 1
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if first_bad_shift is None or drop_1 is None:
    print("Fill in both values above, then run this cell again.")
else:
    true_first = next(s for s, a in enumerate(shift_acc_mlp) if a < 0.8)
    check("first_bad_shift is right", first_bad_shift == true_first, "Loop over shift_acc_mlp with enumerate and stop at the first value below 0.8.")
    check("drop_1 is right", abs(drop_1 - (shift_acc_mlp[0] - shift_acc_mlp[1])) < 1e-6, "Subtract the accuracy at shift 1 from the accuracy at shift 0.")
    RESULTS["p1_first_bad_shift"] = first_bad_shift; RESULTS["p1_drop_1"] = drop_1
```

5. **Step 1.5. The pixel-shuffle experiment.** Here is a stranger test. Pick one fixed random permutation of the 256 pixel positions (the same for every picture) and rearrange the pixels of **every** picture, training and test, in the same way. To a human, the shuffled pictures are unrecognisable noise. A dense layer, though, treats pixel number 17 and pixel number 18 as two unrelated inputs with their own weights; it never uses the fact that they sit next to each other. So if the shuffle is the same everywhere, the network should do just as well. The block shuffles, draws five shuffled pictures, retrains the same MLP and reports the accuracy.

```python
perm = np.random.RandomState(0).permutation(256)          # one fixed shuffle of the 256 pixel positions

def shuffle_pixels(X, perm):
    return X.reshape(len(X), -1)[:, perm].reshape(X.shape)

Xtr_s, Xte_s = shuffle_pixels(Xtr, perm), shuffle_pixels(Xte, perm)
show_images(Xte_s, yte, n=5)

torch.manual_seed(SEED)
mlp_s = make_mlp(64)
h_s = train_classifier(mlp_s, Xtr_s, ytr, Xte_s, yte, epochs=EPOCHS, lr=0.01, seed=0)
print(f"normal pictures:   train {h_mlp['train_acc'][-1]:.3f} | test {h_mlp['test_acc'][-1]:.3f}")
print(f"shuffled pictures: train {h_s['train_acc'][-1]:.3f} | test {h_s['test_acc'][-1]:.3f}")
RESULTS["p1_shuffled_test_acc"] = h_s["test_acc"][-1]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
normal pictures:   train 1.000 | test 1.000
shuffled pictures: train 1.000 | test 1.000
```

   You should see that the five shuffled pictures look like random speckle, and yet the printed accuracies are the same: **1.00 train / 1.00 test** for the normal pictures and **1.00 train / 1.00 test** for the shuffled ones. The dense network found the pattern in pixel positions that no human can read. This is the surprising lesson of the experiment: a dense network gets *no benefit* from the picture's layout. Everything a human uses (a line is a run of neighbouring pixels, a corner is two lines that meet) is invisible to it; the network sees 256 unrelated numbers. That is a waste when a picture has structure, and it is the reason convolution layers will be built around the idea of a neighbourhood.

   Tips and common mistakes:

   - It must be the *same* permutation for training and test pictures, and for every picture. Different shuffles per picture would destroy the information.
   - `X.reshape(len(X), -1)` flattens each picture into a row of 256 numbers; `[:, perm]` reorders the columns; the last `reshape` restores the picture shape.
   - This is true for dense networks only. Convolution layers (Part 2) look at neighbourhoods, so a shuffle would destroy exactly what they are built to use.

   Try also (optional):

   - Shuffle only the *rows* of each picture instead of all pixels. Is the picture recognisable to you? To the MLP?

   *Exercise 1.5:* the network `mlp_s` learned the shuffle `perm`. Create a **different** permutation `perm2 = np.random.RandomState(1).permutation(256)`, shuffle the *test* pictures `Xte` with it, and store the accuracy of `mlp_s` on them in `acc_wrong_perm`. Predict first: will it be near 1.0, near 0.2, or in between? Why it matters: it shows what the network really learned: a map from *positions* to classes.

```python
acc_wrong_perm = None   # accuracy of mlp_s on test pictures shuffled with a DIFFERENT permutation
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if acc_wrong_perm is None:
    print("Fill in acc_wrong_perm above, then run this cell again.")
else:
    perm2_check = np.random.RandomState(1).permutation(256)
    truth = accuracy(mlp_s, shuffle_pixels(Xte, perm2_check), yte)
    check("acc_wrong_perm matches", abs(acc_wrong_perm - truth) < 1e-6, "Use shuffle_pixels(Xte, perm2) and accuracy(mlp_s, ..., yte).")
    check("it is far below the matching-shuffle accuracy", acc_wrong_perm < 0.6, "A wrong shuffle should leave only about chance (0.2).")
    RESULTS["p1_wrong_perm_acc"] = acc_wrong_perm
```

6. **Step 1.6. Imagine and write: what layout means.** A picture's layout carries meaning. In this last step you see the problem in numbers, then you write about it. In the block, you make five noise-free pictures (one of each class) and measure the straight-line (Euclidean) distance between pictures, treating each as a point in 256 dimensions: how far is a vertical bar from the **same bar moved one pixel**, and how far is it from a **horizontal bar**? To a dense network that has no idea about neighbours, these distances are the only notion of "similar" it can build on.

```python
X0, y0 = make_shapes(50, 16, noise=0.0, shift=0, seed=3)
pics = {k: X0[np.where(y0 == k)[0][0], 0] for k in range(5)}      # one noise-free picture of each class
vbar, hbar = pics[1], pics[0]
vbar_moved = np.roll(vbar, 1, axis=1)                               # the same vertical bar, one pixel to the right

def dist(a, b):
    return float(np.sqrt(((a - b) ** 2).sum()))

print(f"vertical bar vs the same bar moved by one pixel: {dist(vbar, vbar_moved):.2f}")
print(f"vertical bar vs a horizontal bar:                {dist(vbar, hbar):.2f}")
print(f"vertical bar vs a cross:                         {dist(vbar, pics[2]):.2f}")
RESULTS["p1_dist_moved"] = dist(vbar, vbar_moved); RESULTS["p1_dist_other_class"] = dist(vbar, hbar)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
vertical bar vs the same bar moved by one pixel: 4.24
vertical bar vs a horizontal bar:                4.00
vertical bar vs a cross:                         2.83
```

   You should see **4.24** for the bar moved by one pixel, and **4.00** for a horizontal bar. In pixel space the *same* bar one pixel to the right is **farther** from the original than a completely different shape is (the cross is only 2.83 away, closer still). A thin bar moved by one pixel no longer overlaps itself at all, while the horizontal bar still shares the centre pixel with it. So for a dense network, "similar picture" does not mean "similar to the eye". A human compares pictures by *structure*; a dense network compares them pixel by pixel.

   Tips and common mistakes:

   - `np.roll(vbar, 1, axis=1)` moves every column one step to the right (the bar is far from the edge, so nothing wraps around).
   - This is a comparison of raw pixels. The network has hidden layers that can learn better features, but they are built from the same position-specific weights.

   Try also (optional):

   - Move the bar by two and three pixels. Does the distance keep growing?
   - Blur the pictures a little (average each pixel with its neighbours) before measuring. What happens to the distance for the moved bar?

   *Exercise 1.6:* write **three short sentences** in the list `layout_notes`, each naming one way in which the layout of a picture carries meaning that a dense network cannot use. Ideas to start from: neighbouring pixels belong together (an edge is a short line of them), the same object can appear in different places, and the same pattern (an eye, a letter) repeats within one picture. Use examples from your school or community (a school ID photo, a jeepney route sign, a market price board).

```python
layout_notes = [
    "",   # 1. ...
    "",   # 2. ...
    "",   # 3. ...
]
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
notes = [n for n in layout_notes if n.strip()]
if len(notes) < 3:
    print("Write three sentences in layout_notes above, then run this cell again.")
else:
    check("three notes of at least 25 characters each", all(len(n) >= 25 for n in notes[:3]), "Write a full sentence for each idea, with an example.")
    RESULTS["p1_layout_notes"] = notes[:3]
```

### Part 2. Convolution by hand (25 min)

*Lesson sections: "Convolution" and "Output size".* Picture a barangay inspector walking along a street with a small stencil, a cardboard frame with a few holes. At every house the inspector lays the frame over the wall, multiplies what shows through each hole by a number printed beside it, adds everything up and writes one number in a notebook. Then the inspector steps to the next house and does the same, with the **same** frame and the **same** printed numbers. That is a convolution: *local* (the frame looks only at a small patch) and *shared* (one set of numbers is used at every position). In this part you do the inspector's job with your own code, first on the lesson's small example, then with the output-size formula, stride, padding, channels and parameter counts, and finally you count how much cheaper a shared stencil is than a dense layer.

One note on the word. What deep-learning libraries call convolution is, strictly speaking, *cross-correlation*: the kernel is not flipped. Step 2.6 shows the difference with numbers. The helper `conv2d_np(x, k, stride, pad)` from the setup cell is the inspector, written in plain NumPy loops.

**Imagine first.** An inspector slides a 5 × 5 stencil over a 28 × 28 picture, one pixel at a time (stride 1), without ever stepping outside the picture (no padding). How many positions does the stencil have along one side, so how big is the output? Write the side length, as a single number, before you read on. (Hint: when the stencil sits at the very left, where is its right edge? How many steps can it take?)

```python
# Imagine first: output size of a 5 x 5 kernel on a 28 x 28 picture, stride 1, no padding.
my_out = None   # replace None with the side length of the output (the output is my_out x my_out)
```

1. **Step 2.1. The lesson's example, by hand and in code.** The lesson's worked example uses a 4 × 4 input `x` and a 2 × 2 kernel `K`. The output cell `y[i, j]` is the sum of the products of the kernel with the 2 × 2 window of `x` that starts at row i, column j: `y[i, j] = Σ x[i+a, j+c] · K[a, c]`. The first block does this for one output cell and prints every single product, so that you can follow the arithmetic with a pencil. The second block computes the whole output with `conv2d_np`, asserts that it equals the lesson's matrix, and checks the result against PyTorch's `F.conv2d` (which expects the shape `(N, C, H, W)`, so you reshape to `(1, 1, H, W)`). Write the nine output numbers on paper before you look at the printout.

```python
def pm(a, d=0):
    """Print a 2-D array as a tidy table."""
    for row in np.atleast_2d(np.asarray(a)):
        print("  ".join(f"{v:{5 + d}.{d}f}" for v in row))

x = np.array([ [1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2] ], dtype=float)
K = np.array([ [1, 0], [0, -1] ], dtype=float)
print("input x (4 x 4):"); pm(x)
print("kernel K (2 x 2):"); pm(K)

i, j = 0, 1                               # the output cell y[0, 1]
print(f"\nthe window of x for y[{i},{j}], and the products with K:")
total = 0.0
for a in range(2):
    for c in range(2):
        prod = x[i + a, j + c] * K[a, c]
        total += prod
        print(f"  x[{i + a},{j + c}] * K[{a},{c}] = {x[i + a, j + c]:.0f} * {K[a, c]:.0f} = {prod:.0f}")
print("sum =", total)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
input x (4 x 4):
    1      2      0      1
    0      1      3      1
    2      1      0      0
    1      0      1      2
kernel K (2 x 2):
    1      0
    0     -1

the window of x for y[0,1], and the products with K:
  x[0,1] * K[0,0] = 2 * 1 = 2
  x[0,2] * K[0,1] = 0 * 0 = 0
  x[1,1] * K[1,0] = 1 * 0 = 0
  x[1,2] * K[1,1] = 3 * -1 = -3
sum = -1.0
```

```python
y = conv2d_np(x, K)
print("y = conv2d_np(x, K), shape", y.shape); pm(y)
y_lesson = np.array([ [0, -1, -1], [-1, 1, 3], [2, 0, -2] ], dtype=float)
assert np.allclose(y, y_lesson), "should be the lesson's matrix"

y_torch = F.conv2d(torch.tensor(x).reshape(1, 1, 4, 4), torch.tensor(K).reshape(1, 1, 2, 2))
print("F.conv2d output shape:", tuple(y_torch.shape))
assert np.allclose(y_torch.numpy()[0, 0], y)
print("conv2d_np, the lesson and F.conv2d agree")
RESULTS["p2_example_y"] = y.tolist()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
y = conv2d_np(x, K), shape (3, 3)
    0     -1     -1
   -1      1      3
    2      0     -2
F.conv2d output shape: (1, 1, 3, 3)
conv2d_np, the lesson and F.conv2d agree
```

   You should see the window for `y[0,1]` made of the cells `x[0,1]=2`, `x[0,2]=0`, `x[1,1]=1`, `x[1,2]=3`. The four products are 2, 0, 0 and -3, and their sum is **-1**, which is the lesson's value for `y[0,1]`. The full output is a 3 × 3 table with the rows `0 -1 -1`, `-1 1 3` and `2 0 -2`; the asserts pass silently and the last line confirms that your NumPy loops, the lesson and PyTorch agree. Notice the shape: a 4 × 4 input and a 2 × 2 kernel give a **3 × 3** output. The kernel can only be placed at 3 positions along each side, because at the fourth position it would stick out of the picture.

   Tips and common mistakes:

   - Only the two diagonal cells of the window count in this example: the zeros in `K` switch the other two off. That is why `K` computes a *difference along the diagonal* (top-left minus bottom-right).
   - `F.conv2d` has no bias unless you pass one as a third argument; `nn.Conv2d` owns a learnable bias.
   - If `np.allclose` fails by a hair, the cause is usually `float32` versus `float64`. Here both are `float64` so the match is exact.

   Try also (optional):

   - Compute `y[1, 1]` by hand (answer: 1) and then print `x[1:3, 1:3] * K` to see the four products.
   - Use the kernel `np.array([ [0, 0], [0, 1] ])`. What does it do to the picture? (It copies the picture, shifted by one cell.)

   *Exercise 2.1:* compute, by hand first and then with the loop from the block above, the output cell `y[1, 2]` (row 1, column 2) and the output cell `y[2, 1]` of the lesson's example. Store them in `y_1_2` and `y_2_1`. Why it matters: reading the window correctly (which rows and columns of `x`) is the skill behind every convolution, and also behind the gradients in Week 4's lesson.

```python
y_1_2 = None   # output cell in row 1, column 2
y_2_1 = None   # output cell in row 2, column 1
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if y_1_2 is None or y_2_1 is None:
    print("Fill in both cells above, then run this cell again.")
else:
    check("y[1,2] is 3", y_1_2 == 3, "The window is x[1:3, 2:4]; multiply by K and add.")
    check("y[2,1] is 0", y_2_1 == 0, "The window is x[2:4, 1:3]; multiply by K and add.")
    RESULTS["p2_y12_y21"] = [y_1_2, y_2_1]
```

2. **Step 2.2. The output-size formula.** How many windows fit? With an input of side `n`, a kernel of side `k`, a stride `s` (how many pixels the stencil moves each step) and padding `p` (how many rows and columns of zeros are added on every side of the picture first), the output side is `floor((n + 2p - k) / s) + 1`. The first block writes this as a Python function `out_size`, prints a table for a 28-pixel input and checks the three examples of the lesson. The second block proves the formula against reality: it runs `conv2d_np` on random pictures for many combinations and compares the real output shape with the formula. A formula that you only believe is much weaker than one that you tested.

```python
def out_size(n, k, s=1, p=0):
    """Side of the output for input side n, kernel k, stride s, padding p."""
    return (n + 2 * p - k) // s + 1

print("input 28 x 28:    k   s   p  -> output side")
for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 1), (5, 1, 0), (5, 1, 2), (5, 2, 2), (7, 1, 3), (7, 2, 3)]:
    print(f"                {k:>3} {s:>3} {p:>3}  -> {out_size(28, k, s, p):>3}")

# the three examples of the lesson
assert out_size(28, 5, 1, 0) == 24 and out_size(28, 3, 1, 1) == 28 and out_size(32, 3, 2, 0) == 15
print("lesson examples: 28,k5,s1,p0 ->", out_size(28, 5, 1, 0), "| 28,k3,s1,p1 ->", out_size(28, 3, 1, 1), "| 32,k3,s2,p0 ->", out_size(32, 3, 2, 0))
print("floor in action: n=8, k=3, s=2, p=0 ->", out_size(8, 3, 2, 0), "(the last column is never used)")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
input 28 x 28:    k   s   p  -> output side
                  3   1   0  ->  26
                  3   1   1  ->  28
                  3   2   1  ->  14
                  5   1   0  ->  24
                  5   1   2  ->  28
                  5   2   2  ->  14
                  7   1   3  ->  28
                  7   2   3  ->  14
lesson examples: 28,k5,s1,p0 -> 24 | 28,k3,s1,p1 -> 28 | 32,k3,s2,p0 -> 15
floor in action: n=8, k=3, s=2, p=0 -> 3 (the last column is never used)
```

```python
rng = np.random.RandomState(0)
n_tested = 0
for n in (7, 8, 12, 28):
    pic = rng.normal(size=(n, n))
    for k, s, p in [(2, 1, 0), (3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1), (5, 1, 2), (5, 2, 0), (4, 2, 1)]:
        real = conv2d_np(pic, rng.normal(size=(k, k)), stride=s, pad=p).shape
        assert real == (out_size(n, k, s, p),) * 2, (n, k, s, p, real)
        n_tested += 1
print(f"the formula matched conv2d_np in all {n_tested} combinations")
RESULTS["p2_formula_tested"] = n_tested
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
the formula matched conv2d_np in all 32 combinations
```

   You should see the table: a 3 × 3 kernel gives 26 with no padding and 28 with padding 1; with stride 2 and padding 1 it gives **14**; a 5 × 5 kernel without padding gives **24** (the answer to *Imagine first*: the stencil can take 24 steps along a side of 28, because its right edge must stay inside, so it can start at columns 0 to 23); 5 × 5 with padding 2 gives 28 again; 7 × 7 with padding 3 keeps 28 and with stride 2 gives 14. The three lesson examples print 24, 28 and 15. The floor matters in the last line: for n = 8, k = 3, s = 2 the stencil lands at columns 0-2, 2-4 and 4-6 and so the output is **3**; column 7 is never used. The second block prints `the formula matched conv2d_np in all 32 combinations`: the asserts would stop the cell on the first mismatch. A pattern worth remembering: with stride 1, the choice `p = (k - 1) / 2` (for odd k) keeps the size, "same" padding.

   Tips and common mistakes:

   - Use integer division `//`, not `/`, or you get a float and, later, an error when you build a layer.
   - If the result is 0 or negative, the kernel is bigger than the padded input. That is the classic way to "shrink the picture to nothing" in a deep stack.
   - Stride and padding change the *size* of the output; they do not change the number of parameters of the layer.

   Try also (optional):

   - Find, by trial in `out_size`, a stride that turns a 28-pixel input with `k=3, p=1` into 7 pixels. (Answer: stride 4.)
   - Chain the function: 28 -> conv5 (p=2) -> pool2 -> conv5 -> pool2. What is the final side? (Answer: 5.)

   *Exercise 2.2:* use `out_size` to compute the output side for (a) a 224-pixel input with a 7 × 7 kernel, stride 2, padding 3 (the first layer of a famous large network), stored in `size_224`, and (b) a 32-pixel input with a 3 × 3 kernel, stride 2, padding 0, stored in `size_32`. Why it matters: output size is the number you need for the Linear layer after the convolutions (Part 5).

```python
size_224 = None   # 224, k=7, s=2, p=3
size_32 = None    # 32, k=3, s=2, p=0
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if size_224 is None or size_32 is None:
    print("Fill in both sizes above, then run this cell again.")
else:
    check("size_224 is 112", size_224 == 112, "(224 + 2*3 - 7) // 2 + 1")
    check("size_32 is 15", size_32 == 15, "(32 + 0 - 3) // 2 + 1 = 29 // 2 + 1 = 15")
    RESULTS["p2_sizes"] = [size_224, size_32]
```

3. **Step 2.3. Stride and padding in action.** Now see stride and padding on an actual picture. Take a 7 × 7 picture of all ones and a 3 × 3 *averaging* kernel (every weight is 1/9). On a perfectly flat picture the average of any complete 3 × 3 window is 1, so every output number should be 1, except where the window hangs over the border and meets the zeros that padding adds. The first block runs the four combinations of stride 1 or 2 and padding 0 or 1 and prints the shapes. The second block prints the output with padding 1 so that you can see the border effect, and draws all four outputs as small pictures.

```python
ones = np.ones((7, 7))
box = np.ones((3, 3)) / 9                    # averages the 9 numbers under it
outs = {}
for s in (1, 2):
    for p in (0, 1):
        outs[(s, p)] = conv2d_np(ones, box, stride=s, pad=p)
        print(f"stride {s}, padding {p}: output shape {outs[(s, p)].shape}   formula says {out_size(7, 3, s, p)}")
print("\noutput with stride 1 and padding 1 (rounded to two decimals):")
pm(outs[(1, 1)], 2)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
stride 1, padding 0: output shape (5, 5)   formula says 5
stride 1, padding 1: output shape (7, 7)   formula says 7
stride 2, padding 0: output shape (3, 3)   formula says 3
stride 2, padding 1: output shape (4, 4)   formula says 4

output with stride 1 and padding 1 (rounded to two decimals):
   0.44     0.67     0.67     0.67     0.67     0.67     0.44
   0.67     1.00     1.00     1.00     1.00     1.00     0.67
   0.67     1.00     1.00     1.00     1.00     1.00     0.67
   0.67     1.00     1.00     1.00     1.00     1.00     0.67
   0.67     1.00     1.00     1.00     1.00     1.00     0.67
   0.67     1.00     1.00     1.00     1.00     1.00     0.67
   0.44     0.67     0.67     0.67     0.67     0.67     0.44
```

```python
fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
for ax, ((s, p), o) in zip(axes, outs.items()):
    ax.imshow(o, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"stride {s}, pad {p}\n{o.shape[0]} x {o.shape[1]}", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
o11 = outs[(1, 1)]
print("corner value:", round(o11[0, 0], 3), "(4 of 9 cells are real) | edge value:", round(o11[0, 3], 3), "(6 of 9) | centre:", round(o11[3, 3], 3))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
corner value: 0.444 (4 of 9 cells are real) | edge value: 0.667 (6 of 9) | centre: 1.0
```

   You should see the shapes (5, 5) for stride 1 / padding 0, (7, 7) for stride 1 / padding 1, (3, 3) for stride 2 / padding 0 and (4, 4) for stride 2 / padding 1: each agrees with the formula. With padding 1 the output has the **same size as the input** (7 × 7), but its values are not all 1: the centre is 1.00, the edges 0.67 (six of the nine cells are real picture, three are padded zeros: 6/9) and the four corners only 0.44 (4/9). So padding preserves the size at a price: the border of the output is a blend of picture and zeros. In the small pictures you see a bright square with a darker frame. With stride 2 you see far fewer positions (the output is about half as wide) and each row of the output skips every other window.

   Tips and common mistakes:

   - `conv2d_np(..., pad=1)` uses `np.pad` with zeros. Other kinds of padding (copying the border) exist in libraries, but zeros is the default.
   - Stride 2 is a cheap way to halve the size without a pooling layer. Pooling (Part 3) is the other way.
   - Border effects matter for real pictures too: objects touching the edge are seen by fewer windows.

   Try also (optional):

   - Use `pad=2` with the 3 × 3 kernel on the 7 × 7 picture. What is the output size, and what values appear in the outermost ring?

   *Exercise 2.3:* how much padding keeps a 7 × 7 picture at 7 × 7 with a **5 × 5** kernel and stride 1? Store it in `pad_k5`. Then do the same for a **7 × 7** kernel and store it in `pad_k7`. Check one of them by running `conv2d_np` on `ones` and printing the shape. Why it matters: "same" padding is how you stack many convolutions without the picture shrinking away.

```python
pad_k5 = None   # padding that keeps 7x7 with a 5x5 kernel, stride 1
pad_k7 = None   # padding that keeps 7x7 with a 7x7 kernel, stride 1
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if pad_k5 is None or pad_k7 is None:
    print("Fill in both paddings above, then run this cell again.")
else:
    s5 = conv2d_np(ones, np.ones((5, 5)), 1, pad_k5).shape
    s7 = conv2d_np(ones, np.ones((7, 7)), 1, pad_k7).shape
    check("pad_k5 keeps the size (output is 7 x 7)", s5 == (7, 7), f"With pad_k5 = {pad_k5} the output is {s5}; use p = (k - 1) / 2.")
    check("pad_k7 keeps the size (output is 7 x 7)", s7 == (7, 7), f"With pad_k7 = {pad_k7} the output is {s7}; use p = (k - 1) / 2.")
    RESULTS["p2_same_padding"] = [pad_k5, pad_k7]
```

4. **Step 2.4. Parameters and channels.** Real pictures have several channels (colour pictures have three), and a convolution layer has several *filters* (kernels), each of which produces one output channel. Each filter has to look at **all** input channels, so it has `k · k · C_in` weights, plus one bias. A layer with `C_out` filters therefore has `k · k · C_in · C_out + C_out` parameters. In this step you compute that formula for three layers by hand, build the matching `nn.Conv2d` layers, compare with `count_params`, and look at the shape of the weight tensor, which has four numbers: `(C_out, C_in, k, k)`.

```python
specs = [("conv1", 1, 6, 5), ("conv2", 6, 16, 5), ("conv3", 16, 32, 3)]    # (name, C_in, C_out, k)
for name, cin, cout, k in specs:
    layer = nn.Conv2d(cin, cout, k)
    by_hand = k * k * cin * cout + cout
    print(f"{name}: {k}*{k}*{cin}*{cout} + {cout} = {by_hand:>5} | count_params = {count_params(layer):>5} | weight {tuple(layer.weight.shape)} | bias {tuple(layer.bias.shape)}")
    assert count_params(layer) == by_hand
RESULTS["p2_conv_params"] = [k * k * cin * cout + cout for _, cin, cout, k in specs]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
conv1: 5*5*1*6 + 6 =   156 | count_params =   156 | weight (6, 1, 5, 5) | bias (6,)
conv2: 5*5*6*16 + 16 =  2416 | count_params =  2416 | weight (16, 6, 5, 5) | bias (16,)
conv3: 3*3*16*32 + 32 =  4640 | count_params =  4640 | weight (32, 16, 3, 3) | bias (32,)
```

   You should see `conv1: 5*5*1*6 + 6 = 156`, `conv2: 5*5*6*16 + 16 = 2416` and `conv3: 3*3*16*32 + 32 = 4640`, each matching `count_params`. The weight shapes are `(6, 1, 5, 5)`, `(16, 6, 5, 5)` and `(32, 16, 3, 3)`: the first number is the number of filters (output channels), the second is the number of input channels each filter must cover, and the last two are the height and width of the stencil. The bias has one number per filter: `(6,)`, `(16,)`, `(32,)`. The first two layers, 156 and 2,416 parameters, are exactly the two convolution layers of the LeNet-style network in the lesson. Notice that none of the numbers contains the picture's height or width.

   Tips and common mistakes:

   - If a filter's weight tensor has `C_in` as its second number, it means every filter sees every input channel. A colour filter is a stack of three 2-D stencils added together.
   - `nn.Linear` stores its weight as `(out, in)` and `nn.Conv2d` as `(out, in, kH, kW)`: in both, the output count comes first.
   - `nn.Conv2d(cin, cout, k)` has `bias=True` by default. Setting `bias=False` removes the `cout` term.

   Try also (optional):

   - Print `conv1.weight[0, 0]` for a fresh `nn.Conv2d(1, 6, 5)` to see the 25 random numbers of one filter.
   - How many parameters does a 1 × 1 convolution from 64 to 32 channels have? (Answer: 2080.)

   *Exercise 2.4:* a network for colour photos starts with `nn.Conv2d(3, 64, 7)`. Compute its number of parameters by hand and store it in `p_conv7`, and write the shape of its weight tensor as a tuple in `w_shape`. Why it matters: the weight shape tells you at a glance what a layer connects, and mistakes in it cause the typical channel errors in Step 5.5.

```python
p_conv7 = None   # parameters of nn.Conv2d(3, 64, 7)
w_shape = None   # shape of its weight tensor, as a tuple
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if p_conv7 is None or w_shape is None:
    print("Fill in both values above, then run this cell again.")
else:
    real = nn.Conv2d(3, 64, 7)
    check("p_conv7 is 9472", p_conv7 == count_params(real) == 9472, "7*7*3*64 weights plus 64 biases.")
    check("w_shape is (64, 3, 7, 7)", tuple(w_shape) == tuple(real.weight.shape), "Order: (filters, input channels, height, width).")
    RESULTS["p2_conv7"] = [p_conv7, tuple(w_shape)]
```

5. **Step 2.5. Local connectivity and weight sharing, counted.** Why are the numbers so small? Two ideas. **Local connectivity**: each output looks at a small patch, not at the whole picture. **Weight sharing**: all positions use the same kernel. To feel the saving, compare a dense layer that produces an output picture of the same size as a 3 × 3 convolution on an n × n picture (each of the n² input pixels connected to each of the n² output pixels, so (n²)² weights) with the 9 weights of the convolution. The block fills a table for n = 8, 16, 32, 64 and 128, checks one entry against a real `nn.Linear`, and plots both curves on a logarithmic axis.

```python
sizes = [8, 16, 32, 64, 128]
dense_w = [(n * n) ** 2 for n in sizes]         # every input pixel to every output pixel
conv_w = [9 for n in sizes]                     # one shared 3 x 3 kernel
print(f"{'n':>5} {'dense weights':>15} {'conv weights':>13} {'ratio':>14}")
for n, d, c in zip(sizes, dense_w, conv_w):
    print(f"{n:>5} {d:>15,} {c:>13} {d // c:>14,}")
assert nn.Linear(8 * 8, 8 * 8, bias=False).weight.numel() == dense_w[0]

plt.figure(figsize=(5.5, 3.4))
plt.semilogy(sizes, dense_w, "o-", c="tab:red", label="dense layer, same output size")
plt.semilogy(sizes, conv_w, "s-", c="tab:blue", label="3 x 3 convolution (9 weights)")
plt.xlabel("picture side n"); plt.ylabel("weights (log scale)"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()
RESULTS["p2_dense_vs_conv_128"] = [dense_w[-1], 9]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
    n   dense weights  conv weights          ratio
    8           4,096             9            455
   16          65,536             9          7,281
   32       1,048,576             9        116,508
   64      16,777,216             9      1,864,135
  128     268,435,456             9     29,826,161
```

   You should see the table: for n = 8 the dense layer needs **4,096** weights and the convolution 9; for n = 16, 65,536; for n = 32, 1,048,576; for n = 64, 16,777,216; for n = 128 it needs **268,435,456** weights against the convolution's unchanged 9. The ratio goes from 455 at n = 8 to about 29,826,161 at n = 128. On the log plot the red line climbs steeply (every doubling of the side multiplies the dense count by 16), while the blue line stays flat at 9: the convolution's cost does not depend on the picture size at all. The assert passes silently: an `nn.Linear` from 64 to 64 values really has 4,096 weights.

   Tips and common mistakes:

   - This compares layers that give an output picture the same size as the input. A dense layer to a *few* units is cheaper, but then the network loses the picture structure early.
   - A convolution layer with many filters has `9 * C_in * C_out` weights, still far from `(n * n) ** 2`.
   - Fewer parameters also means less data is needed to find good values, and less memory.

   Try also (optional):

   - Add `n = 256` to the list. At what point would the dense weights not fit in the memory of a laptop (about 4 bytes per weight)?

   *Exercise 2.5:* for a picture with side n = 256, compute the number of weights of the dense layer with the same output size, `dense_256 = (256 * 256) ** 2`, and the whole-number ratio to the 9 weights of the convolution, `ratio_256`. Why it matters: it makes concrete why a CNN is possible on pictures that a dense network cannot even store.

```python
dense_256 = None   # weights of a dense layer from 256*256 pixels to 256*256 pixels
ratio_256 = None   # dense_256 divided by 9 (whole number)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if dense_256 is None or ratio_256 is None:
    print("Fill in both values above, then run this cell again.")
else:
    check("dense_256 is 4,294,967,296", dense_256 == 4294967296, "(256 * 256) squared = 65536 squared.")
    check("ratio_256 is dense_256 // 9", ratio_256 == dense_256 // 9, "Whole-number division by 9.")
    RESULTS["p2_dense_256"] = dense_256
```

6. **Step 2.6. Cross-correlation versus convolution.** Textbooks on signals define convolution with a **flipped** kernel (turned by 180 degrees: reversed in both directions). Deep-learning libraries, and the helper `conv2d_np`, do not flip: strictly that is *cross-correlation*. Does it matter? In this step you apply a kernel that is not symmetric and its flipped copy to the lesson's `x`, and compare. Then you test a kernel that is its own flip.

```python
Kn = np.array([ [1, 2], [0, -1] ], dtype=float)           # not symmetric
Kf = np.flip(Kn)                                           # np.flip with no axis flips BOTH directions (turn by 180 degrees)
print("kernel:"); pm(Kn); print("flipped kernel:"); pm(Kf)
y_cc, y_flip = conv2d_np(x, Kn), conv2d_np(x, Kf)
print("\nwithout flipping (what deep learning does):"); pm(y_cc)
print("with the flipped kernel (textbook convolution):"); pm(y_flip)
print("same result?", np.allclose(y_cc, y_flip))
Ks = np.array([ [1, 2], [2, 1] ], dtype=float)             # a kernel that is its own flip
print("symmetric kernel unchanged by flip:", np.array_equal(np.flip(Ks), Ks), "| same result:", np.allclose(conv2d_np(x, Ks), conv2d_np(x, np.flip(Ks))))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
kernel:
    1      2
    0     -1
flipped kernel:
   -1      0
    2      1

without flipping (what deep learning does):
    4     -1      1
    1      7      5
    4      0     -2
with the flipped kernel (textbook convolution):
    0      3      7
    5      1     -3
    0      0      4
same result? False
symmetric kernel unchanged by flip: True | same result: True
```

   You should see two different 3 × 3 tables: the unflipped result has the rows 4 -1 1 / 1 7 5 / 4 0 -2 and the flipped result has the rows 0 3 7 / 5 1 -3 / 0 0 4, so for this kernel they are **not the same**. For the symmetric kernel `Ks`, flipping changes nothing and the results are identical. The sentence to remember: *libraries do not flip, because the kernel values are learned. A flipped kernel is just another kernel, so a network that "should have used the flipped version" simply learns the flipped numbers, and nothing is lost.* The flip matters only if you copy a hand-made kernel (like the Sobel filters of Part 3) from a book into a library, or compare with another tool: then you must know which convention the tool uses.

   Tips and common mistakes:

   - `np.flip(K)` flips both axes. `np.flip(K, axis=0)` flips only up-down, `np.flip(K, axis=1)` only left-right.
   - The word "convolution" in this course always means the deep-learning version: no flip.
   - For the backward pass (Part 4) the flip comes back, but as a *consequence* of the chain rule, not as a definition.

   Try also (optional):

   - Show that flipping twice gives back the original kernel: `np.array_equal(np.flip(np.flip(Kn)), Kn)`.
   - Try the Sobel x kernel from Part 3 and its flip: how does the result change? (The sign flips.)

   *Exercise 2.6:* design a 3 × 3 kernel `K_sym` that is **unchanged by the flip** (and not all zeros), store it, and let the Check cell confirm that `conv2d_np(x, K_sym)` equals `conv2d_np(x, np.flip(K_sym))` for the lesson's `x`. Hint: the kernel must read the same forwards and backwards in both directions, like a palindrome. Why it matters: it shows exactly when the flip is harmless.

```python
K_sym = None   # a 3 x 3 numpy array that equals its own flip
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if K_sym is None:
    print("Fill in K_sym above (a 3 x 3 numpy array), then run this cell again.")
else:
    K_sym = np.asarray(K_sym, dtype=float)
    check("K_sym is 3 x 3 and not all zeros", K_sym.shape == (3, 3) and np.abs(K_sym).sum() > 0, "Use np.array with three rows of three numbers.")
    check("K_sym equals its own flip", np.array_equal(K_sym, np.flip(K_sym)), "K[0,0] = K[2,2], K[0,1] = K[2,1], K[0,2] = K[2,0], K[1,0] = K[1,2].")
    check("the two convolutions agree", np.allclose(conv2d_np(x, K_sym), conv2d_np(x, np.flip(K_sym))), "If the kernel equals its flip, the results are equal by definition.")
    RESULTS["p2_k_sym"] = K_sym.tolist()
```

### Part 3. Filters, feature maps and pooling (25 min)

*Lesson sections: "Convolution" and "Stacking".* In Part 2 the stencil's numbers were given to you. In a real CNN they are *learned*, but it helps enormously to see first what a hand-made stencil does. A kernel is a **detector**: it gives a big number where the picture under it looks like the pattern written in its weights, and a small or negative number elsewhere. The grid of those numbers is a **feature map**. In this part you apply classic filters to your shapes, design a detector of your own, see what ReLU and a bias do to a feature map, shrink maps with pooling and measure how pooling tolerates a small shift, and compute how much of the picture a unit in a deep layer can see (its receptive field).

**Imagine first.** Take a picture of a hollow **square** (an outline, one pixel thick) on a blank background and slide a **vertical-edge** filter over it. A vertical-edge filter responds where the brightness changes from left to right. Without running anything, describe in a sentence or two where on the square the output will be large and positive, where it will be large and negative, and where it will be zero. Write your sketch in words in the cell below; you can compare it with the picture in Step 3.1 and Step 3.3.

```python
# Imagine first: a vertical-edge filter on the picture of a hollow square. Where is the output large, negative, zero?
my_sketch = ""   # write one or two sentences between the quotes
```

1. **Step 3.1. Hand-made filters on the five shapes.** Four famous 3 × 3 kernels: the two **Sobel** edge filters (`SOBEL_X` responds to vertical edges, where the brightness changes from left to right; `SOBEL_Y` is its transpose and responds to horizontal edges), a **blur** that averages the nine pixels, and a **sharpen** filter that boosts the centre pixel against its four neighbours. The block takes one picture of each of the five shape classes, applies the four filters with `conv2d_np` and padding 1 (so each feature map is 16 × 16), draws a grid of pictures (one row per class: the original, then the four feature maps; red is positive, blue negative) and prints the strongest absolute response for every class and filter.

```python
SOBEL_X = np.array([ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ], dtype=float)    # vertical edges
SOBEL_Y = SOBEL_X.T                                                          # horizontal edges
BLUR = np.ones((3, 3)) / 9
SHARPEN = np.array([ [0, -1, 0], [-1, 5, -1], [0, -1, 0] ], dtype=float)
FILTERS = {"Sobel x (vertical edges)": SOBEL_X, "Sobel y (horizontal edges)": SOBEL_Y, "blur": BLUR, "sharpen": SHARPEN}

Xs, ys = make_shapes(100, 16, noise=0.05, seed=3)
firsts = [int(np.where(ys == k)[0][0]) for k in range(5)]        # index of the first picture of each class
fig, axes = plt.subplots(5, 5, figsize=(8.5, 8.5))
strongest = np.zeros((5, 4))
for r, idx in enumerate(firsts):
    pic = Xs[idx, 0]
    axes[r, 0].imshow(pic, cmap="gray_r", vmin=0, vmax=1); axes[r, 0].set_ylabel(SHAPE_NAMES[r], fontsize=8)
    for c, (name, k) in enumerate(FILTERS.items(), start=1):
        fmap = conv2d_np(pic, k, stride=1, pad=1)
        strongest[r, c - 1] = np.abs(fmap).max()
        axes[r, c].imshow(fmap, cmap="RdBu_r", vmin=-4, vmax=4)
        if r == 0: axes[r, c].set_title(name, fontsize=7)
for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_title("picture", fontsize=7)
plt.tight_layout(); plt.show()

print(f"{'strongest |response|':>22}" + "".join(f"{n.split(' (')[0]:>14}" for n in FILTERS))
for r in range(5):
    print(f"{SHAPE_NAMES[r]:>22}" + "".join(f"{v:>14.2f}" for v in strongest[r]))
RESULTS["p3_strongest_responses"] = strongest.round(2).tolist()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
  strongest |response|       Sobel x       Sobel y          blur       sharpen
        horizontal bar          2.16          4.33          0.35          4.35
          vertical bar          4.17          2.01          0.35          4.17
                 cross          4.31          4.18          0.57          3.94
                square          4.41          4.33          0.59          3.59
              diagonal          2.26          2.30          0.37          5.48
```

   You should see a 5 × 5 grid. In the **Sobel x** column the vertical bar lights up as a pair of lines, red on its left and blue on its right (the brightness rises to the bar and falls away again), while the horizontal bar gives only weak responses at its two ends. The **Sobel y** column does the opposite: the horizontal bar is outlined and the vertical bar is almost silent. The square gives both: its left and right sides in Sobel x, its top and bottom in Sobel y. The **blur** column is the original picture, smeared and much weaker; the **sharpen** column keeps the shapes crisp with a strong centre. The table agrees: the strongest Sobel x response is 2.16 for the horizontal bar (only the ends) but 4.17 for the vertical bar; Sobel y is 4.33 for the horizontal bar and 2.01 for the vertical bar; blur never exceeds 0.59. What each filter responds to: Sobel x to vertical edges, Sobel y to horizontal edges, blur to nothing in particular (it removes detail), sharpen to bright centre pixels against a darker neighbourhood.

   Tips and common mistakes:

   - A positive and a negative response are both information: the sign tells you on which side of the edge the bright pixels are. That is why the picture uses a diverging colour map (red positive, blue negative).
   - The Sobel kernels sum to zero. A kernel whose weights add to zero gives zero on a flat region, so it reacts to *changes* only; the blur sums to 1 and keeps flat regions as they are.
   - `conv2d_np` with `pad=1` keeps the 16 × 16 size; without padding you would get 14 × 14 maps.

   Try also (optional):

   - Apply the Sobel x filter to the diagonal and the cross. Which parts of them light up?
   - Compute the Sobel magnitude `np.sqrt(fx**2 + fy**2)` for the square: all four sides appear together.

   *Exercise 3.1:* design a 3 × 3 kernel `K_diag` that responds more strongly to the **diagonal** shape (the pictures in `firsts[4]`) than to the horizontal bar (`firsts[0]`) and to the vertical bar (`firsts[1]`). Hint: put positive weights along the main diagonal of the 3 × 3 window (top-left to bottom-right). The Check cell compares the strongest response on each of the three pictures. Why it matters: this is what training does automatically: it finds kernels that respond to useful patterns.

```python
K_diag = None   # a 3 x 3 numpy array
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if K_diag is None:
    print("Fill in K_diag above (a 3 x 3 numpy array), then run this cell again.")
else:
    K_diag = np.asarray(K_diag, dtype=float)
    resp = [conv2d_np(Xs[firsts[k], 0], K_diag, 1, 1).max() for k in (4, 0, 1)]
    print("strongest response: diagonal %.2f | horizontal bar %.2f | vertical bar %.2f" % tuple(resp))
    check("K_diag is 3 x 3", K_diag.shape == (3, 3), "Use np.array with three rows of three numbers.")
    check("responds more to the diagonal than to both bars", resp[0] > resp[1] and resp[0] > resp[2], "Put positive numbers on the main diagonal and zero (or negative) elsewhere.")
    RESULTS["p3_k_diag"] = K_diag.tolist()
```

2. **Step 3.2. Design your own detector and use it as a classifier.** Now use a filter for a job. You will build a detector for a **vertical bar** and turn the whole feature map into **one number per picture**: the largest value of the feature map after ReLU ("is there a vertical bar anywhere?"). The kernel puts weight 2 on the middle column of the 3 × 3 window and weight -1 on the two side columns, so it likes a bright column between two dark ones. Then you take 400 new noisy pictures, keep only the vertical bars and the horizontal bars, compute the one number for each, choose a threshold halfway between the two class averages and report the accuracy of that one-number classifier. A first sanity check compares `conv2d_np` with PyTorch's `F.conv2d`, which is the fast version that you use for many pictures at once.

```python
K_vbar = np.array([ [-1, 2, -1], [-1, 2, -1], [-1, 2, -1] ], dtype=float)    # bright column between dark ones

def feature_scores(X, K):
    """For each picture: the largest value of the ReLU'd feature map of kernel K (padding 1). Fast, via F.conv2d."""
    kt = torch.as_tensor(K, dtype=torch.float32).reshape(1, 1, *K.shape)
    fmap = F.relu(F.conv2d(torch.as_tensor(X), kt, padding=1))
    return fmap.amax(dim=(1, 2, 3)).numpy()

def separation(K, X, y, pos, neg):
    """Threshold halfway between the two class means of the score. Returns accuracy, threshold and the scores of both classes."""
    s = feature_scores(X, K); a, b = s[y == pos], s[y == neg]
    th = (a.mean() + b.mean()) / 2
    acc = ((a > th).sum() + (b <= th).sum()) / (len(a) + len(b))
    return float(acc), float(th), a, b

# the fast version agrees with the loop version
Xd, yd = make_shapes(400, 16, noise=0.15, seed=5)
slow = np.maximum(conv2d_np(Xd[0, 0], K_vbar, 1, 1), 0).max()
assert abs(slow - feature_scores(Xd[:1], K_vbar)[0]) < 1e-4
acc_v, th_v, sv, sh = separation(K_vbar, Xd, yd, pos=1, neg=0)       # label 1 = vertical bar, label 0 = horizontal bar
print(f"vertical bars:   score from {sv.min():.2f} to {sv.max():.2f}, mean {sv.mean():.2f}  ({len(sv)} pictures)")
print(f"horizontal bars: score from {sh.min():.2f} to {sh.max():.2f}, mean {sh.mean():.2f}  ({len(sh)} pictures)")
print(f"threshold {th_v:.2f} -> accuracy {acc_v:.3f}")
RESULTS["p3_vbar_detector_acc"] = acc_v
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
vertical bars:   score from 5.55 to 7.89, mean 6.65  (80 pictures)
horizontal bars: score from 1.36 to 2.83, mean 1.97  (80 pictures)
threshold 4.31 -> accuracy 1.000
```

```python
plt.figure(figsize=(6, 3))
plt.hist(sh, bins=20, alpha=0.7, label="horizontal bars", color="tab:orange")
plt.hist(sv, bins=20, alpha=0.7, label="vertical bars", color="tab:blue")
plt.axvline(th_v, c="k", ls="--"); plt.xlabel("largest ReLU'd response of the vertical-bar kernel"); plt.legend(); plt.tight_layout(); plt.show()
```

   You should see that the two classes separate completely: vertical bars score between 5.55 and 7.89 (mean 6.65) while horizontal bars score only between 1.36 and 2.83 (mean 1.97); the threshold 4.31 gives an accuracy of **1.000**. The histogram shows two separate hills with a gap and the dashed line in the gap. One 3 × 3 kernel, with nine numbers chosen by thinking, plus "take the maximum", is already a classifier for this pair of classes, and it works wherever the bar sits in the picture, because the kernel is slid over every position. Compare with the dense network of Part 1, which needed 16,773 parameters and failed when the shape moved. (This works for two classes that differ by exactly this feature; separating all five classes needs several filters, which a CNN learns.)

   Tips and common mistakes:

   - Taking the maximum over the feature map throws away *where* the bar is and keeps *whether* it is there. This is the idea of global pooling.
   - The kernel's weights sum to zero (-3 + 6 - 3 per row), so a flat region gives zero, and a horizontal bar, which fills the three columns equally, gives zero in its middle and only responds at its ends.
   - A threshold halfway between the class means is the simplest choice; with a gap between the classes, any value in the gap works.

   Try also (optional):

   - Put noise 0.5 into `make_shapes` and see at which noise level the accuracy begins to fall.
   - Use `SOBEL_X` as the kernel. Does it separate the two classes as well as your kernel?

   *Exercise 3.2:* make a kernel `K_h` that detects a **horizontal** bar (the transposed idea of `K_vbar`: a bright row between two dark ones) and let the Check cell use it to separate horizontal bars (label 0) from vertical bars (label 1) on the same pictures. Store the kernel in `K_h`. Why it matters: two kernels, one per direction, are the beginning of a feature set; a CNN layer is just many such detectors working at once.

```python
K_h = None   # a 3 x 3 numpy array that detects a horizontal bar
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if K_h is None:
    print("Fill in K_h above (a 3 x 3 numpy array), then run this cell again.")
else:
    K_h = np.asarray(K_h, dtype=float)
    acc_h, th_h, _, _ = separation(K_h, Xd, yd, pos=0, neg=1)
    check("K_h is 3 x 3", K_h.shape == (3, 3), "Three rows of three numbers.")
    check(f"horizontal-bar detector accuracy {acc_h:.3f} is above 0.95", acc_h > 0.95, "Put 2 on the middle ROW and -1 on the top and bottom rows (the transpose of K_vbar).")
    RESULTS["p3_hbar_detector_acc"] = acc_h
```

3. **Step 3.3. ReLU and bias: what survives.** A convolution layer does not stop at the sum: it adds a **bias** `b` and applies an activation, usually **ReLU**, which sets negative numbers to zero: `relu(z) = max(0, z)`. In terms of the inspector: after summing, the inspector adds a personal "strictness" number and then writes down only positive scores. ReLU keeps the cells that respond positively; a negative bias raises the bar so that only strong responses survive. The block applies `SOBEL_X` to a noise-free square outline, then draws the raw map, the ReLU'd map and the ReLU'd map after subtracting 2, and counts how many cells remain active (greater than zero) for several values of the subtracted bias.

```python
sq_X, sq_y = make_shapes(50, 16, noise=0.0, seed=3)
square = sq_X[int(np.where(sq_y == 3)[0][0]), 0]           # a clean square outline
raw = conv2d_np(square, SOBEL_X, stride=1, pad=1)

fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for ax, (title, m) in zip(axes, [("picture", square), ("Sobel x, raw", raw), ("after ReLU", np.maximum(raw, 0)), ("ReLU(raw - 2)", np.maximum(raw - 2, 0))]):
    ax.imshow(m, cmap="gray_r" if title == "picture" else "RdBu_r", vmin=None if title == "picture" else -4, vmax=None if title == "picture" else 4)
    ax.set_title(title, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()

print("values in the raw map:", sorted(set(raw.ravel().tolist())))
print("cells with positive / negative / zero response:", int((raw > 0).sum()), int((raw < 0).sum()), int((raw == 0).sum()))
for b in range(0, 6):
    print(f"subtract bias {b}: active cells after ReLU = {int((np.maximum(raw - b, 0) > 0).sum())}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
values in the raw map: [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
cells with positive / negative / zero response: 26 26 204
subtract bias 0: active cells after ReLU = 26
subtract bias 1: active cells after ReLU = 18
subtract bias 2: active cells after ReLU = 16
subtract bias 3: active cells after ReLU = 12
subtract bias 4: active cells after ReLU = 0
subtract bias 5: active cells after ReLU = 0
```

   You should see in the raw map a pattern of red and blue lines: red just left of each vertical side of the square (the response of the left edge, where the brightness rises towards the right) and blue to the right of each vertical side, with nearly nothing on the horizontal sides except at the corners. The raw map has **26 positive, 26 negative and 204 zero cells** (values from -4 to 4). After ReLU the 26 negative cells become 0 and only 26 cells remain active; this is how a feature map becomes *sparse*: most cells are silent. Subtracting a bias raises the threshold further: active cells fall from 26 (bias 0) to 18, 16 and 12 for biases 1, 2 and 3, and to 0 for bias 4 (and 5): the detector has become so fussy that nothing is left. That answers the sketch from *Imagine first*: big positive where the brightness rises from left to right (left sides), big negative where it falls (right sides), zero on flat areas and on the horizontal sides.

   Tips and common mistakes:

   - The ReLU'd map keeps only one sign. The two signs of a Sobel filter (a bright-to-dark edge versus a dark-to-bright edge) become two *different* detectors only if the network has two filters, one the negative of the other.
   - In a trained network the bias is learned like the weights. A strongly negative bias makes a filter fussy, a positive bias makes it eager.
   - The count of active cells is a quick measure of how selective a filter is.

   Try also (optional):

   - Plot `np.maximum(-raw, 0)`: this is what the *flipped* Sobel filter would show after ReLU.
   - Use a random 3 × 3 kernel instead of Sobel x. How many cells are active, with bias 0?

   *Exercise 3.3:* find the smallest whole-number `b_sparse` such that after subtracting it, ReLU leaves **fewer than 15** active cells in the feature map of the square (`np.maximum(raw - b_sparse, 0)`). Use the counts printed above, then confirm with code. Why it matters: it connects the bias to the idea of a threshold that decides whether a detector fires.

```python
b_sparse = None   # the smallest whole number so that fewer than 15 cells stay active
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if b_sparse is None:
    print("Fill in b_sparse above, then run this cell again.")
else:
    active = lambda b: int((np.maximum(raw - b, 0) > 0).sum())
    check(f"with bias {b_sparse} fewer than 15 cells are active ({active(b_sparse)})", active(b_sparse) < 15, "Look at the counts printed for subtract bias 0 to 5.")
    check("one step less is not enough", b_sparse == 0 or active(b_sparse - 1) >= 15, "You need the SMALLEST bias that works.")
    RESULTS["p3_b_sparse"] = b_sparse
```

4. **Step 3.4. Pooling by hand, and how pooling forgives a shift.** **Pooling** shrinks a feature map: it slides a small window (usually 2 × 2, with stride 2 so that windows do not overlap) and keeps one number per window, the **maximum** (max pooling) or the **average** (average pooling). It has no parameters. First you write `pool_np` and apply it to a 4 × 4 example, checking against `F.max_pool2d` and `F.avg_pool2d`. Then you test the claim that pooling makes a network less sensitive to small shifts: take 300 noise-free shapes that sit at random positions, make the ReLU'd Sobel x feature map, shift every input picture by **one pixel** to the right (`np.roll`), and measure how much the feature map changes and how much the max-pooled map changes. The size of a change is measured as the length of the difference divided by the length of the original map, so that 0 means unchanged and 1 means as large as the map itself.

```python
def pool_np(x, k=2, mode="max"):
    """Non-overlapping k x k pooling (stride k) of a 2-D array."""
    h, w = x.shape[0] // k, x.shape[1] // k
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            win = x[i * k:(i + 1) * k, j * k:(j + 1) * k]
            out[i, j] = win.max() if mode == "max" else win.mean()
    return out

p4 = np.array([ [1, 3, 2, 1], [4, 6, 5, 1], [7, 2, 0, 1], [3, 1, 8, 9] ], dtype=float)
print("input (4 x 4):"); pm(p4)
print("max pool 2x2:"); pm(pool_np(p4, 2, "max"), 2)
print("average pool 2x2:"); pm(pool_np(p4, 2, "avg"), 2)
t4 = torch.tensor(p4).reshape(1, 1, 4, 4)
assert np.allclose(F.max_pool2d(t4, 2).numpy()[0, 0], pool_np(p4, 2, "max"))
assert np.allclose(F.avg_pool2d(t4, 2).numpy()[0, 0], pool_np(p4, 2, "avg"))
print("pool_np agrees with F.max_pool2d and F.avg_pool2d")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
input (4 x 4):
    1      3      2      1
    4      6      5      1
    7      2      0      1
    3      1      8      9
max pool 2x2:
   6.00     5.00
   7.00     9.00
average pool 2x2:
   3.50     2.25
   3.25     4.50
pool_np agrees with F.max_pool2d and F.avg_pool2d
```

```python
Xn, yn = make_shapes(300, 16, noise=0.0, shift=2, seed=4)            # shapes at random positions, no noise
kx = torch.as_tensor(SOBEL_X, dtype=torch.float32).reshape(1, 1, 3, 3)
fm = F.relu(F.conv2d(torch.as_tensor(Xn), kx, padding=1))                                    # feature maps, 16 x 16
fm_shift = F.relu(F.conv2d(torch.as_tensor(np.roll(Xn, 1, axis=3).copy()), kx, padding=1))   # same pictures moved 1 pixel right

def rel_change(a, b):
    """Average over the pictures of |a - b| / |a| (0 = unchanged, 1 = a change as large as the map itself)."""
    return float(((a - b).norm(dim=(2, 3)) / a.norm(dim=(2, 3)).clamp_min(1e-9)).mean())

def n_changed(a, b):
    return float((a - b).abs().gt(1e-6).float().sum(dim=(1, 2, 3)).mean())

mp, mp_shift = F.max_pool2d(fm, 2), F.max_pool2d(fm_shift, 2)                                # 8 x 8
gp, gp_shift = fm.amax(dim=(2, 3)), fm_shift.amax(dim=(2, 3))                                # global max: one number
print(f"feature map 16x16:   {n_changed(fm, fm_shift):5.1f} of 256 values changed | relative change {rel_change(fm, fm_shift):.2f}")
print(f"max pooled 8x8:      {n_changed(mp, mp_shift):5.1f} of  64 values changed | relative change {rel_change(mp, mp_shift):.2f}")
print(f"global max (1 value): relative change {float(((gp - gp_shift).abs() / gp.clamp_min(1e-9)).mean()):.3f}")
RESULTS["p3_rel_change_fm_pool_global"] = [round(rel_change(fm, fm_shift), 2), round(rel_change(mp, mp_shift), 2)]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
feature map 16x16:    26.2 of 256 values changed | relative change 1.25
max pooled 8x8:        8.1 of  64 values changed | relative change 0.81
global max (1 value): relative change 0.000
```

   You should see for the 4 × 4 example the max pooled map with rows `6 5` and `7 9`, and the average pooled map with rows `3.50 2.25` and `3.25 4.50`; both agree with PyTorch. Each output number summarises one 2 × 2 window, so a 4 × 4 map becomes 2 × 2: the size is halved on each side and the number of values drops to a quarter. In the shift test, moving every picture one pixel changes **26.2 of the 256 values** of the feature map (a relative change of **1.25**), and **8.1 of the 64 values** of the max-pooled map, a relative change of **0.81**. So pooling reduces the relative effect of the shift (from 1.25 to 0.81) but does not remove it: when a bright cell crosses the border between two pooling windows, the output still changes. Note also that the *fraction* of changed values is not smaller after pooling (10 percent against 13 percent), because the pooled map has fewer values: the honest message is "less sensitive", not "immune". The last number, the global maximum, changes by **0.000**, that is, not at all: when you keep one number per map, a small shift does not matter. That is the trick that Step 3.2 used.

   Tips and common mistakes:

   - Max pooling keeps the strongest evidence in each window and discards its exact location; average pooling keeps the mean and blurs.
   - If the input side is odd, `F.max_pool2d(x, 2)` drops the last row and column (it floors), just like the output-size formula.
   - `np.roll` wraps pixels around the edge; the shapes here stay away from the edge, so nothing wraps.

   Try also (optional):

   - Change the shift to 2 pixels and compare the relative changes again.
   - Try `F.avg_pool2d` instead of max pooling in the shift test. Which is less sensitive?

   *Exercise 3.4:* compute the 2 × 2 **average** pooling of the 4 × 4 example `p4` by hand (each output is the mean of a 2 × 2 window) and store the result as a 2 × 2 NumPy array in `avg_pooled`. Do it first on paper, then verify with `pool_np`. Why it matters: average pooling is the other common choice (it spreads the gradient equally in the backward pass, Part 4).

```python
avg_pooled = None   # a 2 x 2 numpy array
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if avg_pooled is None:
    print("Fill in avg_pooled above (a 2 x 2 numpy array), then run this cell again.")
else:
    avg_pooled = np.asarray(avg_pooled, dtype=float)
    check("avg_pooled has shape (2, 2)", avg_pooled.shape == (2, 2), "A 4 x 4 map with 2 x 2 pooling gives 2 x 2.")
    check("avg_pooled has the right numbers", avg_pooled.shape == (2, 2) and np.allclose(avg_pooled, pool_np(p4, 2, "avg")), "Top-left window: (1 + 3 + 4 + 6) / 4 = 3.5.")
    RESULTS["p3_avg_pooled"] = avg_pooled.tolist()
```

5. **Step 3.5. The receptive field.** The **receptive field** of a unit is the patch of the *input picture* that can influence it. A unit in the first 3 × 3 layer sees 3 × 3 pixels. A unit in the next layer sees 3 × 3 *units of the first layer*, which together cover more of the picture. The rule: keep the receptive field `r` and the *jump* `j` (the distance in input pixels between neighbouring units). Start with `r = 1` and `j = 1`. For each layer with kernel `k` and stride `s`: `r = r + (k - 1) · j`, then `j = j · s`. The block implements it as `receptive_field(layers)` for a list of `(k, s)` pairs, prints the table for conv3, pool2, conv3, pool2, conv3 (pooling counts as a layer with kernel 2 and stride 2), and asserts the result.

```python
def receptive_field(layers, verbose=False):
    """layers = list of (kernel, stride). Returns the receptive field after each layer."""
    r, jump, history = 1, 1, []
    for k, s in layers:
        r = r + (k - 1) * jump
        jump = jump * s
        history.append(r)
        if verbose: print(f"  kernel {k}, stride {s}: receptive field {r}, jump {jump}")
    return history

net = [(3, 1), (2, 2), (3, 1), (2, 2), (3, 1)]       # conv3 -> pool2 -> conv3 -> pool2 -> conv3
print("layer by layer for conv3, pool2, conv3, pool2, conv3:")
hist_rf = receptive_field(net, verbose=True)
assert hist_rf == [3, 4, 8, 10, 18]
print("receptive fields:", hist_rf)
print("stacking only 3x3 convolutions (stride 1): layers needed to see 15 pixels ->", next(n for n in range(1, 20) if receptive_field([(3, 1)] * n)[-1] >= 15))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
layer by layer for conv3, pool2, conv3, pool2, conv3:
  kernel 3, stride 1: receptive field 3, jump 1
  kernel 2, stride 2: receptive field 4, jump 2
  kernel 3, stride 1: receptive field 8, jump 2
  kernel 2, stride 2: receptive field 10, jump 4
  kernel 3, stride 1: receptive field 18, jump 4
receptive fields: [3, 4, 8, 10, 18]
stacking only 3x3 convolutions (stride 1): layers needed to see 15 pixels -> 7
```

   You should see the table with receptive fields **3, 4, 8, 10, 18**: the first 3 × 3 layer sees 3 pixels; the pooling layer adds one more (4) and doubles the jump to 2; the next 3 × 3 layer reaches 8 because its neighbours are now 2 pixels apart (4 + 2·2); the second pooling gives 10 and a jump of 4; the last convolution sees 18 pixels (10 + 2·4). The assert passes silently. The last line shows that stacking plain 3 × 3 layers at stride 1 needs **7** layers to see 15 pixels (the field grows by 2 each time); pooling and stride make it grow much faster, because they increase the jump. This is why a CNN stacks layers: the first ones see tiny patches, and the deeper ones see larger and larger parts of the picture, without any single layer having to connect to the whole picture.

   Tips and common mistakes:

   - Pooling counts as a layer with its own kernel and stride in this formula.
   - Padding does not change the receptive field; it only keeps the map size.
   - The receptive field is the largest region that *can* influence a unit; the pixels in the middle of it matter most.

   Try also (optional):

   - Compute the field for the 5 × 5 first layer of LeNet followed by pooling: `receptive_field([ (5, 1), (2, 2) ])`.
   - Replace each pool2 by a stride-2 convolution of kernel 2. Does the field change?

   *Exercise 3.5:* compute the receptive field after each layer for conv5, pool2, conv5, pool2 (the front of the LeNet-style network in the lesson) with `receptive_field`, and store the final one in `rf_lenet`. Then compute how many layers of kernel 5, stride 1 you would need to see at least 29 pixels and store it in `n_layers_29`. Why it matters: it tells you whether the last convolution layer of a network can see the whole 28 × 28 digit.

```python
rf_lenet = None      # final receptive field of conv5, pool2, conv5, pool2
n_layers_29 = None   # how many stacked 5x5 stride-1 layers reach at least 29 pixels
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if rf_lenet is None or n_layers_29 is None:
    print("Fill in both values above, then run this cell again.")
else:
    check("rf_lenet is 16", rf_lenet == 16, "5, then 5 + 1 = 6, then 6 + 4*2 = 14, then 14 + 1*2 = 16.")
    check("n_layers_29 is 7", n_layers_29 == 7, "Each 5x5 stride-1 layer adds 4: 1 + 4n is at least 29.")
    RESULTS["p3_rf_lenet"] = rf_lenet
```

6. **Step 3.6. Imagine and write: what each layer might detect.** A CNN is a stack of detectors, and each layer builds on the one before. The block below prints, for the LeNet-style front end on a 28 × 28 picture, how big the feature map is after each layer (with your `out_size`) and how many input pixels each unit can see (with your `receptive_field`). Use the table as the ruler for your writing: a unit that sees only 5 pixels can detect an edge or a short stroke but not an eye; a unit that sees 14 pixels can detect parts. Then imagine a CNN looking at **faces** and write what the first, second and third layers might detect.

```python
stack = [("conv 5x5, pad 2", 5, 1, 2), ("pool 2x2", 2, 2, 0), ("conv 5x5", 5, 1, 0), ("pool 2x2", 2, 2, 0)]
side, rfs = 28, receptive_field([ (k, s) for _, k, s, _ in stack ])
print(f"{'layer':<18}{'map side':>9}{'unit sees':>11}")
print(f"{'input picture':<18}{side:>9}{1:>11}")
for (name, k, s, p), r in zip(stack, rfs):
    side = out_size(side, k, s, p)
    print(f"{name:<18}{side:>9}{str(r) + ' x ' + str(r):>11}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
layer              map side  unit sees
input picture            28          1
conv 5x5, pad 2          28      5 x 5
pool 2x2                 14      6 x 6
conv 5x5                 10    14 x 14
pool 2x2                  5    16 x 16
```

   You should see the table: the picture is 28; after the first convolution (padding 2) it is still 28 and each unit sees 5 × 5 pixels; after pooling the map is 14 and the field is 6; after the second convolution the map is 10 and each unit sees 14 × 14 pixels; after the last pooling the map is 5 × 5 and each unit sees 16 × 16 of the 28 × 28 digit. The map gets smaller while the field grows. For faces, a typical (not guaranteed!) story is: the first layers react to **edges and small blobs** of light and dark, the middle layers to **parts** (an eye, a nose, a mouth, an outline), and the last layers to **whole faces** or combinations of parts. The word to use is "often", not "always": nobody designs these detectors; they are what training discovers, and the stories we tell about them are partly interpretation. Part 6 looks at the learned filters of your own CNN to see which of these stories fits.

   Tips and common mistakes:

   - Early layers see small patches and are shared by almost every vision task; later layers specialise.
   - The numbers in the table assume the 28 × 28 LeNet-style example from the lesson; with your 16 × 16 shapes the map sizes differ, the receptive fields do not.

   Try also (optional):

   - Add a third convolution to `stack` (kernel 3, stride 1, padding 0) and read the new map size and field.

   *Exercise 3.6:* write **three short sentences** in the list `faces_notes`: what the first, the second and the third convolution layer of a CNN that looks at faces might detect, and why the later layer can detect something bigger (use the receptive-field idea). Use the word "often" at least once to remind yourself that this is a typical story, not a law.

```python
faces_notes = [
    "",   # layer 1: ...
    "",   # layer 2: ...
    "",   # layer 3: ...
]
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
notes = [n for n in faces_notes if n.strip()]
if len(notes) < 3:
    print("Write three sentences in faces_notes above, then run this cell again.")
else:
    check("three notes of at least 25 characters each", all(len(n) >= 25 for n in notes[:3]), "Write a full sentence for each layer.")
    check("you hedged with the word often (or typically)", any(w in " ".join(notes).lower() for w in ("often", "typically", "may", "might")), "Add a hedge: these detectors are often, not always, what a trained CNN finds.")
    RESULTS["p3_faces_notes"] = notes[:3]
```

### Part 4. The backward pass of a convolution, by hand (30 min)

*Lesson section: "Backward".* In Part 2 you computed what a convolution *does* to a picture. Here you compute how its output *depends* on its inputs: the gradients that training needs. You will write the backward pass of a convolution with plain loops on the lesson's 4x4 example, check it three ways (the lesson's numbers, finite differences, PyTorch autograd), see that it is itself a convolution, extend it to stride and padding, pass gradients through ReLU and max-pooling, and finally chain it all in a tiny CNN. ==Because the kernel is shared, the gradient of one kernel cell is a sum over every window, and the gradient of one input cell is a sum over every window that read it.==

**Imagine first.** Look at the lesson's 4x4 picture `x`, the 2x2 kernel `K` and the upstream gradient `G` (it says how much the loss grows when each of the 9 output cells grows by 1). Every output cell is a sum over a 2x2 window of the picture. Which input cell do you expect to receive the **largest gradient in absolute value**: a cell in the middle, which is read by many windows, or a cell at the edge, which is read by few? Write your guess as `(row, column)` in `guess_cell` (counting from 0). You test it in Step 4.3 and at the end of the part.

```python
guess_cell = None       # YOUR TURN: (row, column) of the input cell you expect to get the largest gradient, e.g. (1, 1)
print("Your guess:", guess_cell)
```

1. **Step 4.1. The lesson's example, with explicit loops.** The lesson's example has a 4x4 input `x`, a 2x2 kernel `K = [ [1, 0], [0, -1] ]` and the 3x3 output `y` you computed in Part 2. Suppose the layers after this one tell us how much each output cell matters: that is the **upstream gradient** `G = dL/dy`, a 3x3 matrix here. The backward pass must turn `G` into two things: `dL/dK` (to update the kernel) and `dL/dx` (to hand back to the layer before). The rule follows from the forward formula `y[i][j] = sum over a, b of K[a][b] * x[i+a][j+b]`: the kernel cell `K[a][b]` met input cell `(i+a, j+b)` in the window `(i, j)`, so `dL/dK[a][b]` adds up `G[i][j] * x[i+a][j+b]` over all windows; and the input cell `(i+a, j+b)` was read by window `(i, j)` through `K[a][b]`, so `dL/dx` collects `G[i][j] * K[a][b]` from every window that read it. Write both with four nested loops, so that you can see exactly what is added to what.

```python
# the lesson's example: a 4x4 picture x, a 2x2 kernel K, and the upstream gradient G.
# G[i][j] tells how much the loss L grows when the output cell y[i][j] grows by 1; it comes from the layers after this one.
x = np.array([ [1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2] ], dtype=float)
K = np.array([ [1, 0], [0, -1] ], dtype=float)
G = np.array([ [1, 0, -1], [2, 1, 0], [0, 1, 1] ], dtype=float)

y = conv2d_np(x, K)                      # the forward pass from Part 2
print("y =\n", y)
assert np.allclose(y, [ [0, -1, -1], [-1, 1, 3], [2, 0, -2] ]), "the lesson's y"

def conv_backward_kernel(x, G, kh, kw):
    """dL/dK. Kernel cell (a, b) met input cell (i + a, j + b) in every window (i, j), so add up G[i][j] times that input cell."""
    dK = np.zeros((kh, kw))
    for a in range(kh):
        for b in range(kw):
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    dK[a, b] += G[i, j] * x[i + a, j + b]
    return dK

def conv_backward_input(K, G, in_shape):
    """dL/dx. Output cell (i, j) read input cell (i + a, j + b) through kernel cell (a, b), so it sends back G[i][j] times K[a][b]."""
    dx = np.zeros(in_shape)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            for a in range(K.shape[0]):
                for b in range(K.shape[1]):
                    dx[i + a, j + b] += G[i, j] * K[a, b]
    return dx

dK = conv_backward_kernel(x, G, 2, 2)
dx = conv_backward_input(K, G, x.shape)
print("dL/dK =\n", dK)
print("dL/dx =\n", dx)
assert np.allclose(dK, [ [3, 6], [3, 5] ]), "the lesson's dL/dK"
assert np.allclose(dx, [ [1, 0, -1, 0], [2, 0, 0, 1], [0, -1, 0, 0], [0, 0, -1, -1] ]), "the lesson's dL/dx"
print("Both gradients equal the lesson's matrices.")
RESULTS["p4_dK"] = dK.tolist()
```

   **What you should see.** `y` is the lesson's matrix `[ [0, -1, -1], [-1, 1, 3], [2, 0, -2] ]`, `dL/dK` is `[ [3, 6], [3, 5] ]` and `dL/dx` is `[ [1, 0, -1, 0], [2, 0, 0, 1], [0, -1, 0, 0], [0, 0, -1, -1] ]`. The three `assert` lines pass silently, and the last line prints "Both gradients equal the lesson's matrices." If an assert fails, check that you used `x[i + a, j + b]` (the input cell the kernel cell met) and not `x[i, j]`.

   Try also (optional):

   - Change `G` to all ones. Then `dL/dK` is the sum of every 2x2 window of `x`; check one entry by hand.
   - Change `K` to `[ [0, 0], [0, 1] ]` (a kernel that copies one cell). What does `dL/dx` look like, and why?
   - Set `G` to zeros except `G[1][1] = 1`. Which input cells get a non-zero gradient? (Only the four cells the window (1,1) read.)

   *Exercise 4.1:* Set `G2` to a 3x3 matrix with a single 1 in the centre cell `(1, 1)` and zeros elsewhere. Compute `dK2 = conv_backward_kernel(x, G2, 2, 2)` and `dx2 = conv_backward_input(K, G2, x.shape)`. What does `dK2` equal? Which cells of `dx2` are non-zero? Why it matters: this shows the backward pass as a *routing* of one output's blame back to exactly the cells that produced it.

```python
G2 = None      # YOUR CODE: a 3x3 matrix of zeros with a 1 in the centre
dK2 = None     # YOUR CODE: conv_backward_kernel(x, G2, 2, 2)
dx2 = None     # YOUR CODE: conv_backward_input(K, G2, x.shape)
if G2 is None or dK2 is None or dx2 is None:
    print("Fill in `G2`, `dK2` and `dx2` above, then run this cell again.")
else:
    check("dK2 is the window at (1,1): [ [1, 3], [1, 0] ]", np.allclose(dK2, x[1:3, 1:3]), "with a single 1 in G the kernel gradient is the window that output cell read")
    check("dx2 has non-zeros only in x[1:3, 1:3]", np.count_nonzero(dx2) == 2 and np.allclose(dx2[1:3, 1:3], K), "the window (1,1) sends back K to the four cells it read")
```

   *What it means.* `dK2` equals the 2x2 window of the picture that the output cell `(1, 1)` read, `[ [1, 3], [1, 0] ]`, and `dx2` is zero except for the four cells of that window, where it holds a copy of `K` (two of its entries are 0, so only two of the four cells are non-zero). One unit of blame at one output goes back along the exact paths it came from.

2. **Step 4.2. Verify with finite differences and with autograd.** Matching the lesson's numbers is good evidence, but a **gradient check** is stronger because it needs no answer sheet: it nudges every entry of `x` and `K` up and down by `eps = 1e-5`, recomputes the loss `L = sum(G * y)`, and takes the slope `(L(a + eps) - L(a - eps)) / (2 eps)`. This is the same test you used on the dense network in Week 3. Then you let PyTorch compute the same gradients with `requires_grad=True` and compare all three. Use float64, as in Week 3: in single precision the differences of two nearly equal losses are lost in rounding.

```python
def numeric_grad(f, a, eps=1e-5):
    """Central finite differences: nudge every entry of the array a by +eps and -eps and see how f() changes."""
    g = np.zeros(a.shape)
    for idx in np.ndindex(*a.shape):
        old = a[idx]
        a[idx] = old + eps; up = f()
        a[idx] = old - eps; down = f()
        a[idx] = old                       # always put the entry back
        g[idx] = (up - down) / (2 * eps)
    return g

loss_fn = lambda: float(np.sum(G * conv2d_np(x, K)))     # L = sum(G * y)
print("loss L =", loss_fn())
dx_num = numeric_grad(loss_fn, x)
dK_num = numeric_grad(loss_fn, K)
print("largest |analytic - numeric|  for dx:", np.abs(dx - dx_num).max(), "   for dK:", np.abs(dK - dK_num).max())
assert np.abs(dx - dx_num).max() < 1e-6 and np.abs(dK - dK_num).max() < 1e-6, "finite differences should agree to 1e-6"
```

```python
# the same gradients from PyTorch's autograd (float64, so the comparison is fair)
xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
Kt = torch.tensor(K, dtype=torch.float64, requires_grad=True)
yt = F.conv2d(xt[None, None], Kt[None, None])[0, 0]          # reshape to (1, 1, H, W) and back
L = (torch.tensor(G) * yt).sum()
L.backward()
dx_t, dK_t = xt.grad.numpy(), Kt.grad.numpy()
print("autograd dL/dK =\n", dK_t)
print("three-way difference:  mine vs numeric", np.abs(dK - dK_num).max(), "  mine vs autograd", np.abs(dK - dK_t).max(), "  (dx)", np.abs(dx - dx_t).max())
assert np.allclose(dx, dx_t) and np.allclose(dK, dK_t), "my loops must equal autograd"
RESULTS["p4_gradcheck_err"] = float(max(np.abs(dx - dx_num).max(), np.abs(dK - dK_num).max()))
check("my loops, finite differences and autograd all agree", True)
```

   **What you should see.** The loss is `L = -2.0`. The largest difference between the analytic and the finite-difference gradient is about `6.6e-12` for `dL/dx` and `1.7e-11` for `dL/dK`, far below the limit `1e-6`; against autograd the difference is 0.0 (the numbers are small integers, so both are exact). The cell ends with a tick. If the finite-difference error is large (say 1e-2), you perturbed a copy instead of the array itself, or you did not restore the entry.

   Try also (optional):

   - Make `eps` larger (1e-2, then 1e-1) and then smaller (1e-9, 1e-12) and look at the error. Where is the sweet spot, and why does it get worse on both sides?
   - Use `dtype=np.float32` for `x` and `K` and repeat the check. What happens to the error?
   - Add `RESULTS["p4_gradcheck_err"]` to your notes: it is the number your instructor will look for.

   *Exercise 4.2:* Write a wrong version `conv_backward_kernel_bad` that uses `x[i, j]` instead of `x[i + a, j + b]` and run the gradient check on it. Store the largest error in `bad_error`. Is it above or below 1e-6? Why it matters: a gradient check is only useful if it *fails* when the code is wrong. Seeing it fail once teaches you to trust it when it passes.

```python
def conv_backward_kernel_bad(x, G, kh, kw):
    return np.zeros((kh, kw))          # YOUR CODE: copy conv_backward_kernel and change the input cell it multiplies
bad_error = None                       # YOUR CODE: largest |bad gradient - dK_num|
if bad_error is None:
    print("Fill in `bad_error` above, then run this cell again.")
else:
    check("the wrong version fails the gradient check (error above 1e-6)", bad_error > 1e-6, "if your error is tiny, the 'bad' version is not different from the good one")
    check("the right version passes", np.abs(dK - dK_num).max() < 1e-6, "dK was computed in Step 4.1")
```

   *What it means.* The bad version puts the same number, 3, into all four kernel cells (it is the sum of `G[i][j] * x[i][j]`, which no longer depends on the kernel cell), while the right answer is `[ [3, 6], [3, 5] ]`, so the largest error is 3.0, far above 1e-6, and the check fails. The same check passes for the correct loops, so it separates right from wrong.

3. **Step 4.3. Understand each term (the convgrad picture).** The lesson figure `convgrad` draws every term of these sums. Here you list them. For the input cell `(1, 1)` the terms of `dL/dx[1][1]` are one per window that read it: the windows `(0,0)`, `(0,1)`, `(1,0)` and `(1,1)`, each multiplied by the kernel cell it used. For the kernel cell `(0, 1)` the terms of `dL/dK[0][1]` are one **per output cell**, nine in all: **sum over all positions because the kernel is shared**. A dense layer has one weight per connection, so each weight gets one term per example; a convolution kernel cell is used at every position, so it gets one term per position. The last lines count how many windows read each input cell, which is also how many terms its gradient has.

```python
def terms_dx(p, q):
    """Every term in dL/dx[p][q]: the output cells (i, j) that read the input cell (p, q), through the kernel cell (a, b) = (p - i, q - j)."""
    out = []
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            a, b = p - i, q - j
            if 0 <= a < K.shape[0] and 0 <= b < K.shape[1]:
                out.append((i, j, a, b, G[i, j] * K[a, b]))
    return out

def terms_dK(a, b):
    """Every term in dL/dK[a][b]: one per output cell (i, j), namely G[i][j] times the input cell the kernel cell (a, b) looked at there."""
    return [ (i, j, G[i, j], x[i + a, j + b], G[i, j] * x[i + a, j + b]) for i in range(G.shape[0]) for j in range(G.shape[1]) ]

print("dL/dx[1][1] has", len(terms_dx(1, 1)), "terms:")
for i, j, a, b, t in terms_dx(1, 1):
    print(f"   output ({i},{j}) read it through kernel cell ({a},{b}):  G[{i}][{j}] * K[{a}][{b}] = {G[i, j]:g} * {K[a, b]:g} = {t:g}")
print("   sum =", sum(t[-1] for t in terms_dx(1, 1)), "  (the lesson says dL/dx[1][1] = 0)")
print()
print("dL/dK[0][1] has", len(terms_dK(0, 1)), "terms:")
for i, j, g_ij, x_ij, t in terms_dK(0, 1):
    print(f"   window ({i},{j}):  G[{i}][{j}] * x[{i}][{j + 1}] = {g_ij:g} * {x_ij:g} = {t:g}")
print("   sum =", sum(t[-1] for t in terms_dK(0, 1)), "  (the lesson says dL/dK[0][1] = 6)")

coverage = np.array([ [len(terms_dx(p, q)) for q in range(4)] for p in range(4) ])
print("\nnumber of terms (= windows that read the cell) for every input cell:\n", coverage)
assert coverage.sum() == 9 * 4, "9 windows times 4 kernel cells"
assert sum(t[-1] for t in terms_dx(1, 1)) == dx[1, 1] and sum(t[-1] for t in terms_dK(0, 1)) == dK[0, 1]
```

   **What you should see.** `dL/dx[1][1]` has 4 terms, `-1`, `0`, `0`, `1`, which add to **0**, the lesson's value. `dL/dK[0][1]` has 9 terms `2, 0, -1, 2, 3, 0, 0, 0, 0` that add to **6**, the lesson's value. The window count is `[ [1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1] ]`: corners are read by one window, edges by two, the middle 2x2 by four. Now go back to your Imagine-first guess. The largest absolute gradient is **2**, at cell `(1, 0)`, which is read by only two windows. More terms means more *chances* for a gradient, not a larger one: the terms can cancel (the middle cell `(1,1)` sums to 0) and what counts is `G` times `K`, not the number of terms.

   Try also (optional):

   - Print `terms_dx(0, 0)`: a single term, `G[0][0] * K[0][0]`.
   - Print `terms_dx(1, 0)` and add the terms by hand to get 2.
   - Replace `K` by `[ [1, 1], [1, 1] ]` and recompute the cell that now gets the largest gradient.

   *Exercise 4.3:* Without running code first, answer: how many terms does `dL/dx[0][0]` have, how many does `dL/dx[1][1]` have, how many does `dL/dK[1][1]` have, and what is the value of `dL/dx[0][0]`? Then run the check. Why it matters: the number of terms is the number of times a parameter or a pixel is *used*. Weight sharing makes a kernel cell used many times, which is why its gradient is a sum of many pieces.

```python
n_terms_corner = None     # YOUR ANSWER: how many terms does dL/dx[0][0] have?
n_terms_centre = None     # YOUR ANSWER: how many terms does dL/dx[1][1] have?
n_terms_dK = None         # YOUR ANSWER: how many terms does dL/dK[1][1] have?
value_corner = None       # YOUR ANSWER: the value of dL/dx[0][0]
if None in (n_terms_corner, n_terms_centre, n_terms_dK, value_corner):
    print("Fill in your four answers above, then run this cell again.")
else:
    check("the corner cell is read by one window", n_terms_corner == 1, "only the window at (0,0) touches x[0][0]")
    check("the centre cell x[1][1] is read by four windows", n_terms_centre == 4, "windows (0,0), (0,1), (1,0), (1,1)")
    check("dL/dK[1][1] has one term per output cell: 9", n_terms_dK == 9, "the kernel is shared, so every window contributes")
    check("dL/dx[0][0] is G[0][0] * K[0][0] = 1", abs(value_corner - 1) < 1e-9, "one term only")
```

   *What it means.* The corner cell has one term, `G[0][0] * K[0][0] = 1`. The centre cell has four. Every kernel cell has nine, one per output cell. A shared weight gets the summed blame of all the places it was used.

4. **Step 4.4. The input gradient is a convolution.** Look at the pattern in the matrix `dL/dx` that you computed with loops: every output cell `(i, j)` writes a copy of the kernel, scaled by `G[i][j]`, into the window it came from, and the copies overlap and add. That is exactly the **transposed convolution** (`F.conv_transpose2d`). It is also what you get by taking the **full convolution** of `G` with the kernel flipped in both directions: pad `G` with `k - 1` zeros on each side, then slide the flipped kernel over it as usual. So the backward pass of a convolution is itself a convolution, which is why deep-learning libraries need no new operation for it. (This holds for stride 1 and no padding; with stride 2 the transposed convolution inserts zeros between the entries of `G`.)

```python
Gt = torch.tensor(G)[None, None]            # shape (1, 1, 3, 3)
Kt1 = torch.tensor(K)[None, None]           # shape (1, 1, 2, 2)
dx_ct = F.conv_transpose2d(Gt, Kt1)[0, 0].numpy()          # the "transposed convolution" of G with K
dx_full = conv2d_np(G, np.flip(K), pad=1)                   # the "full" convolution: pad G by k-1 = 1 and slide the flipped kernel
print("conv_transpose2d(G, K) =\n", dx_ct)
print("full convolution of G with the flipped K =\n", dx_full)
assert np.allclose(dx_ct, dx) and np.allclose(dx_full, dx), "the input gradient is a convolution of G"
print("Both equal dL/dx, the matrix your loops produced.")
RESULTS["p4_dx_is_convolution"] = True
```

   **What you should see.** Both printed matrices are `[ [1, 0, -1, 0], [2, 0, 0, 1], [0, -1, 0, 0], [0, 0, -1, -1] ]`, identical to your loops' `dL/dx`. `conv_transpose2d` gives a (1, 1, 4, 4) tensor, so the picture is recovered at its original size; the full convolution takes a 5x5 padded `G` and a 2x2 kernel to a 4x4 result. The assert passes.

   Try also (optional):

   - Try the full convolution without the flip: `conv2d_np(G, K, pad=1)`. Is it still equal? (It is not, here: the flip matters.)
   - Change `K` to a symmetric kernel such as `[ [1, 2], [2, 1] ]`: now the flip changes nothing in this case. Why?
   - Use `conv2d_np(G, np.flip(K))` without padding: what shape do you get, and why is it too small?

   *Exercise 4.4:* Compute the flipped kernel (`np.flip(K)`) and the shape of `G` after padding by 1 on each side. Why it matters: the flip and the padding are the two details people forget when they implement the backward pass of a convolution, and both come from the same fact: output cell `(i, j)` sends its blame to the window it read, and the window is read by a kernel that is mirrored when seen from the input side.

```python
flipped_K = None       # YOUR CODE: the kernel [ [1, 0], [0, -1] ] flipped in both directions (use np.flip)
shape_dx_full = None   # YOUR ANSWER: the shape of the padded G (pad = k - 1 = 1 on each side) as a tuple
if flipped_K is None or shape_dx_full is None:
    print("Fill in `flipped_K` and `shape_dx_full` above, then run this cell again.")
else:
    check("the flipped kernel is [ [-1, 0], [0, 1] ]", np.allclose(flipped_K, [ [-1, 0], [0, 1] ]), "np.flip(K) reverses rows and columns")
    check("padded G is 5 x 5", tuple(shape_dx_full) == (5, 5), "3 + 2 * 1 = 5 in each direction")
```

   *What it means.* The flipped kernel is `[ [-1, 0], [0, 1] ]` and the padded `G` is 5x5. Sliding a 2x2 kernel over a 5x5 matrix gives a 4x4 result, the size of the original picture.

5. **Step 4.5. Stride and padding in the backward pass.** Real layers use stride and padding. The idea does not change; only the bookkeeping does. With stride `s` and padding `p`, output cell `(i, j)` reads the window whose top-left corner is `(i*s, j*s)` in the **padded** picture. The kernel gradient adds `G[i][j]` times that window; the picture gradient adds `G[i][j]` times `K` into that window; and at the end we cut the padding off, because the padding cells are not part of the picture. Write this once for any `(stride, pad)` and test it on a 5x5 input with a 3x3 kernel for four settings, each with a finite-difference check and an autograd check.

```python
def conv_backward(x, K, G, stride=1, pad=0):
    """Gradients of L = sum(G * conv(x, K)) for any stride and padding. Returns (dx, dK).
    Same idea as before, but the window of output cell (i, j) now starts at (i * stride, j * stride) in the PADDED picture."""
    kh, kw = K.shape
    xp = np.pad(x, pad)
    dxp = np.zeros(xp.shape); dK = np.zeros((kh, kw))
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            r, c = i * stride, j * stride
            dK += G[i, j] * xp[r:r + kh, c:c + kw]            # what the kernel saw in this window, weighted by the upstream gradient
            dxp[r:r + kh, c:c + kw] += G[i, j] * K            # what this window sends back to the picture
    h, w = x.shape
    return dxp[pad:pad + h, pad:pad + w], dK        # the padding cells are not part of the picture: drop them

rs = np.random.RandomState(0)
x5, K3 = rs.randn(5, 5), rs.randn(3, 3)
for stride, pad in [ (1, 0), (1, 1), (2, 1), (2, 0) ]:
    out = conv2d_np(x5, K3, stride, pad)
    G5 = rs.randn(*out.shape)
    dx5, dK3 = conv_backward(x5, K3, G5, stride, pad)
    f = lambda: float(np.sum(G5 * conv2d_np(x5, K3, stride, pad)))
    err = max(np.abs(dx5 - numeric_grad(f, x5)).max(), np.abs(dK3 - numeric_grad(f, K3)).max())
    xt5 = torch.tensor(x5, requires_grad=True); Kt5 = torch.tensor(K3, requires_grad=True)
    (torch.tensor(G5) * F.conv2d(xt5[None, None], Kt5[None, None], stride=stride, padding=pad)[0, 0]).sum().backward()
    err_t = max(np.abs(dx5 - xt5.grad.numpy()).max(), np.abs(dK3 - Kt5.grad.numpy()).max())
    print(f"stride {stride}, pad {pad}:  y {out.shape},  dx {dx5.shape},  dK {dK3.shape},  error vs finite differences {err:.1e},  vs autograd {err_t:.1e}")
    assert err < 1e-6 and err_t < 1e-9, "gradient check failed"
```

   **What you should see.** Four lines. Stride 1, pad 0: the output is 3x3. Stride 1, pad 1: 5x5 (padding keeps the size). Stride 2, pad 1: 3x3. Stride 2, pad 0: 2x2. In every case `dx` is (5, 5) and `dK` is (3, 3): a gradient always has the shape of the thing it is the gradient of. The errors are about `1e-10` against finite differences and below `1e-14` against autograd. All the assertions pass.

   Try also (optional):

   - Add the setting `(3, 2)` (stride 3, padding 2). What is the output size by the formula, and does the check still pass?
   - Print `dx5` for stride 2, pad 0 and look for rows or columns that are exactly 0. Which are they and why?
   - Remove the cropping `[pad:pad + ...]` and print the shape of the result for pad 1. What goes wrong?

   *Exercise 4.5:* A 2x2 kernel with stride 2 and no padding on a 5x5 picture: what is the shape of the output, and how many of the 25 input cells receive a gradient of exactly 0? Run `conv_backward` with ones to find out, then explain in a sentence why. Why it matters: when the stride does not divide the picture evenly, the last rows and columns are never read, and a layer cannot learn from pixels it never sees.

```python
out_shape = None     # YOUR ANSWER: the shape of the output of a 2x2 kernel with stride 2, no padding, on a 5x5 picture
zero_cells = None    # YOUR ANSWER: how many of the 25 input cells get a gradient of exactly 0?
if out_shape is None or zero_cells is None:
    print("Fill in `out_shape` and `zero_cells` above, then run this cell again.")
else:
    check("the output is 2 x 2", tuple(out_shape) == (2, 2), "floor((5 - 2) / 2) + 1 = 2")
    xx, KK = np.ones((5, 5)), np.ones((2, 2))
    dx_test, _ = conv_backward(xx, KK, np.ones((2, 2)), stride=2, pad=0)
    check("the number of zero-gradient cells matches the loops", zero_cells == int((dx_test == 0).sum()), "count the cells no window ever reads")
```

   *What it means.* The output is 2x2 and 9 cells have gradient 0: the four windows cover rows 0 to 3 and columns 0 to 3, which is 16 cells, so the last row and the last column are never read.

6. **Step 4.6. Through ReLU and max-pooling.** A CNN does not apply a convolution alone: it follows it with ReLU and max-pooling. Their backward passes are the simplest of all because they only *route* the gradient. ReLU passes the gradient where its input was positive and blocks it elsewhere. Max-pooling sends the whole gradient of a pooled cell to the **one input cell that won the max** and gives nothing to the other three. Combined, in a 2x2 block only the winner gets a gradient, and only if the winner is positive. Here is a 4x4 feature map `M` with some negative values, ReLU, 2x2 max-pooling, and a "next layer" that weighs the four pooled values by `[ [1, 2], [3, 4] ]`. The function `relu_maxpool_backward` does the routing by hand; autograd checks it.

```python
M = np.array([ [1, -2, 3, 0], [-1, 4, -3, 2], [-1, -2, 0, 1], [-3, -1, -2, -1] ], dtype=float)   # a 4x4 feature map
Wt = np.array([ [1, 2], [3, 4] ], dtype=float)                       # the next layer weighs the four pooled values

def relu_maxpool_backward(M, G):
    """Gradient of L = sum(G * maxpool2x2(relu(M))) with respect to M. Only the winner of each 2x2 block, and only if it is positive, gets anything."""
    dM = np.zeros(M.shape)
    for I in range(0, M.shape[0], 2):
        for J in range(0, M.shape[1], 2):
            block = np.maximum(M[I:I + 2, J:J + 2], 0)               # ReLU
            a, b = np.unravel_index(np.argmax(block), block.shape)    # which cell of the block wins the max-pool?
            if M[I + a, J + b] > 0:                                   # ReLU lets the gradient pass only where its input was positive
                dM[I + a, J + b] = G[I // 2, J // 2]
    return dM

pooled = np.array([ [np.maximum(M[I:I + 2, J:J + 2], 0).max() for J in (0, 2)] for I in (0, 2) ])
print("after ReLU and 2x2 max-pool:\n", pooled, "\nL = sum(Wt * pooled) =", float(np.sum(Wt * pooled)))
dM = relu_maxpool_backward(M, Wt)
print("dL/dM by hand =\n", dM)

Mt = torch.tensor(M, requires_grad=True)
L = (torch.tensor(Wt) * F.max_pool2d(F.relu(Mt)[None, None], 2)[0, 0]).sum()
L.backward()
print("autograd =\n", Mt.grad.numpy())
assert np.allclose(dM, Mt.grad.numpy()), "hand gradient must equal autograd"
assert (dM != 0).sum() == 3, "three of the 16 cells receive a gradient"
RESULTS["p4_relu_pool_nonzero"] = int((dM != 0).sum())
```

   **What you should see.** The pooled map is `[ [4, 3], [0, 1] ]` and `L = 14`. The gradient `dL/dM` has three non-zero cells: `2` at `(0, 2)`, `1` at `(1, 1)` and `4` at `(2, 3)`; everything else is 0. Autograd prints the same matrix. The bottom-left block (all four values negative) sends nothing back: ReLU had cut them to 0 and passes no gradient there, so that pooled value (0) and its weight 3 had no influence on the map.

   Try also (optional):

   - Change `M[2][0]` to `5` and check which cell of the bottom-left block now receives the gradient `3`.
   - Make two cells of one block equal (a tie). Does your function agree with autograd? (Frameworks pick one cell by a fixed rule; ties are rare with real data.)
   - Replace the 2x2 pool by a 4x4 pool of the whole map. Which cell receives the gradient?

   *Exercise 4.6:* Count the cells of `M` that receive a non-zero gradient, say what the maximum possible number of such cells is for a 4x4 map with 2x2 pooling, and write one sentence on why the bottom-left block sends back nothing. Why it matters: this is why a CNN's gradient is *sparse*: most cells of a feature map do not directly decide the answer, so only a few of them are corrected on each step.

```python
nonzero_cells = None     # YOUR ANSWER: how many of the 16 cells of M receive a non-zero gradient above?
max_nonzero = None       # YOUR ANSWER: with a 4x4 map and 2x2 pooling, what is the MOST cells that could ever receive a gradient?
why_zero = ""            # YOUR ANSWER (a sentence): why does the bottom-left block send back nothing?
if nonzero_cells is None or max_nonzero is None:
    print("Fill in `nonzero_cells` and `max_nonzero` above (and write `why_zero`), then run this cell again.")
else:
    check("three cells get a gradient", nonzero_cells == 3, "count the non-zero entries of dM")
    check("at most one winner per 2x2 block: 4 cells", max_nonzero == 4, "four blocks, one winner each")
    check("your sentence mentions ReLU or negative", ("relu" in why_zero.lower()) or ("negative" in why_zero.lower()), "all four cells of that block were negative")
```

   *What it means.* Three cells receive a gradient; at most four could (one winner per 2x2 block). The bottom-left block is all negative, ReLU cut it to zero and blocks the gradient on the way back.

7. **Step 4.7. A two-layer chain: conv, ReLU, pool, flatten, linear.** Now put it together on one 6x6 picture: convolution with a 3x3 kernel and a bias (output 4x4), ReLU, 2x2 max-pooling (output 2x2), flatten (4 numbers), a linear layer to two scores, and the cross-entropy loss with the target class 1. First let autograd compute everything and keep the gradient at the convolution output with `retain_grad` (normally PyTorch throws away the gradients of intermediate values). Then use your own loops with that upstream gradient and check that your `dL/dK` equals autograd's. In the second cell, go further and compute the whole chain by hand from the loss, using the pieces you wrote in this part: the softmax-minus-target gradient, the transposed linear layer, the routing of Step 4.6 and the kernel loops of Step 4.1.

```python
# a tiny network on one 6x6 picture:  conv 3x3 (+ bias) -> ReLU -> 2x2 max-pool -> flatten -> linear -> cross-entropy
pic, lab = make_shapes(1, 6, noise=0.1, seed=3)
pic_np = pic[0, 0].astype(np.float64)
rs = np.random.RandomState(0)
Kc = torch.tensor(rs.randn(3, 3), requires_grad=True)             # float64 leaf tensors, so we can compare to 1e-12
bc = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
W2 = torch.tensor(0.5 * rs.randn(2, 4), requires_grad=True)
b2 = torch.zeros(2, dtype=torch.float64, requires_grad=True)
target = torch.tensor([1])

z = F.conv2d(torch.tensor(pic_np)[None, None], Kc[None, None], bc)   # (1, 1, 4, 4)
z.retain_grad()                                                       # keep the gradient of this intermediate value
r = F.relu(z)
p = F.max_pool2d(r, 2)                                                # (1, 1, 2, 2)
logits = p.flatten(1) @ W2.T + b2                                     # (1, 2)
loss = F.cross_entropy(logits, target)
loss.backward()
print("shapes: z", tuple(z.shape), " pooled", tuple(p.shape), " logits", tuple(logits.shape), " loss", round(float(loss), 4))
print("autograd gradients:  dK", tuple(Kc.grad.shape), " dbias", tuple(bc.grad.shape), " dW2", tuple(W2.grad.shape), " db2", tuple(b2.grad.shape))

# (1) my loops, with the upstream gradient taken from autograd at the conv output
G_auto = z.grad[0, 0].numpy()
dK_mine = conv_backward_kernel(pic_np, G_auto, 3, 3)
print("\nupstream gradient at the conv output (4x4):\n", np.round(G_auto, 4))
print("dL/dK from my loops:\n", np.round(dK_mine, 4))
assert np.allclose(dK_mine, Kc.grad.numpy()), "my conv gradient must equal autograd's"
```

```python
# (2) the whole chain by hand, from the loss back to the kernel
probs = torch.softmax(logits.detach(), dim=1)[0].numpy()
d_logits = probs - np.eye(2)[1]                                  # cross-entropy with softmax: probabilities minus the one-hot target
d_flat = W2.detach().numpy().T @ d_logits                          # back through the linear layer
d_pooled = d_flat.reshape(2, 2)                                    # un-flatten
G_hand = relu_maxpool_backward(z.detach().numpy()[0, 0], d_pooled)   # back through max-pool and ReLU (the function of Step 4.6)
dK_hand = conv_backward_kernel(pic_np, G_hand, 3, 3)               # back through the convolution (the loops of Step 4.1)
print("upstream gradient at the conv output, by hand vs autograd, largest difference:", np.abs(G_hand - G_auto).max())
print("dL/dK, by hand vs autograd, largest difference:", np.abs(dK_hand - Kc.grad.numpy()).max())
assert np.allclose(G_hand, G_auto) and np.allclose(dK_hand, Kc.grad.numpy()), "the hand chain must equal autograd"
dW2_hand = np.outer(d_logits, p.detach().numpy().ravel())
assert np.allclose(dW2_hand, W2.grad.numpy()) and np.allclose(d_logits, b2.grad.numpy())
bias_grad_auto = float(bc.grad)
print("Every gradient of the little CNN, computed by hand, equals autograd's.")
RESULTS["p4_chain_ok"] = True
```

   **What you should see.** The shapes are z (1, 1, 4, 4), pooled (1, 1, 2, 2), logits (1, 2), and the loss is `0.3798`. The gradients have the shapes of their parameters: `dK` (3, 3), `dbias` (1,), `dW2` (2, 4), `db2` (2,). The upstream gradient at the conv output (4x4) has only **four non-zero entries**, one per pooling block: `0.0456` at (1,0), `-0.0474` at (1,3), `-0.1158` at (2,3) and `0.1770` at (3,1). Everything else is exactly 0 because max-pooling and ReLU routed it away. Your loops give `dL/dK` equal to autograd's (difference about `1e-17`), and the chain computed by hand from the loss agrees with every autograd gradient.

   Try also (optional):

   - Change the target to class 0 and see which entries of the upstream gradient change sign.
   - Print `bc.grad` and `G_auto.sum()` and compare. This is Exercise 4.7.
   - Replace the max-pool by `F.avg_pool2d`. How many entries of the upstream gradient are non-zero now?

   *Exercise 4.7:* The convolution bias `bc` is one number added to all 16 output cells. Compute its gradient yourself from `G_auto` (one line of NumPy) and store it in `db_conv`. Why it matters: it is the same rule as for the kernel, "sum over all positions", but for a shared number that multiplies a constant 1.

```python
db_conv = None      # YOUR CODE: the gradient of the loss with respect to the conv bias (a single number), from G_auto
if db_conv is None:
    print("Fill in `db_conv` above, then run this cell again.")
else:
    check("db_conv matches autograd", abs(db_conv - bias_grad_auto) < 1e-9, "the bias is added to every output cell, so add up all of G")
```

   *What it means.* The bias is added to every output cell, so its gradient is the sum of the upstream gradient over all positions: `G_auto.sum()`, here about 0.0594, equal to autograd's `bc.grad`.

   **Before you move on (Part 4).** Check that you can say each of these in your own words: the kernel gradient adds `G[i][j]` times the input cell that kernel cell met, over all windows; the input gradient adds `G[i][j]` times the kernel cell, over all windows that read the cell; both are sums over positions because the kernel is shared; the input gradient is a transposed (full) convolution with the flipped kernel; ReLU and max-pooling only route gradients; and a gradient check compares an analytic gradient with a finite difference.

8. **Imagine first, revisited.** Run the next block to compare your Imagine-first prediction with the facts. Compare your Imagine-first guess with the facts. Corners are read by 1 window, the centre by 4, but the largest gradient, 2, is at the edge cell `(1, 0)`. In a markdown cell write whether you were right, and why a cell read by *more* windows does not automatically receive a *larger* gradient.

```python
if guess_cell is None:
    print("You did not write a guess in the Imagine-first cell; the facts are below anyway.")
else:
    print("Your guess:", guess_cell, "- it is read by", coverage[guess_cell], "windows and its gradient is", dx[guess_cell])
big = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(dx)), dx.shape))
print("The largest |dL/dx| is", dx[big], "at cell", big, "which is read by", coverage[big], "windows.")
print("The cells read by the most windows (4) have gradients:", dx[1:3, 1:3].ravel())
RESULTS["p4_largest_cell"] = big
```

### Part 5. Build and train a CNN (30 min)

*Lesson section: "Training".* Now you build a real convolutional network in PyTorch and compare it with a dense network of the same size on the shapes of the lab. You will print the shape and the parameter count after every layer and check them against your by-hand numbers, train both networks on centred shapes, test both on shapes that have moved, train on shifted shapes and count what it costs a dense network to catch up, provoke the three most common shape errors on purpose, and (optionally) add batch normalisation and dropout. ==A convolutional network reuses the same few weights at every position, so it needs far fewer parameters than a dense network and copes better when the pattern appears somewhere new.==

**Imagine first.** Both networks you will compare have about **2,500 parameters**: the CNN has 2,533 and the dense network (MLP) has 2,625. Both will be trained only on shapes in the **centre** of the picture. Then you test them on shapes that have moved 3 pixels away from the centre. Which network do you expect to be better, and by how many percentage points? Write `"cnn"`, `"mlp"` or `"tie"` in `guess_winner`, and your guess for the gap in `guess_gap`. You test it in Step 5.3.

```python
guess_winner = ""       # YOUR TURN: "cnn", "mlp" or "tie": who is better on shifted shapes?
guess_gap = None        # YOUR TURN: by how many percentage points (e.g. 10)? Think of the test set with shift 3.
print("Your guess:", guess_winner if guess_winner else "(nothing yet)", "wins by about", guess_gap, "points")
```

1. **Step 5.1. Build SmallCNN and read its shapes.** `SmallCNN` follows the lesson's recipe on 16x16 grey pictures: `Conv(1, 8, 3, padding 1)`, ReLU, `MaxPool(2)`, `Conv(8, 16, 3, padding 1)`, ReLU, `MaxPool(2)`, `Flatten`, `Linear(16*4*4, 5)`. The padding 1 keeps the size at 16x16 and 8x8 after each convolution; each pooling halves the side, so two poolings leave 4x4 maps, and the last layer reads 16 channels of 4x4 = 256 numbers. The class is built on `nn.Sequential` with named layers so that you can loop over `named_children()`, push a dummy picture through, and print the shape and the parameters of every layer. Then compare each count with the formula `k*k*C_in*C_out + C_out` from Part 2.

```python
from collections import OrderedDict

class SmallCNN(nn.Sequential):
    """Conv(1,8,3,pad 1) - ReLU - MaxPool(2) - Conv(8,16,3,pad 1) - ReLU - MaxPool(2) - Flatten - Linear(16*4*4, 5), for 16x16 pictures."""
    def __init__(self, n_classes=5, c1=8, c2=16, size=16):
        s = size // 4                                  # two poolings divide the side by 4
        super().__init__(OrderedDict([
            ("conv1", nn.Conv2d(1, c1, 3, padding=1)), ("relu1", nn.ReLU()), ("pool1", nn.MaxPool2d(2)),
            ("conv2", nn.Conv2d(c1, c2, 3, padding=1)), ("relu2", nn.ReLU()), ("pool2", nn.MaxPool2d(2)),
            ("flatten", nn.Flatten()), ("fc", nn.Linear(c2 * s * s, n_classes))]))

torch.manual_seed(0)
cnn = SmallCNN()
h = torch.zeros(1, 1, 16, 16)                         # a dummy picture: (batch, channels, height, width)
print(f"{'input':8s} -> {tuple(h.shape)}")
for name, layer in cnn.named_children():
    h = layer(h)
    print(f"{name:8s} -> {str(tuple(h.shape)):16s} parameters: {count_params(layer)}")
print("total parameters:", count_params(cnn))

by_hand = {"conv1": 3 * 3 * 1 * 8 + 8, "conv2": 3 * 3 * 8 * 16 + 16, "fc": 16 * 4 * 4 * 5 + 5}
for name, n in by_hand.items():
    assert count_params(getattr(cnn, name)) == n, name
assert count_params(cnn) == sum(by_hand.values()) == 2533, "80 + 1168 + 1285"
print("By hand:", by_hand, "->", sum(by_hand.values()))
RESULTS["p5_cnn_params"] = count_params(cnn)
```

   **What you should see.** The shapes are `(1, 8, 16, 16)` after `conv1`, `(1, 8, 8, 8)` after the first pooling, `(1, 16, 8, 8)` after `conv2`, `(1, 16, 4, 4)` after the second pooling, `(1, 256)` after `flatten` and `(1, 5)` at the end. The parameters are **80** (conv1: 3*3*1*8 + 8), **1,168** (conv2: 3*3*8*16 + 16) and **1,285** (fc: 256*5 + 5); ReLU, pooling and flatten have none. The total is **2,533**, equal to your by-hand sum (the asserts pass). Notice where the parameters are: more than half are in the last, dense layer.

   Try also (optional):

   - Change `c1` and `c2` to 16 and 32 and predict the total before you run it.
   - Change `padding=1` to `padding=0` in the convolutions: what are the shapes now, and does the model still build? (The Linear layer will complain: you fix that in Step 5.5.)
   - Add `print(cnn)` and read the layer names; then print `cnn.conv1.weight.shape`.

   *Exercise 5.1:* Compute by hand (with the formula) the parameters of the second conv layer, the length of the vector that `Flatten` hands to the Linear layer, and how many parameters the Linear layer would need if we removed both poolings (the map would stay 16x16). Why it matters: the last dense layer is where parameters hide; pooling is not only about shift tolerance, it also shrinks that layer.

```python
params_conv2 = None       # YOUR ANSWER: parameters of the second conv layer (8 -> 16 filters, 3x3), from the formula
flat_size = None          # YOUR ANSWER: how many numbers does Flatten hand to the Linear layer?
params_if_no_pool = None  # YOUR ANSWER: parameters of the Linear layer if we removed BOTH poolings (the map stays 16x16)
if None in (params_conv2, flat_size, params_if_no_pool):
    print("Fill in your three answers above, then run this cell again.")
else:
    check("conv2 has 3*3*8*16 + 16 = 1168 parameters", params_conv2 == 1168, "k*k*C_in*C_out + C_out")
    check("Flatten hands 16 * 4 * 4 = 256 numbers", flat_size == 256, "16 channels, each a 4x4 map after two poolings")
    check("without pooling the Linear layer would have 16*16*16*5 + 5 = 20485", params_if_no_pool == 20485, "the flat vector would have 4096 numbers")
```

   *What it means.* conv2 has 1,168, Flatten gives 256 numbers, and without pooling the Linear layer would need 16*16*16*5 + 5 = 20,485 parameters, eight times more than the whole CNN.

2. **Step 5.2. Train the CNN and an equally small MLP on centred shapes.** To compare fairly, the dense network must be about as large as the CNN. The MLP of Part 1 had 64 hidden units (16,773 parameters); a hidden width of 10 gives `256*10 + 10 + 10*5 + 5 = 2,625` parameters, close to the CNN's 2,533. Both train on the same 600 centred pictures (shift 0, noise 0.15) with Adam, learning rate 0.01, for 30 epochs, and are tested on 300 other centred pictures. The code prints a table and plots the training loss and the test accuracy per epoch. We also record the first epoch at which each model reaches 90 % test accuracy.

```python
def accuracy(model, X, y):
    """Fraction of pictures the model classifies correctly."""
    model.eval()
    with torch.no_grad():
        return float((model(torch.as_tensor(X)).argmax(1) == torch.as_tensor(y)).float().mean())

def make_mlp(hidden, size=16):
    return nn.Sequential(nn.Flatten(), nn.Linear(size * size, hidden), nn.ReLU(), nn.Linear(hidden, 5))

# the training and test sets: centred shapes (shift 0), a little noise
Xtr, ytr = make_shapes(600, 16, noise=0.15, shift=0, seed=1)
Xte, yte = make_shapes(300, 16, noise=0.15, shift=0, seed=2)

torch.manual_seed(0); mlp = make_mlp(10)               # the width 10 gives about as many parameters as the CNN
torch.manual_seed(0); cnn = SmallCNN()
print("parameters:  MLP", count_params(mlp), "   CNN", count_params(cnn))

h_mlp = train_classifier(mlp, Xtr, ytr, Xte, yte, epochs=30, lr=0.01, seed=0)
h_cnn = train_classifier(cnn, Xtr, ytr, Xte, yte, epochs=30, lr=0.01, seed=0)
print(f"{'model':6s} {'params':>7s} {'train acc':>10s} {'test acc':>9s}")
for name, m, h in [ ("MLP", mlp, h_mlp), ("CNN", cnn, h_cnn) ]:
    print(f"{name:6s} {count_params(m):7d} {h['train_acc'][-1]:10.3f} {h['test_acc'][-1]:9.3f}")
first90 = lambda h: next((e + 1 for e, a in enumerate(h["test_acc"]) if a >= 0.9), None)
print("first epoch with test accuracy of at least 0.9:   MLP", first90(h_mlp), "  CNN", first90(h_cnn))
print("training loss at epochs 1, 5, 30:   MLP", [round(h_mlp["loss"][e], 3) for e in (0, 4, 29)], "  CNN", [round(h_cnn["loss"][e], 3) for e in (0, 4, 29)])
RESULTS["p5_centred"] = {"mlp": h_mlp["test_acc"][-1], "cnn": h_cnn["test_acc"][-1]}

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for name, h in [ ("MLP", h_mlp), ("CNN", h_cnn) ]:
    ax[0].plot(h["loss"], label=name); ax[1].plot(h["test_acc"], label=name)
ax[0].set_title("training loss"); ax[1].set_title("test accuracy"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch"); ax[0].legend()
plt.tight_layout(); plt.show()
assert h_cnn["loss"][-1] < h_cnn["loss"][0], "the CNN should learn"
```

   **What you should see.** MLP 2,625 parameters, CNN 2,533. Both reach **train accuracy 1.000 and test accuracy 1.000**. The CNN reaches 90 % test accuracy in epoch 1, the MLP in epoch 2; the training loss falls from about 1.3 in the first epoch to about 0.001 (MLP) or 0.000 (CNN) at the end. On centred shapes there is nothing to tell the two apart: this task is easy, and a small dense network can memorise the positions. The difference appears only when the pictures change, in the next step. (A flat accuracy line at 1.0 is a finding, not a bug.)

   Try also (optional):

   - Plot the learning curves on a log scale (`ax.set_yscale("log")`) to see how fast the loss falls for each model.
   - Train both with `lr=0.001` and report how many epochs each needs to reach 90 %.
   - Make the data harder: `noise=0.5` for training and test. Do the two models still tie?

   *Exercise 5.2:* (a) How many parameters does the MLP with 64 hidden units have (it is the network of Part 1)? Compute it with `count_params(make_mlp(64))`. (b) The MLP `256 - h - 5` has `262*h + 5` parameters. Which integer width `h` gives a count closest to the CNN's 2,533? Why it matters: comparing a big network with a small one would prove nothing about *structure*; fair comparisons match the size.

```python
mlp_params_16 = None     # YOUR CODE: how many parameters would an MLP with 64 hidden units have (256 -> 64 -> 5)?
width_for_cnn = None     # YOUR CODE: which hidden width h gives about 2533 parameters? (262*h + 5 parameters for a 256-h-5 MLP)
if mlp_params_16 is None or width_for_cnn is None:
    print("Fill in `mlp_params_16` and `width_for_cnn` above, then run this cell again.")
else:
    check("256-64-5 has 16773 parameters", mlp_params_16 == 16773, "256*64 + 64 + 64*5 + 5")
    check("a width of about 10 matches the CNN", 9 <= width_for_cnn <= 10, "262*h + 5 is close to 2533 for h = 9 or 10")
```

   *What it means.* The 64-unit MLP has 16,773 parameters (6.6 times the CNN) and the width that matches the CNN is 10 (2,625; the width 9 gives 2,363).

3. **Step 5.3. The shift test, again.** You met this test in Part 1 for the MLP. Now you run both trained models on fresh test sets whose shapes sit up to 0, 1, 2, 3 and 4 pixels away from the centre (300 new pictures each, noise 0.15), and plot accuracy against the shift. Chance is 0.2 (five classes). Whichever way it comes out, report it as measured: the point of the experiment is to find out, not to confirm a story.

```python
shifts = [0, 1, 2, 3, 4]
acc_mlp, acc_cnn = [], []
for s in shifts:
    Xs, ys = make_shapes(300, 16, noise=0.15, shift=s, seed=10 + s)       # fresh pictures, shape moved by up to s pixels
    acc_mlp.append(accuracy(mlp, Xs, ys)); acc_cnn.append(accuracy(cnn, Xs, ys))
print("shift  :", shifts)
print("MLP    :", [round(a, 3) for a in acc_mlp])
print("CNN    :", [round(a, 3) for a in acc_cnn])
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(shifts, acc_mlp, "o-", label="MLP"); ax.plot(shifts, acc_cnn, "s-", label="CNN")
ax.axhline(0.2, ls=":", c="gray"); ax.set_xlabel("shift of the test shapes (pixels)"); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.05); ax.legend()
ax.set_title("Trained on centred shapes"); plt.tight_layout(); plt.show()
RESULTS["p5_shift_mlp"] = [round(a, 3) for a in acc_mlp]; RESULTS["p5_shift_cnn"] = [round(a, 3) for a in acc_cnn]
assert acc_cnn[0] > 0.9 and acc_mlp[0] > 0.9, "both models should be good on centred shapes"
```

   **What you should see.** MLP: `[1.0, 0.38, 0.293, 0.227, 0.13]`. CNN: `[1.0, 0.953, 0.687, 0.483, 0.38]`. At shift 0 they tie. At shift 1 the MLP has already fallen to **0.38**, the CNN to 0.953; at shift 3 the CNN (0.483) is ahead of the MLP (0.227) by about **25.7 points**; at shift 4 the MLP's 0.13 is *below* chance (0.2), the CNN's 0.38 is still above it. So the CNN degrades **less**, but it still degrades: its last layer is a dense layer over a 4x4 map, and the 2x2 poolings absorb only one or two pixels of movement. The same weights found the pattern in the new place; the network is not truly shift-invariant.

   Try also (optional):

   - Print the confusion of the CNN at shift 3 (which classes get mixed up?). A short bar and a cross moved away may look alike.
   - Test at shift 2 with `noise=0.4`: which model loses more?
   - Retrain with another seed (`torch.manual_seed(1)`, `seed=1`) and see if the order of the two lines changes. Report the range.

   *Exercise 5.3:* Compute the gap at shift 3 in percentage points (CNN minus MLP), say who wins, and give a one- or two-sentence reason in terms of what the weights of each model know about position. Why it matters: the reason, not the number, is what you can carry to a new problem.

```python
gap_at_3 = None     # YOUR CODE: CNN accuracy minus MLP accuracy on the shift-3 test set, in percentage points
who_wins = ""       # YOUR ANSWER: "cnn", "mlp" or "tie" at shift 3
why = ""            # YOUR ANSWER: in one or two sentences, why? (think about what each model's weights know about position)
if gap_at_3 is None or not who_wins:
    print("Fill in `gap_at_3` and `who_wins` (and write `why`), then run this cell again.")
else:
    true_gap = 100 * (acc_cnn[3] - acc_mlp[3])
    check("gap_at_3 matches your measurement", abs(gap_at_3 - true_gap) < 0.5, "100 * (acc_cnn[3] - acc_mlp[3])")
    check("who_wins agrees with the numbers", who_wins == ("cnn" if true_gap > 2 else "mlp" if true_gap < -2 else "tie"), "compare the two lines at shift 3")
    check("your explanation has at least 12 words", len(why.split()) >= 12, "write a real sentence")
```

   *What it means.* The gap is about 25.7 points in favour of the CNN. The MLP gives every pixel position its own weights, so a shape in a place it never saw meets weights that learned nothing for it; the CNN uses the same filters at every position.

4. **Step 5.4. Train on shifted pictures: what does the dense network need?.** Now give both models what the test will look like: training pictures whose shapes sit up to 3 pixels from the centre (data augmentation by shifting), tested on shifted pictures (shift 3, 300 new ones). To see how much *data* a dense network needs to catch up, repeat with 150, 300, 600 and 1,200 training pictures, for three models: the small MLP (width 10), the larger MLP (width 64, 16,773 parameters) and the CNN. Twelve trainings of 30 epochs take a few seconds each. Read the table by columns: parameters, number of pictures, accuracy.

```python
# train on shifted pictures (shift 3) and test on shifted pictures; and see how many pictures the MLP needs
Xte3, yte3 = make_shapes(300, 16, noise=0.15, shift=3, seed=30)
rows = []
print(f"{'model':8s} {'params':>7s} {'train pics':>10s} {'train acc':>10s} {'test acc (shift 3)':>19s}")
for n_train in [150, 300, 600, 1200]:
    Xs, ys = make_shapes(n_train, 16, noise=0.15, shift=3, seed=20 + n_train // 150)
    for name, build in [ ("MLP-10", lambda: make_mlp(10)), ("MLP-64", lambda: make_mlp(64)), ("CNN", SmallCNN) ]:
        torch.manual_seed(0); m = build()
        h = train_classifier(m, Xs, ys, Xte3, yte3, epochs=30, lr=0.01, seed=0)
        rows.append((name, count_params(m), n_train, h["train_acc"][-1], h["test_acc"][-1]))
        print(f"{name:8s} {count_params(m):7d} {n_train:10d} {h['train_acc'][-1]:10.3f} {h['test_acc'][-1]:19.3f}")
RESULTS["p5_shifted_training"] = {f"{r[0]}@{r[2]}": round(r[4], 3) for r in rows}

fig, ax = plt.subplots(figsize=(5.5, 3.4))
for name, mk in [ ("MLP-10", "o--"), ("MLP-64", "^--"), ("CNN", "s-") ]:
    pts = [(r[2], r[4]) for r in rows if r[0] == name]
    ax.plot(*zip(*pts), mk, label=name)
ax.set_xscale("log"); ax.set_xlabel("number of training pictures (shift 3)"); ax.set_ylabel("test accuracy (shift 3)"); ax.set_ylim(0, 1.05); ax.legend()
plt.tight_layout(); plt.show()
```

   **What you should see.** With only **150** pictures: MLP-10 0.577, MLP-64 0.647, CNN **0.990**. With 300: 0.783, 0.813 and **1.000**. With 600: 0.950, 0.957, 1.000. With **1,200**: 0.983, 0.980, 1.000. All models reach train accuracy 1.000 (or 0.997), so the difference is entirely in test accuracy. The CNN is nearly perfect from 150 pictures with 2,533 parameters; the dense networks need about **8 times more pictures** to come within two points, and the larger MLP, with 6.6 times the parameters, helps a little at small sizes (0.647 against 0.577 with 150 pictures) and not at all by 1,200 pictures (0.980 against 0.983). Shifting the training pictures made the dense network *learn* every position separately, which costs data; the CNN only needed to see the pattern once.

   Try also (optional):

   - Add the number of training pictures needed by each model to reach 0.95 test accuracy (read it from the table) and compare.
   - Train the CNN on shift-3 pictures and test at shift 0: does it forget the centre?
   - Try 10 epochs instead of 30 for the MLP-64 with 1,200 pictures: does it need the epochs, too?

   *Exercise 5.4:* From the table (the list `rows` holds `(model, parameters, pictures, train accuracy, test accuracy)`), read off the CNN's test accuracy with 300 pictures, the MLP-64's test accuracy with 1,200 pictures, and the ratio of the parameters of the MLP-64 to those of the CNN (one decimal). Why it matters: what a model costs is parameters, data and training time; a good structure saves all three.

```python
cnn_300 = None       # YOUR CODE: the CNN's test accuracy with 300 training pictures (read it from `rows`)
mlp64_1200 = None    # YOUR CODE: the MLP-64's test accuracy with 1200 training pictures
mlp64_params_ratio = None   # YOUR CODE: how many times more parameters does MLP-64 have than the CNN? (rounded to 1 decimal)
if None in (cnn_300, mlp64_1200, mlp64_params_ratio):
    print("Fill in your three values above, then run this cell again.")
else:
    ref = {(r[0], r[2]): r[4] for r in rows}
    check("cnn_300 matches the table", abs(cnn_300 - ref[("CNN", 300)]) < 0.005, "find the row with model CNN and 300 pictures")
    check("mlp64_1200 matches the table", abs(mlp64_1200 - ref[("MLP-64", 1200)]) < 0.005, "find the row with model MLP-64 and 1200 pictures")
    check("MLP-64 has about 6.6 times the CNN's parameters", abs(mlp64_params_ratio - 16773 / 2533) < 0.15, "16773 / 2533")
```

   *What it means.* The CNN with 300 pictures reaches 1.000, the MLP-64 needs 1,200 pictures to reach 0.980, and it has 6.6 times as many parameters. The cost of ignoring structure is paid in parameters and data.

5. **Step 5.5. Wrong-shape errors on purpose.** Most CNN bugs are shape bugs, and PyTorch reports them with long messages. It is much better to meet them here, on purpose, than at midnight before a deadline. The cell builds three broken networks, runs each on a batch of 8 pictures, and prints the first line of the error: (1) no `Flatten` between the conv part and the `Linear` layer, (2) a first conv layer expecting 3 channels when a grey picture has 1, (3) a `Linear` layer sized for 8x8 maps although two poolings leave 4x4. Then it runs the fixed versions. The printed message differs slightly between PyTorch versions; read the *numbers* in it.

```python
def try_it(label, fn):
    """Run fn and print the first line of the error, if there is one."""
    try:
        out = fn(); print(f"{label}: worked, output shape {tuple(out.shape)}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")

xb = torch.zeros(8, 1, 16, 16)                                  # a batch of 8 pictures

# mistake 1: no Flatten between the conv part and the Linear layer
bad1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Linear(8 * 8 * 8, 5))
try_it("1. forgot Flatten      ", lambda: bad1(xb))
# mistake 2: the first conv layer expects 3 channels but a grey picture has 1
bad2 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
try_it("2. wrong in_channels   ", lambda: bad2(xb))
# mistake 3: the Linear layer was sized for 8x8 maps, but two poolings leave 4x4
bad3 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16 * 8 * 8, 5))
try_it("3. wrong Linear input  ", lambda: bad3(xb))

# the fixes
fix1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8 * 8 * 8, 5))
fix2 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
fix3 = SmallCNN()
try_it("1. fixed (Flatten)     ", lambda: fix1(xb))
try_it("2. fixed (in_channels 1)", lambda: fix2(xb))
try_it("3. fixed (16*4*4)      ", lambda: fix3(xb))
assert fix1(xb).shape == (8, 5) and fix3(xb).shape == (8, 5)
# a trick for mistake 3: print the shape before Flatten and read the number off
print("shape before Flatten:", tuple(nn.Sequential(*list(fix3.children())[:6])(xb).shape))
```

   **What you should see.** In the PyTorch version used to write this lab, error 1 is a `RuntimeError` of the form "mat1 and mat2 shapes cannot be multiplied (512x8 and 512x5)": the Linear layer got a 4-D tensor and read its last dimension (8) as the features. Error 2 says the weight has size `[8, 3, 3, 3]` and expected 3 channels but got 1. Error 3 says "(8x256 and 1024x5)": the flat vector has **256** numbers per picture but the Linear layer was built for 16*8*8 = **1,024**. The fixed versions print `(8, 5)`, `(8, 8, 16, 16)` and `(8, 5)`, and the last line shows the shape before `Flatten` as `(8, 16, 4, 4)`: read `16*4*4` off it.

   Try also (optional):

   - Provoke a fourth error: pass a picture without the batch dimension, `torch.zeros(1, 16, 16)`, into `cnn`. Read the message.
   - Provoke a fifth: change `MaxPool2d(2)` to `MaxPool2d(3)` in one place. Which Linear size do you need now? (Hint: out_size.)
   - Pass a 3-channel input to `SmallCNN` and fix the model, not the input.

   *Exercise 5.5:* Write a debugging checklist of **five lines** for shape errors in a CNN. Each line must say what to check or print and why, with at least six words. Why it matters: the checklist is the page you will reread the next time a network does not run.

```python
checklist = [
    # "my first check when a CNN throws a shape error", ... (five lines)
]
for i, line in enumerate(checklist, 1):
    print(f"{i}. {line}")
if len(checklist) < 5:
    print(f"You have {len(checklist)} lines; add {5 - len(checklist)} more, then run this cell again.")
else:
    check("five or more lines", len(checklist) >= 5, "")
    check("every line has at least 6 words", all(len(str(l).split()) >= 6 for l in checklist), "be concrete: name the shape or the layer")
    RESULTS["p5_checklist"] = checklist
```

   *What it means.* A good checklist names the habit (print the shape after every layer), the rule behind each error (Flatten before Linear, in_channels equals the incoming channels) and the arithmetic (in_features of the first Linear layer is channels times height times width after the last pool).

6. **Step 5.6. Optional stretch: batch normalisation and dropout.** Two common additions are **batch normalisation** (`nn.BatchNorm2d`, rescales each channel's activations per batch, as in the lesson on training deep networks) and **dropout** (`nn.Dropout`, randomly zeroes a fraction of the numbers during training so that the network cannot lean on any single one). Add both to `SmallCNN` and train on a harder setting (noise 0.4, shift 2, 300 training pictures) to see if they help. Report the result honestly: with a dataset this small a difference of one or two points is within the noise.

```python
class BNCNN(nn.Sequential):
    """SmallCNN with batch normalisation after each conv and dropout before the last layer."""
    def __init__(self, n_classes=5, c1=8, c2=16, size=16, p_drop=0.25):
        s = size // 4
        super().__init__(OrderedDict([
            ("conv1", nn.Conv2d(1, c1, 3, padding=1)), ("bn1", nn.BatchNorm2d(c1)), ("relu1", nn.ReLU()), ("pool1", nn.MaxPool2d(2)),
            ("conv2", nn.Conv2d(c1, c2, 3, padding=1)), ("bn2", nn.BatchNorm2d(c2)), ("relu2", nn.ReLU()), ("pool2", nn.MaxPool2d(2)),
            ("flatten", nn.Flatten()), ("drop", nn.Dropout(p_drop)), ("fc", nn.Linear(c2 * s * s, n_classes))]))

# a harder setting where regularisation could matter: more noise, shapes shifted by 2
Xa, ya = make_shapes(300, 16, noise=0.4, shift=2, seed=40)
Xb_, yb_ = make_shapes(300, 16, noise=0.4, shift=2, seed=41)
res = {}
for name, build in [ ("plain CNN", SmallCNN), ("CNN + batchnorm + dropout", BNCNN) ]:
    torch.manual_seed(0); m = build()
    h = train_classifier(m, Xa, ya, Xb_, yb_, epochs=30, lr=0.01, seed=0)
    res[name] = h
    print(f"{name:26s} params {count_params(m):5d}   train {h['train_acc'][-1]:.3f}   test {h['test_acc'][-1]:.3f}   gap {h['train_acc'][-1] - h['test_acc'][-1]:+.3f}")
RESULTS["p5_stretch"] = {k: round(v["test_acc"][-1], 3) for k, v in res.items()}
```

   **What you should see.** The plain CNN (2,533 parameters) reaches train **1.000**, test **0.963** (gap +0.037). The version with batch norm and dropout (2,581 parameters) reaches train 1.000, test **0.980** (gap +0.020). That is a gain of 1.7 points on 300 test pictures, which is about 5 pictures: suggestive, not conclusive. To decide you would repeat with several seeds, as you did in Week 3.

   Try also (optional):

   - Run both models with three different seeds and report the mean and the range of the test accuracy.
   - Raise the dropout probability to 0.5. Does the gap shrink further, and what happens to the training accuracy?
   - Remove the dropout and keep batch norm: which of the two caused the gain?

   **Before you move on (Part 5).** Check that you can say each of these in your own words: the shape after a conv layer follows the output-size formula; a conv layer has `k*k*C_in*C_out + C_out` parameters and does not depend on the picture size; a CNN and an MLP with the same number of parameters tie on centred shapes and differ when the shapes move; the CNN is more robust, not invariant; and the three typical shape errors come from a missing `Flatten`, a wrong `in_channels` and a wrong `Linear` input size.

7. **Imagine first, revisited.** Run the next block to compare your Imagine-first prediction with the facts. Compare your Imagine-first guess with the measured gap. In a markdown cell write whether you were right about who wins, and by how much you were off. If you expected the smaller network to lose, which fact about how a convolution uses its weights did you not count?

```python
print("Your guess:", guess_winner if guess_winner else "(none)", "wins by about", guess_gap, "points at shift 3")
gap3 = 100 * (RESULTS["p5_shift_cnn"][3] - RESULTS["p5_shift_mlp"][3])
print(f"Measured at shift 3 (trained on centred shapes): CNN {RESULTS['p5_shift_cnn'][3]:.3f}, MLP {RESULTS['p5_shift_mlp'][3]:.3f}  ->  the CNN is ahead by {gap3:.1f} points")
print("Both models have about 2,500 parameters, so the difference is not about size.")
RESULTS["p5_guess"] = (guess_winner, guess_gap)
```

### Part 6. Look inside the network (20 min)

*Lesson section: "Stacking" and "Training".* A trained network is not a black box if you ask it the right questions. In this part you look at the 8 filters of the first layer of the CNN you trained in Part 5, at the feature maps they produce, at which part of a picture the network relies on (by hiding patches and by taking a gradient), and at what happens when there is too little training data. Use the `cnn` from Step 5.2; if you restarted the notebook, run Part 5 again first. ==The first layer of a CNN is a bank of small filters that the network designed for itself, and you can read them like the hand-made filters of Part 3.==

**Imagine first.** Before training, the 8 filters of the first layer were small random numbers. After training on five kinds of shape (horizontal bar, vertical bar, cross, square, diagonal), what will they look like? Pick the best description and write it in `guess_filters`: `"noise"` (still random), `"blobs"` (smooth patches), `"edges and bars"` (lines at certain angles) or `"copies of the shapes"` (the filter looks like a whole bar or square). Remember that the filter has only 3x3 numbers. You compare in Step 6.1.

```python
guess_filters = ""     # YOUR TURN: after training, will the 8 first-layer filters look like "noise", "blobs", "edges and bars" or "copies of the shapes"?
print("Your guess:", guess_filters if guess_filters else "(nothing yet)")
```

1. **Step 6.1. The learned filters.** The weight tensor of the first convolution has shape `(8, 1, 3, 3)`: 8 filters, 1 input channel, a 3x3 window (the four numbers of Step 2.4). Plot the eight 3x3 filters as small heat maps with one common colour scale (red = positive weight, blue = negative), so that their strengths can be compared. Then measure each filter's **correlation** with four reference kernels: the Sobel x and Sobel y edge filters of Part 3 (written out again here), and a horizontal-bar and a vertical-bar detector. A correlation of +1 means the same pattern, -1 the opposite pattern (a detector for dark-on-light instead of light-on-dark), 0 unrelated.

```python
# the trained CNN of Step 5.2 (centred shapes). If you restarted the notebook, run Part 5 first.
W1 = cnn.conv1.weight.detach().numpy()[:, 0]            # shape (8, 3, 3): eight 3x3 filters (one input channel)
print("first-layer filter bank:", cnn.conv1.weight.shape, "(filters, input channels, height, width)")

fig, axes = plt.subplots(1, 8, figsize=(12, 1.9))
vmax = np.abs(W1).max()                                  # one colour scale for all eight, so sizes can be compared
for i, a in enumerate(axes):
    a.imshow(W1[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax); a.set_title(f"filter {i}", fontsize=8); a.axis("off")
plt.tight_layout(); plt.show()

templates = {
    "Sobel x (vertical edge)":   np.array([ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ], float),
    "Sobel y (horizontal edge)": np.array([ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ], float),
    "horizontal bar":            np.array([ [-1, -1, -1], [2, 2, 2], [-1, -1, -1] ], float),
    "vertical bar":              np.array([ [-1, 2, -1], [-1, 2, -1], [-1, 2, -1] ], float),
}
def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
print(f"{'filter':7s}" + "".join(f"{n[:14]:>16s}" for n in templates) + "   best match")
best_names = []
for i in range(8):
    cs = {n: corr(W1[i], t) for n, t in templates.items()}
    best = max(cs, key=lambda n: abs(cs[n])); best_names.append(best)
    print(f"{i:<7d}" + "".join(f"{v:16.2f}" for v in cs.values()) + f"   {best} ({cs[best]:+.2f})")
RESULTS["p6_best_matches"] = best_names
assert W1.shape == (8, 3, 3)
```

   **What you should see.** The weights have shape `torch.Size([8, 1, 3, 3])`. The strongest matches (best template, correlation): filter 0 vertical bar (+0.67), filter 1 horizontal bar (-0.71), filter 2 Sobel y (+0.47), filter 3 horizontal bar (+0.37), filter 4 **Sobel y (-0.91)**, filter 5 Sobel x (+0.47), filter 6 Sobel y (+0.62), filter 7 Sobel y (-0.47). So the filters are not noise any more: several correlate clearly with a line or edge template, one (filter 4) is almost exactly a horizontal-edge detector (with the signs reversed), and some are weaker mixtures. They are not copies of the hand-made Sobel filters, because the network built what separates these five shapes, not what separates objects in photographs. Their pattern will differ from run to run and between PyTorch versions; the idea (edges and bars, not noise, not whole shapes) should not.

   Try also (optional):

   - Plot the 8 filters of an *untrained* `SmallCNN()` next to the trained ones. How different do they look?
   - Add the diagonal detector `[ [1, 0, 0], [0, 1, 0], [0, 0, 1] ]` to `templates` and see which filters match it.
   - Print `W1.std()` before and after training (use a fresh `SmallCNN()` for the before): did the weights grow?

   *Exercise 6.1:* Count how many of the 8 filters have an absolute correlation above 0.7 with at least one of the four templates, and write two sentences on whether the learned filters look like Sobel edges, like bars, or like something else. Why it matters: a filter bank you can name is a first step to trusting a model; one you cannot name is a reason to test more.

```python
n_filters_edge_like = None    # YOUR CODE: how many of the 8 filters have |correlation| above 0.7 with at least one of the four templates?
my_reading = ""              # YOUR ANSWER: in two sentences, do the learned filters look like Sobel edges, like bars, or like something else?
if n_filters_edge_like is None:
    print("Fill in `n_filters_edge_like` (and write `my_reading`), then run this cell again.")
else:
    true_n = sum(1 for i in range(8) if max(abs(corr(W1[i], t)) for t in templates.values()) > 0.7)
    check("your count matches the table", n_filters_edge_like == true_n, f"count the rows whose best |correlation| exceeds 0.7")
    check("your reading has at least 15 words", len(my_reading.split()) >= 15, "describe what you see")
```

   *What it means.* Two filters (1 and 4) pass the threshold 0.7. The rest are weaker, mixed patterns: the network does not need clean edges, only features that help separate five shapes.

2. **Step 6.2. Feature maps: what each filter responds to.** A **feature map** is the output of one filter over the whole picture (after ReLU). Take one picture of each class (shift 0, small noise), push it through the first convolution and ReLU, and plot the 8 maps next to the input: 5 rows (classes) by 9 columns (input + 8 filters). Then summarise: for every filter and class, the mean activation over all pictures of that class. Look at the table by rows: does a filter prefer one shape?

```python
# one clean-ish picture of each class (shift 0, small noise), through conv1 + ReLU
Xv, yv = make_shapes(100, 16, noise=0.1, shift=0, seed=50)
firsts = [int(np.where(yv == k)[0][0]) for k in range(5)]
with torch.no_grad():
    maps = F.relu(cnn.conv1(torch.as_tensor(Xv[firsts])))[:, :, :, :].numpy()       # shape (5, 8, 16, 16)

fig, axes = plt.subplots(5, 9, figsize=(12, 7))
for k in range(5):
    axes[k, 0].imshow(Xv[firsts[k], 0], cmap="gray_r", vmin=0, vmax=1); axes[k, 0].set_ylabel(SHAPE_NAMES[k], fontsize=8)
    for f in range(8):
        axes[k, f + 1].imshow(maps[k, f], cmap="viridis")
    for a in axes[k]: a.set_xticks([]); a.set_yticks([])
for f in range(8): axes[0, f + 1].set_title(f"filter {f}", fontsize=8)
axes[0, 0].set_title("input", fontsize=8)
plt.tight_layout(); plt.show()

# how strongly does each filter respond to each class? (mean activation over 20 pictures of the class)
with torch.no_grad():
    allmaps = F.relu(cnn.conv1(torch.as_tensor(Xv))).numpy()
strength = np.array([ [allmaps[yv == k, f].mean() for k in range(5)] for f in range(8) ])      # (filters, classes)
print(f"{'filter':7s}" + "".join(f"{n[:10]:>12s}" for n in SHAPE_NAMES) + "   strongest on")
for f in range(8):
    print(f"{f:<7d}" + "".join(f"{v:12.2f}" for v in strength[f]) + f"   {SHAPE_NAMES[int(np.argmax(strength[f]))]}")
RESULTS["p6_filter_pref"] = [SHAPE_NAMES[int(np.argmax(strength[f]))] for f in range(8)]
assert allmaps.shape == (100, 8, 16, 16) and (allmaps >= 0).all(), "ReLU output is never negative"
```

   **What you should see.** The maps have the same size as the input (16x16) because of the padding, and each bright part is a place where that filter found its pattern. In the table, filters **1 and 3 respond about equally to every shape** (0.33 to 0.38), and six filters (0, 2, 4, 5, 6 and 7) are strongest on **squares**, for instance filter 2 with 0.49 on squares against 0.27 on bars. This is a size effect: a square outline has 32 inked pixels, a cross 17, a bar or a diagonal only 9, so a filter that adds up local ink responds more to a square. No filter is dead (the weakest, filter 4, still has 0.10 on squares). Layer 1 reports local pieces; it does not name shapes.

   Try also (optional):

   - Plot the feature maps of a *noisy* picture (noise 0.5). Which filters keep their clean lines and which turn grainy?
   - Compute the table on pictures shifted by 3 pixels. Does any row change its favourite class?
   - Look at filter 4's maps: where is it bright, and does that match its Sobel y correlation of -0.91?

   *Exercise 6.2:* Find the filter with the largest mean activation on squares, count for how many of the 8 filters the strongest class is "square", and write one or two sentences: are the first-layer filters specialists for one shape or generalists, and what does that say about what layer 1 does? Why it matters: it explains why networks are *stacked*: no single early filter can decide, the decision is built from many local reports.

```python
filter_best_square = None   # YOUR ANSWER: the number (0-7) of the filter with the largest mean activation on squares
n_prefer_square = None     # YOUR ANSWER: for how many of the 8 filters is 'square' the class with the largest mean activation?
generic_note = ""          # YOUR ANSWER: one or two sentences: are the first-layer filters specialists for one shape, or generalists? what does that say about what layer 1 does?
if None in (filter_best_square, n_prefer_square):
    print("Fill in `filter_best_square` and `n_prefer_square` (and write `generic_note`), then run this cell again.")
else:
    check("filter_best_square is the argmax of the 'square' column", filter_best_square == int(np.argmax(strength[:, 3])), "look at the column 'square'")
    check("n_prefer_square matches the table", n_prefer_square == int(sum(np.argmax(strength[f]) == 3 for f in range(8))), "count the rows whose strongest class is square")
    check("your note has at least 15 words", len(generic_note.split()) >= 15, "write a real answer")
```

   *What it means.* Filter 2 is strongest on squares and 6 of the 8 filters have squares as their favourite class, but not because they are square detectors: squares simply have the most ink. The first layer is a set of generalists; shape identity is decided later.

3. **Step 6.3. What does the network look at? An occlusion map.** To ask which part of a picture the network relies on, hide a part and see what happens. The **occlusion map** slides a 4x4 patch over the picture, paints it with the background colour (the value 0), and records, for each patch position, the **drop in the probability** of the class (the probability on the untouched picture minus the probability with the patch). A big drop means "this region mattered". We use pictures with noise 0.3 so that the network is not 100 % sure of everything, and a noise-free copy of the same shapes to know where the shape is. Then we compare the average drop of patches that cover part of the shape with those that miss it completely.

```python
def occlusion_map(model, img, cls, patch=4, fill=0.0):
    """Slide a patch x patch square (painted with the value fill; 0 is the background colour) over the picture (img has shape (1, H, W)).
    Returns, for every position, how much the probability of class cls drops, and the probability on the untouched picture."""
    model.eval()
    H, W = img.shape[-2:]
    with torch.no_grad():
        base = float(torch.softmax(model(torch.as_tensor(img)[None]), 1)[0, cls])
        out = np.zeros((H - patch + 1, W - patch + 1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                occ = img.copy(); occ[:, i:i + patch, j:j + patch] = fill
                out[i, j] = base - float(torch.softmax(model(torch.as_tensor(occ)[None]), 1)[0, cls])
    return out, base

# pictures with noise 0.3 (so the network is not 100 % sure of everything); Xc has the SAME shapes without noise and tells us where the shape is
Xo, yo = make_shapes(100, 16, noise=0.3, shift=0, seed=50)
Xc, yc = make_shapes(100, 16, noise=0.0, shift=0, seed=50)
picks = [Xo[i] for i in firsts]                                    # one noisy picture per class
assert (yo == yv).all() and (yc == yv).all(), "same seed, same shapes"
fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
inside, outside = [], []
for k in range(5):
    i = firsts[k]
    om, base = occlusion_map(cnn, Xo[i], k)
    axes[0, k].imshow(Xo[i, 0], cmap="gray_r", vmin=0, vmax=1); axes[0, k].set_title(f"{SHAPE_NAMES[k]}\np = {base:.2f}", fontsize=8)
    axes[1, k].imshow(om, cmap="magma", vmin=0, vmax=1); axes[1, k].set_title(f"largest drop {om.max():.2f}", fontsize=8)
    for a in axes[:, k]: a.axis("off")
    for r in range(om.shape[0]):                      # does a patch that covers a shape pixel hurt more than one that does not?
        for c in range(om.shape[1]):
            (inside if Xc[i, 0, r:r + 4, c:c + 4].max() > 0.5 else outside).append(om[r, c])
plt.tight_layout(); plt.show()
print(f"mean drop of patches that cover part of the shape:   {np.mean(inside):.3f}   ({len(inside)} positions)")
print(f"mean drop of patches that miss the shape completely: {np.mean(outside):.3f}   ({len(outside)} positions)")
RESULTS["p6_occlusion"] = {"inside": round(float(np.mean(inside)), 3), "outside": round(float(np.mean(outside)), 3)}
assert np.mean(inside) > np.mean(outside), "hiding part of the shape should hurt more than hiding background"
```

   **What you should see.** The probabilities on the untouched pictures are 1.00 for bars, square and diagonal and 0.89 for the cross. The largest drops are **0.89** for the cross and **0.94** for the square; for the bars and the diagonal the largest drop is **0.00**. Patches that cover part of the shape drop the probability by **0.147** on average (376 positions); patches that miss the shape completely change it by **-0.012** (469 positions), i.e. not at all. Why zero for the bars? A bar with 9 pixels still looks like a bar when 4 of them are hidden, and the network recognises a short line anywhere: the evidence is redundant. A cross loses its centre and becomes a bar; a square loses a corner and becomes something else. What matters is where the shape's *joints* are.

   Try also (optional):

   - Try `patch=6` and then `patch=8`: does a bar finally show a drop, and at which patch size? (Measure it; a bar has only 9 pixels, so a bigger patch can erase most of it.)
   - Occlude the picture with the *wrong* class: ask for the probability of 'square' while the picture shows a cross.
   - Show the maps on a picture with shift 3: do the bright regions move with the shape?

   *Exercise 6.3:* Repeat the inside/outside measurement but paint the patch grey (`fill=0.5`) instead of with the background colour. Store the two averages in `inside_grey` and `outside_grey`, and write one or two sentences about what changed and what it says about the occlusion method itself. Why it matters: an explanation method can change the thing it explains. The patch is a new object in the picture, and the network reacts to it.

```python
inside_grey = None     # YOUR CODE: repeat the inside / outside measurement of the cell above, but paint the patch GREY (fill=0.5) instead of blank
outside_grey = None    # YOUR CODE: the mean drop of the patches that miss the shape, with the grey patch
grey_note = ""         # YOUR ANSWER: one or two sentences: what changed, and what does it say about the occlusion method itself?
if inside_grey is None or outside_grey is None:
    print("Fill in `inside_grey` and `outside_grey` (and write `grey_note`), then run this cell again.")
else:
    ins_g, out_g = [], []
    for k in range(5):
        om, _ = occlusion_map(cnn, picks[k], k, fill=0.5)
        for r in range(13):
            for c in range(13):
                (ins_g if Xc[firsts[k], 0, r:r + 4, c:c + 4].max() > 0.5 else out_g).append(om[r, c])
    check("inside_grey matches", abs(inside_grey - np.mean(ins_g)) < 0.005, "same loop, fill=0.5")
    check("outside_grey matches", abs(outside_grey - np.mean(out_g)) < 0.005, "same loop, fill=0.5")
    check("your note has at least 12 words", len(grey_note.split()) >= 12, "write a real sentence")
```

   *What it means.* With the grey patch the averages are 0.070 inside and 0.112 outside: the order is reversed. A grey blob looks like part of a shape, so it confuses the network wherever it sits; the method has a side effect, and the fill value must be chosen and reported.

4. **Step 6.4. Gradient saliency.** A faster way to ask "which pixels matter?" is to use the gradient you computed in Part 4, but now with respect to the *input picture*: how much does the score of the predicted class change if this pixel changes a little? PyTorch gives it with `requires_grad=True` on the input and one `backward()` on the class score. The absolute value of this gradient, as an image, is a **saliency map**. To compare it with the occlusion map we add up the saliency over every 4x4 window (so both maps have 13x13 values) and compute the correlation.

```python
def saliency(model, img, cls=None):
    """|d (score of class cls) / d pixel|, shape (H, W). If cls is None use the predicted class."""
    model.eval()
    xin = torch.as_tensor(img)[None].clone().requires_grad_(True)
    score = model(xin)[0]
    c = int(score.argmax()) if cls is None else cls
    score[c].backward()
    return xin.grad[0, 0].abs().numpy(), c

fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
cors = []
for k in range(5):
    i = firsts[k]
    sal, c = saliency(cnn, Xo[i])
    om, _ = occlusion_map(cnn, Xo[i], c)
    box = np.array([ [sal[r:r + 4, q:q + 4].sum() for q in range(13)] for r in range(13) ])       # saliency added up over each 4x4 window: comparable to the occlusion map
    cors.append(float(np.corrcoef(box.ravel(), om.ravel())[0, 1]))
    axes[0, k].imshow(Xo[i, 0], cmap="gray_r", vmin=0, vmax=1); axes[0, k].set_title(f"{SHAPE_NAMES[k]}\npredicted: {SHAPE_NAMES[c]}", fontsize=8)
    axes[1, k].imshow(sal, cmap="magma"); axes[1, k].set_title(f"saliency, corr. with occlusion {cors[-1]:.2f}", fontsize=7)
    for a in axes[:, k]: a.axis("off")
plt.tight_layout(); plt.show()
print("correlation between the occlusion map and the (window-summed) saliency map, per class:", [round(c, 2) for c in cors])
RESULTS["p6_saliency_corr"] = round(float(np.mean(cors)), 2)
assert sal.shape == (16, 16)
```

   **What you should see.** Look at the saliency images: some background pixels are bright as well, not only the shape. The correlation between the occlusion map and the window-summed saliency is `[0.14, 0.16, 0.69, 0.10, 0.43]` for the five classes, about **0.30** on average: weak for the two bars (0.14 and 0.16, where the occlusion map is empty) and for the square (0.10), medium for the diagonal (0.43) and strong for the cross (0.69). The two methods ask different questions: occlusion asks "what if this region disappeared?", saliency asks "what if this pixel changed a tiny bit?"; they agree only when the network responds smoothly. A comparison in one sentence: occlusion is slower but easier to read, saliency is fast but noisy.

   Try also (optional):

   - Smooth the saliency map with a 3x3 mean filter (use `conv2d_np` and the kernel `np.ones((3, 3)) / 9`) and see if the correlation rises.
   - Compute the saliency for a class that is *not* the prediction.
   - Average the saliency over the 20 pictures of one class. Where is the average bright?

   *Exercise 6.4:* For the cross picture, compute what fraction of the total saliency lies on pixels that belong to the shape (use the noise-free copy `Xc` as a mask), and compare it with the fraction of the picture the shape occupies. Write one sentence comparing the occlusion map and the saliency map. Why it matters: a fraction is a number you can put in a report; "it looks about right" is not.

```python
sal_on_shape = None     # YOUR CODE: for the 'cross' picture (picks[2]) what fraction of the total saliency lies on pixels that belong to the shape? (use Xc[firsts[2], 0] > 0.5 as the shape mask)
one_sentence = ""       # YOUR ANSWER: one sentence comparing the occlusion map and the saliency map
if sal_on_shape is None:
    print("Fill in `sal_on_shape` (and write `one_sentence`), then run this cell again.")
else:
    s_, _ = saliency(cnn, picks[2]); mask = Xc[firsts[2], 0] > 0.5
    check("your fraction matches", abs(sal_on_shape - s_[mask].sum() / s_.sum()) < 0.01, "sum of saliency under the mask divided by the total")
    check("the shape has few pixels but a large share", sal_on_shape > mask.mean(), "compare with the fraction of pixels in the mask")
    check("your sentence has at least 10 words", len(one_sentence.split()) >= 10, "write a sentence")
```

   *What it means.* About 22 % of the saliency lies on the 17 shape pixels, which are 6.6 % of the picture: the network is clearly paying more attention to the shape than to the background, but most of the saliency is still on the background.

5. **Step 6.5. Learning curves and overfitting.** One more look inside: what happens when the network has too little to learn from. Train the same `SmallCNN` on 40 and on 400 pictures from a harder setting (noise 0.3, shift 2) for 60 epochs, and test on 300 other pictures. A big gap between the training accuracy and the test accuracy is the signature of **overfitting**: the model has memorised its training pictures, including their noise. It is the picture-world version of the dense network that memorised positions in Part 1.

```python
# the same CNN with 40 or with 400 training pictures, on a harder setting (noise 0.3, shift 2); 60 epochs
Xt, yt_ = make_shapes(300, 16, noise=0.3, shift=2, seed=61)
curves = {}
for n in [40, 400]:
    Xs, ys = make_shapes(n, 16, noise=0.3, shift=2, seed=60)
    torch.manual_seed(0); m = SmallCNN()
    curves[n] = train_classifier(m, Xs, ys, Xt, yt_, epochs=60, lr=0.01, seed=0)
    print(f"{n:4d} pictures:  train {curves[n]['train_acc'][-1]:.3f}   test {curves[n]['test_acc'][-1]:.3f}   gap {curves[n]['train_acc'][-1] - curves[n]['test_acc'][-1]:+.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, n in zip(ax, [40, 400]):
    a.plot(curves[n]["train_acc"], label="train"); a.plot(curves[n]["test_acc"], label="test"); a.set_title(f"{n} training pictures"); a.set_xlabel("epoch"); a.set_ylim(0, 1.05)
ax[0].set_ylabel("accuracy"); ax[0].legend(); plt.tight_layout(); plt.show()
RESULTS["p6_gap"] = {n: round(curves[n]["train_acc"][-1] - curves[n]["test_acc"][-1], 3) for n in curves}
assert curves[40]["train_acc"][-1] >= curves[40]["test_acc"][-1], "a small training set is memorised"
```

   **What you should see.** With **40** pictures: train **1.000**, test **0.813**, a gap of +0.187. With **400**: train 1.000, test **0.993**, a gap of +0.007. In both cases the training accuracy is perfect; only the test accuracy tells the difference. Ten times more data closed almost the whole gap: the curves for 400 pictures lie on top of each other, the curves for 40 do not. Look at the two plots and describe the distance between the train line and the test line in each.

   Try also (optional):

   - Add 100 and 200 to the list of sizes and plot the test accuracy against the number of training pictures.
   - Train the 40-picture run with `BNCNN` from Step 5.6. Does the gap shrink?
   - Use data augmentation: make 40 pictures with `shift=3` from different seeds, and train on those plus the original 40.

   *Exercise 6.5:* Compute the train-minus-test gap for both runs, say which is larger, and name two things you could do to shrink the gap when only 40 pictures are available. Why it matters: in the real world, labelled pictures cost money and time; this is the situation you will most often be in.

```python
gap_40 = None     # YOUR CODE: train accuracy minus test accuracy for the 40-picture run (at the last epoch)
gap_400 = None    # YOUR CODE: the same for 400 pictures
remedy = ""       # YOUR ANSWER: name two things you could do to shrink the gap with only 40 pictures
if gap_40 is None or gap_400 is None:
    print("Fill in `gap_40` and `gap_400` (and write `remedy`), then run this cell again.")
else:
    check("gap_40 matches", abs(gap_40 - RESULTS["p6_gap"][40]) < 0.005, "curves[40]['train_acc'][-1] - curves[40]['test_acc'][-1]")
    check("gap_400 matches", abs(gap_400 - RESULTS["p6_gap"][400]) < 0.005, "curves[400]['train_acc'][-1] - curves[400]['test_acc'][-1]")
    check("the small set has the larger gap", gap_40 > gap_400, "less data, more memorising")
    check("your remedy has at least 8 words", len(remedy.split()) >= 8, "name two things")
```

   *What it means.* The gaps are +0.187 and +0.007. Remedies: more (or augmented) data, dropout or weight decay, a smaller network, early stopping.

   **Before you move on (Part 6).** Check that you can say each of these in your own words: the first-layer filters of a trained CNN are small line- and edge-like patterns that the network designed for itself; a feature map shows where a filter finds its pattern, and early filters are generalists; occlusion measures what happens when a region disappears, saliency measures a local gradient, and both depend on choices (patch, fill, class); and a large train-test gap with little data is overfitting.

6. **Imagine first, revisited.** Run the next block to compare your Imagine-first prediction with the facts. Compare your Imagine-first guess with what you saw in Step 6.1. In a markdown cell write whether "edges and bars" is a fair description of your filters, and which filter was the clearest example.

```python
print("Your guess:", guess_filters if guess_filters else "(none)")
print("Best template for each learned filter:", RESULTS["p6_best_matches"])
RESULTS["p6_guess"] = guess_filters
```

### Part 7. Design your own CNN and reflect (15 min)

*Lesson section: the whole of Week 4.* In this last part you use what you know. You describe a convolutional network for a problem from your own life, check its shapes and parameter count with a small helper, build and train a variant on a harder version of the shapes, stress-test it on pictures it has never seen, print all your key numbers, and answer four reflection questions. ==Designing a CNN means choosing, layer by layer, what the filters should detect, how the picture shrinks, and what you are willing to throw away.==

1. **Step 7.1. Imagination design: a CNN for a problem you know.** Pick a real picture-recognition problem from your school or community. Some ideas: reading the house number on a gate, recognising ripe fruit at the market, sorting recyclables from a phone photo, detecting a flooded road from a jeepney window, telling healthy and spotted leaves. Fill in a **spec**: the input (size and channels), what the first layers should detect, the layer list, the outputs, the loss, how you will test, which augmentation you will use, and one risk. The helper `shape_table(input_shape, layers)` uses the output-size formula of Part 2 to follow a picture through your layer list and add up the parameters, so a wrong kernel or stride shows up as an error before you train anything. The first cell shows a worked example (ripe mango) so you can see the level of detail expected.

```python
def out_size(n, k, s=1, p=0):
    """Side length of the output of a k-wide kernel (or pool window) with stride s and padding p on a side of length n (as in Part 2)."""
    return (n + 2 * p - k) // s + 1

def shape_table(input_shape, layers, show=True):
    """Follow a picture of shape (channels, height, width) through a list of layers; print the shape and the parameters after each.
    Layers:  ("conv", filters, k, stride, pad)   ("relu",)   ("pool", k)   ("globalpool",)   ("flatten",)   ("linear", n_out)
    Returns (final shape, total parameters)."""
    shape, total = tuple(input_shape), 0
    if show: print(f"{'layer':26s} {'output shape':18s} {'parameters':>10s}\n{'input':26s} {str(shape):18s} {'':>10s}")
    for layer in layers:
        kind, params = layer[0], 0
        if kind == "conv":
            _, f, k, s, p = layer; c, h, w = shape
            params = k * k * c * f + f; shape = (f, out_size(h, k, s, p), out_size(w, k, s, p))
        elif kind == "pool":
            c, h, w = shape; shape = (c, out_size(h, layer[1], layer[1]), out_size(w, layer[1], layer[1]))
        elif kind == "globalpool":
            shape = (shape[0], 1, 1)
        elif kind == "flatten":
            shape = (int(np.prod(shape)),)
        elif kind == "linear":
            params = shape[0] * layer[1] + layer[1]; shape = (layer[1],)
        elif kind != "relu":
            raise ValueError("unknown layer " + str(layer))
        total += params
        if show: print(f"{str(layer):26s} {str(shape):18s} {params:10d}")
    if show: print(f"{'total':26s} {'':18s} {total:10d}")
    return shape, total

# an example design card: ripe fruit from a phone photo
example_spec = {
    "problem": "Is the mango in the photo unripe, ripe or overripe? A seller at the market takes a phone photo.",
    "input": "colour photo, cropped to the fruit, resized to 64 x 64 pixels, 3 channels (red, green, blue)",
    "first layers detect": "edges and colour blobs first, then skin texture and spots, then parts of the fruit such as the stem end",
    "layers": "3 x (conv 3x3, ReLU), pooling after the first two, then a global pool and one linear layer",
    "classes": "3: unripe, ripe, overripe",
    "loss": "cross-entropy over the three classes",
    "how to test": "300 photos from a different market and a different day than the training photos; accuracy and the confusion between ripe and overripe",
    "augmentation": "random crops, small rotations, brightness changes (markets have shade and sun)",
    "one risk": "the model may learn the colour of the crate or the light, not the fruit",
}
example_layers = [ ("conv", 16, 3, 1, 1), ("relu",), ("pool", 2), ("conv", 32, 3, 1, 1), ("relu",), ("pool", 2), ("conv", 64, 3, 1, 1), ("relu",), ("globalpool",), ("flatten",), ("linear", 3) ]
final_shape, total = shape_table((3, 64, 64), example_layers)
assert final_shape == (3,) and total == 448 + 4640 + 18496 + 195, "16*27+16, 32*144+32, 64*288+64, 64*3+3"

def print_card(spec):
    print("-" * 78)
    for key, value in spec.items():
        print(f"{key.upper():22s}{value}")
    print("-" * 78)
print_card(example_spec)
```

   **What you should see.** The table for the example (a 3x64x64 colour photo, three conv layers with 16, 32 and 64 filters, pooling after the first two, a global pool and a linear layer to 3 classes) has the parameters **448** (16*3*3*3 + 16), **4,640** (32*16*9 + 32), **18,496** (64*32*9 + 64) and **195** (64*3 + 3), a total of **23,779**, and the final shape is `(3,)`. The assert passes. The card prints every field of the spec on its own line. Compare 23,779 with the 150,529,000 parameters of one dense layer on a 224x224 colour picture (Step 1.2): that is the benefit of convolution in one line.

   Try also (optional):

   - Change the first layer to stride 2 and see how much the parameters and the shapes of the rest change.
   - Remove the global pool and use `("flatten",)` directly: what is the size of the Linear layer then?
   - Write the same design for a 28x28 grey picture of a handwritten digit with 10 classes.

   *Exercise 7.1:* Fill in every entry of `my_spec` (at least three words per entry), set `my_input_shape` and replace `my_layers` with your own layer list. Run the cell: the table must follow your picture to a flat output with as many numbers as you have classes, and the check wants at least two conv layers. Why it matters: a design is a promise you can check; this is the check that costs nothing.

```python
my_spec = {
    "problem": "",              # YOUR DESIGN: the real picture problem from your school or community, in one or two sentences
    "input": "",                # size, channels, what a picture is
    "first layers detect": "",  # what should layer 1, layer 2, deeper layers pick out
    "layers": "",               # the layer list in words: kernel sizes, strides, padding, filters
    "classes": "",              # the outputs
    "loss": "",
    "how to test": "",          # on what data, which number; think of the shifted-and-noisy test of Step 5.3
    "augmentation": "",
    "one risk": "",             # one way the model could fail or mislead
}
my_input_shape = (3, 64, 64)
my_layers = list(example_layers)       # YOUR DESIGN: replace with your own list of layers (see the helper's docstring)
def spec_is_filled(spec):
    return all(len(str(v).split()) >= 3 for v in spec.values())

fs, tot = shape_table(my_input_shape, my_layers)
if not spec_is_filled(my_spec):
    print("Fill in every entry of `my_spec` with at least 3 words, then run this cell again. (The table above shows the example layers until you change them.)")
else:
    print_card(my_spec)
    check("at least 2 conv layers", sum(1 for l in my_layers if l[0] == "conv") >= 2, "a CNN with one conv layer is hardly 'deep'")
    check("the last layer is linear and the output is flat", my_layers[-1][0] == "linear" and len(fs) == 1, "flatten (or globalpool then flatten), then linear")
    check("the output has as many numbers as your classes", fs[0] >= 2, "one output per class")
    RESULTS["p7_spec"] = my_spec; RESULTS["p7_design_params"] = tot
```

   *What it means.* The sample answer designs a flooded-road detector with 2 classes: the shape table ends in `(2,)` with 23,714 parameters. Your own design will differ; the Check only tests that the spec is complete and the shapes are consistent.

2. **Step 7.2. Build and train a variant on a harder setting.** Now try one design idea on the shapes: **a global max-pool instead of Flatten**. After the last convolution, `AdaptiveMaxPool2d(1)` keeps only the strongest response of each channel anywhere in the picture, so the Linear layer asks "is this feature somewhere?" and not "is it here?". With three conv layers (16, 32, 32 filters) and batch norm this design has 14,309 parameters. Train it and the Part 5 CNN on a harder setting: noise 0.4, shapes up to 2 pixels from the centre, 800 training pictures and 300 test pictures, 40 epochs. In your own design you replace `build_my_cnn` by something that follows your spec (keep at least two conv layers).

```python
# a harder setting: more noise (0.4) and shapes up to 2 pixels from the centre
Xtr7, ytr7 = make_shapes(800, 16, noise=0.4, shift=2, seed=70)
Xte7, yte7 = make_shapes(300, 16, noise=0.4, shift=2, seed=71)

def build_my_cnn():
    """The design idea of this part: three conv layers with batch norm, and a GLOBAL max-pool instead of Flatten, so only 'is the feature somewhere?' is kept, not 'where?'.
    Replace it with your own design (keep at least two conv layers)."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 5))

torch.manual_seed(0); base_model = SmallCNN()
torch.manual_seed(0); my_cnn = build_my_cnn()
print("parameters:  Part 5 CNN", count_params(base_model), "   my design", count_params(my_cnn))
h_base = train_classifier(base_model, Xtr7, ytr7, Xte7, yte7, epochs=40, lr=0.01, seed=0)
h_my = train_classifier(my_cnn, Xtr7, ytr7, Xte7, yte7, epochs=40, lr=0.01, seed=0)
for name, h in [ ("Part 5 CNN", h_base), ("my design", h_my) ]:
    print(f"{name:11s} train {h['train_acc'][-1]:.3f}   test {h['test_acc'][-1]:.3f}")
RESULTS["p7_train"] = {"base": round(h_base["train_acc"][-1], 3), "my": round(h_my["train_acc"][-1], 3)}
RESULTS["p7_test"] = {"base": round(h_base["test_acc"][-1], 3), "my": round(h_my["test_acc"][-1], 3)}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for name, h in [ ("Part 5 CNN", h_base), ("my design", h_my) ]:
    ax[0].plot(h["loss"], label=name); ax[1].plot(h["train_acc"], "--", label=name + " train"); ax[1].plot(h["test_acc"], label=name + " test")
ax[0].set_title("training loss"); ax[1].set_title("accuracy"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
assert h_my["loss"][-1] < h_my["loss"][0], "my design should learn"
```

   **What you should see.** The Part 5 CNN has 2,533 parameters and the new design 14,309. Both reach train accuracy **1.000**; test accuracy **0.987** for the Part 5 CNN and **0.990** for the new design, a difference of one picture in 300: on this test set the two designs tie. Nothing is wrong with that: on the training distribution the Part 5 CNN was already good. The difference between the designs, if any, should appear when the test pictures differ from the training pictures, which is the next step. The curves show the training loss falling for both and the test accuracy flat near the top.

   Try also (optional):

   - Replace the global max-pool by a global average-pool (`AdaptiveAvgPool2d(1)`): what happens to the test accuracy?
   - Train with `lr=0.001`: how many epochs does the new design need to reach 0.95?
   - Remove the batch-norm layers: is the gap between training and test accuracy larger?

   *Exercise 7.2:* Change **one** thing in the design (more filters, one more conv layer, no batch norm, Flatten instead of the global pool, a different kernel size) by writing `build_variant`, and train it. Write in `variant_note` what you changed and what happened to the test accuracy, with the numbers. Why it matters: one change at a time is the only way to know which change did what.

```python
def build_variant():
    # YOUR DESIGN: change ONE thing in the design above (more filters, one more conv layer, no batch norm, Flatten instead of the global pool ...)
    return build_my_cnn()
variant_note = ""     # YOUR ANSWER: what did you change and what happened to the test accuracy?
torch.manual_seed(0); variant = build_variant()
n_conv = sum(isinstance(m, nn.Conv2d) for m in variant.modules())
h_var = train_classifier(variant, Xtr7, ytr7, Xte7, yte7, epochs=40, lr=0.01, seed=0)
print(f"your variant: {count_params(variant)} parameters, train {h_var['train_acc'][-1]:.3f}, test {h_var['test_acc'][-1]:.3f}")
check("at least two conv layers", n_conv >= 2, "the task asks for a CNN with 2 or more")
check("the variant learns (test above chance 0.2)", h_var["test_acc"][-1] > 0.3, "something is wrong with the architecture or shapes; print the shapes")
if not variant_note.strip():
    print("Write `variant_note` (what you changed and what happened), then run this cell again.")
else:
    check("your note has at least 15 words", len(variant_note.split()) >= 15, "name the change and the numbers")
    RESULTS["p7_variant"] = {"test": round(h_var["test_acc"][-1], 3), "note": variant_note}
```

   *What it means.* With Flatten instead of the global pool the test accuracy stayed at 0.990 (16,709 parameters), so the choice does not matter on this test set; the next step looks at what it does when the test set is harder.

3. **Step 7.3. Stress test and one fix.** A model is only as good as the situations you tested it in. Take the two models you just trained and test them, unchanged, on new pictures with **noise 0.6 and shift 4** (training had 0.4 and 2). Report the drop in accuracy for both. Then try one fix, **data augmentation**: train the new design on pictures that look like the stress test (800 pictures, noise 0.6, shift 4) and test it on the stress set. Name the limits honestly: the fix trains on the same distribution it is tested on, so it shows what is *possible*, not what happens in the wild.

```python
# stress test: the SAME trained models, test pictures with noise 0.6 and shift 4 (never seen)
Xst, yst = make_shapes(300, 16, noise=0.6, shift=4, seed=72)
acc_st = {"Part 5 CNN": accuracy(base_model, Xst, yst), "my design": accuracy(my_cnn, Xst, yst)}
for name, a in acc_st.items():
    before = RESULTS["p7_test"]["base" if name.startswith("Part") else "my"]
    print(f"{name:11s} test (noise 0.4, shift 2): {before:.3f}   stress (noise 0.6, shift 4): {a:.3f}   drop {before - a:+.3f}")

# one fix to try: train on pictures that look like the stress test (data augmentation), same model, same epochs
Xaug, yaug = make_shapes(800, 16, noise=0.6, shift=4, seed=73)
torch.manual_seed(0); fixed = build_my_cnn()
h_fix = train_classifier(fixed, Xaug, yaug, Xst, yst, epochs=40, lr=0.01, seed=0)
print(f"my design trained on noisy, shifted pictures: stress test accuracy {h_fix['test_acc'][-1]:.3f}")
RESULTS["p7_stress"] = {k: round(v, 3) for k, v in acc_st.items()}; RESULTS["p7_stress_fixed"] = round(h_fix["test_acc"][-1], 3)
assert acc_st["my design"] <= RESULTS["p7_test"]["my"] + 0.02, "a harder test set should not be easier"
```

   **What you should see.** On the stress set the Part 5 CNN falls from 0.987 to **0.513** (a drop of 47.4 points) and the new design from 0.990 to **0.610** (a drop of 38.0 points): the global pool helps by about 10 points in this single run, but both models lose a lot. Trained on pictures like the stress set, the new design reaches **0.870** on it. So most of the drop was *distribution shift* (the model had never seen such pictures), which data can repair; the rest, 13 points, is the difficulty of the task itself at noise 0.6.

   Try also (optional):

   - Stress only the noise (0.6, shift 2) and only the shift (0.4, shift 4) and see which hurts more.
   - Train the fix with noise 0.6 and shift 4 but only 200 pictures: how much of the gain remains?
   - Add `nn.Dropout(0.3)` before the final Linear layer of `build_my_cnn` and repeat the stress test.

   *Exercise 7.3:* Compute how many accuracy points your design lost on the stress test (a number from 0 to 100), and write one fix you would try besides augmentation, with the reason you expect it to help. Why it matters: a stress test that only tells you the model is fragile is half a test; the other half is a plan.

```python
stress_drop = None     # YOUR CODE: how many accuracy points did your design lose on the stress test (a number between 0 and 100)?
my_fix = ""            # YOUR ANSWER: one fix you would try besides augmentation, and why you expect it to help
if stress_drop is None:
    print("Fill in `stress_drop` (and write `my_fix`), then run this cell again.")
else:
    true_drop = 100 * (RESULTS["p7_test"]["my"] - acc_st["my design"])
    check("stress_drop matches", abs(stress_drop - true_drop) < 0.5, "100 * (accuracy before - accuracy on the stress set)")
    check("your fix has at least 12 words", len(my_fix.split()) >= 12, "name the fix and say why")
    RESULTS["p7_my_fix"] = my_fix
```

   *What it means.* The design lost about 38 points. A larger receptive field (more layers or filters), dropout or weight decay, or test-time averaging over a few shifted copies are all reasonable fixes; say why you expect it to help.

4. **Step 7.4. RESULTS summary.** Print everything you stored in `RESULTS` during Parts 4 to 7. This is your summary; copy it, with your written answers, into your notebook or report. The assert checks that at least 15 entries exist, which is true only if you ran every part from the top.

```python
print("RESULTS of Week 4, Parts 4 to 7:")
for k in sorted(RESULTS):
    if k.startswith(("p4", "p5", "p6", "p7")):
        v = RESULTS[k]
        s = str(v)
        print(f"  {k:26s} {s if len(s) < 90 else s[:87] + '...'}")
n_keys = sum(1 for k in RESULTS if k.startswith(("p4", "p5", "p6", "p7")))
print(n_keys, "entries")
assert n_keys >= 15, "run Parts 4 to 7 from the top so that RESULTS is complete"
```

   **What you should see.** About 22 entries, among them `p4_gradcheck_err` (about 1.7e-11), `p5_shift_cnn` (`[1.0, 0.953, 0.687, 0.483, 0.38]`), `p5_shift_mlp` (`[1.0, 0.38, 0.293, 0.227, 0.13]`), `p6_gap` (`{40: 0.187, 400: 0.007}`), `p7_stress` and `p7_stress_fixed` (0.87).

5. **Step 7.5. Four reflection questions.** Answer each question in at least 25 words, in your own words, and cite a number from this lab. (1) Why do convolution layers need far fewer parameters than dense layers? (2) What does weight sharing buy you, and what does it cost? (3) What does pooling throw away? (4) How would you explain the backward pass of a convolution to a friend who knows only that a gradient is a slope?

   *Exercise 7.5:* Write your four answers in the dictionary below. Why it matters: if you can explain it without code, you understand it.

```python
reflection = {
    "1_fewer_parameters": "",
    "2_weight_sharing": "",
    "3_pooling": "",
    "4_explain_backward": "",
}
short = [k for k, v in reflection.items() if len(v.split()) < 25]
if len(short) == len(reflection):
    print("Write your four answers above (at least 25 words each), then run this cell again.")
else:
    check("all four answers have at least 25 words", not short, "still short: " + ", ".join(short))
RESULTS["p7_reflection"] = reflection
```

   *What it means.* A good answer cites a number: the 2,533 parameters of the CNN against the 16,773 of the 64-unit MLP, the 25.7-point gap at shift 3, the 3 non-zero cells of Step 4.6, or the nine terms of a kernel gradient.

   **Before you finish.** Check that you can say each of these in your own words: a convolution layer reuses a few weights at every position, so it has few parameters and some tolerance to shifts; pooling trades position for robustness; the backward pass of a convolution sums over all positions for the kernel and is a transposed convolution for the input; and a design is only trusted after you stress-test it.

6. **Looking back.** Run the next block to compare an Imagine-first guess from Part 5 with the facts. Write one sentence in a markdown cell: did the experiments of this part change what you wrote in your spec about the pooling or the layer sizes?

```python
print("Your problem:", my_spec.get("problem", "") or "(not written)")
print("Did Parts 5 and 7 change what you expected? Compare:")
print("  Part 5 guess:", guess_winner or "(none)", "| measured gap at shift 3 (CNN minus MLP):", round(100 * (RESULTS["p5_shift_cnn"][3] - RESULTS["p5_shift_mlp"][3]), 1), "points")
```

### What you should see (Parts 4 to 7)

- Step 4.1: `y = [ [0, -1, -1], [-1, 1, 3], [2, 0, -2] ]`, `dL/dK = [ [3, 6], [3, 5] ]`, `dL/dx = [ [1, 0, -1, 0], [2, 0, 0, 1], [0, -1, 0, 0], [0, 0, -1, -1] ]`.
- Step 4.2: `L = -2.0`; finite-difference errors about 6.6e-12 (`dx`) and 1.7e-11 (`dK`), below 1e-6; autograd difference 0.
- Step 4.3: `dL/dx[1][1]` has 4 terms adding to 0; `dL/dK[0][1]` has 9 terms adding to 6; windows per input cell `[ [1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1] ]`; the largest `|dL/dx|` is 2, at cell (1, 0).
- Step 4.4: `conv_transpose2d` and the full convolution with the flipped kernel both give the loops' `dL/dx`.
- Step 4.5: output shapes 3x3, 5x5, 3x3 and 2x2 for (stride, pad) = (1, 0), (1, 1), (2, 1), (2, 0); `dx` (5, 5) and `dK` (3, 3) every time; errors about 1e-10 against finite differences and below 1e-14 against autograd.
- Step 4.6: pooled map `[ [4, 3], [0, 1] ]`, `L = 14`, three non-zero gradients (2 at (0, 2), 1 at (1, 1), 4 at (2, 3)).
- Step 4.7: loss 0.3798; four non-zero upstream gradients (0.0456, -0.0474, -0.1158, 0.1770); the hand chain equals autograd to about 1e-17.
- Step 5.1: shapes `(1, 8, 16, 16)`, `(1, 8, 8, 8)`, `(1, 16, 8, 8)`, `(1, 16, 4, 4)`, `(1, 256)`, `(1, 5)`; parameters 80, 1,168, 1,285; total 2,533.
- Step 5.2: MLP (width 10) 2,625 parameters and CNN 2,533 parameters, both with train accuracy 1.000 and test accuracy 1.000; 90 % test accuracy in epoch 2 (MLP) and epoch 1 (CNN).
- Step 5.3: accuracy at shift 0 to 4: MLP `[1.0, 0.38, 0.293, 0.227, 0.13]`, CNN `[1.0, 0.953, 0.687, 0.483, 0.38]`; gap at shift 3 about 25.7 points.
- Step 5.4: test accuracy at shift 3 with 150, 300, 600, 1,200 training pictures: MLP-10 0.577, 0.783, 0.950, 0.983; MLP-64 0.647, 0.813, 0.957, 0.980; CNN 0.990, 1.000, 1.000, 1.000.
- Step 5.5: three `RuntimeError` messages with the shapes `(512x8 and 512x5)`, "expected input to have 3 channels, but got 1", and `(8x256 and 1024x5)`; the fixed versions print `(8, 5)`, `(8, 8, 16, 16)` and `(8, 5)`.
- Step 5.6: plain CNN test 0.963 (gap +0.037), with batch norm and dropout 0.980 (gap +0.020).
- Step 6.1: filters with the best template match: 4 (Sobel y, -0.91), 1 (horizontal bar, -0.71), 0 (vertical bar, +0.67); two filters above 0.7.
- Step 6.2: six of the eight filters respond most to squares (filter 2 the most, 0.49); none is dead.
- Step 6.3: largest drops 0.00 (bars, diagonal), 0.89 (cross), 0.94 (square); mean drop 0.147 for patches on the shape and -0.012 off it; with a grey patch 0.070 inside and 0.112 outside.
- Step 6.4: correlation between occlusion and window-summed saliency `[0.14, 0.16, 0.69, 0.10, 0.43]`; 22 % of the cross picture's saliency lies on the 6.6 % of pixels that are the shape.
- Step 6.5: 40 pictures train 1.000, test 0.813 (gap +0.187); 400 pictures train 1.000, test 0.993 (gap +0.007).
- Step 7.1: example design 448 + 4,640 + 18,496 + 195 = 23,779 parameters.
- Step 7.2: 2,533 parameters (Part 5 CNN) and 14,309 (new design), train 1.000 for both, test 0.987 and 0.990.
- Step 7.3: stress test (noise 0.6, shift 4): 0.513 and 0.610; after training on pictures like the stress set 0.870.

Your numbers should be close to these. PyTorch versions and processors can change the last digits of a training run; the order of magnitude and the conclusions should not. Where they differ a lot, re-run from the Setup cell so the seeds are set again.

### Common mistakes (Parts 4 to 7)

- Using `x[i, j]` instead of `x[i + a, j + b]` in the kernel gradient: the result no longer depends on the kernel cell and the gradient check fails by a factor of order 1.
- Forgetting the flip and the padding when you write the input gradient as a convolution (Step 4.4), or forgetting to cut the padding off when the layer had padding (Step 4.5).
- Running a gradient check in single precision, or with `eps` too large or too small. Use float64 and `eps = 1e-5`.
- Expecting the cell with the most terms to have the largest gradient: terms can cancel, and what counts is the size of `G` times `K`.
- Forgetting that max-pooling gives the gradient to one cell per window only, and that ReLU blocks the gradient where its input was negative.
- Shape errors: no `Flatten` before `Linear`; `in_channels` that does not match the picture; a `Linear` size that ignores the poolings. Print the shape after every layer before you train.
- Comparing a small CNN with a large MLP and concluding something about structure. Match the number of parameters first (Step 5.2).
- Concluding from one run. Where a difference is one or two points on 300 test pictures (Step 5.6, Step 7.2), repeat with other seeds before you say which model is better.
- Treating an explanation map as a fact. The occlusion map depends on the patch and on the fill colour (Step 6.3), and saliency is a local slope (Step 6.4).
- Reading a train accuracy of 1.000 as success. Check the test accuracy and the gap (Step 6.5).

### Deliverables

A short checklist before you submit: every exercise has a written answer under its block; every `None` in a starter cell has been replaced; the code blocks ran from the first to the last without an error; and the printed `RESULTS` summary of Step 7.4 is included.

Submit your work the way your instructor asks (the notebook with outputs, or your JupyterLab files with the printed output and your written answers). For Parts 4 to 7 it must include:

- **Part 4 (18 points):** the loops `conv_backward_kernel` and `conv_backward_input` reproducing the lesson's `dL/dK` and `dL/dx` (4.1), the gradient check with errors below 1e-6 and the autograd comparison, plus your deliberately broken version (4.2), the term lists, the window counts and your Imagine-first guess compared (4.3), the transposed and full convolution (4.4), the generalised `conv_backward` for four settings (4.5), the ReLU and max-pool routing (4.6), and the chain through the small CNN with the conv bias gradient (4.7).
- **Part 5 (16 points):** `SmallCNN` with the shape and parameter table checked by hand (5.1), the CNN-versus-MLP table and curves (5.2), the shift test with the two lines and your reading (5.3), the training-on-shifted-pictures table and your answers (5.4), the three shape errors with their fixes and your five-line checklist (5.5), and, if you did it, the batch-norm and dropout comparison (5.6), with your Imagine-first guess compared.
- **Part 6 (14 points):** the filter heat maps and the correlation table (6.1), the feature maps and the filter-by-class table (6.2), the occlusion maps with the inside and outside averages and the grey-patch comparison (6.3), the saliency maps and the comparison with occlusion (6.4), and the 40-versus-400 learning curves with the gap (6.5).
- **Part 7 (14 points):** your design card and shape table (7.1), the trained variant with its curves (7.2), the stress test with the drop and a fix (7.3), the printed `RESULTS` summary (7.4), and your four reflection answers (7.5).

Together with Parts 1 to 3 the lab is marked out of **100 points**: Part 1 is worth 12, Part 2 is worth 14, Part 3 is worth 12, Part 4 is worth 18, Part 5 is worth 16, Part 6 is worth 14 and Part 7 is worth 14. The rubric is in the section "How Week 4 is assessed". The work graded is the one your instructor specifies.

**Stretch goals.** Extend `conv_backward` to several input and output channels (loop over the channels and check against autograd); implement the backward pass of average-pooling and compare with `F.avg_pool2d`; repeat Step 5.3 with three seeds and plot the mean with a band; add a third convolution layer to `SmallCNN` and compute the receptive field of the last unit (Step 3.5); replace the max-pool of Step 5.1 by stride-2 convolutions and compare accuracy and parameters; compute the occlusion map of Step 6.3 for a picture with shift 3 and see whether the bright region moves with the shape; for your own spec in Step 7.1, write the code with `nn.Sequential`, train it on any 200 pictures you can draw yourself, and report the learning curves.
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions cover the six outcomes. Nine of them need a short calculation (a convolution value, an output size, a parameter count, a dL/dK or dL/dx value, a pooled value, a receptive field), so keep a pencil nearby; the explanations show the arithmetic.

```quiz
{
  "id": "week-04-check",
  "title": "Week 4 quiz: Convolutional neural networks",
  "passMark": 70,
  "questions": [
    {
      "q": "A 32 × 32 colour picture (3 channels) is unrolled and fed to a dense layer with 10 units. How many parameters does that layer have?",
      "options": [
        "30,720",
        "3,072",
        "30,730",
        "10,240"
      ],
      "answer": 2,
      "why": "The picture has 32 × 32 × 3 = 3,072 numbers, so the weights are 3,072 × 10 = 30,720 and the 10 biases make 30,730. 30,720 forgets the biases, 3,072 is only the number of inputs, and 10,240 forgets the three colour channels."
    },
    {
      "q": "You apply one fixed random shuffle to the pixels of every training and every test picture, then retrain a dense network. What do you expect?",
      "options": [
        "About the same accuracy, because a dense layer never used which pixels are neighbours",
        "Much lower accuracy, because the shuffle destroys the edges that dense layers rely on",
        "Much higher accuracy, because the shuffle works like data augmentation",
        "Training is impossible, because the input shape changes after shuffling"
      ],
      "answer": 0,
      "why": "A dense unit has its own weight for every pixel position, and the same shuffle is used for all pictures, so the network can learn the shuffled data about as well. This shows it does not use the layout. The shape does not change, and a fixed shuffle is not augmentation. A CNN, in contrast, would suffer."
    },
    {
      "q": "Which pair of ideas lets a convolution layer find the same pattern anywhere in a picture while using only a few parameters?",
      "options": [
        "Pooling and a larger learning rate",
        "Padding and a stride greater than 1",
        "Fully connected units and dropout",
        "Local connectivity and weight sharing"
      ],
      "answer": 3,
      "why": "Each unit looks at a small patch (local connectivity) and the same kernel is used at every position (weight sharing). Pooling, padding, stride, dropout and the learning rate change other things, but none of them reuses one set of weights across the picture."
    },
    {
      "q": "A 2 × 2 window of a picture has rows (3, 1) and (2, 4). The kernel has rows (1, 2) and (0, −1) and the bias is 2. What is the output value before ReLU?",
      "options": [
        "1",
        "3",
        "11",
        "−1"
      ],
      "answer": 1,
      "why": "3×1 + 1×2 + 2×0 + 4×(−1) = 3 + 2 + 0 − 4 = 1, and adding the bias 2 gives 3. The answer 1 forgets the bias, −1 subtracts the bias, and 11 ignores the minus sign in the kernel."
    },
    {
      "q": "A 32 × 32 picture goes through a 5 × 5 convolution with padding 2 and stride 2. What is the output height?",
      "options": [
        "14",
        "15",
        "16",
        "32"
      ],
      "answer": 2,
      "why": "⌊(32 + 2×2 − 5) / 2⌋ + 1 = ⌊31 / 2⌋ + 1 = 15 + 1 = 16. The answer 14 forgets the padding, 15 forgets the + 1, and 32 would be the result for stride 1."
    },
    {
      "q": "A convolution layer has 16 input channels, 32 filters and 3 × 3 kernels. How many parameters does it have?",
      "options": [
        "4,640",
        "4,608",
        "9,248",
        "512"
      ],
      "answer": 0,
      "why": "3 × 3 × 16 × 32 = 4,608 weights, plus 32 biases, gives 4,640. The answer 4,608 forgets the biases, 9,248 uses 32 input channels by mistake, and 512 is 16 × 32, a 1 × 1 kernel without biases."
    },
    {
      "q": "Why is the gradient with respect to a kernel weight, dL/dK[a, c], a sum over all output positions?",
      "options": [
        "Because the loss is averaged over all the pictures in the batch",
        "Because ReLU adds up the gradients of neighbouring pixels",
        "Because pooling merges the gradients of several different kernels",
        "Because the same weight was used at every position, so each position contributes some of the blame"
      ],
      "answer": 3,
      "why": "With weight sharing, K[a, c] was multiplied by a different pixel at every position, so by the chain rule each position adds G[i, j] × x[i + a, j + c] to its gradient. Averaging over the batch, ReLU and pooling are separate steps and do not explain the sum."
    },
    {
      "q": "A 3 × 3 picture has rows (2, 1, 0), (1, 3, 2), (0, 1, 4). A 2 × 2 kernel gives a 2 × 2 output whose upstream gradient G has rows (1, 2) and (−1, 3). What is dL/dK[0,0]?",
      "options": [
        "18",
        "12",
        "4",
        "10"
      ],
      "answer": 1,
      "why": "dL/dK[0,0] = Σ G[i, j] × x[i, j] over the four positions = 1×2 + 2×1 + (−1)×1 + 3×3 = 2 + 2 − 1 + 9 = 12. The other cells of dL/dK are 4, 10 and 18, so the other options come from reading the wrong pixel for each G."
    },
    {
      "q": "The kernel has rows (1, −1) and (2, 0), and the upstream gradient is G with rows (1, 2) and (−1, 3) for a 3 × 3 picture. What is dL/dx at the centre pixel x[1,1]?",
      "options": [
        "8",
        "6",
        "12",
        "3"
      ],
      "answer": 0,
      "why": "All four windows cover the centre. The terms are G[0,0]·K[1,1] + G[0,1]·K[1,0] + G[1,0]·K[0,1] + G[1,1]·K[0,0] = 1×0 + 2×2 + (−1)×(−1) + 3×1 = 0 + 4 + 1 + 3 = 8. The value 6 comes from a sign slip, 12 is dL/dK[0,0], and 3 counts only one window."
    },
    {
      "q": "Average pooling with a 2 × 2 window and stride 2 is applied to a 4 × 4 map with rows (2,7,1,0), (3,4,5,2), (6,0,1,1) and (2,9,3,3). What is the bottom-left output?",
      "options": [
        "9",
        "17",
        "4.25",
        "4"
      ],
      "answer": 2,
      "why": "The bottom-left window holds 6, 0, 2 and 9. Their sum is 17 and their average is 17 / 4 = 4.25. The answer 9 is the maximum (max pooling), 17 is the sum, and 4 is the average of the top-left window."
    },
    {
      "q": "A 5 × 5 convolution (stride 1) is followed by a 2 × 2 max pool (stride 2) and another 5 × 5 convolution (stride 1). What is the receptive field of a unit in the second convolution?",
      "options": [
        "9",
        "14",
        "10",
        "16"
      ],
      "answer": 1,
      "why": "rf after the first convolution = 5, jump 1. The pool gives rf = 5 + 1×1 = 6 and jump 2. The second convolution gives rf = 6 + 4×2 = 14. The answer 9 ignores the pool’s doubling of the jump, 10 adds the kernel sizes, and 16 would be the field after a second pool."
    },
    {
      "q": "A batch of shape (N, 3, 32, 32) goes through Conv2d(3, 10, 5) with no padding, then MaxPool2d(2), then Flatten. How many numbers per picture does Flatten output?",
      "options": [
        "7,840",
        "2,560",
        "1,690",
        "1,960"
      ],
      "answer": 3,
      "why": "The convolution gives 32 − 5 + 1 = 28, so 28 × 28 × 10; the pool halves it to 14 × 14 × 10 = 1,960. The answer 7,840 forgets the pool, 2,560 and 1,690 come from wrong sizes (16 × 16 and 13 × 13) that do not follow the output-size formula."
    },
    {
      "q": "What does 2 × 2 max pooling give a CNN, and what does it cost?",
      "options": [
        "Smaller maps and a little tolerance to small shifts, at the cost of forgetting exactly where in the window the maximum was",
        "New learnable kernels that raise accuracy, at the cost of more memory for weights",
        "Exact positions of all patterns, at the cost of removing some of the channels",
        "A replacement for the ReLU, at the cost of one extra hyperparameter"
      ],
      "answer": 0,
      "why": "Pooling halves the height and width, so later layers are cheaper, and a small shift inside a window leaves the maximum unchanged; but the exact position within the window is lost. It has no learnable weights, does not keep exact positions, and does not replace ReLU."
    },
    {
      "q": "During backpropagation through a 2 × 2 max pool, the gradient arriving for one window is 0.8. Which cells of that window receive gradient?",
      "options": [
        "All four cells equally, 0.2 each",
        "All four cells, in proportion to their values",
        "Only the cell that held the maximum, which receives 0.8",
        "Only the cell with the smallest value"
      ],
      "answer": 2,
      "why": "Only the winner influenced the output, so only it gets the gradient (the other three get 0). Sharing 0.2 each is the rule for average pooling, and neither proportional sharing nor the smallest cell is a rule of max pooling."
    },
    {
      "q": "A CNN raises the error “mat1 and mat2 shapes cannot be multiplied (32x256 and 400x120)”. What is the most likely cause?",
      "options": [
        "The batch size of 32 is too small for this model",
        "Flatten gives 256 numbers per picture but nn.Linear expects 400, because an earlier padding or kernel size changed the shapes",
        "The learning rate is too high",
        "The ReLU must be moved after the Flatten layer"
      ],
      "answer": 1,
      "why": "In a matrix product the inner sizes must agree: the flattened input has 256 numbers per example and the layer expects 400. Recompute the shapes with the output-size formula, for example a missing padding of 2 in the first convolution turns 400 into 256. The batch size, the learning rate and the ReLU position do not cause this message."
    }
  ]
}
```

## How Week 4 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: Pictures are numbers, and why dense layers struggle (tensor shapes, parameter counts, MLP on shapes, shift test, pixel shuffle) | 12 | Pictures shown and the tensor shape (N, C, H, W) explained; dense and convolution parameter counts done by hand and checked, including 150,529,000; the MLP trained on centred shapes with curves; the shift test plotted with the drop explained by position-specific weights; the pixel-shuffle result measured and explained; three ways layout carries meaning written | Steps run and counts are right; the explanation of the shift drop or of the shuffle is brief | Parameter counts missing or wrong; the shift test or the shuffle experiment absent; conclusions not backed by the printed numbers |
| Part 2: Convolution by hand (worked example, output size, stride and padding, parameters, weight sharing) | 14 | The 4x4 by 2x2 example by hand and with `conv2d_np` equal to the lesson and to `F.conv2d`; `out_size` checked against real outputs, including the three lesson examples; stride and padding shown with shapes and the border effect; layer parameters computed and matched with `count_params`; weights of a dense layer against a conv layer counted and plotted on a log scale; flipped-kernel difference shown | Calculations done and checks pass; the border effect or the weight-sharing plot is incomplete | Output sizes or parameter counts wrong; no comparison with PyTorch |
| Part 3: Filters, feature maps and pooling (hand-made filters, own detector, ReLU and bias, pooling, receptive field) | 12 | Four hand-made filters applied to one shape of each class and explained; an own detector with a measured separation accuracy; ReLU and bias effects shown; pooling by hand with the changed-values count for a one-pixel shift; the receptive field sequence 3, 4, 8, 10, 18 computed and asserted; a careful layer-by-layer story for faces | Everything runs; the explanation of the filters or of the receptive field is general | Filters not applied to a real picture; no separation number; receptive field wrong |
| Part 4: The backward pass of a convolution (loops, gradient check, convolution view, stride and padding, ReLU and pooling, chain) | 18 | `conv_backward_kernel` and `conv_backward_input` reproduce the lesson (`dL/dK = [ [3, 6], [3, 5] ]`) with explicit loops; the gradient check below 1e-6 and agreement with autograd, plus a deliberately broken version that fails; terms listed and the Imagine-first guess compared honestly; transposed and full convolution shown equal; `conv_backward` for four (stride, padding) settings checked; the ReLU and max-pool routing with three non-zero cells; the chain through the small CNN equal to autograd, with the bias gradient | Everything runs and the checks pass; the explanation of the sums over positions or of the routing is brief | A backward function missing or its gradients wrong; no gradient check; stride and padding absent |
| Part 5: Build and train a CNN (SmallCNN, comparison with the MLP, shift test, shifted training, shape errors) | 16 | Shapes and 2,533 parameters checked by hand; CNN and MLP of matched size compared with curves; the shift lines plotted and read honestly (CNN 0.483 against MLP 0.227 at shift 3, and its own drop); the data-size table read correctly (CNN near perfect with 150 pictures, MLP needing about 1,200); three shape errors provoked and fixed with a concrete five-line checklist; the Imagine-first guess compared | Steps run and numbers are right; the reading of the shift plot or of the table is general | Parameter counts or shapes wrong; unfair comparison (different sizes) with no comment; claims not backed by the printed numbers |
| Part 6: Look inside the network (filters, feature maps, occlusion, saliency, overfitting) | 14 | Eight filters plotted with a shared scale and correlated with the templates; feature maps and the filter-by-class table read carefully (generalists, not specialists); occlusion maps with inside and outside averages and the grey-patch effect explained; saliency compared with occlusion with a number and a sentence; 40-versus-400 curves with the gap and two remedies | Everything runs; the readings of the maps are general or partly unsupported | Filters or maps missing; explanation maps presented as facts with no measurement; no learning-curve comparison |
| Part 7: Design and reflect (design card, variant, stress test, summary, reflection) | 14 | Complete spec with the shape table and parameter count that match; a built variant (at least two conv layers) with one change, curves and an honest comparison; stress test with the drop, a tried fix and a reasoned second fix; `RESULTS` summary complete; four specific reflection answers that cite numbers from the lab | Everything present; the design is generic or the answers are general | Spec or reflection missing; no stress test; shapes of the design inconsistent |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave (your instructor will use them to open Week 5):
1. A 28 × 28 greyscale picture goes through a convolution with 6 filters of size 5 × 5, padding 0 and stride 1. Give the output shape and the number of parameters, and say which formula you used for each.
2. In one sentence each, say why dL/dK is a sum over all output positions, and why dL/dx for one pixel adds up only a few terms.
3. Name one thing weight sharing gives you and one thing max pooling takes away.

**Homework / self-study**
- Finish the lab, including the written answers and the Part 7 design.
- Re-read the `equation` figures in this lesson (the dense parameter count, the convolution, the convolution parameter count, the output size, the pooled size, the receptive field, the kernel gradient, the input gradient, the ReLU backward rule and the gradient check) and, for each, write one sentence in your own words about what every symbol means.
- Preview Week 5: it is the Prelim exam week, so review Weeks 1–4 and bring your questions.
