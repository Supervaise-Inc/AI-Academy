> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 18 — Scaling and Deploying AI

**Purpose:** Learn why a big model does not fit on one GPU and why one GPU is too slow, and how ==splitting the work (data, tensor and pipeline parallelism) and using fewer bits per number (mixed precision and quantization)== make a model train, fit and serve.

**Term:** Final (Weeks 16–20) · **Month:** 5 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch a model get too big, then get split and shrunk",
  "caption": "One model, five numbers. Memory shrinks when you use fewer bits per weight, training needs far more memory than the weights, a pipeline of GPUs wastes time while it fills up, a weight rounded to an integer loses a little, and decoding speed is limited by how fast memory can be read. Every number is computed from the formulas you will learn today; GPU memory and bandwidth values are assumed examples.",
  "scenes": [
    { "name": "Weights", "kind": "bars", "prompt": "A 7-billion-parameter model: memory for the weights alone", "outputLabel": "Share of the FP32 size (FP32 28 GB, FP16 14 GB, INT8 7 GB, INT4 3.5 GB)", "bars": [ { "label": "FP32: 28 GB", "value": 1 }, { "label": "FP16: 14 GB", "value": 0.5 }, { "label": "INT8: 7 GB", "value": 0.25 }, { "label": "INT4: 3.5 GB", "value": 0.125 } ] },
    { "name": "Training", "kind": "text", "prompt": "The same 7 B model, but now you want to train it with Adam", "outputLabel": "The arithmetic (assumed example GPU with 80 GB)", "output": "Training needs about 16 bytes per parameter: 7 × 10⁹ × 16 = 112 × 10⁹ bytes = 112 GB. One example GPU with 80 GB cannot hold it, so you need at least 2 GPUs before you even count the activations." },
    { "name": "Idle share", "kind": "bars", "prompt": "A pipeline of 4 GPUs: how much of the time is each GPU idle?", "outputLabel": "Idle share = (p − 1) ÷ (m + p − 1) with p = 4 stages", "bars": [ { "label": "1 micro-batch: 75%", "value": 0.75 }, { "label": "4 micro-batches: 43%", "value": 0.43 }, { "label": "8 micro-batches: 27%", "value": 0.27 }, { "label": "32 micro-batches: 9%", "value": 0.09 } ] },
    { "name": "Rounding", "kind": "text", "prompt": "Store the weight 0.42 as a 4-bit integer", "outputLabel": "Quantization example (largest weight in the group is 0.51)", "output": "scale = 0.51 ÷ 7 = 0.0729. q = round(0.42 ÷ 0.0729) = round(5.76) = 6. Restored value = 6 × 0.0729 = 0.437. The error is 0.017: small, and the weight now takes 4 bits instead of 16." },
    { "name": "Speed limit", "kind": "text", "prompt": "How many tokens per second can a 14 GB model generate?", "outputLabel": "An upper bound from memory bandwidth (assumed 200 GB/s)", "output": "Each new token reads every weight once: 200 GB/s ÷ 14 GB = 14.3 tokens per second at most. The same model in INT4 is 3.5 GB: 200 ÷ 3.5 = 57 tokens per second at most." }
  ]
}
```

## By the end, you can…

1. ==Compute the memory of a model== for inference and for training at FP32, FP16, INT8 and INT4, and say how many GPUs it needs.
2. ==Explain data parallelism== (copy the model, split the batch, average the gradients) and compute its all-reduce traffic and efficiency.
3. ==Explain tensor parallelism== with a split matrix product and ==pipeline parallelism== with its bubble, and compute the bubble.
4. ==Say which parallelism fits which situation==: data, tensor, pipeline, or a combination.
5. ==Explain mixed precision== (FP16, BF16, loss scaling) and ==quantization== (scale, rounding error, outliers) and measure their effect on speed and memory.
6. ==Reason about serving==: latency against throughput, batching, the KV cache and the memory-bound speed of decoding.

Assessed in the Week 18 lab (“Imagine the warehouse and the delivery trucks”) and the quiz. This week uses Week 8 (parameters of a Transformer), Week 9 (C ≈ 6ND and scaling laws), Week 14 (LoRA, QLoRA and 16 bytes per parameter), Week 16 (context budgets) and Week 17 (agents and guardrails). Read each section, then play with the figures under it: they are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. Every formula comes with a figure: select its symbols to read what they mean and step through the worked example. Next week (Week 19) is the capstone build week, and your capstone will be deployed on the school's server, so this lesson is your toolbox for that.

## Why one GPU is not enough {rail=Why scale}

{lede} A [[GPU|A graphics processing unit: a chip with thousands of small cores that do the matrix arithmetic of neural networks very quickly, with its own fast memory.]] has two limits that a big model hits very quickly: ==it runs out of memory, and it runs out of time==. Everything in this lesson is a way to get past one of those two limits, or both.

### The warehouse and the delivery trucks

Imagine the school's AI helpdesk. At first it is a classroom demo: one small model, one laptop. Then the whole district wants it. Think of the GPU as a warehouse with one delivery truck. The warehouse has a fixed floor space: that is the GPU's ==memory==. If your goods (the model's numbers) do not fit on the floor, no amount of driving helps. The truck has a fixed speed: that is the GPU's ==compute==. If you must deliver a million parcels (trillions of training tokens), one truck needs years.

- **Too big for the warehouse** is a *memory* problem. Fix: put the goods in several warehouses (split the model) or pack them smaller (use fewer bits).
- **Too slow for one truck** is a *time* problem. Fix: use many trucks at once (split the work) and keep them all busy.

```viz
{
  "type": "sorter",
  "title": "Memory limit or time limit?",
  "intro": "Each card is a problem. Drop it on the limit that causes it. Some problems are cured by the same trick that cures the other.",
  "bins": [ { "id": "memory", "label": "Memory limit (warehouse too small)" }, { "id": "time", "label": "Time limit (truck too slow)" } ],
  "cards": [
    { "name": "A 70 B model's FP16 weights are 140 GB and the GPU holds 80 GB", "answer": "memory", "why": "The weights alone do not fit on the floor of the warehouse: this is memory." },
    { "name": "Training on 1 trillion tokens would take years on one GPU", "answer": "time", "why": "The model fits, but the number of operations is huge: this is time, and many GPUs working together fix it." },
    { "name": "The Adam optimiser state is 8 bytes per parameter and the run crashes with out of memory", "answer": "memory", "why": "Optimiser state is stored in memory next to the weights; it adds to the floor space needed." },
    { "name": "One training step takes 4 seconds and you need a million steps", "answer": "time", "why": "Steps are too slow, not too big. More GPUs or faster arithmetic shorten the wall-clock time." },
    { "name": "The KV cache of 8 long conversations is larger than the weights", "answer": "memory", "why": "The cache has to be stored in GPU memory while the conversations run; it competes with the weights for space." },
    { "name": "The helpdesk answers one student in 3 s but cannot serve 200 at once", "answer": "time", "why": "Capacity (requests per second) is limited by how fast the GPU can work through them: batching and more GPUs help." },
    { "name": "A batch of 64 long sequences stores too many activations for backpropagation", "answer": "memory", "why": "Activations are kept until the backward pass; they fill the memory even when the model itself fits." },
    { "name": "Generating 500 tokens, one at a time, feels slow to the user", "answer": "time", "why": "Each token needs a pass over the weights, so the delay is a speed limit, not a size limit." }
  ]
}
```

Notice how many problems are about memory even when the model itself fits: activations, optimiser state and the KV cache all use the same floor space.

### Limit 1: memory

==Memory is used by four things.== The ==weights== are the model's numbers. The ==gradients== are one number per weight, computed in training. The [[optimiser state|Extra numbers the optimiser keeps for every weight. Adam keeps two (a running mean of the gradient and of its square), so it adds 8 bytes per parameter when stored in FP32.]] holds the running averages that Adam uses. The [[activations|The intermediate results of the forward pass. Training keeps them so that backpropagation can use them; they grow with batch size and sequence length.]] are the intermediate results of each layer. In inference the [[KV cache|Stored keys and values of all earlier tokens, so that each new token needs only one new row of attention instead of recomputing everything (Weeks 7–8).]] joins the list.

The simplest case is inference: the weights. A [[parameter|One learned number inside a model, usually a weight. A model with 7 billion parameters stores 7 billion such numbers.]] stored in a number format of b bytes costs b bytes. ==Weights memory = number of parameters × bytes per parameter==.

```viz
{
  "type": "equation",
  "title": "Memory of the weights",
  "intro": "Select each part, then step through the example for a 7-billion-parameter model.",
  "formula": [ { "t": "M", "id": "m" }, " = ", { "t": "N", "id": "n" }, " × ", { "t": "b", "id": "b" } ],
  "parts": [
    { "id": "m", "name": "M (memory of the weights)", "meaning": "How many bytes the weights occupy. Divide by 10⁹ to get gigabytes (GB)." },
    { "id": "n", "name": "N (number of parameters)", "meaning": "How many learned numbers the model has. A 7 B model has N = 7 × 10⁹." },
    { "id": "b", "name": "b (bytes per parameter)", "meaning": "FP32 uses 4 bytes, FP16 and BF16 use 2, INT8 uses 1 and INT4 uses 0.5 (half a byte)." }
  ],
  "steps": [
    { "title": "FP16, 2 bytes", "expr": "M = 7 × 10⁹ × 2 = 14 × 10⁹ bytes = 14 GB", "note": "A 7 B model in 16 bits needs 14 GB for its weights." },
    { "title": "FP32, 4 bytes", "expr": "M = 7 × 10⁹ × 4 = 28 GB", "note": "Twice as many bytes per number, twice the memory." },
    { "title": "INT4, half a byte", "expr": "M = 7 × 10⁹ × 0.5 = 3.5 GB", "note": "Four times smaller than FP16. Later you will see that the scales add a little." }
  ]
}
```

Notice that the formula has only one lever you can pull without changing the model: b. Everything about precision, later in this lesson, is about making b smaller.

```viz
{
  "type": "calc",
  "title": "How much memory does a model need?",
  "kind": "memory",
  "intro": "Drag the model size. Set it to 7, then to 70, and read the bars.",
  "note": "Weights only (first four bars) plus the common 16-bytes-per-parameter estimate for mixed-precision training with Adam. Activations, the KV cache and framework overhead are not included."
}
```

At 70 B the FP16 weights are 140 GB. With an example GPU of 80 GB that is 140 ÷ 80 = 1.75, so you need at least ==2 GPUs just to hold the weights==, with no room left for a KV cache. For training the story is far worse.

### Training memory: 16 bytes per parameter

In Week 14 you met the rule that training a model with Adam in mixed precision needs about 16 bytes per parameter: 2 bytes for the 16-bit weights, 2 for the 16-bit gradients, 4 for a 32-bit master copy of the weights and 8 for Adam's two moments (4 bytes each). ==Training memory ≈ 16 × the number of parameters==, before activations.

```viz
{
  "type": "equation",
  "title": "Training memory of a model (Adam, mixed precision)",
  "intro": "Select the parts of the 16 bytes, then run the 7 B and 70 B examples.",
  "formula": [ { "t": "M", "id": "m" }, " ≈ ", { "t": "N", "id": "n" }, " × (", { "t": "2", "id": "w" }, " + ", { "t": "2", "id": "g" }, " + ", { "t": "4", "id": "mc" }, " + ", { "t": "8", "id": "a" }, ") = ", { "t": "16 N", "id": "tot" }, " bytes" ],
  "parts": [
    { "id": "m", "name": "M (training memory)", "meaning": "Bytes needed to hold weights, gradients and optimiser state, not counting activations." },
    { "id": "n", "name": "N (parameters)", "meaning": "The number of learned numbers in the model." },
    { "id": "w", "name": "2 bytes: 16-bit weights", "meaning": "The copy of the weights used by the forward and backward passes." },
    { "id": "g", "name": "2 bytes: 16-bit gradients", "meaning": "One gradient per weight, in 16 bits." },
    { "id": "mc", "name": "4 bytes: master copy", "meaning": "A 32-bit copy of the weights that receives the tiny updates accurately (see the Precision section)." },
    { "id": "a", "name": "8 bytes: Adam's two moments", "meaning": "The running mean of the gradient and of its square, each 4 bytes, for every parameter." },
    { "id": "tot", "name": "16 N", "meaning": "The total: 16 bytes for each of the N parameters." }
  ],
  "steps": [
    { "title": "A 7 B model", "expr": "16 × 7 × 10⁹ = 112 × 10⁹ bytes = 112 GB", "note": "Eight times the 14 GB of FP16 weights." },
    { "title": "How many 80 GB GPUs (assumed)?", "expr": "112 ÷ 80 = 1.4 → at least 2 GPUs", "note": "Activations come on top, so in practice you need more." },
    { "title": "A 70 B model", "expr": "16 × 70 × 10⁹ = 1,120 GB; 1,120 ÷ 80 = 14 → at least 14 GPUs", "note": "And we have not stored a single activation." }
  ]
}
```

Notice that 14 GPUs is a floor, not an estimate of what the run needs. It tells you immediately that one GPU is out of the question.

```viz
{
  "type": "calc",
  "title": "Fine-tuning memory: full, LoRA and QLoRA",
  "kind": "ft_memory",
  "intro": "Set the model to 7 B and press the three buttons. Full fine-tuning needs 112 GB; adapters need far less.",
  "note": "Weights, gradients and Adam state only; activations are not included. Full fine-tuning is 16 bytes per parameter; adapters keep the base weights at 2 bytes (LoRA) or 0.5 bytes (QLoRA) and train a small share."
}
```

This is the same figure as in Week 14. LoRA and QLoRA are themselves memory tricks: they avoid storing gradients and optimiser state for most of the weights, which is why a 7 B model can be fine-tuned on a single GPU.

```viz
{
  "type": "logbars",
  "title": "Memory needed against one GPU",
  "unit": "gigabytes",
  "items": [
    { "label": "7 B weights, INT4", "year": "inference", "value": 3.5, "note": "7 × 10⁹ × 0.5 bytes" },
    { "label": "7 B weights, FP16", "year": "inference", "value": 14, "note": "7 × 10⁹ × 2 bytes" },
    { "label": "One GPU with 80 GB (an example size)", "year": "assumed", "value": 80, "note": "An example size, not the specification of any named GPU" },
    { "label": "7 B training (16 bytes each)", "year": "training", "value": 112, "note": "7 × 10⁹ × 16 bytes, no activations" },
    { "label": "70 B weights, FP16", "year": "inference", "value": 140, "note": "70 × 10⁹ × 2 bytes" },
    { "label": "70 B training (16 bytes each)", "year": "training", "value": 1120, "note": "70 × 10⁹ × 16 bytes, no activations" }
  ],
  "note": "Computed from parameters × bytes. The 80 GB GPU is an assumed example size; real GPUs differ."
}
```

Look at where the 80 GB marker falls: small models fit, but the interesting models (7 B for training, 70 B for anything) sit to its right.

### Limit 2: time

Memory decides whether a model fits. Time decides whether you live to see it trained. Week 9 gave you the rule of thumb ==training compute C ≈ 6 N D== floating-point operations, where N is the number of parameters and D the number of training tokens. Divide by the speed of your hardware to get seconds.

```viz
{
  "type": "equation",
  "title": "Training time from C ≈ 6ND",
  "intro": "Select each part. The speed per GPU is an assumption: real sustained speeds depend on the hardware and the software.",
  "formula": [ { "t": "T", "id": "t" }, " ≈ ", { "t": "6 N D", "id": "c" }, " ÷ (", { "t": "G", "id": "g" }, " × ", { "t": "R", "id": "r" }, ")" ],
  "parts": [
    { "id": "t", "name": "T (training time)", "meaning": "Wall-clock seconds, if all GPUs are kept busy all the time (an ideal)." },
    { "id": "c", "name": "6ND (total compute)", "meaning": "About 6 floating-point operations per parameter per token: 2 forward, 4 backward." },
    { "id": "g", "name": "G (number of GPUs)", "meaning": "How many GPUs share the work." },
    { "id": "r", "name": "R (sustained speed per GPU)", "meaning": "Floating-point operations per second that one GPU really keeps up. Here we assume 200 TFLOP/s = 2 × 10¹⁴." }
  ],
  "steps": [
    { "title": "Total compute", "expr": "C = 6 × 7 × 10⁹ × 10¹² = 4.2 × 10²² FLOPs", "note": "7 B parameters, 1 trillion tokens." },
    { "title": "One GPU (assumed 200 TFLOP/s)", "expr": "T = 4.2 × 10²² ÷ 2 × 10¹⁴ = 2.1 × 10⁸ s ≈ 6.7 years", "note": "One truck delivering for almost seven years." },
    { "title": "1,024 GPUs, perfectly busy", "expr": "T = 2.1 × 10⁸ ÷ 1,024 = 2.05 × 10⁵ s ≈ 2.4 days", "note": "In practice the GPUs are not perfectly busy, which is why communication matters." }
  ]
}
```

Notice the word *perfectly*: to get 1,024 GPUs to deliver a 1,024-fold speed-up, they must spend almost no time waiting for each other. That is exactly the question of the next three sections.

```viz
{
  "type": "scalinglaw",
  "title": "How big should the model be for a given compute budget?",
  "intro": "This is the Week 9 figure. Move the compute budget and see which size is best: bigger is not always better for a fixed budget.",
  "logC": 23.77,
  "logN": 10.8,
  "note": "Parametric loss fit from Besiroglu et al. (2024); the numbers are estimates, not guarantees.",
  "checks": [ { "N": 7e10, "D": 1.4e12, "loss": 1.974 }, { "C": 5.76e23, "optN": 7.2e10, "optRatio": 18.4 } ]
}
```

Because 6ND grows with both the model size and the data, the models that the scaling laws recommend for large budgets are exactly the ones that do not fit on one GPU. Scaling laws and parallelism are two sides of one story.

### Three ways to split the work

There are three basic ways to cut a training or inference job into pieces for several GPUs, and each fits one of the two limits. [[Data parallelism|Copy the whole model to several GPUs and give each a different slice of the batch, then average the gradients.]] ==splits the batch== (each GPU holds the whole model). [[Tensor parallelism|Split each layer's weight matrices across GPUs, so every GPU holds a slice of every layer and the partial results are joined.]] ==splits each layer's matrices== (each GPU holds a slice of every layer). [[Pipeline parallelism|Split the layers into consecutive stages, one stage per GPU, and pass micro-batches from stage to stage.]] ==splits the layers into stages== (each GPU holds some whole layers). Precision tricks sit on top of all three.

```viz
{
  "type": "pipeline",
  "title": "Three ways to split the work",
  "intro": "Step through. Each step shows what is divided between the GPUs.",
  "nodes": [
    { "id": "model", "label": "The model", "sub": "layers of matrices" },
    { "id": "data", "label": "Data parallel", "sub": "split the batch" },
    { "id": "tensor", "label": "Tensor parallel", "sub": "split each matrix" },
    { "id": "pipe", "label": "Pipeline parallel", "sub": "split the layers" },
    { "id": "prec", "label": "Fewer bits", "sub": "mixed precision, quantization" }
  ],
  "steps": [
    { "title": "The problem", "body": "One model with many layers, each layer a large matrix, and a big batch of examples. It is too large or too slow for one GPU.", "active": ["model"], "example": "A 70 B model with 80 layers of large matrices." },
    { "title": "Data parallelism: copy the model, split the batch", "body": "Every GPU holds a full copy and processes its own slice of the batch. Then all GPUs average their gradients. It speeds up training but each GPU still needs the whole model.", "active": ["data", "model"], "example": "8 GPUs, each with the full model and 1/8 of the batch." },
    { "title": "Tensor parallelism: split each matrix", "body": "A big matrix is cut into slices, one per GPU, and the partial results are joined. It lets a layer that is too big for one GPU fit, but it needs fast links because the GPUs talk inside every layer.", "active": ["tensor", "model"], "example": "Each of 4 GPUs holds one quarter of every weight matrix." },
    { "title": "Pipeline parallelism: split the layers into stages", "body": "GPU 1 holds the first layers, GPU 2 the next, and so on. The batch is cut into micro-batches that flow through the stages, so only the activations at the boundaries travel.", "active": ["pipe", "model"], "example": "4 GPUs, each with 20 of 80 layers." },
    { "title": "Fewer bits on top", "body": "Whatever the split, storing each number in fewer bits (FP16, BF16, INT8, INT4) shrinks memory and, often, shortens the time spent reading it.", "active": ["prec"], "example": "FP16 halves the bytes of FP32; INT4 quarters them." }
  ],
  "note": "A conceptual diagram. Real systems combine these ideas."
}
```

Keep this picture in mind: each of the next three sections takes one box of it apart. At the end of Section 4 you will have a rule for choosing.

```viz
{
  "type": "timeline",
  "title": "How scaling became engineering",
  "events": [
    { "year": "2017", "title": "Mixed precision training", "text": "Micikevicius et al. showed that deep networks can be trained with 16-bit numbers by keeping a 32-bit master copy and scaling the loss (the paper appeared first in 2017 and at ICLR in 2018)." },
    { "year": "2018", "title": "GPipe", "text": "Huang et al. described pipeline parallelism with micro-batches, so that a model too big for one device could be split into stages (the paper appeared in 2018 and at NeurIPS in 2019)." },
    { "year": "2019", "title": "Megatron-LM", "text": "Shoeybi et al. split the matrices inside Transformer layers across GPUs (tensor parallelism) to train language models with billions of parameters." },
    { "year": "2019", "title": "ZeRO", "text": "Rajbhandari et al. described how to shard the optimiser state, gradients and parameters across data-parallel GPUs so that no GPU holds a full copy (the paper appeared in 2019 and at SC in 2020)." },
    { "year": "2020", "title": "GPT-3", "text": "Brown et al. reported a language model with 175 billion parameters. Training it required splitting the work across very many GPUs." }
  ]
}
```

Click through the events: the techniques came one at a time, and every time somebody hit a wall (memory or time), a new split was invented.

## Data parallelism {rail=Data parallel}

{lede} ==Data parallelism copies the whole model to every GPU and gives each GPU a different slice of the batch.== Each GPU computes the gradient of its slice, the GPUs [[average|Add the gradients of all GPUs and divide by their number. The result is the same gradient a single GPU would have computed on the whole batch.]] their gradients, and everyone takes the same step. It is the simplest and most common way to use more GPUs.

### Copy, split, average

Imagine four teaching assistants, each with an identical copy of the answer key. A pile of 400 homework sheets is split into four piles of 100. Each assistant marks their pile and writes down how many sheets had each kind of mistake. They then add the four tallies and agree on the new advice to give the class. Every assistant ends with the same advice because they all used the same four tallies. That is ==one step of data parallelism==.

```viz
{
  "type": "pipeline",
  "title": "One data-parallel training step",
  "intro": "Step through one training step on 4 GPUs.",
  "nodes": [
    { "id": "copy", "label": "Same model", "sub": "on every GPU" },
    { "id": "split", "label": "Split the batch", "sub": "one slice per GPU" },
    { "id": "grad", "label": "Local gradients", "sub": "forward + backward" },
    { "id": "avg", "label": "All-reduce", "sub": "average gradients" },
    { "id": "step", "label": "Same update", "sub": "weights stay equal" }
  ],
  "steps": [
    { "title": "Every GPU starts with the same weights", "body": "The model is copied to each GPU. All copies hold identical numbers at the start of every step.", "active": ["copy"], "example": "4 GPUs, 7 B parameters each." },
    { "title": "The batch is split", "body": "A global batch is cut into equal slices, one per GPU, so each GPU sees different examples.", "active": ["split"], "example": "Global batch of 256 → 64 examples per GPU." },
    { "title": "Each GPU computes its own gradient", "body": "Each GPU runs the forward and backward pass on its slice only, with no communication. The gradients differ because the slices differ.", "active": ["grad"], "example": "Four different gradient vectors, each of 7 billion numbers." },
    { "title": "The gradients are averaged (all-reduce)", "body": "An all-reduce adds the four gradient vectors and divides by four, and gives every GPU the same result. This is the only communication in the step.", "active": ["avg"], "example": "Sum of 4 gradients ÷ 4 = the average, available on every GPU." },
    { "title": "Everyone takes the same step", "body": "Each GPU applies the optimiser to the same averaged gradient, so all copies of the weights remain identical for the next step.", "active": ["step", "copy"], "example": "w ← w − learning rate × average gradient." }
  ],
  "note": "A conceptual diagram; real frameworks overlap the communication with the backward pass."
}
```

Notice that nothing is communicated until the gradients are done. The one all-reduce per step is the price of keeping all copies equal.

```viz
{
  "type": "flow",
  "title": "One GPU against four GPUs",
  "modes": [
    {
      "name": "One GPU",
      "tagline": "One copy of the model sees the whole batch of 256, one step at a time.",
      "inputLabel": "Input: a batch of 256 examples",
      "modelLabel": "one GPU, full model",
      "outputLabel": "Output: one gradient, one update",
      "examples": [
        { "label": "Time", "input": "256 examples, 400 ms of compute per 64 examples", "output": "4 × 400 ms = 1,600 ms for the step (assumed example speeds)." },
        { "label": "Memory", "input": "7 B model, 16 bytes per parameter", "output": "112 GB of weights, gradients and optimiser state on this one GPU." }
      ],
      "note": "Assumed example timings."
    },
    {
      "name": "Four GPUs (data parallel)",
      "tagline": "Four copies each see 64 examples, then average their gradients.",
      "inputLabel": "Input: a batch of 256 split in four slices of 64",
      "modelLabel": "four GPUs, full model on each",
      "outputLabel": "Output: the same single update, on every GPU",
      "examples": [
        { "label": "Time", "input": "64 examples per GPU, 400 ms of compute, plus the all-reduce", "output": "400 ms of compute plus the communication time (assumed): close to 4× faster if the communication is small." },
        { "label": "Memory", "input": "7 B model, 16 bytes per parameter", "output": "Still 112 GB on every GPU: data parallelism does not make the model smaller." }
      ],
      "note": "Speeds are assumed. The time is shorter, the memory per GPU is not."
    }
  ]
}
```

Compare the two Memory lines: they are identical. Data parallelism is a cure for the time limit, not for the memory limit.

### Why averaging is exact

A reasonable worry is that four small gradients, averaged, might differ from the gradient of one big batch. For equal-sized slices it does not. The loss of a batch is the mean of the losses of its examples, and the gradient of a mean is the mean of the gradients. Splitting into equal shards and averaging the shard means gives back ==exactly the same number==.

```viz
{
  "type": "equation",
  "title": "Why the average of shard gradients equals the batch gradient",
  "intro": "Toy model: prediction = w × x and loss (w × x − y)² for each example, with w = 1. Select the parts, then follow the two shards.",
  "formula": [ { "t": "g", "id": "g" }, " = ", { "t": "1/K", "id": "k" }, " × (", { "t": "g₁", "id": "g1" }, " + ", { "t": "g₂", "id": "g2" }, " + … + ", { "t": "g_K", "id": "gk" }, ")" ],
  "parts": [
    { "id": "g", "name": "g (batch gradient)", "meaning": "The gradient of the mean loss over the whole batch: what one GPU would compute on all the examples." },
    { "id": "k", "name": "1/K", "meaning": "One over the number of GPUs (shards). It works only when the shards are the same size." },
    { "id": "g1", "name": "g₁ (gradient of shard 1)", "meaning": "The mean of the per-example gradients of the examples on GPU 1." },
    { "id": "g2", "name": "g₂ (gradient of shard 2)", "meaning": "The same for GPU 2." },
    { "id": "gk", "name": "g_K (gradient of the last shard)", "meaning": "The same for the last GPU." }
  ],
  "steps": [
    { "title": "Per-example gradients 2(wx − y)x with w = 1", "expr": "Shard 1: (x=1, y=3) → −4;  (x=2, y=2) → 0.   Shard 2: (x=3, y=6) → −18;  (x=1, y=0) → 2", "note": "For example, 2 × (1×1 − 3) × 1 = −4." },
    { "title": "Mean gradient of each shard", "expr": "g₁ = (−4 + 0) ÷ 2 = −2;   g₂ = (−18 + 2) ÷ 2 = −8", "note": "Each GPU computes this alone." },
    { "title": "Average the shard gradients", "expr": "(g₁ + g₂) ÷ 2 = (−2 − 8) ÷ 2 = −5", "note": "This is what the all-reduce delivers." },
    { "title": "Compare with the whole batch on one GPU", "expr": "(−4 + 0 − 18 + 2) ÷ 4 = −20 ÷ 4 = −5", "note": "The same number. If the shards had different sizes you would weight each by its size." }
  ]
}
```

Notice that the last two lines agree to the digit. This is why data parallelism changes the speed, not the result of training; it is also what you check in the lab.

The global batch is the sum of the slices, so with N GPUs each holding b examples the ==effective batch size is N × b==. A larger batch usually asks for a retuned learning rate; a common rule of thumb is to raise it with the batch, but the best value must be found by experiment.

```viz
{
  "type": "equation",
  "title": "Effective batch size",
  "intro": "Select the parts, then grow the cluster.",
  "formula": [ { "t": "B", "id": "b" }, " = ", { "t": "N", "id": "n" }, " × ", { "t": "b", "id": "per" } ],
  "parts": [
    { "id": "b", "name": "B (effective batch size)", "meaning": "The number of examples that contribute to one weight update." },
    { "id": "n", "name": "N (number of GPUs)", "meaning": "How many data-parallel copies of the model." },
    { "id": "per", "name": "b (examples per GPU)", "meaning": "The slice each GPU processes in one step." }
  ],
  "steps": [
    { "title": "8 GPUs, 32 examples each", "expr": "B = 8 × 32 = 256", "note": "One update uses 256 examples." },
    { "title": "64 GPUs, the same 32 each", "expr": "B = 64 × 32 = 2,048", "note": "Eight times the batch: fewer, larger steps per epoch, so the learning rate often needs retuning." }
  ]
}
```

Notice that adding GPUs changes the batch, not only the speed. That is a hidden knob you must watch.

### The price: communication

The [[all-reduce|A collective operation in which every GPU contributes a tensor and every GPU receives the sum (or average) of all of them.]] moves gradients over the links between GPUs. In the common ring all-reduce, each GPU sends and receives about ==2(N − 1)/N times the size S of the gradients==. For large N this tends to 2S, so it hardly grows as you add GPUs. A [[link|The connection between GPUs, such as a fast link inside one machine or a network between machines. Its bandwidth in GB per second sets how fast an all-reduce can finish.]] that is too slow makes the GPUs wait.

```viz
{
  "type": "equation",
  "title": "All-reduce traffic per GPU",
  "intro": "Select the parts, then run the 7 B example. The link speed is an assumed value.",
  "formula": [ { "t": "traffic", "id": "tr" }, " = 2 × ", { "t": "(N − 1) / N", "id": "ring" }, " × ", { "t": "S", "id": "s" } ],
  "parts": [
    { "id": "tr", "name": "traffic (bytes per GPU)", "meaning": "How many bytes each GPU sends and receives during one all-reduce." },
    { "id": "ring", "name": "(N − 1) / N", "meaning": "N is the number of GPUs. For N = 2 it is 1/2; for N = 8 it is 7/8; for large N it approaches 1." },
    { "id": "s", "name": "S (gradient bytes)", "meaning": "Number of gradient values times bytes per gradient. For 7 B gradients in 16 bits: 14 GB." }
  ],
  "steps": [
    { "title": "Size of the gradients", "expr": "S = 7 × 10⁹ × 2 bytes = 14 GB", "note": "One gradient per parameter, 2 bytes each." },
    { "title": "Traffic for N = 8", "expr": "traffic = 2 × 7/8 × 14 GB = 24.5 GB", "note": "Each GPU sends and receives 24.5 GB per step." },
    { "title": "Time at an assumed 100 GB/s", "expr": "24.5 GB ÷ 100 GB/s = 0.245 s = 245 ms", "note": "A quarter of a second of waiting per step if nothing overlaps." }
  ]
}
```

Notice that 245 ms is not small if a step of compute takes only 400 ms. That brings us to the efficiency.

Efficiency compares the useful work (compute) with the total time (compute plus waiting for communication). In the simplest model, where nothing overlaps: ==efficiency = compute ÷ (compute + communication)==. A speed-up of 8 on 8 GPUs would be an efficiency of 1; real clusters get less.

```viz
{
  "type": "equation",
  "title": "Efficiency of data parallelism",
  "intro": "Select the parts and compute the efficiency of 8 GPUs. Times are assumed example values.",
  "formula": [ { "t": "E", "id": "e" }, " = ", { "t": "t_comp", "id": "tc" }, " ÷ (", { "t": "t_comp", "id": "tc2" }, " + ", { "t": "t_comm", "id": "tm" }, ")" ],
  "parts": [
    { "id": "e", "name": "E (efficiency)", "meaning": "A number between 0 and 1: the share of each step spent on useful computing." },
    { "id": "tc", "name": "t_comp (compute time)", "meaning": "Milliseconds that a GPU needs for the forward and backward pass of its slice." },
    { "id": "tc2", "name": "t_comp again", "meaning": "The same compute time, added to the communication time to give the whole step." },
    { "id": "tm", "name": "t_comm (communication time)", "meaning": "Milliseconds spent in the all-reduce: traffic ÷ link speed." }
  ],
  "steps": [
    { "title": "Compute and communication (assumed)", "expr": "t_comp = 400 ms,  t_comm = 245 ms (7 B model, N = 8, 100 GB/s)", "note": "From the previous example." },
    { "title": "Efficiency", "expr": "E = 400 ÷ (400 + 245) = 0.62", "note": "62% of the time is useful." },
    { "title": "Speed-up on 8 GPUs", "expr": "speed-up = 8 × 0.62 = 4.96", "note": "About 5 times faster than one GPU, not 8. A 4× faster link (400 GB/s, assumed) gives comm 61 ms and E = 0.87." }
  ]
}
```

Notice that the efficiency depends on the ratio of the two times, not on either alone. A faster link, bigger slices (more compute per step) or overlapping communication with computation all raise it.

```viz
{
  "type": "calc",
  "title": "Scaling data parallelism",
  "kind": "dp_scaling",
  "computeMs": 400,
  "bytesPerGrad": 2,
  "intro": "Move the model size, the number of GPUs and the link speed. Find a setting where adding GPUs stops paying off.",
  "note": "Assumed compute time of 400 ms per step and no overlap of communication with computation. Real clusters overlap them and use faster links; use this to build intuition, not to predict a real cluster."
}
```

Slide the link speed down and the efficiency collapses; slide it up and it returns. The number of GPUs matters much less than the link, because the traffic per GPU is almost the same for 8 or 800 GPUs.

### The limit, and what fixes it

Data parallelism has one big limit: ==every GPU still holds the whole model, its gradients and its optimiser state==. A 7 B model needs 112 GB on each of the eight GPUs: copying it eight times does not help the warehouse. Methods such as [[ZeRO|Zero Redundancy Optimizer (Rajbhandari et al., 2019): spread the optimiser state, gradients and, at the highest level, the parameters over the data-parallel GPUs instead of copying them on each.]] and FSDP (fully sharded data parallel) keep the data-parallel recipe but spread the optimiser state, the gradients and even the parameters across the GPUs, gathering them only when needed. You only need the names here.

```viz
{
  "type": "sorter",
  "title": "What travels between GPUs?",
  "intro": "Decide, for each card, what must be communicated in the normal form of each parallelism.",
  "bins": [ { "id": "gradients", "label": "Gradients (once per step)" }, { "id": "activations", "label": "Activations (inside the model)" }, { "id": "nothing", "label": "Nothing between the GPUs" } ],
  "cards": [
    { "name": "Data parallel: end of the backward pass", "answer": "gradients", "why": "The GPUs average their gradients with an all-reduce so that their weights stay equal." },
    { "name": "Pipeline parallel: from stage 1 to stage 2", "answer": "activations", "why": "The output of one stage is the input of the next, so activations cross the stage boundary." },
    { "name": "Tensor parallel: inside every layer", "answer": "activations", "why": "Partial results of a split matrix product are joined, which means an all-reduce of activations in every layer." },
    { "name": "Data parallel: the forward pass on a slice", "answer": "nothing", "why": "Each GPU works on its own slice with its own copy; there is no communication until the gradients are ready." },
    { "name": "Data parallel: each GPU loads its slice of the batch", "answer": "nothing", "why": "Loading data is local; no GPU-to-GPU traffic is needed." },
    { "name": "Two independent inference replicas serving different users", "answer": "nothing", "why": "Replicas never talk to each other; they simply share the incoming requests, which is the cheapest way to add capacity." },
    { "name": "Backward pass through a pipeline", "answer": "activations", "why": "The gradients with respect to activations travel back across stage boundaries, so a pipeline sends data in both directions." }
  ]
}
```

Notice the card about two independent replicas: for serving, the cheapest scale-out is often just more copies behind a load balancer.

### When to use it

Use data parallelism ==when the model (with its optimiser state) fits on one GPU and you want to train faster==. It needs only one all-reduce of the gradients per step, so it works even over slower links, and it is easy to use. When the model no longer fits, you need one of the next two splits, perhaps combined with data parallelism.

## Tensor (model) parallelism {rail=Tensor}

{lede} ==Tensor parallelism splits one layer's weight matrices across GPUs==, so that each GPU holds only a slice of every layer and does only its share of every matrix product. It cures the memory limit even when one layer is too big for one GPU, but the GPUs have to exchange results inside every layer.

### One matrix product, two ways to cut it

A layer computes Y = X·W, where X is the input (rows are tokens or examples) and W is the weight matrix. There are two natural ways to divide W between two GPUs.

- **Column split.** GPU 1 keeps the left columns of W and GPU 2 keeps the right columns. Each computes the matching columns of Y from the full X. ==To join them you simply put the columns next to each other (concatenate)==, with no addition.
- **Row split.** GPU 1 keeps the top rows of W and GPU 2 keeps the bottom rows. Each needs only the matching columns of X. Each produces a full-size partial Y; ==to join them you add them (sum): an all-reduce==.

```viz
{
  "type": "tensorsplit",
  "title": "Splitting Y = X·W over 2 or 4 devices",
  "intro": "Choose column or row, and 2 or 4 devices. Compare each device's slice and the way the results are joined.",
  "X": [[1,2,3,4],[0,1,0,2]],
  "W": [[1,0,2,1],[0,1,1,2],[2,1,0,1],[1,2,1,0]],
  "mode": "column",
  "n": 2,
  "note": "Small integer matrices chosen so that you can check every number by hand.",
  "checks": [
    { "mode": "column", "n": 2, "Y": [[11,13,8,8],[2,5,3,2]] },
    { "mode": "row", "n": 2, "Y": [[11,13,8,8],[2,5,3,2]] },
    { "mode": "row", "n": 4, "Y": [[11,13,8,8],[2,5,3,2]] }
  ]
}
```

Notice that all the splits give the same Y, which is the product computed on one device. In the column split the first row of Y is 11, 13 from GPU 1 followed by 8, 8 from GPU 2. In the row split the first row is (1, 2, 4, 5) + (10, 11, 4, 3) = (11, 13, 8, 8).

```viz
{
  "type": "tensorsplit",
  "title": "A second example: different numbers, same rule",
  "intro": "Try both modes again on different matrices and check the joined result by hand.",
  "X": [[2,0,1,1],[1,1,1,0]],
  "W": [[1,2,0,1],[3,1,1,0],[0,1,2,1],[1,0,1,2]],
  "mode": "row",
  "n": 2,
  "note": "Another small integer example. The result of the split always equals X·W.",
  "checks": [
    { "mode": "column", "n": 2, "Y": [[3,5,3,5],[4,4,3,2]] },
    { "mode": "row", "n": 2, "Y": [[3,5,3,5],[4,4,3,2]] },
    { "mode": "row", "n": 4, "Y": [[3,5,3,5],[4,4,3,2]] }
  ]
}
```

Choose the row split with four devices: two devices get a block of zeros for one row of X, which is allowed. The result is still correct. The only question is which split moves the least data, and that is what a Transformer block decides.

```viz
{
  "type": "equation",
  "title": "Column split and row split of Y = X·W",
  "intro": "Select the parts. The worked example uses the first row of X = (1, 2, 3, 4) and the first example matrix.",
  "formula": [ { "t": "Y", "id": "y" }, " = ", { "t": "X", "id": "x" }, "·[", { "t": "W₁", "id": "w1" }, " | ", { "t": "W₂", "id": "w2" }, "] = [", { "t": "X·W₁", "id": "y1" }, " | ", { "t": "X·W₂", "id": "y2" }, "]   (columns)" ],
  "parts": [
    { "id": "y", "name": "Y (the layer's output)", "meaning": "The same result you would get on one device." },
    { "id": "x", "name": "X (the input)", "meaning": "In a column split every device receives the whole of X." },
    { "id": "w1", "name": "W₁ (left columns of W)", "meaning": "The slice that GPU 1 stores." },
    { "id": "w2", "name": "W₂ (right columns of W)", "meaning": "The slice that GPU 2 stores." },
    { "id": "y1", "name": "X·W₁", "meaning": "GPU 1's result: the left columns of Y." },
    { "id": "y2", "name": "X·W₂", "meaning": "GPU 2's result: the right columns of Y. The two pieces are put side by side." }
  ],
  "steps": [
    { "title": "GPU 1 holds the left two columns", "expr": "X·W₁ = (1,2,3,4) · [[1,0],[0,1],[2,1],[1,2]] = (1+0+6+4, 0+2+3+8) = (11, 13)", "note": "First row of Y, left half." },
    { "title": "GPU 2 holds the right two columns", "expr": "X·W₂ = (1,2,3,4) · [[2,1],[1,2],[0,1],[1,0]] = (2+2+0+4, 1+4+3+0) = (8, 8)", "note": "First row of Y, right half." },
    { "title": "Concatenate", "expr": "Y = [(11, 13) | (8, 8)] = (11, 13, 8, 8)", "note": "No addition needed. For a row split, the partial results are added instead: Y = X₁·W₁ + X₂·W₂." }
  ]
}
```

Notice the difference in what the devices send afterwards: the column split ends with devices holding different pieces of the output, which the next layer may be able to use as they are. The row split ends with devices holding full-size partial sums that must be added.

```viz
{
  "type": "flow",
  "title": "Column split against row split",
  "modes": [
    {
      "name": "Column split",
      "tagline": "Each device holds some columns of W and makes the matching columns of Y.",
      "inputLabel": "Input: the full X on every device",
      "modelLabel": "device k holds columns of W",
      "outputLabel": "Output: a slice of the columns of Y",
      "examples": [
        { "label": "First feed-forward matrix", "input": "X (tokens × d) and W (d × 4d) split into 2 column halves", "output": "Each device gets 2d columns of the wide hidden layer. No communication yet." },
        { "label": "Joining", "input": "Two column slices of Y", "output": "Concatenate when you really need the full Y; often you do not need to." }
      ],
      "note": "The matrix sizes are those of a typical Transformer feed-forward network (hidden width 4d)."
    },
    {
      "name": "Row split",
      "tagline": "Each device holds some rows of W and the matching columns of X, and makes a partial Y.",
      "inputLabel": "Input: a slice of the columns of X on each device",
      "modelLabel": "device k holds rows of W",
      "outputLabel": "Output: a full-size partial Y that must be summed",
      "examples": [
        { "label": "Second feed-forward matrix", "input": "The hidden slice from the first step and the matching rows of the second matrix", "output": "Each device makes a partial output of size tokens × d." },
        { "label": "Joining", "input": "Partial outputs from each device", "output": "Add them with one all-reduce: the only communication of the whole block." }
      ],
      "note": "This is how the two halves of the feed-forward network are joined."
    }
  ]
}
```

Read the two examples side by side: the column split makes the wide hidden layer in slices, and the row split turns those slices back into a full-size output with one sum. That is exactly how Transformer blocks are split.

### The Transformer block

Shoeybi et al. (2019, Megatron-LM) noticed that a Transformer block can be split so that each block needs very little communication. In the [[feed-forward network|The two-matrix part of a Transformer block: expand d numbers to a wider hidden layer (often 4d), apply an activation, then project back to d.]] the first matrix is split by columns, so each GPU makes its own slice of the hidden layer. The activation is applied to each slice separately (no communication). The second matrix is split by rows, so each GPU makes a partial output, and ==one all-reduce adds the partial outputs==. In [[multi-head attention|Attention computed in several heads in parallel, each with its own queries, keys and values (Week 7).]] the heads are independent, so different heads can live on different GPUs. One all-reduce per pair of matrices is all that is needed in the forward pass.

```viz
{
  "type": "pipeline",
  "title": "A feed-forward block split over two GPUs",
  "intro": "Step through the block. Note where the GPUs talk to each other.",
  "nodes": [
    { "id": "x", "label": "Input X", "sub": "same on both GPUs" },
    { "id": "w1", "label": "Matrix 1", "sub": "split by columns" },
    { "id": "act", "label": "Activation", "sub": "on each slice" },
    { "id": "w2", "label": "Matrix 2", "sub": "split by rows" },
    { "id": "sum", "label": "All-reduce", "sub": "add the partials" }
  ],
  "steps": [
    { "title": "Both GPUs hold the same input", "body": "The block's input X (tokens × d) is available on both GPUs, which is the result of the previous all-reduce.", "active": ["x"], "example": "X has 2,048 tokens × 4,096 numbers per example." },
    { "title": "Matrix 1 is split by columns", "body": "Each GPU multiplies X by its half of the first matrix and makes half of the hidden layer. There is no communication.", "active": ["x", "w1"], "example": "Hidden width 16,384 → 8,192 on each GPU." },
    { "title": "The activation acts on each slice", "body": "A non-linear activation works on each hidden number independently, so each GPU applies it to its own slice with no communication.", "active": ["act"], "example": "GELU on 8,192 numbers per token, on each GPU." },
    { "title": "Matrix 2 is split by rows", "body": "Each GPU multiplies its slice of the hidden layer by the matching rows of the second matrix. This gives a full-size partial result.", "active": ["w2"], "example": "Each GPU makes tokens × 4,096 partial numbers." },
    { "title": "One all-reduce adds the partial results", "body": "The two partial outputs are added so that both GPUs hold the complete block output. This is the only communication in the block.", "active": ["sum", "x"], "example": "Output = partial 1 + partial 2, on both GPUs." }
  ],
  "note": "A conceptual diagram of the Megatron-style split; the dimensions are those of an example model."
}
```

Notice that the two matrices are split in different ways on purpose: column then row means the slice produced by one is exactly the slice the other needs, so the only joining happens at the end.

### What it saves and what it costs

Each of N GPUs stores only ==1/N of the weights==, so the weights per GPU are ==M ÷ N==. A 70 B model in FP16 (140 GB) over 4 GPUs is 35 GB each. Nothing is free: the all-reduce moves a tensor of activations of size batch × tokens × d in each block, and it sits on the critical path, because the next layer cannot start before the sum is ready.

```viz
{
  "type": "equation",
  "title": "Memory per GPU under tensor parallelism",
  "intro": "Select the parts and run the 70 B example.",
  "formula": [ { "t": "M_gpu", "id": "mg" }, " = ", { "t": "M", "id": "m" }, " ÷ ", { "t": "N", "id": "n" } ],
  "parts": [
    { "id": "mg", "name": "M_gpu (weights per GPU)", "meaning": "The memory each GPU needs for its slice of the weights." },
    { "id": "m", "name": "M (weights of the whole model)", "meaning": "The bytes of the unsplit weights, parameters × bytes each." },
    { "id": "n", "name": "N (tensor-parallel GPUs)", "meaning": "How many GPUs share every layer." }
  ],
  "steps": [
    { "title": "70 B in FP16", "expr": "M = 70 × 10⁹ × 2 = 140 GB", "note": "Too big for one 80 GB GPU (an example size)." },
    { "title": "Split over 4 GPUs", "expr": "M_gpu = 140 ÷ 4 = 35 GB", "note": "Each GPU now has room left for a KV cache and activations." }
  ]
}
```

Notice that this is the cure for the warehouse problem that data parallelism could not give. The bill arrives as communication.

```viz
{
  "type": "equation",
  "title": "Size of one tensor-parallel all-reduce",
  "intro": "Select the parts and compute the size for an example model.",
  "formula": [ { "t": "A", "id": "a" }, " = ", { "t": "B", "id": "b" }, " × ", { "t": "n", "id": "n" }, " × ", { "t": "d", "id": "d" }, " × ", { "t": "bytes", "id": "by" } ],
  "parts": [
    { "id": "a", "name": "A (bytes in one all-reduce)", "meaning": "The size of the activation tensor that must be summed across the GPUs." },
    { "id": "b", "name": "B (batch size)", "meaning": "How many sequences are processed together." },
    { "id": "n", "name": "n (tokens per sequence)", "meaning": "The number of tokens in each sequence." },
    { "id": "d", "name": "d (model width)", "meaning": "The size of the vector of each token, d_model." },
    { "id": "by", "name": "bytes per value", "meaning": "2 for 16-bit activations." }
  ],
  "steps": [
    { "title": "Example sizes", "expr": "B = 8,  n = 2,048,  d = 4,096,  16-bit (2 bytes)", "note": "A typical training micro-batch for a 7 B-class model (assumed)." },
    { "title": "Bytes per all-reduce", "expr": "A = 8 × 2,048 × 4,096 × 2 = 134,217,728 bytes ≈ 134 MB", "note": "And a block needs about two of these in the forward pass (one for attention, one for the feed-forward network), and more in the backward pass." }
  ]
}
```

Notice how big it is: 134 MB per all-reduce, many times per step, and each one blocks the next layer. That is why tensor parallelism is used ==inside one machine, where GPUs are joined by very fast links==, and rarely across slower network links.

```viz
{
  "type": "sorter",
  "title": "Concatenate or sum?",
  "intro": "For each situation, decide whether the pieces from the devices are joined by concatenation or by a sum (an all-reduce).",
  "bins": [ { "id": "concat", "label": "Concatenate (put side by side)" }, { "id": "sum", "label": "Sum (all-reduce)" } ],
  "cards": [
    { "name": "Column split of a weight matrix", "answer": "concat", "why": "Each device makes different columns of Y, so the pieces are placed next to each other." },
    { "name": "Row split of a weight matrix", "answer": "sum", "why": "Each device makes a full-size partial result, and the true Y is their sum." },
    { "name": "Second matrix of the feed-forward network (split by rows)", "answer": "sum", "why": "The partial outputs must be added; this is the one all-reduce of the block." },
    { "name": "First matrix of the feed-forward network (split by columns)", "answer": "concat", "why": "Each device keeps its own slice of the hidden layer; nothing is added." },
    { "name": "Attention heads on different GPUs, joined before the output projection", "answer": "concat", "why": "The head outputs are concatenated to form the attention output, as in Week 7." },
    { "name": "Output projection of attention (split by rows)", "answer": "sum", "why": "Each GPU holds the rows that match its heads and makes a partial output to add up." },
    { "name": "Gradient averaging in data parallelism", "answer": "sum", "why": "The all-reduce adds the gradients (and divides by N). It is also a sum." },
    { "name": "Two halves of a hidden layer kept on separate devices between two matrices", "answer": "concat", "why": "Nothing needs to be joined, because the next row-split matrix uses exactly those slices." }
  ]
}
```

Notice that sums need communication while concatenation can often be skipped altogether. Splitting column-then-row is the trick that postpones the only sum to the end of the block.

### When to use it

Use tensor parallelism ==when a layer or a model is too big for one GPU and the GPUs are connected by fast links==, typically the GPUs in one machine. It is the only one of the three that lowers the memory of each individual layer. Its cost is frequent communication, so you rarely stretch it across machines.

## Pipeline parallelism {rail=Pipeline}

{lede} ==Pipeline parallelism cuts the model into consecutive stages of layers and puts each stage on its own GPU==, like an assembly line. One batch is cut into small [[micro-batches|Small pieces of a batch sent one after another through the pipeline, so that all stages can work on different pieces at the same time.]] that flow through the stages. It lets a model that is too deep for one GPU fit, and it needs very little communication.

### The assembly line

Imagine washing, drying, ironing and folding laundry, with one person per job. If you give them one big basket, three people wait while the first works, then two wait, and so on. If you split the laundry into small bundles, then as soon as the washer finishes bundle 1 she starts bundle 2 while the dryer takes bundle 1. After the first few bundles everybody is busy. The idle time at the start (the line filling) and at the end (the line emptying) is the [[pipeline bubble|The idle time at the start and the end of a pipeline, when some stages have nothing to work on because the first or last micro-batches have not reached them yet.]].

```viz
{
  "type": "flow",
  "title": "A model that is too deep for one GPU",
  "modes": [
    {
      "name": "One GPU",
      "tagline": "All 80 layers must fit on one device.",
      "inputLabel": "Input: a batch",
      "modelLabel": "one GPU holding layers 1 to 80",
      "outputLabel": "Output: the loss",
      "examples": [
        { "label": "Memory", "input": "70 B model, FP16 weights 140 GB", "output": "Does not fit on one example GPU of 80 GB." },
        { "label": "Result", "input": "Try to load it anyway", "output": "An out-of-memory error before the first step." }
      ],
      "note": "The 80 GB GPU is an assumed example size."
    },
    {
      "name": "Four pipeline stages",
      "tagline": "Each GPU holds 20 consecutive layers; micro-batches flow from stage to stage.",
      "inputLabel": "Input: micro-batches, one after another",
      "modelLabel": "GPU 1 (layers 1–20) → GPU 2 → GPU 3 → GPU 4 (layers 61–80)",
      "outputLabel": "Output: the loss from the last stage",
      "examples": [
        { "label": "Memory", "input": "140 GB of weights split in four stages", "output": "About 35 GB of weights per GPU, with room for activations and a cache." },
        { "label": "Communication", "input": "A micro-batch of 2 × 2,048 tokens × 4,096 numbers × 2 bytes", "output": "33.6 MB crosses each stage boundary once per micro-batch: far less than the many all-reduces of tensor parallelism." }
      ],
      "note": "Illustrative layer counts. 2 × 2,048 × 4,096 × 2 = 33,554,432 bytes."
    }
  ]
}
```

Compare the two Memory lines: splitting by layers also gets the weights to fit, but the traffic is a trickle compared with tensor parallelism, because only the activations at the boundaries travel.

### The bubble

In the schedule below, each row is one GPU (one stage) and each column is one time step. Each coloured cell is one micro-batch being processed by that stage; the outlined cells are the backward pass; grey cells are idle. This is the schedule of GPipe (Huang et al., 2019): all the forward passes first, then all the backward passes.

```viz
{
  "type": "pipesched",
  "title": "A pipeline with 4 stages and 4 micro-batches",
  "intro": "Look at the grey cells in the corners. Then slide m up and watch them become a smaller share of the whole.",
  "p": 4,
  "m": 4,
  "backward": true,
  "note": "One time step is one micro-batch on one stage, and a backward step is taken to cost the same as a forward step (a simplification).",
  "checks": [ { "p": 4, "m": 4, "backward": true, "total": 14, "bubble": 0.4286 }, { "p": 4, "m": 1, "backward": true, "total": 8, "bubble": 0.75 } ]
}
```

With 4 stages and only 4 micro-batches, 43% of all the cells are grey. A GPU is idle for nearly half of the time: that is expensive.

```viz
{
  "type": "pipesched",
  "title": "The same pipeline with 16 micro-batches",
  "intro": "Same 4 stages, four times more micro-batches. Compare the grey corners with the previous figure.",
  "p": 4,
  "m": 16,
  "backward": true,
  "note": "The corners stay the same size (the pipeline still fills and empties) but the busy middle becomes much longer.",
  "checks": [ { "p": 4, "m": 16, "backward": true, "total": 38, "bubble": 0.1579 }, { "p": 4, "m": 8, "backward": true, "total": 22, "bubble": 0.2727 } ]
}
```

The corners did not change, but the middle grew, so the idle share fell to 16%. This is the whole idea: ==more micro-batches shrink the bubble==.

```viz
{
  "type": "equation",
  "title": "The idle share of a pipeline",
  "intro": "Select the parts, then compare 8 and 32 micro-batches on 4 stages.",
  "formula": [ { "t": "idle", "id": "i" }, " = ", { "t": "(p − 1)", "id": "p1" }, " ÷ (", { "t": "m", "id": "m" }, " + ", { "t": "p − 1", "id": "p2" }, ")" ],
  "parts": [
    { "id": "i", "name": "idle (idle share)", "meaning": "The fraction of GPU time that is wasted in the bubble. 0 means no waste." },
    { "id": "p1", "name": "p − 1", "meaning": "p is the number of stages. The pipeline needs p − 1 steps to fill and p − 1 to empty." },
    { "id": "m", "name": "m (micro-batches)", "meaning": "How many pieces the batch is cut into. More pieces make the useful middle longer." },
    { "id": "p2", "name": "p − 1 (in the denominator)", "meaning": "The same fill-and-empty steps, added to the m steps of useful work to give the total length of the forward pass." }
  ],
  "steps": [
    { "title": "p = 4, m = 8", "expr": "idle = 3 ÷ (8 + 3) = 3 ÷ 11 = 0.273 = 27.3%", "note": "More than a quarter of the time is wasted." },
    { "title": "p = 4, m = 32", "expr": "idle = 3 ÷ (32 + 3) = 3 ÷ 35 = 0.086 = 8.6%", "note": "Four times more micro-batches cut the waste by a factor of three." },
    { "title": "More stages, same m", "expr": "p = 8, m = 8: idle = 7 ÷ 15 = 46.7%", "note": "Splitting into more stages makes the bubble worse unless you raise m too." }
  ]
}
```

Notice the last step: asking for more stages to fit a bigger model costs you more idle time, so you must also raise m.

```viz
{
  "type": "equation",
  "title": "Total time of a GPipe step",
  "intro": "Select the parts, then count the time steps for p = 4, m = 8.",
  "formula": [ { "t": "T", "id": "t" }, " = ", { "t": "2", "id": "two" }, " × (", { "t": "m", "id": "m" }, " + ", { "t": "p − 1", "id": "p" }, ")" ],
  "parts": [
    { "id": "t", "name": "T (time steps for forward and backward)", "meaning": "The total length of the schedule, in units of one micro-batch on one stage." },
    { "id": "two", "name": "2 (forward and backward)", "meaning": "The backward pass takes about as long as the forward pass in this simplified model." },
    { "id": "m", "name": "m (micro-batches)", "meaning": "Each stage processes m micro-batches in each pass." },
    { "id": "p", "name": "p − 1 (fill and empty)", "meaning": "The extra steps before the last stage gets its first micro-batch." }
  ],
  "steps": [
    { "title": "Forward pass", "expr": "m + p − 1 = 8 + 3 = 11 steps", "note": "The last micro-batch leaves the last stage at step 11." },
    { "title": "Forward and backward", "expr": "T = 2 × 11 = 22 steps", "note": "Matches the figure when you set p = 4, m = 8." },
    { "title": "Useful work and waste per GPU", "expr": "busy = 2 × 8 = 16 of 22 steps; idle = 6 ÷ 22 = 27.3%", "note": "The same 27.3% as before." }
  ]
}
```

Notice that the total time grows more slowly than the work: doubling m from 16 to 32 does not double T, since the fill-and-empty cost p − 1 is paid once.

There is a price for more micro-batches. Each micro-batch is smaller, so a GPU does less work per kernel launch and is usually less efficient on small matrices. Also, a GPipe-style schedule must keep the activations of all micro-batches in flight until their backward pass, so ==more micro-batches mean more activation memory==, unless you recompute them. In practice you choose m a few times larger than p, as a compromise.

### What travels, and what does not

Only the ==activations at the boundary== between two stages travel, once per micro-batch in the forward pass and the matching gradients in the backward pass. That is a small amount of data compared with the all-reduces of tensor parallelism, so pipeline parallelism ==tolerates slower links==, for example the network between machines.

### Choosing between the three

```viz
{
  "type": "pipeline",
  "title": "Which parallelism should I use?",
  "intro": "Step through the questions in order. The answer to each one is the next step.",
  "nodes": [
    { "id": "fit", "label": "Fits one GPU?", "sub": "weights + optimiser state" },
    { "id": "data", "label": "Data parallel", "sub": "train faster" },
    { "id": "layer", "label": "A layer too big, fast links?", "sub": "inside one machine" },
    { "id": "tensor", "label": "Tensor parallel", "sub": "split the matrices" },
    { "id": "pipe", "label": "Pipeline parallel", "sub": "split the layers" },
    { "id": "all", "label": "Combine all three", "sub": "3D parallelism" }
  ],
  "steps": [
    { "title": "Does the model fit on one GPU?", "body": "Add the weights, gradients and optimiser state (16 bytes per parameter for training) and the activations, and compare with the memory of one GPU.", "active": ["fit"], "example": "A 1.3 B model: 16 × 1.3 = 20.8 GB, which fits in 80 GB (assumed)." },
    { "title": "Yes: use data parallelism", "body": "If it fits, copy it to many GPUs and split the batch: the simplest way to gain speed, with one all-reduce per step.", "active": ["fit", "data"], "example": "Fine-tuning a small classifier on 8 GPUs." },
    { "title": "No, and the GPUs share fast links: use tensor parallelism", "body": "If layers are too large, or you want to cut the memory of every layer, split the matrices across the GPUs of one machine, where fast links make the frequent all-reduces affordable.", "active": ["layer", "tensor"], "example": "A 70 B model on the 4 or 8 GPUs of one server." },
    { "title": "No, very deep model, slower links: use pipeline parallelism", "body": "If the model is deep and the GPUs sit in different machines, cut it into stages. Only the boundary activations cross the slower links. Use enough micro-batches to keep the bubble small.", "active": ["pipe"], "example": "A model split in 4 stages over 4 machines, with m = 16 micro-batches." },
    { "title": "The largest models combine them", "body": "Large training runs use tensor parallelism inside a machine, pipeline parallelism across machines, and data parallelism across groups of machines. Narayanan et al. (2021) described this combination for training very large language models.", "active": ["all", "data", "tensor", "pipe"], "example": "Tensor 8 × pipeline 8 × data 16 = 1,024 GPUs (an illustrative example)." }
  ],
  "note": "A rule of thumb, not a law. Real choices also depend on the software and the hardware."
}
```

Notice that this is a flowchart of questions, in an order: first ask whether the model fits, then ask about the links. Every large run you read about is a product of three numbers, one for each split.

```viz
{
  "type": "sorter",
  "title": "Which parallelism fits each situation?",
  "intro": "Twelve situations. Pick the best answer for each; use the flowchart above if you are not sure.",
  "bins": [ { "id": "data", "label": "Data parallel" }, { "id": "tensor", "label": "Tensor parallel" }, { "id": "pipeline", "label": "Pipeline parallel" }, { "id": "combo", "label": "A combination" } ],
  "cards": [
    { "name": "A 300 M-parameter classifier trains too slowly on one GPU", "answer": "data", "why": "300 M parameters × 16 bytes = 4.8 GB fits easily, so copy it and split the batch." },
    { "name": "A single feed-forward matrix is too big for one GPU's memory", "answer": "tensor", "why": "Only splitting the matrix itself reduces the memory of that layer." },
    { "name": "A very deep model on 4 machines joined by a normal network", "answer": "pipeline", "why": "Only boundary activations cross the slow links, which suits pipeline stages." },
    { "name": "Training a 400 B-parameter model on 2,000 GPUs", "answer": "combo", "why": "Such runs combine tensor, pipeline and data parallelism: 3D parallelism." },
    { "name": "Fine-tuning a 1 B model on 8 GPUs of the same server", "answer": "data", "why": "The model and its optimiser state fit on one GPU, so the simplest scheme is enough." },
    { "name": "Serving a 70 B model on 4 GPUs inside one machine with fast links", "answer": "tensor", "why": "Fast links make the frequent all-reduces cheap and each GPU holds a quarter of the weights." },
    { "name": "Serving 300 users with a model that fits one GPU", "answer": "data", "why": "Run independent replicas behind a load balancer: data parallelism's idea applied to inference." },
    { "name": "A model split across two servers joined by a 25 GB/s network", "answer": "pipeline", "why": "Tensor parallelism across a slow link would stall every layer; stages need far less traffic." },
    { "name": "A 175 B model on 64 GPUs in 8 machines of 8 GPUs", "answer": "combo", "why": "Tensor parallel inside each machine, pipeline across machines and perhaps data parallel on top." },
    { "name": "Your batch is large and the model is small", "answer": "data", "why": "Plenty of data per step and no memory problem: more copies bring speed." },
    { "name": "You want each GPU to hold a quarter of every layer's weights", "answer": "tensor", "why": "That describes tensor parallelism: every GPU holds a slice of every layer." },
    { "name": "You want each GPU to hold a block of consecutive layers", "answer": "pipeline", "why": "That describes pipeline parallelism: the stages are blocks of whole layers." }
  ]
}
```

Notice that each tricky card is decided by one fact: does it fit, how fast are the links, how deep is the model. If you remember these three questions you can reconstruct the table.

```viz
{
  "type": "heatmap",
  "title": "What each parallelism buys and costs",
  "rows": ["Data parallel", "Tensor parallel", "Pipeline parallel"],
  "cols": ["Memory saved per GPU", "Speed gained", "Communication needed", "Complexity"],
  "corner": "method ↓ cost or benefit →",
  "showValues": true,
  "W": [[0.0, 0.9, 0.5, 0.2], [0.9, 0.6, 0.9, 0.7], [0.8, 0.5, 0.2, 0.6]],
  "note": "Illustrative scores from 0 (low) to 1 (high) authored for teaching; they are not measurements and depend on the model, the hardware and the software."
}
```

Read it by columns. Only data parallelism saves no memory; only tensor parallelism needs a lot of communication; only pipeline parallelism pays in idle time rather than in communication. There is no free option.

## Precision: fewer bits per number {rail=Precision}

{lede} ==Precision is the number of bits used to store each number.== Fewer bits mean less memory and less to read, and often faster arithmetic, but also a coarser number. Deep learning is forgiving about this, but not infinitely.

### How a floating-point number is stored

Scientific notation writes 6.02 × 10²³: a sign, a few significant digits (the mantissa) and a size (the exponent). A computer does the same in binary. A [[floating-point number|A number stored as a sign, an exponent and a mantissa, like scientific notation in base 2. The exponent sets the range; the mantissa sets the precision.]] has one sign bit, some exponent bits and some mantissa bits. ==The exponent bits decide the range, and the mantissa bits decide the precision.== Think of a ruler: the exponent chooses which ruler (millimetres or kilometres), and the mantissa is how many tick marks that ruler has.

```viz
{
  "type": "numfmt",
  "title": "How FP32, TF32, BF16 and FP16 store a number",
  "intro": "Type a number or click an example. Compare the stored value and the error in each format, and look at the bit layout.",
  "value": 0.1,
  "examples": [0.1, 3.14159, 70000, 1e-8, 1.0001],
  "note": "FP32: 8 exponent and 23 mantissa bits; TF32: 8 and 10; BF16: 8 and 7; FP16: 5 and 10. Values are rounded to nearest, as computed by the lesson's own formula.",
  "checks": [
    { "x": 0.1, "fmt": "FP16", "stored": 0.0999755859375 },
    { "x": 0.1, "fmt": "BF16", "stored": 0.10009765625 },
    { "x": 70000, "fmt": "FP16", "stored": "inf" },
    { "x": 1e-8, "fmt": "FP16", "stored": 0 }
  ]
}
```

Click 70000 and 1e-8. FP16 turns the first into infinity and the second into zero, while BF16 keeps both with a small error. Click 1.0001: three of the formats cannot see the difference from 1.

### The formats you will meet

- **FP32** (4 bytes: 1 sign, 8 exponent, 23 mantissa) is the traditional default of training. It has a wide range and about 7 decimal digits.
- **FP16** (2 bytes: 1, 5, 10) is half the size, but its range is small: the largest value is 65,504, and values below about 6 × 10⁻⁸ become 0.
- **BF16** (2 bytes: 1, 8, 7) is "brain floating point": ==the same range as FP32 but with only 7 mantissa bits==, so it is coarser than FP16 and almost never overflows.
- **TF32** (1, 8, 10) is a format that some GPUs use inside matrix products while the data stay in FP32 in memory.
- **INT8 and INT4** are integers of 8 and 4 bits. They are used mostly for inference, with a scale, and are the subject of the next section.

```viz
{
  "type": "equation",
  "title": "Relative step of a floating-point format",
  "intro": "Select the parts and compare FP16 with BF16.",
  "formula": [ { "t": "ε", "id": "eps" }, " = 2", { "t": "−m", "id": "m" } ],
  "parts": [
    { "id": "eps", "name": "ε (relative step)", "meaning": "The size of the smallest relative change a format can store: the next number after 1 is 1 + ε." },
    { "id": "m", "name": "−m (minus the mantissa bits)", "meaning": "The number of mantissa bits m appears as the exponent: each extra mantissa bit halves the step." }
  ],
  "steps": [
    { "title": "FP16, m = 10", "expr": "ε = 2⁻¹⁰ = 0.000977", "note": "About 3 decimal digits." },
    { "title": "BF16, m = 7", "expr": "ε = 2⁻⁷ = 0.0078", "note": "Eight times coarser than FP16: about 2 to 3 decimal digits." },
    { "title": "FP32, m = 23", "expr": "ε = 2⁻²³ = 1.19 × 10⁻⁷", "note": "About 7 decimal digits." }
  ]
}
```

Notice that the step is relative: near 1 the gap is about 0.001 in FP16, but near 1,000 it is about 1. A floating-point number is as precise in proportion to its size, not in absolute terms.

```viz
{
  "type": "equation",
  "title": "The largest value of a format",
  "intro": "Select the parts and compute the largest FP16 value.",
  "formula": [ { "t": "max", "id": "mx" }, " = (2 − 2", { "t": "−m", "id": "m" }, ") × 2", { "t": "bias", "id": "b" } ],
  "parts": [
    { "id": "mx", "name": "max (largest finite value)", "meaning": "Anything larger overflows to infinity." },
    { "id": "m", "name": "−m (minus the mantissa bits)", "meaning": "The mantissa can reach 2 − 2⁻ᵐ, just under 2." },
    { "id": "b", "name": "bias (largest exponent)", "meaning": "For FP16 it is 15 (5 exponent bits); for BF16 and FP32 it is 127 (8 exponent bits)." }
  ],
  "steps": [
    { "title": "FP16: m = 10, largest exponent 15", "expr": "max = (2 − 2⁻¹⁰) × 2¹⁵ = 1.9990234 × 32,768 = 65,504", "note": "A 5-bit exponent is not very wide." },
    { "title": "BF16 and FP32: largest exponent 127", "expr": "max ≈ 3.4 × 10³⁸", "note": "That is why BF16 almost never overflows." }
  ]
}
```

Notice how little room FP16 has: a value like 70,000, easily met in a sum of activations, does not fit.

### Overflow and underflow

==FP16 fails in two ways.== [[Overflow|A value bigger than the format's largest number, which becomes infinity (inf). One inf in a sum spoils the whole result.]] happens above 65,504: 70,000 turns into inf. [[Underflow|A value smaller than the format's smallest number, which becomes 0. Tiny gradients disappear this way.]] happens below about 6 × 10⁻⁸ (the smallest FP16 subnormal is 5.96 × 10⁻⁸): a gradient of 10⁻⁸ becomes exactly 0, so that weight stops learning. Gradients in deep networks are often very small, and this is the main danger of training in FP16.

```viz
{
  "type": "numfmt",
  "title": "Gradients near 10⁻⁸ and values near 70,000",
  "intro": "The first example is a tiny gradient and the second is a large activation. Look for the formats that lose them.",
  "value": 1e-8,
  "examples": [1e-8, 1.024e-5, 70000, 65504, 6e-5, 1.0001],
  "note": "1.024e-5 is 10⁻⁸ multiplied by a loss scale of 1,024, as used in the loss-scaling example.",
  "checks": [
    { "x": 1e-8, "fmt": "BF16", "stored": 1.0011717677116394e-8 },
    { "x": 1.024e-5, "fmt": "FP16", "stored": 0.000010251998901367188 },
    { "x": 70000, "fmt": "BF16", "stored": 70144 },
    { "x": 65504, "fmt": "FP16", "stored": 65504 }
  ]
}
```

Compare the first two examples: the tiny gradient is lost in FP16, but the same gradient multiplied by 1,024 is stored with only a 0.1% error. That trick has a name.

```viz
{
  "type": "sorter",
  "title": "Overflow, underflow or just rounding?",
  "intro": "What happens to each value when you store it in FP16?",
  "bins": [ { "id": "over", "label": "Overflow (becomes inf)" }, { "id": "under", "label": "Underflow (becomes 0)" }, { "id": "round", "label": "Stored with a small rounding error" } ],
  "cards": [
    { "name": "70,000", "answer": "over", "why": "It is larger than the largest FP16 value of 65,504." },
    { "name": "A gradient of 0.00000001 (10⁻⁸)", "answer": "under", "why": "It is smaller than the smallest FP16 subnormal of about 6 × 10⁻⁸, so it becomes 0." },
    { "name": "0.1", "answer": "round", "why": "It is stored as 0.0999756 with a relative error of about 0.02%, which is harmless." },
    { "name": "Squared activations summing to 100,000", "answer": "over", "why": "The sum exceeds 65,504 and becomes inf, even if each term is small." },
    { "name": "3.14159", "answer": "round", "why": "It becomes 3.140625: a small rounding error in the mantissa." },
    { "name": "A gradient of 0.00001 (10⁻⁵)", "answer": "round", "why": "It is below the smallest normal FP16 value (6.1 × 10⁻⁵) but above the smallest subnormal (6 × 10⁻⁸), so it survives, stored as 0.0000100136 with a small error." },
    { "name": "A loss gradient of 3 × 10⁻⁹", "answer": "under", "why": "It is below half of the smallest subnormal (2.98 × 10⁻⁸), so it rounds to 0." },
    { "name": "65,000", "answer": "round", "why": "It is just below 65,504 and is stored with a rounding step of 32, so it is rounded but still finite." }
  ]
}
```

Notice that overflow is a catastrophe (inf contaminates everything it touches) while underflow is a silent loss (the learning of those weights stops). Loss scaling is the cure for the second; BF16 is the cure for the first.

### Mixed-precision training

Micikevicius et al. (2018) showed that you do not have to choose between memory and quality. In [[mixed-precision training|Training with 16-bit numbers for the forward and backward passes, while a 32-bit master copy of the weights receives the updates and loss scaling protects small gradients.]] the weights keep a ==32-bit master copy==; the forward and backward passes run on a 16-bit copy; the gradients are converted to 32 bits for the update. The small update (a learning rate times a gradient, often 10⁻⁵ or less) would vanish if added to a 16-bit weight, but is safely added to the 32-bit master.

```viz
{
  "type": "pipeline",
  "title": "One step of mixed-precision training",
  "intro": "Step through. Note which numbers are 16-bit and which stay 32-bit.",
  "nodes": [
    { "id": "master", "label": "Master weights", "sub": "FP32" },
    { "id": "half", "label": "16-bit copy", "sub": "FP16 or BF16" },
    { "id": "fwd", "label": "Forward + loss", "sub": "loss × S" },
    { "id": "bwd", "label": "Backward", "sub": "16-bit gradients × S" },
    { "id": "upd", "label": "Update", "sub": "÷ S, FP32 Adam" }
  ],
  "steps": [
    { "title": "Make a 16-bit copy of the master weights", "body": "The master weights in FP32 are rounded to 16 bits. This copy is used for all the heavy arithmetic of the step.", "active": ["master", "half"], "example": "7 B weights: 28 GB master, 14 GB copy." },
    { "title": "Forward pass in 16 bits, loss multiplied by S", "body": "The model runs in 16 bits and the loss is multiplied by a scale factor S (loss scaling), so that every gradient later is S times larger.", "active": ["half", "fwd"], "example": "S = 1,024, loss 2.3 → scaled loss 2,355." },
    { "title": "Backward pass in 16 bits", "body": "The gradients are computed in 16 bits. Because of S, tiny gradients that would underflow are now representable.", "active": ["bwd", "half"], "example": "Gradient 10⁻⁸ becomes 1.02 × 10⁻⁵ in the 16-bit result." },
    { "title": "Convert to 32 bits and divide by S", "body": "The gradients are converted to FP32 and divided by S to undo the scaling. If any value is inf or NaN, the step is skipped and S is lowered.", "active": ["bwd", "upd"], "example": "1.02 × 10⁻⁵ ÷ 1,024 = 1.0 × 10⁻⁸ again." },
    { "title": "Update the master weights in FP32", "body": "The optimiser updates the master weights with the full-precision gradient. Next step, the 16-bit copy is made again.", "active": ["upd", "master"], "example": "w ← w − lr × gradient, in FP32." }
  ],
  "note": "A conceptual diagram of the recipe of Micikevicius et al. (2018). Frameworks automate it (for example torch.autocast and a gradient scaler in PyTorch)."
}
```

Notice that nothing about the model changes: only where the numbers are stored and when they are converted. This is why the technique became standard.

Why multiply by S? By the chain rule, every gradient in the network is proportional to the loss, so multiplying the loss by S multiplies all gradients by S. ==Loss scaling shifts the gradients up into the range where FP16 can store them==, and the division by S in FP32 shifts them back.

```viz
{
  "type": "equation",
  "title": "Loss scaling",
  "intro": "Select the parts, then follow a gradient of 10⁻⁸ through the steps. The stored values come from rounding to FP16.",
  "formula": [ { "t": "g", "id": "g" }, " = ", { "t": "FP16(g_true × S)", "id": "sc" }, " ÷ ", { "t": "S", "id": "s" } ],
  "parts": [
    { "id": "g", "name": "g (recovered gradient)", "meaning": "The gradient that the optimiser uses, after unscaling in FP32." },
    { "id": "sc", "name": "FP16(g_true × S)", "meaning": "The true gradient multiplied by S and stored in FP16. Multiplying first keeps it out of the underflow zone." },
    { "id": "s", "name": "S (loss scale)", "meaning": "A factor such as 1,024. Dynamic loss scaling raises it while no overflow occurs and halves it when an inf appears." }
  ],
  "steps": [
    { "title": "Without scaling", "expr": "FP16(10⁻⁸) = 0", "note": "10⁻⁸ is below the smallest FP16 subnormal, so the gradient is lost." },
    { "title": "Multiply by S = 1,024", "expr": "10⁻⁸ × 1,024 = 1.024 × 10⁻⁵ → FP16 stores 1.0252 × 10⁻⁵", "note": "Now it is a representable number." },
    { "title": "Divide by S in FP32", "expr": "1.0252 × 10⁻⁵ ÷ 1,024 = 1.0012 × 10⁻⁸", "note": "The gradient is recovered with an error of about 0.1%: instead of 100% loss." }
  ]
}
```

Notice that S must not be too big either: a scaled gradient above 65,504 overflows to inf. Dynamic loss scaling searches for the largest safe S automatically, and that is what you implement in the lab.

Because BF16 has the range of FP32 (down to about 10⁻³⁸ for normal numbers), ==BF16 usually needs no loss scaling==. Its price is a coarser mantissa, which deep learning usually tolerates. Many recent training runs prefer BF16 on hardware that supports it.

### What it saves, and what it costs

Mixed precision halves the memory of the activations and of the working copies of the weights, and your total for weights, gradients and optimiser state stays about 16 bytes per parameter (the master copy and Adam's moments are still 32-bit): so ==it does not shrink the model's training state much, but it shrinks the activations and speeds up the arithmetic==. For speed, GPUs have special units for 16-bit matrix products ([[Tensor Cores|Special units on some NVIDIA GPUs that multiply small matrices in 16-bit (and other) formats much faster than ordinary cores.]]). On hardware without fast 16-bit arithmetic, such as many CPUs, the conversions can make a model slower. The lab measures exactly this on your machine, so do not assume a speed-up.

```viz
{
  "type": "flow",
  "title": "FP32 training against mixed-precision training",
  "modes": [
    {
      "name": "FP32 training",
      "tagline": "Everything in 32 bits: simple and safe, but heavy.",
      "inputLabel": "Input: a batch",
      "modelLabel": "FP32 weights, activations and gradients",
      "outputLabel": "Output: an updated model",
      "examples": [
        { "label": "Memory", "input": "7 B parameters", "output": "Weights 28 GB, gradients 28 GB and Adam state 56 GB: 112 GB, also 16 bytes per parameter, plus activations in 32 bits." },
        { "label": "Speed", "input": "Matrix products in FP32", "output": "No use of the fast 16-bit units, so on a GPU with Tensor Cores it is usually slower." }
      ],
      "note": "In pure FP32, 4 + 4 + 8 = 16 bytes per parameter as well, but all the activations are 32-bit."
    },
    {
      "name": "Mixed precision",
      "tagline": "Compute in 16 bits, keep a 32-bit master copy.",
      "inputLabel": "Input: a batch",
      "modelLabel": "16-bit compute, FP32 master weights and Adam",
      "outputLabel": "Output: an updated model of similar quality",
      "examples": [
        { "label": "Memory", "input": "7 B parameters", "output": "About 16 bytes per parameter in total (2 + 2 + 4 + 8) but activations are about half the size." },
        { "label": "Speed", "input": "Matrix products in 16 bits", "output": "Usually faster on hardware with fast 16-bit support; on a CPU without it the lab shows a slowdown." }
      ],
      "note": "Never assume the speed-up: measure it on your hardware."
    }
  ]
}
```

Notice that the two accounts of 16 bytes per parameter are different sums that happen to total the same: 4 + 4 + 8 in FP32, and 2 + 2 + 4 + 8 in mixed precision. The savings of mixed precision are in the activations and in the speed, not in the model state.

```viz
{
  "type": "sorter",
  "title": "Which format would you use?",
  "intro": "Pick the usual choice for each job. Where two answers are reasonable, the card says so.",
  "bins": [ { "id": "fp32", "label": "FP32 (32 bits)" }, { "id": "half", "label": "FP16 or BF16 (16 bits)" }, { "id": "int", "label": "INT8 or INT4 (integers)" } ],
  "cards": [
    { "name": "The master copy of the weights in mixed-precision training", "answer": "fp32", "why": "Tiny updates need a precise number to land in; that is why the master copy is kept in 32 bits." },
    { "name": "The forward and backward passes of training", "answer": "half", "why": "This is the heavy arithmetic that 16-bit units accelerate." },
    { "name": "Adam's two moments", "answer": "fp32", "why": "They are running averages of tiny numbers and are usually kept in 32 bits." },
    { "name": "Weights of a model for serving on a small GPU", "answer": "int", "why": "INT8 or INT4 shrink the memory by 2 to 4 times compared with FP16 and often speed up decoding." },
    { "name": "Gradients in FP16 training", "answer": "half", "why": "They are computed in 16 bits (with loss scaling), then converted for the update." },
    { "name": "Activations during training", "answer": "half", "why": "Storing them in 16 bits halves the memory they need." },
    { "name": "The sum that accumulates inside a matrix product", "answer": "fp32", "why": "Hardware usually accumulates the products in 32 bits so that many small terms do not lose precision." },
    { "name": "A 70 B model that must fit on one GPU for inference", "answer": "int", "why": "At 4 bits the weights take about 35 GB, which fits where 140 GB of FP16 did not." }
  ]
}
```

Notice the sum-accumulation card: even "16-bit" matrix products are usually accumulated in 32 bits. The format is chosen per job, not per model.

## Quantization {rail=Quantization}

{lede} [[Quantization|Storing numbers as small integers (such as 8 or 4 bits) plus a scale, instead of as 16- or 32-bit floating-point numbers. It mostly shrinks a trained model for inference.]] ==rounds a trained model's weights to a few integer levels and remembers one scale to turn them back==. A 4-bit weight takes a quarter of the memory of a 16-bit weight, and there is much less to read from memory for every token.

### The idea: a ruler with fewer marks

Imagine measuring everyone in the class with a ruler. A ruler with a mark every millimetre is precise but needs a big drawing to print. A ruler with a mark every centimetre is a sixth of the size and every height is off by at most half a centimetre. Quantization does this to weights. A 4-bit integer has only 16 values (from −8 to 7, or −7 to 7 in the symmetric scheme below), so you must choose how much real distance one integer step represents. That distance is the [[scale|The real-number size of one integer step in quantization. A weight is stored as the integer q and restored as q times the scale.]].

In the symmetric scheme, the largest weight in a group, by absolute value, is mapped to the largest integer. With b bits the integers go from −(2ᵇ⁻¹ − 1) to 2ᵇ⁻¹ − 1, so scale = max|w| ÷ (2ᵇ⁻¹ − 1). To quantize a weight w, divide by the scale and round to the nearest integer; to restore it, multiply back.

```viz
{
  "type": "equation",
  "title": "Symmetric quantization",
  "intro": "Select the parts, then quantize 8 weights to 4 bits by hand.",
  "formula": [ { "t": "scale", "id": "sc" }, " = max|w| ÷ (2", { "t": "b−1", "id": "b" }, " − 1);   ", { "t": "q", "id": "q" }, " = round(w ÷ scale);   ", { "t": "ŵ", "id": "wh" }, " = q × scale" ],
  "parts": [
    { "id": "sc", "name": "scale", "meaning": "The size of one integer step, found from the largest absolute weight and the number of bits." },
    { "id": "b", "name": "b − 1 (bits minus one)", "meaning": "One bit is the sign, so b bits give integers up to 2ᵇ⁻¹ − 1. With 4 bits the largest integer is 7; with 8 bits it is 127." },
    { "id": "q", "name": "q (the stored integer)", "meaning": "The weight divided by the scale and rounded to the nearest whole number. This is what takes only b bits." },
    { "id": "wh", "name": "ŵ (the restored weight)", "meaning": "q times the scale. It is close to the original w but not equal." }
  ],
  "steps": [
    { "title": "The weights and the scale", "expr": "w = (0.42, −0.18, 0.07, −0.51, 0.30, 0.11, −0.26, 0.45);  max|w| = 0.51;  scale = 0.51 ÷ 7 = 0.0729", "note": "4 bits: the largest integer is 7." },
    { "title": "The integers", "expr": "q = round(w ÷ 0.0729) = (6, −2, 1, −7, 4, 2, −4, 6)", "note": "For example 0.42 ÷ 0.0729 = 5.76 → 6, and 0.11 ÷ 0.0729 = 1.51 → 2." },
    { "title": "Restore", "expr": "ŵ = q × 0.0729 = (0.437, −0.146, 0.073, −0.510, 0.291, 0.146, −0.291, 0.437)", "note": "0.42 became 0.437 (error 0.017) and 0.11 became 0.146 (error 0.036)." }
  ]
}
```

Notice that the weights are now 8 numbers of 4 bits plus a single scale. The restored values are close but not equal: that gap is the price.

How big can the gap get? Rounding to the nearest integer moves a number by at most half a step, so ==the error of each weight is at most scale ÷ 2==. With a smaller scale (more bits, or a smaller maximum) the error shrinks.

```viz
{
  "type": "equation",
  "title": "The rounding error of quantization",
  "intro": "Select the parts and compare 4 bits with 8 bits for the same weights.",
  "formula": [ "|", { "t": "w − ŵ", "id": "err" }, "| ≤ ", { "t": "scale", "id": "sc" }, " ÷ 2" ],
  "parts": [
    { "id": "err", "name": "w − ŵ (the error)", "meaning": "The difference between the original weight and the restored weight." },
    { "id": "sc", "name": "scale", "meaning": "The size of one integer step. Rounding to the nearest integer cannot be off by more than half a step." }
  ],
  "steps": [
    { "title": "4 bits, max|w| = 0.51", "expr": "scale = 0.0729 → error ≤ 0.0364", "note": "The largest error in the example, 0.036, is just under this limit." },
    { "title": "8 bits, the same weights", "expr": "scale = 0.51 ÷ 127 = 0.0040 → error ≤ 0.0020", "note": "Eighteen times smaller scale, eighteen times smaller worst-case error." }
  ]
}
```

Notice that each extra bit halves the scale and so halves the error. The next figure lets you try this on 16 weights.

```viz
{
  "type": "quant",
  "title": "Rounding smooth weights to fewer bits",
  "intro": "Start with the smooth weights and slide the number of bits from 8 down to 2. Watch the mean squared error and the restored values.",
  "bits": 8,
  "group": 0,
  "sets": [
    { "name": "Smooth weights", "w": [0.42, -0.18, 0.07, -0.51, 0.30, 0.11, -0.26, 0.45, 0.22, -0.35, 0.15, -0.08, 0.38, -0.47, 0.02, 0.27] },
    { "name": "One outlier", "w": [0.12, -0.08, 0.05, -0.15, 0.10, 0.04, -0.09, 0.14, 0.07, -0.11, 0.06, -0.03, 0.13, -0.14, 0.01, 2.40] }
  ],
  "note": "Made-up weights for teaching. Symmetric quantization with scale = max|w| ÷ (2^(bits−1) − 1).",
  "checks": [
    { "set": 0, "bits": 8, "group": 0, "mse": 1.1020522e-6 },
    { "set": 0, "bits": 4, "group": 0, "mse": 4.0255102e-4 },
    { "set": 0, "bits": 3, "group": 0, "mse": 0.00278125 }
  ]
}
```

At 8 bits the error is almost invisible; at 4 bits it is visible but modest; at 2 bits the weights collapse to very few levels. The error of each bit setting rises fast, because each bit you remove doubles the step.

### The outlier problem

One extra-large weight, an [[outlier|A weight or activation much larger in size than the others. Because the scale is set by the largest value, one outlier stretches the scale and wastes the integer levels on the small values.]], spoils ==the scale for every other weight==. The scale is set by the largest number, so one outlier stretches the scale and leaves the small weights with only a few integer levels, or none. Dettmers et al. (2022, LLM.int8()) reported that large language models contain a few such outlier features, which makes naive 8-bit quantization hurt, and they proposed handling those outliers separately.

```viz
{
  "type": "quant",
  "title": "One outlier, and the fix of per-group scales",
  "intro": "The second set has one weight of 2.40. Use 4 bits and one scale for everything, then give every 4 weights their own scale (group 4).",
  "bits": 4,
  "group": 0,
  "sets": [
    { "name": "One outlier", "w": [0.12, -0.08, 0.05, -0.15, 0.10, 0.04, -0.09, 0.14, 0.07, -0.11, 0.06, -0.03, 0.13, -0.14, 0.01, 2.40] },
    { "name": "Smooth weights", "w": [0.42, -0.18, 0.07, -0.51, 0.30, 0.11, -0.26, 0.45, 0.22, -0.35, 0.15, -0.08, 0.38, -0.47, 0.02, 0.27] }
  ],
  "note": "Made-up weights. With one scale and 4 bits, the outlier forces a scale of 2.40 ÷ 7 = 0.343 and every other weight rounds to 0.",
  "checks": [
    { "set": 0, "bits": 4, "group": 0, "mse": 0.00895 },
    { "set": 0, "bits": 4, "group": 4, "mse": 0.0023073980 },
    { "set": 0, "bits": 8, "group": 0, "mse": 3.0166160e-5 }
  ]
}
```

With one scale for all 16 weights at 4 bits, every weight except the outlier becomes 0: the model would be destroyed. With a scale per group of 4, only the group that holds the outlier suffers, and the error falls to about a quarter. The fix works because the damage of an outlier is limited to its own group.

### The scales are not free

Per-group scales help, but each scale takes memory too. If every group of g weights has one 16-bit scale, each weight carries an extra 16 ÷ g bits. ==Bits per weight = bits + 16 ÷ g==. Smaller groups are more accurate and use more memory: a trade-off.

```viz
{
  "type": "equation",
  "title": "Bits per weight, scales included",
  "intro": "Select the parts and compute the memory of a 7 B model in 4-bit weights with a group size of 64.",
  "formula": [ { "t": "bits/weight", "id": "bw" }, " = ", { "t": "b", "id": "b" }, " + ", { "t": "16", "id": "s" }, " ÷ ", { "t": "g", "id": "g" } ],
  "parts": [
    { "id": "bw", "name": "bits/weight (effective)", "meaning": "The true storage cost per weight when the scales are included." },
    { "id": "b", "name": "b (bits of each weight)", "meaning": "The integer width: for example 4." },
    { "id": "s", "name": "16 (bits per scale)", "meaning": "A scale stored as a 16-bit number." },
    { "id": "g", "name": "g (group size)", "meaning": "How many weights share one scale. Larger groups use fewer scales." }
  ],
  "steps": [
    { "title": "Effective bits for g = 64", "expr": "4 + 16 ÷ 64 = 4 + 0.25 = 4.25 bits", "note": "The scales add 6% to the bits." },
    { "title": "A 7 B model", "expr": "7 × 10⁹ × 4.25 ÷ 8 = 3.72 × 10⁹ bytes = 3.72 GB", "note": "Slightly above the 3.5 GB of the naive count." },
    { "title": "A 70 B model", "expr": "70 × 10⁹ × 4.25 ÷ 8 = 37.2 GB", "note": "Fits on one example 80 GB GPU with room for a cache, where the 140 GB of FP16 did not." }
  ]
}
```

Notice that the "4-bit" model is really a 4.25-bit model once the scales are counted. Quote the real number when you plan memory.

### How quantization is done in practice

- [[Post-training quantization|Quantizing a model after it is trained, using only the finished weights and perhaps a small calibration set. It is fast and needs no retraining.]] (PTQ) takes a trained model and rounds it. It is quick and the usual choice for LLMs. Frantar et al. (2022, GPTQ) became known for a one-shot method that quantizes the weights of large language models down to 3 or 4 bits with little loss.
- [[Quantization-aware training|Training (or fine-tuning) with the rounding simulated in the forward pass, so the model learns to be robust to it.]] (QAT) simulates the rounding during training, so the model adapts to it. It usually gives better quality at low bits but costs a training run. Jacob et al. (2018) described a scheme for inference using only integer arithmetic.
- **Weight-only** quantization stores the weights as integers but computes with 16-bit activations. It saves memory and bandwidth. **Weight-and-activation** quantization also quantizes the activations so that the matrix products themselves use integers; it can be faster but is harder, because activations have outliers.
- **Dynamic quantization** stores the weights as integers and quantizes the activations on the fly. PyTorch offers this under the name `quantize_dynamic`; the lab uses it.
- **QLoRA** (Dettmers et al., 2023, Week 14) keeps the frozen base model in 4 bits and trains small LoRA adapters on top, so that you can fine-tune a large model on one GPU.

The adapters of QLoRA rest on the Week 14 idea that a fine-tuning update is often close to a low-rank matrix, so a few numbers describe it. The figure below is the Week 14 picture, as a reminder of why a small adapter on top of a frozen, quantized base is enough.

```viz
{
  "type": "lowrank",
  "title": "Why a small adapter can be enough",
  "intro": "A pretend fine-tuning update ΔW. Slide the rank from 1 up and watch how little is lost with rank 1.",
  "matrix": [[0.9, 1.8, -0.9, 0.45, 1.35, 0.0], [0.62, 1.28, -0.58, 0.35, 0.9, 0.05], [-0.3, -0.55, 0.3, -0.15, -0.45, 0.02], [0.4, 0.85, -0.4, 0.2, 0.62, -0.03]],
  "rank": 1,
  "max": 3,
  "note": "A made-up update matrix, authored so that one direction dominates; real updates are only approximately low-rank.",
  "checks": [ { "sv": [3.514, 0.089] }, { "rank": 1, "err": 0.029 } ]
}
```

With rank 1 the update is rebuilt with only about 3% relative error, from 10 numbers instead of 24. In QLoRA the big frozen matrices are the ones that are quantized, and the small trained adapters stay in 16 bits.

```viz
{
  "type": "pipeline",
  "title": "Quantizing one layer",
  "intro": "Step through the quantization of one weight matrix and its use.",
  "nodes": [
    { "id": "w", "label": "Weights", "sub": "FP16 or FP32" },
    { "id": "scale", "label": "Find scales", "sub": "max|w| per group" },
    { "id": "round", "label": "Round", "sub": "q = round(w ÷ scale)" },
    { "id": "store", "label": "Store", "sub": "integers + scales" },
    { "id": "run", "label": "Run", "sub": "q × scale in the matmul" }
  ],
  "steps": [
    { "title": "Start with the trained weights", "body": "A trained matrix in 16 or 32 bits. Quantization does not retrain it; it changes only how the numbers are stored.", "active": ["w"], "example": "A 4096 × 4096 matrix: 16.8 million weights, 33.6 MB in FP16." },
    { "title": "Find one scale per group", "body": "For each group of g weights, find the largest absolute value and divide by the largest integer. Small groups limit the damage of outliers.", "active": ["w", "scale"], "example": "g = 64, 4 bits: scale = max|w| ÷ 7 for each group of 64." },
    { "title": "Round to integers", "body": "Divide every weight by its group's scale and round to the nearest integer in the allowed range. The error is at most half a scale.", "active": ["round"], "example": "0.42 ÷ 0.0729 = 5.76 → 6." },
    { "title": "Store integers and scales", "body": "Only the b-bit integers and the 16-bit scales are kept. This is where the memory is saved: 4.25 bits per weight instead of 16.", "active": ["store"], "example": "33.6 MB → 8.9 MB for the matrix (4.25 ÷ 16 of the size)." },
    { "title": "Use it at run time", "body": "When the layer runs, the kernel reads the small integers, multiplies by the scale and does the matrix product. Because there is less to read from memory, decoding is often faster.", "active": ["run", "store"], "example": "A layer of 4-bit weights reads about a quarter of the bytes." }
  ],
  "note": "16.8 million × 2 bytes = 33.6 MB; 16.8 million × 4.25 ÷ 8 = 8.9 MB."
}
```

Notice that quantization changes only the storage of a matrix that was already trained. It is why a few minutes of rounding can shrink a model that took weeks to train.

```viz
{
  "type": "calc",
  "title": "The memory of a model at 16, 8 and 4 bits",
  "kind": "memory",
  "intro": "Set the model to 7 B and read the FP16, INT8 and INT4 bars; then set it to 70 B.",
  "note": "Weights only, not counting the scales: 7 B is 14, 7 and 3.5 GB; 70 B is 140, 70 and 35 GB. With scales (group size 64) the INT4 bar grows by about 6%."
}
```

At 70 B, INT4 is the first setting that fits on one example 80 GB GPU. That is why quantization is the first thing people reach for when a model does not fit.

```viz
{
  "type": "flow",
  "title": "An FP16 model against an INT8 model",
  "modes": [
    {
      "name": "FP16 model",
      "tagline": "2 bytes per weight: accurate, but twice as large as INT8.",
      "inputLabel": "Input: a prompt",
      "modelLabel": "16-bit weights",
      "outputLabel": "Output: the reference answer",
      "examples": [
        { "label": "7 B memory", "input": "7 × 10⁹ weights × 2 bytes", "output": "14 GB for the weights." },
        { "label": "Decoding", "input": "Each token reads all the weights", "output": "14 GB to read per token: at an assumed 200 GB/s, at most 14.3 tokens per second." }
      ],
      "note": "Bandwidth is an assumed example value."
    },
    {
      "name": "INT8 model",
      "tagline": "1 byte per weight plus a few scales: half the memory, usually almost the same answers.",
      "inputLabel": "Input: the same prompt",
      "modelLabel": "8-bit integer weights with scales",
      "outputLabel": "Output: an answer very close to the FP16 one",
      "examples": [
        { "label": "7 B memory", "input": "7 × 10⁹ weights × 1 byte", "output": "7 GB for the weights, plus a small amount for the scales." },
        { "label": "Decoding", "input": "Each token reads all the weights", "output": "7 GB to read per token: at an assumed 200 GB/s, at most 28.6 tokens per second, twice the FP16 bound." }
      ],
      "note": "Usually the quality drop at 8 bits is small, but you must measure it on your own task."
    }
  ]
}
```

Notice that the two benefits arrive together: the model is smaller, and each token needs fewer bytes to be read. That second effect comes back in the Serving section.

```viz
{
  "type": "curve",
  "title": "Rounding error halves with every extra bit",
  "intro": "x is the number of bits added to a starting width. The curve is the error left, as a fraction, if each bit halves it (slope 0.5). Move the slider to see a different factor.",
  "xRange": [0, 8],
  "yRange": [-3, 0],
  "logY": true,
  "series": [ { "fn": "decay", "label": "Error left after x extra bits" } ],
  "param": { "label": "Factor per extra bit (0.5 = halving)", "min": 0.3, "max": 0.7, "step": 0.01, "value": 0.5 },
  "alt": "A straight line on a log scale falling from 1 at zero extra bits to about 0.004 at eight extra bits, because each extra bit halves the step of the rounding.",
  "note": "Idealised: the step (scale) halves with each bit, so the worst-case rounding error halves too. Real model quality also depends on outliers, calibration and the task, so measure it. Usually the drop is small at 8 bits and grows below 4 bits."
}
```

On the log scale, the line is straight: that is what halving at every step looks like. In a real model the accuracy usually stays almost flat from 16 down to 8 bits, then falls faster below 4; where it falls depends on the model and the method, so you measure it.

```viz
{
  "type": "timeline",
  "title": "Quantization and serving milestones",
  "events": [
    { "year": "2018", "title": "Integer-arithmetic-only inference", "text": "Jacob et al. described a quantization scheme that lets a network run inference using only integer arithmetic, aimed at efficient devices." },
    { "year": "2022", "title": "LLM.int8()", "text": "Dettmers et al. reported that large language models have rare outlier features that break plain 8-bit quantization and proposed a way to treat them separately." },
    { "year": "2022", "title": "GPTQ", "text": "Frantar et al. introduced a one-shot post-training method that quantizes the weights of large language models to about 3 or 4 bits with little loss in quality." },
    { "year": "2023", "title": "QLoRA", "text": "Dettmers et al. fine-tuned a 4-bit quantized frozen base model with LoRA adapters, so that large models can be fine-tuned on one GPU." },
    { "year": "2023", "title": "PagedAttention and vLLM", "text": "Kwon et al. introduced PagedAttention, which stores the KV cache in blocks like memory pages, and the vLLM serving system built on it." }
  ]
}
```

Click through: the story is the same as in training. Every time a model did not fit, someone found a way to store it in fewer bits.

```viz
{
  "type": "sorter",
  "title": "What does each technique save?",
  "intro": "Drop each technique on what it mainly improves. Be careful: some techniques save neither.",
  "bins": [ { "id": "memory", "label": "Mainly memory" }, { "id": "speed", "label": "Mainly speed" }, { "id": "both", "label": "Both memory and speed" }, { "id": "neither", "label": "Neither (it keeps things stable)" } ],
  "cards": [
    { "name": "4-bit weight-only quantization of a decoding LLM", "answer": "both", "why": "Fewer bytes are stored and fewer bytes are read per token, so decoding is also faster." },
    { "name": "Data parallelism", "answer": "speed", "why": "It splits the batch to shorten the time per epoch; every GPU still holds the whole model." },
    { "name": "Tensor parallelism", "answer": "memory", "why": "Each GPU stores only a slice of every matrix. It is mostly a way to fit a big model, at a communication cost." },
    { "name": "Loss scaling", "answer": "neither", "why": "It protects small gradients from underflow; it neither saves memory nor speeds up anything." },
    { "name": "Batching requests together in serving", "answer": "speed", "why": "The cost of reading the weights is shared, which raises the number of requests per second." },
    { "name": "Storing weights as INT8 but computing in FP16 (weight-only)", "answer": "memory", "why": "The matrices are smaller in storage; the arithmetic itself is not accelerated." },
    { "name": "Lowering the learning rate when training diverges", "answer": "neither", "why": "It is a stability remedy and has nothing to do with how much memory or time a step costs." },
    { "name": "A faster link between the GPUs of a data-parallel job", "answer": "speed", "why": "It shortens the all-reduce and so raises the efficiency; the memory is unchanged." }
  ]
}
```

Notice the first card: a technique is more valuable when it relieves both limits at once. That is why quantization is so widely used for serving.

## Serving a model {rail=Serving}

{lede} Training happens once; ==serving is what students and teachers experience every day==. A deployed model is judged by how fast it answers one person ([[latency|The time one request waits from sending to the full answer. It is measured in milliseconds or seconds, and users feel it directly.]]) and by how many people it can serve at once ([[throughput|How much work the system completes per second, for example requests or tokens per second. It sets how many users one machine can serve.]]). These two goals pull in different directions.

### Two phases of generating text

When a model answers, it works in two phases. In the [[prefill|The first phase of generation: the model reads the whole prompt in one parallel pass and fills the KV cache. It is compute-bound.]] phase it reads the whole prompt at once, in parallel, which keeps the GPU busy and is limited by compute. In the [[decode|The second phase: the model generates one token at a time. Each token needs a pass over all the weights, so it is usually limited by memory bandwidth.]] phase it generates one token at a time, and each token needs a pass over all the weights, so decoding is usually ==memory-bound: limited by how fast the bytes can be read, not by arithmetic==.

```viz
{
  "type": "sorter",
  "title": "Prefill or decode?",
  "intro": "Which phase does each statement describe?",
  "bins": [ { "id": "prefill", "label": "Prefill (reading the prompt)" }, { "id": "decode", "label": "Decode (writing the answer)" } ],
  "cards": [
    { "name": "Processes all the prompt tokens in one parallel pass", "answer": "prefill", "why": "All the prompt tokens are known, so their matrix products can be done together." },
    { "name": "Produces one new token per pass over the weights", "answer": "decode", "why": "Each token depends on the one before it, so tokens cannot be produced in parallel." },
    { "name": "Fills the KV cache for the first time", "answer": "prefill", "why": "The keys and values of all prompt tokens are computed and stored here." },
    { "name": "Is mostly limited by memory bandwidth", "answer": "decode", "why": "For each token the whole set of weights is read for very little arithmetic." },
    { "name": "Is mostly limited by compute (matrix arithmetic)", "answer": "prefill", "why": "A long prompt gives many tokens per weight read, so arithmetic dominates." },
    { "name": "Is what the user feels as the time to the first word", "answer": "prefill", "why": "The first word appears only after the whole prompt has been processed." },
    { "name": "Is what the user feels as the speed of the words appearing", "answer": "decode", "why": "The rate of the stream of words is the decode speed in tokens per second." },
    { "name": "Appends one row of keys and values to the cache per token", "answer": "decode", "why": "Each generated token adds its own key and value to the cache." }
  ]
}
```

Notice that the two phases feel different to a user: the wait before the first word and the speed of the words that follow.

### The speed limit of decoding

For each generated token, the GPU must read every weight once. If the model takes G gigabytes and the [[memory bandwidth|How many gigabytes per second the memory can deliver to the processor. It limits how fast the weights can be read.]] is B gigabytes per second, then you cannot do better than B ÷ G tokens per second, however fast the arithmetic is. ==Tokens per second ≤ memory bandwidth ÷ model bytes.==

```viz
{
  "type": "equation",
  "title": "The decoding speed limit",
  "intro": "Select the parts and compute the bound for a 7 B model at 16 bits and at 4 bits. The bandwidth is an assumed example value.",
  "formula": [ { "t": "tokens/s", "id": "t" }, " ≤ ", { "t": "B", "id": "b" }, " ÷ ", { "t": "G", "id": "g" } ],
  "parts": [
    { "id": "t", "name": "tokens/s (upper bound)", "meaning": "The most tokens per second one sequence can possibly be generated at, when memory is the limit." },
    { "id": "b", "name": "B (memory bandwidth)", "meaning": "How many gigabytes per second the memory can deliver. We assume 200 GB/s as an example." },
    { "id": "g", "name": "G (model size in GB)", "meaning": "The bytes of all the weights: parameters times bytes per parameter." }
  ],
  "steps": [
    { "title": "7 B model in FP16", "expr": "G = 7 × 2 = 14 GB;  tokens/s ≤ 200 ÷ 14 = 14.3", "note": "About 14 tokens a second at most." },
    { "title": "The same model in INT8", "expr": "G = 7 GB;  tokens/s ≤ 200 ÷ 7 = 28.6", "note": "Twice as fast: half the bytes." },
    { "title": "The same model in INT4", "expr": "G = 3.5 GB;  tokens/s ≤ 200 ÷ 3.5 = 57.1", "note": "Four times the FP16 bound. The scales make it a little less." }
  ]
}
```

Notice that the arithmetic speed of the GPU does not appear anywhere in the formula. That is what memory-bound means.

```viz
{
  "type": "calc",
  "title": "Decoding speed against model size and bits",
  "kind": "decode_speed",
  "intro": "Choose the model size and 16, 8 or 4 bits, then change the memory bandwidth. Find the settings that give 30 tokens per second.",
  "note": "The bandwidth values are assumptions; the bound ignores the KV cache and overheads, so real speeds are lower. It applies to one sequence at a time (batch 1)."
}
```

Move the bandwidth slider and the bound moves in proportion; change the bits and the bound moves in inverse proportion to the bytes. That is why quantization speeds up decoding.

### Batching: sharing the weight reads

If the GPU reads the whole model to produce one token for one user, ==it wastes the read==. Reading the weights once to produce one token for each of 8 users is nearly as cheap as for one user. That is [[batching|Processing several requests together in one pass of the model, so that the cost of reading the weights is shared.]]. A batch of b requests takes a fixed time t₀ (reading the weights and other fixed costs) plus c for each request: t = t₀ + c·b. The price is waiting: the first request in a batch waits for the batch to fill.

```viz
{
  "type": "equation",
  "title": "Batching: time, capacity and latency",
  "intro": "Select the parts. The timings are made up (t₀ = 40 ms, c = 5 ms, 50 requests per second arriving).",
  "formula": [ { "t": "t", "id": "t" }, " = ", { "t": "t₀", "id": "t0" }, " + ", { "t": "c·b", "id": "cb" }, ";   capacity = 1000·", { "t": "b", "id": "b" }, " ÷ t;   wait = (b − 1) ÷ (2", { "t": "λ", "id": "l" }, ")" ],
  "parts": [
    { "id": "t", "name": "t (batch time in ms)", "meaning": "How long the model needs to process one batch." },
    { "id": "t0", "name": "t₀ (fixed cost in ms)", "meaning": "The time that does not depend on the batch size, mostly reading the weights." },
    { "id": "cb", "name": "c·b (per-request cost)", "meaning": "c milliseconds for each of the b requests in the batch." },
    { "id": "b", "name": "b (batch size)", "meaning": "The number of requests processed together." },
    { "id": "l", "name": "λ (arrival rate)", "meaning": "How many requests arrive per second. The average wait to fill a batch is (b − 1) ÷ (2λ) seconds." }
  ],
  "steps": [
    { "title": "Batch time", "expr": "t = 40 + 5 × 8 = 80 ms", "note": "b = 8 requests." },
    { "title": "Capacity", "expr": "1000 × 8 ÷ 80 = 100 requests per second", "note": "Twice the 50 per second that arrive: stable." },
    { "title": "Waiting to fill the batch", "expr": "wait = (8 − 1) ÷ (2 × 50) = 0.07 s = 70 ms", "note": "On average, a request waits 70 ms for the others." },
    { "title": "Average latency", "expr": "latency = 70 + 80 = 150 ms", "note": "With b = 1 the time is 45 ms, but capacity is only 22 per second, so the queue would grow without limit." }
  ]
}
```

Notice that batching trades a little delay (70 ms of waiting) for much more capacity (100 against 22 per second). With b = 1 the server cannot keep up with 50 requests a second at all.

```viz
{
  "type": "batching",
  "title": "Latency against throughput as the batch grows",
  "intro": "Move the batch size b. Find the batch size with the lowest average latency, and watch the capacity.",
  "t0": 40,
  "c": 5,
  "lambda": 50,
  "b": 8,
  "note": "Made-up timings (t₀ = 40 ms, c = 5 ms) for teaching; measure your own in the lab.",
  "checks": [ { "b": 8, "lambda": 50, "latency": 150, "capacity": 100 }, { "b": 4, "lambda": 50, "latency": 90, "capacity": 66.67 } ]
}
```

With a gentle arrival rate the best batch is neither the smallest (which cannot keep up) nor the largest (which makes everyone wait to fill it).

```viz
{
  "type": "batching",
  "title": "A busier server with cheaper requests",
  "intro": "Here requests arrive at 100 per second and each one is cheap. Is the best batch size different?",
  "t0": 30,
  "c": 2,
  "lambda": 100,
  "b": 16,
  "note": "Made-up timings (t₀ = 30 ms, c = 2 ms). Real servers also queue requests and use continuous batching.",
  "checks": [ { "b": 16, "lambda": 100, "latency": 137, "capacity": 258.06 }, { "b": 4, "lambda": 100, "latency": 53, "capacity": 105.26 } ]
}
```

Because requests arrive faster, filling a batch is quick, so larger batches are affordable and the best size moves up. The right batch depends on the arrival rate, which is why servers adjust it on the fly.

```viz
{
  "type": "flow",
  "title": "No batching against batching",
  "modes": [
    {
      "name": "No batching",
      "tagline": "Each request is served alone, in the order it arrives.",
      "inputLabel": "Input: requests arriving one by one",
      "modelLabel": "one request per pass over the weights",
      "outputLabel": "Output: one answer at a time",
      "examples": [
        { "label": "Capacity", "input": "t₀ = 40 ms, c = 5 ms, b = 1", "output": "45 ms per request: 22 requests per second at most." },
        { "label": "At 50 per second", "input": "More arrive than can be served", "output": "The queue grows without limit: the latency explodes." }
      ],
      "note": "Made-up timings."
    },
    {
      "name": "Batching (b = 8)",
      "tagline": "Eight requests share one pass over the weights.",
      "inputLabel": "Input: requests collected into batches of 8",
      "modelLabel": "one pass over the weights for 8 requests",
      "outputLabel": "Output: eight answers per batch",
      "examples": [
        { "label": "Capacity", "input": "80 ms per batch of 8", "output": "100 requests per second, enough for the 50 that arrive." },
        { "label": "Latency", "input": "70 ms waiting plus 80 ms of service", "output": "150 ms on average: slower per request, but the system keeps up." }
      ],
      "note": "Made-up timings; the average wait is (b − 1) ÷ (2λ)."
    }
  ]
}
```

Notice that the batched server is the one that works, even though each answer takes longer than a single request would. [[Continuous batching|A serving technique in which new requests join a batch that is already running, instead of waiting for a fresh batch.]], used by modern serving software, lets new requests join a batch that is already running instead of waiting for a fresh one; Kwon et al. (2023) built vLLM, with PagedAttention, on this idea.

### The KV cache

When the model generates token n, attention needs the keys and values of all the earlier tokens. Without a cache you would recompute them for every token, and the cost of the whole answer would grow like n² (Week 8). The [[KV cache|Stored keys and values of earlier tokens, kept in GPU memory so each new token only adds one new row and attends to the stored ones.]] keeps them, so each new token computes only its own key and value and attends to the stored ones, at a cost proportional to n. ==The price is memory.==

```viz
{
  "type": "causalmask",
  "title": "Why the cache is valid: the past never changes",
  "intro": "Switch the causal mask on. Each row only looks at itself and earlier tokens, so the keys and values of earlier tokens never change when a new token arrives.",
  "tokens": ["the", "cat", "sat", "down"],
  "scores": [[0.5, 0.2, 0.1, 0.3], [1.0, 1.0, 0.4, 0.2], [0.3, 1.2, 0.6, 0.1], [0.2, 0.9, 0.4, 0.8]],
  "row": 3,
  "note": "Illustrative attention scores. The point is the shape: with the causal mask the earlier rows are the same whatever comes next, so their keys and values can be stored.",
  "checks": [ { "causal": true, "row": 1, "col": 0, "value": 0.5 } ]
}
```

Because the mask hides the future, the earlier tokens' keys and values do not depend on later tokens. That is the whole reason you can store them once and reuse them.

```viz
{
  "type": "equation",
  "title": "Size of the KV cache",
  "intro": "Select the parts and compute the cache for 8 conversations of 4,096 tokens on a 7 B-class model.",
  "formula": [ { "t": "KV", "id": "kv" }, " = ", { "t": "2", "id": "two" }, " × ", { "t": "L", "id": "l" }, " × ", { "t": "n", "id": "n" }, " × ", { "t": "d", "id": "d" }, " × ", { "t": "bytes", "id": "by" }, " × ", { "t": "batch", "id": "bt" } ],
  "parts": [
    { "id": "kv", "name": "KV (cache bytes)", "meaning": "The memory taken by all the stored keys and values." },
    { "id": "two", "name": "2 (keys and values)", "meaning": "Each token stores one key vector and one value vector." },
    { "id": "l", "name": "L (layers)", "meaning": "Every layer has its own keys and values. We use 32 layers." },
    { "id": "n", "name": "n (tokens per sequence)", "meaning": "The number of tokens stored so far: prompt plus generated." },
    { "id": "d", "name": "d (model width)", "meaning": "The size of each key and value vector summed over the heads: 4,096 here." },
    { "id": "by", "name": "bytes per value", "meaning": "2 for a 16-bit cache, 1 for an 8-bit cache." },
    { "id": "bt", "name": "batch (sequences)", "meaning": "How many conversations are kept in the cache at the same time." }
  ],
  "steps": [
    { "title": "Per token, per sequence", "expr": "2 × 32 × 4,096 × 2 bytes = 524,288 bytes ≈ 0.5 MB", "note": "Every new token adds half a megabyte to its sequence's cache." },
    { "title": "One sequence of 4,096 tokens", "expr": "0.524 MB × 4,096 = 2.15 GB", "note": "A single long conversation takes more than two gigabytes." },
    { "title": "A batch of 8", "expr": "2 × 32 × 4,096 × 4,096 × 2 × 8 = 17.18 × 10⁹ bytes = 17.2 GB", "note": "More than the 14 GB of the FP16 weights: 1.23 times as big." }
  ]
}
```

Notice that the cache grows with both the length of the conversations and the number of users, and can become bigger than the model. This is why long contexts and many users are expensive.

```viz
{
  "type": "calc",
  "title": "The KV cache of a 7 B-class model",
  "kind": "kv_cache",
  "layers": 32,
  "d": 4096,
  "paramsB": 7,
  "intro": "Set 4,096 tokens and a batch of 8 with a 16-bit cache, then change to an 8-bit cache. Raise the tokens and watch the cache overtake the weights.",
  "note": "Layers and width are those of a typical 7 B-class model, used as an example. The weights are counted in 16 bits."
}
```

Switching the cache to 8 bits halves it. Just like the weights, the cache can be quantized, at some risk to quality.

```viz
{
  "type": "tfblock",
  "title": "Where the 7 billion parameters live",
  "intro": "A decoder with width 4,096, a feed-forward width of 16,384 and 32 layers. Look at how the total divides between attention, feed-forward and embeddings.",
  "vocab": 32000,
  "d": 4096,
  "heads": 32,
  "dff": 16384,
  "layers": 32,
  "ctx": 4096,
  "posLearned": false,
  "tied": false,
  "note": "An example configuration close to a 7 B model, with the simple two-matrix feed-forward network of Week 8; real models vary. Parameters: 6.7 B.",
  "checks": [ { "total": 6706307072, "block": 201379840, "attn": 67125248, "ffn": 134238208 } ]
}
```

Notice that the layers hold almost all the parameters (6.44 of 6.71 billion). The same d and layer count that give the weights also give the KV cache: both come from L and d.

### Planning a deployment

You can now plan capacity on paper before you buy anything. The recipe has four steps and each uses a formula from this lesson.

```viz
{
  "type": "pipeline",
  "title": "Capacity planning recipe",
  "intro": "Step through the four checks. Use assumed numbers for your own hardware.",
  "nodes": [
    { "id": "mem", "label": "Memory fits?", "sub": "weights + KV + activations" },
    { "id": "speed", "label": "Tokens per second", "sub": "bandwidth ÷ bytes" },
    { "id": "batch", "label": "Batch size", "sub": "from the latency target" },
    { "id": "cost", "label": "Cost", "sub": "per million tokens" },
    { "id": "ok", "label": "Decision", "sub": "build, split or shrink" }
  ],
  "steps": [
    { "title": "Does it fit?", "body": "Add the weights (parameters × bytes), the KV cache for the longest conversations and the batch, and the activations, and compare with the GPU memory. If not, quantize, shorten the context or split the model.", "active": ["mem"], "example": "7 B in FP16 = 14 GB; KV for 8 × 4,096 = 17.2 GB; total 31.2 GB, which fits in an assumed 80 GB." },
    { "title": "How fast can one sequence decode?", "body": "Divide the memory bandwidth by the model bytes to get the bound on tokens per second for one sequence.", "active": ["speed"], "example": "200 GB/s ÷ 14 GB = 14.3 tokens per second at most (assumed bandwidth)." },
    { "title": "Choose the batch from the latency target", "body": "A larger batch raises throughput and latency. Choose the largest batch whose latency, including the wait, still meets the target.", "active": ["batch"], "example": "Target 200 ms average: b = 8 gives 150 ms in the made-up example." },
    { "title": "Compute the cost per million tokens", "body": "Divide the price of the machine per hour by the tokens it produces per hour, and scale to a million.", "active": ["cost"], "example": "2.00 per hour ÷ (400 tokens/s × 3,600) × 10⁶ = about 1.39 per million tokens (assumed)." },
    { "title": "Decide", "body": "If every check passes, deploy; if memory fails, quantize or split; if speed or cost fails, quantize more, batch better or add replicas.", "active": ["ok"], "example": "A 7 B model quantized to INT4 may be enough for the whole school helpdesk." }
  ],
  "note": "Every number is an assumed example for teaching; the formulas are the point."
}
```

Notice the order: memory is a yes-or-no gate, speed and cost are dials. Run the gate first.

```viz
{
  "type": "pipeline",
  "title": "One request on its way through a server",
  "intro": "Follow one student's question from arrival to the last word.",
  "nodes": [
    { "id": "queue", "label": "Queue", "sub": "wait for a batch" },
    { "id": "prefill", "label": "Prefill", "sub": "read the prompt" },
    { "id": "decode", "label": "Decode", "sub": "token by token" },
    { "id": "cache", "label": "KV cache", "sub": "grows each token" },
    { "id": "reply", "label": "Reply", "sub": "streamed to the user" }
  ],
  "steps": [
    { "title": "The request waits in a queue", "body": "The request arrives and joins others. The server collects a batch or, with continuous batching, slots it into a batch that is already running.", "active": ["queue"], "example": "Wait of about (b − 1) ÷ (2λ) on average with fixed batches." },
    { "title": "Prefill reads the prompt", "body": "The whole prompt is processed in one parallel pass, and the keys and values of every prompt token are stored in the cache.", "active": ["prefill", "cache"], "example": "A 500-token prompt is processed in one pass." },
    { "title": "Decode writes one token at a time", "body": "For each new token the model reads all the weights, attends to the cache and appends the new key and value.", "active": ["decode", "cache"], "example": "Speed bounded by bandwidth ÷ model bytes." },
    { "title": "The cache grows", "body": "After each token the cache is a little larger. Long answers and long prompts use more memory.", "active": ["cache"], "example": "0.5 MB per token for the 7 B-class example." },
    { "title": "The reply is streamed", "body": "Tokens are sent as they are produced, so the user sees words appear while the model is still writing.", "active": ["reply", "decode"], "example": "The first word appears after prefill; later words follow at the decode rate." }
  ],
  "note": "A simplified view of a serving system."
}
```

Notice that the first word's delay is the queue plus the prefill, and every later word is paced by decoding.

### Tail latency and service levels

An average hides the unlucky users. If 100 students ask a question, the 95th [[percentile|The value below which a given share of measurements fall. The p95 latency is the time within which 95 out of 100 requests finish.]] latency (p95) is the time that 95 of them beat, and p99 is for 99. Averages of 150 ms can hide a p99 of several seconds when the system is busy or a batch is unlucky. A [[service-level objective|A target for the quality of a service, such as "p95 latency under 2 seconds", against which the system is monitored and planned.]] (SLO) states a target such as "p95 under 2 seconds", and ==you plan capacity against the tail, not the average==. Queues grow quickly as a server approaches full capacity, which is why you leave headroom.

```viz
{
  "type": "equation",
  "title": "Cost per million tokens",
  "intro": "Select the parts. The price and the speeds are assumed examples, not real quotes.",
  "formula": [ { "t": "cost", "id": "c" }, " = ", { "t": "P", "id": "p" }, " ÷ (", { "t": "r", "id": "r" }, " × 3,600) × 10⁶" ],
  "parts": [
    { "id": "c", "name": "cost (per million tokens)", "meaning": "What it costs to generate one million tokens, in the same currency as P." },
    { "id": "p", "name": "P (price per hour)", "meaning": "What the machine costs to run for an hour. We assume 2.00 as an example." },
    { "id": "r", "name": "r (tokens per second)", "meaning": "The total tokens the machine generates per second, summed over all users." }
  ],
  "steps": [
    { "title": "One user, no batching", "expr": "r = 14.3: cost = 2.00 ÷ (14.3 × 3,600) × 10⁶ = 38.9 per million", "note": "A machine serving one sequence is costly per token." },
    { "title": "A batch with 400 tokens per second in total", "expr": "cost = 2.00 ÷ (400 × 3,600) × 10⁶ = 1.39 per million", "note": "About 28 times cheaper: batching and quantization help the cost as well as the speed." }
  ]
}
```

Notice that the price of the machine hardly changes: what changes the cost per token is how many tokens it produces. Serving well is mostly about keeping the hardware busy.

### Deployment practice

Putting a model into service takes more than speed.

- **Version** the model, its settings and the prompt template, so you know which one produced an answer.
- **Monitor** latency (p50, p95, p99), errors, memory use and cost, and the quality of the answers (a small test set that you run regularly).
- Keep a **rollback**: a way to return to the previous version in minutes.
- **Autoscale** replicas when demand changes, since replicas are the cheapest form of data parallelism for serving.
- Protect **privacy** (what is logged, who sees it) and keep the **guardrails** of Week 17 in front of the model.
- Software examples: vLLM, TensorRT-LLM and NVIDIA Triton Inference Server are examples of systems that do batching, cache management and scheduling for you. In this course you use one of these ideas, not necessarily one of these programs.

### Your capstone

Your Structural Inspection Assistant will be deployed on the school's server (see the [Capstone](/capstone) page). Do not guess what that machine can do; read its hardware report in the lab and use the formulas of this lesson. A short list of what to measure: the memory used by the weights and the cache; the tokens per second for one user and for several; the p95 latency under a realistic load; the effect of quantization on both speed and accuracy; and the cost or energy per request. A table with these numbers, with the method that produced them, is worth more than any claim.

## Key takeaways

- ==One GPU has two limits, memory and time==: weights cost parameters × bytes, training costs about 16 bytes per parameter, and C ≈ 6ND decides how long it takes.
- ==Data parallelism copies the model and averages gradients== (cures time, costs an all-reduce of 2(N − 1)/N × S); ==tensor parallelism splits each matrix== (cures memory, needs fast links); ==pipeline parallelism splits the layers== (little traffic, but an idle bubble of (p − 1)/(m + p − 1)).
- ==Mixed precision== computes in 16 bits with a 32-bit master copy and loss scaling (FP16 overflows above 65,504 and underflows near 6 × 10⁻⁸; BF16 has the range of FP32); it speeds up GPUs that support it, but you must measure.
- ==Quantization== rounds weights to a few integer levels with a scale (error at most scale ÷ 2): 4-bit weights are a quarter of FP16 plus the scales, outliers need per-group scales, and fewer bytes also speed up decoding.
- ==Serving is latency against throughput==: decoding is bounded by bandwidth ÷ model bytes, batching shares the weight reads, and the KV cache (2 × L × n × d × bytes × batch) can be larger than the model.

<!-- LAB-START -->
## Lab: Imagine the warehouse and the delivery trucks {rail=Laboratory}

{lede} This lab is called *Imagine the warehouse and the delivery trucks*. A company ships goods from one warehouse with one truck. The warehouse is **too small** for all the goods: that is the **memory** limit of one GPU, and it decides whether a model fits at all. The truck is **too slow** to deliver before the deadline: that is the **time** limit, and it decides how long training takes. You can build several warehouses and run several trucks (**parallelism**), pack the goods into smaller boxes (**fewer bits per number**), and organise the deliveries so that no truck waits (**batching**). *How much memory and how much time does a model really need, and what does each trick buy, on the computer in front of you?* You build a small Transformer and count its memory exactly, average gradients by hand and in real separate processes, split matrices and layers across simulated devices, write number formats and quantization from scratch, simulate a server and a KV cache, and finish by planning the deployment of three models. Everything is measured, with a fixed random seed.

**The numbers depend on the machine, and the lab says so.** Memory sizes (bytes of weights, gradients, optimiser state and cache) are counted exactly from tensors and are the same everywhere. Timings are not: the lab prints a **hardware report** first, and every part says what it measured *on this machine*. We ran the lab on a laptop CPU with no GPU, and there the results are sometimes the opposite of what you read about GPUs: 16-bit arithmetic was *slower* than 32-bit, and several "devices" that were really processes on the same CPU gave no speed-up. That is not a mistake in the lab; it is the lesson that a trick is only as good as the hardware under it. If you have a GPU (for instance on Google Colab), the notebook **also measures on the GPU** wherever that is interesting. We could not test those GPU cells ourselves, so if one of them fails, tell your instructor and carry on: nothing else depends on it. Prefer the **ratios** (how many times slower, how many times smaller) to the absolute times, and write checks that test properties, not exact milliseconds.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start the notebook, read the hardware report, run a smoke test | 10 |
| 1 | Imagine the warehouse: memory and time | 25 |
| 2 | Data parallelism by hand and for real | 30 |
| 3 | Tensor and pipeline parallelism | 30 |
| 4 | Mixed precision | 30 |
| 5 | Quantization | 30 |
| 6 | Serving: latency, throughput and the KV cache | 30 |
| 7 | Plan the deployment and reflect | 20 |

The total is about 3 hours. Almost none of it is waiting: the longest cells run for under a minute (the multi-process cells of Part 2 start a few small Python processes on your own computer). While a cell runs, write your predictions and answers for the next step. The troubleshooting table and the assessed deliverable are at the end of the lab.

By the end of the lab you can:

- count the parameters of a Transformer and measure, from real tensors, the memory of its weights, gradients, optimiser state and activations, and say how each grows with model size, batch and sequence length;
- show by calculation and by a real multi-process run that data parallelism with averaged gradients equals big-batch training, and measure what the all-reduce costs;
- split a layer across devices (column split and row split), split a model into pipeline stages, and compute the bubble;
- explain and measure what mixed precision and quantization do to memory, speed and accuracy, including when they do **not** help;
- reason about serving (latency, throughput, batching, the KV cache) and plan a deployment from a memory and a bandwidth budget.

### Setup

1. Log in to the portal, start JupyterLab, create a folder `week18` and a notebook `lab18.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names that earlier blocks create (`device`, `time_ms`, `n_params`, `param_bytes`, `RESULTS`, `check` and so on), and each part re-creates what it needs from the Setup names. If you restart the kernel, run the setup cells again and then the blocks of each part from its top.
2. **No dataset and no model are downloaded.** You need Python 3 with NumPy, Matplotlib and **PyTorch** (the CPU version is enough; on Google Colab all of them are installed). If `import torch` fails on your own computer, the command is `pip install torch numpy matplotlib`; on a shared computer, tell your instructor and do not install packages yourself. Part 2 starts a few small Python processes on your own computer that talk to each other through `localhost`; nothing leaves your computer.
3. A CPU is enough for everything. The seed is `SEED = 42`. Counts, sizes and formulas must match the numbers in this lab exactly; **timings will differ** from machine to machine and from run to run by a few percent (more if other programs are running), and the lab tells you which numbers are timings.
4. Write your answers in markdown cells under the block where the exercise appears. Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. In Colab you can choose **Runtime → Change runtime type** and pick a GPU to see the GPU measurements; do the lab at least once without one.
6. Paste the next block into a cell and run it. It loads the libraries, fixes the seed, **finds out what hardware you have**, and defines the helpers that all the parts share: `time_ms(fn, repeat, warmup)` (the average time of a function in milliseconds; on a GPU it waits for the GPU to finish before it stops the clock), `n_params(model)` and `param_bytes(model)` (the number of parameters, and the bytes they occupy, counted from the tensors), `state_mb(model)` (the size of a saved checkpoint), `human_bytes(x)`, and the flags `HAS_CUDA` and `device`. It also times a matrix product in 32-bit and in bfloat16 and prints their ratio `CPU_BF16_RATIO`, which Part 4 uses.

```python
import os, re, io, gc, math, time, json, random, platform, subprocess, sys, warnings
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
HAS_CUDA = torch.cuda.is_available()
device = "cuda" if HAS_CUDA else "cpu"
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone

def time_ms(fn, repeat=5, warmup=1):
    """Average time of fn() in milliseconds over `repeat` runs after `warmup` runs. On a GPU it waits for the GPU to finish before it stops the clock."""
    for _ in range(warmup): fn()
    if HAS_CUDA: torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeat): fn()
    if HAS_CUDA: torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeat * 1000

def n_params(model):
    """Number of parameters of a model."""
    return sum(p.numel() for p in model.parameters())

def param_bytes(model):
    """Bytes that the parameters of a model occupy (number of values times bytes per value, counted exactly)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())

def state_mb(model):
    """Size in megabytes of the model's state dict when saved (what a checkpoint file would weigh)."""
    b = io.BytesIO(); torch.save(model.state_dict(), b); return len(b.getvalue()) / 1e6

def human_bytes(x):
    """Write a number of bytes as B, KB, MB or GB (powers of 1000)."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1000 or u == "TB": return f"{x:.0f} {u}" if u == "B" else f"{x:.2f} {u}"
        x /= 1000

def _cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except Exception: pass
    return platform.processor() or platform.machine()

# ---- What hardware do we have, and is half precision fast on it? ----
_a32 = torch.randn(512, 512); _b32 = torch.randn(512, 512)
_t32 = time_ms(lambda: _a32 @ _b32, repeat=3)
_a16 = _a32.bfloat16(); _b16 = _b32.bfloat16()
_t16 = time_ms(lambda: _a16 @ _b16, repeat=2)
CPU_BF16_RATIO = _t16 / _t32      # above 1: bfloat16 matrix products are slower than 32-bit on this CPU
print("Python", platform.python_version(), "| PyTorch", torch.__version__, "| threads:", torch.get_num_threads())
print("CPU:", _cpu_name())
if HAS_CUDA:
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB of memory, compute capability {props.major}.{props.minor}")
else:
    print("GPU: none found (CUDA is not available); the GPU measurements are skipped and the text says what to expect.")
print(f"A 512x512 matrix product on this CPU: float32 {_t32:.2f} ms, bfloat16 {_t16:.2f} ms ({CPU_BF16_RATIO:.1f}x {'slower' if CPU_BF16_RATIO > 1 else 'faster'}).")
print("Quantization engines available:", torch.backends.quantized.supported_engines)
```

   The block prints a hardware report like the one below. **Your numbers will be different, and that is expected.** Copy these lines into your notes: your CPU name, the number of threads, whether a GPU was found (with its name and memory), and the bfloat16 ratio. Part 7 asks you for them.

```text
Python 3.12.3 | PyTorch 2.15.0+cpu | threads: 6
CPU: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
GPU: none found (CUDA is not available); the GPU measurements are skipped and the text says what to expect.
A 512x512 matrix product on this CPU: float32 1.16 ms, bfloat16 244.98 ms (211.8x slower).
Quantization engines available: ['qnnpack', 'onednn', 'x86', 'fbgemm']
```

   Read the second-last line. On the machine that ran this lab, a 16-bit matrix product is many times **slower** than the 32-bit one, because this CPU has no fast 16-bit arithmetic. If your ratio is below 1, your hardware does have fast 16-bit arithmetic (as GPUs with Tensor Cores do), and Part 4 will give you the opposite result from ours: that is a finding, not an error.

7. Paste the next block into a second cell. `check(...)` prints a tick or a cross with a hint, and `RESULTS` is the dictionary where you keep your key numbers.

```python
if "RESULTS" not in globals():
    RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
```

8. **Smoke test.** Run this block to check the things the lab needs: the timer, the byte counters, the process tools for Part 2 (a child Python process and the `gloo` communication backend) and a free network port on your own computer.

```python
import socket
import torch.distributed as dist
smoke_t = time_ms(lambda: torch.randn(256, 256) @ torch.randn(256, 256), repeat=3)
smoke_model = nn.Linear(100, 10)
smoke_child = subprocess.run([sys.executable, "-c", "print(6*7)"], capture_output=True, text=True, timeout=60)
with socket.socket() as _s:
    _s.bind(("127.0.0.1", 0)); smoke_port = _s.getsockname()[1]
print("device:", device, "| threads:", torch.get_num_threads(), "| CPU_BF16_RATIO: %.1f" % CPU_BF16_RATIO)
print("a 256 x 256 matrix product takes %.2f ms here" % smoke_t)
print("Linear(100, 10): %d parameters = %d bytes (%s)" % (n_params(smoke_model), param_bytes(smoke_model), human_bytes(param_bytes(smoke_model))))
print("child Python process answered:", smoke_child.stdout.strip(), "| torch.distributed available:", dist.is_available(), "| gloo:", dist.is_available() and dist.is_gloo_available())
print("a free local port was found:", smoke_port > 0)
check("setup works", n_params(smoke_model) == 1010 and param_bytes(smoke_model) == 4040 and smoke_child.stdout.strip() == "42" and dist.is_available() and dist.is_gloo_available() and smoke_port > 0,
      "Run the setup cell first. If torch.distributed is missing, Part 2 shows what to do.")
```

   The block prints (the timing and the thread count depend on your computer; the other lines must match):

```text
device: cpu | threads: 6 | CPU_BF16_RATIO: 211.8
a 256 x 256 matrix product takes 0.93 ms here
Linear(100, 10): 1010 parameters = 4040 bytes (4.04 KB)
child Python process answered: 42 | torch.distributed available: True | gloo: True
a free local port was found: True
✓ setup works
```

If something goes wrong:

- `NameError: name 'time_ms' is not defined` (or `n_params`, `device`, `check` and so on): the setup block has not run in this session. Run it again from the top, then the helper block.
- `ModuleNotFoundError: No module named 'torch'`: PyTorch is not installed. On Colab it always is; on your own computer use the command of Setup step 2. On a shared computer, ask your instructor.
- The smoke test prints a cross for "setup works": read which of the lines is different. A missing `gloo` means that your PyTorch was built without distributed support: Part 2 will then print what it could not do, and you can read the output of this lab for that part. A `Linear(100, 10)` that does not have 1,010 parameters means that you changed the cell.
- A cell is very slow (minutes instead of seconds): another program is using the CPU, or you are on a very small shared computer. Close other programs; the heavy cells use `QUICK` only for the authors' tests, so do not change it.
- Part 2 prints "the multi-process run failed": some platforms (certain Windows or notebook setups, or a firewall) do not let the processes connect on `localhost`. Run the cell again, then read the note under it. The later parts do not need it.
- Your timings differ from the ones in this lab by a factor of two or three: that is normal, because every computer is different. Compare **ratios** and **signs** (which is faster, which is smaller), and check that sizes and counts match exactly.
- A GPU cell fails: it was written for the usual case (one CUDA GPU) and could not be tested by the authors. Skip it, tell your instructor, and carry on.

### Part 1. Imagine the warehouse: memory and time (25 min)

*Lesson sections: "Why one GPU is not enough" and "Serving a model" (the memory budget).* Picture the company again. The **warehouse** holds everything that must be at hand at the same moment: the goods themselves (the **weights**), the notes on how to change each good (the **gradients**), two ledgers per good that the optimiser keeps (**Adam's** running averages), and the half-finished work lying on the benches (the **activations**). The **truck** is the speed of the computer: how many arithmetic operations it can do each second. A model fails to train for one of two reasons: ==the warehouse is too small, or the truck is too slow==. In this part you build a small Transformer, count what its warehouse holds **exactly** (bytes taken from the real tensors, not from a formula), measure how the benches fill up when the batch or the text gets longer, scale the numbers up to 7-billion and 70-billion-parameter models, and finish with the truck: how long would training take?

**Imagine first.** (1) A Transformer decoder with d = 256, 4 layers and a vocabulary of 4,000 tokens: how many parameters does it have, to the nearest million? (2) Training it with AdamW in 32-bit numbers: how many **bytes per parameter** does the optimiser step need in total (weights, gradients and the two Adam ledgers)? (3) You double the batch size. By what factor does the memory of the **activations** grow? (4) How many 80 GB GPUs are needed just to *hold* the training state of a 70-billion-parameter model? Write your guesses in the block below (replace each `None`), then run it. You compare with Steps 1.1, 1.2, 1.3 and 1.4.

```python
# Imagine first: the warehouse
my_guess_params_m = None   # parameters of the small model, in millions (a number)
my_guess_bytes_per_param = None   # bytes per parameter during an AdamW step in 32-bit (a whole number)
my_guess_act_factor = None   # growth factor of the activation memory when the batch doubles (a number)
my_guess_gpus_70b = None   # number of 80 GB GPUs to hold the training state of a 70 B model (a whole number)
```

1. **Step 1.1. Build a small Transformer and count what it has.** The block below writes a decoder-only Transformer by hand, in the same shape as the models of Weeks 7 and 8: token and position embeddings, `L` blocks (layer norm, causal self-attention, layer norm, feed-forward network with a 4 d hidden layer, each with a residual connection), a final layer norm and an output layer that gives one score per word of the vocabulary. The attention is split into small modules (`P1Scores`, `P1Mix`) so that the next steps can see every intermediate tensor. We use `d = 256`, 4 layers, 4 heads, a vocabulary of 4,000 and a context of up to 256 tokens, and set `P1_CTX`, `P1_V` and so on as the shared settings of this part.

   The formula of Week 8 says that a block has about 12 d² parameters (4 d² in attention, 8 d² in the feed-forward network), so the model should have about 12 L d² for the blocks plus the embeddings. We count the real parameters and compare. The exact count also contains the small biases and layer-norm parameters: 13 d per block.

```python
P1_D, P1_L, P1_H, P1_V, P1_CTX = 256, 4, 4, 4000, 256

class P1Scores(nn.Module):
    """Attention probabilities: softmax(q k^T / sqrt(dh)) with a causal mask. Output shape (batch, heads, T, T)."""
    def forward(self, q, k):
        T = q.shape[-2]
        att = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        return att.masked_fill(mask, float("-inf")).softmax(-1)

class P1Mix(nn.Module):
    """Mix the value vectors with the attention probabilities."""
    def forward(self, p, v):
        return p @ v

class P1Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.scores, self.mix = P1Scores(), P1Mix()
    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        o = self.mix(self.scores(q, k), v)
        return self.proj(o.transpose(1, 2).reshape(B, T, d))

class P1Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(d), P1Attn(d, h), nn.LayerNorm(d)
        self.fc1, self.act, self.fc2 = nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))

class P1LM(nn.Module):
    def __init__(self, V=P1_V, d=P1_D, L=P1_L, h=P1_H, ctx=P1_CTX):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([P1Block(d, h) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)
    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

torch.manual_seed(SEED)
p1_model = P1LM()
p1_n = n_params(p1_model)
p1_approx_blocks = 12 * P1_L * P1_D ** 2
p1_embed = 2 * P1_V * P1_D + P1_CTX * P1_D
print(f"parameters counted from the tensors: {p1_n:,}  ({p1_n / 1e6:.2f} million)")
print(f"12 L d^2 (blocks only):             {p1_approx_blocks:,}")
print(f"embeddings: tokens + output layer + positions = {p1_embed:,}")
print(f"12 L d^2 + embeddings:              {p1_approx_blocks + p1_embed:,}  ({100 * (p1_approx_blocks + p1_embed) / p1_n:.1f}% of the real count)")
print(f"bytes of the parameters in float32: {param_bytes(p1_model):,} = {human_bytes(param_bytes(p1_model))}")
assert abs(p1_approx_blocks + p1_embed - p1_n) / p1_n < 0.02
```

   The block prints (your numbers must match; timings depend on your computer):

```text
parameters counted from the tensors: 5,273,088  (5.27 million)
12 L d^2 (blocks only):             3,145,728
embeddings: tokens + output layer + positions = 2,113,536
12 L d^2 + embeddings:              5,259,264  (99.7% of the real count)
bytes of the parameters in float32: 21,092,352 = 21.09 MB
```

   The model has **5.27 million parameters**. The formula 12 L d² gives 3.15 million for the four blocks, and the three embedding tables add 2.11 million: in this *small* model the embeddings and the output layer are **40 %** of everything, which is why the rule "12 L d²" only works for big models (in a 7-billion-parameter model the blocks dominate). The formula plus the embeddings is within 0.3 % of the real count; the remainder is the biases and the layer norms (13 d per block). At 4 bytes per float32 value the weights occupy **21.09 MB**.

   *Exercise 1.1:* Count the parameters with a formula. Your task: write `p1_count(V, d, L, ctx)` that returns the **exact** number of parameters of `P1LM` (with `h` heads: the number of heads does not change the count) from the sizes alone, without building the model. Use what Week 8 taught and the exact pieces of one block: the attention has `qkv` (3 d² + 3 d) and `proj` (d² + d), the feed-forward network has `fc1` (4 d² + 4 d) and `fc2` (4 d² + d), and each of the two layer norms has 2 d parameters. Then add the token table, the position table, the final layer norm and the output layer (no bias). Why it matters: you will use this function again to size models that are too big to build.

   Replace the starter, run it, then run the **Check** cell.

```python
def p1_count(V, d, L, ctx):
    # your code here
    return None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if p1_count(P1_V, P1_D, P1_L, P1_CTX) is None:
    print("Write p1_count above, then run this cell again.")
else:
    check("exact count equals the real model", p1_count(P1_V, P1_D, P1_L, P1_CTX) == p1_n, f"you got {p1_count(P1_V, P1_D, P1_L, P1_CTX):,}, the model has {p1_n:,}; check the biases and the layer norms")
    _other = P1LM(V=500, d=64, L=2, h=2, ctx=32)
    check("also right for another size", p1_count(500, 64, 2, 32) == n_params(_other), "the formula must work for any V, d, L, ctx")
    RESULTS["p1_params"] = p1_n
```

   A possible answer, and what it means:

```python
def p1_count(V, d, L, ctx):
    attn = (3 * d * d + 3 * d) + (d * d + d)        # qkv and proj
    ffn = (4 * d * d + 4 * d) + (4 * d * d + d)     # fc1 and fc2
    block = attn + ffn + 2 * (2 * d)                # two layer norms (weight and bias each)
    return V * d + ctx * d + L * block + 2 * d + d * V
```

   A block has 12 d² + 13 d parameters: the 12 d² is the part that Week 8 uses, and the 13 d is the small remainder. Because the count is exact, a model that is too big to build can be sized with the same function.

   Below the exercise we run our version of the formula, so that the cells after it work without your answer.

```python
def p1_count(V, d, L, ctx):
    attn = (3 * d * d + 3 * d) + (d * d + d)
    ffn = (4 * d * d + 4 * d) + (4 * d * d + d)
    block = attn + ffn + 2 * (2 * d)
    return V * d + ctx * d + L * block + 2 * d + d * V
assert p1_count(P1_V, P1_D, P1_L, P1_CTX) == p1_n
print("exact formula:", f"{p1_count(P1_V, P1_D, P1_L, P1_CTX):,}", "= real count", f"{p1_n:,}")
```

   The block prints (your numbers must match; timings depend on your computer):

```text
exact formula: 5,273,088 = real count 5,273,088
```

2. **Step 1.2. Measure the memory of one training step.** Training keeps four things per parameter. The **weights** `p`, the **gradients** `p.grad` (one per weight), and the two ledgers of **AdamW**, `exp_avg` (a running average of the gradient) and `exp_avg_sq` (a running average of its square). We run one real AdamW step on a batch and then add up the bytes of **every tensor** that the optimiser holds: `numel() * element_size()`. Nothing is estimated.

   The lesson says "16 bytes per parameter". There are two accountings that both give 16. In **plain 32-bit training** it is 4 (weights) + 4 (gradients) + 8 (two Adam ledgers) = 16. In **mixed-precision training** (Part 4) the working copies are 16-bit and a 32-bit **master copy** is kept: 2 (16-bit weights) + 2 (16-bit gradients) + 4 (master weights) + 8 (Adam) = 16. Same total, different pieces. The activations come on top in both.

```python
torch.manual_seed(SEED)
p1_m = P1LM()
p1_opt = torch.optim.AdamW(p1_m.parameters(), lr=1e-3)
p1_ids = torch.randint(0, P1_V, (8, 128))
p1_loss = F.cross_entropy(p1_m(p1_ids[:, :-1]).reshape(-1, P1_V), p1_ids[:, 1:].reshape(-1))
p1_loss.backward()
p1_opt.step()

p1_w = sum(p.numel() * p.element_size() for p in p1_m.parameters())
p1_g = sum(p.grad.numel() * p.grad.element_size() for p in p1_m.parameters())
p1_a = sum(s["exp_avg"].numel() * s["exp_avg"].element_size() for s in p1_opt.state.values())
p1_b = sum(s["exp_avg_sq"].numel() * s["exp_avg_sq"].element_size() for s in p1_opt.state.values())
p1_total = p1_w + p1_g + p1_a + p1_b
print(f"weights:               {p1_w:>12,} bytes = {p1_w / p1_n:.1f} per parameter")
print(f"gradients:             {p1_g:>12,} bytes = {p1_g / p1_n:.1f} per parameter")
print(f"Adam exp_avg:          {p1_a:>12,} bytes = {p1_a / p1_n:.1f} per parameter")
print(f"Adam exp_avg_sq:       {p1_b:>12,} bytes = {p1_b / p1_n:.1f} per parameter")
print(f"total training state:  {p1_total:>12,} bytes = {p1_total / p1_n:.1f} per parameter = {human_bytes(p1_total)}")
print("mixed-precision accounting: 2 + 2 + 4 + 8 =", 2 + 2 + 4 + 8, "bytes per parameter")
assert p1_total == 16 * p1_n
```

   The block prints (your numbers must match; timings depend on your computer):

```text
weights:                 21,092,352 bytes = 4.0 per parameter
gradients:               21,092,352 bytes = 4.0 per parameter
Adam exp_avg:            21,092,352 bytes = 4.0 per parameter
Adam exp_avg_sq:         21,092,352 bytes = 4.0 per parameter
total training state:    84,369,408 bytes = 16.0 per parameter = 84.37 MB
mixed-precision accounting: 2 + 2 + 4 + 8 = 16 bytes per parameter
```

   The optimiser step holds **84,369,408 bytes**, exactly **16 bytes per parameter** (84.37 MB for this model): 4 for the weights, 4 for the gradients and 4 + 4 for the two Adam ledgers. The state of training is therefore **four times** the size of the weights alone. Note what is *not* in this number: the activations, which depend on the batch and on the length of the text (Step 1.3), and temporary buffers.

   *Exercise 1.2:* Measure the optimiser state yourself. Your task: write `p1_state_bytes(model, opt)` that returns a dictionary with the four keys `"weights"`, `"grads"`, `"exp_avg"` and `"exp_avg_sq"`, each the total number of bytes (`numel() * element_size()`) of the matching tensors **after** one optimiser step. The optimiser's tensors are in `opt.state`, a dictionary from each parameter to a dictionary of tensors. Why it matters: when a training job runs out of memory, adding up the real tensors tells you which of the four is the problem.

   Replace the starter, run it, then run the **Check** cell.

```python
def p1_state_bytes(model, opt):
    # your code here
    return {}
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
_d = p1_state_bytes(p1_m, p1_opt)
if not _d:
    print("Write p1_state_bytes above, then run this cell again.")
else:
    check("four keys", set(_d) == {"weights", "grads", "exp_avg", "exp_avg_sq"}, "use exactly the four key names")
    check("weights and gradients are right", _d.get("weights") == p1_w and _d.get("grads") == p1_g, "sum numel() * element_size() over the parameters, and over p.grad")
    check("both Adam ledgers are right", _d.get("exp_avg") == p1_a and _d.get("exp_avg_sq") == p1_b, "they are the values of opt.state[p]")
    check("the total is 16 bytes per parameter", sum(_d.values()) == 16 * p1_n, "4 + 4 + 4 + 4 bytes in float32")
    RESULTS["p1_bytes_per_param"] = sum(_d.values()) / p1_n if _d else None
```

   A possible answer, and what it means:

```python
def p1_state_bytes(model, opt):
    nb = lambda t: t.numel() * t.element_size()
    params = list(model.parameters())
    return {"weights": sum(nb(p) for p in params),
            "grads": sum(nb(p.grad) for p in params),
            "exp_avg": sum(nb(opt.state[p]["exp_avg"]) for p in params),
            "exp_avg_sq": sum(nb(opt.state[p]["exp_avg_sq"]) for p in params)}
```

   Each of the four has the same number of values as the model has parameters, and each value is a 4-byte float32: that is why the total is exactly 16 bytes per parameter. (If a gradient is `None`, the optimiser has not run yet.)

   Below the exercise we run our version, so that the cells after it work without your answer.

```python
def p1_state_bytes(model, opt):
    nb = lambda t: t.numel() * t.element_size()
    params = list(model.parameters())
    return {"weights": sum(nb(p) for p in params), "grads": sum(nb(p.grad) for p in params),
            "exp_avg": sum(nb(opt.state[p]["exp_avg"]) for p in params), "exp_avg_sq": sum(nb(opt.state[p]["exp_avg_sq"]) for p in params)}
RESULTS["p1_bytes_per_param"] = sum(p1_state_bytes(p1_m, p1_opt).values()) / p1_n
print(p1_state_bytes(p1_m, p1_opt))
RESULTS.setdefault("p1_params", p1_n)
```

   The block prints (your numbers must match; timings depend on your computer):

```text
{'weights': 21092352, 'grads': 21092352, 'exp_avg': 21092352, 'exp_avg_sq': 21092352}
```

3. **Step 1.3. Measure the activations: the benches fill up with the batch.** The **activations** are the intermediate results of the forward pass that the backward pass needs again. We measure them with a **forward hook** on every small module: a hook is a function that PyTorch calls with the module's output, and ours adds up `numel() * element_size()`. Because we hooked only the *leaf* modules (those with no children) and our attention is split into leaves, nothing is counted twice. As a cross-check we also ask autograd itself: `torch.autograd.graph.saved_tensors_hooks` reports every tensor that is **saved for the backward pass**, and we add up the distinct memory blocks (without the weights).

   We then run a grid of batch sizes and sequence lengths and look at how the activation memory grows. Think about the formula first: a Transformer block stores 16 d numbers per token (the outputs of the layer norms, the attention projections, the feed-forward hidden layer twice and so on) plus, for every head, a table of T × T attention probabilities.

```python
def p1_act_bytes(model, B, T):
    """Activation bytes of one forward pass: (sum of the outputs of all leaf modules, bytes of tensors that autograd saved for backward)."""
    total, hooks = [0], []
    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(lambda mod, inp, out: total.__setitem__(0, total[0] + out.numel() * out.element_size())))
    saved, wptr = {}, {p.untyped_storage().data_ptr() for p in model.parameters()}
    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in wptr: saved[ptr] = t.untyped_storage().nbytes()
        return t
    ids = torch.randint(0, P1_V, (B, T))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(ids)
    for h in hooks: h.remove()
    return total[0], sum(saved.values())

p1_grid = [(B, T) for B in (1, 2, 4, 8) for T in (32, 64, 128, 256)]
p1_rows = [(B, T) + p1_act_bytes(p1_m, B, T) for (B, T) in p1_grid]
print(" batch  tokens  hook bytes (MB)  autograd saved (MB)")
for B, T, hb, sb in p1_rows:
    print(f"{B:6d} {T:7d} {hb / 1e6:15.2f} {sb / 1e6:20.2f}")
p1_x = {(B, T): hb for B, T, hb, sb in p1_rows}
print("doubling the batch: 8 x 128 over 4 x 128 =", round(p1_x[(8, 128)] / p1_x[(4, 128)], 3))
print("doubling the length: 4 x 256 over 4 x 128 =", round(p1_x[(4, 256)] / p1_x[(4, 128)], 3))
assert abs(p1_x[(8, 128)] / p1_x[(4, 128)] - 2) < 0.01
```

   The block prints (your numbers must match; timings depend on your computer):

```text
 batch  tokens  hook bytes (MB)  autograd saved (MB)
     1      32            2.77                 2.24
     1      64            5.68                 4.61
     1     128           11.88                 9.78
     1     256           25.85                21.78
     2      32            5.51                 4.47
     2      64           11.29                 9.20
     2     128           23.63                19.49
     2     256           51.45                43.30
     4      32           10.99                 8.93
     4      64           22.51                18.39
     4     128           47.12                38.90
     4     256          102.63                86.33
     8      32           21.95                17.85
     8      64           44.96                36.76
     8     128           94.11                77.74
     8     256          205.00               172.39
doubling the batch: 8 x 128 over 4 x 128 = 1.997
doubling the length: 4 x 256 over 4 x 128 = 2.178
```

   Doubling the **batch** doubles the activation memory (a factor of 1.997; not exactly 2.000 only because the position embedding is the same for every sequence): every sequence in the batch has its own benches. Doubling the **length** more than doubles it (2.18): that is the T × T attention table growing with the square of the length. The two measurements are of the same size and grow in the same way: 205 MB for the module outputs against 172 MB for the tensors that autograd really saves, at the largest setting. The hook count is about 19 % higher because some outputs (the last output of each residual branch, the final scores) are not needed again by the backward pass; so the hook count is a slight over-estimate and autograd's count is the exact need. We fit the hook count. Now fit the formula to the measurements.

```python
p1_Bs = np.array([r[0] for r in p1_rows], float); p1_Ts = np.array([r[1] for r in p1_rows], float); p1_y = np.array([r[2] for r in p1_rows], float)
p1_A = np.stack([p1_Bs * p1_Ts, p1_Bs * p1_Ts ** 2, p1_Ts], axis=1)
p1_coef, *_ = np.linalg.lstsq(p1_A, p1_y, rcond=None)
p1_theory = (4 * (P1_L * 16 * P1_D + 2 * P1_D + P1_V), 4 * P1_L * P1_H, 4 * P1_D)   # bytes per token, per (token x T) for the attention tables, per position of the position embedding
p1_fit = p1_A @ p1_coef
print("fitted:  bytes = %.0f x (B T) + %.0f x (B T^2) + %.0f x T" % tuple(p1_coef))
print("theory:  bytes = %.0f x (B T) + %.0f x (B T^2) + %.0f x T   [4 x (16 d L + 2 d + V), 4 x L x heads, 4 x d]" % p1_theory)
print("largest relative error of the fit: %.2e" % np.max(np.abs(p1_fit - p1_y) / p1_y))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for B in (1, 2, 4, 8):
    ax[0].plot([32, 64, 128, 256], [p1_x[(B, T)] / 1e6 for T in (32, 64, 128, 256)], "o-", label=f"batch {B}")
ax[0].set_xlabel("tokens per sequence T"); ax[0].set_ylabel("activation memory (MB)"); ax[0].set_title("grows with T, faster than a line"); ax[0].legend()
for T in (32, 64, 128, 256):
    ax[1].plot([1, 2, 4, 8], [p1_x[(B, T)] / 1e6 for B in (1, 2, 4, 8)], "o-", label=f"T = {T}")
ax[1].set_xlabel("batch size B"); ax[1].set_ylabel("activation memory (MB)"); ax[1].set_title("grows like a straight line in B"); ax[1].legend()
plt.tight_layout(); plt.show()
assert np.allclose(p1_coef, p1_theory, rtol=1e-6)
```

   The block prints (your numbers must match; timings depend on your computer):

```text
fitted:  bytes = 83584 x (B T) + 64 x (B T^2) + 1024 x T
theory:  bytes = 83584 x (B T) + 64 x (B T^2) + 1024 x T   [4 x (16 d L + 2 d + V), 4 x L x heads, 4 x d]
largest relative error of the fit: 1.82e-13
```

   The fit reproduces the theory to six digits: the activation memory is **83,584 bytes for every token in the batch plus 64 bytes for every token times the length** (and a tiny 1,024 bytes per position for the position embedding, which does not depend on the batch), which is exactly what counting the layers predicts (83,584, 64 and 1,024). The first term is the linear part (16 d numbers per token and layer, plus 2 d and the output layer's V scores per token); the second is the attention tables (L layers × h heads × T numbers per token). For a long context the second term wins, which is why long contexts are expensive and why **FlashAttention** (Week 8) avoids storing the table.

   Compare with the weights: the optimiser state of this model is 84.37 MB, and one batch of 8 sequences of 256 tokens needs **205 MB** of activations, which is **2.4 times** as much. The weights and the optimiser state do not depend on the batch; the activations do. That is why a bigger batch is the first thing that fails when memory runs out, and why people trade time for memory by recomputing activations (named in the lesson as activation checkpointing).

```python
if HAS_CUDA:
    torch.cuda.reset_peak_memory_stats()
    _g = P1LM().to("cuda"); _o = torch.optim.AdamW(_g.parameters(), lr=1e-3)
    _ids = torch.randint(0, P1_V, (8, 128), device="cuda")
    F.cross_entropy(_g(_ids[:, :-1]).reshape(-1, P1_V), _ids[:, 1:].reshape(-1)).backward(); _o.step()
    torch.cuda.synchronize()
    p1_peak = torch.cuda.max_memory_allocated()
    print(f"GPU peak memory of one training step: {p1_peak / 1e6:.1f} MB; weights + grads + Adam = {16 * p1_n / 1e6:.1f} MB; the rest is activations and temporary buffers")
    del _g, _o
else:
    print("No GPU found: this cell is skipped. On a GPU it prints torch.cuda.max_memory_allocated() for the same step;")
    print("expect a number a little above the optimiser state plus the activations of Step 1.3.")
```

   The block prints (your numbers must match; timings depend on your computer):

```text
No GPU found: this cell is skipped. On a GPU it prints torch.cuda.max_memory_allocated() for the same step;
expect a number a little above the optimiser state plus the activations of Step 1.3.
```

4. **Step 1.4. The memory table for big models.** Now the formula becomes a function. The weights of a model with `P` billion parameters at `b` bits take `P × 10⁹ × b / 8` bytes, which is `P × b / 8` gigabytes (with 1 GB = 10⁹ bytes, as in the lesson). Training needs 16 bytes **in total** per parameter (the weights are included in that number), plus the activations. Then the number of GPUs is the memory divided by the memory of one GPU, rounded **up**, because you cannot buy half a GPU. We assume GPUs with 80 GB; this is an example size, not a specification of any product.

   *Exercise 1.3:* The memory function. Your task: write three functions. `weights_gb(P, bits)` returns the weight memory in GB of a model with `P` billion parameters stored with `bits` bits per value. `train_gb(P)` returns the training state in GB at 16 bytes per parameter (without activations). `gpus_needed(gb, gpu_gb=80)` returns the number of GPUs of `gpu_gb` GB needed to hold `gb` GB (round up). The lesson's numbers must come out: 7 B at 32, 16, 8 and 4 bits is 28, 14, 7 and 3.5 GB, training 7 B is 112 GB, and for 70 B: 140 GB to hold (FP16) and 1,120 GB to train.

   Replace the starter, run it, then run the **Check** cell.

```python
def weights_gb(P, bits):
    # your code here
    return None

def train_gb(P):
    # your code here
    return None

def gpus_needed(gb, gpu_gb=80):
    # your code here
    return None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if weights_gb(7, 16) is None or train_gb(7) is None or gpus_needed(100) is None:
    print("Write the three functions above, then run this cell again.")
else:
    check("7 B at 32, 16, 8, 4 bits", [weights_gb(7, b) for b in (32, 16, 8, 4)] == [28, 14, 7, 3.5], "P * bits / 8 gigabytes")
    check("training 7 B takes 112 GB", abs(train_gb(7) - 112) < 1e-9, "16 bytes per parameter: 16 * P gigabytes")
    check("70 B: 140 GB to hold in FP16, 1,120 GB to train", abs(weights_gb(70, 16) - 140) < 1e-9 and abs(train_gb(70) - 1120) < 1e-9, "same formulas with P = 70")
    check("GPUs: 2 hold 70 B in FP16 and 14 train it", gpus_needed(140) == 2 and gpus_needed(1120) == 14, "round up: -(-gb // gpu_gb), or math.ceil")
    check("rounds up, never down", gpus_needed(80) == 1 and gpus_needed(80.1) == 2, "80 GB fits one GPU exactly, 80.1 GB needs two")
    RESULTS["p1_gpus_70b_train"] = gpus_needed(train_gb(70))
```

   A possible answer, and what it means:

```python
def weights_gb(P, bits):
    return P * bits / 8          # P billion values * bits / 8 bytes, in gigabytes

def train_gb(P):
    return 16 * P                # 16 bytes per parameter: P billion parameters -> 16 * P gigabytes

def gpus_needed(gb, gpu_gb=80):
    return math.ceil(gb / gpu_gb)
```

   The unit trick: a billion parameters at 1 byte each is 1 GB, so the number of gigabytes is just the number of billions times the bytes per parameter. The 14 GPUs for 70 B are a **lower bound**: activations, the KV cache and unused space on each GPU come on top, and real systems split the state unevenly.

   Below the exercise we run our version, so that the cells after it work. Then the lesson's table: for each model size and precision, the memory to hold the weights and the number of 80 GB GPUs; and the memory to train.

```python
def weights_gb(P, bits): return P * bits / 8
def train_gb(P): return 16 * P
def gpus_needed(gb, gpu_gb=80): return math.ceil(gb / gpu_gb)

assert [weights_gb(7, b) for b in (32, 16, 8, 4)] == [28, 14, 7, 3.5]
assert train_gb(7) == 112 and weights_gb(70, 16) == 140 and train_gb(70) == 1120
assert [gpus_needed(weights_gb(70, b)) for b in (32, 16, 8, 4)] == [4, 2, 1, 1] and gpus_needed(train_gb(70)) == 14
print("model   precision   weights GB   80 GB GPUs to hold")
for P in (7, 70):
    for b, name in ((32, "FP32"), (16, "FP16"), (8, "INT8"), (4, "INT4")):
        print(f"{P:3d} B   {name:9s} {weights_gb(P, b):9.1f}   {gpus_needed(weights_gb(P, b)):10d}")
print()
for P in (7, 70):
    print(f"training {P:2d} B (16 bytes per parameter, no activations): {train_gb(P):7,.0f} GB -> at least {gpus_needed(train_gb(P))} GPUs of 80 GB")
RESULTS["p1_gpus_70b_train"] = gpus_needed(train_gb(70))
```

   The block prints (your numbers must match; timings depend on your computer):

```text
model   precision   weights GB   80 GB GPUs to hold
  7 B   FP32           28.0            1
  7 B   FP16           14.0            1
  7 B   INT8            7.0            1
  7 B   INT4            3.5            1
 70 B   FP32          280.0            4
 70 B   FP16          140.0            2
 70 B   INT8           70.0            1
 70 B   INT4           35.0            1

training  7 B (16 bytes per parameter, no activations):     112 GB -> at least 2 GPUs of 80 GB
training 70 B (16 bytes per parameter, no activations):   1,120 GB -> at least 14 GPUs of 80 GB
```

   Read the table from the top. A 7 B model needs 14 GB in FP16, which fits one 80 GB GPU for *inference* but not for *training* (112 GB). A 70 B model needs 140 GB in FP16: 2 GPUs just to hold it, and **14** to train it (1,120 GB). Precision is the cheapest lever: 4-bit weights bring 70 B down to 35 GB, which fits one GPU. These are counts of bytes; they hold on every machine.

5. **Step 1.5. The truck: the time that training takes.** The other limit is time. Training a model takes about **C ≈ 6 N D** floating-point operations (FLOPs) for `N` parameters and `D` training tokens (Week 9): 2 N for the forward pass and 4 N for the backward pass, per token. To turn FLOPs into seconds we need the speed of the machine, in FLOP/s. We **measure it**: a matrix product of two n × n matrices costs 2 n³ operations, so dividing by the measured time gives the FLOP/s that this computer reaches on the operation that dominates a Transformer. Then we time a real training step of our small model and compare it with the 6 N D prediction.

```python
p1_n_mm = 1024
p1_a_mm, p1_b_mm = torch.randn(p1_n_mm, p1_n_mm), torch.randn(p1_n_mm, p1_n_mm)
p1_t_mm = time_ms(lambda: p1_a_mm @ p1_b_mm, repeat=5, warmup=2) / 1000
p1_flops_s = 2 * p1_n_mm ** 3 / p1_t_mm
print(f"1024 x 1024 matrix product: {p1_t_mm * 1000:.1f} ms -> {p1_flops_s / 1e9:.0f} GFLOP/s on this machine")

p1_m2 = P1LM(); p1_o2 = torch.optim.AdamW(p1_m2.parameters(), lr=1e-3)
p1_batch = torch.randint(0, P1_V, (8, 129))
def p1_step():
    p1_o2.zero_grad()
    F.cross_entropy(p1_m2(p1_batch[:, :-1]).reshape(-1, P1_V), p1_batch[:, 1:].reshape(-1)).backward(); p1_o2.step()
p1_t_step = time_ms(p1_step, repeat=3 if QUICK else 5, warmup=1) / 1000
p1_tokens = 8 * 128
p1_pred = 6 * p1_n * p1_tokens / p1_flops_s
print(f"one training step on {p1_tokens} tokens: {p1_t_step * 1000:.0f} ms measured; 6 N D / (matmul speed) predicts {p1_pred * 1000:.0f} ms")
print(f"-> the step reached {100 * 6 * p1_n * p1_tokens / p1_t_step / p1_flops_s:.0f}% of the matrix-product speed")

p1_C = 6 * 7e9 * 1e12
p1_sec_here = p1_C / p1_flops_s
p1_gpu_rate, p1_gpus = 200e12, 1024      # ASSUMED: a sustained 200 TFLOP/s per GPU and a cluster of 1,024 GPUs
print(f"\n7 B parameters on 1 T tokens: C = 6 N D = {p1_C:.2e} FLOPs")
print(f"  on this machine ({p1_flops_s / 1e9:.0f} GFLOP/s):   {p1_sec_here:.2e} s = {p1_sec_here / 3.156e7:,.0f} years")
print(f"  on 1 assumed GPU (200 TFLOP/s):   {p1_C / p1_gpu_rate:.2e} s = {p1_C / p1_gpu_rate / 3.156e7:.1f} years")
print(f"  on 1,024 assumed GPUs (perfect):  {p1_C / p1_gpu_rate / p1_gpus:.2e} s = {p1_C / p1_gpu_rate / p1_gpus / 86400:.1f} days")
assert abs(p1_C - 4.2e22) < 1e18
```

   The block prints (your numbers must match; timings depend on your computer):

```text
1024 x 1024 matrix product: 15.7 ms -> 137 GFLOP/s on this machine
one training step on 1024 tokens: 346 ms measured; 6 N D / (matmul speed) predicts 237 ms
-> the step reached 68% of the matrix-product speed

7 B parameters on 1 T tokens: C = 6 N D = 4.20e+22 FLOPs
  on this machine (137 GFLOP/s):   3.07e+11 s = 9,726 years
  on 1 assumed GPU (200 TFLOP/s):   2.10e+08 s = 6.7 years
  on 1,024 assumed GPUs (perfect):  2.05e+05 s = 2.4 days
```

   On the machine that ran this lab the matrix product reaches **137 GFLOP/s**. A real training step of our small model takes 346 ms, and the 6 N D formula, with the measured matrix speed, predicts 237 ms: the formula is a **floor**, because a real step also does the attention, layer norms, softmax and optimiser update, which are not matrix products and run slower. Training a 7-billion-parameter model on a trillion tokens needs 6 × 7 × 10⁹ × 10¹² = **4.2 × 10²² FLOPs**. At this machine's speed that is about **9,726 years**, which is absurd; at an *assumed* sustained 200 TFLOP/s one GPU would still need 6.7 years, and 1,024 such GPUs working perfectly together about **2.4 days**. The 200 TFLOP/s and the 1,024 GPUs are assumptions for the arithmetic, not facts about any product, and "perfectly together" is what Parts 2 and 3 show you must pay for.

6. **Step 1.6. Memory or time? A decision.** Both limits can bind, one, or neither. Write a function that takes a scenario and says which. A scenario is a model of `P` billion parameters that is either **trained** on `D` billion tokens before a deadline, or only **held** for inference, on GPUs with 80 GB, at an assumed sustained 200 TFLOP/s. The rules: memory is a problem if the state (16 bytes per parameter for training, 2 bytes for FP16 inference) is above the GPU memory; time is a problem if one GPU would need more than the deadline (training only).

```python
# The four scenarios, and your guesses (write them BEFORE you write the function)
SCENARIOS = [("train 7 B on 1,000 B tokens in 30 days", 7, "train", 1000, 30),
             ("serve 70 B in FP16", 70, "infer", 0, None),
             ("train 0.3 B on 300 B tokens in 7 days", 0.3, "train", 300, 7),
             ("fine-tune 1 B on 0.1 B tokens in 3 days", 1, "train", 0.1, 3)]
my_answers = [None, None, None, None]   # write "memory", "time", "both" or "neither" for each scenario
```

   *Exercise 1.4:* Memory or time?. Your task: write `which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12)` that returns `"memory"`, `"time"`, `"both"` or `"neither"`. `mode` is `"train"` or `"infer"`. For `"train"` use `train_gb(P)` for memory and the time `6 * P * 1e9 * D_b * 1e9 / rate` seconds against `days * 86400`. For `"infer"` use `weights_gb(P, 16)` and ignore time. Then classify the four scenarios in the list `my_answers` (replace each `None` by one of the four words) **before** you run the function on them, and compare. The four scenarios are in the cell above: (1) train a 7 B model on 1,000 B tokens in 30 days; (2) serve a 70 B model in FP16 for the district; (3) train a 0.3 B model on 300 B tokens in 7 days; (4) fine-tune a 1 B model on 0.1 B tokens in 3 days.

```python
def which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12):
    # your code here
    return None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if which_limit(7, "train", 1000, 30) is None:
    print("Write which_limit above, then run this cell again.")
else:
    got = [which_limit(P, mode, D, days) for (_, P, mode, D, days) in SCENARIOS]
    check("scenario 1 needs both", got[0] == "both", "112 GB does not fit 80 GB, and 6.7 years is more than 30 days")
    check("scenario 2 is a memory problem", got[1] == "memory", "140 GB against 80 GB; inference has no training deadline here")
    check("scenario 3 is a time problem", got[2] == "time", "4.8 GB fits easily; 5.4e20 FLOPs at 2e14 FLOP/s is about 31 days > 7")
    check("scenario 4 is neither", got[3] == "neither", "16 GB fits; 6e17 FLOPs is under an hour")
    print("your guesses :", my_answers); print("the function :", got)
    RESULTS["p1_guess_hits"] = sum(a == b for a, b in zip(my_answers, got))
```

   A possible answer, and what it means:

```python
def which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12):
    if mode == "train":
        mem_bad = train_gb(P) > gpu_gb
        time_bad = days is not None and 6 * P * 1e9 * D_b * 1e9 / rate > days * 86400
    else:
        mem_bad = weights_gb(P, 16) > gpu_gb
        time_bad = False
    return "both" if mem_bad and time_bad else "memory" if mem_bad else "time" if time_bad else "neither"
```

   Training a 0.3 B model on 300 B tokens is the interesting one: it fits in memory with room to spare (4.8 GB), but at one GPU's assumed speed it takes about 31 days, so the problem is the truck, not the warehouse. The fix is **data parallelism** (Part 2): several copies of the model, each reading a slice of the data.

   Below the exercise we run our version, so that the next cell works.

```python
def which_limit(P, mode, D_b=0, days=None, gpu_gb=80, rate=200e12):
    if mode == "train":
        mem_bad = train_gb(P) > gpu_gb
        time_bad = days is not None and 6 * P * 1e9 * D_b * 1e9 / rate > days * 86400
    else:
        mem_bad = weights_gb(P, 16) > gpu_gb
        time_bad = False
    return "both" if mem_bad and time_bad else "memory" if mem_bad else "time" if time_bad else "neither"
p1_got = [which_limit(P, mode, D, days) for (_, P, mode, D, days) in SCENARIOS]
for (name, P, mode, D, days), g in zip(SCENARIOS, p1_got):
    t_days = 6 * P * 1e9 * D * 1e9 / 200e12 / 86400 if mode == "train" else 0
    print(f"{name:45s} memory {train_gb(P) if mode == 'train' else weights_gb(P, 16):7.1f} GB   one-GPU time {t_days:9.2f} days  ->  {g}")
assert p1_got == ["both", "memory", "time", "neither"]
```

   The block prints (your numbers must match; timings depend on your computer):

```text
train 7 B on 1,000 B tokens in 30 days        memory   112.0 GB   one-GPU time   2430.56 days  ->  both
serve 70 B in FP16                            memory   140.0 GB   one-GPU time      0.00 days  ->  memory
train 0.3 B on 300 B tokens in 7 days         memory     4.8 GB   one-GPU time     31.25 days  ->  time
fine-tune 1 B on 0.1 B tokens in 3 days       memory    16.0 GB   one-GPU time      0.03 days  ->  neither
```

   The answers are **both**, **memory**, **time** and **neither**. Notice scenario 3: a small model can still be too slow, and scenario 2: a model can be fine in time and impossible in memory. The next parts attack each limit with a different tool. **Data parallelism** (Part 2) attacks time when the model fits; **tensor and pipeline parallelism** (Part 3) attack memory when it does not; **mixed precision and quantization** (Parts 4 and 5) shrink the bytes; **batching and the KV cache** (Part 6) decide what a server can hold.

### Part 2. Data parallelism by hand and for real (30 min)

*Lesson section: "Data parallelism".* The company's order book is too big for one truck to finish in time (the time limit of Part 1). So it buys **K identical trucks**, gives each a copy of the same plan (the model) and a **slice of the day's orders** (the batch). Each truck works out how the plan should change, they meet at the end of the day, **average** their notes, and everybody updates the plan in exactly the same way, so all the copies stay identical. That meeting is the **all-reduce**, and it is the price of ==data parallelism: the compute is shared, but every step must send the gradients around==. In this part you do the averaging by hand and prove it equals one big batch, then run **real** training in several processes with `torch.distributed`, and measure what the meeting costs on this machine.

**Imagine first.** (1) You split a batch of 32 examples into 4 shards of 8, compute the gradient of the mean loss on each shard and **average** the 4 gradients. How different is the result from the gradient on all 32 examples at once: not at all (up to rounding), a little, or a lot? (2) If the shards had **different sizes** (10, 20 and 34 examples) and you still took the plain average of the three gradients, would the result be right? (3) A model with 100 MB of gradients is averaged by 4 processes on this computer. The all-reduce moves 2(N − 1)/N × S bytes per process. Roughly how long does it take: under 1 ms, about 10 ms, about 100 ms, or about a second? Write your guesses in the block below (replace each `None`), then run it. You compare with Steps 2.1, 2.2 and 2.5.

```python
# Imagine first: the trucks
my_guess_equal = None    # "same", "little" or "lot": equal shards, averaged gradients against the full-batch gradient
my_guess_unequal = None  # True if the plain average of unequal shards is still right, else False
my_guess_ms = None       # milliseconds for an all-reduce of 100 MB between 4 processes on this computer (a number)
```

1. **Step 2.1. Split the batch, average the gradients.** We use a small regression network (8 inputs, 16 hidden units, 1 output) and a batch of 32 examples. `torch.autograd.grad(loss, params)` returns the gradient of a loss with respect to the parameters **without** storing it in `p.grad`, which is exactly what a shard needs. We compute the gradient of the mean squared error on the **whole batch**, then on each of K = 4 shards of 8 examples, average the four gradients and compare. The loss is a **mean** over the examples, so a gradient on a shard is the average of 8 per-example gradients, and the average of four such averages over equal shards is the average over all 32.

```python
def p2_make(seed=SEED):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))

def p2_grads(model, X, Y):
    """Gradient of the mean squared error of (X, Y) with respect to every parameter, as a list of tensors."""
    params = list(model.parameters())
    return list(torch.autograd.grad(F.mse_loss(model(X), Y), params))

torch.manual_seed(SEED)
p2_X, p2_Y = torch.randn(32, 8), torch.randn(32, 1)
p2_net = p2_make()
p2_full = p2_grads(p2_net, p2_X, p2_Y)
p2_K = 4
p2_shards = [p2_grads(p2_net, p2_X[i::p2_K], p2_Y[i::p2_K]) for i in range(p2_K)]       # shard i takes every 4th example, starting at i
p2_mean = [sum(g[j] for g in p2_shards) / p2_K for j in range(len(p2_full))]
p2_maxdiff = max((a - b).abs().max().item() for a, b in zip(p2_mean, p2_full))
p2_scale = max(g.abs().max().item() for g in p2_full)
print("tensors in the gradient:", len(p2_full), "| numbers:", sum(g.numel() for g in p2_full))
print(f"largest gradient entry: {p2_scale:.4f}")
print(f"largest difference between the average of 4 shard gradients and the full-batch gradient: {p2_maxdiff:.2e}")
print("shard 0 vs shard 1 (first layer weight), first 3 entries:", p2_shards[0][0].flatten()[:3].tolist(), p2_shards[1][0].flatten()[:3].tolist())
assert p2_maxdiff < 1e-6
```

   The block prints (your numbers must match; timings depend on your computer):

```text
tensors in the gradient: 4 | numbers: 161
largest gradient entry: 0.5255
largest difference between the average of 4 shard gradients and the full-batch gradient: 5.96e-08
shard 0 vs shard 1 (first layer weight), first 3 entries: [-0.01589033752679825, -0.009875942021608353, 0.011371574364602566] [0.00793654378503561, 0.01935095340013504, -0.02958216890692711]
```

   The average of the four shard gradients differs from the full-batch gradient by at most **6.0e-08**, while the gradient entries themselves are as large as 0.53: the difference is only the rounding of 32-bit numbers (adding in a different order). The individual shard gradients are quite different from one another (look at the first three entries printed above), but their **average is exact**. That is the whole idea of data parallelism: averaging gradients costs nothing in accuracy.

   *Exercise 2.1:* Average the gradients of the shards. Your task: write `average_grads(grad_lists, sizes=None)`. `grad_lists` is a list with one entry per shard, and each entry is a list of gradient tensors (one per parameter, as returned by `p2_grads`). Return a list with the averaged gradient of each parameter. When `sizes` (the number of examples in each shard) is given, use a **weighted** average with weights `size / sum(sizes)`; when it is `None`, give every shard the same weight. Why it matters: Step 2.2 shows that the weights are necessary when the shards are not equal.

   Replace the starter, run it, then run the **Check** cell.

```python
def average_grads(grad_lists, sizes=None):
    # your code here
    return None
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if average_grads(p2_shards) is None:
    print("Write average_grads above, then run this cell again.")
else:
    _a = average_grads(p2_shards)
    check("equal shards: matches the full-batch gradient", all(torch.allclose(x, y, atol=1e-6) for x, y in zip(_a, p2_full)), "add the shard gradients and divide by the number of shards")
    _g1, _g2 = p2_grads(p2_net, p2_X[:10], p2_Y[:10]), p2_grads(p2_net, p2_X[10:], p2_Y[10:])
    _w = average_grads([_g1, _g2], sizes=[10, 22])
    check("weighted: matches the full-batch gradient for shards of 10 and 22", all(torch.allclose(x, y, atol=1e-6) for x, y in zip(_w, p2_full)), "weight each shard by size / total size")
    check("returns one tensor per parameter", len(_a) == len(p2_full) and all(x.shape == y.shape for x, y in zip(_a, p2_full)), "keep the shapes of the parameters")
    RESULTS["p2_avg_grads_ok"] = True
```

   A possible answer, and what it means:

```python
def average_grads(grad_lists, sizes=None):
    K = len(grad_lists)
    w = [1 / K] * K if sizes is None else [s / sum(sizes) for s in sizes]
    return [sum(w[i] * grad_lists[i][j] for i in range(K)) for j in range(len(grad_lists[0]))]
```

   `j` runs over the parameters and `i` over the shards. The weights add up to 1, so the result is a weighted mean. In a real system this sum is what `all_reduce` computes (with the sum divided by the number of processes afterwards).

   Below the exercise we run our version, so that the next steps work without your answer.

```python
def average_grads(grad_lists, sizes=None):
    K = len(grad_lists)
    w = [1 / K] * K if sizes is None else [s / sum(sizes) for s in sizes]
    return [sum(w[i] * grad_lists[i][j] for i in range(K)) for j in range(len(grad_lists[0]))]
assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(average_grads(p2_shards), p2_full))
print("average_grads works on the 4 equal shards")
```

   The block prints (your numbers must match; timings depend on your computer):

```text
average_grads works on the 4 equal shards
```

2. **Step 2.2. Unequal shards need a weighted average.** What if one truck gets more orders than another? Take shards of 10, 20 and 2 examples (the sizes of a batch of 32 that does not divide by 3). The plain average of the three shard gradients gives the small shard the same say as the big one, which is wrong. The weighted average with weights `size / total` is right. We measure both errors against the full-batch gradient.

```python
p2_sizes = [10, 20, 2]
p2_edges = np.cumsum([0] + p2_sizes)
p2_uneq = [p2_grads(p2_net, p2_X[a:b], p2_Y[a:b]) for a, b in zip(p2_edges[:-1], p2_edges[1:])]
def p2_err(gl): return max((a - b).abs().max().item() for a, b in zip(gl, p2_full))
p2_err_plain, p2_err_w = p2_err(average_grads(p2_uneq)), p2_err(average_grads(p2_uneq, sizes=p2_sizes))
print("shard sizes:", p2_sizes)
print(f"plain average of the three shard gradients: largest error {p2_err_plain:.4f}")
print(f"weighted average (size / total):            largest error {p2_err_w:.2e}")
print("relative size of the plain-average error: %.0f%% of the largest gradient entry" % (100 * p2_err_plain / p2_scale))
assert p2_err_w < 1e-6 and p2_err_plain > 100 * p2_err_w
```

   The block prints (your numbers must match; timings depend on your computer):

```text
shard sizes: [10, 20, 2]
plain average of the three shard gradients: largest error 0.1386
weighted average (size / total):            largest error 1.19e-07
relative size of the plain-average error: 26% of the largest gradient entry
```

   The plain average is wrong by **0.139**, which is 26 % of the largest gradient entry, while the weighted average is exact again (1e-07). In practice frameworks avoid the problem by giving every process the same number of examples, or by **dropping or padding** the last batch; but when you write the loss with `sum` instead of `mean`, or when sequences have different lengths, the same weighting question comes back.

3. **Step 2.3. One step with averaged gradients equals one big-batch step.** The last piece: after the averaging, every copy takes **the same** SGD step, `p ← p − lr × gradient`. We take one step on a copy of the model with the full-batch gradient, and one step with the averaged shard gradients on another copy, and assert that the two models are the same afterwards. The consequence for the learning rate: with K copies each reading b examples, the **effective batch size** is K × b, so the data-parallel run is a big-batch run, and the common (heuristic, not a law) advice is to scale the learning rate with it and warm it up.

```python
import copy
p2_lr = 0.1
p2_a, p2_b = copy.deepcopy(p2_net), copy.deepcopy(p2_net)
with torch.no_grad():
    for p, g in zip(p2_a.parameters(), p2_full): p -= p2_lr * g                           # one big-batch step
    for p, g in zip(p2_b.parameters(), average_grads(p2_shards)): p -= p2_lr * g          # one data-parallel step
p2_pd = max((x - y).abs().max().item() for x, y in zip(p2_a.parameters(), p2_b.parameters()))
p2_move = max((x - y).abs().max().item() for x, y in zip(p2_a.parameters(), p2_net.parameters()))
print(f"largest difference between the two models after one step: {p2_pd:.2e}")
print(f"largest change of any parameter during the step:           {p2_move:.2e}")
print(f"effective batch size: {p2_K} shards x 8 examples = {p2_K * 8}")
assert p2_pd < 1e-6
```

   The block prints (your numbers must match; timings depend on your computer):

```text
largest difference between the two models after one step: 1.49e-08
largest change of any parameter during the step:           5.26e-02
effective batch size: 4 shards x 8 examples = 32
```

   The two models differ by 1e-08, while one step moved some parameter by 0.053: they are the same model. This is why data parallelism is called **exact**: it is not an approximation of big-batch training, it *is* big-batch training, spread over several machines. What it does **not** change is that every copy holds the full weights, gradients and optimiser state (Part 1): data parallelism buys **time**, not memory.

4. **Step 2.4. A real data-parallel run with `torch.distributed`.** Now the real thing: several **processes**, each with its own copy of the model, connected by `torch.distributed`. We use the **gloo** backend, which works on any CPU (on GPUs the usual backend is NCCL). `DistributedDataParallel` (DDP) wraps a model: after the backward pass it runs an all-reduce over the gradients in the background, so every process ends the step with the same averaged gradient.

   The code of one process is written to a small temporary file and started with `subprocess`, one process per rank, because a Jupyter kernel cannot be split into several processes. Each rank takes its own slice of the same batch, trains 5 steps with SGD, and prints a **checksum** (the sum of the absolute values of all parameters). We then train the same model on the whole batch in **this** process and compare. Two things must be true: all ranks agree with each other, and they agree with the single-process run. The cell is written defensively: if processes cannot be started on your platform, it catches the problem and says what to do.

```python
import socket, tempfile, shutil
P2_DDP_SCRIPT = '''
import os, sys, json, time, datetime, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:" + os.environ["PORT"], rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(128, 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 1))
g = torch.Generator().manual_seed(1)
X = torch.randn(256, 128, generator=g); Y = torch.randn(256, 1, generator=g)
shard = slice(rank * 256 // world, (rank + 1) * 256 // world)
ddp = DDP(model)
opt = torch.optim.SGD(ddp.parameters(), lr=0.05)
def step(sync=True):
    opt.zero_grad()
    if sync: nn.functional.mse_loss(ddp(X[shard]), Y[shard]).backward()
    else:
        with ddp.no_sync(): nn.functional.mse_loss(ddp(X[shard]), Y[shard]).backward()
    opt.step()
for _ in range(5): step()
checksum = sum(float(p.detach().double().abs().sum()) for p in model.parameters())
def timeit(sync, n=20):
    step(sync); dist.barrier(); t = time.perf_counter()
    for _ in range(n): step(sync)
    dist.barrier(); return (time.perf_counter() - t) / n
t_local = timeit(False); t_sync = timeit(True)
print("RESULT " + json.dumps({"rank": rank, "checksum": checksum, "t_local": t_local, "t_sync": t_sync}))
dist.destroy_process_group()
'''
P2_AR_SCRIPT = '''
import os, json, time, datetime, torch, torch.distributed as dist
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:" + os.environ["PORT"], rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
SIZES = [1000, 10000, 100000, 1000000, 10000000] + ([] if os.environ.get("NB_TEST") else [100000000])
res = {}
for nbytes in SIZES:
    x = torch.ones(nbytes // 4); reps = 30 if nbytes <= 1000000 else 5
    dist.all_reduce(x); dist.barrier(); t = time.perf_counter()
    for _ in range(reps): dist.all_reduce(x)
    res[nbytes] = (time.perf_counter() - t) / reps
if rank == 0: print("RESULT " + json.dumps(res))
dist.destroy_process_group()
'''

def p2_launch(script_text, world, timeout=120):
    """Run script_text in `world` processes (ranks 0..world-1) that talk through gloo on this computer. Returns the list of dictionaries that the ranks printed after 'RESULT ', or None (with a message) if something fails."""
    tmp = tempfile.mkdtemp(prefix="w18_")
    try:
        path = os.path.join(tmp, "worker.py")
        open(path, "w").write(script_text)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        threads = max(1, torch.get_num_threads() // world)
        procs = []
        for r in range(world):
            env = dict(os.environ, RANK=str(r), WORLD_SIZE=str(world), PORT=str(port), THREADS=str(threads), OMP_NUM_THREADS=str(threads))
            procs.append(subprocess.Popen([sys.executable, path], env=env, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        results = []
        try:
            for p in procs:
                out, err = p.communicate(timeout=timeout)
                if p.returncode != 0:
                    print("a worker failed:", err.strip().splitlines()[-1] if err.strip() else "no message"); return None
                results += [json.loads(l[7:]) for l in out.splitlines() if l.startswith("RESULT ")]
        except subprocess.TimeoutExpired:
            for p in procs: p.kill()
            print(f"the workers did not finish in {timeout} s and were stopped"); return None
        return results
    except Exception as e:
        print("could not start worker processes:", repr(e)); return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def p2_single(steps=5):
    """The same training in this process on the whole batch (what the ranks must reproduce together)."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(128, 1024), nn.Tanh(), nn.Linear(1024, 1024), nn.Tanh(), nn.Linear(1024, 1))
    g = torch.Generator().manual_seed(1)
    X = torch.randn(256, 128, generator=g); Y = torch.randn(256, 1, generator=g)
    o = torch.optim.SGD(m.parameters(), lr=0.05)
    def st():
        o.zero_grad(); F.mse_loss(m(X), Y).backward(); o.step()
    for _ in range(steps): st()
    return m, st

p2_ddp = {}
for W in (2, 4):
    t0 = time.perf_counter()
    r = p2_launch(P2_DDP_SCRIPT, W)
    if r is None:
        print(f"world size {W}: the multi-process run failed; see the note below the cell.")
    else:
        p2_ddp[W] = sorted(r, key=lambda d: d["rank"])
        p2_ranks = p2_ddp[W]
        p2_cs = [round(d["checksum"], 6) for d in p2_ranks]
        print(f"world size {W}: {len(r)} processes finished in {time.perf_counter() - t0:.1f} s; checksums of the ranks:", p2_cs)
p2_m1, p2_st1 = p2_single()
p2_check1 = sum(float(p.detach().double().abs().sum()) for p in p2_m1.parameters())
print(f"single process, whole batch: checksum {p2_check1:.6f}")
for W, r in p2_ddp.items():
    cs = [d["checksum"] for d in r]
    print(f"world size {W}: spread between ranks {max(cs) - min(cs):.2e}; difference from single process {abs(cs[0] - p2_check1):.2e} (relative {abs(cs[0] - p2_check1) / p2_check1:.1e})")
    assert max(cs) - min(cs) < 1e-6 * p2_check1 and abs(cs[0] - p2_check1) < 1e-5 * p2_check1
```

   The block prints (your numbers must match; timings depend on your computer):

```text
world size 2: 2 processes finished in 7.3 s; checksums of the ranks: [22252.749442, 22252.749442]
world size 4: 4 processes finished in 5.4 s; checksums of the ranks: [22252.749443, 22252.749443, 22252.749443, 22252.749443]
single process, whole batch: checksum 22252.749443
world size 2: spread between ranks 0.00e+00; difference from single process 4.24e-07 (relative 1.9e-11)
world size 4: spread between ranks 0.00e+00; difference from single process 7.99e-08 (relative 3.6e-12)
```

   With 2 and with 4 processes, **all the ranks end with the same parameters** (the checksums agree), and they equal the parameters of ordinary training on the whole batch in one process, up to floating-point rounding: the relative difference is below one part in 100,000. That is what DDP guarantees, and what Step 2.1 proved on paper: averaged shard gradients are the full-batch gradient.

   *If this cell printed "the multi-process run failed":* on some platforms (certain Windows or notebook setups, or a firewall that blocks local ports) the processes cannot connect. Try running the cell again (a random free port is chosen each time), close other heavy programs, or run the notebook as a normal Python script. The rest of the lab does not depend on it, because the later steps use the numbers that you measured here only if they exist, and say so if they do not.

5. **Step 2.5. How long does the meeting take? Measure the all-reduce.** The all-reduce is the communication of data parallelism, so we measure it. Each process holds a tensor of S bytes and `dist.all_reduce` replaces it by the **sum** over all processes. We time it for sizes from 1 KB to 100 MB with 2 and with 4 processes. In the ring algorithm each process sends and receives about **2(N − 1)/N × S** bytes (the lesson's formula), so the **effective bandwidth** is that traffic divided by the time. For tiny messages the time is dominated by a fixed **latency** (the cost of starting a message), for big ones by the bandwidth. We estimate the latency as the time of the 1 KB message and the bandwidth from the slope between the two largest sizes. We run the measurement twice and keep the faster time of each size, because other programs on your computer disturb timings.

```python
p2_ar = {}
for W in (2, 4):
    runs = [p2_launch(P2_AR_SCRIPT, W, timeout=240) for _ in range(1 if QUICK else 2)]       # two launches; we keep the faster time of each size (less noise)
    runs = [r[0] for r in runs if r]
    if runs: p2_ar[W] = {int(k): min(float(r[k]) for r in runs) for k in runs[0]}
p2_bw, p2_alpha = {}, {}
if p2_ar:
    print(f"{'size':>9}", *[f"{'N=' + str(W) + ' time':>12}{'GB/s':>8}" for W in p2_ar])
    for nb in sorted(next(iter(p2_ar.values()))):
        row = [f"{human_bytes(nb):>9}"]
        for W in p2_ar:
            traffic = 2 * (W - 1) / W * nb
            row.append(f"{p2_ar[W][nb] * 1000:9.3f} ms{traffic / p2_ar[W][nb] / 1e9:8.2f}")
        print(*row)
    for W, d in p2_ar.items():
        sz = sorted(d); a, b = sz[-2], sz[-1]
        p2_bw[W] = 2 * (W - 1) / W * (b - a) / (d[b] - d[a])
        first = sz[0]
        p2_alpha[W] = d[first]
        print(f"N = {W}: latency about {p2_alpha[W] * 1e6:.0f} microseconds, bandwidth about {p2_bw[W] / 1e9:.1f} GB/s (from the two largest sizes)")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for W, d in p2_ar.items():
        sz = sorted(d)
        ax[0].loglog(sz, [d[s] * 1000 for s in sz], "o-", label=f"N = {W}")
        ax[1].semilogx(sz, [2 * (W - 1) / W * s / d[s] / 1e9 for s in sz], "o-", label=f"N = {W}")
    ax[0].set_xlabel("message size S (bytes)"); ax[0].set_ylabel("all-reduce time (ms)"); ax[0].set_title("flat for small S (latency), a line for large S"); ax[0].legend()
    ax[1].set_xlabel("message size S (bytes)"); ax[1].set_ylabel("effective bandwidth (GB/s)"); ax[1].set_title("small messages waste the link"); ax[1].legend()
    plt.tight_layout(); plt.show()
    assert all(p2_ar[W][max(p2_ar[W])] > p2_ar[W][min(p2_ar[W])] for W in p2_ar)
else:
    print("no all-reduce measurements: see the note below the cell")
```

   The block prints (your numbers must match; timings depend on your computer):

```text
     size     N=2 time    GB/s     N=4 time    GB/s
  1.00 KB     0.294 ms    0.00     0.923 ms    0.00
 10.00 KB     0.326 ms    0.03     0.881 ms    0.02
100.00 KB     0.353 ms    0.28     1.051 ms    0.14
  1.00 MB     0.805 ms    1.24     1.834 ms    0.82
 10.00 MB     6.147 ms    1.63    14.073 ms    1.07
100.00 MB    55.880 ms    1.79   137.754 ms    1.09
N = 2: latency about 294 microseconds, bandwidth about 1.8 GB/s (from the two largest sizes)
N = 4: latency about 923 microseconds, bandwidth about 1.1 GB/s (from the two largest sizes)
```

   On the machine that ran this lab, a message of 1 KB takes about 294 microseconds between 2 processes: that is **latency**, because the message is tiny. The time hardly changes up to 100 KB (0.29 ms for 1 KB, 0.35 ms for 100 KB) and after that it grows in proportion to the size. For the largest messages the effective bandwidth reaches about **1.8 GB/s with 2 processes** and **1.1 GB/s with 4** (the 4 processes share the same cores and memory, so they do not get faster). Two lessons. First, **many small messages are slow**: one big all-reduce beats a thousand small ones, which is why DDP groups the gradients into buckets of several megabytes. Second, these numbers are for processes that share one computer's memory; a real cluster link is faster inside a machine and much slower between machines, and its numbers have to be measured the same way. Your own numbers will differ from ours, and will vary a little from run to run: that is the point of measuring.

   Now we use the measured bandwidth in the lesson's formulas.

6. **Step 2.6. Predict the cost of a step, then compare with the DDP run.** The lesson gives two formulas. The **traffic** per process is 2(N − 1)/N × S bytes, so the all-reduce time is about `latency + traffic / bandwidth`. The **efficiency** of a data-parallel step is compute ÷ (compute + communication), and the speed-up over one copy is N × efficiency (if the all-reduce does not overlap with the computation; real systems overlap part of it, which only helps). For the lesson's example, a 7-billion-parameter model has 14 GB of FP16 gradients; with N = 8 the traffic is 2 × 7/8 × 14 = **24.5 GB** per process, and at an assumed 100 GB/s that is **245 ms**.

   *Exercise 2.2:* The data-parallel cost function. Your task: write `dp_cost(Tc, S, N, bw, alpha=0.0)` that returns a dictionary with the keys `"traffic"` (bytes per process: 2(N − 1)/N × S), `"comm"` (seconds: `alpha + traffic / bw`), `"eff"` (efficiency: `Tc / (Tc + comm)`) and `"speedup"` (`N × eff`). `Tc` is the compute time of one step in seconds, `S` the size of the gradients in bytes, `bw` the bandwidth in bytes per second. Check your function on the lesson's example: Tc = 1 s, S = 14 GB, N = 8, bw = 100 GB/s.

   Replace the starter, run it, then run the **Check** cell.

```python
def dp_cost(Tc, S, N, bw, alpha=0.0):
    # your code here
    return {}
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
_c = dp_cost(1.0, 14e9, 8, 100e9)
if not _c:
    print("Write dp_cost above, then run this cell again.")
else:
    check("traffic is 24.5 GB", abs(_c["traffic"] - 24.5e9) < 1e3, "2 * (N - 1) / N * S with N = 8 and S = 14e9")
    check("all-reduce takes 245 ms", abs(_c["comm"] - 0.245) < 1e-6, "traffic / bw")
    check("efficiency about 0.803 and speed-up about 6.42", abs(_c["eff"] - 1 / 1.245) < 1e-6 and abs(_c["speedup"] - 8 / 1.245) < 1e-5, "eff = Tc / (Tc + comm); speedup = N * eff")
    check("one process has no traffic", dp_cost(1.0, 14e9, 1, 100e9)["traffic"] == 0 and abs(dp_cost(1.0, 14e9, 1, 100e9)["speedup"] - 1) < 1e-12, "with N = 1 the formula gives 0 bytes and a speed-up of 1")
    RESULTS["p2_example_speedup"] = round(_c["speedup"], 2)
```

   A possible answer, and what it means:

```python
def dp_cost(Tc, S, N, bw, alpha=0.0):
    traffic = 2 * (N - 1) / N * S
    comm = alpha + traffic / bw
    eff = Tc / (Tc + comm)
    return {"traffic": traffic, "comm": comm, "eff": eff, "speedup": N * eff}
```

   With N = 1 the traffic is 0 and, with the default `alpha = 0.0`, the communication time is 0 too: a single process does no all-reduce at all. The 245 ms on a 1 s step is a 20 % tax, which is why a fast link, or overlapping the all-reduce with the backward pass, matters.

   Below the exercise we run our version. Then we compare the prediction with the measured DDP runs of Step 2.4: the **compute** time is the time of a step without synchronisation (`no_sync`), the total is the normal step, and the difference is the communication. We predict that difference from the measured latency and bandwidth of Step 2.5, using the real gradient size of the DDP model (its parameters times 4 bytes).

```python
def dp_cost(Tc, S, N, bw, alpha=0.0):
    traffic = 2 * (N - 1) / N * S
    comm = alpha + traffic / bw
    eff = Tc / (Tc + comm)
    return {"traffic": traffic, "comm": comm, "eff": eff, "speedup": N * eff}
RESULTS["p2_example_speedup"] = round(dp_cost(1.0, 14e9, 8, 100e9)["speedup"], 2)
assert abs(dp_cost(1.0, 14e9, 8, 100e9)["traffic"] - 24.5e9) < 1e3 and abs(dp_cost(1.0, 14e9, 8, 100e9)["comm"] - 0.245) < 1e-9

p2_S = sum(p.numel() for p in p2_m1.parameters()) * 4
p2_t_single = time_ms(p2_st1, repeat=10, warmup=2) / 1000     # one step of the whole batch in this process (all threads)
p2_rows = {}
print(f"gradient size of the DDP model: {human_bytes(p2_S)}; one step of the whole batch in one process: {p2_t_single * 1000:.1f} ms")
for W in p2_ddp:
    ranks = p2_ddp[W]
    tl = float(np.mean([d["t_local"] for d in ranks]))
    ts = float(np.mean([d["t_sync"] for d in ranks]))
    if W in p2_bw:
        pred = dp_cost(tl, p2_S, W, p2_bw[W], alpha=p2_alpha[W])
        p2_rows[W] = (tl, ts, ts - tl, pred["comm"], tl / ts, pred["eff"])
        print(f"N = {W}: compute {tl * 1000:6.2f} ms | step with all-reduce {ts * 1000:6.2f} ms | communication measured {(ts - tl) * 1000:6.2f} ms, predicted {pred['comm'] * 1000:6.2f} ms | efficiency measured {tl / ts:.2f}, predicted {pred['eff']:.2f}")
        print(f"       speed-up over one process on the whole batch: {p2_t_single / ts:.2f} x")
```

   The block prints (your numbers must match; timings depend on your computer):

```text
gradient size of the DDP model: 4.73 MB; one step of the whole batch in one process: 9.4 ms
N = 2: compute  20.08 ms | step with all-reduce  37.27 ms | communication measured  17.19 ms, predicted   2.91 ms | efficiency measured 0.54, predicted 0.87
       speed-up over one process on the whole batch: 0.25 x
N = 4: compute  15.70 ms | step with all-reduce  28.44 ms | communication measured  12.74 ms, predicted   7.42 ms | efficiency measured 0.55, predicted 0.68
       speed-up over one process on the whole batch: 0.33 x
```

   For the all-reduce of 4.73 MB the formula predicted 2.9 ms with 2 processes and 7.4 ms with 4; the measured difference between a synchronised and an unsynchronised step was 17.2 ms and 12.7 ms (a ratio of 5.9 and 1.7). The formula gives the right order of magnitude and the right trend (more processes, more communication), but it counts only the bytes on the wire: a real step also pays for splitting the gradients into buckets, for the Python side of DDP and for the processes competing for the same cores, so on this machine the measurement is above the prediction by a factor that is the **overhead** that the formula leaves out. The efficiency measured was 0.54 with 2 processes and 0.55 with 4, against 0.87 and 0.68 predicted.

   **And the honest result: there is no real speed-up here.** One process on the whole batch takes 9.4 ms; two processes take 37.2 ms and four 28.4 ms per step. All the "GPUs" of this experiment are the **same CPU**: the processes share the same cores (each got an equal share of the threads) and the same memory, so the total compute does not grow, while the all-reduce is extra work. On real separate GPUs each process has its own compute, and the speed-up approaches N when the communication is small, as in the next step.

7. **Step 2.7. Speed-up against the number of GPUs, for three link speeds.** With your `dp_cost` function we can ask the question of the lesson: how far does data parallelism scale? Take the 7-billion-parameter example (14 GB of gradients), a compute time of 1 s per step on each GPU (assumed), and N from 1 to 64. We draw the speed-up for three **assumed** link speeds, 25, 100 and 400 GB/s, next to the ideal line, and for comparison a fourth curve with the bandwidth that **you measured** on this machine in Step 2.5. Because the traffic 2(N − 1)/N × S approaches 2 S as N grows, the communication time stops growing, so the efficiency levels off instead of falling to zero, as in Amdahl's law for a part of a program that does not get faster.

```python
p2_Ns = np.arange(1, 65)
p2_links = {"25 GB/s (assumed, a slow link)": 25e9, "100 GB/s (assumed)": 100e9, "400 GB/s (assumed, a fast link)": 400e9}
if p2_bw: p2_links["measured on this machine (gloo, N = 2)"] = p2_bw[2]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(p2_Ns, p2_Ns, "k--", label="ideal: N")
p2_sp = {}
for name, bw in p2_links.items():
    p2_sp[name] = np.array([dp_cost(1.0, 14e9, int(n), bw)["speedup"] for n in p2_Ns])
    ax.plot(p2_Ns, p2_sp[name], label=name)
ax.set_xlabel("number of GPUs N"); ax.set_ylabel("speed-up over 1 GPU"); ax.set_title("data parallelism: 14 GB of gradients, 1 s of compute per step")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
for name, v in p2_sp.items(): print(f"{name:42s}: speed-up at N = 8: {v[7]:5.2f}, at N = 64: {v[63]:5.2f}, efficiency at N = 64: {v[63] / 64:.2f}")
vals = list(p2_sp.values())
assert all(v[63] < 64 for v in vals) and np.all(p2_sp["25 GB/s (assumed, a slow link)"] <= p2_sp["100 GB/s (assumed)"] + 1e-12) and np.all(p2_sp["100 GB/s (assumed)"] <= p2_sp["400 GB/s (assumed, a fast link)"] + 1e-12)
```

   The block prints (your numbers must match; timings depend on your computer):

```text
25 GB/s (assumed, a slow link)            : speed-up at N = 8:  4.04, at N = 64: 30.44, efficiency at N = 64: 0.48
100 GB/s (assumed)                        : speed-up at N = 8:  6.43, at N = 64: 50.17, efficiency at N = 64: 0.78
400 GB/s (assumed, a fast link)           : speed-up at N = 8:  7.54, at N = 64: 59.87, efficiency at N = 64: 0.94
measured on this machine (gloo, N = 2)    : speed-up at N = 8:  0.55, at N = 64:  3.94, efficiency at N = 64: 0.06
```

   With the assumed 400 GB/s link, 64 GPUs give a speed-up of **59.9** (efficiency 0.94); with 100 GB/s, **50.2**; with 25 GB/s, only **30.4**. At N = 8 the three links give 7.5, 6.4 and 4.0. The curves bend away from the ideal line and the slower the link the earlier they bend. The speed of the link, not the number of GPUs, sets the limit; and with the bandwidth measured on this machine the 14 GB of gradients would take far longer to average than to compute, so the speed-up at N = 8 would be only 0.55: a lesson in why a laptop's processes are not a cluster. Data parallelism is the right first tool when **the model fits on one GPU and you want speed**; when it does not fit, Part 3 splits the model itself.

   *Check your numbers against Step 2.6:* the speed-up formula uses the same `dp_cost` function that you wrote in Exercise 2.2. The link speeds of 25, 100 and 400 GB/s are assumptions chosen for the example, not specifications of any product.

### Part 3. Tensor and pipeline parallelism (30 min)

*Lesson sections: "Tensor (model) parallelism" and "Pipeline parallelism".* Back to the warehouse. Part 2 copied the whole warehouse four times and shared the orders. That fails when a single crate is too big for one warehouse, or when the whole stock does not fit in one. There are two ways out. You can **cut the crate** and let each worker carry a piece, so that the workers must meet and add up their pieces after every step: that is **tensor parallelism**. Or you can build an **assembly line**: the first warehouse does the first layers, the second warehouse the next ones, and the goods flow from one to the next: that is **pipeline parallelism**. In this part you build both by hand, on one computer, with "devices" that are only pieces of a matrix or groups of layers, and you check with numbers that the cut and the line give *exactly* the same answer as the unsplit model. Then you measure what each costs: a sum after every layer for the cut, and an idle **bubble** at the start and end for the line. ==Splitting the work never changes the answer; it changes what has to travel and who waits.== Nothing here needs a GPU: the arithmetic is the same on a real set of GPUs, but the time of the communication is not, and we say clearly where a number is assumed.

**Imagine first.** (1) In one feed-forward block of a Transformer that is cut over several devices the Megatron way (first matrix by columns, second by rows), how many **all-reduces** (sums across devices) does the forward pass need? (2) With the block cut over 4 devices, what share of the weights does one device hold? (3) A pipeline with 4 stages and only 4 micro-batches: what share of the time is the average stage idle? (4) The same pipeline with 16 micro-batches? Write your guesses in the block below (replace each `None`) and compare in Step 3.9.

```python
# Imagine first: splitting the work
my_guess_allreduce = None   # all-reduces (sums across devices) in the forward pass of ONE Megatron-style feed-forward block, however many devices (a whole number)
my_guess_memory = None      # the weights of one device when a block is split over 4 devices, as a share of the unsplit weights (a number from 0 to 1)
my_guess_bubble4 = None     # idle share of a pipeline with 4 stages and 4 micro-batches (a number from 0 to 1)
my_guess_bubble16 = None    # idle share of a pipeline with 4 stages and 16 micro-batches (a number from 0 to 1)
```

```python
# Helpers that this part needs (re-declared here so that Part 3 runs on its own).
if "RESULTS" not in globals(): RESULTS = {}
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
```

1. **Step 3.1. Cut a matrix product by columns.** A layer computes Y = X·W + b, where X holds the activations (one row per token), W is the weight matrix with d_in rows and d_out columns, and b is a bias. **Column split:** cut W into n blocks of columns, give block i to device i (with the matching piece of b), and let every device multiply the *whole* X by its block. Each device then owns a slice of the output columns, and gluing the slices side by side (a **concatenation**, not a sum) gives Y. Because the columns of the output do not depend on each other, nothing has to be added up. Run the cell: it cuts a 64 × 96 matrix over 2 and over 4 simulated devices and compares with the unsplit result.

   ```python
   # One matrix product Y = X @ W + b, cut by columns across n simulated devices (no GPU needed: a "device" is just a piece of the matrix).
   torch.manual_seed(SEED)
   p3_batch, p3_din, p3_dout = 5, 64, 96
   p3_x = torch.randn(p3_batch, p3_din)
   p3_W = torch.randn(p3_din, p3_dout) / 8
   p3_b = torch.randn(p3_dout)
   p3_full = p3_x @ p3_W + p3_b                      # the unsplit result: the reference

   def p3_col_split(x, W, b, n):
       """Column split: each device holds a block of COLUMNS of W (and the matching part of b) and computes its own slice of the output."""
       Ws = torch.chunk(W, n, dim=1); bs = torch.chunk(b, n, dim=0)
       parts = [x @ w + bb for w, bb in zip(Ws, bs)]
       return torch.cat(parts, dim=-1), parts

   p3_col_err = {}
   for n in (2, 4):
       y, parts = p3_col_split(p3_x, p3_W, p3_b, n)
       p3_col_err[n] = (y - p3_full).abs().max().item()
       shard = torch.chunk(p3_W, n, dim=1)[0]
       print(f"{n} devices: W shard {tuple(shard.shape)} out of {tuple(p3_W.shape)}, each device's output slice {tuple(parts[0].shape)}, "
             f"max |split - full| = {p3_col_err[n]:.1e}")
   print("every device needed the WHOLE input x:", tuple(p3_x.shape), "| the pieces are glued together (concatenated), nothing is added up")
   assert all(e < 1e-5 for e in p3_col_err.values())
   ```

   The block prints (your numbers must match):

   ```text
   2 devices: W shard (64, 48) out of (64, 96), each device's output slice (5, 48), max |split - full| = 0.0e+00
   4 devices: W shard (64, 24) out of (64, 96), each device's output slice (5, 24), max |split - full| = 7.2e-07
   every device needed the WHOLE input x: (5, 64) | the pieces are glued together (concatenated), nothing is added up
   ```

   The split result is the same as the full result: the largest difference is **7e-07** (for 2 devices 0e+00, for 4 devices 7e-07). A column split computes every output number with the same additions in the same order as the unsplit product, so it is often *exactly* equal; a row split (next) adds the pieces in a different order, and in float32 that changes the last digits slightly (Part 4 of this lab shows why). Notice what each device needed: the **whole input** X (so X must be available on every device), and only a **block of the weights**.

   *Exercise 3.1:* The row split. Your task: write `row_split_forward(x, W, b, n)` for the other way of cutting. **Row split:** cut W into n blocks of **rows** and cut X into n blocks of **columns** to match (the pieces of the input that meet those rows). Device i multiplies its piece of X by its block of W and gets a **partial result** of the *full* output shape. Those partial results are not slices of the answer: they must be **added**. Return `(y, parts)`, where `parts` is the list of partial results and `y` is their sum plus the bias, with the bias added **once** (not on every device). Why it matters: the sum is the communication of tensor parallelism, an all-reduce on real GPUs, and you will now count how often it happens.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   def row_split_forward(x, W, b, n):
       # your code here. Return (y, parts): y is the sum of the partial results plus the bias (added ONCE), parts is the list of the n partial results.
       return None, []
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if row_split_forward(p3_x, p3_W, p3_b, 2)[0] is None:
       print("Write row_split_forward above, then run this cell again.")
   else:
       for n in (2, 4, 8):
           y, parts = row_split_forward(p3_x, p3_W, p3_b, n)
           check(f"{n} devices: the sum of the partial results equals the full output", y.shape == p3_full.shape and (y - p3_full).abs().max().item() < 1e-5,
                 "Cut W into n blocks of ROWS (torch.chunk(W, n, dim=0)) and x into n blocks of COLUMNS (dim=-1); add the n products; add b once at the end.")
       y, parts = row_split_forward(p3_x, p3_W, p3_b, 4)
       check("one partial result alone is NOT the answer", len(parts) == 4 and (parts[0] - p3_full).abs().max().item() > 0.5, "Each part has the full output shape but only a share of the sum.")
       RESULTS["p3_row_split"] = True
   ```

   A possible answer, and what it means:

   ```python
   def row_split_forward(x, W, b, n):
       Ws = torch.chunk(W, n, dim=0)             # blocks of ROWS of W
       xs = torch.chunk(x, n, dim=-1)            # the matching blocks of COLUMNS of x
       parts = [xi @ w for xi, w in zip(xs, Ws)]  # each device: a partial result of the FULL output shape
       return sum(parts) + b, parts              # the sum (an all-reduce on real GPUs), then the bias once
   ```

   The cut is along the shared dimension of the product: X·W is a sum over the d_in index, so cutting that index into blocks and adding the block products gives the same sum. That is why the row split ends in an addition while the column split ends in a concatenation. The mistake to avoid is the bias: if every device added b, the sum would contain it n times (the next step shows the error). On real GPUs the line `sum(parts)` is a single **all-reduce**: every device ends up with the full sum.

2. **Step 3.2. What the row split gets wrong if you are careless.** Here is our version of the row split, and the two classic mistakes: adding the bias on every device, and gluing the partial results together instead of adding them. Seeing the errors makes it clear why the sum is not optional.

   ```python
   # Row split with the reference function, and the two classic mistakes.
   def p3_row_split(x, W, b, n):
       Ws = torch.chunk(W, n, dim=0); xs = torch.chunk(x, n, dim=-1)
       parts = [xi @ w for xi, w in zip(xs, Ws)]
       return sum(parts) + b, parts

   y, parts = p3_row_split(p3_x, p3_W, p3_b, 4)
   p3_row_err = (y - p3_full).abs().max().item()
   p3_part_err = [(p - p3_full).abs().max().item() for p in parts]
   p3_bias_err = ((sum(parts) + 4 * p3_b) - p3_full).abs().max().item()     # mistake: every device adds the bias
   print(f"row split over 4 devices: max |sum - full| = {p3_row_err:.1e}")
   print("a single partial result differs from the answer by up to", [round(e, 2) for e in p3_part_err])
   print(f"mistake 1, the bias added on all 4 devices: error {p3_bias_err:.2f} (it is counted 4 times, so 3 times too many)")
   print("mistake 2, gluing the partial results together instead of adding them: shape", tuple(torch.cat(parts, dim=-1).shape), "instead of", tuple(p3_full.shape))
   assert p3_row_err < 1e-5 and p3_bias_err > 1
   ```

   The block prints (your numbers must match):

   ```text
   row split over 4 devices: max |sum - full| = 9.5e-07
   a single partial result differs from the answer by up to [3.85, 3.5, 4.06, 4.08]
   mistake 1, the bias added on all 4 devices: error 7.94 (it is counted 4 times, so 3 times too many)
   mistake 2, gluing the partial results together instead of adding them: shape (5, 384) instead of (5, 96)
   ```

   The row split is exact too (1e-06). One partial result alone is far from the answer (off by up to 4.1), because each device saw only a part of the sum. Adding the bias on all 4 devices makes the error **7.94**, and gluing the pieces gives a matrix of the wrong shape. The rule: **column split, then concatenate (or just keep the slices where they are); row split, then sum.**

3. **Step 3.3. The Megatron-style feed-forward block: one sum per block.** A Transformer feed-forward block is Y = GELU(X·W1 + b1)·W2 + b2, with W1 of size d × 4d and W2 of size 4d × d. Megatron-LM (Shoeybi et al. 2019) cuts the first matrix by **columns** and the second by **rows**, and that choice is the clever part: the columns of the first layer's output are exactly the rows that the second matrix needs, so **each device runs both layers on its own piece without talking to the others**, and only the final partial results are added. The activation function acts on each element separately, so it can run on every shard. In the cell the class `P3Comm` plays the network: every call of `all_reduce` is counted and its bytes are added up. We cut the block over 1, 2, 4 and 8 devices, and then try the alternative (first matrix by rows) to see why it needs a second sum.

   ```python
   # A Megatron-style feed-forward block: first matrix by columns, activation on each shard, second matrix by rows, ONE sum at the end.
   class P3Comm:
       """Counts the all-reduces (sums across devices) and the bytes of the tensors that are summed."""
       def __init__(self): self.count = 0; self.bytes = 0
       def all_reduce(self, parts):
           self.count += 1; self.bytes += parts[0].numel() * parts[0].element_size()
           return sum(parts)

   torch.manual_seed(SEED)
   p3_d, p3_h = 64, 256                                     # width d and hidden size h = 4d, as in a real Transformer
   p3_W1 = torch.randn(p3_d, p3_h) / 8; p3_b1 = torch.randn(p3_h) * 0.1
   p3_W2 = torch.randn(p3_h, p3_d) / 16; p3_b2 = torch.randn(p3_d) * 0.1
   p3_xf = torch.randn(4, 10, p3_d)                         # batch 4, 10 tokens
   def p3_ffn_full(x): return F.gelu(x @ p3_W1 + p3_b1) @ p3_W2 + p3_b2

   def p3_ffn_tp(x, n, comm):
       W1s = torch.chunk(p3_W1, n, dim=1); b1s = torch.chunk(p3_b1, n, dim=0); W2s = torch.chunk(p3_W2, n, dim=0)
       parts = [F.gelu(x @ w1 + bb1) @ w2 for w1, bb1, w2 in zip(W1s, b1s, W2s)]   # every device works alone: no communication in here
       return comm.all_reduce(parts) + p3_b2                                          # the only communication of the block

   p3_ref = p3_ffn_full(p3_xf)
   p3_ffn_rows = []
   for n in (1, 2, 4, 8):
       comm = P3Comm()
       err = (p3_ffn_tp(p3_xf, n, comm) - p3_ref).abs().max().item()
       p3_ffn_rows.append((n, comm.count, comm.bytes, err))
       print(f"{n} devices: all-reduces per block = {comm.count}, bytes summed = {comm.bytes}, max error = {err:.1e}")
   assert all(r[1] == 1 and r[3] < 1e-5 for r in p3_ffn_rows)

   # What if the first matrix were split by ROWS? The partial results must be added BEFORE the activation, because gelu(a + b) is not gelu(a) + gelu(b).
   n = 4
   pre = [xi @ w for xi, w in zip(torch.chunk(p3_xf, n, dim=-1), torch.chunk(p3_W1, n, dim=0))]
   wrong = sum(F.gelu(p + p3_b1 / n) for p in pre)                        # activation on each partial result, then add: wrong
   right = F.gelu(sum(pre) + p3_b1)                                       # add first (an all-reduce), then the activation
   p3_gelu_wrong = (wrong - F.gelu(p3_xf @ p3_W1 + p3_b1)).abs().max().item()
   p3_gelu_right = (right - F.gelu(p3_xf @ p3_W1 + p3_b1)).abs().max().item()
   print(f"first matrix split by rows: activation before the sum: error {p3_gelu_wrong:.2f}; sum first, then activation: error {p3_gelu_right:.1e} (but that costs a second all-reduce)")
   assert p3_gelu_wrong > 0.05 and p3_gelu_right < 1e-5
   ```

   The block prints (your numbers must match):

   ```text
   1 devices: all-reduces per block = 1, bytes summed = 10240, max error = 0.0e+00
   2 devices: all-reduces per block = 1, bytes summed = 10240, max error = 1.4e-06
   4 devices: all-reduces per block = 1, bytes summed = 10240, max error = 1.4e-06
   8 devices: all-reduces per block = 1, bytes summed = 10240, max error = 1.4e-06
   first matrix split by rows: activation before the sum: error 1.46; sum first, then activation: error 1.2e-06 (but that costs a second all-reduce)
   ```

   For every number of devices the block needs **1 all-reduce** and the answer is right to 1e-06. The tensor that is summed has 2560 numbers (10240 bytes): batch 4 × 10 tokens × d = 64 values of 4 bytes. The alternative is worse in two ways. If the first matrix were cut by rows, its partial results would have to be added **before** the GELU, because GELU of a sum is not the sum of the GELUs: doing it the wrong way gives an error of **1.46**, doing it the right way works (1e-06) but costs a **second** all-reduce. Cutting columns first and rows second is the cheapest order: one sum per block.

   *Exercise 3.2:* How big is one all-reduce?. Your task: write `allreduce_bytes(batch, tokens, d, bytes_per_value)`, the size in bytes of the activation tensor that one all-reduce adds up (batch × tokens × d values). Then compute the size in megabytes for the lesson's example (batch 8, 2,048 tokens, d = 4,096, 16-bit values) and the number of all-reduces in one forward pass of a 32-layer model that needs 2 per layer. Why it matters: this tensor travels at every layer, so the speed of the link decides whether tensor parallelism pays off.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   def allreduce_bytes(batch, tokens, d, bytes_per_value):
       # your code here: the size in bytes of the tensor that one all-reduce of the activations sums
       return None

   p3_lesson_mb = None     # the size in megabytes (1 MB = 1,000,000 bytes) for batch 8, 2,048 tokens, d = 4,096, 16-bit values (2 bytes), rounded to a whole number
   p3_per_forward = None   # the number of all-reduces in one forward pass of a 32-layer model, with 2 all-reduces per layer (one for attention, one for the feed-forward block)
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if allreduce_bytes(1, 1, 1, 1) is None or p3_lesson_mb is None or p3_per_forward is None:
       print("Write allreduce_bytes and fill in the two variables above, then run this cell again.")
   else:
       check("the demo tensor of the FFN above", allreduce_bytes(4, 10, p3_d, 4) == p3_ffn_rows[1][2], "batch x tokens x d x bytes per value: 4 x 10 x 64 x 4.")
       check("the lesson example is 134 MB", p3_lesson_mb == 134, "8 x 2048 x 4096 x 2 bytes, divided by 1,000,000.")
       check("64 all-reduces per forward pass", p3_per_forward == 64, "2 per layer, 32 layers.")
       RESULTS["p3_allreduce_mb"] = p3_lesson_mb
   ```

   A possible answer, and what it means:

   ```python
   def allreduce_bytes(batch, tokens, d, bytes_per_value):
       return batch * tokens * d * bytes_per_value

   p3_lesson_mb = round(allreduce_bytes(8, 2048, 4096, 2) / 1e6)      # 134
   p3_per_forward = 2 * 32                                            # 64
   ```

   The size is batch × tokens × d × bytes per value: 8 × 2,048 × 4,096 × 2 = 134,217,728 bytes, about **134 MB**, and it does not depend on the number of devices (it is the size of the activations, not of the weights). A 32-layer model has 2 sums per layer (one after attention and one after the feed-forward block), so **64 all-reduces** in every forward pass. Step 3.5 puts a time on them.

4. **Step 3.4. Attention heads across devices.** Attention is already built from independent **heads**: each head has its own slice of the query, key and value matrices and attends on its own. So the natural cut gives each device a few heads: the columns of Wq, Wk and Wv that belong to those heads (a column split), and the matching rows of the output matrix Wo (a row split). Each device computes its heads and its partial result of the output projection; one sum finishes the layer. The cell writes a causal multi-head attention with 8 heads from scratch and runs it on 1, 2, 4 and 8 devices.

   ```python
   # Attention heads split across devices: each device owns some heads (columns of Wq, Wk, Wv) and the matching rows of Wo; one sum at the end.
   torch.manual_seed(SEED)
   p3_H, p3_T, p3_B = 8, 12, 2
   p3_dh = p3_d // p3_H
   p3_Wq, p3_Wk, p3_Wv, p3_Wo = [torch.randn(p3_d, p3_d) / 8 for _ in range(4)]
   p3_xa = torch.randn(p3_B, p3_T, p3_d)
   p3_mask = torch.triu(torch.ones(p3_T, p3_T, dtype=torch.bool), diagonal=1)      # causal mask: a token never looks ahead

   def p3_heads(x, Wq, Wk, Wv, nh):
       """Attention of nh heads (the matrices hold exactly those heads); returns the concatenated head outputs."""
       B, T, _ = x.shape
       q, k, v = [(x @ W).view(B, T, nh, p3_dh).transpose(1, 2) for W in (Wq, Wk, Wv)]
       att = (q @ k.transpose(-1, -2) / math.sqrt(p3_dh)).masked_fill(p3_mask, float("-inf")).softmax(-1)
       return (att @ v).transpose(1, 2).reshape(B, T, nh * p3_dh)

   def p3_attn_full(x): return p3_heads(x, p3_Wq, p3_Wk, p3_Wv, p3_H) @ p3_Wo

   def p3_attn_tp(x, n, comm):
       cols = [torch.chunk(W, n, dim=1) for W in (p3_Wq, p3_Wk, p3_Wv)]              # each device gets H/n heads
       rows = torch.chunk(p3_Wo, n, dim=0)
       parts = [p3_heads(x, cols[0][i], cols[1][i], cols[2][i], p3_H // n) @ rows[i] for i in range(n)]
       return comm.all_reduce(parts)

   p3_aref = p3_attn_full(p3_xa)
   p3_attn_rows = []
   for n in (1, 2, 4, 8):
       comm = P3Comm()
       err = (p3_attn_tp(p3_xa, n, comm) - p3_aref).abs().max().item()
       p3_attn_rows.append((n, p3_H // n, comm.count, err))
       print(f"{n} devices: {p3_H // n} heads on each, all-reduces = {comm.count}, max error = {err:.1e}")
   assert all(r[2] == 1 and r[3] < 1e-5 for r in p3_attn_rows)
   print("a whole Transformer layer (attention + feed-forward) therefore needs 2 all-reduces in the forward pass")
   ```

   The block prints (your numbers must match):

   ```text
   1 devices: 8 heads on each, all-reduces = 1, max error = 0.0e+00
   2 devices: 4 heads on each, all-reduces = 1, max error = 3.6e-07
   4 devices: 2 heads on each, all-reduces = 1, max error = 4.8e-07
   8 devices: 1 heads on each, all-reduces = 1, max error = 6.0e-07
   a whole Transformer layer (attention + feed-forward) therefore needs 2 all-reduces in the forward pass
   ```

   Every split gives the same output as the unsplit attention (largest error 6e-07) with **one all-reduce**, so a whole layer, attention plus feed-forward, costs 2 sums in the forward pass. The number of devices cannot exceed the number of heads (8 here) with this cut: that is a real limit of the method.

5. **Step 3.5. Memory per device, and what travels.** The benefit is memory. The weights of a block are cut into n parts, so each device holds about 1/n of them. The cell **measures** it: it takes the shards of device 0 (attention and feed-forward) and adds up the bytes of the tensors. Then it puts numbers on the traffic for a realistic size, with a ring all-reduce that moves 2(N − 1)/N × S bytes per device (the formula of Part 2). The two link speeds are **assumptions** for the example, not specifications of any product.

   ```python
   # What one device holds, measured from the shards, and what travels between devices.
   def p3_device0_bytes(n):
       shards = [torch.chunk(p3_W1, n, dim=1)[0], torch.chunk(p3_b1, n, dim=0)[0], torch.chunk(p3_W2, n, dim=0)[0], p3_b2]   # b2 is added once, after the sum: kept whole
       shards += [torch.chunk(W, n, dim=1)[0] for W in (p3_Wq, p3_Wk, p3_Wv)]
       shards.append(torch.chunk(p3_Wo, n, dim=0)[0])
       return sum(t.numel() * t.element_size() for t in shards)

   p3_mem = {n: p3_device0_bytes(n) for n in (1, 2, 4, 8)}
   print(f"{'devices':>8} {'weights on device 0':>20} {'share of unsplit':>17} {'1/n':>6}")
   for n, b in p3_mem.items():
       print(f"{n:8d} {human_bytes(b):>20} {b / p3_mem[1]:17.3f} {1 / n:6.3f}")
   assert all(abs(p3_mem[n] / p3_mem[1] - 1 / n) < 0.01 for n in p3_mem)

   # the traffic for a realistic size (the numbers are the lesson's example; the link speeds are ASSUMED)
   p3_S = 8 * 2048 * 4096 * 2                                 # bytes of one all-reduce: batch 8, 2,048 tokens, d = 4,096, 2 bytes
   p3_N = 8
   p3_traffic = 2 * (p3_N - 1) / p3_N * p3_S                  # bytes each device sends and receives in a ring all-reduce
   print(f"\none all-reduce: {p3_S / 1e6:.0f} MB of tensor, {p3_traffic / 1e6:.0f} MB of traffic per device (8 devices); 64 of them per forward pass of 32 layers")
   p3_link = {}
   for name, gbps in (("a fast link inside one machine (assumed 100 GB/s)", 100), ("an ordinary network between machines (assumed 10 GB/s)", 10)):
       p3_link[gbps] = 64 * p3_traffic / (gbps * 1e9) * 1000
       print(f"  {name}: {p3_link[gbps]:.0f} ms of communication per forward pass")
   ```

   The block prints (your numbers must match):

   ```text
    devices  weights on device 0  share of unsplit    1/n
          1            197.89 KB             1.000  1.000
          2             99.07 KB             0.501  0.500
          4             49.66 KB             0.251  0.250
          8             24.96 KB             0.126  0.125

   one all-reduce: 134 MB of tensor, 235 MB of traffic per device (8 devices); 64 of them per forward pass of 32 layers
     a fast link inside one machine (assumed 100 GB/s): 150 ms of communication per forward pass
     an ordinary network between machines (assumed 10 GB/s): 1503 ms of communication per forward pass
   ```

   Device 0 holds **0.501** of the weights with 2 devices, **0.251** with 4 and **0.126** with 8: slightly above 1/n, because the bias b2 is kept whole on every device. The price is the traffic: a batch of 8 × 2,048 tokens at d = 4,096 in 16 bits is one all-reduce of 134 MB, and with 8 devices each device moves 235 MB. For 64 of them, an assumed 100 GB/s link spends about **150 ms** of every forward pass on communication, an assumed 10 GB/s network between machines about **1503 ms**. That is why tensor parallelism is used **inside one machine**, where the links are fast, and why the next method is the choice when the links are slow.

6. **Step 3.6. Pipeline parallelism: cut by layers.** Now cut the model the other way: by **layers**. A model of 8 layers (each a 1,024 × 1,024 linear layer and a GELU) is divided into p **stages**, a list of `nn.Sequential`; stage 1 does the first layers, stage 2 the next ones. If a whole batch went through stage by stage, p − 1 of the p devices would wait all the time. So the batch is cut into m **micro-batches** that follow each other: as soon as stage 1 passes micro-batch 1 on, it starts micro-batch 2. The cell runs the stages one after the other on a single computer (the *order* of the arithmetic is what a pipeline does; the *overlap* in time is what the next step simulates) and checks the output against the unsplit model. Then it **measures the time of each stage** for one micro-batch, once for an equal split (2 layers each) and once for an uneven one (1, 1, 2 and 4 layers).

   ```python
   # Pipeline parallelism: 8 layers, cut into p stages; the batch is cut into m micro-batches that flow through the stages.
   torch.manual_seed(SEED)
   p3_width = 1024
   p3_layers = [nn.Sequential(nn.Linear(p3_width, p3_width), nn.GELU()) for _ in range(8)]
   p3_model = nn.Sequential(*p3_layers).eval()
   p3_xp = torch.randn(64, p3_width)

   def p3_make_stages(sizes):
       """sizes: how many layers each stage gets, for example [2, 2, 2, 2]."""
       assert sum(sizes) == len(p3_layers)
       out, i = [], 0
       for s in sizes:
           out.append(nn.Sequential(*p3_layers[i:i + s]).eval()); i += s
       return out

   def p3_run_pipeline(x, stages, m):
       outs = []
       for mb in torch.chunk(x, m, dim=0):          # micro-batch after micro-batch ...
           for st in stages: mb = st(mb)            # ... through stage after stage (on real GPUs the stages work at the same time)
           outs.append(mb)
       return torch.cat(outs)

   p3_pipe_err = {}
   with torch.no_grad():
       p3_ref_out = p3_model(p3_xp)
       for p, m in ((1, 1), (2, 4), (4, 8), (8, 16)):
           p3_pipe_err[(p, m)] = (p3_run_pipeline(p3_xp, p3_make_stages([8 // p] * p), m) - p3_ref_out).abs().max().item()
           print(f"p = {p} stages, m = {m:2d} micro-batches of {64 // m}: max |pipeline - unsplit model| = {p3_pipe_err[(p, m)]:.1e}")
   assert all(e < 1e-5 for e in p3_pipe_err.values())

   # measure the time of each stage for one micro-batch of 16 rows
   def p3_stage_times(sizes, rows=16):
       mb = torch.randn(rows, p3_width); ts = []
       with torch.no_grad():
           for st in p3_make_stages(sizes):
               ts.append(time_ms(lambda: st(mb), repeat=20 if not QUICK else 5, warmup=3)); mb = st(mb)
       return ts
   p3_t_equal = p3_stage_times([2, 2, 2, 2])
   p3_t_uneven = p3_stage_times([1, 1, 2, 4])
   print("\nmeasured time per micro-batch (ms), 2 layers per stage :", [round(t, 3) for t in p3_t_equal])
   print("measured time per micro-batch (ms), layers 1, 1, 2, 4  :", [round(t, 3) for t in p3_t_uneven])
   ```

   The block prints (your numbers must match):

   ```text
   p = 1 stages, m =  1 micro-batches of 64: max |pipeline - unsplit model| = 0.0e+00
   p = 2 stages, m =  4 micro-batches of 16: max |pipeline - unsplit model| = 9.3e-09
   p = 4 stages, m =  8 micro-batches of 8: max |pipeline - unsplit model| = 1.1e-08
   p = 8 stages, m = 16 micro-batches of 4: max |pipeline - unsplit model| = 9.3e-09

   measured time per micro-batch (ms), 2 layers per stage : [0.53, 0.618, 0.558, 0.56]
   measured time per micro-batch (ms), layers 1, 1, 2, 4  : [0.239, 0.22, 0.549, 1.45]
   ```

   The pipeline gives the same output as the unsplit model for every p and m (largest difference 1e-08), so cutting a batch into micro-batches loses nothing in the forward pass for a model without batch statistics (layers such as batch normalisation, which look at the whole batch, need extra care). On the machine that ran this lab one micro-batch takes about **0.57 ms** per equal stage, and for the uneven split the slowest stage takes 1.45 ms against 0.22 ms for the fastest: its four layers cost about four times as much as one layer.

7. **Step 3.7. The GPipe schedule, the bubble, and the slowest stage.** Now we let the stages work **at the same time** and compute when everything finishes. The rule of the GPipe schedule: stage i can start micro-batch j when two things are true: it has finished micro-batch j − 1, and stage i − 1 has finished micro-batch j. `gpipe_schedule(stage_times, m)` follows this rule for given stage times, returns every (stage, micro-batch, start, end) and the **makespan** (the finishing time of the last micro-batch). The **idle share** is 1 − (busy time) ÷ (p × makespan): the part of the devices' time that is wasted, the **bubble**. The cell draws a Gantt chart for our two measured splits, checks the lesson's formula (p − 1) ÷ (m + p − 1) against the simulation for 18 combinations, and shows what unequal stages do. We simulate the forward pass; a training step runs the backward pass the same way, and the bubble has the same share.

   ```python
   # The GPipe schedule: stage i can start micro-batch j when it has finished micro-batch j - 1 AND stage i - 1 has finished micro-batch j.
   def gpipe_schedule(stage_times, m):
       """Return (events, makespan). events: (stage, micro_batch, start, end)."""
       p = len(stage_times); free = [0.0] * p; done = {}; events = []
       for j in range(m):
           for i in range(p):
               start = max(free[i], done.get((i - 1, j), 0.0))
               end = start + stage_times[i]
               done[(i, j)] = end; free[i] = end; events.append((i, j, start, end))
       return events, max(free)

   def p3_idle(stage_times, m):
       events, span = gpipe_schedule(stage_times, m)
       return 1 - sum(e[3] - e[2] for e in events) / (len(stage_times) * span)

   def p3_gantt(ax, stage_times, m, title):
       events, span = gpipe_schedule(stage_times, m)
       cmap = plt.get_cmap("tab10")
       for i, j, s, e in events:
           ax.barh(i, e - s, left=s, color=cmap(j % 10), edgecolor="white")
           if m <= 8: ax.text((s + e) / 2, i, str(j + 1), ha="center", va="center", color="white", fontsize=8)
       ax.set_yticks(range(len(stage_times))); ax.set_yticklabels([f"stage {i + 1}" for i in range(len(stage_times))]); ax.invert_yaxis()
       ax.set_xlabel("time (ms)"); ax.set_title(f"{title}\nmakespan {span:.2f} ms, idle {p3_idle(stage_times, m):.0%}", fontsize=9)

   p3_m = 4
   fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
   p3_gantt(axes[0], p3_t_equal, p3_m, "measured times, 2 layers per stage")
   p3_gantt(axes[1], p3_t_uneven, p3_m, "measured times, layers 1, 1, 2, 4")
   plt.tight_layout(); plt.show()

   # verify the bubble formula with equal stage times
   p3_formula_err = 0
   for p in (2, 4, 8):
       for m in (1, 2, 4, 8, 16, 32):
           sim = p3_idle([1.0] * p, m); formula = (p - 1) / (m + p - 1)
           p3_formula_err = max(p3_formula_err, abs(sim - formula))
   assert p3_formula_err < 1e-12
   print(f"simulated idle share vs (p - 1) / (m + p - 1), 18 combinations of p and m: largest difference {p3_formula_err:.1e}")
   print(f"{'p':>3} " + " ".join(f"m={m:<5d}" for m in (1, 2, 4, 8, 16, 32)))
   for p in (2, 4, 8):
       print(f"{p:3d} " + " ".join(f"{(p - 1) / (m + p - 1):7.1%}" for m in (1, 2, 4, 8, 16, 32)))

   # the slowest stage sets the pace
   p3_unequal = [1.0, 1.0, 1.0, 4.0]
   _, p3_span_unequal = gpipe_schedule(p3_unequal, 8)
   p3_span_formula = sum(p3_unequal) + (8 - 1) * max(p3_unequal)
   print(f"\nstage times {p3_unequal}, m = 8: makespan {p3_span_unequal}, formula sum + (m - 1) x slowest = {p3_span_formula}, idle share {p3_idle(p3_unequal, 8):.1%} "
         f"(equal stages with the same total work: {p3_idle([1.75] * 4, 8):.1%})")
   assert p3_span_unequal == p3_span_formula

   # speed-up of the measured pipeline against one device that runs the micro-batches one after another
   p3_seq = lambda ts, m: m * sum(ts)
   p3_speed = {m: p3_seq(p3_t_equal, m) / gpipe_schedule(p3_t_equal, m)[1] for m in (1, 4, 16, 64)}
   print("speed-up of 4 stages (measured stage times) over one device:", {m: round(v, 2) for m, v in p3_speed.items()}, "(the ideal is 4)")

   fig, ax = plt.subplots(figsize=(5.5, 3.2))
   for p in (2, 4, 8):
       ms = np.arange(1, 65)
       ax.plot(ms, [(p - 1) / (m + p - 1) for m in ms], label=f"p = {p}")
       ax.plot([1, 2, 4, 8, 16, 32], [p3_idle([1.0] * p, m) for m in (1, 2, 4, 8, 16, 32)], "k.")
   ax.set_xlabel("micro-batches m"); ax.set_ylabel("idle share (the bubble)"); ax.set_title("the formula (lines) and the simulation (dots)"); ax.legend(); plt.tight_layout(); plt.show()
   ```

   The block prints (your numbers must match):

   ```text
   simulated idle share vs (p - 1) / (m + p - 1), 18 combinations of p and m: largest difference 5.6e-17
     p m=1     m=2     m=4     m=8     m=16    m=32   
     2   50.0%   33.3%   20.0%   11.1%    5.9%    3.0%
     4   75.0%   60.0%   42.9%   27.3%   15.8%    8.6%
     8   87.5%   77.8%   63.6%   46.7%   30.4%   17.9%

   stage times [1.0, 1.0, 1.0, 4.0], m = 8: makespan 35.0, formula sum + (m - 1) x slowest = 35.0, idle share 60.0% (equal stages with the same total work: 27.3%)
   speed-up of 4 stages (measured stage times) over one device: {1: 1.0, 4: 2.2, 16: 3.14, 64: 3.52} (the ideal is 4)
   ```

   Three things to notice. First, the **formula is exact** for equal stages: the simulation and the formula differ by at most 6e-17 over all 18 combinations; with p = 4 the bubble falls from 75% (one micro-batch: no pipelining at all) to 42.9% (m = 4), 27.3% (m = 8) and 8.6% (m = 32). Second, **the slowest stage sets the pace**: with stage times 1, 1, 1 and 4 the makespan is the sum plus (m − 1) times the slowest stage, 35 time units for m = 8, and the idle share is 60% against 27.3% for four equal stages with the same total work. Balance the stages by their *time*, not by their number of layers. Third, with our measured equal stages the speed-up over one device is 2.20 with 4 micro-batches and 3.52 with 64, against an ideal of 4: more micro-batches keep the devices busy, but each micro-batch is smaller (so each matrix product is less efficient on a real GPU), and every micro-batch in flight keeps its activations in memory.

   *Exercise 3.3:* The bubble as a function. Your task: write `bubble(p, m)`, the idle share of a pipeline with p stages and m micro-batches of equal stage time. Store in `p3_bubble_4_8` its value for p = 4 and m = 8, and in `min_m` the **smallest number of micro-batches** that brings the idle share of a 4-stage pipeline below 10 percent. Why it matters: you choose m, and this tells you how many you need.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   def bubble(p, m):
       # your code here: the idle share of a pipeline with p stages and m micro-batches, equal stage times
       return None

   p3_bubble_4_8 = None    # idle share for p = 4, m = 8, as a number from 0 to 1
   min_m = None            # the smallest number of micro-batches that brings the idle share of a 4-stage pipeline below 10 percent (0.10)
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if bubble(2, 2) is None or p3_bubble_4_8 is None or min_m is None:
       print("Write bubble and fill in the two variables above, then run this cell again.")
   else:
       check("bubble(4, 8) is 3/11", abs(bubble(4, 8) - 3 / 11) < 1e-9 and abs(p3_bubble_4_8 - 3 / 11) < 1e-9, "(p - 1) / (m + p - 1) with p = 4, m = 8.")
       check("bubble matches the simulation", all(abs(bubble(p, m) - p3_idle([1.0] * p, m)) < 1e-9 for p in (2, 3, 5) for m in (1, 3, 9)), "Compare with p3_idle([1.0] * p, m).")
       check("min_m brings it below 10 percent, and one less does not", bubble(4, min_m) < 0.10 <= bubble(4, min_m - 1), "Try m = 20, 25, 30 ... in your function.")
       RESULTS["p3_bubble"] = (round(p3_bubble_4_8, 4), min_m)
   ```

   A possible answer, and what it means:

   ```python
   def bubble(p, m):
       return (p - 1) / (m + p - 1)

   p3_bubble_4_8 = bubble(4, 8)                                   # 3/11 = 0.2727
   min_m = next(m for m in range(1, 1000) if bubble(4, m) < 0.10)   # 28
   ```

   The idle share is (p − 1) ÷ (m + p − 1). For p = 4 and m = 8 it is 3 ÷ 11 = **27.3 %**. Below 10 percent we need 3 ÷ (m + 3) < 0.10, so m + 3 > 30, m > 27: the smallest whole number is **m = 28** (3 ÷ 31 = 9.7 %). Remember what the 28 costs: each micro-batch is 1/28 of the batch, and the batch must be big enough to be cut that finely.

8. **Step 3.8. Which parallelism, and why? The decision table.** Three kinds of parallelism, one question each: *does the model fit on one device? How fast are the links? How deep is the model?* The cell prints eight situations with an answer and a reason. Read the table once, then do the exercise below it (the exercise asks you to decide three of the eight *before* you look at the reasons).

   ```python
   # The decision table. Eight situations; for each, which kind of parallelism, and why.
   P3_SITUATIONS = [
       ("1B-parameter model, fits on one GPU; you have 8 GPUs and want to train faster", "data", "the model fits, so copy it; only the gradients travel, once per step"),
       ("70B-parameter model (140 GB at 16 bits) on one machine with 8 GPUs and fast links", "tensor", "each layer is split over the GPUs; the per-layer all-reduces need fast links, and that is what one machine has"),
       ("a 96-layer model cut over 4 machines that are joined by an ordinary network", "pipeline", "only the activations at the stage borders cross the slow network"),
       ("a model with 400 billion parameters on hundreds of GPUs in many machines", "combination", "tensor inside a machine, pipeline across machines, data on top (often called 3D parallelism)"),
       ("one weight matrix is bigger than the memory of one GPU", "tensor", "only splitting the matrix itself helps: stages and copies still hold whole layers"),
       ("serving a 7B model to many users; the model fits on one GPU", "data", "run replicas, each with a full copy, and share the requests"),
       ("2 GPUs joined by a slow link, the model fits on each, you want throughput", "data", "one gradient exchange per step is far cheaper than an all-reduce in every layer"),
       ("a 30-layer model of 48 GB on four GPUs with 16 GB each, joined by a slow link", "pipeline", "cut by layers, 12 GB each; little traffic, which suits a slow link"),
   ]
   for i, (s, a, r) in enumerate(P3_SITUATIONS, 1):
       print(f"{i}. {s}\n   -> {a}: {r}")
   ```

   The block prints (your numbers must match):

   ```text
   1. 1B-parameter model, fits on one GPU; you have 8 GPUs and want to train faster
      -> data: the model fits, so copy it; only the gradients travel, once per step
   2. 70B-parameter model (140 GB at 16 bits) on one machine with 8 GPUs and fast links
      -> tensor: each layer is split over the GPUs; the per-layer all-reduces need fast links, and that is what one machine has
   3. a 96-layer model cut over 4 machines that are joined by an ordinary network
      -> pipeline: only the activations at the stage borders cross the slow network
   4. a model with 400 billion parameters on hundreds of GPUs in many machines
      -> combination: tensor inside a machine, pipeline across machines, data on top (often called 3D parallelism)
   5. one weight matrix is bigger than the memory of one GPU
      -> tensor: only splitting the matrix itself helps: stages and copies still hold whole layers
   6. serving a 7B model to many users; the model fits on one GPU
      -> data: run replicas, each with a full copy, and share the requests
   7. 2 GPUs joined by a slow link, the model fits on each, you want throughput
      -> data: one gradient exchange per step is far cheaper than an all-reduce in every layer
   8. a 30-layer model of 48 GB on four GPUs with 16 GB each, joined by a slow link
      -> pipeline: cut by layers, 12 GB each; little traffic, which suits a slow link
   ```

   The pattern in the table: **data** parallelism whenever the model fits (it only exchanges gradients once per step, or nothing at all when serving); **tensor** parallelism when a layer or the model is too big for one device and the links are fast (a sum in every layer); **pipeline** parallelism when the model is deep and the links are slow (only the activations at the stage borders travel); and a **combination** when one method is not enough. The largest training runs use all three at once (Narayanan et al. 2021 describe how to combine them).

   *Exercise 3.4:* Decide three situations. Your task: for situations **2**, **3** and **6** of the table, write in `my_choice` one of `'data'`, `'tensor'`, `'pipeline'` or `'combination'`, and write one sentence of reasoning for each in `my_reason`. The check tells you whether your choice agrees with ours. Why it matters: the table is easy to follow and hard to apply; the skill is to ask the three questions yourself.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   # Before you read our answers above in detail: decide situations 2, 3 and 6 from the table. Write 'data', 'tensor', 'pipeline' or 'combination'.
   my_choice = {2: None, 3: None, 6: None}
   my_reason = {2: "", 3: "", 6: ""}      # one sentence each, in your own words
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if None in my_choice.values():
       print("Fill in my_choice for situations 2, 3 and 6, then run this cell again.")
   else:
       for k in (2, 3, 6):
           check(f"situation {k}", my_choice[k] == P3_SITUATIONS[k - 1][1], f"Think about what has to move between devices, and how fast the link is. Our answer has a reason: {P3_SITUATIONS[k - 1][2]}.")
       RESULTS["p3_decisions"] = dict(my_choice)
   ```

   A possible answer, and what it means:

   ```python
   my_choice = {2: "tensor", 3: "pipeline", 6: "data"}
   my_reason = {2: "the weights do not fit on one GPU and the 8 GPUs of one machine have fast links, so split each layer",
                3: "the network is slow, and a pipeline only sends the activations between stages",
                6: "the model fits on one GPU, so replicas give throughput with no communication between them"}
   ```

   Situation 2 (70 B parameters, 140 GB at 16 bits, 8 GPUs with fast links in one machine): **tensor**, because the weights do not fit on one GPU and the links inside a machine are fast. Situation 3 (a 96-layer model over 4 machines on an ordinary network): **pipeline**, because only the activations at the stage borders cross the slow network. Situation 6 (serving a 7 B model that fits on one GPU): **data**, run copies of the model and share the requests, with no communication between copies. Your own sentences should name the same three things: does it fit, how fast is the link, how deep is it.

9. **Step 3.9. Compare with your Imagine guesses.** The next cell prints the true values next to your guesses.

   ```python
   print(f"(1) all-reduces in one Megatron-style FFN block: {p3_ffn_rows[1][1]}   your guess: {my_guess_allreduce}")
   print(f"(2) weights per device over 4 devices: {p3_mem[4] / p3_mem[1]:.3f} of the unsplit   your guess: {my_guess_memory}")
   print(f"(3) idle share p = 4, m = 4: {(4 - 1) / (4 + 4 - 1):.3f}   your guess: {my_guess_bubble4}")
   print(f"(4) idle share p = 4, m = 16: {(4 - 1) / (16 + 4 - 1):.3f}   your guess: {my_guess_bubble16}")
   RESULTS["p3_ffn_allreduces"] = p3_ffn_rows[1][1]
   RESULTS["p3_mem_share_4"] = round(p3_mem[4] / p3_mem[1], 3)
   RESULTS["p3_bubble_4_4"] = round(3 / 7, 3)
   RESULTS["p3_bubble_4_16"] = round(3 / 19, 3)
   RESULTS["p3_speedup_m16"] = round(p3_speed[16], 2)
   ```

   The block prints (your numbers must match):

   ```text
   (1) all-reduces in one Megatron-style FFN block: 1   your guess: None
   (2) weights per device over 4 devices: 0.251 of the unsplit   your guess: None
   (3) idle share p = 4, m = 4: 0.429   your guess: None
   (4) idle share p = 4, m = 16: 0.158   your guess: None
   ```

   Your numbers against ours: **one** all-reduce per block, **0.25** of the weights on each of 4 devices, a bubble of **42.9%** with 4 micro-batches and **15.8%** with 16. Many people guess that the bubble with 4 micro-batches is small, because 4 stages feels like 4 times the work: the formula says that with p stages you need many more than p micro-batches. Remember the cost of each cut: tensor parallelism adds a sum in every layer and needs fast links; pipeline parallelism adds a bubble and needs balanced stages.

   ### What real systems add

   You built the idea, not the system. Real training frameworks add much more: **Megatron-LM** implements the tensor cuts you wrote (with the communication done by the GPU communication library, NCCL) and pipeline schedules that interleave forward and backward micro-batches (the **1F1B** schedule and its variants) to shrink the memory that waiting micro-batches hold; **GPipe** (Huang et al. 2019) is the pipeline schedule you simulated; **DeepSpeed ZeRO** (Rajbhandari et al. 2020) and PyTorch **FSDP** keep data parallelism but cut the optimiser state, gradients and weights across the copies; and for mixture-of-experts models there is **expert parallelism**, which sends each token to the device that holds its expert. Next, Part 4 looks at the other way to fit a model: make every number smaller.

### Part 4. Mixed precision (30 min)

*Lesson section: "Precision: fewer bits per number".* Back to the warehouse. Every number in a model is a box on a shelf, and the size of the box is the number of bits. A 32-bit box is roomy and precise; a 16-bit box is half the size, so you can store twice as many in the same warehouse and move twice as many on each truck. But a smaller box has two limits: it cannot hold very **big** numbers (it overflows) and it cannot tell apart very **close** numbers (it rounds), and worst of all it flattens very **small** numbers to zero. **Mixed precision** is the manager's compromise: keep one precise master copy of the weights in the big boxes, do the heavy arithmetic in small boxes, and rescale the tiny numbers so that they survive the trip. In this part you open the boxes: you decode the bits of a number, write your own rounding to any format and check it against PyTorch, watch numbers overflow and vanish, train a network with emulated 16-bit arithmetic with and without **loss scaling**, and finally measure what real 16-bit arithmetic does for speed and memory *on this machine*. ==The honest result may surprise you: on a computer without fast 16-bit hardware, 16 bits can be slower, not faster.==

**Imagine first.** (1) What is the largest finite number a 16-bit float (float16) can hold? (2) What does float16 store for 70,000? (3) You add 0.0001 to a float16 running total 100,000 times; the true total is 10. What total do you expect? (4) What does float16 store for the gradient 1e-8? Write your guesses in the block below (replace each `None`) and compare in Step 4.11.

```python
# Imagine first: the numbers inside the computer
my_guess_largest = None   # the largest finite number that a 16-bit float (float16) can hold
my_guess_70000 = None     # what float16 stores for 70000 (a number, or float("inf") if you think it is not stored)
my_guess_sum = None       # add 0.0001 to a float16 running total 100,000 times (the true answer is 10): what total do you expect?
my_guess_tiny = None      # what float16 stores for the gradient 1e-8 (a number)
```

```python
# Helpers that this part needs (re-declared here so that Part 4 runs on its own).
import struct
if "RESULTS" not in globals(): RESULTS = {}
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
```

1. **Step 4.1. The bits of a number.** A floating-point number is stored like scientific notation in binary: a **sign** bit, an **exponent** (which power of two) and a **mantissa** (the digits after the leading 1). The three formats of this part split their bits differently: **float32** has 1 + 8 + 23 bits, **float16** has 1 + 5 + 10, and **bfloat16** (brain float) has 1 + 8 + 7: it keeps the 8 exponent bits of float32 (the same range) but only 7 mantissa bits (less precision). A normal number is (−1)^sign × 2^(exponent field − bias) × (1 + mantissa ÷ 2^m), with bias = 2^(e−1) − 1 (15 for float16, 127 for the other two). The cell prints the bit fields of five numbers in the three formats; `struct` reads the bits of a float32, and NumPy and torch give the 16-bit ones.

   ```python
   # The bits of a number: sign, exponent, mantissa.
   FORMATS = {"float32": (8, 23), "float16": (5, 10), "bfloat16": (8, 7)}      # (exponent bits, mantissa bits)

   def p4_bits(x, fmt):
       """The bit pattern of x in a format, as an integer."""
       if fmt == "float32": return struct.unpack(">I", struct.pack(">f", x))[0]
       if fmt == "float16": return int(np.array(x, dtype=np.float16).view(np.uint16))
       return int(torch.tensor(x, dtype=torch.float32).bfloat16().view(torch.int16).item()) & 0xFFFF       # bfloat16

   def p4_fields(bits, fmt):
       e, m = FORMATS[fmt]
       return bits >> (e + m), (bits >> m) & ((1 << e) - 1), bits & ((1 << m) - 1)

   for x in (0.15625, -2.5, 1 / 3, 70000.0, 1e-8):
       print(f"x = {x!r}")
       for fmt in FORMATS:
           e, m = FORMATS[fmt]
           s, ex, ma = p4_fields(p4_bits(x, fmt), fmt)
           print(f"  {fmt:9s} {s:1b} {ex:0{e}b} {ma:0{m}b}   sign {s}, exponent field {ex:3d}, mantissa field {ma}")
   ```

   The block prints (your numbers must match):

   ```text
   x = 0.15625
     float32   0 01111100 01000000000000000000000   sign 0, exponent field 124, mantissa field 2097152
     float16   0 01100 0100000000   sign 0, exponent field  12, mantissa field 256
     bfloat16  0 01111100 0100000   sign 0, exponent field 124, mantissa field 32
   x = -2.5
     float32   1 10000000 01000000000000000000000   sign 1, exponent field 128, mantissa field 2097152
     float16   1 10000 0100000000   sign 1, exponent field  16, mantissa field 256
     bfloat16  1 10000000 0100000   sign 1, exponent field 128, mantissa field 32
   x = 0.3333333333333333
     float32   0 01111101 01010101010101010101011   sign 0, exponent field 125, mantissa field 2796203
     float16   0 01101 0101010101   sign 0, exponent field  13, mantissa field 341
     bfloat16  0 01111101 0101011   sign 0, exponent field 125, mantissa field 43
   x = 70000.0
     float32   0 10001111 00010001011100000000000   sign 0, exponent field 143, mantissa field 571392
     float16   0 11111 0000000000   sign 0, exponent field  31, mantissa field 0
     bfloat16  0 10001111 0001001   sign 0, exponent field 143, mantissa field 9
   x = 1e-08
     float32   0 01100100 01010111100110001110111   sign 0, exponent field 100, mantissa field 2870391
     float16   0 00000 0000000000   sign 0, exponent field   0, mantissa field 0
     bfloat16  0 01100100 0101100   sign 0, exponent field 100, mantissa field 44
   ```

   Read the rows. For 0.15625 = 1.25 × 2^−3 the exponent field of float32 is 124 = 127 − 3 and the mantissa field starts with 01 (the .25 of 1.25). Float16 stores the same number with exponent field 12 = 15 − 3. For **70,000** float16 shows the exponent field **31** (all ones) and mantissa 0: that is the code for **infinity**, because 70,000 is above the largest float16, while bfloat16 stores it with a mantissa of only 7 bits. For **1e-8** float16 shows exponent 0 and mantissa 0: the number was **flattened to zero**, while bfloat16 keeps it with exponent field 100.

   *Exercise 4.1:* Read the bits back. Your task: write `decode_bits(bits, e_bits, m_bits)`, which turns an integer bit pattern back into the number it stands for, for any format. Use the formula of Step 4.1 for normal numbers; when the exponent field is 0 the number is **subnormal** (no leading 1: the value is 2^(1 − bias) × mantissa ÷ 2^m); when the exponent field is all ones it is infinity (mantissa 0) or not-a-number. Why it matters: a subnormal is how a format stretches its range downwards, at the price of fewer digits, and it is exactly where tiny gradients live.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   def decode_bits(bits, e_bits, m_bits):
       # your code here: the value (a Python float) that the integer `bits` stands for in a format with e_bits exponent bits and m_bits mantissa bits.
       # Normal numbers:    (-1)^sign * 2^(exponent - bias) * (1 + mantissa / 2^m_bits),  with bias = 2^(e_bits - 1) - 1
       # Subnormal numbers (exponent field 0): (-1)^sign * 2^(1 - bias) * (mantissa / 2^m_bits).  All ones in the exponent field: inf (mantissa 0) or nan.
       return None
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if decode_bits(0, 5, 10) is None:
       print("Write decode_bits above, then run this cell again.")
   else:
       check("float16 0x3C00 is 1.0", decode_bits(0x3C00, 5, 10) == 1.0, "exponent field 15 = bias, mantissa 0: 2^0 x 1.")
       check("float16 0x7BFF is 65504", decode_bits(0x7BFF, 5, 10) == 65504.0, "exponent field 30, mantissa 1023: 2^15 x (1 + 1023/1024).")
       check("float16 0x0001 is 2^-24, the smallest subnormal", decode_bits(0x0001, 5, 10) == 2.0 ** -24, "exponent field 0: 2^(1 - 15) x (1 / 1024).")
       check("float16 0xC000 is -2", decode_bits(0xC000, 5, 10) == -2.0, "sign bit 1.")
       check("float16 0x7C00 is inf", decode_bits(0x7C00, 5, 10) == float("inf"), "exponent field all ones and mantissa 0.")
       check("bfloat16 0x3F80 is 1.0", decode_bits(0x3F80, 8, 7) == 1.0, "bias 127.")
       check("float32 0x3E800000 is 0.25", decode_bits(0x3E800000, 8, 23) == 0.25, "bias 127, exponent field 125.")
       _ok = all(decode_bits(p4_bits(v, f), *FORMATS[f]) == float(torch.tensor(v).to({"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[f]).item())
                 for v in (0.1, -3.75, 1234.5, 6e-5, 3e-7) for f in FORMATS)
       check("decoding the bits of real numbers gives back what torch stores", _ok, "Compare with torch.tensor(v).half().item().")
       RESULTS["p4_decode"] = True
   ```

   A possible answer, and what it means:

   ```python
   def decode_bits(bits, e_bits, m_bits):
       bias = 2 ** (e_bits - 1) - 1
       sign = -1.0 if (bits >> (e_bits + m_bits)) & 1 else 1.0
       exp = (bits >> m_bits) & ((1 << e_bits) - 1)
       man = bits & ((1 << m_bits) - 1)
       if exp == (1 << e_bits) - 1:
           return sign * float("inf") if man == 0 else float("nan")
       if exp == 0:                                   # subnormal: no hidden 1, fixed exponent
           return sign * 2.0 ** (1 - bias) * (man / 2 ** m_bits)
       return sign * 2.0 ** (exp - bias) * (1 + man / 2 ** m_bits)
   ```

   The three cases are the whole standard. A **normal** number has a hidden leading 1, so its value is 2^(exponent − bias) × (1 + mantissa ÷ 2^m). A **subnormal** has exponent field 0, no hidden 1, and the fixed exponent 1 − bias, so the numbers below the smallest normal one are evenly spaced (for float16 the spacing is 2^−24, about 6 × 10^−8, which is the smallest positive number it can hold). The all-ones exponent is reserved for infinity and not-a-number. Check 0x7BFF in float16: exponent field 30, mantissa 1023, so 2^15 × (1 + 1023 ÷ 1024) = 65,504, the largest finite float16.

2. **Step 4.2. Write the rounding yourself.** Storing a number in a format means **rounding** it to the nearest number the format can hold (when it is exactly halfway, to the neighbour with an even last digit). Around a value with exponent e the neighbours of a format with m mantissa bits are 2^(e − m) apart; below the smallest normal number the gap stops shrinking (the subnormals); above the largest number the result is infinity. That is all `round_to_format(x, e_bits, m_bits)` does, in a few lines of NumPy. The cell compares it with PyTorch's own conversion on 20 hand-picked values (including overflow, subnormals and halfway cases) and on 100,000 random values spread over 18 powers of ten, for float32, float16 and bfloat16.

   ```python
   # Our own rounding to any format with e_bits exponent bits and m_bits mantissa bits (round to nearest, ties to even).
   def round_to_format(x, e_bits, m_bits):
       x = np.asarray(x, dtype=np.float64)
       bias = 2 ** (e_bits - 1) - 1
       emin, emax = 1 - bias, bias
       max_normal = (2 - 2.0 ** -m_bits) * 2.0 ** emax
       ax = np.abs(x)
       e = np.floor(np.log2(np.where(ax > 0, ax, 1.0)))     # the power of two just below |x|
       e = np.maximum(e, emin)                              # below the smallest normal number the spacing stops shrinking: these are the subnormals
       step = 2.0 ** (e - m_bits)                           # the gap between two neighbouring numbers of the format around |x|
       r = np.rint(ax / step) * step                        # np.rint rounds halves to the even neighbour, like the hardware
       r = np.where(r > max_normal, np.inf, r)              # too big for the format: infinity
       r = np.where(np.isfinite(ax), r, ax)                 # inf and nan stay as they are
       return np.copysign(r, x)

   p4_torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
   def p4_torch_round(x32, fmt):
       return torch.tensor(x32, dtype=torch.float32).to(p4_torch_dtype[fmt]).double().numpy()

   p4_special = np.array([0.0, 1.0, -1.0, 0.1, 1 / 3, 3.14159265, 65504.0, 65519.0, 65520.0, 70000.0, -1e5, 6.1035e-5, 6.0e-5,
                          5.96e-8, 3.0e-8, 2.9e-8, 1e-8, -2.5e-5, 1e30, 123456.789], dtype=np.float32)
   rng = np.random.default_rng(SEED)
   p4_random = (rng.choice([-1, 1], 100_000) * 10.0 ** rng.uniform(-12, 6, 100_000)).astype(np.float32)
   print(f"{'x':>12} {'ours float16':>14} {'torch float16':>14} {'ours bfloat16':>14} {'torch bfloat16':>14}")
   for v in p4_special[:20]:
       print(f"{float(v):12.6g} {float(round_to_format(v, 5, 10)):14.6g} {float(p4_torch_round(v, 'float16')):14.6g} {float(round_to_format(v, 8, 7)):14.6g} {float(p4_torch_round(v, 'bfloat16')):14.6g}")
   p4_same = {}
   for fmt, (e, m) in FORMATS.items():
       for name, data in (("20 chosen values", p4_special), ("100,000 random values", p4_random)):
           ours = round_to_format(data.astype(np.float64), e, m); ref = p4_torch_round(data, fmt)
           p4_same[(fmt, name)] = bool(np.array_equal(ours, ref))
       print(f"{fmt:9s} identical to torch on the 20 values: {p4_same[(fmt, '20 chosen values')]}, on the 100,000 random values: {p4_same[(fmt, '100,000 random values')]}")
   assert all(p4_same.values())
   ```

   The block prints (your numbers must match):

   ```text
              x   ours float16  torch float16  ours bfloat16 torch bfloat16
              0              0              0              0              0
              1              1              1              1              1
             -1             -1             -1             -1             -1
            0.1      0.0999756      0.0999756       0.100098       0.100098
       0.333333       0.333252       0.333252       0.333984       0.333984
        3.14159        3.14062        3.14062        3.14062        3.14062
          65504          65504          65504          65536          65536
          65519          65504          65504          65536          65536
          65520            inf            inf          65536          65536
          70000            inf            inf          70144          70144
        -100000           -inf           -inf         -99840         -99840
     6.1035e-05    6.10352e-05    6.10352e-05    6.10352e-05    6.10352e-05
          6e-05    6.00219e-05    6.00219e-05    6.00815e-05    6.00815e-05
       5.96e-08    5.96046e-08    5.96046e-08    5.96046e-08    5.96046e-08
          3e-08    5.96046e-08    5.96046e-08    3.00352e-08    3.00352e-08
        2.9e-08              0              0    2.89874e-08    2.89874e-08
          1e-08              0              0    1.00117e-08    1.00117e-08
       -2.5e-05   -2.49743e-05   -2.49743e-05    -2.5034e-05    -2.5034e-05
          1e+30            inf            inf    1.00026e+30    1.00026e+30
         123457            inf            inf         123392         123392
   float32   identical to torch on the 20 values: True, on the 100,000 random values: True
   float16   identical to torch on the 20 values: True, on the 100,000 random values: True
   bfloat16  identical to torch on the 20 values: True, on the 100,000 random values: True
   ```

   Our 12-line function gives **exactly** the same numbers as torch for all three formats, on the 20 chosen values and on all 100,000 random ones: nothing mysterious happens when a tensor is converted. Look at the table: 0.1 becomes 0.0999756 in float16 and 0.100098 in bfloat16 (the error is 0.02 % and 0.1 %); **65,519 rounds down to 65,504 but 65,520 rounds to infinity** (halfway between the largest number and the next power of two, ties go to the even neighbour, which is out of range); 3 × 10^−8 becomes 5.96 × 10^−8 while 2.9 × 10^−8 becomes **0**; and in bfloat16 the same values survive, because its exponent has the range of float32.

   *Exercise 4.2:* Range and precision of a format. Your task: write `fmt_info(e_bits, m_bits)` returning `(largest, smallest_normal, epsilon)`. With bias = 2^(e − 1) − 1: the largest finite number is (2 − 2^−m) × 2^bias, the smallest normal number is 2^(1 − bias), and the **machine epsilon** (the relative gap between a number and its neighbour) is 2^−m. The check compares your function with `torch.finfo`. Why it matters: these three numbers are everything you need to decide whether a value can overflow, vanish or lose digits in a format.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   def fmt_info(e_bits, m_bits):
       # your code here: return (largest, smallest_normal, epsilon) for a format with e_bits exponent bits and m_bits mantissa bits.
       # bias = 2^(e_bits - 1) - 1;  largest = (2 - 2^-m_bits) * 2^bias;  smallest normal = 2^(1 - bias);  epsilon = 2^-m_bits (the relative gap between neighbours)
       return None
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   if fmt_info(5, 10) is None:
       print("Write fmt_info above, then run this cell again.")
   else:
       L, S, E = fmt_info(5, 10)
       check("float16: largest 65,504", L == 65504.0, "(2 - 2^-10) x 2^15.")
       check("float16: smallest normal 6.1e-5", abs(S - 6.1035e-5) < 1e-8, "2^(1 - 15) = 2^-14.")
       check("float16: epsilon 9.77e-4", abs(E - 9.765625e-4) < 1e-12, "2^-10.")
       L2, S2, E2 = fmt_info(8, 7)
       check("bfloat16: epsilon 7.8e-3", abs(E2 - 0.0078125) < 1e-12, "2^-7.")
       check("bfloat16: largest is about 3.39e38 and the smallest normal about 1.18e-38", abs(L2 / 3.3895e38 - 1) < 1e-3 and abs(S2 / 1.1755e-38 - 1) < 1e-3, "Same bias 127 as float32.")
       check("agrees with torch.finfo for the three formats", all(abs(fmt_info(*FORMATS[f])[0] / torch.finfo(p4_torch_dtype[f]).max - 1) < 1e-6 and abs(fmt_info(*FORMATS[f])[2] / torch.finfo(p4_torch_dtype[f]).eps - 1) < 1e-6 for f in FORMATS), "torch.finfo(torch.float16).max and .eps.")
       RESULTS["p4_fmt_info"] = (L, round(S, 8), E)
   ```

   A possible answer, and what it means:

   ```python
   def fmt_info(e_bits, m_bits):
       bias = 2 ** (e_bits - 1) - 1
       return (2 - 2.0 ** -m_bits) * 2.0 ** bias, 2.0 ** (1 - bias), 2.0 ** -m_bits
   ```

   For float16 (e = 5, m = 10, bias 15) the largest number is (2 − 2^−10) × 2^15 = **65,504**, the smallest normal one is 2^−14 = **6.1 × 10^−5** and epsilon is 2^−10 = **9.77 × 10^−4**, about 3 decimal digits. For bfloat16 (e = 8, m = 7, bias 127) the largest number is about 3.39 × 10^38 and the smallest normal one about 1.18 × 10^−38 (the range of float32), but epsilon is 2^−7 = **7.8 × 10^−3**, only about 2 decimal digits. So: **float16 has more digits but a small range; bfloat16 has a huge range but fewer digits.**

3. **Step 4.3. The range and precision table.** The cell computes the table for the three formats with the formulas, checks it against `torch.finfo` and against the numbers of the lesson, and measures the **largest relative rounding error** on 100,000 random numbers inside each format's normal range. Rounding to nearest can be wrong by at most half the gap, so the relative error must stay below epsilon ÷ 2.

   ```python
   # Range and precision of the three formats.
   print(f"{'format':9s} {'largest':>11} {'smallest normal':>16} {'smallest subnormal':>19} {'epsilon':>10} {'digits':>7}")
   p4_info = {}
   for fmt, (e, m) in FORMATS.items():
       bias = 2 ** (e - 1) - 1
       p4_info[fmt] = ((2 - 2.0 ** -m) * 2.0 ** bias, 2.0 ** (1 - bias), 2.0 ** (1 - bias - m), 2.0 ** -m)
       L, S, S2, E = p4_info[fmt]
       print(f"{fmt:9s} {L:11.4g} {S:16.4g} {S2:19.4g} {E:10.3g} {-math.log10(E):7.1f}")
       fi = torch.finfo(p4_torch_dtype[fmt]); assert abs(L / fi.max - 1) < 1e-6 and abs(S / fi.tiny - 1) < 1e-6 and abs(E / fi.eps - 1) < 1e-6
   assert p4_info["float16"][0] == 65504 and abs(p4_info["float16"][1] - 6.1e-5) < 1e-6 and abs(p4_info["float16"][3] - 9.77e-4) < 1e-5 and abs(p4_info["bfloat16"][3] - 7.8e-3) < 1e-4
   # the largest rounding error, relative to the value, measured on random values inside the normal range
   p4_rel = {}
   for fmt, (e, m) in FORMATS.items():
       L, S, _, E = p4_info[fmt]
       v = 10.0 ** rng.uniform(math.log10(S) + 0.5, min(math.log10(L) - 0.5, 30), 100_000)
       p4_rel[fmt] = float(np.max(np.abs(round_to_format(v, e, m) - v) / v))
       print(f"{fmt:9s} largest relative rounding error on 100,000 random values: {p4_rel[fmt]:.2e}   (half of epsilon: {E / 2:.2e})")
       assert p4_rel[fmt] <= E / 2 * 1.0001
   ```

   The block prints (your numbers must match):

   ```text
   format        largest  smallest normal  smallest subnormal    epsilon  digits
   float32     3.403e+38        1.175e-38           1.401e-45   1.19e-07     6.9
   float16      6.55e+04        6.104e-05            5.96e-08   0.000977     3.0
   bfloat16     3.39e+38        1.175e-38           9.184e-41    0.00781     2.1
   float32   largest relative rounding error on 100,000 random values: 5.94e-08   (half of epsilon: 5.96e-08)
   float16   largest relative rounding error on 100,000 random values: 4.85e-04   (half of epsilon: 4.88e-04)
   bfloat16  largest relative rounding error on 100,000 random values: 3.89e-03   (half of epsilon: 3.91e-03)
   ```

   The measured worst-case relative errors are **5.9e-08** (float32), **4.9e-04** (float16) and **3.9e-03** (bfloat16): they sit right at epsilon ÷ 2, as the theory says. In words: float16 stores a number to about 0.05 percent, bfloat16 to about 0.4 percent, float32 to 6 × 10^−6 percent. Notice the trade: from float16 to bfloat16 we give up 3 mantissa bits (8 times less precise) and gain 3 exponent bits (the largest number grows from 6.5 × 10^4 to 3.4 × 10^38).

4. **Step 4.4. Overflow, underflow and a sum that stops growing.** Three small experiments show how a format fails. (1) **Overflow:** store 70,000. (2) **Underflow:** store gradients from 10^−3 down to 10^−8 in float16 and in bfloat16. (3) **A running total:** add float16(0.0001) to a float16 total 100,000 times; the total stops growing when the gap between neighbouring float16 numbers is more than twice the number you add, because the sum then rounds back to the old total. We also try the same sum in float32 and with `torch.sum` on a half-precision tensor. Finally a small table shows **loss scaling** in numbers: a gradient of 10^−8 multiplied by a scale before it is rounded and divided by the same scale after.

   ```python
   # Overflow and underflow.
   p4_over = {f: float(round_to_format(70000.0, *FORMATS[f])) for f in FORMATS}
   print("70000 stored as:", {f: v for f, v in p4_over.items()})
   print(f"{'value':>9} {'float16':>12} {'rel. error':>11} {'bfloat16':>12} {'rel. error':>11}")
   for v in (1e-3, 1e-5, 1e-7, 6e-8, 3e-8, 1e-8):
       h = float(round_to_format(v, 5, 10)); b = float(round_to_format(v, 8, 7))
       print(f"{v:9.0e} {h:12.4g} {abs(h - v) / v:11.1%} {b:12.4g} {abs(b - v) / v:11.1%}")

   # a running total of 100,000 small numbers, in 16 bits and in 32 bits
   p4_x = np.float16(1e-4)
   p4_run16 = np.cumsum(np.full(100_000, p4_x, dtype=np.float16), dtype=np.float16)[-1]       # the total itself is kept in float16
   p4_run32 = np.cumsum(np.full(100_000, p4_x, dtype=np.float32), dtype=np.float32)[-1]
   p4_run_t = torch.full((100_000,), float(p4_x), dtype=torch.float16).sum().item()
   print(f"\n100,000 additions of float16(0.0001) = {float(p4_x):.6g}: true total {100_000 * float(p4_x):.4f}")
   print(f"  running total kept in float16: {float(p4_run16):.4f}   | kept in float32: {float(p4_run32):.4f}   | torch half .sum() (accumulates in 32 bits inside): {p4_run_t:.4f}")
   # why it stops: at a total of 0.25 the gap between neighbouring float16 numbers is 2^-12
   print(f"  the gap between float16 numbers near {float(p4_run16):.2f} is {2.0 ** (math.floor(math.log2(float(p4_run16))) - 10):.2e}; half of it is more than 0.0001, so adding 0.0001 changes nothing")
   assert float(p4_run16) < 1 and abs(float(p4_run32) - 10) < 0.1 and p4_over["float16"] == float("inf") and np.isfinite(p4_over["bfloat16"])

   # loss scaling in numbers: a gradient of 1e-8 survives if it is multiplied before it is rounded and divided afterwards
   print(f"\n{'loss scale':>11} {'1e-8 x scale':>13} {'stored in float16':>18} {'divided back':>13} {'error':>8}")
   for s in (1, 1024, 65536):
       stored = float(round_to_format(1e-8 * s, 5, 10)); back = stored / s
       print(f"{s:11d} {1e-8 * s:13.3e} {stored:18.4e} {back:13.4e} {abs(back - 1e-8) / 1e-8:8.1%}")
   ```

   The block prints (your numbers must match):

   ```text
   70000 stored as: {'float32': 70000.0, 'float16': inf, 'bfloat16': 70144.0}
       value      float16  rel. error     bfloat16  rel. error
       1e-03        0.001        0.0%    0.0009995        0.1%
       1e-05    1.001e-05        0.1%    1.001e-05        0.1%
       1e-07    1.192e-07       19.2%    1.001e-07        0.1%
       6e-08     5.96e-08        0.7%    6.007e-08        0.1%
       3e-08     5.96e-08       98.7%    3.004e-08        0.1%
       1e-08            0      100.0%    1.001e-08        0.1%

   100,000 additions of float16(0.0001) = 0.000100017: true total 10.0017
     running total kept in float16: 0.2500   | kept in float32: 10.0112   | torch half .sum() (accumulates in 32 bits inside): 10.0000
     the gap between float16 numbers near 0.25 is 2.44e-04; half of it is more than 0.0001, so adding 0.0001 changes nothing

    loss scale  1e-8 x scale  stored in float16  divided back    error
             1     1.000e-08         0.0000e+00    0.0000e+00   100.0%
          1024     1.024e-05         1.0252e-05    1.0012e-08     0.1%
         65536     6.554e-04         6.5517e-04    9.9972e-09     0.0%
   ```

   **Overflow:** 70,000 becomes **infinity** in float16 and 70,144 in bfloat16. **Underflow:** the float16 error grows from 0.1 % at 10^−5 to 19% at 10^−7 (the subnormals are coarse), the value 3 × 10^−8 is stored as 5.96 × 10^−8 (an error of almost 100 %) and **1e-8 becomes 0**; bfloat16 keeps 0.1 % of precision all the way down. **The running total** is the most instructive: the true answer is 10.00, a float16 total stops at **0.25**, float32 reaches 10.01, and `torch.sum` on a float16 tensor reaches 10.00 because it *accumulates in 32 bits inside* and only rounds the final result: the libraries already use this trick. The scale table shows loss scaling at work: without scaling the gradient 10^−8 is lost (100 % error); multiplied by 1,024 it survives with an error of 0.1 %, by 65,536 with almost none. Remember the rule that follows: **keep sums and master weights in 32 bits.**

5. **Step 4.5. Emulating 16 bits on any computer.** The rounding that matters for accuracy can be reproduced without any special hardware: round every value to 16 bits and go back to float32, `x.half().float()` (or `.bfloat16().float()`). The numbers have exactly the precision and range of 16-bit values, but they still occupy 4 bytes and the arithmetic is the ordinary 32-bit one. We use this **emulation** in the next steps to study the *accuracy* of mixed precision. It cannot say anything about *speed* or *memory*: for that we need the real thing (Step 4.8). The cell checks that the emulation equals your `round_to_format`, counts the bytes and times the extra conversions.

   ```python
   # The "16-bit storage, fast 32-bit compute" trick: round to 16 bits and go back to float32. The rounding is real; the speed is not.
   torch.manual_seed(SEED)
   p4_t = torch.randn(1000, 1000) * 100
   p4_emu = p4_t.half().float()
   print("x.half().float() equals our round_to_format(x, 5, 10):", bool(np.array_equal(p4_emu.double().numpy(), round_to_format(p4_t.double().numpy(), 5, 10))))
   print("bytes per value: float32", p4_t.element_size(), "| real half tensor", p4_t.half().element_size(), "| emulated (half().float())", p4_emu.element_size(), "<- no memory saved")
   print(f"largest rounding error of the emulation: {(p4_emu - p4_t).abs().max().item():.4f} on values up to {p4_t.abs().max().item():.0f}")
   p4_a, p4_b = torch.randn(512, 512), torch.randn(512, 512)
   p4_t_plain = time_ms(lambda: p4_a @ p4_b, repeat=5)
   p4_t_emu = time_ms(lambda: p4_a.half().float() @ p4_b.half().float(), repeat=5)
   print(f"512 x 512 matrix product: plain float32 {p4_t_plain:.2f} ms, with both operands rounded to 16 bits first {p4_t_emu:.2f} ms ({p4_t_emu / p4_t_plain:.1f}x): the emulation costs time, it does not save any")
   ```

   The block prints (your numbers must match):

   ```text
   x.half().float() equals our round_to_format(x, 5, 10): True
   bytes per value: float32 4 | real half tensor 2 | emulated (half().float()) 4 <- no memory saved
   largest rounding error of the emulation: 0.1249 on values up to 483
   512 x 512 matrix product: plain float32 1.27 ms, with both operands rounded to 16 bits first 1.01 ms (0.8x): the emulation costs time, it does not save any
   ```

   The emulation is **identical** to `round_to_format(x, 5, 10)`, but it keeps 4 bytes per value (a real half tensor has 2), and the two extra conversions made the matrix product 0.8 times as slow as plain float32. Emulation is a tool for studying numbers, never a speed-up.

   *Exercise 4.3:* A dynamic loss scaler. Your task: write the class `LossScaler`. It keeps a number `scale` (starting at 2^16 = 65,536). The method `update(found_inf)` is called after every training step: when the gradients overflowed (`found_inf` is `True`) the step is skipped, the scale is **multiplied by `backoff`** (0.5) and the count of good steps starts again from 0; when they did not, the count of good steps goes up by one, and after `interval` good steps in a row the scale is **multiplied by `growth`** (2.0) and the count restarts. Why it matters: the best scale is the largest one that does not overflow, and it changes during training, so the scaler searches for it by trial: it grows slowly and backs off fast.

   Replace the starter, run it, then run the **Check** cell.

   Write your answer in this starter block:

   ```python
   class LossScaler:
       # your code here. Keep self.scale (start 2**16). update(found_inf): after an overflow multiply the scale by backoff (0.5) and restart the count of good steps;
       # after `interval` good steps in a row multiply the scale by growth (2.0) and restart the count.
       def __init__(self, init_scale=2.0 ** 16, growth=2.0, backoff=0.5, interval=50):
           pass
       def update(self, found_inf):
           pass
   ```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

   ```python
   try:
       _s = LossScaler(init_scale=1024.0, interval=3)
       _ok = hasattr(_s, "scale")
   except Exception:
       _ok = False
   if not _ok:
       print("Write the class LossScaler above, then run this cell again.")
   else:
       _s = LossScaler(init_scale=1024.0, interval=3)
       _s.update(True);  a = _s.scale
       for _ in range(2): _s.update(False)
       b = _s.scale
       _s.update(False); c = _s.scale
       _s.update(True); _s.update(False); _s.update(False); d = _s.scale
       check("an overflow halves the scale", a == 512.0, "scale *= backoff when found_inf is True.")
       check("two good steps change nothing (interval 3)", b == 512.0, "Count the good steps in a row.")
       check("the third good step doubles the scale", c == 1024.0, "scale *= growth after `interval` good steps, then restart the count.")
       check("an overflow restarts the count of good steps", d == 512.0, "After the overflow only 2 good steps follow: no growth yet.")
       RESULTS["p4_scaler"] = True
   ```

   A possible answer, and what it means:

   ```python
   class LossScaler:
       def __init__(self, init_scale=2.0 ** 16, growth=2.0, backoff=0.5, interval=50):
           self.scale, self.growth, self.backoff, self.interval, self.good = init_scale, growth, backoff, interval, 0
       def update(self, found_inf):
           if found_inf:
               self.scale *= self.backoff; self.good = 0
           else:
               self.good += 1
               if self.good == self.interval:
                   self.scale *= self.growth; self.good = 0
   ```

   The scaler is a tiny feedback controller. Overflow is dangerous (an infinity in a gradient would ruin the weights), so it reacts at once: halve the scale and skip that step. Underflow is silent and mild, so it tests upwards only after `interval` quiet steps. Over time the scale settles near the largest power of two that the gradients can stand. This is the same idea as the scaler of PyTorch's automatic mixed precision, written in a dozen lines.

6. **Step 4.6. Mixed-precision training from scratch: which gradients vanish?.** Now a network. It has 6 hidden layers of 64 units and a regression task with 128 outputs, and the loss is the **mean** over 4,096 samples and 128 outputs, which makes every gradient small (each output error is divided by about half a million). The master weights stay in float32. For the forward and backward pass the weights, the inputs, every layer output and **every gradient on its way back** are rounded to the 16-bit format by a custom autograd function (the emulation of Step 4.5). The first experiment takes the starting weights and computes the gradients once in plain float32 and once through float16 or bfloat16 copies, with different **loss scales**: the loss is multiplied by the scale before the backward pass and the gradients are divided by it after. We count the gradient entries that are non-zero in float32 but **zero** after the 16-bit pass (flushed to zero), and the entries that became infinite or not-a-number (overflow).

   ```python
   # Mixed-precision training from scratch. A deep regression network; the master weights stay float32; forward and backward run on "16-bit" copies (rounded with x.to(dtype).float()).
   class P4Round(torch.autograd.Function):
       """Rounds the values on the way forward AND the gradients on the way back to a 16-bit format: this is what 16-bit tensors do to a backward pass."""
       @staticmethod
       def forward(ctx, x, dt): ctx.dt = dt; return x.to(dt).float()
       @staticmethod
       def backward(ctx, g): return g.to(ctx.dt).float(), None

   P4_N, P4_IN, P4_W, P4_OUT, P4_DEPTH = 4096, 16, 64, 128, 6
   _g = torch.Generator().manual_seed(SEED)
   P4_X = torch.randn(P4_N, P4_IN, generator=_g)
   P4_TS = 0.3                                   # the size of the targets
   P4_Y = torch.tanh(P4_X @ (torch.randn(P4_IN, P4_OUT, generator=_g) / 4)) * P4_TS

   def p4_init(seed=SEED):
       torch.manual_seed(seed)
       dims = [P4_IN] + [P4_W] * P4_DEPTH + [P4_OUT]
       ws = [(torch.randn(a, b) * (0.8 * math.sqrt(2 / a))).requires_grad_() for a, b in zip(dims[:-1], dims[1:])]
       outs = dims[1:]
       bs = [torch.zeros(b, requires_grad=True) for b in outs]
       return ws + bs

   def p4_forward(params, x, dt=None):
       """dt None: plain float32. Otherwise every weight, input and layer output is rounded to dt (forward) and every gradient too (backward)."""
       ws, bs = params[:len(params) // 2], params[len(params) // 2:]
       r = (lambda t: P4Round.apply(t, dt)) if dt is not None else (lambda t: t)
       h = r(x)
       for i, (w, b) in enumerate(zip(ws, bs)):
           h = r(h @ r(w) + r(b))
           if i < len(ws) - 1: h = torch.relu(h)
       return h

   def p4_grads(params, dt, scale):
       """Gradients of the master weights computed through the 16-bit copies with a loss scale, and divided by the scale afterwards."""
       for p in params: p.grad = None
       loss = ((p4_forward(params, P4_X, dt) - P4_Y) ** 2).mean()
       (loss * scale).backward()
       return [p.grad / scale for p in params], loss.item()

   # gradients at the starting point: how many that are not zero in float32 become zero in float16 (and how many overflow)?
   p4_params0 = p4_init()
   p4_g32, p4_loss0 = p4_grads(p4_params0, None, 1.0)
   p4_nz = sum(int((g != 0).sum()) for g in p4_g32)
   p4_flush = {}
   print(f"loss at the start: {p4_loss0:.4f}; gradient entries that are not zero in float32: {p4_nz:,}")
   print(f"{'format':9s} {'loss scale':>10} {'flushed to zero':>16} {'overflow (inf/nan)':>19}")
   for dt, name in ((torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
       for s in (1.0, 1024.0, 65536.0, 2.0 ** 24):
           g, _ = p4_grads(p4_params0, dt, s)
           bad = sum(int((~torch.isfinite(t)).sum()) for t in g)
           flushed = sum(int(((a != 0) & (b == 0)).sum()) for a, b in zip(p4_g32, g))
           p4_flush[(name, s)] = (flushed / p4_nz, bad)
           print(f"{name:9s} {s:10.0f} {flushed / p4_nz:16.2%} {bad:19,d}")
   ```

   The block prints (your numbers must match):

   ```text
   loss at the start: 0.1875; gradient entries that are not zero in float32: 29,203
   format    loss scale  flushed to zero  overflow (inf/nan)
   float16            1            0.64%                   0
   float16         1024            0.00%                   0
   float16        65536            0.00%                   0
   float16     16777216            0.00%               5,155
   bfloat16           1            0.00%                   0
   bfloat16        1024            0.00%                   0
   bfloat16       65536            0.00%                   0
   bfloat16    16777216            0.00%                   0
   ```

   At the starting point 29,203 gradient entries are non-zero. With float16 and **no scaling 0.64%** of them are flushed to zero; a scale of 1,024 brings it to 0.00%. A scale that is **too big** causes the opposite failure: with 2^24 there are 5,155 overflowed entries (infinity or not-a-number), which would destroy the weights. So the scale must be neither too small nor too large, which is why it is adjusted during training. **bfloat16 needs none of this:** it flushes 0.00% at scale 1 and never overflows, because its range is that of float32.

7. **Step 4.7. Training runs: scaling or not?.** Five settings train the same network for 200 steps with Adam (the Adam update runs on the float32 master weights): plain float32; float16 without scaling; float16 with a fixed scale of 1,024; float16 with the dynamic scaler (starting at 2^16, with `interval = 25`: an overflow skips the step and halves the scale); and bfloat16 without scaling. Every setting is trained from 3 different starting weights; the table gives the mean final loss, its range, the mean number of skipped steps and the mean share of gradient entries flushed to zero (measured every 25 steps against float32 gradients at the same weights). This cell is the slowest of the part: it takes about a minute on a laptop CPU.

   ```python
   # Training runs: float32, float16 without scaling, float16 with a fixed scale, float16 with the dynamic LossScaler, bfloat16 without scaling.
   class P4Scaler:
       """The same dynamic loss scaler as in Exercise 4.3 (copied here so that this cell runs on its own)."""
       def __init__(self, init_scale=2.0 ** 16, growth=2.0, backoff=0.5, interval=50):
           self.scale, self.growth, self.backoff, self.interval, self.good = init_scale, growth, backoff, interval, 0
       def update(self, found_inf):
           if found_inf: self.scale *= self.backoff; self.good = 0
           else:
               self.good += 1
               if self.good == self.interval: self.scale *= self.growth; self.good = 0

   def p4_eval(params):
       with torch.no_grad(): return ((p4_forward(params, P4_X) - P4_Y) ** 2).mean().item()

   def p4_train(dt, mode, steps, seed=SEED, lr=3e-3, probe_every=25):
       params = p4_init(seed)
       opt = torch.optim.Adam(params, lr=lr)
       scaler = P4Scaler(interval=25) if mode == "dynamic" else None
       scale = {"none": 1.0, "fixed": 1024.0, "dynamic": 2.0 ** 16}[mode] if dt is not None else 1.0
       curve, skipped, scales, flushed = [], 0, [], []
       for s in range(steps):
           if scaler: scale = scaler.scale
           grads, loss = p4_grads(params, dt, scale)
           found_inf = not all(bool(torch.isfinite(g).all()) for g in grads)
           if scaler: scaler.update(found_inf); scales.append(scale)
           if found_inf: skipped += 1; continue
           if dt is not None and s % probe_every == 0:
               g32, _ = p4_grads(params, None, 1.0)
               nz = sum(int((g != 0).sum()) for g in g32)
               flushed.append(sum(int(((a != 0) & (b == 0)).sum()) for a, b in zip(g32, grads)) / nz)
           for p, g in zip(params, grads): p.grad = g
           opt.step()
           if s % 20 == 0: curve.append(p4_eval(params))
       return {"loss": p4_eval(params), "curve": curve, "skipped": skipped, "scales": scales, "flushed": float(np.mean(flushed)) if flushed else 0.0}

   P4_STEPS = 60 if QUICK else 200
   P4_SEEDS = (SEED,) if QUICK else (SEED, SEED + 1, SEED + 2)        # three different starting weights for every setting
   P4_SETTINGS = (("float32", None, "none"), ("float16, no scaling", torch.float16, "none"), ("float16, fixed scale 1024", torch.float16, "fixed"),
                  ("float16, dynamic scale", torch.float16, "dynamic"), ("bfloat16, no scaling", torch.bfloat16, "none"))
   p4_all = {name: [p4_train(dt, mode, P4_STEPS, seed=sd) for sd in P4_SEEDS] for name, dt, mode in P4_SETTINGS}
   p4_runs = {name: rs[0] for name, rs in p4_all.items()}              # the first seed, for the plot
   p4_summary = {name: {"loss": float(np.mean([r["loss"] for r in rs])), "lo": min(r["loss"] for r in rs), "hi": max(r["loss"] for r in rs),
                        "skipped": float(np.mean([r["skipped"] for r in rs])), "flushed": float(np.mean([r["flushed"] for r in rs]))} for name, rs in p4_all.items()}
   print(f"{'run (mean of ' + str(len(P4_SEEDS)) + ' starts)':28s} {'final loss':>11} {'range':>19} {'skipped steps':>14} {'gradients flushed to 0':>23}")
   for name, r in p4_summary.items():
       print(f"{name:28s} {r['loss']:11.5f} {r['lo']:9.5f}-{r['hi']:.5f} {r['skipped']:14.1f} {r['flushed']:23.2%}")
   p4_dyn = p4_runs["float16, dynamic scale"]
   print("dynamic scale at steps 0, 1, 2, 3, 30, 60, last (first start):", [p4_dyn["scales"][min(i, len(p4_dyn["scales"]) - 1)] for i in (0, 1, 2, 3, 30, 60, len(p4_dyn["scales"]) - 1)])
   fig, ax = plt.subplots(figsize=(6, 3.4))
   for name, r in p4_runs.items(): ax.plot(np.arange(len(r["curve"])) * 20, r["curve"], label=name)
   ax.set_yscale("log"); ax.set_xlabel("optimiser step"); ax.set_ylabel("loss (float32 evaluation)"); ax.set_title("mixed-precision training, emulated 16 bits"); ax.legend(fontsize=7); plt.tight_layout(); plt.show()
   ```

   The block prints (your numbers must match):

   ```text
   run (mean of 3 starts)        final loss               range  skipped steps  gradients flushed to 0
   float32                          0.00719   0.00585-0.00916            0.0                   0.00%
   float16, no scaling              0.00773   0.00586-0.00898            0.0                   6.20%
   float16, fixed scale 1024        0.00717   0.00595-0.00902            0.0                   0.04%
   float16, dynamic scale           0.00732   0.00586-0.00920            1.0                   0.01%
   bfloat16, no scaling             0.00770   0.00686-0.00929            0.0                   0.10%
   dynamic scale at steps 0, 1, 2, 3, 30, 60, last (first start): [65536.0, 65536.0, 65536.0, 65536.0, 131072.0, 262144.0, 4194304.0]
   ```

   Read this table honestly. **The final loss is almost the same in every setting**: the means lie within 8% of each other, and the range over different starts (for float32: 0.00585 to 0.00916) is as wide as the differences between settings. On this small problem, with Adam, loss scaling did **not** improve the final loss. What it changed is the gradients: without scaling **6.2%** of the non-zero gradient entries were flushed to zero (more than at the start, because the errors, and so the gradients, shrink as the network learns), with a fixed scale of 1,024 only 0.04% and with the dynamic scaler 0.01%. Adam divides each gradient by its own running size, which hides much of the damage here; with plain SGD, with larger networks or with smaller losses the zeros would matter more, and the usual practice is to scale. The dynamic scaler skipped 1 step(s) on average and then raised its scale, from 65,536 to 4,194,304 in the first run, because the gradients kept shrinking. **bfloat16** without any scaling trained as well as the others (mean final loss 0.00770 against 0.00719 for float32), with only 0.10% flushed: that is why modern training prefers it when the hardware has it.

8. **Step 4.8. The real thing on this computer: autocast.** Emulation told us about accuracy. Now we measure **speed and memory** with real 16-bit arithmetic. `torch.autocast` runs the matrix products and the linear layers in 16 bits and keeps the sensitive operations (sums, normalisation, softmax, losses) in 32 bits. The model is a small Transformer decoder of the Part 1 type (4 layers, d = 256, 4,000 tokens in the vocabulary, 5.2 M parameters). The cell (1) measures the **activation memory** with forward hooks that add up the bytes of every layer's output, in float32 and under autocast; (2) times one complete training step (forward, backward, AdamW) in float32 and with autocast bfloat16; and (3) times a plain 512 × 512 matrix product in float32, bfloat16 and float16. The Setup cell already measured the bfloat16 ratio of this computer: it was 232 times slower than float32 when the notebook started.

   ```python
   # Real autocast: the speed and the activation memory of a small Transformer decoder (the Part 1 style model) in float32 and under torch.autocast.
   class P4Tiny(nn.Module):
       def __init__(self, vocab=4000, d=256, layers=4, heads=4, T=64):
           super().__init__()
           self.emb = nn.Embedding(vocab, d); self.pos = nn.Parameter(torch.zeros(T, d))
           self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu") for _ in range(layers)])
           self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)
           self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))
       def forward(self, idx):
           n = idx.shape[1]
           h = self.emb(idx) + self.pos[:n]
           for b in self.blocks: h = b(h, src_mask=self.mask[:n, :n])
           return self.head(self.norm(h))

   def p4_activation_bytes(model, idx, ctx=None):
       """Sum of the bytes of the output of every leaf module in one forward pass (a forward hook on each)."""
       total = [0]; hooks = []
       def hook(m, i, o):
           for t in (o if isinstance(o, tuple) else (o,)):
               if torch.is_tensor(t): total[0] += t.numel() * t.element_size()
       for m in model.modules():
           if len(list(m.children())) == 0: hooks.append(m.register_forward_hook(hook))
       try:
           with (ctx() if ctx else contextlib_nullcontext()): model(idx)
       finally:
           for h in hooks: h.remove()
       return total[0]
   import contextlib
   contextlib_nullcontext = contextlib.nullcontext

   torch.manual_seed(SEED)
   p4_model = P4Tiny().to(device)
   p4_idx = torch.randint(0, 4000, (8, 64), device=device)
   p4_tgt = torch.randint(0, 4000, (8, 64), device=device)
   p4_opt = torch.optim.AdamW(p4_model.parameters(), lr=1e-4)
   print(f"model: {n_params(p4_model) / 1e6:.2f} M parameters, {human_bytes(param_bytes(p4_model))} in float32")

   def p4_step(autocast_dtype=None):
       ctx = torch.autocast(device, dtype=autocast_dtype) if autocast_dtype is not None else contextlib.nullcontext()
       with ctx: loss = F.cross_entropy(p4_model(p4_idx).float().view(-1, 4000), p4_tgt.view(-1))
       p4_opt.zero_grad(); loss.backward(); p4_opt.step()

   p4_act32 = p4_activation_bytes(p4_model, p4_idx)
   p4_act16 = p4_activation_bytes(p4_model, p4_idx, lambda: torch.autocast(device, dtype=torch.bfloat16))
   print(f"activation bytes of one forward pass (batch 8, 64 tokens): float32 {human_bytes(p4_act32)}, under autocast bfloat16 {human_bytes(p4_act16)} -> {p4_act16 / p4_act32:.2f} of float32")
   p4_reps = 1
   p4_ms32 = time_ms(lambda: p4_step(None), repeat=p4_reps)
   p4_msbf = time_ms(lambda: p4_step(torch.bfloat16), repeat=p4_reps)
   print(f"one training step on {device}: float32 {p4_ms32:.0f} ms, autocast bfloat16 {p4_msbf:.0f} ms -> {p4_msbf / p4_ms32:.2f}x {'slower' if p4_msbf > p4_ms32 else 'faster'}")
   p4_mm32 = time_ms(lambda: p4_a @ p4_b, repeat=3)
   p4_mmbf = time_ms(lambda: p4_a.bfloat16() @ p4_b.bfloat16(), repeat=2)
   p4_mmf16 = time_ms(lambda: p4_a.half() @ p4_b.half(), repeat=2)
   print(f"512 x 512 matrix product: float32 {p4_mm32:.2f} ms, bfloat16 {p4_mmbf:.2f} ms ({p4_mmbf / p4_mm32:.1f}x), float16 {p4_mmf16:.2f} ms ({p4_mmf16 / p4_mm32:.1f}x); the Setup measured the bfloat16 ratio {CPU_BF16_RATIO:.1f}x")
   ```

   The block prints (your numbers must match):

   ```text
   model: 5.23 M parameters, 20.91 MB in float32
   activation bytes of one forward pass (batch 8, 64 tokens): float32 36.50 MB, under autocast bfloat16 20.87 MB -> 0.57 of float32
   one training step on cpu: float32 112 ms, autocast bfloat16 4337 ms -> 38.87x slower
   512 x 512 matrix product: float32 1.34 ms, bfloat16 240.85 ms (180.2x), float16 288.11 ms (215.5x); the Setup measured the bfloat16 ratio 232.5x
   ```

   Two results, one good and one bad. **Memory:** under autocast the layer outputs total 20.87 MB instead of 36.50 MB, **0.57** of the float32 size: not exactly one half, because some operations (the layer norms, for example) stay in float32. **Speed:** on the machine that ran this lab the training step took 112 ms in float32 and **4337 ms** with autocast, **39 times slower**, and the plain bfloat16 matrix product was 180 times slower than float32 (float16 216 times). This processor has no fast 16-bit arithmetic: the 16-bit product is carried out by slow fallback code. It is a real result, and exactly what the lesson warned about: **mixed precision is a speed-up only on hardware built for it** (GPUs with Tensor Cores, and CPUs with native bfloat16 instructions). Your own computer may give a different ratio; if it is below 1, it has such hardware. The memory saving, in contrast, does not depend on the machine.

9. **Step 4.9. The same on a GPU, if you have one.** If the notebook finds a GPU (`HAS_CUDA`), this cell times a 4,096 × 4,096 matrix product and a training step of the same model in float32, float16 and bfloat16 under `torch.autocast('cuda')`, and reads the **peak memory** with `torch.cuda.max_memory_allocated()`. Without a GPU the cell only prints a note. **The author of this lab could not run this cell on a GPU**, so there are no GPU numbers in this text: read the output of your own run, and expect the 16-bit matrix product to be several times faster than float32 on a GPU with Tensor Cores, and the peak memory of a training step to fall (activations halve; the master weights and the optimiser state stay in 32 bits).

   ```python
   # The same on a GPU, if there is one (this cell is skipped without a GPU; the author could not test it on a GPU, so read its output carefully).
   p4_gpu = {}
   if HAS_CUDA:
       torch.cuda.reset_peak_memory_stats()
       a = torch.randn(4096, 4096, device=device); b = torch.randn(4096, 4096, device=device)
       for name, dt in (("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
           aa, bb = a.to(dt), b.to(dt)
           p4_gpu[name] = {"matmul_ms": time_ms(lambda: aa @ bb, repeat=20, warmup=3)}
       for name, dt in (("float32", None), ("float16", torch.float16), ("bfloat16", torch.bfloat16)):
           torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
           p4_gpu[name]["step_ms"] = time_ms(lambda: p4_step(dt), repeat=10, warmup=2)
           p4_gpu[name]["peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
       print(f"{'format':9s} {'4096 matmul (ms)':>17} {'training step (ms)':>19} {'peak memory (MB)':>17}")
       for name, r in p4_gpu.items(): print(f"{name:9s} {r['matmul_ms']:17.2f} {r['step_ms']:19.1f} {r['peak_mb']:17.0f}")
   else:
       print("No GPU here: the GPU measurement is skipped. On a GPU with Tensor Cores the 16-bit matrix product is usually several times faster than float32, and the peak memory of a training step falls; run this cell on one to see your own numbers.")
   ```

   The block prints (your numbers must match):

   ```text
   No GPU here: the GPU measurement is skipped. On a GPU with Tensor Cores the 16-bit matrix product is usually several times faster than float32, and the peak memory of a training step falls; run this cell on one to see your own numbers.
   ```

10. **Step 4.10. The honest summary.** One table with what 16-bit arithmetic did on this machine: the memory it saved, the speed it cost (or gave), and the accuracy.

    ```python
    # The honest summary of what 16-bit arithmetic did, on THIS machine.
    print(f"{'what we measured':46s} {'result':>14}")
    rows = [("activation bytes under autocast / float32", f"{p4_act16 / p4_act32:.2f}"),
            ("training step time, autocast bfloat16 / float32", f"{p4_msbf / p4_ms32:.2f}x"),
            ("512 matrix product time, bfloat16 / float32", f"{p4_mmbf / p4_mm32:.1f}x"),
            ("emulated float16 training: final loss / float32 loss", f"{p4_summary['float16, dynamic scale']['loss'] / p4_summary['float32']['loss']:.3f}"),
            ("float16 without scaling: gradients flushed to zero", f"{p4_summary['float16, no scaling']['flushed']:.2%}"),
            ("bfloat16 without scaling: final loss / float32 loss", f"{p4_summary['bfloat16, no scaling']['loss'] / p4_summary['float32']['loss']:.3f}")]
    for k, v in rows: print(f"{k:46s} {v:>14}")
    RESULTS["p4_activation_ratio"] = round(p4_act16 / p4_act32, 2)
    RESULTS["p4_autocast_time_ratio"] = round(p4_msbf / p4_ms32, 2)
    RESULTS["p4_bf16_matmul_ratio"] = round(p4_mmbf / p4_mm32, 1)
    RESULTS["p4_flushed_noscale"] = round(p4_summary["float16, no scaling"]["flushed"], 4)
    RESULTS["p4_loss_ratio_dynamic"] = round(p4_summary["float16, dynamic scale"]["loss"] / p4_summary["float32"]["loss"], 3)
    ```

    The block prints (your numbers must match):

    ```text
    what we measured                                       result
    activation bytes under autocast / float32                0.57
    training step time, autocast bfloat16 / float32         38.87x
    512 matrix product time, bfloat16 / float32            180.2x
    emulated float16 training: final loss / float32 loss          1.019
    float16 without scaling: gradients flushed to zero          6.20%
    bfloat16 without scaling: final loss / float32 loss          1.071
    ```

    The summary: mixed precision **saved memory** (the activations fell to 0.57 of their size), **cost speed on this CPU** (about 39 times slower), and **did not hurt accuracy** in the emulated runs (final loss 1.019 times the float32 loss with dynamic scaling, within the spread between starts). Which of the three matters depends on your hardware, and that is the whole point of measuring before you decide. Part 5 attacks the same memory problem from another side: not 16 bits but 8, 4 or fewer, for the weights only.

11. **Step 4.11. Compare with your Imagine guesses.** The next cell prints the true values next to your guesses.

    ```python
    print(f"(1) largest float16: {p4_info['float16'][0]:.0f}   your guess: {my_guess_largest}")
    print(f"(2) float16(70000): {p4_over['float16']}   your guess: {my_guess_70000}")
    print(f"(3) float16 running total of 100,000 x 0.0001: {float(p4_run16):.4f} (true 10)   your guess: {my_guess_sum}")
    print(f"(4) float16(1e-8): {float(round_to_format(1e-8, 5, 10))}   your guess: {my_guess_tiny}")
    ```

    The block prints (your numbers must match):

    ```text
    (1) largest float16: 65504   your guess: None
    (2) float16(70000): inf   your guess: None
    (3) float16 running total of 100,000 x 0.0001: 0.2500 (true 10)   your guess: None
    (4) float16(1e-8): 0.0   your guess: None
    ```

    Your numbers against ours: the largest float16 is **65,504**; 70,000 becomes **infinity**; the float16 running total sticks at **0.25** instead of 10 (most people guess "about 10, a little off": the failure is total, not small); and the gradient 1e-8 becomes **zero**. Notice that the first three are about *range and gaps*, and the fourth is the reason why loss scaling exists.

### Part 5. Quantization (30 min)

*Lesson section: "Quantization".* A weight in a trained model is a 32-bit number, but the model does not need that many digits: if every weight is stored as a small **integer** (8 bits, 4 bits) and one **scale** number per group of weights, the model is four to eight times smaller and, because decoding is limited by how fast the weights can be read from memory, it can also run faster. The price is a rounding error. ==In this part you write symmetric quantization yourself, measure how the error grows when the bits shrink and when a few outlier weights stretch the scale, quantize a trained classifier, and then use PyTorch's own dynamic quantization to measure the real size and the real speed on this machine.== Everything is measured on small models that run in seconds; the memory arithmetic for a 7-billion and a 70-billion-parameter model is exact and does not depend on the machine.

**Imagine first.** You keep every weight of a small trained classifier in only **8 bits**, then in **4 bits**, then in **2 bits** (one scale for each row of the weight matrix). Before you run anything, guess how the accuracy changes. Then you ask PyTorch to quantize a wide network (1024 → 4096 → 1024) to 8-bit integers, and you guess how much **smaller** the file becomes and how much **faster** the forward pass runs at batch size 1 on this machine. Write your guesses in the starter cell.

```python
# Imagine first: what does it cost to throw digits away?
guess_acc_8bit = None        # YOUR GUESS: test accuracy with 8-bit weights, in percent of the full-precision accuracy (100 = no change)
guess_acc_4bit = None        # YOUR GUESS: the same for 4-bit weights, in percent of the full-precision accuracy
guess_acc_2bit = None        # YOUR GUESS: the same for 2-bit weights, in percent of the full-precision accuracy
guess_size_ratio = None      # YOUR GUESS: how many times smaller is the state dict of the 8-bit dynamic-quantized wide network? (a number)
guess_speedup_b1 = None      # YOUR GUESS: how many times faster is its forward pass at batch size 1 on this machine? (a number, 1 = same, 0.5 = twice as slow)
```

```python
# Helpers for Part 5 (all names start with p5_ or P5_).
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
if "RESULTS" not in globals():
    RESULTS = {}      # a place to keep your key numbers; the last part of the lab prints them together
import copy
warnings.filterwarnings("ignore", message=".*quantiz.*")      # quantize_dynamic prints a deprecation notice in newer PyTorch (see Step 5.5)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def p5_dequantize(q, scales):
    """Turn integers and scales back into approximate weights: w is about q times scale. `scales` has one column per group in each row."""
    R, C = q.shape; ns = scales.shape[1]; g = C // ns
    return (q.reshape(R, ns, g) * scales[:, :, None]).reshape(R, C)
def p5_mse(a, b):
    """Mean squared difference between two arrays."""
    return float(np.mean((a - b) ** 2))
def p5_rel_err(a, b):
    """Relative error of b with respect to a: the size of the difference divided by the size of a (a tensor)."""
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(a))
print("helpers ready")
```

It prints:

```text
helpers ready
```

1. **Step 5.1. Symmetric quantization from scratch.** **Symmetric quantization** with `b` bits uses the integers from −qmax to +qmax, where `qmax = 2^(b−1) − 1` (127 for 8 bits, 7 for 4 bits, 1 for 2 bits). A **scale** says how big one integer step is: `scale = max|w| / qmax`, so that the largest weight in the group lands exactly on ±qmax. Each weight becomes `q = round(w / scale)` (an integer, clipped to the range), and we get an approximation back with `w ≈ q × scale`. The rounding error of every weight is at most half a step, `scale / 2`. A group is the set of weights that share one scale: the whole matrix (`group=None`), one row (`group="row"`) or blocks of `g` neighbours inside each row (`group=g`, for example 64).

   A tiny example by hand, 4 bits (qmax = 7), the weights `0.9, −0.31, 0.04, 0.5`: the scale is 0.9 / 7 = 0.1286, the integers are round(7, −2.41, 0.31, 3.89) = `7, −2, 0, 4`, and the approximate weights are `0.9, −0.257, 0, 0.514`. Notice that the small weight 0.04 vanished: it is smaller than half a step (0.064).

```python
def p5_quantize(w, bits, group=None):
    """Symmetric quantization: returns the integer array q and the scales (see the exercise text for the shapes)."""
    return None   # YOUR CODE: split w into groups, scale = max|group| / qmax, q = clip(round(w / scale), -qmax, qmax)
```

   A possible answer (try first, then compare):

```python
def p5_quantize(w, bits, group=None):
    qmax = 2 ** (bits - 1) - 1
    R, C = w.shape
    if group is None: blocks = w.reshape(1, 1, R * C)          # one scale for the whole matrix
    elif group == "row": blocks = w.reshape(R, 1, C)           # one scale per row
    else: blocks = w.reshape(R, C // group, group)             # one scale per block of `group` weights
    scales = np.abs(blocks).max(axis=2, keepdims=True) / qmax
    scales = np.where(scales == 0, 1.0, scales)
    q = np.clip(np.round(blocks / scales), -qmax, qmax).astype(np.int32)
    return q.reshape(R, C), scales[:, :, 0]
q, s = p5_quantize(np.array([ [0.9, -0.31, 0.04, 0.5] ]), 4)
print("integers", q.ravel().tolist(), "| scale %.4f" % s[0, 0], "| back", np.round(p5_dequantize(q, s), 3).ravel().tolist())
```

The only subtle line is the reshape: it turns the matrix into blocks of weights that share a scale (one block for the whole matrix, one per row, or several per row), so that a single `max` over the last axis finds the largest weight of every group at once. `np.round` rounds halves to the nearest even integer, which does not matter here. In a real implementation the integers are packed (two 4-bit values in one byte); we keep them in 32-bit integers because we only measure the error.

   It prints:

```text
integers [7, -2, 0, 4] | scale 0.1286 | back [0.9, -0.257, 0.0, 0.514]
```

   Running the check on this answer prints:

```text
✓ the example gives the integers 7, -2, 0, 4
✓ the integers are of integer type and in the range -7 to 7
✓ the error of every weight is at most scale / 2
✓ the scales have shapes (1, 1), (8, 1) and (8, 4)
✓ 8 bits are more exact than 4 bits
```

2. **Step 5.2. Error against bits, with and without outliers.** How much does each bit buy? We take a random weight matrix of 256 by 256 numbers from a Gaussian distribution (mean 0, standard deviation 1, like the weights of a freshly initialised layer, only larger) and quantize it with one scale for the whole matrix at 8, 6, 4, 3 and 2 bits. We measure the **mean squared error (MSE)**. A short calculation says that rounding to steps of size `scale` gives an error that is spread evenly between −scale/2 and +scale/2, whose mean square is `scale² / 12`; we print that prediction next to the measurement.

   Then the same matrix with **outliers**: we set 16 of the 65,536 weights (0.024 percent) to plus or minus 20 standard deviations. Real language models have a few such huge weights and activations, and they are the main reason why quantization is hard. Watch what one huge number does to the scale of the whole matrix.

```python
# Step 5.2. Error against bits for a Gaussian matrix and for the same matrix with 16 outliers
p5_rng = np.random.default_rng(SEED)
P5_W = p5_rng.normal(0, 1, (256, 256))
P5_WO = P5_W.copy(); p5_idx = p5_rng.choice(P5_W.size, 16, replace=False)
P5_WO.flat[p5_idx] = p5_rng.choice([-1.0, 1.0], 16) * 20 * P5_W.std()
P5_BITS = [8, 6, 4, 3, 2]
p5_rows = []
for bits in P5_BITS:
    r = []
    for W in (P5_W, P5_WO):
        q, s = p5_quantize(W, bits); r += [p5_mse(W, p5_dequantize(q, s)), float(s[0, 0] ** 2 / 12)]
    p5_rows.append(r)
print("bits | Gaussian: MSE   predicted | with outliers: MSE   predicted | MSE ratio (outliers / Gaussian)")
for bits, r in zip(P5_BITS, p5_rows):
    print("%4d | %17.2e %10.2e | %22.2e %10.2e | %10.1f" % (bits, r[0], r[1], r[2], r[3], r[2] / r[0]))
RESULTS["p5_mse_4bit_gauss"] = float("%.3g" % p5_rows[2][0]); RESULTS["p5_mse_4bit_outlier"] = float("%.3g" % p5_rows[2][2])
plt.figure(figsize=(5.2, 3.6))
plt.semilogy(P5_BITS, [r[0] for r in p5_rows], "o-", label="Gaussian weights")
plt.semilogy(P5_BITS, [r[2] for r in p5_rows], "s-", label="same, with 16 outliers")
plt.gca().invert_xaxis(); plt.xlabel("bits per weight"); plt.ylabel("mean squared error"); plt.title("One scale for the whole matrix"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

   It prints:

```text
bits | Gaussian: MSE   predicted | with outliers: MSE   predicted | MSE ratio (outliers / Gaussian)
   8 |          1.29e-04   1.30e-04 |               2.08e-03   2.08e-03 |       16.1
   6 |          2.19e-03   2.17e-03 |               3.47e-02   3.49e-02 |       15.9
   4 |          4.26e-02   4.26e-02 |               6.08e-01   6.84e-01 |       14.3
   3 |          2.30e-01   2.32e-01 |               1.00e+00   3.72e+00 |        4.3
   2 |          9.64e-01   2.09e+00 |               1.01e+00   3.35e+01 |        1.0
```

   **What you should see.**

   The table has five rows, from 8 bits down to 2 bits. For the Gaussian matrix the measured MSE is **0.000129 at 8 bits, 0.0426 at 4 bits and 0.23 at 3 bits**, and the prediction `scale² / 12` matches it to within about 1 percent for 8, 6, 4 and 3 bits. Every bit that you remove multiplies the error by about **4** (the step doubles, and the error is the square of the step): from 8 to 6 bits the MSE grows 17 times, which is 4 for each of the two bits. With the 16 outliers the error is about **15 times larger** at 8, 6 and 4 bits (2.1 × 10⁻³, 0.035 and 0.61), although the outliers are only 0.024 percent of the weights; at 3 and 2 bits the ratio shrinks because the error cannot exceed the size of the weights themselves (at 2 bits only the integers −1, 0 and +1 exist, and nearly every ordinary weight becomes 0). The figure shows two parallel lines, one about 15 times above the other.

   *What it means.* The formula `scale² / 12` is accurate while the steps are small; at 2 bits (and for the outlier matrix at 3 bits) it overestimates, because the predicted noise would be bigger than the weights. The outlier matrix is the important lesson: **the scale is set by the largest weight, so a handful of big values degrades every other weight in the group**. A 15-fold increase of the error is like losing about two bits.

3. **Step 5.3. One scale for everything, one per row, one per group.** One huge weight makes the scale large, and a large scale makes the steps large for **every** weight in the group. The cure is to give each small group its own scale, so that an outlier spoils only its own neighbours. Here is the outlier matrix at 4 bits with the whole matrix as one group, one scale per row (256 weights), and one scale per block of 128, 64, 32 and 16 weights. Smaller groups are more exact but need **more scales**, and every scale costs 16 bits (a half-precision number). So the real storage per weight is **bits + 16 / g** for groups of `g` weights.

```python
def p5_bits_per_weight(bits, group):
    """Storage per weight in bits, counting one 16-bit scale per group."""
    return None   # YOUR CODE
def p5_size_gb(params_billion, bits, group=None):
    """Size in GB of a model with params_billion billion weights stored with `bits` bits and groups of `group`."""
    return None   # YOUR CODE
```

   A possible answer (try first, then compare):

```python
def p5_bits_per_weight(bits, group):
    return bits if group is None else bits + 16 / group
def p5_size_gb(params_billion, bits, group=None):
    return params_billion * p5_bits_per_weight(bits, group) / 8
print("%.4f bits, %.2f GB, %.2f GB" % (p5_bits_per_weight(4, 64), p5_size_gb(7, 4, 64), p5_size_gb(70, 4, 64)))
```

A billion weights at 8 bits is one byte each, so 1 GB: that is why `params_billion * bits / 8` gives gigabytes directly. The 16 / group term shrinks as the groups grow: 0.25 bits for g = 64, 0.125 for g = 128, 0.0625 for a whole row of 256. A scale is stored once per group, whatever the number of bits.

   It prints:

```text
4.2500 bits, 3.72 GB, 37.19 GB
```

   Running the check on this answer prints:

```text
✓ 4 bits with groups of 64 cost 4.25 bits per weight
✓ 4 bits with groups of 128 cost 4.125 bits per weight
✓ without groups the cost is just the bits
✓ 7 billion weights at 4 bits, groups of 64: 3.72 GB
```

```python
# Step 5.3. The outlier matrix at 4 bits with different groups
P5_SCHEMES = [("whole matrix", None, 256 * 256), ("one scale per row", "row", 256), ("blocks of 128", 128, 128), ("blocks of 64", 64, 64), ("blocks of 32", 32, 32), ("blocks of 16", 16, 16)]
p5_sch = []
print("scheme                | scales | bits per weight | MSE at 4 bits | MSE of the Gaussian matrix, same scheme")
for name, grp, gsize in P5_SCHEMES:
    q, s = p5_quantize(P5_WO, 4, grp); mse_o = p5_mse(P5_WO, p5_dequantize(q, s))
    q2, s2 = p5_quantize(P5_W, 4, grp); mse_g = p5_mse(P5_W, p5_dequantize(q2, s2))
    bpw = p5_bits_per_weight(4, gsize); p5_sch.append((name, s.size, bpw, mse_o, mse_g))
    print("%-21s | %6d | %15.4f | %13.3f | %10.3f" % (name, s.size, bpw, mse_o, mse_g))
    assert abs(bpw - (4 + 16 * s.size / P5_WO.size)) < 1e-9          # the formula bits + 16 / g counts the scales exactly
RESULTS["p5_group64_gain"] = round(p5_sch[0][3] / p5_sch[3][3], 1)
print("Groups of 64 cut the error of the outlier matrix by a factor of %.1f compared with one scale, for %.2f extra bits per weight." % (RESULTS["p5_group64_gain"], p5_sch[3][2] - 4))
```

   It prints:

```text
scheme                | scales | bits per weight | MSE at 4 bits | MSE of the Gaussian matrix, same scheme
whole matrix          |      1 |          4.0002 |         0.608 |      0.043
one scale per row     |    256 |          4.0625 |         0.053 |      0.016
blocks of 128         |    512 |          4.1250 |         0.032 |      0.014
blocks of 64          |   1024 |          4.2500 |         0.021 |      0.012
blocks of 32          |   2048 |          4.5000 |         0.014 |      0.010
blocks of 16          |   4096 |          5.0000 |         0.009 |      0.007
Groups of 64 cut the error of the outlier matrix by a factor of 29.5 compared with one scale, for 0.25 extra bits per weight.
```

   **What you should see.**

   With one scale for the whole matrix the 4-bit error is **0.608** (the first row, 1 scale). One scale per row cuts it to **0.053**, blocks of 64 to **0.021** and blocks of 16 to **0.009**. The last line says that groups of 64 reduce the error of the outlier matrix **29.5 times** for **0.25 extra bits per weight** (4.25 instead of 4.0002). The right-hand column, the same schemes on the Gaussian matrix, shows that groups also help without outliers, but only a little (0.043 to 0.012): their real use is the protection against outliers. The assertion in the cell confirms that the formula `bits + 16 / g` counts the scales exactly. The Check lines of Exercise 5.2 print four ticks.

   *What it means.* Scales cost memory, and the cost grows quickly for small groups: blocks of 16 reach 5 bits per weight, so a "4-bit" model is really a 5-bit model. Groups of 64 or 128 are the usual compromise: about 4.1 to 4.25 bits per weight, and most of the benefit.

4. **Step 5.4. Quantize a trained classifier.** Error in the weights is only interesting if it changes what the model **does**. We train a small classifier on a made-up task (6 classes, 24 numbers per example, overlapping groups, so that the accuracy is high but not perfect) with two hidden layers (24 → 128 → 64 → 6), for 100 steps of Adam: a second or two. Then we replace every weight matrix by its quantized-and-dequantized version (one scale per row, your `p5_quantize`) at 8, 6, 4, 3 and 2 bits, keep the biases as they are, and measure the **test accuracy** and the **relative output error** (the size of the change in the output numbers divided by the size of the output). This is called **simulated** or *fake* quantization: the weights are rounded, but the arithmetic is still in 32 bits, which is exactly what we need to measure the damage of the rounding.

```python
# Step 5.4. Train a small classifier, then quantize its weights to fewer and fewer bits
P5_K, P5_D = 6, 24
p5_rng = np.random.default_rng(SEED); p5_means = p5_rng.normal(0, 1.0, (P5_K, P5_D))
def p5_data(n):
    y = p5_rng.integers(0, P5_K, n); X = p5_means[y] + p5_rng.normal(0, 1.6, (n, P5_D))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)
P5_Xtr, P5_ytr = p5_data(3000); P5_Xte, P5_yte = p5_data(2000)
torch.manual_seed(SEED)
P5_NET = nn.Sequential(nn.Linear(P5_D, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, P5_K))
p5_opt = torch.optim.Adam(P5_NET.parameters(), lr=1e-2, weight_decay=1e-3)
for _ in range(100):
    p5_opt.zero_grad(); F.cross_entropy(P5_NET(P5_Xtr), P5_ytr).backward(); p5_opt.step()
def p5_acc(model, X=None, y=None):
    """Test accuracy (percent) of a classifier."""
    X = P5_Xte if X is None else X; y = P5_yte if y is None else y
    with torch.no_grad(): return float((model(X).argmax(1) == y).float().mean() * 100)
def p5_fake_quant(model, bits, group="row", only=None):
    """A copy of the model whose Linear weights are replaced by quantized-then-dequantized weights. `only` = list of layer positions to quantize (default all)."""
    m = copy.deepcopy(model)
    for i, layer in enumerate(m):
        if isinstance(layer, nn.Linear) and (only is None or i in only):
            q, s = p5_quantize(layer.weight.detach().numpy().astype(np.float64), bits, group)
            layer.weight.data = torch.tensor(p5_dequantize(q, s), dtype=torch.float32)
    return m
P5_BASE_ACC = p5_acc(P5_NET)
with torch.no_grad(): P5_REF_OUT = P5_NET(P5_Xte)
print("full precision: train accuracy %.1f %%, test accuracy %.1f %%, parameters %d, float32 size %s" % (p5_acc(P5_NET, P5_Xtr, P5_ytr), P5_BASE_ACC, n_params(P5_NET), human_bytes(param_bytes(P5_NET))))
p5_curve = []
print("bits | test accuracy | share of full-precision accuracy | relative output error")
for bits in P5_BITS:
    m = p5_fake_quant(P5_NET, bits)
    with torch.no_grad(): err = p5_rel_err(P5_REF_OUT, m(P5_Xte))
    a = p5_acc(m); p5_curve.append((bits, a, err))
    print("%4d | %12.1f %% | %30.1f %% | %20.3f" % (bits, a, 100 * a / P5_BASE_ACC, err))
RESULTS["p5_acc_full"] = round(P5_BASE_ACC, 1)
for bits, a, err in p5_curve: RESULTS["p5_acc_%dbit" % bits] = round(a, 1)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
ax[0].plot([c[0] for c in p5_curve], [c[1] for c in p5_curve], "o-"); ax[0].axhline(P5_BASE_ACC, ls="--", c="gray"); ax[0].invert_xaxis(); ax[0].set_xlabel("bits per weight"); ax[0].set_ylabel("test accuracy (%)"); ax[0].set_title("Accuracy against bits")
ax[1].semilogy([c[0] for c in p5_curve], [c[2] for c in p5_curve], "o-", c="C1"); ax[1].invert_xaxis(); ax[1].set_xlabel("bits per weight"); ax[1].set_ylabel("relative output error"); ax[1].set_title("Output error against bits")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

   It prints:

```text
full precision: train accuracy 100.0 %, test accuracy 82.1 %, parameters 11846, float32 size 47.38 KB
bits | test accuracy | share of full-precision accuracy | relative output error
   8 |         82.1 % |                          100.1 % |                0.004
   6 |         81.8 % |                           99.7 % |                0.015
   4 |         81.9 % |                           99.8 % |                0.068
   3 |         79.8 % |                           97.3 % |                0.174
   2 |         63.2 % |                           77.0 % |                0.629
```

   **What you should see.**

   The classifier reaches **82.1 percent** on the test set (100 percent on the training set: with 3,000 examples and 11,846 weights it has memorised the training data, which does not matter for this experiment: we measure how much the **test** accuracy moves). With your quantizer the test accuracy is **82.1 percent at 8 bits, 81.8 at 6, 81.9 at 4, 79.8 at 3 and 63.2 at 2 bits**. The relative output error grows by a factor of 2 to 4 for every bit removed, from 0.004 at 8 bits to 0.068 at 4 bits and 0.629 at 2 bits. The figure shows an accuracy curve that is flat down to 4 bits, bends at 3 bits and **falls off a cliff at 2 bits**.

   *What it means.* The accuracy hardly moves while the output error is below a few percent: the decision of a classifier depends on which output is largest, not on its exact value, so it tolerates noise much better than the weights do. 8 and 6 bits are free here, 4 bits costs nothing measurable, and **the cliff is between 3 and 2 bits**. Do not read these numbers as universal: bigger models and harder tasks fall off earlier, and the accuracy of a trained language model must be measured on its own task.

5. **Step 5.5. The real thing: PyTorch dynamic quantization.** Now we let PyTorch do it. **Dynamic quantization** stores the weights of every `nn.Linear` as 8-bit integers (with scales) and, when the model runs, quantizes the incoming activations on the fly and multiplies integers with integers. One call does it: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`. We measure what is real: the size of the saved **state dict**, the **latency** of a forward pass at batch size 1, 8 and 64, the output error, and (for the classifier) the accuracy. We try the small classifier and a **wide** network, 1024 → 4096 → 1024 (8.4 million weights), which is large enough for the speed to matter.

   **A note on the warning.** In recent PyTorch versions `quantize_dynamic` prints a deprecation notice: the quantization tools are moving to a separate library, `torchao`. The function still works in the version that this lab uses, so we silence the notice; if a later version removes it, look in the documentation of `torchao` for the same idea.

```python
# Step 5.5. quantize_dynamic on the classifier and on a wide network
try:
    torch.backends.quantized.engine = "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines else torch.backends.quantized.engine
    p5_dyn_ok = True
except Exception as e:
    p5_dyn_ok = False; print("dynamic quantization is not available here:", e)
def p5_dyn(model):
    """An 8-bit dynamic-quantized copy of the model."""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8)
torch.manual_seed(SEED)
P5_WIDE = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024)).eval()
P5_NET.eval()
if p5_dyn_ok:
    P5_NETQ = p5_dyn(P5_NET); P5_WIDEQ = p5_dyn(P5_WIDE)
    print("engine:", torch.backends.quantized.engine)
    print("%-26s | float32 | int8 dynamic | ratio" % "state dict (MB)")
    p5_sz = {}
    for name, a, b in (("classifier (24-128-64-6)", P5_NET, P5_NETQ), ("wide (1024-4096-1024)", P5_WIDE, P5_WIDEQ)):
        p5_sz[name] = (state_mb(a), state_mb(b)); print("%-26s | %7.3f | %12.3f | %4.2fx" % (name, p5_sz[name][0], p5_sz[name][1], p5_sz[name][0] / p5_sz[name][1]))
    with torch.no_grad():
        xw = torch.randn(64, 1024); ow, oq = P5_WIDE(xw), P5_WIDEQ(xw)
        print("output error of the wide network: %.2f %% of the output size" % (100 * p5_rel_err(ow, oq)))
        print("classifier: accuracy %.1f %% -> %.1f %%, output error %.2f %%" % (P5_BASE_ACC, p5_acc(P5_NETQ), 100 * p5_rel_err(P5_REF_OUT, P5_NETQ(P5_Xte))))
    print("batch | wide float32 (ms) | wide int8 (ms) | speed-up | classifier float32 (us) | int8 (us) | speed-up")
    p5_lat = {}
    for bsz in (1, 8, 64):
        x = torch.randn(bsz, 1024); xs = P5_Xte[:bsz]
        with torch.no_grad():
            a = time_ms(lambda: P5_WIDE(x), repeat=20, warmup=3); b = time_ms(lambda: P5_WIDEQ(x), repeat=20, warmup=3)
            c = time_ms(lambda: P5_NET(xs), repeat=200, warmup=10) * 1000; d = time_ms(lambda: P5_NETQ(xs), repeat=200, warmup=10) * 1000
        p5_lat[bsz] = (a, b); print("%5d | %17.2f | %14.2f | %7.2fx | %23.1f | %9.1f | %6.2fx" % (bsz, a, b, a / b, c, d, c / d))
    RESULTS["p5_wide_size_ratio"] = round(p5_sz["wide (1024-4096-1024)"][0] / p5_sz["wide (1024-4096-1024)"][1], 2)
    RESULTS["p5_wide_speedup_b1"] = round(p5_lat[1][0] / p5_lat[1][1], 2); RESULTS["p5_wide_speedup_b64"] = round(p5_lat[64][0] / p5_lat[64][1], 2)
    assert 3.5 < RESULTS["p5_wide_size_ratio"] < 4.1       # four times fewer bytes per weight
    RESULTS["p5_dyn_err_pct"] = round(100 * p5_rel_err(ow, oq), 2)
```

   It prints:

```text
engine: fbgemm
state dict (MB)            | float32 | int8 dynamic | ratio
classifier (24-128-64-6)   |   0.050 |        0.017 | 2.93x
wide (1024-4096-1024)      |  33.577 |        8.413 | 3.99x
output error of the wide network: 2.36 % of the output size
classifier: accuracy 82.1 % -> 82.1 %, output error 1.81 %
batch | wide float32 (ms) | wide int8 (ms) | speed-up | classifier float32 (us) | int8 (us) | speed-up
    1 |              1.17 |           0.35 |    3.32x |                    54.1 |     184.4 |   0.29x
    8 |              2.52 |           0.67 |    3.74x |                    68.5 |     188.2 |   0.36x
   64 |              5.89 |           2.63 |    2.24x |                    74.7 |     190.8 |   0.39x
```

   **What you should see.**

   On the machine that ran this lab: the wide network's state dict shrinks from **33.6 MB to 8.4 MB (3.99 times smaller)**, the size of the 8-bit weights plus a few scales; the small classifier shrinks only 2.9 times, because its biases and the file overhead are a larger share. The output of the wide network changes by **2.4 percent** of its size, and the classifier keeps its accuracy (82.1 percent before and after; output error 1.8 percent). The latency of the wide network drops by a factor of about **4 to 5 at batch 1 and 8, and 2 to 3 at batch 64** (for example 1.1 ms to 0.25 ms at batch 1). The surprise is the small classifier: its int8 version is about **3 times slower** (40 microseconds against 125 microseconds at batch 1).

   *What it means.* Quantization is not magic: for a tiny network the cost of quantizing the activations on the fly and calling the integer kernels is larger than the saving, while for a wide network that reads 100 MB of weights per forward pass, reading a quarter of the bytes helps a lot. The speed-up falls at batch 64 because with many inputs the arithmetic, not the memory, starts to limit the speed: that is the same story as in Step 6.1. These timings are for this CPU (with its 8-bit integer kernels); on another CPU or on a GPU they can be very different, so measure on the hardware that you will deploy to.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

6. **Step 5.6. Which layer hurts most?** Not every layer is equally fragile. We quantize the three weight matrices of the classifier **one at a time** to 3 bits (one scale per row) and leave the other two in full precision, and compare with quantizing all three. The layer that loses most accuracy is the one that you would keep at higher precision: real quantization tools do exactly this, for instance by leaving the first and the last layer in 16 bits.

```python
# Step 5.6. Per-layer sensitivity at 3 bits
P5_LAYERS = [0, 2, 4]       # positions of the three Linear layers in the Sequential
P5_LNAME = {0: "layer 1 (24 -> 128)", 2: "layer 2 (128 -> 64)", 4: "layer 3 (64 -> 6)"}
p5_sens = {}
print("quantized layer      | weights | test accuracy | change | relative output error")
for pos in P5_LAYERS + [None]:
    m = p5_fake_quant(P5_NET, 3, "row", only=None if pos is None else [pos])
    with torch.no_grad(): err = p5_rel_err(P5_REF_OUT, m(P5_Xte))
    a = p5_acc(m); name = "all three layers" if pos is None else P5_LNAME[pos]
    nw = sum(P5_NET[i].weight.numel() for i in P5_LAYERS) if pos is None else P5_NET[pos].weight.numel()
    p5_sens[name] = (a, err); print("%-20s | %7d | %12.1f %% | %+5.1f | %.3f" % (name, nw, a, a - P5_BASE_ACC, err))
p5_worst = min((k for k in p5_sens if k != "all three layers"), key=lambda k: p5_sens[k][0])
RESULTS["p5_worst_layer"] = p5_worst
print("Most sensitive single layer at 3 bits:", p5_worst)
```

   It prints:

```text
quantized layer      | weights | test accuracy | change | relative output error
layer 1 (24 -> 128)  |    3072 |         81.7 % |  -0.4 | 0.109
layer 2 (128 -> 64)  |    8192 |         81.7 % |  -0.4 | 0.046
layer 3 (64 -> 6)    |     384 |         80.9 % |  -1.1 | 0.113
all three layers     |   11648 |         79.8 % |  -2.3 | 0.174
Most sensitive single layer at 3 bits: layer 3 (64 -> 6)
```

   **What you should see.**

   Quantizing one layer at a time to 3 bits changes the test accuracy by **−0.4 points for layer 1, −0.4 for layer 2 and −1.1 for layer 3**; all three together lose **2.3 points** (79.8 percent), a little more than the sum of the three (1.9): the errors add up. The **most sensitive single layer is the last one** (64 → 6, with only 384 weights), although it is by far the smallest: its output error is 0.113, as large as the first layer's (0.109), and more than twice that of the middle layer (0.046), which holds most of the weights.

   *What it means.* Sensitivity has nothing to do with size. The last layer produces the scores that are compared directly, so its errors are not averaged out by later layers; the biggest layer is the most robust here. Real tools exploit this: the small, sensitive layers (the first and the last, the embeddings, the normalisation layers) are kept in 16 bits, and the big, tolerant middle layers are quantized hard, which saves most of the memory for little loss.

7. **Step 5.7. Is decoding memory-bound on this machine?** The lesson says that generating text one token at a time is limited by **memory bandwidth**, not by arithmetic: to produce one token the computer must read **every weight once**, and does only about two operations per weight it reads. Then the speed limit is `tokens per second ≤ bandwidth ÷ model bytes`. Quantizing to 8 bits makes the model four times smaller, so the bound is four times higher. Let us test the claim on one big matrix, 4096 by 4096 (16.8 million weights, 67 MB in float32), the shape of a layer in a 7-billion-parameter model: a matrix-vector product (batch 1) in float32 and in 8-bit dynamic quantization, with the time per byte and the bandwidth that the timing implies. First we measure this machine's bandwidth by reading a big array (128 MB), which is larger than any cache.

```python
def p5_decode_bound(bandwidth_gb_s, model_gb):
    """Upper bound on tokens per second when every token needs one read of the whole model."""
    return None   # YOUR CODE
```

   A possible answer (try first, then compare):

```python
def p5_decode_bound(bandwidth_gb_s, model_gb):
    return bandwidth_gb_s / model_gb
print("%.1f and %.1f tokens per second" % (p5_decode_bound(200, 14), p5_decode_bound(200, 3.5)))
```

Units do the work: gigabytes per second divided by gigabytes per token gives tokens per second. The bound ignores the arithmetic, the cache and the overhead, so real systems reach a fraction of it; but it tells you which resource limits you, and why halving the bytes (a smaller number of bits) nearly doubles the speed of decoding.

   It prints:

```text
14.3 and 57.1 tokens per second
```

   Running the check on this answer prints:

```text
✓ 14 GB at 200 GB/s: 14.3 tokens per second
✓ 3.5 GB at 200 GB/s: 57.1 tokens per second
✓ twice the bandwidth gives twice the bound
```

```python
# Step 5.7. Bandwidth of this machine, then a matrix-vector product in float32 and in int8
p5_big = torch.ones(32_000_000)                                  # 128 MB of float32: larger than any cache
p5_t_read = time_ms(lambda: p5_big.sum(), repeat=10, warmup=2) / 1000
P5_BW = p5_big.numel() * 4 / p5_t_read / 1e9                     # GB/s read by a sum over the array
del p5_big
torch.manual_seed(SEED)
p5_lin = nn.Sequential(nn.Linear(4096, 4096, bias=False)).eval()
p5_linq = p5_dyn(p5_lin) if p5_dyn_ok else None
p5_bytes32 = param_bytes(p5_lin); p5_bytes8 = p5_lin[0].weight.numel()          # int8 = one byte per weight (the scales are tiny)
print("Bandwidth measured by reading a 128 MB array: %.1f GB/s" % P5_BW)
print("float32 weights: %.1f MB | int8 weights: %.1f MB" % (p5_bytes32 / 1e6, p5_bytes8 / 1e6))
print("batch | float32 (ms) | float32 GB/s read | int8 (ms) | int8 GB/s read | speed-up | bound from bandwidth: float32, int8 (matrix products per second)")
p5_mv = {}
for bsz in (1, 64):
    x = torch.randn(bsz, 4096)
    with torch.no_grad():
        a = time_ms(lambda: p5_lin(x), repeat=30, warmup=5); b = time_ms(lambda: p5_linq(x), repeat=30, warmup=5) if p5_linq is not None else float("nan")
    p5_mv[bsz] = (a, b)
    print("%5d | %12.2f | %17.1f | %9.2f | %14.1f | %7.2fx | %.0f, %.0f" % (bsz, a, p5_bytes32 / (a / 1000) / 1e9, b, p5_bytes8 / (b / 1000) / 1e9, a / b, p5_decode_bound(P5_BW, p5_bytes32 / 1e9), p5_decode_bound(P5_BW, p5_bytes8 / 1e9)))
RESULTS["p5_bandwidth_gb_s"] = round(P5_BW, 1)
RESULTS["p5_mv_speedup_b1"] = round(p5_mv[1][0] / p5_mv[1][1], 2); RESULTS["p5_mv_speedup_b64"] = round(p5_mv[64][0] / p5_mv[64][1], 2)
print("Time per megabyte read at batch 1: float32 %.3f ms, int8 %.3f ms; at batch 64: float32 %.3f ms, int8 %.3f ms." % (p5_mv[1][0] / (p5_bytes32 / 1e6), p5_mv[1][1] / (p5_bytes8 / 1e6), p5_mv[64][0] / (p5_bytes32 / 1e6), p5_mv[64][1] / (p5_bytes8 / 1e6)))
```

   It prints:

```text
Bandwidth measured by reading a 128 MB array: 32.5 GB/s
float32 weights: 67.1 MB | int8 weights: 16.8 MB
batch | float32 (ms) | float32 GB/s read | int8 (ms) | int8 GB/s read | speed-up | bound from bandwidth: float32, int8 (matrix products per second)
    1 |         2.64 |              25.4 |      0.52 |           32.5 |    5.11x | 485, 1938
   64 |        15.48 |               4.3 |      5.43 |            3.1 |    2.85x | 485, 1938
Time per megabyte read at batch 1: float32 0.039 ms, int8 0.031 ms; at batch 64: float32 0.231 ms, int8 0.324 ms.
```

   **What you should see.**

   The bandwidth probe reads about **30 GB/s** on this machine (31 to 33 in our runs). At batch 1 the float32 matrix–vector product takes about **2.3 ms**, which means that it reads the 67 MB of weights at about **29 GB/s**: **the same as the bandwidth of the machine, so this product is memory-bound** (the bound from bandwidth, about 470 products per second, is close to the measured 430). The int8 version reads a quarter of the bytes and is about **5 times faster** (0.47 ms): the speed-up is a little above the 4 that the byte count predicts, because the int8 kernel is also more efficient and a part of its 16.8 MB sits in the processor's cache, which is faster than main memory. At batch 64 the picture changes: the time per product is only 5 to 6 times larger although the work is 64 times larger, and the bytes read per second drop to about 5 GB/s: the weights are read once and **used 64 times**, so the arithmetic, not the memory, sets the time, and the speed-up of int8 over float32 is smaller (about 3 times).

   *What it means.* The claim of the lesson holds on this laptop: decoding one token at a time is limited by how fast the weights can be read, so fewer bytes means more tokens per second, and quantization helps. It also holds that **batching changes the regime**: with many sequences in the batch, every byte read is used many times and the machine becomes compute-bound. A GPU has far more memory bandwidth (and far more arithmetic) than this CPU, but the same two regimes.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

8. **Step 5.8. Weight-only 4-bit: the arithmetic for 7 B and 70 B.** Large language models are usually quantized **weight-only**: the weights are stored in 4 or 8 bits (with group scales), and are converted back to 16-bit numbers inside the matrix product, while the activations stay in 16 bits. The size arithmetic does not depend on the machine, so we can do it exactly with your function from Exercise 5.2, with the lesson's numbers as assertions, and add the decoding speed limit of Exercise 5.3 at an **assumed** bandwidth of 200 GB/s (an example, not a real device: replace it by the figure for your own hardware).

```python
# Step 5.8. Sizes of 7 B and 70 B models at 16, 8 and 4 bits, and the decoding bound
P5_ASSUMED_BW = 200.0       # GB/s, an assumption for the table: use the figure of your hardware
print("model | format                       | bits per weight | size (GB) | decoding bound at an assumed %d GB/s (tokens/s)" % P5_ASSUMED_BW)
for pb in (7, 70):
    for name, bits, g in (("16-bit", 16, None), ("8-bit", 8, None), ("4-bit, groups of 128", 4, 128), ("4-bit, groups of 64", 4, 64)):
        gb = p5_size_gb(pb, bits, g)
        print("%3d B | %-28s | %15.3f | %9.2f | %10.1f" % (pb, name, p5_bits_per_weight(bits, g), gb, p5_decode_bound(P5_ASSUMED_BW, gb)))
assert abs(p5_size_gb(7, 16) - 14) < 1e-9 and abs(p5_size_gb(7, 8) - 7) < 1e-9 and abs(p5_size_gb(7, 4) - 3.5) < 1e-9      # the lesson: 14, 7, 3.5 GB
assert abs(p5_size_gb(70, 16) - 140) < 1e-9 and abs(p5_size_gb(70, 8) - 70) < 1e-9 and abs(p5_size_gb(70, 4) - 35) < 1e-9
assert abs(p5_size_gb(7, 4, 64) - 3.71875) < 1e-9 and abs(p5_size_gb(70, 4, 64) - 37.1875) < 1e-9
RESULTS["p5_7b_int4_g64_gb"] = round(p5_size_gb(7, 4, 64), 2); RESULTS["p5_70b_int4_g64_gb"] = round(p5_size_gb(70, 4, 64), 2)
print("An 80 GB GPU holds a 70 B model at 4 bits (%.1f GB of weights) with room for the cache; at 16 bits (140 GB) it needs two GPUs for the weights alone." % p5_size_gb(70, 4, 64))
```

   It prints:

```text
model | format                       | bits per weight | size (GB) | decoding bound at an assumed 200 GB/s (tokens/s)
  7 B | 16-bit                       |          16.000 |     14.00 |       14.3
  7 B | 8-bit                        |           8.000 |      7.00 |       28.6
  7 B | 4-bit, groups of 128         |           4.125 |      3.61 |       55.4
  7 B | 4-bit, groups of 64          |           4.250 |      3.72 |       53.8
 70 B | 16-bit                       |          16.000 |    140.00 |        1.4
 70 B | 8-bit                        |           8.000 |     70.00 |        2.9
 70 B | 4-bit, groups of 128         |           4.125 |     36.09 |        5.5
 70 B | 4-bit, groups of 64          |           4.250 |     37.19 |        5.4
An 80 GB GPU holds a 70 B model at 4 bits (37.2 GB of weights) with room for the cache; at 16 bits (140 GB) it needs two GPUs for the weights alone.
```

   **What you should see.**

   The sizes are exact. A 7-billion-parameter model needs **14 GB at 16 bits, 7 GB at 8 bits, 3.5 GB at 4 bits** and **3.72 GB at 4 bits with groups of 64** (4.25 bits per weight); a 70-billion-parameter model needs **140, 70, 35 and 37.19 GB**. The assertions confirm the lesson's numbers. At an assumed 200 GB/s the decoding bound rises from 14.3 tokens per second for the 7 B model in 16 bits to 53.8 in 4 bits with groups of 64 (3.8 times: a little less than 4 because of the scales); for the 70 B model it rises from 1.4 to 5.4.

   *What it means.* A 70 B model in 16 bits needs two 80 GB GPUs just for the weights, and in 4 bits it fits on one with room for the cache. The decoding bound is the most optimistic number: it ignores the cache, the arithmetic, the overhead and the time to talk between GPUs.

9. **Step 5.9. Compare with your guesses, and the limits of what we did.** Run the cell below to put your guesses next to the measurements (if you left a guess as `None` it shows "(not written)").

```python
# Step 5.9. Imagine first against the measurements
def p5_g(x, unit=""):
    return "(not written)" if x is None else "%s%s" % (x, unit)
full = RESULTS["p5_acc_full"]
print("what you guessed -> what was measured")
print("8-bit accuracy:   %s -> %.1f %% of full-precision accuracy" % (p5_g(guess_acc_8bit, " %"), 100 * RESULTS["p5_acc_8bit"] / full))
print("4-bit accuracy:   %s -> %.1f %%" % (p5_g(guess_acc_4bit, " %"), 100 * RESULTS["p5_acc_4bit"] / full))
print("2-bit accuracy:   %s -> %.1f %%" % (p5_g(guess_acc_2bit, " %"), 100 * RESULTS["p5_acc_2bit"] / full))
print("state dict size:  %s times smaller -> %.2f times smaller" % (p5_g(guess_size_ratio), RESULTS.get("p5_wide_size_ratio", float("nan"))))
print("speed at batch 1: %s times faster -> %.2f times faster" % (p5_g(guess_speedup_b1), RESULTS.get("p5_wide_speedup_b1", float("nan"))))
```

   It prints:

```text
what you guessed -> what was measured
8-bit accuracy:   (not written) -> 100.0 % of full-precision accuracy
4-bit accuracy:   (not written) -> 99.8 %
2-bit accuracy:   (not written) -> 77.0 %
state dict size:  (not written) times smaller -> 3.99 times smaller
speed at batch 1: (not written) times faster -> 3.32 times faster
```

   **What you should see.**

   The cell compares your guesses with the numbers. For the classifier, **8-bit weights keep 100 percent of the accuracy, 4-bit weights 99.8 percent and 2-bit weights only 77 percent**. The wide network's state dict becomes **3.99 times smaller** and its forward pass at batch 1 about **4 to 5 times faster** on this machine. If you guessed that 4 bits would hurt much more and that quantization would only save memory, now you know that on a wide network it can also save time, when the model is limited by memory reads, and that the cliff is lower than most people expect.

**The limits of this part, written down.** (1) We quantized small models; large language models have **outliers** in their weights and especially in their **activations** (a few channels with values a hundred times larger than the rest), and a plain 8-bit scheme for the activations can fail on them: the method LLM.int8() (Dettmers et al., 2022) was designed around this problem, and 4-bit methods such as GPTQ (Frantar et al., 2022) choose the rounding of each weight to minimise the error of the layer's *output* instead of the error of the weight. (2) **Weight-only** quantization saves memory and memory bandwidth, but the arithmetic is still in 16 bits; **weight-and-activation** quantization (INT8 multiplied by INT8) also speeds up the arithmetic where the hardware supports it, and needs **calibration data** (a few hundred typical inputs) to choose the ranges of the activations. (3) **Post-training** quantization, which we did, rounds a trained model; **quantization-aware training** trains with the rounding inside the loop and usually loses less at 4 bits and below. (4) We measured accuracy on a toy task: **always re-measure the quality of your own model on your own task** after quantizing it, ideally including the rare inputs that matter most.

### Part 6. Serving: latency, throughput and the KV cache (30 min)

*Lesson section: "Serving a model".* A trained model is only useful when people can use it, and a server has to balance two things that pull in opposite directions: **latency** (how long one person waits) and **throughput** (how many requests per second the machine finishes, and therefore what each costs). ==In this part you measure why batching raises throughput, simulate a server with random arrivals to see the price in waiting time, write the KV cache of a Transformer decoder and check that it gives exactly the same text faster, and do the memory and cost arithmetic of serving a 7-billion-parameter model.== The measurements are on this machine; the memory sizes are counted exactly from tensors and formulas. Prices and the sizes of the "example GPU" are **assumptions** that are flagged where they appear.

**Imagine first.** A restaurant kitchen can cook one dish in 10 minutes, but the oven holds 8 dishes, and a batch of 8 also takes about 12 minutes. A model server is like that: running **one** request through the network costs nearly as much as running **several** together, because the weights are read from memory once either way. Before you run anything, guess:

1. A batch of 64 inputs against a batch of 1: how many times **longer** does the forward pass take (a number: 1 would mean no extra time, 64 would mean no gain from batching)?
2. So how many times more **requests per second** does the batch of 64 finish than the batch of 1?
3. Generating 160 tokens one by one **without** a KV cache (re-computing the whole sequence at every step), how many times **slower** is the last token than with a cache? Write your guesses in the starter cell.

```python
# Imagine first: what does batching buy, and what does a cache buy?
guess_batch64_time_ratio = None     # YOUR GUESS: time of a forward pass at batch 64 divided by the time at batch 1 (a number)
guess_throughput_gain = None        # YOUR GUESS: requests per second at batch 64 divided by requests per second at batch 1 (a number)
guess_nocache_slowdown = None       # YOUR GUESS: time of the 160th token without a cache divided by its time with a cache (a number)
```

```python
# Helpers for Part 6 (all names start with p6_ or P6_).
if "check" not in globals():
    def check(name, ok, hint=""):
        """Print a tick or a cross with a hint, so you know whether your answer is right."""
        print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
if "RESULTS" not in globals():
    RESULTS = {}      # a place to keep your key numbers; the last part of the lab prints them together

def p6_weights_gb(params_billion, bits, group=None):
    """Size in GB (powers of 1000) of the weights: bits per weight plus one 16-bit scale per group, if there are groups."""
    return params_billion * (bits if group is None else bits + 16 / group) / 8
def p6_median_ms(fn, repeat=25, warmup=3):
    """Median time of fn() in milliseconds (the median ignores the odd slow run). On a GPU it waits for the GPU."""
    for _ in range(warmup): fn()
    ts = []
    for _ in range(repeat):
        if HAS_CUDA: torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if HAS_CUDA: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))
print("helpers ready")
```

   It prints:

```text
helpers ready
```

10. **Step 6.1. Measure batching.** We time the forward pass of a wide network (1024 → 4096 → 4096 → 1024, 25 million weights, a stand-in for the feed-forward part of a Transformer) for batch sizes 1, 2, 4, 8, 16, 32 and 64. With a small batch the computer spends most of its time **reading the weights** (about 100 MB of them), and that costs the same for one input or for several; with a large batch the arithmetic starts to dominate and the time grows with the batch. A straight line describes the simplest version of the story: **time = t₀ + c × b**, where `t₀` is the fixed cost of a batch (reading the weights, starting the work) and `c` the extra cost of every additional request. One warning about this machine: for a **single** input PyTorch uses a different routine (a matrix-vector product), which on many CPUs is faster than the general matrix product used for two or more inputs. So we time all seven batch sizes but **fit the line on batch sizes 2 to 64**, and show batch 1 next to the line.

```python
def p6_fit(batch_sizes, times_ms):
    """Least-squares line time = t0 + c * b. Returns (t0, c)."""
    return None   # YOUR CODE: slope, intercept = np.polyfit(...); return the intercept first
```

   A possible answer (try first, then compare):

```python
def p6_fit(batch_sizes, times_ms):
    c, t0 = np.polyfit(batch_sizes, times_ms, 1)
    return float(t0), float(c)
print("t0 = %.1f, c = %.2f" % p6_fit([1, 2, 4, 8, 16], [12, 14, 18, 26, 42]))
```

`np.polyfit(x, y, 1)` returns the coefficients from the highest power to the lowest, so the slope `c` comes first and the intercept `t0` second. The slope is the extra time for one more request in the batch; the intercept is what a batch costs even when it is empty, which is the price that batching shares among the requests.

   It prints:

```text
t0 = 10.0, c = 2.00
```

   Running the check on this answer prints:

```text
✓ exact data t = 10 + 2 b gives t0 = 10 and c = 2
✓ noisy data: t0 is close to 5 and c is close to 0.3
```

```python
# Step 6.1. Time a forward pass at batch sizes 1 ... 64, fit t0 + c b, and plot
torch.manual_seed(SEED)
P6_MLP = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1024)).eval()
P6_BS = [1, 2, 4, 8, 16, 32, 64]
p6_ts = []
with torch.no_grad():
    for bsz in P6_BS:
        x = torch.randn(bsz, 1024); p6_ts.append(p6_median_ms(lambda: P6_MLP(x), repeat=(5 if QUICK else 25), warmup=3))
P6_T0, P6_C = p6_fit(P6_BS[1:], p6_ts[1:])          # fitted on batch sizes 2 to 64: batch 1 uses a special, faster routine (see the text)
p6_pred = np.array([P6_T0 + P6_C * b for b in P6_BS]); p6_r2 = 1 - np.sum((np.array(p6_ts[1:]) - p6_pred[1:]) ** 2) / np.sum((np.array(p6_ts[1:]) - np.mean(p6_ts[1:])) ** 2)
print("model: %d weights (%s as float32)" % (n_params(P6_MLP), human_bytes(param_bytes(P6_MLP))))
print("batch | time (ms) | requests per second | time per request (ms) | line t0 + c b (ms)")
for bsz, t, pr in zip(P6_BS, p6_ts, p6_pred):
    print("%5d | %9.2f | %19.0f | %21.3f | %7.2f" % (bsz, t, 1000 * bsz / t, t / bsz, pr))
print("fit on batch sizes 2 to 64: t0 = %.2f ms, c = %.3f ms per request, R squared = %.3f" % (P6_T0, P6_C, p6_r2))
print("batch 1 against the line: measured %.2f ms, the line says %.2f ms" % (p6_ts[0], p6_pred[0]))
print("batch 64 takes %.1f times as long as batch 1 and finishes %.1f times as many requests per second." % (p6_ts[-1] / p6_ts[0], (64 / p6_ts[-1]) / (1 / p6_ts[0])))
RESULTS["p6_t0_ms"] = round(P6_T0, 2); RESULTS["p6_c_ms"] = round(P6_C, 3)
RESULTS["p6_b64_time_ratio"] = round(p6_ts[-1] / p6_ts[0], 1); RESULTS["p6_throughput_gain_b64"] = round((64 / p6_ts[-1]) / (1 / p6_ts[0]), 1)
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
ax[0].plot(P6_BS, p6_ts, "o", label="measured"); ax[0].plot(P6_BS, p6_pred, "-", label="t0 + c b"); ax[0].set_xlabel("batch size b"); ax[0].set_ylabel("time of a batch (ms)"); ax[0].set_title("Latency of a batch"); ax[0].legend()
ax[1].plot(P6_BS, [1000 * b / t for b, t in zip(P6_BS, p6_ts)], "o-", c="C2"); ax[1].set_xlabel("batch size b"); ax[1].set_ylabel("requests per second"); ax[1].set_title("Throughput")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

   It prints:

```text
model: 25175040 weights (100.70 MB as float32)
batch | time (ms) | requests per second | time per request (ms) | line t0 + c b (ms)
    1 |      3.66 |                 273 |                 3.663 |    9.60
    2 |     10.04 |                 199 |                 5.019 |    9.76
    4 |      9.86 |                 406 |                 2.464 |   10.08
    8 |     11.08 |                 722 |                 1.385 |   10.72
   16 |     11.81 |                1355 |                 0.738 |   12.00
   32 |     14.08 |                2272 |                 0.440 |   14.56
   64 |     19.91 |                3214 |                 0.311 |   19.67
fit on batch sizes 2 to 64: t0 = 9.44 ms, c = 0.160 ms per request, R squared = 0.992
batch 1 against the line: measured 3.66 ms, the line says 9.60 ms
batch 64 takes 5.4 times as long as batch 1 and finishes 11.8 times as many requests per second.
```

   **What you should see.**

   Batching is cheap. On the machine that ran this lab the 100 MB network takes about **3.6 ms at batch 1** (the special single-row routine) and then **9.5 ms at batch 2, 4 and 8** (flat: the time is the cost of reading the weights), 12.6 at batch 16, 15 at batch 32 and **17 ms at batch 64**. The line fitted on batch sizes 2 to 64 has **t₀ of about 9.5 ms** and **c of about 0.13 ms per request**, with R² around 0.9 (a straight line is only an approximation: the real curve is flat first and then rises). Batch 64 takes about **4.7 times as long as batch 1** and finishes about **13.5 times as many requests per second** (about 3,800 against 280). The time per request falls from 3.6 ms to 0.27 ms. The two plots show a latency that rises slowly and a throughput that rises almost in proportion to the batch.

   *What it means.* A fixed cost shared among many requests is why servers batch. The price is that every request in the batch has to wait for the slowest part of the batch and, in a real server, for the batch to fill (Step 6.2). On a GPU the shape is the same with much larger `t₀`-to-`c` ratios.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

11. **Step 6.2. Simulate a server.** Requests do not arrive in a neat row. We model them as a **Poisson process**: the gaps between arrivals are random (exponential), with an average rate of λ requests per second. The server collects requests into a **batch**: it takes the first waiting request, then waits until it has `b` requests **or** a **timeout** has passed since that first request arrived (or until the server is free again, if it is busy), and serves the batch in `t₀ + c × (size of the batch)` milliseconds, using the `t₀` and `c` that you just measured. We record the latency of every request (from its arrival to the end of its batch) and report the median (**p50**) and the tail (**p95**, **p99**: the time that 95 or 99 percent of the requests beat). The lesson's formula for the average latency without a timeout is **wait + service = (b − 1) / (2λ) + t₀ + c b**: a request waits on average for half of the other `b − 1` arrivals to complete its batch.

```python
def p6_formula_latency(lam, b, t0, c):
    """Predicted average latency in ms: waiting for the batch to fill plus the service time."""
    return None   # YOUR CODE
```

   A possible answer (try first, then compare):

```python
def p6_formula_latency(lam, b, t0, c):
    return (b - 1) / (2 * lam) * 1000 + t0 + c * b
print("%.0f ms" % p6_formula_latency(50, 8, 40, 5))
```

The first term is the average wait: if requests arrive every 1 / lam seconds, the first request of a batch waits for `b - 1` arrivals and the last waits for none, so on average half of `b - 1` gaps. The second term is the service time of the batch. Batching trades the first term (which grows with `b`) for a cheaper second term per request (`t0 / b` shrinks).

   It prints:

```text
150 ms
```

   Running the check on this answer prints:

```text
✓ the lesson's example (lam 50, b 8, t0 40, c 5) gives 150 ms
✓ with b = 1 there is no waiting: the latency is t0 + c
✓ a higher arrival rate shortens the wait
```

```python
# Step 6.2. The simulator, and a table over batch sizes and arrival rates
def p6_simulate(lam, b, timeout_ms, t0, c, n=20000, seed=SEED):
    """Poisson arrivals (lam per second), a batcher (up to b requests, or the timeout after the first one), service t0 + c * size.
    Returns the latencies in ms (one per request), the sizes of the batches and the throughput (requests per second)."""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1000.0 / lam, n))          # arrival times in ms
    lat = np.empty(n); sizes = []; i = 0; free = 0.0
    while i < n:
        last = min(i + b - 1, n - 1)
        close = max(free, min(arr[last], arr[i] + timeout_ms))  # the batch closes when it is full or on timeout, but not before the server is free
        j = max(i + 1, min(int(np.searchsorted(arr, close, side="right")), i + b, n))
        free = close + t0 + c * (j - i)
        lat[i:j] = free - arr[i:j]; sizes.append(j - i); i = j
    return lat, sizes, n / free * 1000
P6_N = 4000 if QUICK else 20000
p6_cap1 = 1000 / (P6_T0 + P6_C)
P6_LAMS = [round(0.6 * p6_cap1, -1), round(1.5 * p6_cap1, -1), round(4 * p6_cap1, -1)]
P6_BSIZES = [1, 2, 4, 8, 16, 32]
p6_sim = {}
print("fit: t0 = %.2f ms, c = %.3f ms; capacity at b = 1: %.0f requests per second; no timeout (the batch waits until it is full)" % (P6_T0, P6_C, p6_cap1))
for lam in P6_LAMS:
    print("\narrival rate %d per second" % lam)
    print("  b | capacity (req/s) | stable | p50 (ms) | p95 (ms) | p99 (ms) | throughput | mean (ms) | formula mean (ms)")
    for b in P6_BSIZES:
        lat, sizes, thr = p6_simulate(lam, b, 1e12, P6_T0, P6_C, n=P6_N)
        cap = 1000 * b / (P6_T0 + P6_C * b); p6_sim[(lam, b)] = (np.percentile(lat, 50), np.percentile(lat, 95), np.percentile(lat, 99), thr, lat.mean(), cap)
        print("%3d | %16.0f | %6s | %8.1f | %8.1f | %8.1f | %10.0f | %9.1f | %9.1f" % (b, cap, "yes" if cap > lam else "NO", *p6_sim[(lam, b)][:3], thr, lat.mean(), p6_formula_latency(lam, b, P6_T0, P6_C)))
```

   It prints:

```text
fit: t0 = 9.44 ms, c = 0.160 ms; capacity at b = 1: 104 requests per second; no timeout (the batch waits until it is full)

arrival rate 60 per second
  b | capacity (req/s) | stable | p50 (ms) | p95 (ms) | p99 (ms) | throughput | mean (ms) | formula mean (ms)
  1 |              104 |    yes |     12.3 |     34.3 |     49.6 |         60 |      16.1 |       9.6
  2 |              205 |    yes |     12.2 |     48.0 |     73.5 |         60 |      18.5 |      18.1
  4 |              397 |    yes |     26.9 |     88.1 |    125.8 |         60 |      34.7 |      35.1
  8 |              746 |    yes |     61.1 |    160.1 |    212.6 |         60 |      68.7 |      69.1
 16 |             1334 |    yes |    128.1 |    295.9 |    365.0 |         60 |     136.6 |     137.0
 32 |             2199 |    yes |    263.5 |    550.6 |    656.2 |         60 |     271.3 |     272.9

arrival rate 160 per second
  b | capacity (req/s) | stable | p50 (ms) | p95 (ms) | p99 (ms) | throughput | mean (ms) | formula mean (ms)
  1 |              104 |     NO |  34221.1 |  63905.3 |  66700.8 |        104 |   33815.0 |       9.6
  2 |              205 |    yes |     18.3 |     39.8 |     53.5 |        160 |      20.3 |      12.9
  4 |              397 |    yes |     17.0 |     39.3 |     53.5 |        160 |      19.5 |      19.5
  8 |              746 |    yes |     29.6 |     66.7 |     86.4 |        160 |      32.5 |      32.6
 16 |             1334 |    yes |     55.5 |    118.5 |    144.4 |        160 |      58.7 |      58.9
 32 |             2199 |    yes |    107.9 |    215.6 |    255.2 |        160 |     110.8 |     111.4

arrival rate 420 per second
  b | capacity (req/s) | stable | p50 (ms) | p95 (ms) | p99 (ms) | throughput | mean (ms) | formula mean (ms)
  1 |              104 |     NO |  72457.4 | 137241.9 | 143060.3 |        104 |   72302.3 |       9.6
  2 |              205 |     NO |  25267.7 |  47577.6 |  49622.5 |        205 |   25113.2 |      10.9
  4 |              397 |     NO |   1643.9 |   2755.4 |   2920.9 |        397 |    1526.7 |      13.6
  8 |              746 |    yes |     18.2 |     32.1 |     39.6 |        421 |      19.2 |      19.1
 16 |             1334 |    yes |     28.6 |     52.6 |     62.4 |        421 |      29.8 |      29.9
 32 |             2199 |    yes |     50.1 |     91.1 |    106.2 |        421 |      51.2 |      51.5
```

   **What you should see.**

   Look at the three blocks, one per arrival rate. The capacity column is `1000 b / (t₀ + c b)`; "stable" means that the capacity exceeds the arrival rate. **At the lowest rate (60 per second)** every batch size is stable, but latency **grows with b**: the mean is about 16 ms for b = 1 and about 270 ms for b = 32, because a request waits for the batch to fill (the p95 is 35 ms against 550 ms). **At 160 per second, above the capacity of b = 1 (104 per second)**, the batch size 1 is not stable: its median latency is about 35 *seconds*; b = 2 to 32 are fine, and b = 4 has the lowest p95 (39 ms). **At 410 per second** b = 1, 2 and 4 are all unstable (capacities 104, 204 and 398), and b = 8 is the smallest stable batch (p95 about 32 ms). In the stable cases the simulated **mean latency agrees with the lesson's formula** to within about 1 to 2 percent (for example 32.3 against 32.4 ms for b = 8 at 160 per second); the formula is **wrong at b = 1 and b = 2 at the lowest rate** (16.3 against 9.7 ms) and wherever the load is close to the capacity, because it counts the wait for the batch to fill and the service time but **not the queue**, which appears when requests arrive while the server is busy. The throughput column equals the arrival rate when the server is stable, and equals the capacity when it is not.

   *What it means.* Batching is a trade: it buys capacity and costs waiting time, and the best batch size depends on the traffic. The formula is a good guide when the server has spare capacity and the batch is large enough, and it is silent about the queue. The p95 and p99 are 2 to 3 times the median: a service-level objective is written on the tail, not on the mean.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

12. **Step 6.3. Too much load, and a latency target.** Two questions are left. What happens when the arrival rate is **above** the capacity of the server? The queue grows without limit: every request waits a little longer than the one before. We watch the latency of requests 0 to 20,000 for `b = 1` at the middle arrival rate, where the capacity is too small. And how do you choose the batch size? A typical rule: the **largest batch whose p95 latency stays under a target** (the service-level objective, SLO), because a bigger batch means fewer machines for the same traffic.

```python
def p6_pick(rows, target_ms):
    """The largest batch size whose p95 latency is at most the target, or None."""
    return None   # YOUR CODE
```

   A possible answer (try first, then compare):

```python
def p6_pick(rows, target_ms):
    ok = [b for b, p95 in rows if p95 <= target_ms]
    return max(ok) if ok else None
print(p6_pick([(1, 500), (2, 60), (4, 45), (8, 90), (16, 300)], 100))
```

Notice that the p95 is **not** monotonic in `b`: at low load a small batch is unstable (the queue grows), a medium batch is best, and a large one makes people wait for the batch to fill. That is why the rule picks the largest batch that still meets the target and not simply the smallest p95.

   It prints:

```text
8
```

   Running the check on this answer prints:

```text
✓ the largest b under the target is chosen
✓ the target is inclusive
✓ no answer gives None
```

```python
# Step 6.3. The queue that never empties, and the choice of b under a target
lam_mid = P6_LAMS[1]
lat1, _, thr1 = p6_simulate(lam_mid, 1, 1e12, P6_T0, P6_C, n=P6_N)
print("arrival rate %d per second against a capacity of %.0f at b = 1:" % (lam_mid, p6_cap1))
for k in (0.0, 0.25, 0.5, 0.75, 1.0):
    i = min(int(k * (P6_N - 1)), P6_N - 1); print("  request %5d: latency %9.1f ms" % (i, lat1[i]))
print("throughput %.0f requests per second: it cannot exceed the capacity, so the rest waits." % thr1)
P6_TARGET = round(6 * (P6_T0 + P6_C), 0)
print("\nLatency target: p95 at most %.0f ms (six times the time of a single request)" % P6_TARGET)
for lam in P6_LAMS:
    rows = [(b, p6_sim[(lam, b)][1]) for b in P6_BSIZES]
    best_p95 = min(rows, key=lambda r: r[1])
    print("  arrival rate %4d: b with the lowest p95 = %d (%.1f ms), largest b under the target = %s" % (lam, best_p95[0], best_p95[1], p6_pick(rows, P6_TARGET)))
RESULTS["p6_queue_latency_end_ms"] = round(float(lat1[-1]), 0)
plt.figure(figsize=(5.4, 3.5)); plt.plot(lat1[::20]); plt.xlabel("request number (every 20th)"); plt.ylabel("latency (ms)"); plt.title("Arrival rate above capacity (b = 1)"); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

   It prints:

```text
arrival rate 160 per second against a capacity of 104 at b = 1:
  request     0: latency       9.6 ms
  request  4999: latency   17277.0 ms
  request  9999: latency   34219.9 ms
  request 14999: latency   50437.0 ms
  request 19999: latency   67356.6 ms
throughput 104 requests per second: it cannot exceed the capacity, so the rest waits.

Latency target: p95 at most 58 ms (six times the time of a single request)
  arrival rate   60: b with the lowest p95 = 1 (34.3 ms), largest b under the target = 2
  arrival rate  160: b with the lowest p95 = 4 (39.3 ms), largest b under the target = 4
  arrival rate  420: b with the lowest p95 = 8 (32.1 ms), largest b under the target = 16
```

   **What you should see.**

   Above the capacity the queue never empties: for b = 1 at 160 requests per second against a capacity of 104, the latency of request 0 is 9.7 ms, of request 5,000 about **17.5 seconds**, and of request 20,000 about **68 seconds**; the throughput stays at the capacity (104 per second) and the figure shows a straight, rising line. With the target "p95 at most six times the time of one request" (58 ms) the rule of Exercise 6.3 picks **b = 2 at 60 requests per second, b = 4 at 160 and b = 16 at 410**. The Check lines print three ticks.

   *What it means.* The right batch size grows with the load: at low traffic a small batch is best (nobody waits for others), at high traffic you need a larger batch to keep up, and the SLO caps it from above. In practice servers add a **timeout** so that a lonely request is not held back for a batch that never fills, and measure the p95 under real traffic.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

13. **Step 6.4. The KV cache, written by hand.** Generating text means running the decoder again for every new token. Without a cache, each step **recomputes the keys and values of all the earlier tokens**, which are exactly the same as in the previous step: the cost of step `n` grows with `n`. With a **KV cache** you keep the keys and values of the past in memory and compute only the new token's row: the projections cost the same at every step, and only the attention itself (the new query against `n` stored keys) still reads `n` entries. The price is **memory**: for every layer, every position and every sequence in the batch you store one key vector and one value vector of size `d`.

   Below is a tiny decoder written from scratch (2 layers, 8 heads, `d` = 256, a vocabulary of 200, random weights: it produces gibberish but it is a real, causal, multi-head Transformer decoder), in two versions: `p6_generate_nocache` recomputes everything at each step, and `p6_generate_cache` keeps the keys and values. We generate 160 tokens with both, assert that **the tokens are identical and the scores agree**, and time every step.

```python
def p6_kv_bytes(layers, n, d, bytes_per_value, batch):
    """Bytes of the KV cache: keys and values, for every layer, position and sequence."""
    return None   # YOUR CODE
```

   A possible answer (try first, then compare):

```python
def p6_kv_bytes(layers, n, d, bytes_per_value, batch):
    return 2 * layers * n * d * bytes_per_value * batch
print("%.2f GB" % (p6_kv_bytes(32, 4096, 4096, 2, 8) / 1e9))
```

Per token, per layer, you keep a key and a value of `d` numbers each; multiplied by the layers, the tokens and the sequences, with the bytes of each number, that is the whole cache. At 4,096 tokens and a batch of 8, the cache of this model (17.2 GB) is already larger than the 14 GB of its 16-bit weights. Models with grouped-query attention store fewer key and value heads, which divides the formula by a constant; the formula here is the plain multi-head case.

   It prints:

```text
17.18 GB
```

   Running the check on this answer prints:

```text
✓ the lesson's example needs 17.2 GB
✓ the cache of 1 sequence of 1 token in 1 layer, d = 10, 4 bytes: 80 bytes
✓ doubling the context doubles the cache
```

```python
# Step 6.4. A tiny decoder with and without a KV cache: same tokens, different cost
P6_V, P6_D, P6_H, P6_L = 200, 256, 8, 2
_g = torch.Generator().manual_seed(SEED); _s = P6_D ** -0.5
P6_EMB = torch.randn(P6_V, P6_D, generator=_g); P6_OUT = torch.randn(P6_V, P6_D, generator=_g)      # separate input and output tables
P6_W = [dict(q=torch.randn(P6_D, P6_D, generator=_g) * _s, k=torch.randn(P6_D, P6_D, generator=_g) * _s, v=torch.randn(P6_D, P6_D, generator=_g) * _s,
             o=torch.randn(P6_D, P6_D, generator=_g) * _s, f1=torch.randn(P6_D, 4 * P6_D, generator=_g) * _s, f2=torch.randn(4 * P6_D, P6_D, generator=_g) * (4 * P6_D) ** -0.5) for _ in range(P6_L)]
_pos = torch.arange(1024)[:, None] / (10000 ** (torch.arange(0, P6_D, 2)[None, :] / P6_D))
P6_POS = torch.zeros(1024, P6_D); P6_POS[:, 0::2] = torch.sin(_pos); P6_POS[:, 1::2] = torch.cos(_pos)
def p6_norm(x): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
def p6_heads(x): return x.reshape(x.shape[0], x.shape[1], P6_H, P6_D // P6_H).transpose(1, 2)        # (batch, heads, tokens, d / heads)
def p6_attend(q, K, V, causal):
    s = q @ K.transpose(-1, -2) / math.sqrt(P6_D // P6_H)
    if causal: s = s.masked_fill(torch.triu(torch.ones(s.shape[-2], s.shape[-1], dtype=torch.bool), 1), float("-inf"))
    return (torch.softmax(s, -1) @ V).transpose(1, 2).reshape(q.shape[0], q.shape[2], P6_D)
def p6_logits_full(tokens):
    """No cache: run all the tokens through the decoder and return the scores for the last position."""
    n_tok = tokens.shape[1]; x = P6_EMB[tokens] + P6_POS[:n_tok]
    for w in P6_W:
        h = p6_norm(x); x = x + p6_attend(p6_heads(h @ w["q"]), p6_heads(h @ w["k"]), p6_heads(h @ w["v"]), True) @ w["o"]
        x = x + torch.relu(p6_norm(x) @ w["f1"]) @ w["f2"]
    return p6_norm(x)[:, -1] @ P6_OUT.T
def p6_logits_step(tok, t, cache):
    """With a cache: process one new token at position t, append its key and value to the cache, return its scores."""
    x = (P6_EMB[tok] + P6_POS[t])[:, None, :]
    for l, w in enumerate(P6_W):
        h = p6_norm(x); k, v = p6_heads(h @ w["k"]), p6_heads(h @ w["v"])
        cache[l] = (k, v) if cache[l] is None else (torch.cat([cache[l][0], k], 2), torch.cat([cache[l][1], v], 2))
        x = x + p6_attend(p6_heads(h @ w["q"]), cache[l][0], cache[l][1], False) @ w["o"]
        x = x + torch.relu(p6_norm(x) @ w["f1"]) @ w["f2"]
    return p6_norm(x)[:, 0] @ P6_OUT.T
def p6_generate_nocache(prompt, n_new):
    toks = prompt.clone(); times = []; lg = []
    for _ in range(n_new):
        t = time.perf_counter(); s = p6_logits_full(toks); times.append((time.perf_counter() - t) * 1000)
        lg.append(s); toks = torch.cat([toks, s.argmax(-1, keepdim=True)], 1)
    return toks, times, lg
def p6_generate_cache(prompt, n_new):
    cache = [None] * P6_L; toks = prompt.clone(); times = []; lg = []
    for t in range(prompt.shape[1] - 1): p6_logits_step(prompt[:, t], t, cache)          # fill the cache with the prompt
    nxt = prompt[:, -1]
    for _ in range(n_new):
        t0_ = time.perf_counter(); s = p6_logits_step(nxt, toks.shape[1] - 1, cache); times.append((time.perf_counter() - t0_) * 1000)
        lg.append(s); nxt = s.argmax(-1); toks = torch.cat((toks, nxt.unsqueeze(1)), 1)
    return toks, times, lg, cache
P6_NNEW = 40 if QUICK else 160
P6_PROMPT = torch.tensor([ [5, 17, 42, 8] ])
with torch.no_grad():
    p6_ta, p6_tn, p6_ln = p6_generate_nocache(P6_PROMPT, P6_NNEW)
    p6_tb, p6_tc, p6_lc, p6_cache = p6_generate_cache(P6_PROMPT, P6_NNEW)
p6_maxdiff = max(float((a - b).abs().max()) for a, b in zip(p6_ln, p6_lc))
assert torch.equal(p6_ta, p6_tb), "the cached decoder must produce exactly the same tokens"
assert p6_maxdiff < 1e-3
print("generated %d tokens from a prompt of %d: identical tokens: %s | largest difference between the scores of the two versions: %.1e" % (P6_NNEW, P6_PROMPT.shape[1], bool(torch.equal(p6_ta, p6_tb)), p6_maxdiff))
print("first generated tokens:", p6_ta[0, 4:14].tolist())
print("position | no cache (ms/token) | with cache (ms/token) | no cache / cache   (average over 10 steps)")
p6_rows4 = []
for end in range(10, P6_NNEW + 1, 30 if not QUICK else 10):
    a = float(np.mean(p6_tn[end - 10:end])); b = float(np.mean(p6_tc[end - 10:end])); p6_rows4.append((P6_PROMPT.shape[1] + end, a, b))
    print("%8d | %19.3f | %21.3f | %10.1f" % (P6_PROMPT.shape[1] + end, a, b, a / b))
RESULTS["p6_nocache_slowdown"] = round(p6_rows4[-1][1] / p6_rows4[-1][2], 1)
RESULTS["p6_nocache_growth"] = round(float(np.mean(p6_tn[-10:]) / np.mean(p6_tn[:10])), 1); RESULTS["p6_cache_growth"] = round(float(np.mean(p6_tc[-10:]) / np.mean(p6_tc[:10])), 1)
print("From the first to the last 10 tokens, the time per token grows %.1f times without a cache and %.1f times with one." % (RESULTS["p6_nocache_growth"], RESULTS["p6_cache_growth"]))
plt.figure(figsize=(5.6, 3.6)); plt.plot(range(P6_PROMPT.shape[1], P6_PROMPT.shape[1] + P6_NNEW), p6_tn, label="no cache"); plt.plot(range(P6_PROMPT.shape[1], P6_PROMPT.shape[1] + P6_NNEW), p6_tc, label="with KV cache")
plt.xlabel("position of the new token"); plt.ylabel("time for one token (ms)"); plt.title("Time per generated token"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

   It prints:

```text
generated 160 tokens from a prompt of 4: identical tokens: True | largest difference between the scores of the two versions: 2.7e-05
first generated tokens: [87, 87, 87, 79, 110, 41, 41, 41, 41, 41]
position | no cache (ms/token) | with cache (ms/token) | no cache / cache   (average over 10 steps)
      14 |               0.948 |                 0.745 |        1.3
      44 |               1.604 |                 0.706 |        2.3
      74 |               2.114 |                 0.944 |        2.2
     104 |               2.810 |                 0.769 |        3.7
     134 |               4.000 |                 0.780 |        5.1
     164 |               4.301 |                 0.948 |        4.5
From the first to the last 10 tokens, the time per token grows 4.5 times without a cache and 1.3 times with one.
```

   **What you should see.**

   The two versions generate the **same 160 tokens** and their scores agree to about **3 × 10⁻⁵** (the difference of the float32 rounding of two ways to compute the same numbers). Without a cache, the time to generate a token **grows with the position**: from about 0.8 ms near the start to about **3.7 ms near the end** (4 to 5 times more); with the cache it stays between **0.55 and 0.65 ms** (it grows only 1.1 times, because the attention still reads all the stored keys). At position 164 the cache makes a token about **6 times faster**. The figure shows a rising curve and a nearly flat one.

   *What it means.* Without a cache the total work to generate `n` tokens grows with the **square** of `n`; with the cache it grows (almost) linearly. The speed-up is modest here because this toy is small and each step is dominated by the fixed cost of calling many small operations: in a real model, at thousands of tokens, the difference between recomputing and remembering is the difference between unusable and usable. We did not change a single weight: the cache is only a trade of memory for time.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

14. **Step 6.5. Count the cache from the real tensors.** The formula of Exercise 6.4 should match the bytes of the tensors that the cache really holds. We run the cached decoder for a **batch of 4 sequences** and 40 new tokens, add up the `nbytes` of every key and value tensor, and compare with your function (4 bytes per value, because this toy runs in float32). Then we look at what the same formula says for a 7-billion-parameter-like model.

```python
# Step 6.5. Cache bytes: measured from the tensors against the formula
P6_PROMPT4 = torch.tensor([ [5, 17, 42, 8], [1, 2, 3, 4], [9, 99, 19, 91], [7, 7, 7, 7] ])
with torch.no_grad():
    toks4, _, _, cache4 = p6_generate_cache(P6_PROMPT4, 40)
p6_measured = sum(k.nbytes + v.nbytes for k, v in cache4)
n_cached = cache4[0][0].shape[2]
p6_formula = p6_kv_bytes(P6_L, n_cached, P6_D, 4, 4)
print("the cache holds %d positions for %d sequences and %d layers: measured %d bytes (%s), formula %d bytes" % (n_cached, 4, P6_L, p6_measured, human_bytes(p6_measured), p6_formula))
assert p6_measured == p6_formula == 2 * P6_L * n_cached * P6_D * 4 * 4
print("in 16 bits the same cache would need %s; in 8 bits %s" % (human_bytes(p6_kv_bytes(P6_L, n_cached, P6_D, 2, 4)), human_bytes(p6_kv_bytes(P6_L, n_cached, P6_D, 1, 4))))
print("model weights of this toy: %s (float32); the cache is %.0f %% of them" % (human_bytes(4 * (2 * P6_V * P6_D + P6_L * 12 * P6_D * P6_D)), 100 * p6_measured / (4 * (2 * P6_V * P6_D + P6_L * 12 * P6_D * P6_D))))
RESULTS["p6_cache_bytes_toy"] = int(p6_measured)
```

   It prints:

```text
the cache holds 43 positions for 4 sequences and 2 layers: measured 704512 bytes (704.51 KB), formula 704512 bytes
in 16 bits the same cache would need 352.26 KB; in 8 bits 176.13 KB
model weights of this toy: 6.70 MB (float32); the cache is 11 % of them
```

   **What you should see.**

   The cache holds 43 positions (3 prompt tokens stored in advance and one more for each of the 40 steps) for 4 sequences and 2 layers: **704,512 bytes measured, 704,512 bytes by the formula**, and the assertion passes. In 16 bits the same cache would need 352 KB and in 8 bits 176 KB. Even for this toy the cache is already **11 percent of the model's weights** after only 43 tokens, and it grows with every token.

   *What it means.* The formula `2 × L × n × d × bytes × batch` is not an approximation but an exact count of the tensors: keys and values (the 2), for each layer, each position, each of the `d` numbers, each sequence in the batch. A 16-bit cache halves it, an 8-bit cache quarters it (at some cost in quality that you must test), and fewer key and value heads (grouped-query attention) divide it again.

15. **Step 6.6. The memory budget of a 7 B-like server.** Does a model **fit** on a GPU? The memory must hold the weights **and** the KV cache of every sequence that is being served (plus activations and overhead, which we ignore here, so our maximum batch is an optimistic upper bound). We use a 7-billion-parameter-like configuration (32 layers, `d` = 4,096) and an **assumed example GPU with 80 GB of memory** (not a real device: put the memory of your own hardware in `P6_MEM_GB`). The first table shows the cache in GB for several batch sizes and context lengths. The second shows the **largest batch** that fits at a context of 4,096 tokens with 16-bit or 4-bit weights (groups of 64) and with a 16-bit or 8-bit cache.

```python
# Step 6.6. Weights plus cache against the memory of an (assumed) 80 GB GPU
P6_MEM_GB = 80.0                    # an assumption for the example: use the memory of your hardware
P6_LAYERS, P6_DMODEL, P6_PARAMS_B = 32, 4096, 7
def p6_budget(params_b, bits, group, layers, d, n, batch, kv_bytes, mem_gb):
    """Return (weights GB, cache GB, total GB, fits?) for serving `batch` sequences of `n` tokens."""
    w = p6_weights_gb(params_b, bits, group); kv = p6_kv_bytes(layers, n, d, kv_bytes, batch) / 1e9
    return w, kv, w + kv, w + kv <= mem_gb
def p6_max_batch(params_b, bits, group, layers, d, n, kv_bytes, mem_gb):
    """The largest number of sequences of n tokens that fit next to the weights (0 if even the weights do not fit)."""
    w = p6_weights_gb(params_b, bits, group); per_seq = p6_kv_bytes(layers, n, d, kv_bytes, 1) / 1e9
    return max(0, int((mem_gb - w) // per_seq))
print("KV cache in GB (16-bit values), %d layers, d = %d" % (P6_LAYERS, P6_DMODEL))
print("context | " + " | ".join("batch %2d" % b for b in (1, 8, 32)))
for n in (2048, 4096, 8192):
    print("%7d | " % n + " | ".join("%8.2f" % p6_budget(P6_PARAMS_B, 16, None, P6_LAYERS, P6_DMODEL, n, b, 2, P6_MEM_GB)[1] for b in (1, 8, 32)))
assert abs(p6_budget(7, 16, None, 32, 4096, 4096, 8, 2, 80)[1] - 17.18) < 0.01              # the lesson's example
print("\nLargest batch at 4096 tokens on %.0f GB (optimistic: activations and overhead ignored)" % P6_MEM_GB)
print("weights              | cache values | weights GB | cache per sequence GB | largest batch")
p6_mb = {}
for wname, bits, grp in (("16-bit", 16, None), ("4-bit, groups of 64", 4, 64)):
    for kvb, kname in ((2, "16-bit"), (1, "8-bit")):
        m = p6_max_batch(P6_PARAMS_B, bits, grp, P6_LAYERS, P6_DMODEL, 4096, kvb, P6_MEM_GB); p6_mb[(wname, kname)] = m
        print("%-20s | %12s | %10.2f | %21.3f | %13d" % (wname, kname, p6_weights_gb(P6_PARAMS_B, bits, grp), p6_kv_bytes(P6_LAYERS, 4096, P6_DMODEL, kvb, 1) / 1e9, m))
        assert p6_budget(P6_PARAMS_B, bits, grp, P6_LAYERS, P6_DMODEL, 4096, m, kvb, P6_MEM_GB)[3] and not p6_budget(P6_PARAMS_B, bits, grp, P6_LAYERS, P6_DMODEL, 4096, m + 1, kvb, P6_MEM_GB)[3]
assert p6_mb[("16-bit", "16-bit")] == 30 and p6_mb[("4-bit, groups of 64", "16-bit")] == 35
RESULTS["p6_maxbatch_fp16"] = p6_mb[("16-bit", "16-bit")]; RESULTS["p6_maxbatch_int4"] = p6_mb[("4-bit, groups of 64", "16-bit")]; RESULTS["p6_maxbatch_int4_kv8"] = p6_mb[("4-bit, groups of 64", "8-bit")]
```

   It prints:

```text
KV cache in GB (16-bit values), 32 layers, d = 4096
context | batch  1 | batch  8 | batch 32
   2048 |     1.07 |     8.59 |    34.36
   4096 |     2.15 |    17.18 |    68.72
   8192 |     4.29 |    34.36 |   137.44

Largest batch at 4096 tokens on 80 GB (optimistic: activations and overhead ignored)
weights              | cache values | weights GB | cache per sequence GB | largest batch
16-bit               |       16-bit |      14.00 |                 2.147 |            30
16-bit               |        8-bit |      14.00 |                 1.074 |            61
4-bit, groups of 64  |       16-bit |       3.72 |                 2.147 |            35
4-bit, groups of 64  |        8-bit |       3.72 |                 1.074 |            71
```

   **What you should see.**

   For the 7 B-like configuration the cache of **one** sequence of 4,096 tokens is **2.15 GB** in 16 bits; the first table shows 1.07 GB at 2,048 tokens, 17.2 GB for a batch of 8 at 4,096 (the lesson's number, asserted) and **137 GB for a batch of 32 at 8,192 tokens**, far beyond one 80 GB device. On the assumed 80 GB GPU, the largest batch at 4,096 tokens is **30 sequences with 16-bit weights and 16-bit cache**, **35 with 4-bit weights** (the weights shrink from 14 to 3.72 GB, which frees room for five more sequences), **61 with 16-bit weights and an 8-bit cache** and **71 with both**. The assertions check each result and that one more sequence does not fit.

   *What it means.* At long contexts the cache, not the weights, is what limits the number of users per GPU: quantizing the weights gained 5 sequences, quantizing the cache gained 31. Halving the context also doubles the batch. Remember that these are upper bounds: activations, the framework and fragmentation take a share of the memory.

16. **Step 6.7. What does a million tokens cost?** Cost per million tokens is the price of the machine for the time it takes: `price per hour ÷ (tokens per second × 3,600) × 1,000,000`. The tokens per second of decoding come from the memory-bound rule of Part 5: each step reads the weights **once** plus the cache of **every** sequence in the batch, so a step takes `(weights + b × cache per sequence) ÷ bandwidth` and produces `b` tokens. The inputs below are **assumptions**, not real prices or devices: a bandwidth of 200 GB/s, a price of 2.00 per hour (any currency), a context of 4,096 tokens. They are only here to show the **shape** of the answer: batching and quantization both lower the cost, and the cache sets the limit.

```python
# Step 6.7. Cost per million generated tokens (ALL numbers are assumptions)
P6_BW_GB_S, P6_PRICE_HOUR = 200.0, 2.00
def p6_cost_per_m(tokens_per_s, price_per_hour):
    """Cost of one million tokens when the machine makes `tokens_per_s` tokens per second and costs `price_per_hour`."""
    return price_per_hour / (tokens_per_s * 3600) * 1e6
def p6_decode_tps(weights_gb, kv_seq_gb, batch, bw_gb_s):
    """Memory-bound tokens per second: a step reads the weights once and the cache of each of the `batch` sequences."""
    return batch / ((weights_gb + batch * kv_seq_gb) / bw_gb_s)
assert abs(p6_cost_per_m(1000, 3.6) - 1.0) < 1e-9                    # 1000 tokens/s at 3.60 an hour: 1.00 per million tokens
print("assumed: bandwidth %.0f GB/s, price %.2f per hour, context 4096 tokens, 16-bit cache" % (P6_BW_GB_S, P6_PRICE_HOUR))
print("weights              | batch | tokens/s (bound) | step (ms) | cost per million tokens")
p6_cost = {}
for wname, bits, grp in (("16-bit", 16, None), ("4-bit, groups of 64", 4, 64)):
    w = p6_weights_gb(P6_PARAMS_B, bits, grp); kvs = p6_kv_bytes(P6_LAYERS, 4096, P6_DMODEL, 2, 1) / 1e9
    for b in (1, 8, 30):
        tps = p6_decode_tps(w, kvs, b, P6_BW_GB_S); p6_cost[(wname, b)] = p6_cost_per_m(tps, P6_PRICE_HOUR)
        print("%-20s | %5d | %16.1f | %9.1f | %10.2f" % (wname, b, tps, 1000 * b / tps, p6_cost[(wname, b)]))
RESULTS["p6_cost_fp16_b1"] = round(p6_cost[("16-bit", 1)], 1); RESULTS["p6_cost_fp16_b30"] = round(p6_cost[("16-bit", 30)], 2); RESULTS["p6_cost_int4_b30"] = round(p6_cost[("4-bit, groups of 64", 30)], 2)
print("Batch 30 against batch 1 at 16 bits: %.1f times cheaper per token (the same machine, shared by more requests)." % (p6_cost[("16-bit", 1)] / p6_cost[("16-bit", 30)]))
```

   It prints:

```text
assumed: bandwidth 200 GB/s, price 2.00 per hour, context 4096 tokens, 16-bit cache
weights              | batch | tokens/s (bound) | step (ms) | cost per million tokens
16-bit               |     1 |             12.4 |      80.7 |      44.85
16-bit               |     8 |             51.3 |     155.9 |      10.83
16-bit               |    30 |             76.5 |     392.1 |       7.26
4-bit, groups of 64  |     1 |             34.1 |      29.3 |      16.30
4-bit, groups of 64  |     8 |             76.6 |     104.5 |       7.26
4-bit, groups of 64  |    30 |             88.0 |     340.7 |       6.31
Batch 30 against batch 1 at 16 bits: 6.2 times cheaper per token (the same machine, shared by more requests).
```

   **What you should see.**

   With the assumed numbers (200 GB/s, 2.00 per hour, a context of 4,096 tokens), a 16-bit 7 B-like model at batch 1 can decode at most **12.4 tokens per second**, which costs **44.85 per million tokens**; at batch 8 the bound is 51 tokens per second (**10.83** per million), and at the largest batch that fits, 30, it is 76.5 tokens per second (**7.26**): batching makes a token **6.2 times cheaper**. With 4-bit weights the cost at batch 1 falls to 16.30 and at batch 30 to **6.31**. The step time grows with the batch (81 ms at batch 1, 392 ms at batch 30): every sequence in the batch must wait for the cache of all the others to be read.

   *What it means.* The shape matters more than the numbers, which are assumptions: cost per token falls steeply with the batch at first and then flattens, because at a large batch the **cache**, not the weights, is what each step reads; quantizing the weights helps most at small batches (where the weights dominate). A real service chooses the batch that meets its latency target (Step 6.3) and then counts the cost.

   **Continuous batching, named but not built.** The server of Step 6.2 collects a batch, serves it, and only then starts the next one: a long answer holds up the short ones in its batch. Modern serving systems use **continuous batching** (also called in-flight batching): after every decoding step, finished sequences leave the batch and waiting requests join it, so the GPU never idles waiting for the slowest sequence. Together with a cache that is stored in small blocks instead of one long strip per sequence (the idea of PagedAttention in the vLLM system, Kwon et al., 2023), it is what lets real servers keep the batch large and the memory full. We do not implement it here; the capacity planner in Part 7 uses the same memory and bandwidth formulas.

17. **Step 6.8. Compare with your guesses.** Run the cell below to put your guesses next to the measurements (if you left a guess as `None` it shows "(not written)").

```python
# Step 6.8. Imagine first against the measurements
def p6_g(x):
    return "(not written)" if x is None else "%s" % x
print("what you guessed -> what was measured")
print("time at batch 64 divided by time at batch 1:  %s -> %.1f" % (p6_g(guess_batch64_time_ratio), RESULTS["p6_b64_time_ratio"]))
print("throughput gain of batch 64:                  %s -> %.1f" % (p6_g(guess_throughput_gain), RESULTS["p6_throughput_gain_b64"]))
print("slowdown of the last token without a cache:   %s -> %.1f" % (p6_g(guess_nocache_slowdown), RESULTS["p6_nocache_slowdown"]))
```

   It prints:

```text
what you guessed -> what was measured
time at batch 64 divided by time at batch 1:  (not written) -> 5.4
throughput gain of batch 64:                  (not written) -> 11.8
slowdown of the last token without a cache:   (not written) -> 4.5
```

   **What you should see.**

   The three comparisons are the heart of the part. **Batch 64 took about 4.7 times as long as batch 1 but finished about 13.5 times as many requests per second**: batching is nearly free until the arithmetic starts to count. **Without a cache the last of 160 tokens was about 6 times slower** than with one, and the gap widens as the context grows. If your guesses were far from these numbers, look at which assumption was wrong: the idea that "twice the work takes twice the time" is true only when the machine is busy with arithmetic; for a model that waits for memory it is not.

   *A word about timings.* The time figures in this paragraph come from a run on a quiet machine (nothing else running); the table printed above comes from your own run and can differ by tens of percent, and much more if other programs are using the processor. The **ratios, the order of the rows and the shape of the curves** are what to compare, not the milliseconds.

   ### ✏️ Reflect: Part 6

   1. Your server has a fixed `t0` and `c`. At which arrival rates does a batch of 1 stop being stable, and what does the batch size have to be to stay stable at twice that rate? Use the capacity formula `1000 b / (t0 + c b)`.
   2. Why does the p95 latency of the larger batches **rise** at low arrival rates, although they are more efficient? What would a **timeout** change?
   3. The cache for a 4,096-token context was larger than the weights of the model at a batch of 8. Name two ways to make it smaller, and say what each costs.

### Part 7. Plan the deployment and reflect (20 min)

*Lesson sections: "Serving a model" and "Key takeaways".* In Parts 1 to 6 you measured one thing at a time: memory, a gradient average, a split matrix, a pipeline bubble, a number format, a quantizer and a batching server. A real deployment needs all of them at once, and it needs one person to say: *this model, on this hardware, for this many users, at this cost, and this is what we do when it breaks.* ==In this part you write a small capacity planner, run it for three deployments, turn the result into a checklist and a design card, and collect the honest findings of the whole week.== Every price, bandwidth and memory size in this part is **ASSUMED**: it is a parameter that you replace with the numbers of your own hardware report from the Setup, not a specification of any real product.

**Imagine first.** You run the school helpdesk model. Before you run anything, guess three things. (1) Can a 70-billion-parameter model in 16-bit numbers fit on **one** 80 GB GPU? (2) For that 70 B model, which is cheaper per million tokens: four 80 GB GPUs with 16-bit weights, or one 80 GB GPU with 4-bit weights? (3) A 7 B model with 4-bit weights, on a GPU that reads 300 GB per second: about how many tokens per second can **one stream** of text get, at most? Write your guesses, then see how close you were.

```python
# Imagine first: memory, cost and speed of a deployment
guess_70b_fits_one_gpu = None    # YOUR GUESS: "yes" or "no" (70 B in 16-bit numbers on one 80 GB GPU?)
guess_cheaper = None             # YOUR GUESS: "four GPUs fp16" or "one GPU int4" (cheaper per million tokens?)
guess_tokens_s = None            # YOUR GUESS: tokens per second for one stream of a 7 B model, 4-bit weights, 300 GB/s (upper bound)
```

1. **Step 7.1. The memory of a deployment.** A deployment has to hold two things in GPU memory all the time: the **weights** (parameters times bytes per parameter, Part 1 and Part 5) and the **KV cache** (2 · layers · tokens · d · bytes per value, for every request in the batch, Part 6). The formulas of the lesson were `weights = parameters × bits / 8` and `KV = 2 × L × n × d × bytes × batch`. Large models today usually share each key and value among several query heads (**grouped-query attention**), which makes the cache smaller by a factor `kv_share` (for example 8 key-value heads for 64 query heads is `kv_share = 0.125`); for a plain multi-head model `kv_share = 1`.

   *Exercise 7.1:* Write `p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0)`. It returns the pair `(weights_gb, kv_gb)` in gigabytes (1 GB is 10⁹ bytes; `params_b` is in billions). Why it matters: every decision in this part starts from these two numbers, and you have measured both in earlier parts. The check compares your function with the lesson's figures (7 B: 28, 14, 7 and 3.5 GB; 70 B: 140 GB) and with a hand computation of a cache.

```python
def p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0):
    """Return (weights_gb, kv_gb)."""
    return None   # YOUR CODE: weights from the parameters and bits; KV cache from layers, tokens, d, bytes per value, batch and kv_share
```

   A possible answer (try first, then compare):

```python
def p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share=1.0):
    weights_gb = params_b * 1e9 * w_bits / 8 / 1e9
    kv_gb = 2 * layers * ctx * d_model * (kv_bits / 8) * batch * kv_share / 1e9
    return weights_gb, kv_gb
```

   The factor 2 is for keys and values, `layers * ctx * d_model` is the number of stored values per request and sequence position, `kv_bits / 8` converts to bytes, and the batch multiplies because every request keeps its own cache.

   Running the check on this answer prints:

```text
✓ 7 B at 32, 16, 8 and 4 bits is 28, 14, 7 and 3.5 GB
✓ 70 B at 16 bits is 140 GB
✓ KV cache of 7 B, 4,096 tokens, batch 8, 16-bit is 17.18 GB
✓ kv_share 0.125 makes the cache 8 times smaller
✓ the cache grows with the batch, the weights do not
```

2. **Step 7.2. The capacity planner.** Now we add the other two limits of the lesson. **Speed:** decoding reads all the weights and the cache for every token, so with total memory bandwidth `bw × N` the step takes at least `(weights + KV) / (bw × N)` seconds, and one stream gets at most the inverse of that in tokens per second. When the model is split over N GPUs with tensor parallelism (Part 3) every layer needs two all-reduces of `batch × d × 2` bytes, with traffic `2(N − 1)/N × S` (Part 2); we add that time using the link speed. **Cost:** a real server reaches only a fraction of the bound, which we call `eff` and assume to be 0.5; the price per million tokens is `GPUs × price per hour / (tokens per second × 3600) × 10⁶`.

   The function below does all this and gives **advice**. Read it before you run it: every rule is one you can argue with. The 10 percent that `reserve` keeps free is an assumed allowance for activations and the framework (Part 1), and the threshold of 100 GB/s for a link that is fast enough for tensor parallelism is also assumed. The function uses the `p7_memory` of Step 7.1.

```python
import math

def p7_plan(params_b, w_bits, kv_bits, ctx, batch, gpu_gb, bw_gb_s, n_gpus, link_gb_s,
            layers=32, d_model=4096, kv_share=1.0, price_gpu_hr=2.0, eff=0.5, reserve=0.10):
    """Capacity plan for serving one model. Every hardware number is an ASSUMPTION you pass in."""
    weights_gb, kv_gb = p7_memory(params_b, w_bits, kv_bits, ctx, batch, layers, d_model, kv_share)
    total_gb = weights_gb + kv_gb
    usable_gb = n_gpus * gpu_gb * (1 - reserve)
    fits = total_gb <= usable_gb
    step_s = total_gb / (bw_gb_s * n_gpus)                                  # time to read weights and cache once
    comm_s = 0.0
    if n_gpus > 1 and link_gb_s > 0 and layers > 0:                         # two all-reduces per layer when tensor-parallel
        size_gb = batch * d_model * 2 / 1e9
        comm_s = 2 * layers * (2 * (n_gpus - 1) / n_gpus) * size_gb / link_gb_s
    stream_bound = 1 / (step_s + comm_s)                                    # tokens/s of one stream, upper bound
    total_tps = batch * stream_bound * eff                                  # expected tokens/s for the whole batch
    cost = n_gpus * price_gpu_hr / (total_tps * 3600) * 1e6
    if not fits:
        need = math.ceil(total_gb / (gpu_gb * (1 - reserve)))
        advice = "does not fit: needs at least %d GPUs of this size, or fewer bits, a shorter context or a smaller batch" % need
    elif total_gb <= gpu_gb * (1 - reserve):
        advice = "fits on one GPU: keep the model whole and add replicas (data parallel) when more users arrive"
    elif link_gb_s >= 100:
        advice = "split over %d GPUs with tensor parallelism (the link is fast enough for two all-reduces per layer)" % n_gpus
    else:
        advice = "link too slow for tensor parallelism: use pipeline stages and micro-batches between the boxes"
    return {"weights_gb": weights_gb, "kv_gb": kv_gb, "total_gb": total_gb, "usable_gb": usable_gb, "fits": fits,
            "comm_ms": comm_s * 1000, "stream_tps": stream_bound, "total_tps": total_tps, "cost_per_m": cost, "advice": advice}

# Sanity checks against numbers you can do by hand
a_ = p7_plan(7, 16, 16, 4096, 8, gpu_gb=80, bw_gb_s=2000, n_gpus=1, link_gb_s=0)
check("one stream of 7 B 16-bit, 1 GPU of 2,000 GB/s: bound is 2000 / (14 + 17.18) tokens/s", abs(a_["stream_tps"] - 2000 / (14 + 17.179869184)) < 1e-6, "no communication on one GPU")
check("a 70 B model in 16 bits does not fit on one 80 GB GPU", not p7_plan(70, 16, 16, 1, 1, 80, 2000, 1, 0, layers=80, d_model=8192)["fits"], "140 GB is more than 72 GB usable")
check("more GPUs add all-reduce time", p7_plan(7, 16, 16, 4096, 8, 80, 2000, 4, 300)["comm_ms"] > 0 and p7_plan(7, 16, 16, 4096, 8, 80, 2000, 1, 300)["comm_ms"] == 0, "communication only with N above 1")
print("one stream bound for the check case: %.1f tokens/s; expected for the batch of 8 at eff 0.5: %.1f tokens/s; cost %.2f per million tokens (assumed price)" % (a_["stream_tps"], a_["total_tps"], a_["cost_per_m"]))
```

   It prints:

```text
✓ one stream of 7 B 16-bit, 1 GPU of 2,000 GB/s: bound is 2000 / (14 + 17.18) tokens/s
✓ a 70 B model in 16 bits does not fit on one 80 GB GPU
✓ more GPUs add all-reduce time
one stream bound for the check case: 64.1 tokens/s; expected for the batch of 8 at eff 0.5: 256.6 tokens/s; cost 2.17 per million tokens (assumed price)
```

3. **Step 7.3. Three deployments.** Now we plan three deployments. **All hardware numbers are assumed**: `MY_BOX` is where you put the numbers of *your* machine from the hardware report of the Setup (the memory of your GPU, or of your RAM if you have no GPU, and a bandwidth you measured in Part 5 or looked up in the documentation of your hardware). We use deliberately round values that are not those of any product.

   1. **The school helpdesk model** (7 B, 4-bit weights with scales, so 4.25 bits per weight, 2,048 tokens of context, 4 requests at once) on **one box**.
   2. **A 70 B model for a district** (16-bit weights, 4,096 tokens, 16 requests at once, grouped-query attention with `kv_share = 0.125`, 80 GB GPUs): on 4 GPUs with 16-bit weights, and on 1 GPU with 4-bit weights.
   3. **The Structural Inspection Assistant on the school's server**: a small image classifier (assumed 25 million parameters, 32-bit) **plus** the same 7 B language model at 4.25 bits with 4,096 tokens, one request at a time. Both must fit in the same memory.

   *Exercise 7.3:* Run the three deployments through `p7_plan` and print one line each. Fill in `answers_73` with the strings `"fits"` or `"does not fit"` for the helpdesk box (key `"helpdesk"`), for the 70 B model on 4 GPUs in 16 bits (`"district_fp16"`) and for the 70 B model on one GPU in 4 bits (`"district_int4"`). Why it matters: the planner is only useful if you read its verdict before you buy anything. The check compares your three words with the planner.

```python
MY_BOX = {"gpu_gb": 24, "bw_gb_s": 300, "price_gpu_hr": 1.0}      # ASSUMED: replace with the numbers of your own hardware report

P7_SCEN = {
    "helpdesk":      dict(params_b=7,  w_bits=4.25, kv_bits=16, ctx=2048, batch=4,  n_gpus=1, link_gb_s=0,   layers=32, d_model=4096, **MY_BOX),
    "district_fp16": dict(params_b=70, w_bits=16,   kv_bits=16, ctx=4096, batch=16, n_gpus=4, link_gb_s=300, layers=80, d_model=8192, kv_share=0.125, gpu_gb=80, bw_gb_s=2000, price_gpu_hr=2.0),
    "district_int4": dict(params_b=70, w_bits=4.25, kv_bits=16, ctx=4096, batch=16, n_gpus=1, link_gb_s=0,   layers=80, d_model=8192, kv_share=0.125, gpu_gb=80, bw_gb_s=2000, price_gpu_hr=2.0),
}
p7_res = {k_: p7_plan(**v_) for k_, v_ in P7_SCEN.items()}
print("%-14s %7s %6s %7s %5s %8s %9s %9s" % ("deployment", "weights", "KV", "total", "fits", "stream", "batch", "$ per M"))
for k_, r_ in p7_res.items():
    print("%-14s %5.1f G %4.1f G %5.1f G %5s %6.1f t/s %7.1f t/s %9.2f" % (k_, r_["weights_gb"], r_["kv_gb"], r_["total_gb"], "yes" if r_["fits"] else "NO", r_["stream_tps"], r_["total_tps"], r_["cost_per_m"]))
for k_, r_ in p7_res.items(): print("  %-14s %s" % (k_, r_["advice"]))
answers_73 = {"helpdesk": None, "district_fp16": None, "district_int4": None}    # YOUR ANSWERS: "fits" or "does not fit"
```

   It prints:

```text
deployment     weights     KV   total  fits   stream     batch   $ per M
helpdesk         3.7 G  4.3 G   8.0 G   yes   37.4 t/s    74.9 t/s      3.71
district_fp16  140.0 G 21.5 G 161.5 G   yes   49.0 t/s   392.3 t/s      5.67
district_int4   37.2 G 21.5 G  58.7 G   yes   34.1 t/s   272.7 t/s      2.04
  helpdesk       fits on one GPU: keep the model whole and add replicas (data parallel) when more users arrive
  district_fp16  split over 4 GPUs with tensor parallelism (the link is fast enough for two all-reduces per layer)
  district_int4  fits on one GPU: keep the model whole and add replicas (data parallel) when more users arrive
```

   A possible answer (try first, then compare):

```python
answers_73 = {"helpdesk": "fits", "district_fp16": "fits", "district_int4": "fits"}
```

   All three fit, but for different reasons: the helpdesk model is small, the 16-bit district model needs all four GPUs, and the 4-bit one fits on a single GPU because the weights are four times smaller. If you change `MY_BOX` to a smaller memory, the first answer changes: that is the experiment to try.

   Running the check on this answer prints:

```text
✓ helpdesk: fits
✓ district_fp16: fits
✓ district_int4: fits
```

```python
# Step 7.3 (continued). The Structural Inspection Assistant: classifier plus language model in the same memory
cls_ = p7_plan(0.025, 32, 16, 1, 1, layers=0, d_model=1, gpu_gb=16, bw_gb_s=200, n_gpus=1, link_gb_s=0)
llm_ = p7_plan(7, 4.25, 16, 4096, 1, gpu_gb=16, bw_gb_s=200, n_gpus=1, link_gb_s=0, layers=32, d_model=4096, price_gpu_hr=1.0)
tot_ = cls_["weights_gb"] + llm_["total_gb"]
print("classifier weights %.3f GB | language model weights %.2f GB + KV cache %.2f GB | together %.2f GB of %.1f GB usable -> %s" % (cls_["weights_gb"], llm_["weights_gb"], llm_["kv_gb"], tot_, 16 * 0.9, "fits" if tot_ <= 16 * 0.9 else "does not fit"))
print("language model alone: stream bound %.1f tokens/s, expected %.1f tokens/s, about %.2f per million tokens (assumed price)" % (llm_["stream_tps"], llm_["total_tps"], llm_["cost_per_m"]))
check("the classifier is tiny next to the language model", cls_["weights_gb"] < 0.05 * llm_["weights_gb"], "100 MB against 3.7 GB")
check("the two models fit together in the assumed 16 GB", tot_ <= 16 * 0.9, "lower the context or the bits")
RESULTS["p7_inspection_gb"] = round(tot_, 2)
w_only_ = MY_BOX["bw_gb_s"] / p7_memory(7, 4.25, 16, 1, 1, 32, 4096)[0]
print("guess 3 of the Imagine step: weights alone (3.72 GB) at %d GB/s give at most %.1f tokens/s for one stream; with the KV cache of the helpdesk plan %.1f" % (MY_BOX["bw_gb_s"], w_only_, p7_res["helpdesk"]["stream_tps"]))
```

   It prints:

```text
classifier weights 0.100 GB | language model weights 3.72 GB + KV cache 2.15 GB | together 5.97 GB of 14.4 GB usable -> fits
language model alone: stream bound 34.1 tokens/s, expected 17.0 tokens/s, about 16.30 per million tokens (assumed price)
✓ the classifier is tiny next to the language model
✓ the two models fit together in the assumed 16 GB
guess 3 of the Imagine step: weights alone (3.72 GB) at 300 GB/s give at most 80.7 tokens/s for one stream; with the KV cache of the helpdesk plan 37.4
```

   *What it means.* The three guesses of the Imagine step: (1) **no**, 70 B in 16 bits needs 140 GB and one 80 GB GPU offers 72 GB after the 10 percent reserve; (2) **one GPU with 4-bit weights** was cheaper in this model (2.04 against 5.67 per million tokens at the assumed prices), because it needs a quarter of the GPUs and has no all-reduce, even though each stream is a little slower (34 against 49 tokens per second); (3) about 80 tokens per second if you count only the weights, and about 37 once the cache of four requests is read as well: the cache is not free. The prices are only as good as the assumptions, and the accuracy of the 4-bit model is **not** in this table: Part 5 showed that you must measure it. The helpdesk and the inspection assistant fit on a small box; the 70 B model needs either four GPUs and a fast link, or quantization.

4. **Step 7.4. What moves the cost? Batch size and bits.** The planner says that cost per million tokens falls when more requests share one pass over the weights (Part 6), until memory runs out because every request needs its own KV cache. Let us find that edge for the helpdesk model. We sweep the batch size for 4-bit and for 16-bit weights on the assumed box, and find the largest batch that fits. **Caution:** the planner's speed bound assumes that decoding stays memory-bound; at large batches the arithmetic starts to limit too (Part 6 measured `t₀ + c·b`), so the real cost curve flattens earlier than this one.

```python
def p7_max_batch(**kw):
    kw = dict(kw); b_ = 1
    while b_ < 4096 and p7_plan(**dict(kw, batch=b_ + 1))["fits"]: b_ += 1
    return b_ if p7_plan(**dict(kw, batch=b_))["fits"] else 0

base_ = dict(P7_SCEN["helpdesk"])
sweep_ = {}
for bits_ in (4.25, 16):
    kw_ = dict(base_, w_bits=bits_)
    sweep_[bits_] = {"max_batch": p7_max_batch(**kw_), "rows": [(b_, p7_plan(**dict(kw_, batch=b_))) for b_ in (1, 2, 4, 8, 16, 32, 64)]}
print("%-6s %-5s %8s %6s %10s %8s" % ("bits", "batch", "total", "fits", "tokens/s", "$ per M"))
for bits_, d_ in sweep_.items():
    for b_, r_ in d_["rows"]:
        print("%-6s %-5d %6.1f G %6s %10.1f %8.2f" % (bits_, b_, r_["total_gb"], "yes" if r_["fits"] else "NO", r_["total_tps"], r_["cost_per_m"]))
    print("largest batch that fits with %s-bit weights: %d" % (bits_, d_["max_batch"]))
fig_, ax_ = plt.subplots(figsize=(6, 3.2))
for bits_, d_ in sweep_.items():
    ok_ = [(b_, r_["cost_per_m"]) for b_, r_ in d_["rows"] if r_["fits"] ]
    ax_.plot([x for x, _ in ok_], [y for _, y in ok_], "o-", label="%s-bit weights (stops at the memory limit)" % bits_)
ax_.set_xscale("log", base=2); ax_.set_yscale("log"); ax_.set_xlabel("batch size"); ax_.set_ylabel("assumed price per million tokens"); ax_.legend(fontsize=8); ax_.grid(alpha=.3)
plt.tight_layout(); plt.show()
RESULTS["p7_max_batch_int4"] = sweep_[4.25]["max_batch"]; RESULTS["p7_max_batch_fp16"] = sweep_[16]["max_batch"]
check("4-bit weights leave room for a larger batch than 16-bit weights", sweep_[4.25]["max_batch"] > sweep_[16]["max_batch"], "smaller weights leave more memory for the cache")
falls_ = []
for bits_, d_ in sweep_.items():
    fit_costs_ = [r_["cost_per_m"] for _, r_ in d_["rows"] if r_["fits"] ]
    falls_.append(fit_costs_[0] > fit_costs_[-1])
check("cost per million tokens falls as the batch grows (while it fits)", all(falls_), "one pass over the weights serves more requests")
```

   It prints:

```text
bits   batch    total   fits   tokens/s  $ per M
4.25   1        4.8 G    yes       31.3     8.87
4.25   2        5.9 G    yes       51.1     5.43
4.25   4        8.0 G    yes       74.9     3.71
4.25   8       12.3 G    yes       97.5     2.85
4.25   16      20.9 G    yes      114.8     2.42
4.25   32      38.1 G     NO      126.1     2.20
4.25   64      72.4 G     NO      132.5     2.10
largest batch that fits with 4.25-bit weights: 16
16     1       15.1 G    yes       10.0    27.91
16     2       16.1 G    yes       18.6    14.95
16     4       18.3 G    yes       32.8     8.47
16     8       22.6 G     NO       53.1     5.23
16     16      31.2 G     NO       77.0     3.61
16     32      48.4 G     NO       99.3     2.80
16     64      82.7 G     NO      116.1     2.39
largest batch that fits with 16-bit weights: 7
✓ 4-bit weights leave room for a larger batch than 16-bit weights
✓ cost per million tokens falls as the batch grows (while it fits)
```

   *What it means.* On the assumed 24 GB box, going from batch 1 to batch 16 cuts the price by a factor of 3.7 (8.87 to 2.42) with 4-bit weights, and the 4-bit model fits a batch of 16 while the 16-bit one stops at 7: smaller weights leave room for more caches, so quantization buys both a smaller model **and** a cheaper token. The rows marked `NO` are only a calculation: they show where memory would run out. The curve flattens because the cache read starts to dominate, and in a real server it would flatten even earlier, because at a large batch the arithmetic limits the speed as well (the line `t₀ + c·b` of Part 6).

5. **Step 7.5. The deployment checklist and your design card.** A number from a planner is not a deployment. A deployment also has a **precision** and a **parallelism** decision, a **batching** policy, a **service level objective** (SLO: for example "95 percent of answers in under 3 seconds", the p95 of Part 6), **monitoring**, **versioning**, a **rollback** plan, **privacy and guardrails** (Week 17) and a **cost**. The function below fills in the checklist for the helpdesk from the planner result and prints it as a markdown table, so you can paste it into a report.

```python
CHECK_FIELDS = ("precision", "parallelism", "batching", "slo", "monitoring", "versioning", "rollback", "privacy_guardrails", "cost")

def p7_checklist(name, plan, slo_p95_s):
    return {
        "precision": "4.25-bit weights (4 bits plus a scale for every group of 64, Part 5) and a 16-bit KV cache; accuracy rechecked on our own questions after quantizing.",
        "parallelism": "none: the whole model uses %.1f GB of %.1f GB, so we keep one copy per box and add replicas when users grow (data parallel)." % (plan["total_gb"], MY_BOX["gpu_gb"]),
        "batching": "the server waits at most 50 ms to form a batch of up to %d requests (Part 6), because a larger batch lowers the cost but raises the wait." % RESULTS.get("p7_max_batch_int4", 8),
        "slo": "95 percent of answers (p95) in under %.0f seconds, measured over each week, and fewer than 1 request in 100 dropped." % slo_p95_s,
        "monitoring": "log the p50, p95 and p99 latency, the queue length, the GPU memory in use and the share of answers that the Week 17 rails blocked, and alert when p95 is above the SLO.",
        "versioning": "every model file carries a name, the quantization settings and a checksum, and the evaluation results of that file are stored next to it.",
        "rollback": "keep the previous model file loaded on a second box; if p95 or the blocked share doubles for 10 minutes, send the traffic back to it by hand.",
        "privacy_guardrails": "the input, tool and output rails of Week 17 stay on, a human reads a sample of 20 conversations every Friday, and logs hide names and are deleted after 30 days.",
        "cost": "about %.2f per million tokens at the assumed price of %.2f per GPU hour and half of the speed bound (assumed)." % (plan["cost_per_m"], MY_BOX["price_gpu_hr"]),
    }

ref_card = p7_checklist("helpdesk", p7_res["helpdesk"], 3)
print("| item | decision |\n|---|---|")
for f_ in CHECK_FIELDS: print("| %s | %s |" % (f_, ref_card[f_]))
```

   It prints:

```text
| item | decision |
|---|---|
| precision | 4.25-bit weights (4 bits plus a scale for every group of 64, Part 5) and a 16-bit KV cache; accuracy rechecked on our own questions after quantizing. |
| parallelism | none: the whole model uses 8.0 GB of 24.0 GB, so we keep one copy per box and add replicas when users grow (data parallel). |
| batching | the server waits at most 50 ms to form a batch of up to 16 requests (Part 6), because a larger batch lowers the cost but raises the wait. |
| slo | 95 percent of answers (p95) in under 3 seconds, measured over each week, and fewer than 1 request in 100 dropped. |
| monitoring | log the p50, p95 and p99 latency, the queue length, the GPU memory in use and the share of answers that the Week 17 rails blocked, and alert when p95 is above the SLO. |
| versioning | every model file carries a name, the quantization settings and a checksum, and the evaluation results of that file are stored next to it. |
| rollback | keep the previous model file loaded on a second box; if p95 or the blocked share doubles for 10 minutes, send the traffic back to it by hand. |
| privacy_guardrails | the input, tool and output rails of Week 17 stay on, a human reads a sample of 20 conversations every Friday, and logs hide names and are deleted after 30 days. |
| cost | about 3.71 per million tokens at the assumed price of 1.00 per GPU hour and half of the speed bound (assumed). |
```

   *Exercise 7.5:* Fill in `my_card` for **your own deployment** (for example the helpdesk of your school, a study helper or the Structural Inspection Assistant) with the nine fields of `CHECK_FIELDS`, at least 8 words in each. Use the planner for your memory and cost numbers. Why it matters: the card makes you decide the monitoring, the rollback and the human step before the first user arrives. The check tests the structure: the `slo` must contain a number and the word p95, the `cost` a number, the `precision` a number format, and the `privacy_guardrails` one of the protections of Week 17. It never judges the content.

```python
my_card = {f_: "" for f_ in CHECK_FIELDS}      # YOUR TEXT: nine fields, each at least 8 words
```

   The reference card printed above is one possible answer for the helpdesk. A shorter one, for the inspection assistant:

   A possible answer (try first, then compare):

```python
my_card = {
    "precision": "The language model uses 4-bit weights with groups of 64 and the classifier stays in 32-bit floats, rechecked on 200 test photos.",
    "parallelism": "None: classifier and language model together need about 6 GB of the 16 GB server, so one copy on one box is enough.",
    "batching": "One request at a time, because an engineer waits for each draft, so we choose a low wait over a low cost.",
    "slo": "95 percent of drafts (p95) in under 20 seconds and 0 verdicts released without an engineer's approval.",
    "monitoring": "Log the confidence of every photo, the latency p95, the share of drafts with a verdict removed by the rail, and the engineer's rejections.",
    "versioning": "The classifier and the language model files are versioned together, with the test results of that pair stored next to them.",
    "rollback": "Keep the previous pair on disk and switch back with one command if the rejection rate of the engineer doubles in a week.",
    "privacy_guardrails": "The Week 17 verdict rail stays on, the licensed engineer approves every record, and photos are deleted after 90 days.",
    "cost": "About 16 per million tokens at the assumed price of 1.00 per GPU hour, which is small because a draft is only a few hundred tokens.",
}
```



   Running the check on this answer prints:

```text
✓ every field has at least 8 words
✓ slo has a number and p95
✓ cost has a number
✓ precision names a number format
✓ privacy_guardrails names a protection of Week 17
```

6. **Step 7.6. Your honest findings table, and what comes next.** The most useful page of a report is the one with the experiments that did not work. In the first cell below, copy the numbers you measured in Parts 1 to 6 into `my_numbers` (the earlier parts print them; replace each `None`). The cell then prints them next to the numbers of this part.

   *Exercise 7.6:* Fill in `findings` with at least 6 dictionaries with the keys `tried`, `number` and `verdict`, drawing rows from Parts 1 to 7. Include at least one result that did **not** work or surprised you (on a CPU-only machine, half precision is the obvious candidate), at least one row about the communication or the bubble (Part 2 or 3), and one row from Part 7. Why it matters: a row with a number and an honest verdict is what makes a deployment decision checkable. The check tests the structure only.

```python
my_numbers = {   # copy these from your own output of Parts 1 to 6
    "training memory per parameter in bytes, measured (Part 1)": None,
    "gradient average equals the full-batch gradient: largest difference (Part 2)": None,
    "all-reduce bandwidth on this machine in GB/s, and DDP speed-up (Part 2)": None,
    "tensor-parallel split equals the full block: largest difference (Part 3)": None,
    "pipeline bubble: formula and simulation for your p and m (Part 3)": None,
    "bfloat16 against float32 time on this CPU, and on a GPU if you have one (Part 4)": None,
    "int8 dynamic quantization: size, speed-up and output error (Part 5)": None,
    "batch size where throughput stops improving, and the KV cache bytes (Part 6)": None,
}
print("%-82s %s" % ("what", "your number"))
for k_, v_ in my_numbers.items(): print("%-82s %s" % (k_, v_))
print("%-82s %s" % ("70 B: price per million tokens, 4 GPUs 16-bit / 1 GPU 4-bit (Part 7, assumed prices)", "%s / %s" % (RESULTS.get("p7_cost_fp16_4gpu"), RESULTS.get("p7_cost_int4_1gpu"))))
print("%-82s %s" % ("helpdesk: largest batch that fits, 4-bit / 16-bit weights (Part 7)", "%s / %s" % (RESULTS.get("p7_max_batch_int4"), RESULTS.get("p7_max_batch_fp16"))))
findings = []      # YOUR ROWS: {"tried": "...", "number": "...", "verdict": "..."}
```

   It prints:

```text
what                                                                               your number
training memory per parameter in bytes, measured (Part 1)                          None
gradient average equals the full-batch gradient: largest difference (Part 2)       None
all-reduce bandwidth on this machine in GB/s, and DDP speed-up (Part 2)            None
tensor-parallel split equals the full block: largest difference (Part 3)           None
pipeline bubble: formula and simulation for your p and m (Part 3)                  None
bfloat16 against float32 time on this CPU, and on a GPU if you have one (Part 4)   None
int8 dynamic quantization: size, speed-up and output error (Part 5)                None
batch size where throughput stops improving, and the KV cache bytes (Part 6)       None
70 B: price per million tokens, 4 GPUs 16-bit / 1 GPU 4-bit (Part 7, assumed prices) 5.67 / 2.04
helpdesk: largest batch that fits, 4-bit / 16-bit weights (Part 7)                 16 / 7
```

   A possible answer (try first, then compare):

```python
findings = [
    {"tried": "mixed precision on this CPU (Part 4)", "number": "your Part 4 ratio of bfloat16 time to float32 time", "verdict": "it did not speed anything up here, because this CPU has no fast 16-bit arithmetic; a GPU with Tensor Cores behaves differently"},
    {"tried": "int8 dynamic quantization of the wide MLP (Part 5)", "number": "your Part 5 size ratio and speed-up", "verdict": "4 times smaller and faster on this machine, with a small output error: quantization helped where mixed precision did not"},
    {"tried": "all-reduce between processes (Part 2)", "number": "your Part 2 bandwidth in GB/s and DDP speed-up", "verdict": "no real speed-up with gloo on one CPU, because the processes share the same cores and memory; the formula still explains the traffic"},
    {"tried": "the pipeline bubble (Part 3)", "number": "your Part 3 formula and simulated idle share", "verdict": "they matched for equal stages, and unequal stages made it worse, because the slowest stage sets the pace"},
    {"tried": "batching a server (Part 6)", "number": "your Part 6 batch size and KV cache bytes", "verdict": "throughput grew with the batch until memory or arithmetic became the limit, at the price of a longer wait for each request"},
    {"tried": "the capacity planner for a 70 B model (Part 7)", "number": "%s per million tokens with 4 GPUs in 16 bits, %s with 1 GPU in 4 bits (assumed prices)" % (RESULTS.get("p7_cost_fp16_4gpu"), RESULTS.get("p7_cost_int4_1gpu")), "verdict": "4-bit weights were cheaper because they need fewer GPUs, but I must recheck the accuracy on my own questions before I believe it"},
]
print("%-48s | %-44s | %s" % ("tried", "number", "verdict"))
for x in findings: print("%-48s | %-44s | %s" % (x["tried"][:48], x["number"][:44], x["verdict"][:90]))
```

   Rows that say "your Part N" must be replaced by what *you* measured. Keep the negative rows: they are the ones a reader trusts.

   Running the check on this answer prints:

```text
tried                                            | number                                       | verdict
mixed precision on this CPU (Part 4)             | your Part 4 ratio of bfloat16 time to float3 | it did not speed anything up here, because this CPU has no fast 16-bit arithmetic; a GPU w
int8 dynamic quantization of the wide MLP (Part  | your Part 5 size ratio and speed-up          | 4 times smaller and faster on this machine, with a small output error: quantization helped
all-reduce between processes (Part 2)            | your Part 2 bandwidth in GB/s and DDP speed- | no real speed-up with gloo on one CPU, because the processes share the same cores and memo
the pipeline bubble (Part 3)                     | your Part 3 formula and simulated idle share | they matched for equal stages, and unequal stages made it worse, because the slowest stage
batching a server (Part 6)                       | your Part 6 batch size and KV cache bytes    | throughput grew with the batch until memory or arithmetic became the limit, at the price o
the capacity planner for a 70 B model (Part 7)   | 5.67 per million tokens with 4 GPUs in 16 bi | 4-bit weights were cheaper because they need fewer GPUs, but I must recheck the accuracy o
✓ at least 6 rows
✓ every row has tried, number and verdict
✓ every number cell contains a digit
✓ at least one verdict says it did not work or surprised you
✓ at least one row is about communication or the bubble
✓ at least one row is about Part 7
```

   **Reflection questions.**

   Write two or three sentences for each, in a markdown cell, using your own numbers.

   1. Which of the two limits, memory or time, decided your helpdesk plan, and which number in the planner showed it?
   2. Mixed precision, quantization, batching, tensor parallelism, pipelines and data parallelism each solve a different problem. Pick the two that mattered most for **your** hardware and say why the others did not.
   3. Which result of this week did you not expect, and what did you do to check that it was real and not a mistake in your code?
   4. What would you measure on the real hardware before you promise a school a response time (the p95 of Part 6)?
   5. Which of your assumed numbers (a price, a bandwidth, the 50 percent efficiency) would change the decision most if it were wrong by a factor of two? Test it by changing it in `MY_BOX` and running Step 7.3 again.

   **What comes next.**

   You now have the whole chain of the week: the warehouse (memory, Part 1), the trucks that share a load (data, tensor and pipeline parallelism, Parts 2 and 3), the cheaper boxes (mixed precision and quantization, Parts 4 and 5), the delivery desk (batching and the KV cache, Part 6) and the plan that puts them together (Part 7). The lesson of the week is that **a deployment is a set of trade-offs that you can measure**: memory against bits, speed against communication, throughput against waiting, cost against accuracy. Bring your design card and your findings table to **Week 19**, where you build the capstone, and to the **Capstone** itself: the Structural Inspection Assistant needs exactly this plan (what runs where, at what precision, with what response time and with which guardrails from Week 17) and a licensed engineer who still makes every pass or fail decision.

### Troubleshooting

Find your symptom in the left column, then read the cause and the fix. If the same error appears again after the fix, restart the runtime and run the Setup cell and the part again from the top: many errors come from a cell that was run twice or out of order. Remember that **timings depend on the machine**: a laptop CPU, a Colab CPU and a GPU give different numbers, so compare ratios and the properties that the checks test, not the exact milliseconds in the text.

| Symptom | Likely cause | What to do |
|---|---|---|
| `NameError: name 'check' is not defined` (or `RESULTS`, `time_ms`, `param_bytes`, `HAS_CUDA`, `QUICK`) | The Setup cell was not run in this session, or the runtime was restarted. | Run the Setup cell from the top, then the part you want. The first code cell of a part defines `check` and `RESULTS` if they are missing. |
| The hardware report says `GPU: none found`, and the GPU cells print nothing | The runtime has no GPU. In Colab choose Runtime, Change runtime type, and a GPU; on your own machine a GPU needs a PyTorch build with CUDA. | Nothing is wrong: every GPU cell is skipped on purpose and the text says what to expect. Do the CPU parts and read the GPU paragraphs. If you do have a GPU, restart the runtime after changing it and run the Setup again. |
| A GPU cell fails with `CUDA out of memory` | The batch, the sequence length or the model is too big for your GPU, or earlier cells still hold memory. | Lower the batch or the sequence length, run `torch.cuda.empty_cache()` after `del` of big tensors, or restart the runtime. The Part 1 formulas tell you how much memory to expect before you run. |
| Mixed precision (`torch.autocast`, bfloat16, float16) is **slower** than float32 | The CPU has no fast 16-bit arithmetic, so the numbers are converted back and forth. This is the correct result on many laptops and on a CPU-only Colab runtime, and `CPU_BF16_RATIO` in the report shows it. | Report it honestly: on this hardware mixed precision saves memory but not time. On a GPU with Tensor Cores (or a CPU with native bfloat16) it is usually faster; the notebook measures that automatically when `HAS_CUDA` is true. |
| `RuntimeError` or `RuntimeWarning` about `float16` or `bfloat16` not implemented on the CPU | Some operations have no 16-bit version on the CPU in your PyTorch version. | Use bfloat16 instead of float16 on the CPU, or keep the operation in float32 and cast the result, as the Part 4 emulation (`x.half().float()`) does. |
| Your own `round_to_format` disagrees with `torch.float16` on a few values | A rounding edge case: ties must round to the nearest even mantissa, subnormals have a fixed exponent, and values above the largest number become infinity. | Print the three bit patterns (yours, torch, the exact value) for the first value that differs and fix that case first. |
| The multi-process cell (`torch.distributed`, `gloo`) prints a timeout, a port error or `Address already in use` | Another process holds the port, a previous run is still alive, or the platform does not allow child processes (some notebook services and Windows set-ups). | The cell catches the error and prints a message. Run it again (it picks a free port), close old notebooks, and if it still fails read the hand-written data-parallel cells instead: they show the same result without processes. Never run it from a runtime that has little memory. |
| The ranks of the real DDP run print different checksums | The ranks started from different weights (the seed must be set **before** the model is built) or the data shards were not the same on every run. | Set the same seed on every rank before building the model, shard the batch by rank, and compare again to the single-process result with a tolerance of about 1e-5, not exact equality. |
| The all-reduce bandwidth is much lower than you expected, or the DDP run shows no speed-up | `gloo` on a CPU sends data through memory that the processes share with the computation, so there is no extra hardware to gain. | This is the honest result for one machine: report the measured GB/s and use it in the formulas. A real speed-up needs separate GPUs and a fast link. |
| `assert` fails when you compare a split computation with the full one (tensor parallel, gradient average) | Compared with exact equality instead of a tolerance, the shards are unequal (the average must be weighted), or a row split was concatenated instead of summed. | Use `torch.allclose` with `atol` of about 1e-5 for float32. Column splits **concatenate**, row splits **add**. For unequal shards weight each gradient by its share of the examples. |
| The pipeline simulation does not match the bubble formula | The stage times are not equal (the formula assumes equal stages), or the backward pass was left out of one side. | Use equal stage times for the check, then change one stage: the slowest stage sets the pace and the idle share rises above the formula. |
| `quantize_dynamic` prints a deprecation warning, or says that the quantization engine is not available (`NoQEngine`) | In recent PyTorch versions the quantization tools are moving to the `torchao` library, and some platforms (for example some ARM builds) ship only some engines. | The warning is harmless: the lab filters it and the code still works. If the engine is missing, run `torch.backends.quantized.supported_engines` and set `torch.backends.quantized.engine` to one of the names it lists. |
| `quantize_dynamic` does not make the model smaller or faster | Only `nn.Linear` layers are quantized (convolutions and attention written by hand are not), and a tiny layer is dominated by overhead. | Check which layers are `Linear`, look at the `state_dict` size, and time the **wide** MLP of Part 5 at batch 1, where the weights are read from memory and the gain is clearest. |
| The int8 model runs on the CPU only, or fails on `cuda` | PyTorch's dynamic quantization is a CPU feature. | Quantize on the CPU. For a GPU you would use a library made for GPU quantization; the lab only teaches the idea and the arithmetic. |
| Timings change from run to run, or a ratio in your table is far from the text | Other programs use the CPU, the machine throttles when hot, or the thread count differs. | Close other programs, repeat the measurement, use the median of several runs, and say how many threads you used (`torch.get_num_threads()`). A ratio is more stable than a time. |
| Part 7: `p7_memory` gives a number that is 8 or 1,000 times too small or too large | Bits and bytes were mixed up (divide bits by 8), or gigabytes were computed with 1,024 instead of 1,000. | The lab uses 1 GB equal to 10 to the power 9 bytes and `bits / 8` bytes per value. Recompute the 7 B at 16 bits case by hand: 14 GB. |
| Part 7: the planner says a model does not fit when you expected it to | Your `gpu_gb` is too small, the 10 percent reserve is subtracted, or the KV cache of a large batch and context is bigger than you thought. | Print `weights_gb`, `kv_gb` and `usable_gb`. Lower the batch or the context, use fewer bits, or add GPUs, and see which change fixes it. |
| Part 7: your cost per million tokens is very different from the text | You changed `MY_BOX`, the efficiency `eff`, or the price per GPU hour (all assumed). | A different number is a different experiment: say which assumption you changed. The lesson is how strongly the cost depends on these assumptions. |
| An exercise check prints `Fill in ... above` | The starter still has `None`, an empty string or an empty list. | Replace it with your answer and run the check cell again. |
| The check fails on a design card or a findings table | A field is too short, a number is missing, or a key has the wrong name. | Read the hint after the cross. The checks test the structure only: they never judge the content. |
| Your answer cells run, but later cells fail | You ran a cell twice, or an exercise function still returns `None`. | Run the "possible answer" cell below the exercise to restore a working version, then continue. |
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions, most of them short calculations with the numbers of this lesson. Work them out on paper first.

```quiz
{
  "id": "week-18-check",
  "title": "Week 18 quiz: Scaling and deploying AI",
  "passMark": 70,
  "questions": [
    { "q": "How much memory do the weights of a 13-billion-parameter model need in FP16?", "options": ["13 GB", "26 GB", "52 GB", "6.5 GB"], "answer": 1, "why": "Memory = parameters × bytes = 13 × 10⁹ × 2 = 26 GB. 13 GB would be one byte per weight (INT8), 52 GB would be FP32 and 6.5 GB would be INT4." },
    { "q": "Training a 30-billion-parameter model with Adam in mixed precision needs about 16 bytes per parameter. How much memory is that, before the activations?", "options": ["60 GB", "120 GB", "480 GB", "240 GB"], "answer": 2, "why": "16 × 30 × 10⁹ = 480 × 10⁹ bytes = 480 GB. 60 GB is 2 bytes per parameter (just the FP16 weights), 120 GB is 4 bytes, and 240 GB is 8 bytes." },
    { "q": "In one data-parallel step each GPU needs 300 ms to compute and then waits 100 ms for the all-reduce. What is the efficiency, compute ÷ (compute + communication)?", "options": ["0.75", "0.33", "0.25", "1.33"], "answer": 0, "why": "300 ÷ (300 + 100) = 0.75. 0.33 and 0.25 are the communication share inverted or misread, and an efficiency can never be above 1." },
    { "q": "The gradients of a 4-billion-parameter model are stored in 16 bits (S = 8 GB). With N = 4 GPUs, how much traffic does each GPU send and receive in a ring all-reduce, 2(N − 1)/N × S?", "options": ["6 GB", "8 GB", "16 GB", "12 GB"], "answer": 3, "why": "2 × 3/4 × 8 = 12 GB. 8 GB forgets the factor 2(N − 1)/N, 6 GB uses (N − 1)/N alone, and 16 GB uses 2S, the limit for very large N." },
    { "q": "X = (2, 1) is multiplied by W = [[1, 0, 3, 1], [2, 1, 0, 1]] with a column split over two GPUs. GPU 1 holds the first two columns of W. What does GPU 1 compute?", "options": ["(4, 1)", "(6, 3)", "(10, 4)", "(3, 6)"], "answer": 0, "why": "GPU 1 computes X times the first two columns: (2·1 + 1·2, 2·0 + 1·1) = (4, 1). (6, 3) is GPU 2's piece, (10, 4) would be the sum of the two pieces, and a column split joins by concatenation, not by adding." },
    { "q": "A pipeline has p = 4 stages and the batch is cut into m = 12 micro-batches. What share of the time is each GPU idle, (p − 1)/(m + p − 1)?", "options": ["25%", "15%", "20%", "30%"], "answer": 2, "why": "3 ÷ (12 + 3) = 3 ÷ 15 = 0.20 = 20%. 25% would be m = 9, and 15% and 30% do not match the formula." },
    { "q": "A 70 B model must be trained over two machines joined by an ordinary network, and each machine holds 8 GPUs with fast links. Which split is the best fit for the connection between the two machines?", "options": ["Pipeline parallelism between the machines", "Tensor parallelism across the network", "Pure data parallelism with a full copy on every GPU", "Quantizing the gradients to 4 bits"], "answer": 0, "why": "Only the activations at stage boundaries cross the slower network in a pipeline. Tensor parallelism needs an all-reduce in every layer and would stall on a slow link, and a full copy of a 70 B model with its optimiser state does not fit on one GPU." },
    { "q": "A gradient of 2 × 10⁻⁹ is stored in FP16 without loss scaling. What is stored?", "options": ["2 × 10⁻⁹ with a small rounding error", "inf", "65,504", "0"], "answer": 3, "why": "The smallest FP16 subnormal is about 6 × 10⁻⁸, and 2 × 10⁻⁹ is well below half of it, so the value underflows to 0. FP16 does not overflow for small numbers, and 65,504 is its largest value." },
    { "q": "Which loss scale S would lift a gradient of 10⁻⁸ to about 10⁻⁵ so that FP16 can store it?", "options": ["10", "1,024", "100", "1,000,000"], "answer": 1, "why": "10⁻⁸ × 1,024 = 1.02 × 10⁻⁵. S = 10 gives 10⁻⁷ and S = 100 gives 10⁻⁶, which are not the 10⁻⁵ asked for, and S = 10⁶ gives 10⁻², which would risk overflow for larger gradients." },
    { "q": "Symmetric 4-bit quantization of weights whose largest absolute value is 0.8. What is the scale?", "options": ["0.8 ÷ 8 = 0.100", "0.8 ÷ 15 = 0.053", "0.8 ÷ 7 = 0.114", "0.8 ÷ 16 = 0.050"], "answer": 2, "why": "A 4-bit symmetric scheme uses integers from −7 to 7, so the scale is max|w| ÷ (2³ − 1) = 0.8 ÷ 7 = 0.114. The other options divide by the number of levels (16), by 15 or by 8, which do not match the sign bit." },
    { "q": "8-bit symmetric quantization of weights with largest absolute value 1.27 gives a scale of 0.01. What is the largest possible error of any weight?", "options": ["0.005", "0.01", "0.0079", "0.05"], "answer": 0, "why": "Rounding to the nearest integer is off by at most half a step: scale ÷ 2 = 0.005. A full step (0.01) would be the error of truncating, not of rounding to nearest." },
    { "q": "4-bit weights with one 16-bit scale for every group of 32 weights. How many bits does each weight really cost?", "options": ["4.0", "4.25", "5.0", "4.5"], "answer": 3, "why": "bits + 16 ÷ g = 4 + 16 ÷ 32 = 4.5. 4.25 is the value for a group of 64, and 4.0 ignores the scales." },
    { "q": "A model of 4 GB in memory is decoded on hardware with a memory bandwidth of 100 GB/s (assumed). What is the upper bound on tokens per second for one sequence?", "options": ["4", "25", "400", "0.04"], "answer": 1, "why": "Each token reads all 4 GB, so the bound is 100 ÷ 4 = 25 tokens per second. 400 inverts the division by mistake and 4 and 0.04 do not correspond to either ratio." },
    { "q": "A decoder has 24 layers and width 2,048. What is the KV cache for 4 sequences of 2,048 tokens with a 16-bit cache, 2 × L × n × d × bytes × batch?", "options": ["1.61 GB", "0.81 GB", "0.40 GB", "3.22 GB"], "answer": 0, "why": "2 × 24 × 2,048 × 2,048 × 2 × 4 = 1.61 × 10⁹ bytes. 0.81 GB forgets the factor 2 for keys and values, 0.40 GB also forgets the batch, and 3.22 GB uses 4 bytes per value." },
    { "q": "A batch of 4 requests takes t₀ + c·b with t₀ = 20 ms and c = 10 ms, and requests arrive at 40 per second. What is the average latency, including the wait for the batch to fill, (b − 1)/(2λ), plus the service time?", "options": ["60 ms", "37.5 ms", "97.5 ms", "150 ms"], "answer": 2, "why": "Service = 20 + 10 × 4 = 60 ms. Wait = 3 ÷ (2 × 40) = 0.0375 s = 37.5 ms. Latency = 37.5 + 60 = 97.5 ms. 60 ms forgets the wait and 37.5 ms forgets the service." }
  ]
}
```

## How Week 18 is assessed

**Your lab (100 points).** You submit the notebook with its outputs (or your JupyterLab files with the printed output) and answer five written questions. The questions are the assessment: each one asks for **your own measured numbers** from the notebook, so keep the output of the cells named below (a screenshot of the printed result is enough; do not retype numbers you did not see). Memory sizes, parameter counts, formulas and equality checks are counted exactly and repeat on every machine; **times, bandwidths and speed-ups depend on your hardware** (a laptop CPU, a Colab CPU or a GPU), so for those you quote **your measured value** and say which machine produced it. An answer that reports your own, different, timing and reads it correctly (for example "bfloat16 was slower on my CPU, so mixed precision saved memory but not time") earns full marks. Answers that repeat a number from the text without running the cell do not. If you have no GPU, the GPU cells are skipped and you are not penalised: say that you had none. All prices, bandwidths and memory sizes in Part 7 are assumptions that you chose; the school, the helpdesk and the inspection assistant are invented for the lab.

**What to keep from each part (screenshots or copied numbers).**

| Part | Keep the output of | The numbers to have at hand |
|---|---|---|
| 1. Memory and time | the parameter count of your small Transformer next to the formula; the training-memory sum (weights, gradients, optimiser state); the activation-memory plot against batch size and sequence length; the memory table for 7 B and 70 B; the time estimate from your machine's measured FLOPs per second; the four memory-or-time decisions | parameters counted and predicted; bytes per parameter measured for one AdamW step (compare with 16); how the activation bytes grew with batch and with sequence length; 28, 14, 7 and 3.5 GB for 7 B and 112 GB for training it; 140 GB for 70 B at 16 bits and the number of 80 GB GPUs; your measured FLOPs per second and the time that 6ND implies for 7 B on 1 T tokens on your machine and on 1,024 assumed GPUs |
| 2. Data parallelism | the shard-gradient check (equal and unequal shards); the one-step equality check; the real multi-process run (2 and 4 processes) with the parameter checksums of the ranks; the all-reduce time against size plot; the predicted and measured step time; the speed-up plot for three link speeds | the largest difference between the averaged shard gradients and the full-batch gradient; why unequal shards need weights; whether all ranks agreed and matched single-process training; your measured all-reduce bandwidth in GB/s; traffic 2(N - 1)/N times S; the efficiency compute over compute plus communication; what happened to the step time with 2 and 4 processes on your machine |
| 3. Tensor and pipeline parallelism | the column-split and row-split checks; the split feed-forward block and the attention heads on 1, 2 and 4 simulated devices; the pipeline output check; the measured stage times; the Gantt chart; the bubble table; the decision table | the largest difference between the split and full outputs; the number of all-reduces per block; the memory per device (weights divided by N) and the size of one all-reduce; the simulated idle share next to (p - 1)/(m + p - 1) for equal stages, and what unequal stages did; your answers for the three checked situations |
| 4. Mixed precision | the decoded bits of the three formats; the check of your `round_to_format`; the range and precision table; the overflow and underflow runs; the loss-scaling comparison; the real autocast timing (and the GPU results if you have one) | largest values and machine epsilons (65,504 and 6.1e-5; 9.77e-4 for float16, 7.8e-3 for bfloat16); what 70,000 and 1e-8 became in float16; the number of zero gradients and the final loss with and without loss scaling; your `CPU_BF16_RATIO` and the autocast slowdown or speed-up on your machine; the saving of activation memory |
| 5. Quantization | the check that your `quantize` matches the lesson; the error against bits for the Gaussian and outlier matrices; the per-tensor, per-row and per-group table; the accuracy against bits curve; the `quantize_dynamic` size, latency and error; the per-layer sensitivity; the memory-bound check | MSE at 8, 6, 4, 3 and 2 bits and the effect of outliers; bits per weight bits plus 16/g; the bit width where your classifier's accuracy fell off; size ratio (about 4) and speed-up at batch 1, 8 and 64; the layer that hurt most; your measured bandwidth and the tokens per second bound; the INT4 sizes for 7 B and 70 B |
| 6. Serving | the batch-size timing table and the fit of t0 plus c times b; the server simulation (p50, p95, p99, throughput); the KV-cache equality check and the time per token with and without the cache; the cache bytes check; the memory budget and the maximum batch; the cost table | your t0 and c; the batch size that gave the lowest p95 under the target; the latency and throughput as the arrival rate grew past capacity; the cache bytes and the check against 2 times L times n times d times bytes times batch; the maximum batch for an assumed 80 GB GPU at 16-bit and 4-bit weights; the cost per million tokens for your assumed price |
| 7. Plan the deployment | the passing checks of `p7_memory`; the three deployment lines of the planner; the batch sweep table and plot; the printed checklist; your design card; your findings table | weights and cache in GB for each deployment and whether it fits; the assumed hardware numbers you used (`MY_BOX`); the largest batch that fits with 4-bit and with 16-bit weights; the price per million tokens for 4 GPUs at 16 bits and for 1 GPU at 4 bits (with the prices you assumed); your nine-field card and at least six findings rows |

**The five questions (100 points).** Write each answer in full sentences under its question, in your own words. Where a question says "your measured value", quote it with its unit and say what machine it came from.

| # | Question | Marks | What a full-mark answer contains |
|---|---|---|---|
| 1 | **The warehouse: memory and time** (Part 1). Explain what fills the memory of a model and how long training takes, using your own measurements. | 20 | **(5)** the weights: your counted parameters next to 12 times L times d squared plus the embeddings, and the bytes at 32 and 16 bits. **(5)** training memory: the bytes of parameters, gradients and optimiser state that you summed for one AdamW step, the bytes per parameter you found next to the lesson's 16 (and the two accountings: 4 plus 4 plus 8 in 32-bit, and the mixed-precision count), and what 16 bytes means for 7 B (112 GB). **(5)** activations: how the memory measured by your hooks grew with the batch and with the sequence length, and one sentence on why that is a third thing to budget next to weights and optimiser state. **(5)** time: your measured FLOPs per second, the time that C of about 6ND gives for 7 B on 1 T tokens on your machine (absurdly long) and on 1,024 assumed GPUs, and the verdict for one of the four scenarios: memory or time? |
| 2 | **Many devices, one model** (Parts 2 and 3). Show how data, tensor and pipeline parallelism each split the work, what each costs in communication, and what you measured. | 25 | **(6)** data parallelism: why the average of equal-shard gradients equals the full-batch gradient (the largest difference you measured), why unequal shards need weights, and whether the ranks of your real run agreed with single-process training. **(6)** communication: your measured all-reduce bandwidth, the traffic 2(N - 1)/N times S for one size, the predicted efficiency, and an honest sentence on why gloo on one machine gave no real speed-up (your measured step times). **(6)** tensor parallelism: column splits concatenate and row splits add; the Megatron-style block needs one sum for two matrices; the memory per device and the size of the all-reduce from your output. **(7)** pipeline parallelism: the simulated idle share next to (p - 1)/(m + p - 1) for your p and m, what unequal stage times did, and your decision (data, tensor, pipeline or a combination, with reasons) for one situation of the decision table |
| 3 | **Fewer bits** (Parts 4 and 5). Compare mixed precision and quantization: what each saves, what each costs, and what you measured on your hardware. | 25 | **(6)** the formats: the sign, exponent and mantissa bits of float32, float16 and bfloat16, the largest value and epsilon of each, and why float16 needs loss scaling and bfloat16 does not (range against precision). **(6)** mixed-precision training: what happened to the gradients and the final loss with and without loss scaling, with your counts of zero gradients, and how your `LossScaler` reacted to an overflow. **(6)** the hardware: your `CPU_BF16_RATIO` and the autocast timing on your machine (and the GPU numbers if you had one), and the honest verdict on what mixed precision saved here (memory, time, accuracy). **(7)** quantization: the error against bits with and without outliers, how per-group scales helped, the bit width where your classifier's accuracy fell off, and the size and speed-up of `quantize_dynamic` at batch 1 and 64, with one sentence on why a quantized model can be both smaller and faster when decoding is memory-bound |
| 4 | **The delivery desk** (Part 6). Explain batching, queues and the KV cache with your own measurements. | 15 | **(4)** batching: your fitted t0 and c, what the line says about the cost of one extra request, and the batch size where throughput stopped improving. **(4)** the queue: your p50, p95 and p99 for one setting, the batch size that gave the lowest p95 under the target, and what happened when the arrival rate went above capacity. **(4)** the KV cache: the check that cached and uncached generation give identical outputs, the time per token with and without the cache, and your measured cache bytes next to 2 times L times n times d times bytes times batch. **(3)** the budget: the maximum batch for an assumed 80 GB GPU at 16-bit and 4-bit weights, and the cost per million tokens from your assumed price (flagged as assumed) |
| 5 | **Plan the deployment** (Part 7). Use the capacity planner and the checklist to argue for one deployment, and say what you do not know. | 15 | **(4)** the planner: for one of the three deployments, the weights, the KV cache, whether it fits and the advice it gave, with the hardware numbers you assumed (`MY_BOX`) flagged as assumed. **(4)** the trade-off: the price per million tokens for 4 GPUs at 16 bits against 1 GPU at 4 bits, and the largest batch with 4-bit and 16-bit weights, with the reason that smaller weights leave room for more cache. **(4)** your design card: precision, parallelism, batching, an SLO with a p95 number, monitoring, versioning, rollback, privacy and the Week 17 guardrails, and cost, with one sentence that links one field to a number you measured in Parts 1 to 6. **(3)** the honest part: one result that surprised you or did not work (for example half precision on a CPU), and the one assumption that would change your decision most if it were wrong by a factor of two |

**Reference values for the instructor and for self-checking.** Formula values do not depend on the machine: 7 B parameters need 28, 14, 7 and 3.5 GB at 32, 16, 8 and 4 bits, training it needs about 112 GB with 16 bytes per parameter, and 70 B needs 140 GB at 16 bits and 1,120 GB for training; float16 has a largest value of 65,504, a smallest normal number of about 6.1e-5 and an epsilon of about 9.77e-4, bfloat16 has an epsilon of about 7.8e-3; the pipeline idle share is (p - 1)/(m + p - 1); all-reduce traffic is 2(N - 1)/N times the data size. Part 7 with the assumed numbers of the lab (a 24 GB box at 300 GB/s and 1.00 per GPU hour; 80 GB GPUs at 2,000 GB/s, a link of 300 GB/s and 2.00 per GPU hour; half of the speed bound reached): the helpdesk model (7 B, 4.25 bits, 2,048 tokens, batch 4) needs 3.7 GB of weights and 4.3 GB of cache, fits on one GPU and costs about 3.71 per million tokens; the 70 B district model needs 140.0 GB of weights and 21.5 GB of cache (grouped-query attention with 0.125 of the keys and values) on 4 GPUs at about 5.67 per million tokens, or 37.2 GB of weights on 1 GPU at 4.25 bits at about 2.04; the inspection assistant (a 25 million parameter classifier and the 7 B model, 4,096 tokens) needs about 5.97 GB; the largest helpdesk batch is 16 with 4.25-bit weights and 7 with 16-bit weights. **Timings, bandwidths and speed-ups** (the all-reduce bandwidth, the DDP step times, the bfloat16 and float16 ratios, the quantization speed-up, t0 and c) depend on the machine: on the machine that ran this lab (a six-core laptop CPU without a GPU) bfloat16 matrix products were slower than float32 by a large factor, mixed precision was slower, and int8 dynamic quantization was faster and about four times smaller; a different machine can show different ratios, and a mark is given for reading **your** numbers correctly, not for matching these.


## Exit ticket and homework

Three questions before you leave. Answer each in two or three sentences.

1. A friend says: "I will make my training faster by putting the model on 8 GPUs with data parallelism, and then I can train a model eight times larger." What is right and what is wrong? Use the words memory and time.
2. You quantize a 7 B model from FP16 to 4 bits and decoding gets faster, but mixed precision made your CPU run slower. Explain both results with one idea each (bytes read, and hardware support).
3. Your helpdesk model answers one student in 45 ms but cannot keep up with 50 requests a second. Name one change that raises capacity and one thing it costs.

**Homework:** finish the lab and write down the numbers you measured. Re-read the `equation` figures of this lesson and try to redo each worked example without looking. Preview Week 19, the capstone build week and the mock NCA-GENL exam. Keep working on your capstone: write down which of the checks of the capacity-planning recipe your capstone must pass.
