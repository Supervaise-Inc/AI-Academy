> Content based on the NVIDIA Generative AI and DLI Deep Learning Teaching Kits, used under the Supervaise × NVIDIA partnership.

# Week 13 — Diffusion Models

**Purpose:** Learn how a network can make a picture by ==starting from pure noise and removing a little of it at a time==, and why the network that does the removing is a UNet that is told how noisy its input is.

**Term:** Pre-final (Weeks 11–15) · **Month:** 4 · **Time:** about 60 minutes for the lesson, then a 3-hour lab

```viz
{
  "type": "hook",
  "title": "Watch noise turn into a picture",
  "caption": "The house and the loss curve are made up to show the idea; a real diffusion model gives different pictures and numbers. The signal and step numbers in the other tabs are exact arithmetic that you will redo in this lesson and in the lab.",
  "scenes": [
    { "name": "Noise to picture", "kind": "denoise", "prompt": "A diffusion model starts from pure static and removes a little noise at every step.", "outputLabel": "A picture appearing out of noise (a simple house, illustrative)", "image": ["000000000000","000009900000","000099990000","000999999000","009999999900","099999999990","000999999000","000990099000","000990099000","000990099000","000999999000","000000000000"] },
    { "name": "Fog on the photo", "kind": "bars", "prompt": "How much of the original picture is left after t of 1,000 noising steps?", "outputLabel": "Share of the picture kept, the square root of alpha-bar (rounded)", "bars": [ { "label": "t = 100", "value": 0.95 }, { "label": "t = 300", "value": 0.63 }, { "label": "t = 500", "value": 0.28 }, { "label": "t = 1000", "value": 0.01 } ] },
    { "name": "A real number", "kind": "text", "prompt": "The noise schedule keeps a fraction ᾱ of the picture’s strength at step t.", "outputLabel": "Computed with the linear schedule of this lesson", "output": "At t = 300 the fraction is ᾱ = 0.3964, so the clean picture is multiplied by 0.63 and fresh noise is added with weight 0.78: the noise already outweighs the picture. At t = 1000, ᾱ is about 0.00004, so the picture is multiplied by 0.0064 and what is left is, for practical purposes, pure static." },
    { "name": "Learning to denoise", "kind": "loss", "prompt": "The network is trained to guess the noise that was added. The loss is a plain squared error.", "outputLabel": "Noise-prediction loss over training (illustrative)" },
    { "name": "The price", "kind": "text", "prompt": "To draw one picture the model must run its network once per denoising step.", "outputLabel": "Computed: network calls for one picture", "output": "A GAN (Week 12) makes a picture with 1 pass through its generator. The original diffusion sampler takes 1,000 steps, so it calls the network 1,000 times: 1,000 times the work. Faster samplers get good pictures in about 20 to 50 calls, which is still 20 to 50 times more than a GAN." }
  ]
}
```

## By the end, you can…

1. ==Describe the forward process and compute x_t from x₀== for a given step t, using the one-line closed form, and describe the reverse process that undoes it.
2. ==Explain the noise schedule and ᾱ_t==: compute ᾱ as a running product, read how much picture and how much noise are left at step t, and compare the linear and cosine schedules with the signal-to-noise ratio.
3. ==Write the training loss and say why predicting the noise is a good target==: one network, one squared error, a random step per example, no adversary.
4. ==Describe one reverse step and the sampling loop==, say why a little fresh noise is added, and say how DDIM reaches a picture in far fewer steps.
5. ==Explain why the denoiser needs the time step and why a UNet with skip connections suits it==: same shape in and out, fine detail, global context and a time embedding.
6. ==Explain in words classifier-free guidance, latent diffusion and how diffusion compares with GANs==, and say what part diffusion could and could not play in the capstone.

Assessed in the Week 13 lab (“Imagine the sculptor and the fog”) and the quiz. Read each section, then play with the figure under it: the figures are practice, so you cannot break anything. Terms with a dotted underline open a short definition when you select them. The lab refers back to this lesson: look for pointers such as “Lab Part 1”.

## The forward process: adding noise {rail=Forward}

{lede} A [[diffusion model|A generative model that learns to undo a slow, step-by-step noising of data. It makes new data by starting from random noise and removing a little noise at each step.]] begins with a strange first move: ==destroy the picture on purpose, a little at a time, until nothing is left but static==. That destruction is called the [[forward process|The fixed recipe that turns a clean picture into noise by adding a small amount of Gaussian noise at each of T steps. Nothing in it is learned.]]. It needs no learning at all. The learning is in running it backwards.

### The fog on the window

Picture a photograph of the Mayon Volcano that you hold up to a window during a heavy rainstorm. Each minute the glass fogs a little more. After one minute you can still see every detail. After ten you see the cone and the clouds, but the grass is gone. After an hour you see only a grey blur. If you could keep going you would reach a screen of television static where no picture is left.

Now imagine a patient artist who has watched that fog thousands of times. Shown any foggy window, the artist can say: “This is the fog from one minute earlier, with a little more fog added.” Give the artist a window of pure static and ask them to undo the fog minute by minute, and a picture appears. ==That artist is the neural network of this lesson==, and each “minute” is one of T steps.

There is a second way to see it. A sculptor looks at a block of marble and sees a statue inside; each tap of the chisel removes a little stone. ==Diffusion is the sculptor run on noise==: the block is pure static, each tap removes a little noise, and the statue is a picture. The sculptor knows what statues look like because the network has seen many thousands of pictures being buried in fog.

```viz
{
  "type": "pipeline",
  "title": "From a photograph to static, step by step",
  "intro": "Step through the forward process: the same picture, a little more buried at each stage. The numbers belong to one pixel with the value 1.0 and the noise drawn once.",
  "nodes": [
    { "id": "x0", "label": "x₀", "sub": "clean picture" },
    { "id": "x1", "label": "x₁", "sub": "a pinch of noise" },
    { "id": "x2", "label": "x₂", "sub": "a little more" },
    { "id": "dots", "label": "…", "sub": "998 more steps" },
    { "id": "xT", "label": "x_T", "sub": "pure static" }
  ],
  "steps": [
    { "title": "Start with a real picture", "body": "x₀ is a picture from the training set, for example a photo of a school building scaled so every pixel lies between −1 and 1.", "active": ["x0"], "example": "One pixel: x₀ = 1.0." },
    { "title": "Add a pinch of noise", "body": "To get x₁, shrink the picture very slightly and add a small amount of Gaussian noise. At the start of the linear schedule the noise is tiny, so x₁ looks almost the same as x₀.", "active": ["x0", "x1"], "example": "β₁ = 0.0001: the noise has weight √0.0001 = 0.01." },
    { "title": "Keep going", "body": "Each step takes the previous result and does the same thing again, with slightly more noise each time. After many steps the picture is faded and noisy.", "active": ["x1", "x2", "dots"], "example": "At t = 300 the picture keeps only 0.63 of its strength." },
    { "title": "End in pure static", "body": "After T = 1,000 steps almost nothing of x₀ is left, and x_T looks like a fresh draw of Gaussian noise. That is the key fact: the end of the forward process is a distribution we can sample from.", "active": ["dots", "xT"], "example": "At t = 1,000 the picture keeps 0.0064 of its strength." }
  ]
}
```

Notice the direction of every arrow: from clean to noisy. During training the network always sees this direction. At the end of the lesson it will walk the other way, from static to picture.

### One noising step

How exactly is “a little more noise” added? Each step has a small number β_t (beta, between 0 and 1) that says how much noise to add at step t. The noisy picture x_t is made from the previous one x_(t−1) by shrinking it a bit and mixing in fresh noise ε (epsilon). The noise is [[Gaussian noise|Random numbers from a bell curve. Standard Gaussian noise has mean 0 and spread (standard deviation) 1, written N(0, 1). In a picture, every pixel gets its own independent draw.]]: every pixel gets its own independent draw from a bell curve with mean 0 and spread 1.

```viz
{
  "type": "equation",
  "title": "One forward step",
  "intro": "Select each symbol to read what it means, then follow the worked example for a single pixel.",
  "formula": [ { "t": "x_t", "id": "xt" }, " = ", { "t": "√(1 − β_t)", "id": "keep" }, " · ", { "t": "x_(t−1)", "id": "xprev" }, " + ", { "t": "√β_t", "id": "add" }, " · ", { "t": "ε", "id": "eps" } ],
  "parts": [
    { "id": "xt", "name": "x_t (the noisier picture)", "meaning": "The picture after step t. It has the same shape as x₀: one number per pixel (and per colour channel)." },
    { "id": "keep", "name": "√(1 − β_t) (how much to keep)", "meaning": "A number just below 1 that shrinks the previous picture slightly. With β_t = 0.02 it is 0.98995." },
    { "id": "xprev", "name": "x_(t−1) (the picture before the step)", "meaning": "The result of the previous step. For t = 1 it is the clean picture x₀." },
    { "id": "add", "name": "√β_t (how much noise to add)", "meaning": "The weight of the fresh noise. A small β_t means a small weight. With β_t = 0.02 it is 0.14142." },
    { "id": "eps", "name": "ε (fresh noise)", "meaning": "A new draw from the standard Gaussian N(0, 1), one independent number per pixel. It is drawn again at every step." }
  ],
  "steps": [
    { "title": "Choose the numbers", "expr": "x_(t−1) = 0.8, β_t = 0.02, ε = 0.5", "note": "One pixel. The value 0.02 is the largest β in the linear schedule; early steps use much smaller values." },
    { "title": "Shrink the old value", "expr": "√(1 − 0.02) × 0.8 = 0.98995 × 0.8 = 0.79196", "note": "The picture’s strength falls by about 1 percent." },
    { "title": "Add the noise", "expr": "√0.02 × 0.5 = 0.14142 × 0.5 = 0.07071", "note": "The noise has weight 0.14, so a draw of 0.5 moves the pixel by about 0.07." },
    { "title": "Add them up", "expr": "x_t = 0.79196 + 0.07071 = 0.86267", "note": "The pixel moved from 0.8 to 0.86267: a small push in the direction of the noise." },
    { "title": "Why these two weights", "expr": "(1 − β_t) + β_t = 1", "note": "The squares of the weights add to 1, so if the picture has spread 1, the new picture also has spread 1: the numbers do not slowly blow up or shrink away." }
  ]
}
```

Notice how gentle one step is: the pixel moved by 0.06 on a scale where a pixel can be anywhere from −1 to 1. ==One step ruins almost nothing; a thousand of them ruin everything.== The forward process is a [[Markov chain|A sequence in which each item depends only on the one just before it. Here x_t is made from x_(t−1) and fresh noise, and the older history does not matter.]]: each step uses only the previous picture and new randomness.

### The noise schedule

The list of numbers β₁, β₂, …, β_T is the [[noise schedule|The list of numbers β_1 … β_T that sets how much noise is added at each step. A common choice rises in a straight line from 0.0001 to 0.02 over T = 1,000 steps.]]. A common choice, from the original DDPM paper by Ho, Jain and Abbeel (2020), is a straight line: β₁ = 0.0001, rising evenly to β₁₀₀₀ = 0.02. ==Early steps add very little noise and late steps add more.== Two helper quantities come from β.

- **α_t = 1 − β_t** is the fraction kept at step t. It is just below 1.
- **ᾱ_t** (“alpha-bar”) is the running product α₁ × α₂ × … × α_t. It says how much of the original picture’s strength survives after t steps in a row.

```viz
{
  "type": "equation",
  "title": "From β to ᾱ",
  "intro": "Select each symbol, then multiply the first three steps of the linear schedule by hand.",
  "formula": [ { "t": "ᾱ_t", "id": "abar" }, " = ", { "t": "α_1", "id": "a1" }, " · ", { "t": "α_2", "id": "a2" }, " ⋯ ", { "t": "α_t", "id": "at" }, ",   where ", { "t": "α_s", "id": "as" }, " = 1 − ", { "t": "β_s", "id": "beta" } ],
  "parts": [
    { "id": "abar", "name": "ᾱ_t (alpha-bar)", "meaning": "The product of all the α values up to step t. It starts near 1 (nothing lost yet) and falls towards 0 (nothing left)." },
    { "id": "a1", "name": "α_1", "meaning": "The fraction kept at step 1. With the linear schedule it is 1 − 0.0001 = 0.9999." },
    { "id": "a2", "name": "α_2", "meaning": "The fraction kept at step 2, slightly smaller than α_1 because β grows." },
    { "id": "at", "name": "α_t", "meaning": "The fraction kept at step t. Multiplying many numbers just below 1 gives a number that shrinks steadily." },
    { "id": "as", "name": "α_s (any step s)", "meaning": "One step’s keep fraction, equal to 1 − β_s." },
    { "id": "beta", "name": "β_s (noise level at step s)", "meaning": "The schedule value: 0.0001 at s = 1 rising in a straight line to 0.02 at s = 1000, so β_s = 0.0001 + 0.0199 × (s − 1)/999." }
  ],
  "steps": [
    { "title": "The first three β", "expr": "β_1 = 0.0001, β_2 = 0.00011992, β_3 = 0.00013984", "note": "Each is the previous one plus 0.0199/999 = 0.0000199." },
    { "title": "The three α", "expr": "α_1 = 0.9999, α_2 = 0.99988008, α_3 = 0.99986016", "note": "Each α is simply 1 minus its β." },
    { "title": "Multiply", "expr": "ᾱ_3 = 0.9999 × 0.99988008 × 0.99986016 = 0.99964028", "note": "After three steps 99.96 percent of the strength is left." },
    { "title": "Keep multiplying", "expr": "ᾱ_100 = 0.8970, ᾱ_300 = 0.3964, ᾱ_1000 = 0.0000404", "note": "A computer does these 1,000 multiplications in a blink. Notice how the small losses pile up." }
  ]
}
```

Why a product? Each step multiplies the picture by √α_t, so ==after t steps the picture has been multiplied by √(α₁ α₂ ⋯ α_t) = √ᾱ_t==. Play with the schedule below: move the slider and watch ᾱ fall.

```viz
{
  "type": "schedule",
  "title": "The linear noise schedule",
  "intro": "Drag the step slider. The curves show √ᾱ_t (the weight of the picture), √(1 − ᾱ_t) (the weight of the noise) and ᾱ_t itself.",
  "T": 1000,
  "kind": "linear",
  "t": 300,
  "note": "These are exact values for the linear schedule β from 0.0001 to 0.02 over T = 1,000 steps, computed in the browser.",
  "checks": [ { "kind": "linear", "t": 300, "abar": 0.3964 }, { "kind": "linear", "t": 1000, "abar": 4.04e-5 }, { "kind": "linear", "t": 100, "abar": 0.8970 } ]
}
```

At t = 300 the picture curve and the noise curve are at 0.63 and 0.78: the noise already has the larger weight. At t = 1,000 the picture curve is nearly flat on the floor. In Lab Part 1 you build this schedule yourself and assert ᾱ_300 = 0.3964 and ᾱ_1000 = 4.04e-5.

### The shortcut: jump straight to step t

A thousand sequential steps would make training painfully slow if you had to loop through them to get x_t. Luckily you do not. Because every step adds Gaussian noise, ==the whole chain collapses into a single formula that jumps from x₀ to any x_t in one line==.

```viz
{
  "type": "equation",
  "title": "The closed form: x_t straight from x₀",
  "intro": "Select each symbol. The worked example uses t = 300, and then a three-pixel picture.",
  "formula": [ { "t": "x_t", "id": "xt" }, " = ", { "t": "√ᾱ_t", "id": "sig" }, " · ", { "t": "x₀", "id": "x0" }, " + ", { "t": "√(1 − ᾱ_t)", "id": "noi" }, " · ", { "t": "ε", "id": "eps" } ],
  "parts": [
    { "id": "xt", "name": "x_t (the noisy picture at step t)", "meaning": "What the network will see during training. It is made in one line from x₀ and one noise draw." },
    { "id": "sig", "name": "√ᾱ_t (picture weight)", "meaning": "How strongly the clean picture is still present. It is 1 at the start and falls towards 0." },
    { "id": "x0", "name": "x₀ (the clean picture)", "meaning": "A training picture, with pixels scaled to lie between −1 and 1." },
    { "id": "noi", "name": "√(1 − ᾱ_t) (noise weight)", "meaning": "How strongly fresh noise is present. It is 0 at the start and rises towards 1. The squares of the two weights add to exactly 1." },
    { "id": "eps", "name": "ε (one noise draw)", "meaning": "A single draw from N(0, 1) per pixel. This one draw replaces all t little noises of the chain, and it is the quantity the network will learn to guess." }
  ],
  "steps": [
    { "title": "Look up the schedule at t = 300", "expr": "ᾱ_300 = 0.3964", "note": "From the schedule figure above." },
    { "title": "The two weights", "expr": "√0.3964 = 0.6296 (picture),  √(1 − 0.3964) = √0.6036 = 0.7769 (noise)", "note": "The squares are 0.3964 + 0.6036 = 1." },
    { "title": "One pixel", "expr": "x₀ = 0.5, ε = −1  →  x_300 = 0.6296 × 0.5 + 0.7769 × (−1) = 0.3148 − 0.7769 = −0.4621", "note": "The noise has flipped the pixel’s sign." },
    { "title": "Three pixels at once", "expr": "x₀ = (0.8, −0.5, 0.2), ε = (0.5, −1.2, 0.3)  →  x_300 = (0.8921, −1.2471, 0.3590)", "note": "Picture parts 0.5037, −0.3148, 0.1259; noise parts 0.3885, −0.9323, 0.2331. Each pixel is done on its own." },
    { "title": "At t = 1,000", "expr": "x_1000 = 0.0064 · x₀ + 1.0000 · ε", "note": "Essentially just ε: the picture has gone." }
  ]
}
```

Notice that at t = 300 the noise weight (0.78) is bigger than the picture weight (0.63), yet the picture is not gone: ==noise is random and the picture is not, so a clever network can still find the picture under it==. The three-pixel row shows another thing: after noising, a pixel (−1.2471) can lie outside the original range of −1 to 1.

### Why the shortcut is true

You do not need a proof, but you should see why it works. The idea is that ==a sum of independent Gaussians is again a Gaussian, and their variances add up==. [[Variance|The average squared distance from the mean; the square of the standard deviation. For independent random numbers, the variance of a sum is the sum of the variances.]] is a measure of spread. After one step, the noise in x_t has variance β_t. After two steps there is the old noise (shrunk by α) plus the new noise, and the variances add. Carry on, and the total noise variance after t steps is always 1 − ᾱ_t. You can check the bookkeeping for one step with real numbers.

```viz
{
  "type": "equation",
  "title": "Checking the variance bookkeeping at t = 300",
  "intro": "The noise variance after step t should be 1 − ᾱ_t. If the chain is built right, the old noise (shrunk) plus the new noise must add up to exactly that.",
  "formula": [ { "t": "α_t", "id": "al" }, " · (1 − ", { "t": "ᾱ_(t−1)", "id": "abp" }, ") + ", { "t": "β_t", "id": "be" }, " = 1 − ", { "t": "ᾱ_t", "id": "ab" } ],
  "parts": [
    { "id": "al", "name": "α_t", "meaning": "The keep fraction of step t. The old noise is multiplied by √α_t, so its variance is multiplied by α_t." },
    { "id": "abp", "name": "ᾱ_(t−1)", "meaning": "The product up to the previous step. Then 1 − ᾱ_(t−1) is the variance of the noise already in x_(t−1)." },
    { "id": "be", "name": "β_t = 1 − α_t", "meaning": "The variance of the new noise added at step t (its weight is √β_t, and weights are squared to get variance)." },
    { "id": "ab", "name": "ᾱ_t", "meaning": "The product up to step t. The formula claims the total noise variance is 1 − ᾱ_t." }
  ],
  "steps": [
    { "title": "Numbers at t = 300", "expr": "α_300 = 0.993944, ᾱ_299 = 0.398835, β_300 = 0.006056", "note": "β_300 = 0.0001 + 0.0199 × 299/999." },
    { "title": "Old noise, shrunk", "expr": "0.993944 × (1 − 0.398835) = 0.993944 × 0.601165 = 0.597524", "note": "The noise that was already there loses a little." },
    { "title": "Add the new noise", "expr": "0.597524 + 0.006056 = 0.603580", "note": "The variances add because the new draw is independent of the old noise." },
    { "title": "Compare with the formula", "expr": "1 − ᾱ_300 = 1 − 0.396420 = 0.603580", "note": "They agree to six decimals. The chain and the shortcut give the same noise level." }
  ]
}
```

In Lab Part 1 you test the shortcut a different way: you noise one pixel 10,000 times with 300 single steps and 10,000 times with the shortcut, and the means and variances agree.

### Seeing it: the picture dissolves

Slide t in the figure below. At t = 0 you see a clean house; as t grows it dissolves into noise, exactly according to the formula above.

```viz
{
  "type": "diffusion",
  "title": "A house dissolving into noise",
  "image": ["000000000000","000009900000","000099990000","000999999000","009999999900","099999999990","000999999000","000990099000","000990099000","000990099000","000999999000","000000000000"],
  "t": 300,
  "note": "A small 12 × 12 shape drawn for teaching. The noise in the figure is random, so a new pattern appears each time."
}
```

Move t from 0 to 100: very little changes. Around 300 the shape is hard to see but not gone. By 1,000 you see only static. The schedule is gentle at the start on purpose, so each reverse step has an easy job.

### Signal-to-noise ratio

To compare schedules in one number, people use the [[signal-to-noise ratio|SNR: how much stronger the picture’s part is than the noise’s part. Here SNR_t = ᾱ_t divided by (1 − ᾱ_t), the squared picture weight over the squared noise weight.]]. It is the squared picture weight divided by the squared noise weight, and it works like the “how loud is the music compared with the hiss” knob on a radio.

```viz
{
  "type": "equation",
  "title": "Signal-to-noise ratio",
  "intro": "Select the parts, then follow the numbers for three steps of the linear schedule.",
  "formula": [ { "t": "SNR_t", "id": "snr" }, " = ", { "t": "ᾱ_t", "id": "num" }, " / (1 − ", { "t": "ᾱ_t", "id": "den" }, ")" ],
  "parts": [
    { "id": "snr", "name": "SNR_t (signal-to-noise ratio)", "meaning": "Above 1 the picture part is stronger than the noise part. Below 1 the noise is stronger. It equals (picture weight / noise weight) squared." },
    { "id": "num", "name": "ᾱ_t (squared picture weight)", "meaning": "The square of √ᾱ_t, the weight of the clean picture in the closed form." },
    { "id": "den", "name": "1 − ᾱ_t (squared noise weight)", "meaning": "The square of √(1 − ᾱ_t), the weight of the noise in the closed form." }
  ],
  "steps": [
    { "title": "At t = 100", "expr": "0.8970 / (1 − 0.8970) = 0.8970 / 0.1030 = 8.71", "note": "The picture part is 8.7 times as strong as the noise part." },
    { "title": "At t = 300", "expr": "0.3964 / 0.6036 = 0.657", "note": "The noise part is now stronger: SNR has dropped below 1." },
    { "title": "At t = 500", "expr": "0.0786 / 0.9214 = 0.0853", "note": "The picture part is about one-twelfth of the noise part." },
    { "title": "Where SNR crosses 1", "expr": "ᾱ_t = 0.5, which for the linear schedule first happens at t = 260", "note": "Of the 1,000 steps, 259 have SNR above 1 and 741 have SNR below 1." }
  ]
}
```

==The SNR is the fair way to compare schedules==, because it measures what the network has to work with. A good rule: ==the network learns best when it sees a spread of noise levels, not mostly hopeless ones.== That brings up a weakness of the linear schedule.

### The cosine schedule

The linear schedule destroys the picture quickly. Of its 1,000 steps, ==175 have ᾱ_t below 0.001==: the picture is already gone, and those steps teach the network little. Nichol and Dhariwal (2021) proposed the [[cosine schedule|A noise schedule defined through ᾱ_t = f(t)/f(0) with f(t) = cos² of a shifted, scaled t/T. It destroys the picture more slowly at the start and the end than the linear schedule.]], which destroys information more slowly. Open the cosine tab in the figure and compare.

```viz
{
  "type": "schedule",
  "title": "The cosine schedule",
  "intro": "Switch to the cosine tab and drag the slider. Compare ᾱ at t = 300 and 500 with the linear schedule shown for reference.",
  "T": 1000,
  "kind": "cosine",
  "t": 500,
  "note": "Cosine schedule of Nichol and Dhariwal (2021) with offset s = 0.008 and β clipped at 0.999; values are computed in the browser.",
  "checks": [ { "kind": "cosine", "t": 300, "abar": 0.787 }, { "kind": "cosine", "t": 500, "abar": 0.494 }, { "kind": "linear", "t": 500, "abar": 0.0786 } ]
}
```

With the cosine schedule ᾱ_300 = 0.787 (the linear one: 0.396) and ᾱ_500 = 0.494 (the linear one: 0.0786). The SNR at t = 300 is 3.69 instead of 0.657. Only 21 steps have ᾱ below 0.001, and SNR stays above 1 until t = 496. Is that “better”? It depends on the data and on T; ==a schedule is a design choice, not a law of nature==. You compare them side by side in Lab Part 2.

### Reading the schedule as a table

The heat map below shows the same two numbers for several steps and both schedules: the picture weight and the noise weight. Switch tabs.

```viz
{
  "type": "heatmap",
  "title": "Picture weight and noise weight at several steps",
  "intro": "Rows are steps t. The first column is √ᾱ_t (how much picture), the second is √(1 − ᾱ_t) (how much noise). Switch between the two schedules.",
  "rows": ["t = 1","t = 50","t = 100","t = 200","t = 300","t = 500","t = 800","t = 1000"],
  "cols": ["picture √ᾱ","noise √(1−ᾱ)"],
  "corner": "step ↓ weight →",
  "showValues": true,
  "max": 1,
  "tabs": [
    { "name": "Linear", "caption": "Linear schedule: the picture weight is 0.63 at t = 300 and almost zero by t = 800.", "W": [[0.9999,0.01],[0.9854,0.1702],[0.9471,0.3209],[0.8118,0.5839],[0.6296,0.7769],[0.2803,0.9599],[0.0391,0.9992],[0.0064,1.0]] },
    { "name": "Cosine", "caption": "Cosine schedule: the picture weight is still 0.89 at t = 300 and 0.70 at t = 500.", "W": [[1.0,0.0064],[0.996,0.0894],[0.9859,0.1671],[0.948,0.3183],[0.8871,0.4616],[0.7027,0.7114],[0.3067,0.9518],[0.0,1.0]] }
  ],
  "note": "Values computed from the two schedules (rounded to four places). The cosine value at t = 1000 is zero to many decimals."
}
```

Notice how the linear picture weight falls to 0.28 by t = 500, while the cosine one is still 0.70. ==Same final destination (pure noise), different pace.== The reverse process has to follow the same pace backwards, so the pace matters.

### Sorting forward and reverse

Before you move on, check your picture of the two directions. Put each statement under the process it describes; some apply to both.

```viz
{
  "type": "sorter",
  "title": "Forward process, reverse process, or both?",
  "intro": "Sort each statement. Remember: the forward process is the fixed fog; the reverse process is what the network learns.",
  "bins": [ { "id": "forward", "label": "Forward process" }, { "id": "reverse", "label": "Reverse process" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "A fixed recipe that needs no learning", "answer": "forward", "why": "Adding noise follows the schedule β_t exactly. No parameters are trained for it." },
    { "name": "Needs a trained neural network", "answer": "reverse", "why": "Removing noise is the hard direction. A network has to guess what was added." },
    { "name": "Ends at (almost) pure Gaussian noise", "answer": "forward", "why": "After T = 1,000 steps ᾱ is about 0.00004, so x_T is practically N(0, I)." },
    { "name": "Starts from a fresh draw of N(0, I)", "answer": "reverse", "why": "Sampling a new picture starts from random static and works backwards." },
    { "name": "Can jump from x₀ to x_t in one line", "answer": "forward", "why": "The closed form x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε needs no loop." },
    { "name": "Is repeated about T times in the DDPM sampler", "answer": "reverse", "why": "The original sampler makes one denoising step per t, from T down to 1." },
    { "name": "Uses the schedule numbers β_t and ᾱ_t", "answer": "both", "why": "Both the training recipe and the reverse step use the same schedule, so it must be fixed in advance." },
    { "name": "Keeps the picture’s shape (one number per pixel)", "answer": "both", "why": "Every x_t, noisy or clean, has the same shape as the picture. The network’s input and output have that shape too." }
  ]
}
```

### The whole chain on one page

Here is the forward process as a short story you can follow with real numbers: eight pixels, a “pulse” picture of values −1 and 1, and one fixed noise draw ε reused for every t, which is how the closed form works.

```viz
{
  "type": "pipeline",
  "title": "Eight pixels disappearing",
  "intro": "Step through the same eight-pixel signal at five noise levels. The numbers come from the closed form with the linear schedule and one noise draw.",
  "nodes": [
    { "id": "t0", "label": "t = 0", "sub": "clean" },
    { "id": "t100", "label": "t = 100", "sub": "light fog" },
    { "id": "t300", "label": "t = 300", "sub": "heavy fog" },
    { "id": "t500", "label": "t = 500", "sub": "mostly noise" },
    { "id": "t1000", "label": "t = 1000", "sub": "static" }
  ],
  "steps": [
    { "title": "The clean signal", "body": "A bump: two pixels at −1, then three at +1, then three at −1. You can read the bump easily.", "active": ["t0"], "example": "x₀ = (−1, −1, 1, 1, 1, −1, −1, −1)" },
    { "title": "Light fog", "body": "At t = 100 the picture weight is 0.947 and the noise weight 0.321. The bump is slightly blurred, but the shape is clear.", "active": ["t0", "t100"], "example": "x₁₀₀ = (−0.06, −0.97, 0.64, 0.97, 0.90, −0.65, −1.18, −1.34)" },
    { "title": "Heavy fog", "body": "At t = 300 the noise weight 0.777 is larger than the picture weight 0.630. The first pixel got a big noise draw (2.76) and has even changed sign. A person can still half-see the bump.", "active": ["t100", "t300"], "example": "x₃₀₀ = (1.51, −0.68, −0.11, 0.69, 0.52, 0.09, −1.20, −1.58)" },
    { "title": "Mostly noise", "body": "At t = 500 the picture weight is only 0.280. The values are now close to the noise draw itself.", "active": ["t300", "t500"], "example": "x₅₀₀ = (2.37, −0.35, −0.63, 0.36, 0.15, 0.60, −0.98, −1.45)" },
    { "title": "Static", "body": "At t = 1,000 the signal is multiplied by 0.0064. The row is the noise draw, to two decimals.", "active": ["t500", "t1000"], "example": "x₁₀₀₀ = (2.75, −0.08, −0.94, 0.09, −0.13, 0.91, −0.74, −1.23)" }
  ]
}
```

Check yourself on the arithmetic: the first pixel at t = 100 is 0.947 × (−1) + 0.321 × 2.76 = −0.06. ==The same noise draw is used at every t==, so the rows differ only in the two weights. In a real training run a new ε is drawn each time.

### Imagine first, then look

Before the lab’s Part 1 you are asked to predict how much of the picture is left after 300 of 1,000 noising steps. You already know the answer: the picture is multiplied by 0.63, which is about 63 percent of its original strength, and noise now has the bigger weight. ==The forward process is the easy half of diffusion.== The rest of the lesson is about the other half.

## Training: predict the noise {rail=Training}

{lede} The forward process is a fixed recipe, so what is there to learn? The answer is the reverse: given a noisy picture x_t and the step t, the network must work out what the picture looked like before the fog. ==The trick that makes this easy to train is to ask the network for the noise that was added, not for the clean picture.== Because we made that noise ourselves, ==we know the right answer exactly==, and training becomes ordinary [[supervised learning|Training a model on examples that come with the right answer, so the loss compares the model’s output with the known answer.]] with a squared error.

### Why not ask for the picture directly?

Suppose you show a network the blurred photo x_300 and ask for x₀. Many different clean pictures could have produced the same blurred photo, so the question is hard and the network learns to give a vague average. Suppose instead you ask for x_(t−1), the picture one step earlier. That step is tiny, so the answer is almost the input, and the useful information is the small difference. Both questions hide the same fact: ==all the information you need is in the noise==.

Look again at the closed form: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε. If you know x_t, t and ε, you can solve for x₀. So ==finding the noise ε and finding the clean picture x₀ are the same job==, and the noise is the easier thing to ask for: it has a fixed, simple shape (independent Gaussian numbers), the same at every noise level. That is the idea of the DDPM paper.

Think of a teacher who photocopies a page, scribbles random marks on it, and gives you the scribbled copy. The scribbles are the noise. Your job is to circle every scribble. The teacher can mark your answer perfectly because the teacher drew the scribbles. A [[denoiser|The neural network ε_θ that takes a noisy picture x_t and the step number t and outputs its guess of the noise that was added. It is trained, and it is the only learned part of a diffusion model.]] does this for every amount of scribbling, from a faint dusting to a page covered in ink.

### The loss: a mean squared error

The network is called ε_θ (epsilon with the learnable weights θ). It takes two inputs, the noisy picture x_t and the step t, and returns a guess of the noise with the same shape as the picture. The training loss compares the guess with the real noise.

```viz
{
  "type": "equation",
  "title": "The training loss: guess the noise",
  "intro": "Select each symbol, then follow the worked example with a two-pixel picture.",
  "formula": [ { "t": "L", "id": "loss" }, " = E ‖ ", { "t": "ε", "id": "eps" }, " − ", { "t": "ε_θ", "id": "net" }, "(", { "t": "x_t", "id": "xt" }, ", ", { "t": "t", "id": "t" }, ") ‖²" ],
  "parts": [
    { "id": "loss", "name": "L (the loss)", "meaning": "The average squared distance between the real noise and the network’s guess. Training lowers L by gradient descent, as in Week 3." },
    { "id": "eps", "name": "ε (the real noise)", "meaning": "The noise we drew when we built x_t from x₀. It is the training target and we know it exactly." },
    { "id": "net", "name": "ε_θ (the denoiser)", "meaning": "The neural network with learnable weights θ (theta). Its output has the same shape as the picture." },
    { "id": "xt", "name": "x_t (the noisy input)", "meaning": "The picture after the closed-form noising, x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε." },
    { "id": "t", "name": "t (the step)", "meaning": "How noisy the input is. The network needs it because the same-looking input means different things at different noise levels." }
  ],
  "steps": [
    { "title": "The real noise and the guess", "expr": "ε = (0.5, −1.2),  ε_θ(x_t, t) = (0.3, −0.9)", "note": "Two pixels. The network is close but not exact." },
    { "title": "Differences", "expr": "ε − ε_θ = (0.5 − 0.3, −1.2 − (−0.9)) = (0.2, −0.3)", "note": "The first pixel is off by 0.2, the second by 0.3." },
    { "title": "Square and add", "expr": "‖ε − ε_θ‖² = 0.2² + (−0.3)² = 0.04 + 0.09 = 0.13", "note": "This is the sum of squares, as in the formula." },
    { "title": "Mean over pixels", "expr": "0.13 / 2 = 0.065", "note": "In code the loss is usually averaged over pixels (F.mse_loss), which divides by the number of pixels. The two differ only by a constant." },
    { "title": "What E means", "expr": "average over x₀, t and ε", "note": "In practice that average is a batch: many pictures, each with its own random t and its own ε." }
  ]
}
```

==Notice that nothing in this loss mentions “realistic” or “fake”.== It only asks: how close is the guess to a number we already know? In Lab Part 4 you compute this loss yourself on a batch of two-dimensional points, and in Lab Part 6 on 16 × 16 pictures.

### One training iteration

Training a diffusion model is a very short recipe. Step through it.

```viz
{
  "type": "pipeline",
  "title": "One training iteration of a diffusion model",
  "intro": "Step through what happens for one batch. Each picture gets its own random step t and its own noise ε.",
  "nodes": [
    { "id": "x0", "label": "Clean picture x₀", "sub": "from the data" },
    { "id": "t", "label": "Random step t", "sub": "1 … T" },
    { "id": "eps", "label": "Noise ε", "sub": "N(0, I)" },
    { "id": "xt", "label": "Noisy picture x_t", "sub": "closed form" },
    { "id": "net", "label": "Denoiser", "sub": "ε_θ(x_t, t)" },
    { "id": "loss", "label": "MSE loss", "sub": "‖ε − ε_θ‖²" }
  ],
  "steps": [
    { "title": "Take a picture, pick a random step", "body": "Draw a clean picture x₀ from the training set and a random step t between 1 and T. Different pictures in the batch get different values of t, so the network sees every noise level.", "active": ["x0", "t"], "example": "x₀ = a 16 × 16 shape, t = 300." },
    { "title": "Draw the noise", "body": "Sample ε from N(0, I): one independent number per pixel. Keep it, because it is the answer the network must find.", "active": ["eps"], "example": "ε has 256 numbers for a 16 × 16 grey picture." },
    { "title": "Build the noisy picture", "body": "Use the closed form to make x_t in one line. No loop over the 300 earlier steps is needed.", "active": ["x0", "t", "eps", "xt"], "example": "x_300 = 0.6296 · x₀ + 0.7769 · ε." },
    { "title": "Predict the noise", "body": "Feed x_t and t into the denoiser. It outputs ε_θ(x_t, t), the same shape as the input.", "active": ["xt", "t", "net"], "example": "Input and output are both 16 × 16 grids." },
    { "title": "Compare and update", "body": "Compute the mean squared error between ε and the guess, backpropagate (Week 3), and take a gradient step. Then repeat with a new batch.", "active": ["eps", "net", "loss"], "example": "loss = mean((ε − ε_θ(x_t, t))²)." }
  ]
}
```

Notice there is no second network, no game and no “real or fake” label. The only thing the algorithm needs from the data is the pictures themselves.

### Why this works so well

Four properties make this recipe forgiving, compared with the GAN game of Week 12.

1. ==The target is known exactly.== In a GAN the generator’s target is a moving judgement from another network. Here the target is a number we generated ourselves.
2. **There is no adversary.** One network, one loss. The loss goes down as the network gets better, so it is a trustworthy signal, unlike a GAN’s two losses that can both look fine while the pictures are bad.
3. **Each noise level is a separate small task.** At t = 10 the network cleans a faint dusting; at t = 800 it finds blobs under heavy snow. One network handles all of them because it is told t.
4. **Coverage.** Training is not a game, so there is no pressure to fall into [[mode collapse|A GAN failure from Week 12 in which the generator learns to make only a few outputs that fool the discriminator.]]. The network is trained on every training picture at many noise levels, so it is pushed to cover the whole data set. (Coverage is a tendency, not a guarantee.) In Lab Part 4 you check this on the eight-blob data set of Week 12.

A caution: ==the loss does not go to zero, even for a perfect network==. The noise ε is random, and at moderate noise levels the network cannot tell exactly which part of x_t is noise and which is picture, so a floor remains. You will measure this floor in Lab Part 4.

### Random t for every example, and the batch average

Why a random step for each example, instead of going through 1 to T in order? Because ==we want a network that is good at every noise level at once==, and a random t in each example gives a balanced mixture in every batch, like a teacher who mixes easy and hard questions in every worksheet.

```viz
{
  "type": "equation",
  "title": "The loss over a batch of three examples",
  "intro": "A tiny batch of three (picture, step) pairs. Each has its own t and its own two-pixel noise. The loss per example is the mean squared error; the batch loss is their average.",
  "formula": [ { "t": "L_batch", "id": "lb" }, " = (1 / ", { "t": "B", "id": "b" }, ") · Σ_i mean( ", { "t": "ε_i", "id": "ei" }, " − ", { "t": "ε̂_i", "id": "ehi" }, " )²" ],
  "parts": [
    { "id": "lb", "name": "L_batch (the batch loss)", "meaning": "The number the optimiser lowers at each iteration: the average of the per-example losses." },
    { "id": "b", "name": "B (the batch size)", "meaning": "How many (picture, step, noise) triples are in the batch. The lab uses 64 for pictures and 256 for two-dimensional points." },
    { "id": "ei", "name": "ε_i (the real noise of example i)", "meaning": "The noise that was added to example i, drawn with its own random step t_i." },
    { "id": "ehi", "name": "ε̂_i (the guess for example i)", "meaning": "The denoiser’s output ε_θ(x_(t_i), t_i) for example i." }
  ],
  "steps": [
    { "title": "Example 1, t = 50 (little noise)", "expr": "ε = (0.4, −0.7), ε̂ = (0.35, −0.6)  →  (0.05² + 0.1²)/2 = 0.00625", "note": "Illustrative numbers authored for teaching." },
    { "title": "Example 2, t = 300", "expr": "ε = (0.5, −1.2), ε̂ = (0.3, −0.9)  →  (0.2² + 0.3²)/2 = 0.065", "note": "This is the example of the previous figure." },
    { "title": "Example 3, t = 800 (heavy noise)", "expr": "ε = (−1.1, 0.2), ε̂ = (−0.2, 0.5)  →  (0.9² + 0.3²)/2 = 0.45", "note": "A poor guess: errors like this are common early in training." },
    { "title": "Average", "expr": "L_batch = (0.00625 + 0.065 + 0.45) / 3 = 0.17375", "note": "The big example dominates: the optimiser pays most attention to what it gets most wrong." }
  ]
}
```

==A random t is how one network ends up trained on all 1,000 noise levels.== A [[batch|A small group of training examples processed together in one step. The loss and the gradient are averaged over the group.]] of 64 pictures gives 64 different noise levels at once.

### Other things the network could predict

==Predicting ε is the usual choice, not the only one.== A network can instead predict the clean picture x₀ directly, or a mix of the two called v (the “v-prediction”). They carry the same information, because you can convert one into another with the closed form. People choose among them for training stability and for how the result behaves at the noisiest steps. This lesson uses ε-prediction throughout.

### What a perfect denoiser would compute

Here is a worked example that shows what the network is trying to be. Suppose the data are very simple: every training picture is a single pixel with the exact value x₀ = 2. Then a perfect network could compute the noise exactly from x_t, because the only unknown is ε, and x_t = √ᾱ_t · 2 + √(1 − ᾱ_t) · ε.

```viz
{
  "type": "equation",
  "title": "A perfect denoiser when every picture is x₀ = 2",
  "intro": "If you know the data, you can turn the closed form around. Select each symbol and follow t = 300 with a noisy pixel x_t = 1.4.",
  "formula": [ { "t": "ε̂", "id": "eh" }, " = (", { "t": "x_t", "id": "xt" }, " − ", { "t": "√ᾱ_t", "id": "sa" }, " · ", { "t": "x₀", "id": "x0" }, ") / ", { "t": "√(1 − ᾱ_t)", "id": "sn" } ],
  "parts": [
    { "id": "eh", "name": "ε̂ (the best guess of the noise)", "meaning": "What a perfect denoiser outputs when it knows that x₀ is always the same." },
    { "id": "xt", "name": "x_t (the noisy pixel)", "meaning": "The only thing the network sees, together with t." },
    { "id": "sa", "name": "√ᾱ_t (picture weight)", "meaning": "0.6296 at t = 300." },
    { "id": "x0", "name": "x₀ (the clean value)", "meaning": "Known here to be 2. In a real data set the network does not know x₀ and must learn what typical pictures look like." },
    { "id": "sn", "name": "√(1 − ᾱ_t) (noise weight)", "meaning": "0.7769 at t = 300." }
  ],
  "steps": [
    { "title": "Expected picture part", "expr": "√ᾱ_300 × x₀ = 0.6296 × 2 = 1.2592", "note": "This is where x_t would sit with zero noise." },
    { "title": "What is left over", "expr": "x_t − 1.2592 = 1.4 − 1.2592 = 0.1408", "note": "The pixel is a little above the expected value, so the noise was positive." },
    { "title": "Divide by the noise weight", "expr": "ε̂ = 0.1408 / 0.7769 = 0.1812", "note": "The estimated noise draw was 0.18." },
    { "title": "Check by rebuilding", "expr": "0.6296 × 2 + 0.7769 × 0.1812 = 1.2592 + 0.1408 = 1.4", "note": "It reproduces x_t, so the guess is consistent." }
  ]
}
```

A real data set is not a single point, so the network cannot do this exactly: it has to learn the shape of the data. But the formula also reveals something about the *size* of the job at each noise level. If the noise guess is off by 0.1, the implied clean picture is off by 0.1 × √(1 − ᾱ_t)/√ᾱ_t. At t = 50 that factor is 0.17 (an error of 0.017), at t = 300 it is 1.23 (0.123), at t = 800 it is 25.5 (an error of 2.55), and at t = 1,000 it is 157. This is why a sampler does not try to jump from heavy noise straight to a picture: it takes many small steps and lets the network correct itself as the fog thins.

### Watch a perfect denoiser work

Now play with a one-dimensional world where the data are two bumps, one at −2 and one at 3. This figure uses the *exact* noise predictor for that data (computed from a formula, no neural network), so it shows what a perfectly trained denoiser would do. Press Play and watch 600 dots that start as random noise (a single bell curve) split into two groups on top of the true density.

```viz
{
  "type": "sampler1d",
  "title": "What a perfect denoiser would do",
  "intro": "Press Play (or drag the slider). The dots start as pure noise at the right-hand end of the schedule and are denoised step by step towards the two bumps.",
  "means": [-2, 3],
  "weights": [0.5, 0.5],
  "sd0": 0.4,
  "T": 200,
  "n": 600,
  "seed": 5,
  "stochastic": true,
  "note": "One-dimensional data, T = 200 steps, β rising linearly from 0.001 to 0.06 (so ᾱ_T is about 0.002). The noise predictor is the exact formula, not a trained network.",
  "checks": [ { "meanMin": 0.2, "meanMax": 0.8, "leftMin": 0.4, "leftMax": 0.6 } ]
}
```

Notice three things. First, the dots begin as one blob, yet end up in two separate groups: nothing told them which bump to join. Second, about half end up in the left bump (here 50 percent) and the final mean is close to the data mean 0.5. Third, each group has a spread of about 0.42, close to the true 0.4: the process covers the width of each bump instead of collapsing onto its centre. In Lab Part 3 you write this loop yourself with the exact predictor and get the same numbers.

### Training diffusion versus training a GAN

Put the two training stories next to each other. The pictures in the tabs are described in words, not drawn by a model.

```viz
{
  "type": "flow",
  "title": "Training a GAN versus training a diffusion model",
  "modes": [
    {
      "name": "GAN (Week 12)",
      "tagline": "Two networks play a game; each loss depends on the other network.",
      "inputLabel": "Input: noise z and real pictures",
      "modelLabel": "Generator G against discriminator D",
      "outputLabel": "Output: a generator that makes a picture in one pass",
      "examples": [
        { "label": "What is the target?", "input": "G’s target is “fool D”, and D keeps changing", "output": "A moving target that can oscillate or collapse" },
        { "label": "Can you trust the loss?", "input": "Both losses can look fine while the pictures are poor", "output": "You must look at samples and use a judge" },
        { "label": "What can go wrong?", "input": "Mode collapse, vanishing gradients, instability", "output": "Many tricks (Week 12) were invented to cope" }
      ],
      "note": "Illustrative summary of Week 12: no real training run is shown."
    },
    {
      "name": "Diffusion (this week)",
      "tagline": "One network learns to guess the noise it is shown; the target is known exactly.",
      "inputLabel": "Input: a noisy picture x_t and the step t",
      "modelLabel": "One denoiser ε_θ",
      "outputLabel": "Output: a guess of the noise, same shape as the input",
      "examples": [
        { "label": "What is the target?", "input": "The noise ε we drew ourselves", "output": "A fixed, known number for every pixel" },
        { "label": "Can you trust the loss?", "input": "A plain squared error that goes down as the network improves", "output": "Still look at samples, but the loss is a steady signal" },
        { "label": "What can go wrong?", "input": "Too little training, too few sampling steps, a wrong schedule", "output": "Blurry or noisy pictures, not a collapse onto one picture" }
      ],
      "note": "These are general tendencies, not guarantees. Results depend on the model, the data and the settings."
    }
  ]
}
```

Switch tabs and compare the rows. ==The price of the smooth training is paid later, at sampling time==, when the denoiser has to be called many times. That is the next section.

### Sorting training and sampling

Check what you know. Decide whether each statement is about training, about sampling (making a new picture), or true of both.

```viz
{
  "type": "sorter",
  "title": "Training, sampling, or both?",
  "intro": "Sort each statement. Training teaches the denoiser; sampling uses it to make a new picture.",
  "bins": [ { "id": "training", "label": "Training" }, { "id": "sampling", "label": "Sampling" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "Picks a random step t for each picture", "answer": "training", "why": "Each training example gets its own random t so the network sees every noise level in every batch." },
    { "name": "Starts from a draw of pure noise x_T", "answer": "sampling", "why": "A new picture is generated by starting from static; in training we start from real pictures." },
    { "name": "Needs the real noise ε as the answer to compare with", "answer": "training", "why": "The squared error compares the guess with the noise we added. At sampling time nobody knows the true noise." },
    { "name": "Calls the denoiser once per step, many times in a row", "answer": "sampling", "why": "The sampler loops from t = T down to 1, calling the network at each step." },
    { "name": "Feeds the network a noisy picture and the step t", "answer": "both", "why": "The denoiser’s input is always (x_t, t); only where x_t comes from differs." },
    { "name": "Updates the network’s weights by gradient descent", "answer": "training", "why": "The weights are frozen when you sample; only the picture changes." },
    { "name": "Uses the closed form to make x_t in one line", "answer": "training", "why": "Training builds x_t from x₀ and ε with the shortcut; the sampler gets x_t from the previous step." },
    { "name": "Uses the noise schedule β_t and ᾱ_t", "answer": "both", "why": "Training needs ᾱ_t for the closed form and sampling needs β_t and ᾱ_t for the reverse step." }
  ]
}
```

==Training is the cheap, stable half of diffusion. Sampling is where the time goes.==

## Sampling: from noise to image {rail=Sampling}

{lede} Once the denoiser is trained, making a new picture is a loop. ==Start from pure noise x_T, ask the network which noise it sees, remove a little of it, and repeat down to t = 1.== This is the sculptor’s method: many small taps instead of one big cut, and each tap is easy because the network is only asked to undo one thin layer of fog.

### The reverse step

The standard rule is called [[DDPM|Denoising Diffusion Probabilistic Models, the 2020 method of Ho, Jain and Abbeel. Its sampler removes the predicted noise one step at a time and adds a little fresh noise at each step.]] sampling. At step t the sampler has x_t and the network’s guess ε̂ = ε_θ(x_t, t). It computes the previous picture x_(t−1) in three small moves: subtract the part of the picture that the noise accounts for, scale the result back up to undo the step’s shrinking, and add a little *fresh* noise.

```viz
{
  "type": "equation",
  "title": "One reverse (denoising) step",
  "intro": "Select each symbol. The worked example is one pixel at t = 300 of the linear schedule.",
  "formula": [ { "t": "x_(t−1)", "id": "xp" }, " = (1 / ", { "t": "√α_t", "id": "sa" }, ") · ( ", { "t": "x_t", "id": "xt" }, " − ", { "t": "β_t / √(1 − ᾱ_t)", "id": "coef" }, " · ", { "t": "ε_θ(x_t, t)", "id": "eh" }, " ) + ", { "t": "σ_t", "id": "sig" }, " · ", { "t": "z", "id": "z" } ],
  "parts": [
    { "id": "xp", "name": "x_(t−1) (the less noisy picture)", "meaning": "The result of this step: one notch cleaner than x_t." },
    { "id": "sa", "name": "√α_t (undo the shrink)", "meaning": "The forward step multiplied the picture by √α_t, so the reverse step divides by it. Because α_t is just below 1, this is a very slight enlargement." },
    { "id": "xt", "name": "x_t (the current picture)", "meaning": "What the sampler holds now. At the start it is pure noise x_T." },
    { "id": "coef", "name": "β_t / √(1 − ᾱ_t) (how much noise to remove)", "meaning": "A step-specific weight that says how much of the network’s noise guess to subtract. It is small, so each step removes only a thin layer." },
    { "id": "eh", "name": "ε_θ(x_t, t) (the network’s guess)", "meaning": "The denoiser’s estimate of the noise in x_t. It has the same shape as the picture." },
    { "id": "sig", "name": "σ_t (size of the fresh noise)", "meaning": "In DDPM a common simple choice is σ_t = √β_t. It is small, and the same schedule that adds noise in training sets it." },
    { "id": "z", "name": "z (fresh noise)", "meaning": "A new draw from N(0, I) at every step, except the last step (t = 1) where z is set to 0 so the final picture is not noisy." }
  ],
  "steps": [
    { "title": "Look up the schedule at t = 300", "expr": "β_300 = 0.006056, α_300 = 0.993944, ᾱ_300 = 0.3964, √(1 − ᾱ_300) = 0.7769", "note": "β_300 = 0.0001 + 0.0199 × 299/999." },
    { "title": "How much to subtract", "expr": "β/√(1 − ᾱ) = 0.006056 / 0.7769 = 0.007795", "note": "Per unit of predicted noise the sampler removes only 0.0078." },
    { "title": "Subtract the predicted noise", "expr": "x_t = 0.9, ε̂ = 0.6  →  0.9 − 0.007795 × 0.6 = 0.9 − 0.004677 = 0.895323", "note": "One pixel with illustrative values." },
    { "title": "Undo the shrink", "expr": "0.895323 / √0.993944 = 0.895323 / 0.996967 = 0.898046", "note": "A slight enlargement." },
    { "title": "Add fresh noise", "expr": "σ = √0.006056 = 0.07782. With z = 0.3: 0.07782 × 0.3 = 0.023346, so x_(t−1) = 0.898046 + 0.023346 = 0.921393", "note": "With z = 0 (the last step) the answer would stay 0.898046." }
  ]
}
```

Notice how small every move is: the pixel went from 0.9 to 0.921, and most of that came from the fresh noise, not from the subtraction. A single reverse step does not paint a picture. ==A thousand of them, each guided by the network, do.==

### Why add fresh noise at all?

It may seem odd to remove noise and then put some back. There are two reasons, and one of them is the heart of why diffusion produces *variety*.

- The reverse step is the average of what the earlier picture could have been, plus the right amount of spread around that average. Without the spread, every starting noise would be pushed along the same smooth path towards the *average* of the picture it is near.
- ==Fresh noise lets the sampler explore.== Two sculptors starting from two different blocks of marble should not both end up carving the same average statue.

The next figure shows the effect with real numbers, so you can see it instead of taking it on trust.

```viz
{
  "type": "sampler1d",
  "title": "Stochastic and deterministic reverse steps",
  "intro": "The data are now 70 percent in the left bump (−2) and 30 percent in the right bump (3). Press Play with the fresh-noise switch on, then turn it off and play again.",
  "means": [-2, 3],
  "weights": [0.7, 0.3],
  "sd0": 0.4,
  "T": 200,
  "n": 600,
  "seed": 8,
  "stochastic": true,
  "note": "Exact noise predictor, T = 200, β from 0.001 to 0.06. Turning off the fresh noise z is a teaching experiment, not a recommended sampler: it is not DDIM.",
  "checks": [ { "stochastic": true, "meanMin": -0.8, "meanMax": -0.3, "leftMin": 0.62, "leftMax": 0.78 }, { "stochastic": false, "meanMax": -1.8, "leftMin": 0.95 } ]
}
```

With the fresh noise on, about 71 percent of the dots end in the left bump (the true share is 70 percent), and the spread inside each bump is about 0.39 to 0.43, close to the true 0.4. With the fresh noise off, *all* dots slide into the big left bump (100 percent) and the spread inside it shrinks to about 0.02. The sampler has collapsed: ==without fresh noise the process stops covering the data and rushes to the nearest and biggest peak==. You repeat both runs in Lab Part 3.

### The sampling loop

Put the reverse step in a loop and you have the whole generator.

```viz
{
  "type": "pipeline",
  "title": "The sampling loop",
  "intro": "Step through one trip from static to picture. Count the network calls.",
  "nodes": [
    { "id": "noise", "label": "Pure noise x_T", "sub": "a draw of N(0, I)" },
    { "id": "net", "label": "Denoiser", "sub": "guess ε at step t" },
    { "id": "step", "label": "Reverse step", "sub": "x_t → x_(t−1)" },
    { "id": "loop", "label": "t − 1", "sub": "repeat" },
    { "id": "img", "label": "Picture x₀", "sub": "the result" }
  ],
  "steps": [
    { "title": "Start with static", "body": "Draw x_T from N(0, I), one independent number per pixel. Different static gives a different final picture, so the starting noise works like a seed.", "active": ["noise"], "example": "A 16 × 16 grid of random numbers." },
    { "title": "Ask the network", "body": "Give the denoiser the current picture and the step number. It returns its guess of the noise hiding the picture.", "active": ["noise", "net"], "example": "The call at t = 1,000 is the first of 1,000." },
    { "title": "Take one step back", "body": "Apply the reverse-step formula: subtract a thin layer of predicted noise, scale back up and add a little fresh noise. The result is slightly cleaner.", "active": ["net", "step", "loop"], "example": "x₁₀₀₀ → x₉₉₉ → x₉₉₈ → …" },
    { "title": "Repeat until t = 1", "body": "At the last step set z = 0 so the output is not left with a layer of noise. The loop has called the network T times.", "active": ["loop", "step", "img"], "example": "1,000 steps = 1,000 network calls for one picture." }
  ]
}
```

Notice the arithmetic of the loop: one picture costs T network calls. A GAN makes the same kind of picture with one pass. ==Diffusion pays for its stable training with a slow sampler.==

```viz
{
  "type": "equation",
  "title": "The cost of sampling",
  "intro": "Total time is the number of network calls times the time per call. The 0.05 seconds per call is an assumption chosen to be easy to calculate with, not a measurement of a particular GPU.",
  "formula": [ { "t": "time", "id": "time" }, " = ", { "t": "N", "id": "n" }, " × ", { "t": "c", "id": "c" } ],
  "parts": [
    { "id": "time", "name": "time (seconds for one picture)", "meaning": "How long a user waits to see the result." },
    { "id": "n", "name": "N (network calls)", "meaning": "One for a GAN, T for DDPM (1,000 here), fewer for faster samplers. If the model uses guidance (later in this lesson) each step makes two calls." },
    { "id": "c", "name": "c (seconds per call)", "meaning": "How long one forward pass through the network takes. It depends on the model size and the hardware." }
  ],
  "steps": [
    { "title": "A GAN", "expr": "1 × 0.05 = 0.05 seconds", "note": "One pass through the generator, as in Week 12." },
    { "title": "DDPM with 1,000 steps", "expr": "1,000 × 0.05 = 50 seconds", "note": "1,000 times as slow as the GAN." },
    { "title": "A faster sampler with 50 steps", "expr": "50 × 0.05 = 2.5 seconds", "note": "1,000 / 50 = 20 times faster than DDPM." },
    { "title": "With 20 steps", "expr": "20 × 0.05 = 1 second", "note": "1,000 / 20 = 50 times faster than DDPM, and still 20 times slower than a GAN." }
  ]
}
```

### Watching a picture come out

The figure below is the forward slider again, but now with a different shape and the idealised reverse direction. In the real process the reverse step is made by the network; here the picture is cleaned by an idealised rule to show the shape of the process.

```viz
{
  "type": "diffusion",
  "title": "A tree dissolving and being restored",
  "image": ["0000000000000000","0000000990000000","0000009999000000","0000099999900000","0000009999000000","0000099999900000","0000999999990000","0009999999999000","0000099999900000","0000999999990000","0009999999999000","0000000990000000","0000000990000000","0000000990000000","0000000000000000","0000000000000000"],
  "t": 500,
  "note": "A 16 × 16 shape drawn for teaching. The reverse direction is idealised: a real model must predict the noise at every step."
}
```

Slide t from 1,000 down to 0 and watch the tree return. At t = 500 the shape is a ghost: ᾱ_500 = 0.0786, so the picture weight is only 0.28. A sampler that starts at the very end and walks back sees a tree appear first as a vague blob of darkness near the middle, then as a trunk and a crown, and only in the last hundred steps as clean edges. ==Early steps decide the large layout; late steps polish the detail.==

### Fewer steps: DDIM

A thousand calls is slow. A second family of samplers needs far fewer. [[DDIM|Denoising Diffusion Implicit Models, by Song, Meng and Ermon (2021). A sampler that reuses the same trained network but takes big strides along the noise schedule, so good pictures need only about 20 to 50 steps.]] (Song, Meng and Ermon, 2021) uses the *same* trained denoiser but a different update. The idea is to use the network’s noise guess to estimate the clean picture x̂₀ (a formula you saw in the Training section), and then jump straight to a much earlier step, building the earlier picture from the estimated clean picture and the estimated noise.

```viz
{
  "type": "equation",
  "title": "A DDIM step (deterministic, with η = 0)",
  "intro": "Two lines: first estimate the clean picture, then re-noise it to the earlier step s. The worked example jumps from t = 300 to s = 250.",
  "formula": [ { "t": "x̂₀", "id": "xh" }, " = (", { "t": "x_t", "id": "xt" }, " − √(1 − ᾱ_t) · ", { "t": "ε̂", "id": "eh" }, ") / √ᾱ_t,      ", { "t": "x_s", "id": "xs" }, " = ", { "t": "√ᾱ_s", "id": "as" }, " · x̂₀ + ", { "t": "√(1 − ᾱ_s)", "id": "bs" }, " · ε̂" ],
  "parts": [
    { "id": "xh", "name": "x̂₀ (the estimated clean picture)", "meaning": "Where the picture would end up if the network’s noise guess were exactly right. It is a guess, and a rough one when the noise is heavy." },
    { "id": "xt", "name": "x_t (the current picture)", "meaning": "The picture at the current, noisier step t." },
    { "id": "eh", "name": "ε̂ (the network’s noise guess)", "meaning": "ε_θ(x_t, t). The same guess is used twice: to estimate x̂₀ and to rebuild x_s." },
    { "id": "xs", "name": "x_s (the picture at the earlier step s)", "meaning": "The result of the jump. In DDIM, s can be much smaller than t − 1: for 50 steps over T = 1,000 each jump is about 20 steps long." },
    { "id": "as", "name": "√ᾱ_s (picture weight at s)", "meaning": "From the same schedule table as before. s is chosen from a list of a few steps spread across 1 … T." },
    { "id": "bs", "name": "√(1 − ᾱ_s) (noise weight at s)", "meaning": "The matching noise weight at the earlier step." }
  ],
  "steps": [
    { "title": "Numbers", "expr": "t = 300, s = 250: ᾱ_300 = 0.3964, ᾱ_250 = 0.5241;  x_t = 1.4, ε̂ = 0.8", "note": "Illustrative values for one pixel." },
    { "title": "Estimate the clean picture", "expr": "x̂₀ = (1.4 − 0.7769 × 0.8) / 0.6296 = (1.4 − 0.6215) / 0.6296 = 1.2364", "note": "If the noise guess is right, the clean pixel is about 1.24." },
    { "title": "Weights at s = 250", "expr": "√0.5241 = 0.7239 (picture),  √(1 − 0.5241) = 0.6899 (noise)", "note": "Less noise than at t = 300." },
    { "title": "Rebuild the earlier picture", "expr": "x_250 = 0.7239 × 1.2364 + 0.6899 × 0.8 = 0.8951 + 0.5519 = 1.4470", "note": "No fresh noise was added: with η = 0 the update is deterministic. The same x_t always leads to the same x_s." }
  ]
}
```

Why does this still give good pictures? The network was trained at every noise level, so it can give a sensible noise guess at any step on the list, and the update stays consistent with the closed form (it moves the picture to a point that the forward process could have produced). ==Quality usually falls slowly as you take fewer steps==, not suddenly, and then drops sharply when the strides are too long (a very small number such as 5 steps). Newer solvers and a trick called distillation (training a student network to do in few steps what a teacher does in many) push the number of steps lower still. DDIM also has a setting η (eta) that adds back some fresh noise: η = 0 is fully deterministic, and η = 1 behaves like DDPM. The lab uses η = 0.

| Sampler | Steps N | Calls per picture | Time at 0.05 s per call | Speed-up over DDPM |
|---|---|---|---|---|
| DDPM | 1,000 | 1,000 | 50 s | 1× |
| DDIM | 50 | 50 | 2.5 s | 20× |
| DDIM | 20 | 20 | 1 s | 50× |
| DDIM | 5 | 5 | 0.25 s | 200× (quality usually drops) |

The times assume 0.05 seconds per call, which is made up for easy arithmetic; the speed-up column is exact. In Lab Part 6 you time your own tiny model and judge the quality at 100, 50, 20, 10 and 5 steps.

```viz
{
  "type": "flow",
  "title": "Many steps against few steps",
  "modes": [
    {
      "name": "1,000 steps (DDPM)",
      "tagline": "Slow but careful: every thin layer of fog is removed with fresh noise.",
      "inputLabel": "Input: static x_T",
      "modelLabel": "Denoiser called 1,000 times",
      "outputLabel": "Output: a picture",
      "examples": [
        { "label": "Cost", "input": "1,000 network calls", "output": "About 50 seconds at an assumed 0.05 s per call" },
        { "label": "Quality", "input": "The sampler the models were trained for", "output": "Usually the reference quality" }
      ],
      "note": "Illustrative description, not a measurement of a particular model."
    },
    {
      "name": "20 to 50 steps (DDIM)",
      "tagline": "The same network, bigger strides, no fresh noise.",
      "inputLabel": "Input: the same static x_T",
      "modelLabel": "The same trained denoiser, called 20 to 50 times",
      "outputLabel": "Output: a similar picture, much faster",
      "examples": [
        { "label": "Cost", "input": "20 to 50 network calls", "output": "1 to 2.5 seconds at the assumed 0.05 s per call" },
        { "label": "Quality", "input": "Usually close to the 1,000-step result", "output": "Falls off gradually as you cut steps" },
        { "label": "Repeatability", "input": "Same static, same settings", "output": "The same picture every time (deterministic)" }
      ],
      "note": "The outputs are described in words, not drawn; you will measure the trade-off yourself in Lab Part 6."
    },
    {
      "name": "5 steps",
      "tagline": "The strides are now too long for the network to follow.",
      "inputLabel": "Input: the same static x_T",
      "modelLabel": "The same denoiser, called 5 times",
      "outputLabel": "Output: often blurry or smeared",
      "examples": [
        { "label": "Cost", "input": "5 network calls", "output": "0.25 seconds at the assumed 0.05 s per call" },
        { "label": "Quality", "input": "Each jump spans about 200 schedule steps", "output": "Typically blurry; smaller models degrade sooner" }
      ],
      "note": "A general tendency. Distilled models are trained specially to do well in a handful of steps."
    }
  ]
}
```

Switch between the tabs. The trade is speed against care, and it is a *setting*: ==the same trained network can be run with different samplers==, and you pick the number of steps at sampling time.

### Sorting the two samplers

```viz
{
  "type": "sorter",
  "title": "DDPM sampler, DDIM sampler, or both?",
  "intro": "Sort each statement. Remember that both use the same trained denoiser.",
  "bins": [ { "id": "ddpm", "label": "DDPM sampler" }, { "id": "ddim", "label": "DDIM sampler (η = 0)" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "Adds fresh noise z at every step except the last", "answer": "ddpm", "why": "The DDPM update contains the term σ_t·z. DDIM with η = 0 drops it." },
    { "name": "Gives the same picture every time from the same static", "answer": "ddim", "why": "With η = 0 the update is deterministic, so identical inputs give identical outputs." },
    { "name": "Typically uses on the order of a thousand steps", "answer": "ddpm", "why": "It takes one step per schedule step, so T calls." },
    { "name": "Can take strides of 20 or more schedule steps", "answer": "ddim", "why": "It estimates x̂₀ and jumps to any earlier step s, not only t − 1." },
    { "name": "Uses the same trained noise-predicting network", "answer": "both", "why": "The samplers differ only in the update rule. The network is trained once." },
    { "name": "Needs a noise schedule with ᾱ_t", "answer": "both", "why": "Both read ᾱ at the steps they visit." },
    { "name": "Starts from pure noise and moves towards a clean picture", "answer": "both", "why": "Both samplers begin with a draw of N(0, I) and end with a picture. They differ in how they get there." },
    { "name": "Is one way of buying speed with the same network", "answer": "ddim", "why": "Fewer steps means fewer network calls, hence 20 to 50 times faster than 1,000 steps." }
  ]
}
```

Whichever sampler you choose, one idea stays the same: ==generation is a loop of small, learned corrections starting from noise==. Next we open the box and ask what kind of network can make those corrections.

## The denoiser: a UNet with a time embedding {rail=UNet}

{lede} You now know what the denoiser must do: take a noisy picture and the step number, and return the noise that was added. ==That job puts three demands on the network==: its output must have the same shape as its input, it must keep fine detail at full resolution, and it must know how noisy the input is. A UNet with a time embedding meets all three, which is why nearly every diffusion model of the last few years has one at its core.

### What the denoiser needs

Think about what a mural restorer needs when a barangay mural has been scrubbed by a flood. First, the restorer works on the *whole wall* and must give back a repaired wall, the same size as the damaged one. Second, the restorer needs the small details: the fine line of a rooftop, the curve of a letter. Third, the restorer needs the big picture: this is a fiesta scene, so the shape in the corner must be bunting, not a bird. And fourth, the restorer needs to know *how badly* the wall was damaged: a light scrub calls for a touch-up; a washed-out wall calls for repainting whole shapes.

The denoiser has the same four needs.

| Need | Why | What provides it |
|---|---|---|
| Same shape in and out | Noise has one number per pixel, so the guess ε̂ must be a grid the size of the picture | A network that ends where it started: an encoder and a decoder |
| Fine detail | Noise at small t is a thin dusting; telling it from texture needs pixel-level information | Skip connections that carry full-resolution features |
| Global context | At large t you must decide the layout of the whole picture | An encoder that shrinks the picture so each position sees a wide area |
| Know the noise level | The same-looking input needs different treatment at t = 20 and t = 900 | A time embedding added to the blocks |

```viz
{
  "type": "pipeline",
  "title": "What goes in and what comes out",
  "intro": "Step through the denoiser’s interface. The sizes are for the 16 × 16 grey pictures used in the lab.",
  "nodes": [
    { "id": "xt", "label": "x_t", "sub": "1 × 16 × 16 picture" },
    { "id": "t", "label": "t", "sub": "one number, e.g. 300" },
    { "id": "unet", "label": "UNet", "sub": "the denoiser" },
    { "id": "eps", "label": "ε̂", "sub": "1 × 16 × 16 noise guess" }
  ],
  "steps": [
    { "title": "Two inputs", "body": "The network receives the noisy picture and, separately, the step number. Neither alone is enough: the picture does not say how noisy it is.", "active": ["xt", "t"], "example": "x_t has 256 numbers; t is the single number 300." },
    { "title": "One network", "body": "A UNet reads both. The picture goes through the convolutions; the step number is turned into a vector and added inside every block.", "active": ["xt", "t", "unet"], "example": "135,713 parameters in the small version of this lesson." },
    { "title": "One output, same shape as the picture", "body": "The network returns a guess of the noise for every pixel. Because the output has the same shape as x_t, subtracting a fraction of it makes sense pixel by pixel.", "active": ["unet", "eps"], "example": "ε̂ has 256 numbers, one per pixel." }
  ]
}
```

==Notice that the output is not a label or a single score.== It is a whole picture of numbers. That is why the classification networks of Week 4, which squeeze a picture into a few class scores, are not the right shape on their own.

### The UNet: a U-shaped network

The [[UNet|An encoder-decoder convolutional network drawn as a U. The encoder halves the resolution and doubles the channels; the decoder reverses that; skip connections copy early feature maps to the decoder.]] was introduced by Ronneberger, Fischer and Brox (2015) for a different job, biomedical image segmentation, where the network had to label every pixel of a microscope image. It is the same shape of problem: a picture in, a picture-sized answer out. It has three parts.

- The **encoder** (the downhill side of the U) repeats a block of convolutions and then halves the height and width with a [[stride-2 convolution|A convolution whose window jumps 2 pixels at a time instead of 1, so the output has half the height and width.]]. As the picture shrinks, the number of channels grows. Each position at the lower resolution sees a wider area of the picture, so the encoder gathers context.
- The [[bottleneck|The middle of the U-shaped network, where the picture is smallest and has the most channels.]] (the bottom of the U) works at the lowest resolution with the most channels. It sees the picture as a whole.
- The **decoder** (the uphill side) doubles the resolution again, using the [[transpose convolution|A learnable upsampling layer (Week 12): each input cell stamps a scaled copy of a kernel onto a larger output, and overlaps add. With kernel 2 and stride 2 it exactly doubles the height and width.]] you learned last week, and mixes in the skip connections.

A [[skip connection|A direct link that copies a feature map from an early layer to a later layer of the same size, so that fine detail lost when the picture was shrunk can be used again. In a UNet the copies are joined on to the decoder’s feature maps.]] is the key to the UNet. Without skips, everything the decoder knows about fine detail would have to squeeze through the narrow bottleneck. With them, the decoder at each level receives a copy of what the encoder saw at that same resolution. Move the controls in the figure to see the shapes and the parameter counts.

```viz
{
  "type": "unet",
  "title": "A small UNet denoiser for 16 × 16 pictures",
  "intro": "Change the base channels, the number of levels and the time-embedding size. Watch the table of shapes and parameters, and the dashed skip connections.",
  "size": 16,
  "inCh": 1,
  "base": 16,
  "levels": 3,
  "timeDim": 32,
  "note": "Conventions: one block = two 3 × 3 convolutions plus a time projection; downsampling = a 3 × 3 convolution with stride 2; upsampling = a 2 × 2 transpose convolution with stride 2; the decoder concatenates the skip, so its channels double; a final 1 × 1 convolution gives one channel. Real UNets differ in the details.",
  "checks": [ { "total": 135713 }, { "layer": "time embedding", "params": 2112 }, { "layer": "down block 1", "params": 3008 }, { "layer": "down block 2", "params": 14944 }, { "layer": "bottleneck", "params": 57536 }, { "layer": "up block 2", "params": 28768 }, { "layer": "up block 1", "params": 7472 } ]
}
```

Read the table from top to bottom. The picture shrinks 16 → 8 → 4 while the channels grow 16 → 32 → 64, and then the decoder undoes both. Notice that the total is 135,713 parameters, of which the bottleneck alone is 57,536 (42 percent): ==most of the weights sit at the lowest resolution where the channels are widest==. In Lab Part 5 you build this exact network and assert the same total.

### Counting the parameters of one block

You can check the table by hand. Each convolution with a 3 × 3 kernel from C_in to C_out channels has 9·C_in·C_out weights and C_out biases. A block has two of them and a small linear layer for the time (explained below).

```viz
{
  "type": "equation",
  "title": "Parameters in one UNet block",
  "intro": "Select each symbol. The worked example counts down block 1 (1 → 16 channels, time size 32) and then up block 1 (32 → 16 channels).",
  "formula": [ { "t": "P_block", "id": "p" }, " = ( 9 · ", { "t": "C_in", "id": "cin" }, " · ", { "t": "C_out", "id": "cout" }, " + ", { "t": "C_out", "id": "cout2" }, " ) + ( 9 · C_out² + C_out ) + ( ", { "t": "D", "id": "d" }, " · C_out + C_out )" ],
  "parts": [
    { "id": "p", "name": "P_block (parameters of one block)", "meaning": "All the weights and biases that the block learns." },
    { "id": "cin", "name": "C_in (input channels)", "meaning": "The number of feature maps going into the block. After a skip connection this has doubled." },
    { "id": "cout", "name": "C_out (output channels)", "meaning": "The number of feature maps the block produces." },
    { "id": "cout2", "name": "+ C_out (biases)", "meaning": "One bias per output channel of the first convolution." },
    { "id": "d", "name": "D (time-embedding size)", "meaning": "The length of the time vector (32 in the small UNet, 128 in the example later in this section). A linear layer maps it to C_out numbers, costing D·C_out + C_out parameters." }
  ],
  "steps": [
    { "title": "Down block 1: first convolution", "expr": "9 × 1 × 16 + 16 = 160", "note": "A 3 × 3 kernel for each of 16 filters on a 1-channel picture, plus 16 biases." },
    { "title": "Second convolution", "expr": "9 × 16 × 16 + 16 = 2,320", "note": "16 channels in, 16 out." },
    { "title": "Time projection", "expr": "32 × 16 + 16 = 528", "note": "Maps the 32-number time vector to one number per channel." },
    { "title": "Total for down block 1", "expr": "160 + 2,320 + 528 = 3,008", "note": "It matches the table." },
    { "title": "Up block 1 after the skip", "expr": "9 × 32 × 16 + 16 = 4,624;  2,320;  528  →  4,624 + 2,320 + 528 = 7,472", "note": "The input has 32 channels because 16 come from the upsampling and 16 are copied from the skip. It matches the table." }
  ]
}
```

==The skip connection costs nothing by itself (copying is free)==, but it makes the next block’s first convolution bigger: 4,624 weights instead of the 2,320 it would have had with 16 channels in. That is the price of keeping the detail.

### Why the skips matter

Imagine the decoder had no skips. It would start from the 4 × 4 bottleneck and rebuild a 16 × 16 picture from it. The bottleneck knows the layout (“a crown over a trunk”) but has thrown away exactly where each edge was. The rebuilt picture would be a blur of the right shape. With skips the decoder also receives the 16 × 16 feature maps from the encoder’s first block, which still know where the edges are. ==Skip connections give the decoder both the big picture and the fine detail.== People usually report that removing them makes outputs blurrier, and in Lab Part 5 and Part 6 you test this by setting the skips to zero.

### The encoder side, shape by shape

The next figure shows only the encoder path of a UNet-like network and what it does to the shapes and the [[receptive field|The patch of the original picture that one output number depends on. It grows as you stack convolutions, and faster if some of them use stride 2.]], the patch of the original picture that one output number depends on.

```viz
{
  "type": "cnnstack",
  "title": "An encoder path: shapes and receptive field",
  "intro": "Five 3 × 3 convolutions with padding 1. Layers 2 and 4 use stride 2 to halve the resolution. Change a stride to 1 and watch what happens to the shapes and to the receptive-field column.",
  "input": [16, 16, 1],
  "layers": [
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1, "stride": 2 },
    { "type": "conv", "k": 3, "filters": 32, "pad": 1 },
    { "type": "conv", "k": 3, "filters": 32, "pad": 1, "stride": 2 },
    { "type": "conv", "k": 3, "filters": 64, "pad": 1 }
  ],
  "checks": [ { "layer": 0, "shape": [16, 16, 16], "params": 160, "rf": 3 }, { "layer": 1, "shape": [8, 8, 16], "params": 2320, "rf": 5 }, { "layer": 3, "shape": [4, 4, 32], "params": 9248, "rf": 13 }, { "layer": 4, "shape": [4, 4, 64], "params": 18496, "rf": 21 }, { "total": 34864 } ]
}
```

Look at the receptive-field column: 3, 5, 9, 13, 21. After the fifth layer, each of the 16 positions at 4 × 4 depends on a 21 × 21 patch, which is wider than the whole 16 × 16 picture, so the network is looking at everything at once. That is the global context. ==Downsampling is what made it cheap.==

```viz
{
  "type": "cnnstack",
  "title": "Five convolutions at full resolution",
  "intro": "The same five layers without any stride-2 layer. Compare the receptive field and the shapes with the previous figure.",
  "input": [16, 16, 1],
  "layers": [
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 },
    { "type": "conv", "k": 3, "filters": 16, "pad": 1 }
  ],
  "checks": [ { "layer": 0, "shape": [16, 16, 16], "params": 160, "rf": 3 }, { "layer": 2, "shape": [16, 16, 16], "params": 2320, "rf": 7 }, { "layer": 4, "shape": [16, 16, 16], "params": 2320, "rf": 11 }, { "total": 9440 } ]
}
```

Here the receptive field grows only 3, 5, 7, 9, 11: each 3 × 3 layer adds just 2. After five layers a position still sees only an 11 × 11 patch. Compare the cost: the full-resolution stack does about 2,396,160 multiply-adds for these shapes, and the downsampling stack about 921,600, roughly 2.6 times fewer, while seeing a wider area. This is exactly why the UNet shrinks the picture. The receptive-field growth rule is simple.

```viz
{
  "type": "equation",
  "title": "How a receptive field grows",
  "intro": "Each convolution layer widens the receptive field by (k − 1) times the current jump; a stride multiplies the jump. Follow the two stacks of five layers.",
  "formula": [ { "t": "RF_new", "id": "rfn" }, " = ", { "t": "RF", "id": "rf" }, " + (", { "t": "k", "id": "k" }, " − 1) · ", { "t": "jump", "id": "j" }, ",      ", { "t": "jump_new", "id": "jn" }, " = jump · ", { "t": "stride", "id": "s" } ],
  "parts": [
    { "id": "rfn", "name": "RF_new (receptive field after the layer)", "meaning": "The side length, in input pixels, of the patch that one output number depends on." },
    { "id": "rf", "name": "RF (before the layer)", "meaning": "The receptive field of the layer below. At the input it is 1." },
    { "id": "k", "name": "k (kernel size)", "meaning": "3 in a UNet’s usual convolution." },
    { "id": "j", "name": "jump", "meaning": "The distance, in input pixels, between neighbouring positions at this layer. It starts at 1 and doubles after each stride-2 layer." },
    { "id": "jn", "name": "jump_new", "meaning": "The jump for the next layer." },
    { "id": "s", "name": "stride", "meaning": "1 for an ordinary convolution, 2 for a downsampling one." }
  ],
  "steps": [
    { "title": "Five stride-1 layers", "expr": "RF = 1 → 3 → 5 → 7 → 9 → 11  (jump stays 1)", "note": "Each layer adds (3 − 1) × 1 = 2." },
    { "title": "Layer 1 of the encoder path", "expr": "RF = 1 + 2 × 1 = 3, jump 1", "note": "Same as in the stack with no stride." },
    { "title": "Layer 2 (stride 2)", "expr": "RF = 3 + 2 × 1 = 5, then jump = 1 × 2 = 2", "note": "The layer itself widens by 2, but later layers now step 2 pixels at a time." },
    { "title": "Layer 3", "expr": "RF = 5 + 2 × 2 = 9", "note": "Now each layer adds 4." },
    { "title": "Layers 4 and 5", "expr": "RF = 9 + 2 × 2 = 13, jump = 4;  RF = 13 + 2 × 4 = 21", "note": "21 covers the whole 16 × 16 picture." }
  ]
}
```

The two stacks differ in one decision only, but together they show why every image network that needs global context also shrinks the picture.

```viz
{
  "type": "flow",
  "title": "A UNet or a plain stack of convolutions at full size?",
  "modes": [
    {
      "name": "Plain stack at full size",
      "tagline": "Every layer keeps 16 × 16. Simple, but each layer sees only a small patch.",
      "inputLabel": "Input: noisy 16 × 16 picture",
      "modelLabel": "Five 3 × 3 convolutions, no downsampling",
      "outputLabel": "Output: 16 × 16 noise guess",
      "examples": [
        { "label": "Receptive field", "input": "3, 5, 7, 9, 11 after five layers", "output": "Never sees the whole 16 × 16 picture" },
        { "label": "Cost", "input": "About 2.4 million multiply-adds for these shapes", "output": "Every layer works on all 256 positions" },
        { "label": "Fine detail", "input": "Kept: full resolution all the way", "output": "Good detail, poor context" }
      ],
      "note": "The numbers were computed for the shapes in the two cnnstack figures above."
    },
    {
      "name": "UNet",
      "tagline": "Shrink to gather context, grow back, and copy detail across with skips.",
      "inputLabel": "Input: noisy 16 × 16 picture",
      "modelLabel": "Encoder, bottleneck, decoder, skips",
      "outputLabel": "Output: 16 × 16 noise guess",
      "examples": [
        { "label": "Receptive field", "input": "3, 5, 9, 13, 21 along the encoder", "output": "Covers the whole picture by the bottleneck" },
        { "label": "Cost", "input": "About 0.9 million multiply-adds for the encoder path above", "output": "Deeper layers work on only 64 or 16 positions" },
        { "label": "Fine detail", "input": "Recovered through the skip connections", "output": "Good detail and good context" }
      ],
      "note": "The encoder path is the cnnstack figure above; the multiply-adds are for that path only."
    }
  ]
}
```

Switch tabs: the UNet wins on both counts of the last two rows. Remember that ==shrinking buys context and cheapness, skips buy back detail==.

### The decoder grows the picture back

The decoder doubles the resolution with the transpose convolution of Week 12. In the small UNet the upsampler uses a 2 × 2 kernel with stride 2. Try it on a 2 × 2 input, and recall the stamp picture: each input cell stamps a scaled copy of the kernel, and with kernel 2 and stride 2 the stamps do not overlap, so no checkerboard artefact appears.

```viz
{
  "type": "conv",
  "title": "A 2 × 2 transpose convolution with stride 2",
  "mode": "transpose",
  "stride": 2,
  "input": [[1, 2], [3, 4]],
  "kernel": [[1, 0], [0, 1]]
}
```

The output is 4 × 4: the side is (2 − 1) × 2 + 2 = 4. Each of the four input numbers has stamped a diagonal onto its own 2 × 2 block. A learned kernel would stamp something more useful than a diagonal. The next figure recalls the other upsamplers of Week 12 for comparison.

```viz
{
  "type": "upsample",
  "title": "Three ways to double a picture’s side",
  "intro": "Switch the tabs and compare nearest neighbour, bilinear and the transpose convolutions. A UNet decoder usually uses one of the learnable ones (or an upsample followed by a convolution).",
  "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  "start": 4,
  "max": 9,
  "note": "Recalled from Week 12. The small UNet of this lesson uses the 2 × 2 transpose convolution with stride 2."
}
```

### Telling the network how noisy the input is

The UNet’s blocks do not know, by themselves, that this picture is at t = 20 or t = 900. The same image of faint speckle means “polish the edges” at t = 20 and could mean “this is nearly pure noise” at t = 900, and the right answer is very different. So the step number is fed in as a [[time embedding|A vector computed from the step number t, usually with sines and cosines, then passed through a small network and added into the UNet’s blocks so that every layer knows the noise level.]]. We cannot feed the plain integer t, because a single number is awkward for a network to use. Instead we turn t into a vector with the sine and cosine waves of [[sinusoidal positional encoding|A way to turn a position number into a vector of sines and cosines at many different frequencies, so that nearby positions get similar vectors. Week 8 used it for word positions; diffusion uses it for the step number.]] from Week 8, applied to the step number instead of a word position.

```viz
{
  "type": "equation",
  "title": "The sinusoidal time embedding",
  "intro": "Select each symbol. The worked example uses a 4-number embedding (d = 4) for three steps. Even entries are sines and odd entries are cosines.",
  "formula": [ { "t": "emb(t)", "id": "emb" }, "_(2k) = sin( ", { "t": "t", "id": "t" }, " / 10000^(2k/", { "t": "d", "id": "d" }, ") ),      emb(t)_(2k+1) = cos( t / 10000^(2k/d) )" ],
  "parts": [
    { "id": "emb", "name": "emb(t) (the time embedding)", "meaning": "A vector of d numbers that describes the step t. In the small UNet d = 32." },
    { "id": "t", "name": "t (the step number)", "meaning": "An integer from 1 to T. Different steps give different vectors, and nearby steps give similar ones." },
    { "id": "d", "name": "d (embedding size)", "meaning": "The number of entries in the vector. Pairs (sine, cosine) share a frequency, so there are d/2 frequencies, from fast to slow." }
  ],
  "steps": [
    { "title": "The frequencies for d = 4", "expr": "k = 0: 1/10000⁰ = 1;   k = 1: 1/10000^(2/4) = 1/100 = 0.01", "note": "A fast pair and a slow pair." },
    { "title": "t = 1", "expr": "(sin 1, cos 1, sin 0.01, cos 0.01) = (0.8415, 0.5403, 0.0100, 1.0000)", "note": "The slow pair is almost (0, 1): at small t it barely moves." },
    { "title": "t = 100", "expr": "(sin 100, cos 100, sin 1, cos 1) = (−0.5064, 0.8623, 0.8415, 0.5403)", "note": "The slow pair now equals the fast pair of t = 1, because 100 × 0.01 = 1." },
    { "title": "t = 300", "expr": "(sin 300, cos 300, sin 3, cos 3) = (−0.9998, −0.0221, 0.1411, −0.9900)", "note": "Every step gets its own pattern, and the slow pair tells coarse position (early versus late) while the fast pair tells fine position." }
  ]
}
```

Notice that the vector for t = 100 is not like the one for t = 1. The wave pairs act like the hands of a clock: the fast hand tells exactly where within a short cycle, and the slow hand tells roughly where overall. You can explore the full pattern below.

```viz
{
  "type": "posenc",
  "title": "Time embeddings as a pattern",
  "dModel": 16,
  "positions": 24
}
```

Read the figure as “rows are steps, columns are embedding entries”. Left columns change quickly from row to row and right columns slowly. Select a row to see how similar the other rows are. In Lab Part 5 you compute these vectors by hand for t = 1, 100 and 300 and compare nearby steps.

```viz
{
  "type": "heatmap",
  "title": "Which steps look alike? Cosine similarity of 32-number time embeddings",
  "intro": "Rows and columns are steps. A value near 1 means the two steps have almost the same embedding; a value near 0 means they are unrelated.",
  "rows": ["t = 1","t = 100","t = 101","t = 110","t = 200","t = 300"],
  "cols": ["t = 1","t = 100","t = 101","t = 110","t = 200","t = 300"],
  "corner": "step ↓ step →",
  "showValues": true,
  "max": 1,
  "W": [[1.0,0.5,0.58,0.38,0.42,0.17],[0.5,1.0,0.96,0.63,0.58,0.51],[0.58,0.96,1.0,0.62,0.5,0.42],[0.38,0.63,0.62,1.0,0.28,0.37],[0.42,0.58,0.5,0.28,1.0,0.58],[0.17,0.51,0.42,0.37,0.58,1.0]],
  "note": "Computed with the Week 8 sinusoidal formula for d = 32 (even entries sine, odd entries cosine). Values are rounded to two places."
}
```

Look at the row for t = 100: it is 0.96 similar to t = 101, 0.63 to t = 110 and 0.58 to t = 200. ==Neighbouring steps get similar embeddings==, which is just what you want: a network that behaves well at t = 100 should behave almost the same at t = 101.

### From an embedding to the feature maps

The sinusoidal vector is then passed through a small network of two linear layers with an activation between them, which lets the model learn what it needs from the step number. For the small UNet the sinusoidal vector has 32 numbers and the MLP maps 32 → 32 → 32, which is exactly the 2,112 parameters in the first row of the UNet table: 2 × (32 × 32 + 32).

```viz
{
  "type": "mlp",
  "title": "The time MLP of the small UNet",
  "intro": "A 32-number sinusoidal embedding goes through two linear layers. The table counts the weights and biases.",
  "layers": [32, 32, 32],
  "note": "Measured by the formula inputs × outputs + outputs for each layer: 2 × (32 × 32 + 32) = 2,112. The activation between the layers is a nonlinearity such as SiLU."
}
```

Each block then turns the time vector into one number per channel with its own linear layer, and **adds** that number to every pixel of the corresponding feature map. This is the same idea as a bias, except that the bias depends on t. It is a per-channel shift, and the shape arithmetic is a broadcast.

```viz
{
  "type": "equation",
  "title": "Adding the time to a feature map",
  "intro": "A time vector of 128 numbers becomes one number per channel, which is added to every pixel of a C × H × W feature map.",
  "formula": [ { "t": "h'", "id": "hp" }, "[c, i, j] = ", { "t": "h", "id": "h" }, "[c, i, j] + ", { "t": "b(t)", "id": "bt" }, "[c],      ", { "t": "b(t)", "id": "bt2" }, " = W · emb(t) + β" ],
  "parts": [
    { "id": "hp", "name": "h′ (the feature map after adding the time)", "meaning": "Same shape as h: C channels of H × W numbers." },
    { "id": "h", "name": "h (the feature map before)", "meaning": "What the first convolution of the block produced, with shape C × H × W." },
    { "id": "bt", "name": "b(t)[c] (the time bias for channel c)", "meaning": "One number per channel. The same number is added at every pixel (i, j) of channel c; this copying is called broadcasting." },
    { "id": "bt2", "name": "b(t) (the time bias vector)", "meaning": "The linear layer’s output. W has shape C × D and the bias has C entries, so it costs D·C + C parameters." }
  ],
  "steps": [
    { "title": "The sizes", "expr": "time vector D = 128, channels C = 64, feature map 64 × 8 × 8", "note": "An example of a mid-sized block; the numbers are illustrative." },
    { "title": "Linear layer", "expr": "128 → 64, parameters 128 × 64 + 64 = 8,256", "note": "Gives one number per channel." },
    { "title": "Reshape for adding", "expr": "64 numbers → shape (64, 1, 1)", "note": "A trailing pair of size-1 axes so that each number can be spread over the whole map." },
    { "title": "Broadcast add", "expr": "(64, 8, 8) + (64, 1, 1) → (64, 8, 8); 64 × 8 × 8 = 4,096 additions", "note": "The shape is unchanged. Each channel is shifted by its own amount, and the amounts depend on t." }
  ]
}
```

Because every block gets its own copy of this shift, ==one network handles every noise level==: at t = 20 the shifts push the features towards “fine detail”, at t = 900 towards “coarse shapes”. Without the time input the network must guess the noise level from the picture alone, which is sometimes possible (a very noisy picture looks different from a nearly clean one) but wasteful and unreliable. You test this in Lab Part 5, where you train the same denoiser with and without the time embedding.

```viz
{
  "type": "pipeline",
  "title": "Where the time goes in",
  "intro": "Step through one UNet block with its time embedding.",
  "nodes": [
    { "id": "t", "label": "Step t", "sub": "an integer" },
    { "id": "sin", "label": "Sinusoidal embedding", "sub": "32 numbers" },
    { "id": "mlp", "label": "Small MLP", "sub": "32 → 32 → 32" },
    { "id": "lin", "label": "Linear per block", "sub": "→ one number per channel" },
    { "id": "add", "label": "Add to feature map", "sub": "broadcast" },
    { "id": "blk", "label": "Rest of the block", "sub": "activation, second convolution" }
  ],
  "steps": [
    { "title": "The step number", "body": "The integer t (for example 300) is the only thing about the noise level that the network is told.", "active": ["t"], "example": "t = 300." },
    { "title": "Sines and cosines", "body": "The sinusoidal formula turns t into a vector of d numbers whose pattern is different for every t and similar for nearby t.", "active": ["t", "sin"], "example": "32 numbers between −1 and 1." },
    { "title": "A small MLP", "body": "Two linear layers let the model learn which parts of that pattern matter.", "active": ["sin", "mlp"], "example": "Shared by all the blocks of the UNet." },
    { "title": "One number per channel in each block", "body": "Each block has its own linear layer that maps the time vector to as many numbers as the block has channels.", "active": ["mlp", "lin"], "example": "For a block with 32 channels: 32 numbers." },
    { "title": "Add and carry on", "body": "The numbers are added to the feature map after the first convolution, then the block continues. The shape does not change.", "active": ["lin", "add", "blk"], "example": "(32, 8, 8) + (32, 1, 1) → (32, 8, 8)." }
  ]
}
```

### Bigger UNets: more levels, attention, and Transformers

Real image generators are bigger versions of the same design. Slide the controls below to a 32 × 32 picture with four levels and 32 base channels, and watch the totals grow past two million.

```viz
{
  "type": "unet",
  "title": "A larger UNet for 32 × 32 pictures",
  "intro": "Four levels, 32 base channels and a 64-number time embedding. Compare the total with the small one and find the biggest block.",
  "size": 32,
  "inCh": 1,
  "base": 32,
  "levels": 4,
  "timeDim": 64,
  "note": "Same conventions as the small UNet. Production models are much larger and add more pieces (normalisation, attention).",
  "checks": [ { "total": 2172865 }, { "layer": "bottleneck", "params": 901888 }, { "layer": "up block 1", "params": 29792 } ]
}
```

The total is 2,172,865 and the bottleneck is again the biggest part (901,888, about 42 percent). Larger UNets add [[self-attention|A layer, introduced in Week 7, in which every position of a feature map looks at every other position and mixes information according to how relevant each is. It gives a network global context in one layer.]] layers at their lowest resolutions, where the grid is small enough to afford them (Week 7), so that every position can look at every other. Some newer models replace the UNet by a Transformer that works on patches of the picture, for example the diffusion Transformer (DiT) of Peebles and Xie (2023). The shape of the job is the same: noisy tokens and a time input in, noise out.

### Sorting the parts

```viz
{
  "type": "sorter",
  "title": "Which part of the UNet does the job?",
  "intro": "Sort each job to the part of the denoiser that provides it.",
  "bins": [ { "id": "encoder", "label": "Encoder" }, { "id": "decoder", "label": "Decoder" }, { "id": "skip", "label": "Skip connection" }, { "id": "time", "label": "Time embedding" } ],
  "cards": [
    { "name": "Halves the height and width while increasing the channels", "answer": "encoder", "why": "Downsampling convolutions shrink the picture, and each block adds channels, so the later layers see a wider area." },
    { "name": "Brings the picture back to its original size with transpose convolutions", "answer": "decoder", "why": "The decoder doubles the side again at each level until the output matches the input shape." },
    { "name": "Carries full-resolution edges from an early layer to a late one", "answer": "skip", "why": "The copy lets the decoder use detail that the narrow bottleneck threw away." },
    { "name": "Tells the network whether this is step 20 or step 900", "answer": "time", "why": "It is the only place where the noise level enters the network." },
    { "name": "Doubles the number of channels at a join, by concatenation", "answer": "skip", "why": "Concatenating a copy with the upsampled map gives twice the channels: 16 + 16 = 32 in the small UNet." },
    { "name": "Gives each position a wider and wider receptive field", "answer": "encoder", "why": "Stride-2 convolutions make the jump larger, so the receptive field grows 3, 5, 9, 13, 21 in the example." },
    { "name": "Becomes one number per channel that is added to a feature map", "answer": "time", "why": "A linear layer maps the time vector to C numbers, and the sum is broadcast over height and width." },
    { "name": "Produces the final noise guess with the same shape as the input", "answer": "decoder", "why": "The decoder’s last layer is a 1 × 1 convolution that gives one channel at full resolution." }
  ]
}
```

By now you can answer the question the course asks: ==why a UNet with a time embedding?== Because the job needs a picture-shaped output with fine detail, wide context and knowledge of the noise level, and a UNet with skip connections and a time embedding is built for exactly those four things.

## Making it useful: guidance and latent diffusion {rail=Guidance}

{lede} So far the model makes a random picture from the training data: you press the button and get *something*. A useful tool needs two more things. You need to steer it (“a jeepney on a rainy street”), and it must be cheap enough to run on one GPU. ==Conditioning with classifier-free guidance gives you the steering; latent diffusion gives you the cheapness.== Together they are the recipe behind [[Stable Diffusion|An open text-to-image latent diffusion model made of a text encoder, a UNet denoiser that works in latent space, and an image decoder.]]. This section is conceptual: you will read the ideas, do the arithmetic, and (optionally) run the real model in the lab.

### Conditioning: giving the denoiser a hint

==Conditioning means giving the denoiser extra input c== besides x_t and t, so that the noise guess becomes ε_θ(x_t, t, c). The condition might be a class label (“cat”, “crack”, “digit 7”) or the embedding of a text prompt. For text, a separate encoder turns the prompt into a sequence of vectors, much like the CLIP text encoder you met in Week 11, and the UNet reads those vectors through [[cross-attention|An attention layer in which the queries come from one sequence (the image features) and the keys and values come from another (the prompt). It lets every picture position look at the words that matter for it.]] layers, an application of the attention of Week 7. Each picture position asks “which words are relevant to me?” and mixes in what the words say.

```viz
{
  "type": "pipeline",
  "title": "A conditioned denoiser",
  "intro": "Step through how a text prompt reaches the noise guess. Shapes are for Stable Diffusion v1 and are examples.",
  "nodes": [
    { "id": "prompt", "label": "Prompt", "sub": "“a jeepney in the rain”" },
    { "id": "enc", "label": "Text encoder", "sub": "words → vectors" },
    { "id": "xt", "label": "Noisy latent x_t + step t", "sub": "the picture so far" },
    { "id": "unet", "label": "UNet with cross-attention", "sub": "picture queries, word keys" },
    { "id": "eps", "label": "Noise guess ε̂", "sub": "depends on the prompt" }
  ],
  "steps": [
    { "title": "Turn the words into vectors", "body": "A text encoder reads the prompt and returns one vector per token. In Stable Diffusion v1 the encoder is a CLIP text model and the result is 77 vectors of 768 numbers each.", "active": ["prompt", "enc"], "example": "77 × 768 numbers, the same for every step of the loop." },
    { "title": "Keep the usual inputs", "body": "The noisy picture and the step number go into the UNet exactly as before. The prompt’s vectors are an extra input that does not change at every step.", "active": ["xt", "unet"], "example": "x_t and t as in the rest of this lesson." },
    { "title": "Look at the words with cross-attention", "body": "Inside the UNet, cross-attention layers let each picture position read the prompt’s vectors. The noise guess therefore depends on what you asked for.", "active": ["enc", "unet"], "example": "The position near the road asks which words talk about roads." },
    { "title": "A guess that depends on the prompt", "body": "The output is still a noise guess with the same shape as the input, but it is now ε_θ(x_t, t, c). Run in a loop, it steers the static towards pictures that match the prompt.", "active": ["unet", "eps"], "example": "Same prompt, different static: different jeepneys." }
  ]
}
```

Notice that the prompt only changes the *noise guess*. The sampler loop of the previous section is untouched.

### Classifier-free guidance

A conditioned model that follows the prompt weakly would give pictures that match roughly. A neat trick, [[classifier-free guidance|A way to make a conditioned diffusion model follow its condition more strongly. At each step the network is run with and without the condition, and the result is pushed further in the direction that the condition adds.]] (Ho and Salimans, 2021), makes it follow strongly without training a separate classifier. It has two parts.

1. **Training with the condition sometimes dropped.** For a small fraction of the training examples (often something like 10 to 20 percent) the condition is replaced by an empty one, written ∅. The same network therefore learns two jobs: guess the noise given the prompt, and guess the noise with no prompt at all.
2. **Sampling with both.** At each step run the network twice, once with the prompt c and once with ∅, and combine the two guesses.

```viz
{
  "type": "equation",
  "title": "Classifier-free guidance",
  "intro": "Select each symbol. The worked example uses a two-pixel noise guess and the guidance scale w = 3.",
  "formula": [ { "t": "ε̃", "id": "et" }, " = ", { "t": "ε(x_t, ∅)", "id": "eu" }, " + ", { "t": "w", "id": "w" }, " · ( ", { "t": "ε(x_t, c)", "id": "ec" }, " − ε(x_t, ∅) )" ],
  "parts": [
    { "id": "et", "name": "ε̃ (the guided guess)", "meaning": "The noise guess actually used in the reverse step." },
    { "id": "eu", "name": "ε(x_t, ∅) (unconditional guess)", "meaning": "What the network says with an empty prompt. It describes noise in a picture like those in the data set." },
    { "id": "w", "name": "w (guidance scale)", "meaning": "How far to push in the direction of the prompt. w = 0 ignores the prompt, w = 1 is the plain conditional guess, w > 1 exaggerates it." },
    { "id": "ec", "name": "ε(x_t, c) (conditional guess)", "meaning": "What the network says given the prompt c." }
  ],
  "steps": [
    { "title": "The two guesses", "expr": "ε(x_t, ∅) = (0.2, −0.1),  ε(x_t, c) = (0.5, −0.4)", "note": "Illustrative values for two pixels." },
    { "title": "The direction the prompt adds", "expr": "ε(x_t, c) − ε(x_t, ∅) = (0.3, −0.3)", "note": "This is what the prompt changed about the guess." },
    { "title": "Push w = 3 times as far", "expr": "3 × (0.3, −0.3) = (0.9, −0.9)", "note": "Three times the prompt’s effect." },
    { "title": "Add back to the unconditional guess", "expr": "ε̃ = (0.2, −0.1) + (0.9, −0.9) = (1.1, −1.0)", "note": "The guided guess is far larger than either input: that is the exaggeration." },
    { "title": "Other values of w", "expr": "w = 0: (0.2, −0.1);  w = 1: (0.5, −0.4);  w = 7.5: (2.45, −2.35)", "note": "w = 1 gives back the plain conditional guess." }
  ]
}
```

Notice that the formula takes the unconditional guess and adds w copies of “what the prompt changed”. The [[guidance scale|The number w in classifier-free guidance. Higher w follows the prompt more closely and gives less variety.]] is the only knob. Higher values make the picture match the prompt more closely and look less varied; very high values often give harsh, over-saturated pictures. Values often around 7 to 8 are common in Stable Diffusion tools. ==Guidance has a cost: two network calls per step==, so a 25-step sampler with guidance makes 50 calls.

```viz
{
  "type": "flow",
  "title": "The guidance scale w",
  "modes": [
    {
      "name": "Low w (about 1)",
      "tagline": "The prompt is a gentle hint.",
      "inputLabel": "Input: static and a prompt",
      "modelLabel": "ε̃ with a small push towards the prompt",
      "outputLabel": "Output: varied, loosely related pictures",
      "examples": [
        { "label": "Prompt match", "input": "Weak", "output": "The picture may ignore parts of the prompt" },
        { "label": "Variety", "input": "High", "output": "Different static gives very different pictures" }
      ],
      "note": "A general tendency described in words, not a measurement."
    },
    {
      "name": "Medium w (often around 7 to 8)",
      "tagline": "A common balance.",
      "inputLabel": "Input: static and a prompt",
      "modelLabel": "ε̃ with a clear push",
      "outputLabel": "Output: pictures that follow the prompt",
      "examples": [
        { "label": "Prompt match", "input": "Good", "output": "Most details in the prompt appear" },
        { "label": "Variety", "input": "Moderate", "output": "Different static still gives different pictures" }
      ],
      "note": "A typical default range, not a rule. The best value depends on the model and the prompt."
    },
    {
      "name": "High w (15 or more)",
      "tagline": "The prompt is shouted.",
      "inputLabel": "Input: static and a prompt",
      "modelLabel": "ε̃ with a very large push",
      "outputLabel": "Output: loud, repetitive pictures",
      "examples": [
        { "label": "Prompt match", "input": "Very strong", "output": "Often exaggerated" },
        { "label": "Variety", "input": "Low", "output": "Colours can look harsh and over-saturated" }
      ],
      "note": "A general tendency; try it yourself in the optional lab stretch if a GPU is available."
    }
  ]
}
```

==The pattern is a trade between obeying the prompt and staying varied and natural.== In Lab Part 6.8 (an optional stretch) you train a small class-conditional model, drop the label 15 percent of the time, and measure the trade with a judge classifier.

### Latent diffusion: working in a smaller space

A 512 × 512 colour picture has 786,432 numbers. A UNet running 50 steps over that many numbers is slow and memory-hungry. [[Latent diffusion|Running the diffusion process in the compressed latent space of an autoencoder instead of in pixel space. Pictures are encoded to a small latent, denoised there, and decoded back to pixels at the end.]] (Rombach and colleagues, 2022) solves this by compressing the picture first. An [[autoencoder|A pair of networks: an encoder that compresses a picture into a small latent tensor and a decoder that rebuilds the picture from it. Week 12 showed that a plain autoencoder reconstructs but does not invent.]] is trained once to squeeze pictures into a small latent tensor and expand them back. The diffusion model then lives entirely in the latent space, and ==only the very last step decodes to pixels==.

```viz
{
  "type": "equation",
  "title": "How much smaller is the latent?",
  "intro": "Select the parts. The worked example uses the sizes of Stable Diffusion v1: a 512 × 512 × 3 picture and a 64 × 64 × 4 latent.",
  "formula": [ { "t": "ratio", "id": "r" }, " = ( ", { "t": "H·W·C", "id": "pix" }, " ) / ( ", { "t": "h·w·c", "id": "lat" }, " )" ],
  "parts": [
    { "id": "r", "name": "ratio (compression)", "meaning": "How many times fewer numbers the diffusion model has to handle." },
    { "id": "pix", "name": "H·W·C (the picture)", "meaning": "Height × width × colour channels of the picture in pixel space." },
    { "id": "lat", "name": "h·w·c (the latent)", "meaning": "Height × width × channels of the latent. In Stable Diffusion v1 the side is 8 times smaller and there are 4 channels." }
  ],
  "steps": [
    { "title": "Pixel picture", "expr": "512 × 512 × 3 = 786,432", "note": "The numbers a pixel-space model would denoise." },
    { "title": "Latent", "expr": "64 × 64 × 4 = 16,384", "note": "Each side is 512/64 = 8 times smaller." },
    { "title": "Ratio", "expr": "786,432 / 16,384 = 48", "note": "48 times fewer numbers to denoise at every step." },
    { "title": "Attention table", "expr": "positions: 64 × 64 = 4,096 against 512 × 512 = 262,144, a factor of 64;  scores: 4,096² = 16,777,216 against 262,144² = 68,719,476,736, a factor of 4,096", "note": "Attention compares every position with every other, so its cost grows with the square of the number of positions. That is why shrinking the grid helps even more than the 48 suggests." }
  ]
}
```

To feel the square law, use the attention calculator below. Set the sequence length n to 4,096 and then to a much larger number, and compare the n × n score table.

```viz
{
  "type": "calc",
  "kind": "attn_cost",
  "title": "Why a smaller grid helps attention"
}
```

Notice how fast the score table grows. At the latent’s 4,096 positions it is about 16.8 million scores; at 262,144 positions it would be 68.7 billion. ==Diffusion in latent space is what lets a UNet with attention run on a single GPU.==

```viz
{
  "type": "nested",
  "title": "Where Stable Diffusion sits",
  "intro": "Click a ring or its chip to read it. The outermost ring is the widest idea.",
  "levels": [
    { "name": "Generative models", "blurb": "Models that make new data like their training data: GANs, VAEs, language models and diffusion models.", "examples": ["GAN", "VAE", "GPT", "diffusion"] },
    { "name": "Diffusion models", "blurb": "Generative models that learn to reverse a gradual noising process and sample by denoising step by step.", "examples": ["DDPM", "DDIM"] },
    { "name": "Latent diffusion models", "blurb": "Diffusion done in the compressed latent space of an autoencoder, which makes it much cheaper than pixel space.", "examples": ["Rombach et al. 2022"] },
    { "name": "Stable Diffusion", "blurb": "An open text-to-image latent diffusion model: a text encoder, a UNet with cross-attention, and an image decoder.", "examples": ["Stable Diffusion v1"] }
  ]
}
```

### The three parts of Stable Diffusion

Stable Diffusion, the model of the lab’s optional stretch, is three networks in a row. Each was trained in its own way, and together they turn a sentence into a picture.

```viz
{
  "type": "pipeline",
  "title": "Stable Diffusion at a glance (v1 sizes)",
  "intro": "Step through the three parts. The shapes are for Stable Diffusion v1 and are examples, not a recipe for every model.",
  "nodes": [
    { "id": "prompt", "label": "Prompt", "sub": "text" },
    { "id": "text", "label": "Text encoder", "sub": "77 × 768" },
    { "id": "noise", "label": "Latent noise", "sub": "4 × 64 × 64" },
    { "id": "unet", "label": "UNet denoiser", "sub": "in latent space" },
    { "id": "vae", "label": "Image decoder", "sub": "latent → pixels" },
    { "id": "img", "label": "Picture", "sub": "3 × 512 × 512" }
  ],
  "steps": [
    { "title": "Read the prompt", "body": "The text encoder turns the prompt into a sequence of vectors that the UNet will read by cross-attention. This happens once per prompt.", "active": ["prompt", "text"], "example": "77 token vectors of 768 numbers." },
    { "title": "Start from latent noise", "body": "The sampler starts from random noise in the latent space, not from a noisy 512 × 512 picture. This is 16,384 numbers, 48 times fewer than the final picture.", "active": ["noise"], "example": "4 × 64 × 64 random numbers." },
    { "title": "Denoise in the latent space", "body": "The UNet, guided by the prompt’s vectors and by classifier-free guidance, removes noise step by step. All the heavy work happens here on the small grid.", "active": ["text", "noise", "unet"], "example": "num_inference_steps calls (twice that many network passes with guidance)." },
    { "title": "Decode to pixels", "body": "Only at the end does the image decoder expand the clean latent into a full picture. It is called once, not once per step.", "active": ["unet", "vae", "img"], "example": "4 × 64 × 64 → 3 × 512 × 512." }
  ]
}
```

In the `diffusers` library the two settings you will play with are `num_inference_steps` (how many denoising steps the sampler takes) and `guidance_scale` (the w of classifier-free guidance). The optional GPU stretch of Lab Part 7 runs the model if your instructor has it cached; it needs no download and no `pip install`.

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
g = torch.Generator("cuda").manual_seed(42)          # the same static every time
image = pipe("a jeepney on a rainy street", num_inference_steps=30,
             guidance_scale=7.5, generator=g).images[0]
```

Run the call with `num_inference_steps` set to 5, 15 and 30, and with `guidance_scale` set to 1, 7.5 and 15, always with the same seed and prompt, and compare. Do this only if the model is in the offline cache and a GPU is free.

```viz
{
  "type": "sorter",
  "title": "Which part of Stable Diffusion?",
  "intro": "Sort each job to the network that does it.",
  "bins": [ { "id": "text", "label": "Text encoder" }, { "id": "unet", "label": "UNet denoiser (latent space)" }, { "id": "vae", "label": "Image decoder" } ],
  "cards": [
    { "name": "Turns the words of the prompt into 77 vectors", "answer": "text", "why": "It reads the prompt once and produces the vectors that the UNet reads by cross-attention." },
    { "name": "Is called once per denoising step", "answer": "unet", "why": "The sampler asks the UNet for a noise guess at every step, which is why it dominates the running time." },
    { "name": "Expands a 4 × 64 × 64 latent into a 3 × 512 × 512 picture", "answer": "vae", "why": "The decoder is the final stage: it converts the clean latent back to pixels." },
    { "name": "Contains cross-attention layers that look at the prompt", "answer": "unet", "why": "The prompt’s vectors enter the UNet through cross-attention, so the noise guess depends on the text." },
    { "name": "Runs only once for each prompt, before the loop starts", "answer": "text", "why": "The prompt does not change during sampling, so its vectors can be computed a single time." },
    { "name": "Is called exactly once per picture, after the loop ends", "answer": "vae", "why": "Decoding happens once, on the finished latent, not at every step." }
  ]
}
```

### Diffusion and GANs side by side

In Week 12 you built a GAN. Now you can compare the two families properly. Remember that these are typical tendencies, not laws, and that both families have many variants.

| | GAN (Week 12) | Diffusion (this week) |
|---|---|---|
| How it is trained | A game between a generator and a discriminator | One network, a plain squared error on the noise |
| Is the training stable? | Often delicate: balance, collapse, vanishing gradients | Typically steady |
| Coverage of the data | Can collapse onto a few modes | Typically covers the variety of the data |
| Network calls to make a picture | 1 | About 20 to 50 with a fast sampler, up to 1,000 with DDPM |
| How you steer it | Conditional GANs, a label fed in | Conditioning, cross-attention and guidance |
| Judged by | The judge idea of Week 12 | The same judge idea (Lab Part 6) |

```viz
{
  "type": "sorter",
  "title": "GAN, diffusion, or both?",
  "intro": "Sort each statement. Use the table above and Week 12.",
  "bins": [ { "id": "gan", "label": "GAN" }, { "id": "diffusion", "label": "Diffusion" }, { "id": "both", "label": "Both" } ],
  "cards": [
    { "name": "Makes a picture in a single forward pass of the generator", "answer": "gan", "why": "A GAN generator maps noise to a picture in one pass, so sampling is fast." },
    { "name": "Is trained with a plain mean squared error", "answer": "diffusion", "why": "The denoiser is trained to match the noise we added, with no discriminator." },
    { "name": "Can fall into mode collapse", "answer": "gan", "why": "A generator can learn to produce only a few outputs that fool the discriminator. Diffusion’s training objective does not reward this." },
    { "name": "Needs many network calls to make one picture", "answer": "diffusion", "why": "Sampling repeats the denoiser for each step: about 20 to 50 with DDIM and up to 1,000 with DDPM." },
    { "name": "Can be conditioned on a class label or a text embedding", "answer": "both", "why": "Conditional GANs and conditioned diffusion models both take an extra input. Diffusion adds guidance on top." },
    { "name": "Starts from random noise", "answer": "both", "why": "A GAN generator starts from a latent vector z; a diffusion sampler starts from the noisy picture x_T." },
    { "name": "Needs a discriminator network during training", "answer": "gan", "why": "The discriminator is the adversary that defines the GAN game. Diffusion has none." },
    { "name": "Is judged well by a classifier’s confidence, entropy and class coverage", "answer": "both", "why": "The Week 12 judge idea does not care how the pictures were made, so you can measure both families with the same tool." }
  ]
}
```

### Using these tools responsibly

Everything in Week 12 about generated images applies here, and often with more force, because a text prompt makes realistic pictures easy for anyone. Four points are worth remembering. ==Synthetic images can mislead if they are passed off as real photographs.== Pictures of real people need consent. Models are trained on large collections of images, and artists and photographers have raised concerns about their work being used without permission, so ask where training data came from. Finally, labelling helps: [[watermarking|Hiding a signal inside generated pictures so that a tool can later recognise them as generated.]] and provenance records (a signed note of where a picture came from) are two ways to mark generated images, and neither is perfect.

What should a school do? Keep it simple: have a clear policy that says what is allowed and that generated images must be labelled, teach students to check the source of a picture before sharing it, and never judge a picture as fake on the word of a detector alone. These are general ideas, not a legal guide.

### A short history

The ideas of this lesson arrived over about seven years. Click the events.

```viz
{
  "type": "timeline",
  "title": "Milestones of diffusion models",
  "events": [
    { "year": "2015", "title": "The first diffusion model", "text": "Sohl-Dickstein and colleagues proposed learning to reverse a gradual noising process, the idea behind everything in this lesson." },
    { "year": "2020", "title": "DDPM", "text": "Ho, Jain and Abbeel showed that noise prediction with a simple loss gives high-quality pictures, and gave the sampler you studied." },
    { "year": "2021", "title": "DDIM, the cosine schedule and guidance", "text": "Song, Meng and Ermon introduced DDIM for faster sampling; Nichol and Dhariwal proposed the cosine schedule; Ho and Salimans proposed classifier-free guidance." },
    { "year": "2022", "title": "Latent diffusion and Stable Diffusion", "text": "Rombach and colleagues moved the diffusion process into the latent space of an autoencoder, which made text-to-image models far cheaper to run." }
  ]
}
```

A second timeline places the denoiser’s own ingredients on the same line.

```viz
{
  "type": "timeline",
  "title": "Where the denoiser’s parts come from",
  "events": [
    { "year": "2014", "title": "GANs", "text": "Goodfellow and colleagues introduced the generator and discriminator game that you studied in Week 12." },
    { "year": "2015", "title": "The U-Net", "text": "Ronneberger, Fischer and Brox designed the U-shaped network with skip connections for biomedical image segmentation." },
    { "year": "2017", "title": "The Transformer", "text": "Vaswani and colleagues introduced attention-only models, whose attention layers later found their way into the UNets of diffusion models." },
    { "year": "2020", "title": "A UNet inside DDPM", "text": "The DDPM denoiser was a UNet-style network with a time embedding, the design you built in this lesson." },
    { "year": "2023", "title": "Diffusion Transformers", "text": "Peebles and Xie showed that a Transformer working on patches can replace the UNet as the denoiser (DiT)." }
  ]
}
```

==Notice that the UNet predates diffusion by five years==: it was borrowed from medical image segmentation because it fits the denoising job so well.

## Capstone opens this week {rail=Capstone}

The Structural Inspection Assistant capstone opens in Week 13 and runs to Week 20. It is a team project that triages inspection photos and drafts a summary for an engineer to review; the course outline gives it as 20 percent of the final grade, and your instructor announces the full weighting. The pipeline is: photo in, classify (crack, spalling, corrosion or none), find checklist items with RAG, then draft the summary. It uses vision models (ViT or CLIP), RAG over checklists, and NeMo Guardrails. One rule is fixed: ==a licensed engineer makes every pass/fail decision, and the assistant never signs off on safety==. Teams have a Team lead, a Technical lead and a Responsible-AI lead. A capstone proposal session takes place in Week 15, and your instructor announces dates and team arrangements. Read the full brief on the [Capstone](/capstone) page, and think about which of the models you have met so far could handle each step.

### Where could diffusion help, and where not?

You will probably not use a diffusion model as a main part of the assistant, because its main jobs (classify a photo, find checklist items, draft a summary) are done by other models. But a diffusion model could play a ==supporting role==. One idea is to make extra synthetic training pictures of the classes that are rare in your photo collection, for example early-stage corrosion, to balance a small data set. If you try this, treat it as a hypothesis to test, not as a guaranteed improvement.

- **Realism is a risk.** A generated crack might look unlike a real one. ==A classifier trained on unrealistic cracks can learn the wrong cues.==
- **Labels still need a person.** A generated picture is labelled by the prompt, but the prompt does not guarantee the picture shows what you asked for. Have a person check a sample.
- **Test only on real photos.** Synthetic pictures may help training, but a score measured on synthetic pictures says nothing about the real world. ==Keep the test set real and untouched.==
- **Never alter evidence.** A generative model must never be used to clean up, sharpen, “repair” or fill in a photo that an engineer will use to decide whether a structure is safe, because it can invent texture that was not there. ==The engineer must see the original photo.==
- **Say what you did.** Label any synthetic data, and write it up in your documentation and Responsible-AI notes.

```viz
{
  "type": "pipeline",
  "title": "Where a diffusion model could (and could not) touch the capstone pipeline",
  "intro": "Step through the Structural Inspection Assistant and see which step could use synthetic data and which must stay untouched.",
  "nodes": [
    { "id": "photo", "label": "Photo in", "sub": "real, unaltered" },
    { "id": "classify", "label": "Classify", "sub": "crack, spalling, corrosion, none" },
    { "id": "rag", "label": "Find checklist items", "sub": "RAG" },
    { "id": "draft", "label": "Draft summary", "sub": "for the engineer" },
    { "id": "eng", "label": "Engineer decides", "sub": "every pass/fail" }
  ],
  "steps": [
    { "title": "Photo in: never touched", "body": "The photo is the evidence. A generative model must not enhance, repair or extend it. If a photo is poor, ask for a better one.", "active": ["photo"], "example": "Original file kept, unchanged." },
    { "title": "Classify: the one place synthetic data might help", "body": "To balance a small collection, you could add some synthetic pictures of a rare class to the training set only. Keep the test set real and have a person check a sample of the synthetic pictures.", "active": ["photo", "classify"], "example": "Train on real + a labelled share of synthetic; test on real only." },
    { "title": "RAG and summary: not a job for diffusion", "body": "Finding checklist items and drafting text are language tasks. Diffusion for images does not help here.", "active": ["rag", "draft"], "example": "Use the retrieval and language models from earlier weeks instead." },
    { "title": "Engineer decides", "body": "A licensed engineer makes every pass/fail decision. No generated picture ever replaces an inspection.", "active": ["draft", "eng"], "example": "The assistant drafts; the engineer signs off." }
  ]
}
```

Lab Part 7.4 asks you to write a short plan for how a generative model could (or could not) help your capstone, covering where synthetic images might help, what could go wrong, how you would evaluate it with a judge and a human check, and what you will not do.

## Key takeaways

- ==The forward process adds noise by a fixed recipe, x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε==. With the linear schedule ᾱ_300 = 0.3964 (weights 0.630 and 0.777) and ᾱ_1000 = 4.04e-5, so the end of the forward process is pure noise; SNR_300 = 0.657.
- ==The network is trained to predict the noise with a mean squared error==, at a random step for each example. The target is known exactly and there is no adversary, so training is usually steadier than a GAN’s.
- ==Sampling starts from noise and loops==: x_(t−1) = (x_t − β_t/√(1 − ᾱ_t) · ε̂)/√α_t + σ_t · z. It costs one network call per step (1,000 for DDPM against 1 for a GAN); DDIM reuses the same network with 20 to 50 bigger steps, a 20 to 50 times speed-up.
- ==The denoiser is a UNet with a time embedding==: the same shape in and out, an encoder for context, a decoder with skip connections for detail, and a sinusoidal embedding of t added to every block so that one network handles every noise level.
- ==Classifier-free guidance, ε̃ = ε(∅) + w·(ε(c) − ε(∅)), steers the model towards a prompt, and latent diffusion runs the loop in a latent 48 times smaller==, which is how Stable Diffusion is steerable and affordable. Diffusion trades GANs’ one-pass speed for steady training and wide coverage.

<!-- LAB-START -->
## Lab: Imagine the sculptor and the fog {rail=Laboratory}

This lab is called *Imagine the sculptor and the fog*. Week 13 asks a question that sounds like a riddle: how can a network make a picture by *removing noise*? Picture a sculptor who reveals a statue by chipping away a block of stone, and a photograph that slowly disappears into fog. A diffusion model learns to run that film backwards. You answer the riddle by building the story from the bottom, with numbers and pictures that you make yourself. *How do you destroy a picture on purpose?* You write the forward process by hand: the noise schedule, one step, the closed form that jumps straight to step t, the signal-to-noise ratio, and real pictures disappearing into static; then you compare the linear and the cosine schedule and count the steps that a schedule wastes. *How do you undo one step?* You write the reverse step, derive the perfect noise predictor for a simple data set and watch pure noise turn into two bumps with no network at all, and then you break the loop on purpose. Next you train a first denoiser on a ring of eight blobs and compare it with the GAN of Week 12, take the time embedding and the UNet apart, and train a complete diffusion model (DDPM) on the line shapes of Week 4, sample from it with the slow and the fast sampler (DDIM) and judge the pictures with a classifier. You finish with guidance, the arithmetic of latent diffusion, the cost of sampling, an optional Stable Diffusion experiment on a GPU, and a plan for how a generative model could (and could not) help in the capstone. ==The goal is to replace the mystery of "AI that draws pictures from noise" by a recipe that you computed, trained and sampled yourself, and to learn that a diffusion model is trained by an ordinary loss (predict the noise) with no game against an opponent.== Every part begins with an **Imagine first** moment: you write a prediction before you run any code and compare it with what really happens. Each part points back to a section of the lesson. Nothing is downloaded in this lab, except in one optional GPU experiment at the end: every data set is drawn with NumPy, everything runs on a CPU, and the longest training run takes about two minutes. This page contains the setup and the first three parts (the forward process, noise schedules, the reverse step and a perfect denoiser); the other four parts, the deliverables and the marking scheme follow in the same lab.

| Part | What you do | Minutes |
|---|---|---|
| Setup | Start JupyterLab, paste the helper cells, run a smoke test | 10 |
| 1 | The forward process by hand | 25 |
| 2 | Noise schedules | 20 |
| 3 | The reverse step and a perfect denoiser | 25 |
| 4 | Train a denoiser on 2-D data | 25 |
| 5 | Time embeddings, the UNet and its parts | 30 |
| 6 | A DDPM on tiny pictures | 35 |
| 7 | Guidance, latent space, responsible use and the capstone | 10 |

The total is 180 minutes, about 3 hours. The marking scheme and the deliverables are at the end of the lab.

### Setup

1. Log in to the portal, start JupyterLab on the DGX Spark, create a folder `week13` and a notebook `lab13.ipynb`. Run each block of this lab in its own cell, in order: later blocks use names (`make_schedule`, `q_sample`, `abar`, `reverse_step`, `RESULTS` and so on) that earlier blocks create. If you restart the kernel, run the setup cells and then the blocks again from the top (every block of the first three parts takes seconds, so this is quick).
2. Do not install packages: PyTorch, NumPy and Matplotlib are already installed. Do not use `!pip install`, Google Drive or `google.colab` code in the Spark notebook. **Nothing is downloaded in this lab**: the real data of this week (a bell curve, eight Gaussian blobs, the five line shapes of Week 4) are generated by the helper functions below. The only exception is an optional experiment at the end of the lab (Part 7) that uses a pretrained Stable Diffusion model: it runs only if the instructor has put that model in the cache of the Spark, and it is not required.
3. A CPU is enough. Every block of Parts 1 to 3 finishes in a few seconds, because it only uses NumPy; the training runs of Parts 4 to 6 take up to about two minutes each. The GPU of the Spark is shared by the whole class and gives no advantage on networks and pictures this small, so leave it alone. The first helper cell prints which device it found.
4. Write your answers in markdown cells under the block where the exercise appears (Insert a cell, change its type to Markdown). Where an exercise says "store it in a variable", write the variable in a code cell: Part 7 collects your numbers in the dictionary `RESULTS`.
5. The same lab is available as a **Google Colab notebook**. Use the "Open in Colab" card at the top of this lab if you work from home or on a laptop: it has the same parts, the same numbering and the same expected results, plus automatic Check cells and a worked answer for every exercise. The code on this page is the complete working version, so on the Spark you can type or paste it directly. In the Colab notebook you write the key lines yourself first.
6. Paste these four blocks into four cells and run them. The first fixes the random seed (so your numbers match the ones printed in this lab), defines the empty `RESULTS` dictionary and the `check(...)` function. The second defines the helpers for the two data sets of the numeric parts: `sample_real_1d` (numbers from a bell curve), `make_mixture` (points from eight Gaussian blobs on a circle, the data of Part 4) and `modes_covered`, which counts how many blobs a set of generated points reaches. The third defines `show_grid` to look at small pictures. The fourth defines `make_shapes`, the five line shapes of Week 4 (the "real pictures" of the diffusion model in Part 6), the parameter counter and a trainer for a picture classifier (the judge of Part 6):

```python
import os, gc, math, random, re, time, warnings
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
def sample_real_1d(n, mu=4.0, sigma=1.0, seed=0):
    """n numbers from a normal (bell-curve) distribution with mean mu and standard deviation sigma, as a float32 array of shape (n, 1)."""
    return (mu + sigma * np.random.RandomState(seed).randn(n, 1)).astype(np.float32)

def make_mixture(n=2000, modes=8, radius=3.0, sigma=0.15, seed=0):
    """n points in the plane from `modes` equal Gaussian blobs (spread sigma) whose centres lie on a circle of the given radius.
    Returns X float32 (n, 2), the blob index of every point and the (modes, 2) array of centres."""
    r = np.random.RandomState(seed)
    ang = 2 * np.pi * np.arange(modes) / modes
    centres = np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1)
    k = r.randint(0, modes, n)
    return (centres[k] + sigma * r.randn(n, 2)).astype(np.float32), k, centres.astype(np.float32)

def modes_covered(points, centres, sigma=0.15, min_share=0.02, radius_sigmas=3.0):
    """How well do generated 2-D points cover a ring of blobs? Returns (covered, quality): `covered` = the number of blobs that receive at least
    min_share of all points within radius_sigmas * sigma of the centre; `quality` = the fraction of points that land within that distance of ANY centre."""
    points = np.asarray(points); d = np.linalg.norm(points[:, None, :] - centres[None], axis=2)
    near = d <= radius_sigmas * sigma
    covered = int(sum(near[:, j].mean() >= min_share for j in range(len(centres))))
    return covered, float(near.any(axis=1).mean())
```

```python
def show_grid(imgs, titles=None, cols=8, scale=1.0):
    """Show a grid of small pictures. imgs: array (N, 1, H, W) or (N, H, W) with values in [-1, 1] or [0, 1] (rescaled automatically)."""
    imgs = np.asarray(imgs); n = len(imgs); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols * scale, 1.35 * rows * scale))
    for a in np.ravel(axes): a.axis("off")
    lo, hi = (-1, 1) if imgs.min() < -0.01 else (0, 1)
    for i, a in enumerate(np.ravel(axes)[:n]):
        a.imshow(imgs[i].reshape(imgs[i].shape[-2:]), cmap="gray_r", vmin=lo, vmax=hi)
        if titles is not None: a.set_title(str(titles[i]), fontsize=7)
    plt.tight_layout(); plt.show()
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

   - `check(name, ok, hint)` prints a tick or a cross with a hint, so you can test your own answers;
   - `sample_real_1d(n, mu, sigma, seed)` returns `n` numbers from a bell curve N(mu, sigma squared) as an array of shape (n, 1);
   - `make_mixture(n, modes, radius, sigma, seed)` returns `n` points from `modes` Gaussian blobs whose centres lie on a circle, the index of each point's blob and the array of centres: the "real data" of Part 4, a data set with several **modes**;
   - `modes_covered(points, centres, sigma, min_share, radius_sigmas)` returns two numbers for a set of points: how many blobs receive at least 2 percent of the points (diversity) and what fraction of all points lie within 3 standard deviations of a blob centre (quality);
   - `show_grid(imgs, titles, cols, scale)` draws a grid of small pictures (values in [-1, 1] or [0, 1]);
   - `make_shapes(n, size, noise, shift, seed)` returns `n` greyscale pictures of five line shapes (`SHAPE_NAMES`) as an array of shape (n, 1, 16, 16), and their labels;
   - `count_params(model)` counts the trainable numbers of a model; `train_classifier(model, Xtr, ytr, Xte, yte, epochs, lr, opt, batch, seed, verbose)` trains any PyTorch classifier and returns its learning curves (`loss`, `train_acc`, `test_acc`);
   - `RESULTS` is a dictionary where you keep your key numbers; Part 7 prints it as your summary.

   Notice what is *not* in the helpers: the noise schedule, the denoising networks and the samplers. You write them yourself in this lab.

7. Run this smoke test in the next cell. It checks that the helpers work:

```python
Xs, ys = make_shapes(10, seed=SEED)
x1 = sample_real_1d(2000, seed=SEED)
Xm, km, cm_ = make_mixture(400, seed=SEED)
print("make_shapes:", Xs.shape, ys[:10], "  classes:", SHAPE_NAMES)
print("sample_real_1d:", x1.shape, x1.dtype, " mean", round(float(x1.mean()), 3), " std", round(float(x1.std()), 3))
print("make_mixture:", Xm.shape, " blobs:", len(cm_), " modes_covered on the real data:", modes_covered(Xm, cm_))
tiny = nn.Sequential(nn.Flatten(), nn.Linear(256, 5))
hist = train_classifier(tiny, Xs, ys, Xs, ys, epochs=5, lr=0.01, seed=SEED)
print("parameters of the tiny classifier:", count_params(tiny), " loss in epoch 1 and 5:", round(hist["loss"][0], 3), round(hist["loss"][-1], 3))
show_grid(Xs, [SHAPE_NAMES[k] for k in ys], cols=5)
check("helpers work", Xs.shape == (10, 1, 16, 16) and count_params(tiny) == 1285 and hist["loss"][-1] < hist["loss"][0] and modes_covered(Xm, cm_)[0] == 8, "Run the setup cells first.")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
make_shapes: (10, 1, 16, 16) [3 1 0 0 2 2 4 4 3 1]   classes: ['horizontal bar', 'vertical bar', 'cross', 'square', 'diagonal']
sample_real_1d: (2000, 1) float32  mean 4.045  std 0.988
make_mixture: (400, 2)  blobs: 8  modes_covered on the real data: (8, 0.9925)
parameters of the tiny classifier: 1285  loss in epoch 1 and 5: 1.712 0.662
✓ helpers work
```

   You should see `Running on: cpu` after the first cell. The smoke test then prints the shape of each data set, the first ten labels, the mean and standard deviation of 2,000 numbers from the bell curve (close to 4 and 1), the number of blobs reached by real mixture data (8), the size of a very small classifier and its loss going down, a figure with ten small shape pictures, and a tick.

If something goes wrong:

- `NameError: name 'xyz' is not defined`: you skipped a block or restarted the kernel. Run the setup cells and the earlier blocks again, in order.
- Plots do not appear: run `%matplotlib inline` once in a cell, or check that the block ends with `plt.show()`.
- Your numbers differ from the ones printed here: make sure the setup cell ran with `SEED = 42`, and that you did not run a block twice that uses a random generator without resetting it. The numbers of Parts 1 to 3 (schedules, values, counts, spreads) must match; the numbers of the training runs in the later parts can differ a little between library versions and hardware (training is not exactly deterministic across machines), but the qualitative results are stable.
- A cell seems frozen: stop it (the square button), restart the kernel and run again from the top. If a cell takes more than two or three minutes, you probably changed a loop count or a data size.

### Part 1. The forward process by hand (25 min)

*Lesson section: "Forward".* Everything in this week rests on one fixed recipe: **take a picture and destroy it slowly, in many tiny steps, by adding a little random noise each time.** This is the *forward process*. Think of a photograph on a window in heavy rain, or a barangay mural slowly washed by floods: at first you can still read every detail, after a while only the big shapes are left, and in the end there is only grey fog. Nothing in this recipe is learned. It is a formula, and because it is a formula, we can compute exactly what the picture looks like after any number of steps, and we know exactly which noise was added. That known noise is the answer key that the network of the next parts will be trained on. In this part you build the recipe by hand: the schedule that says how much noise each step adds, one step, the shortcut that jumps straight to step t, a real picture disappearing into fog, the signal-to-noise ratio that measures how much of the picture is left, and the proof by experiment that the end of the process is pure noise.

**Imagine first.** Picture a clean 16 by 16 picture of a cross. We add noise in 1,000 tiny steps, a little more each time. After 300 steps, what is left of the original? Before you run any code, write two predictions. (1) How much of the picture's *strength* is left after 300 of the 1,000 steps, as a percentage between 0 and 100 (100 means nothing has changed, 0 means nothing is left)? (2) At which step t do you think a person can no longer recognise the cross at all? You compare with Steps 1.1 and 1.4.

```python
# Imagine first: the fog
my_guess_left = None   # percent of the original picture's strength that is left after 300 of 1,000 noising steps (0 to 100)
my_guess_gone = None   # the step t (1 to 1000) at which you think a shape can no longer be recognised
```

1. **Step 1.1. The noise schedule.** The **noise schedule** is the list of numbers beta_1, beta_2, ..., beta_T that says how much noise step t adds. We use the **linear schedule** of the original paper: beta grows in a straight line from 0.0001 (a whisper) to 0.02 over T = 1,000 steps. From beta we compute alpha_t = 1 - beta_t (how much of the picture survives step t) and the running product **abar_t** (written ᾱ_t in the lesson), the product alpha_1 * alpha_2 * ... * alpha_t: how much of the original picture survives *all t steps together*. We store each list with index 0 meaning "no step yet" (beta_0 = 0, abar_0 = 1), so that `abar[300]` really is abar_300. The block builds the schedule, prints the numbers of the lesson and draws the four curves that matter: beta_t, abar_t, the weight of the picture sqrt(abar_t) and the weight of the noise sqrt(1 - abar_t).

```python
def make_schedule(T=1000, b0=1e-4, b1=0.02):
    """Linear noise schedule. Returns three arrays of length T + 1, indexed by the step t (index 0 means 'no noise yet'):
    beta[t], alpha[t] = 1 - beta[t] and abar[t] = alpha[1] * alpha[2] * ... * alpha[t]."""
    beta = np.concatenate(([0.0], np.linspace(b0, b1, T)))
    alpha = 1.0 - beta
    abar = np.cumprod(alpha)
    return beta, alpha, abar

beta, alpha, abar = make_schedule()
print("beta[1], beta[300], beta[1000] :", beta[1], round(beta[300], 5), beta[1000])
print(f"abar[300]  = {abar[300]:.4f}   sqrt(abar) = {np.sqrt(abar[300]):.3f}   sqrt(1 - abar) = {np.sqrt(1 - abar[300]):.3f}")
print(f"abar[1000] = {abar[1000]:.3e}")
assert abs(abar[300] - 0.3964) < 0.001 and abs(abar[1000] - 4.04e-5) < 0.1e-5

t = np.arange(1001)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(t, beta, color="C3"); ax[0].set_title("beta_t: noise added per step"); ax[0].set_xlabel("step t")
ax[1].plot(t, abar, label="abar_t"); ax[1].plot(t, np.sqrt(abar), label="sqrt(abar_t): picture weight")
ax[1].plot(t, np.sqrt(1 - abar), label="sqrt(1 - abar_t): noise weight"); ax[1].legend(); ax[1].set_xlabel("step t")
plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
beta[1], beta[300], beta[1000] : 0.0001 0.00606 0.02
abar[300]  = 0.3964   sqrt(abar) = 0.630   sqrt(1 - abar) = 0.777
abar[1000] = 4.036e-05
```

   You should see three printed lines and a figure with two panels. The first line shows that beta_1 is 0.0001, beta_300 is 0.00606 and beta_1000 is 0.02: each single step adds very little noise. The second line is the answer to the first Imagine question: abar_300 is 0.3964, so the weight of the clean picture in x_300 is sqrt(abar_300) = 0.630 and the weight of the noise is sqrt(1 - abar_300) = 0.777. In words: after 300 of the 1,000 steps, **63 percent of the picture's strength is left** (but only 39.6 percent of its power, which is the square of the strength), and the noise already has the bigger weight. At the end abar_1000 is 4.04e-5, which is 0.00004: nothing of the picture remains. In the figure, the left panel shows beta climbing in a straight line, and the right panel shows abar_t falling like a slide: slowly at first, quickest in the middle, flat near zero at the end. The picture weight sqrt(abar_t) and the noise weight sqrt(1 - abar_t) cross where abar = 0.5, at about step 260. The silent assertion checks the two numbers of the lesson (0.3964 and 4.04e-5).

   *Exercise 1.1:* a *faster* schedule: let beta grow from 0.0001 to 0.04 instead of 0.02 (call `make_schedule(1000, 1e-4, 0.04)`). Store the value of abar at t = 300 in `abar_fast_300`, and the first step at which abar is below 0.001 (the picture is gone for all practical purposes) in `t_gone_fast`. Why it matters: the schedule is a choice, and you can see right away how a more aggressive choice changes how many steps are really needed.

```python
abar_fast_300 = None   # abar at t = 300 for the faster schedule
t_gone_fast = None     # the first step t at which abar is below 0.001
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if abar_fast_300 is None or t_gone_fast is None:
    print("Fill in abar_fast_300 and t_gone_fast above, then run this cell again.")
else:
    _, _, ab2 = make_schedule(1000, 1e-4, 0.04)
    check("abar_fast_300 is right", abs(abar_fast_300 - ab2[300]) < 1e-4, "Call make_schedule(1000, 1e-4, 0.04) and read abar[300].")
    check("abar_fast_300 is smaller than for the slow schedule", abar_fast_300 < abar[300], "More noise per step leaves less of the picture.")
    check("t_gone_fast is right", t_gone_fast == int(np.argmax(ab2 < 0.001)), "Find the first index where abar is below 0.001 (np.argmax of a boolean array gives the first True).")
    RESULTS["p1_fast"] = (round(float(abar_fast_300), 4), int(t_gone_fast))
```

   A possible answer, and what it means:

```python
_, _, abar_fast = make_schedule(1000, 1e-4, 0.04)
abar_fast_300 = float(abar_fast[300])
t_gone_fast = int(np.argmax(abar_fast < 0.001))
print(abar_fast_300, t_gone_fast)
```

   With beta ending at 0.04, abar_300 drops to 0.1606 (against 0.3964), and the picture is gone (abar below 0.001) at step 584 instead of step 826. Doubling the final beta does not halve the time: the effect compounds, because abar is a product.

2. **Step 1.2. One forward step.** One step of the forward process is: x_t = sqrt(1 - beta_t) * x_(t-1) + sqrt(beta_t) * eps, where eps is a fresh random number from the standard bell curve N(0, 1). The first part *shrinks* the old picture a tiny bit; the second part *adds* a tiny bit of fresh noise. The two weights are chosen so that the total variance stays the same (we check this in the exercise). We use the example of the lesson, one pixel with the value 0.8, beta = 0.02 and a noise draw of eps = 0.5, and then apply five steps one after the other with a hand-written loop to a picture with four pixels, so that you see the pixels drift.

```python
def q_step(x, beta_t, eps):
    """One noising step: shrink the picture a little and add a little fresh noise."""
    return np.sqrt(1 - beta_t) * x + np.sqrt(beta_t) * eps

x1 = q_step(0.8, 0.02, 0.5)
print(f"signal part  sqrt(1 - 0.02) * 0.8 = {np.sqrt(0.98) * 0.8:.4f}")
print(f"noise part   sqrt(0.02) * 0.5     = {np.sqrt(0.02) * 0.5:.4f}")
print(f"new pixel                         = {x1:.4f}")
assert abs(x1 - 0.8627) < 1e-3

rng = np.random.RandomState(SEED)
x = np.array([0.8, -0.5, 0.0, 1.0])                 # a picture with four pixels
print("start         ", " ".join(f"{v:7.3f}" for v in x))
for k in range(1, 6):                               # five steps, written out by hand (beta = 0.02 each time)
    x = q_step(x, 0.02, rng.randn(len(x)))
    print(f"after step {k}  ", " ".join(f"{v:7.3f}" for v in x))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
signal part  sqrt(1 - 0.02) * 0.8 = 0.7920
noise part   sqrt(0.02) * 0.5     = 0.0707
new pixel                         = 0.8627
start            0.800  -0.500   0.000   1.000
after step 1     0.862  -0.515   0.092   1.205
after step 2     0.820  -0.542   0.314   1.302
after step 3     0.746  -0.460   0.245   1.223
after step 4     0.773  -0.726  -0.001   1.131
after step 5     0.622  -0.674  -0.129   0.920
```

   You should see the three pieces of the worked example and then five lines for the four-pixel picture. The signal part is sqrt(0.98) * 0.8 = 0.7920 (the pixel shrank from 0.8 to 0.792), the noise part is sqrt(0.02) * 0.5 = 0.0707, and the new pixel is **0.8627**, the number in the lesson. In the five-step table the four pixels start at 0.8, -0.5, 0.0 and 1.0 and wander away from these values in random directions: after five steps the first pixel has moved from 0.8 to 0.622, the third from 0.0 to -0.129. After only five steps the picture is still easy to recognise. Each step is a small push, and 1,000 small pushes add up. The assertion checks 0.8627 to three decimals.

   *Exercise 1.2:* check the claim that a forward step *preserves the variance*. Take 100,000 pixels that all start at exactly 1.0 (so they have no spread at all), apply one step with beta = 0.02 and store the variance of the result in `var_const`. Then take 100,000 pixels drawn from N(0, 1) (variance 1), apply one step and store the variance of the result in `var_normal`. Use a `RandomState(SEED)` for the noise. Why it matters: a process that kept growing or shrinking in size would be impossible to train a network on; because the variance stays at 1, the picture and the noise always live on the same scale.

```python
var_const = None    # variance after one step (beta = 0.02) of 100,000 pixels that all start at exactly 1.0
var_normal = None   # variance after one step (beta = 0.02) of 100,000 pixels that start as N(0, 1) numbers
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if var_const is None or var_normal is None:
    print("Fill in var_const and var_normal above, then run this cell again.")
else:
    check("var_const is close to beta = 0.02", abs(var_const - 0.02) < 0.003, "All pixels start equal, so the only spread is the new noise: sqrt(beta) * eps has variance beta.")
    check("var_normal is close to 1", abs(var_normal - 1.0) < 0.03, "(1 - beta) * 1 + beta * 1 = 1: the step keeps the variance.")
    RESULTS["p1_var"] = (round(float(var_const), 4), round(float(var_normal), 3))
```

   A possible answer, and what it means:

```python
rng = np.random.RandomState(SEED)
n = 100000
var_const = float(q_step(np.ones(n), 0.02, rng.randn(n)).var())
var_normal = float(q_step(rng.randn(n), 0.02, rng.randn(n)).var())
print(var_const, var_normal)
```

   `var_const` is 0.0200, almost exactly beta = 0.02: all the spread comes from the new noise, with weight sqrt(0.02), whose variance is 0.02. `var_normal` is 0.997, close to 1: (1 - beta) * 1 + beta * 1 = 1. The small differences are sampling noise from using 100,000 numbers instead of infinitely many. This is why the process is called *variance preserving*: if the picture has variance 1 and the noise has variance 1, so does the result.

3. **Step 1.3. The closed form: jump to step t.** Running 1,000 steps one after the other is slow, and there is a shortcut. A pixel after t steps is a sum of the original pixel (shrunk many times) and many independent bell-curve noises (each shrunk by the later steps). **A sum of independent bell curves is again a bell curve, with the variances added**, so all those noises together are just *one* bell-curve noise. The result is the **closed form**: x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps, with a single eps from N(0, 1). It lets us jump from the clean picture to *any* step t in one move, which is what makes training fast. The block checks this claim by experiment. It starts 10,000 copies of one pixel (x_0 = 0.8) and (a) runs 300 single steps with fresh noise each time, and (b) makes the same number of samples with one jump to t = 300. The two groups do not use the same random numbers, so we cannot compare them number by number: we compare their mean and their variance, and the theory (mean sqrt(abar_300) * 0.8, variance 1 - abar_300).

```python
def q_sample(x0, t, eps, abar):
    """Jump straight from the clean picture x0 to step t: sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps."""
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps

rng = np.random.RandomState(SEED)
N = 10000
x = np.full(N, 0.8)                                  # 10,000 copies of one pixel with value 0.8
for k in range(1, 301):                              # (a) 300 single steps, fresh noise each time
    x = q_step(x, beta[k], rng.randn(N))
xb = q_sample(np.full(N, 0.8), 300, rng.randn(N), abar)   # (b) one jump with the closed form
th_mean, th_var = np.sqrt(abar[300]) * 0.8, 1 - abar[300]
print(f"(a) 300 steps  : mean {x.mean():.3f}  variance {x.var():.3f}")
print(f"(b) closed form: mean {xb.mean():.3f}  variance {xb.var():.3f}")
print(f"theory         : mean {th_mean:.3f}  variance {th_var:.3f}")
assert abs(x.mean() - xb.mean()) < 0.02 and abs(x.var() - xb.var()) < 0.02
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
(a) 300 steps  : mean 0.500  variance 0.604
(b) closed form: mean 0.505  variance 0.601
theory         : mean 0.504  variance 0.604
```

   You should see three lines. With 300 single steps the 10,000 samples have a mean of 0.500 and a variance of 0.604; with the closed form they have a mean of 0.505 and a variance of 0.601; the theory says 0.504 and 0.604. The three agree within the sampling error (the assertion allows 0.02 in the mean and in the variance). So 300 little pushes and one big push give the same bell curve: **the closed form is exact, not an approximation.** Notice also what the numbers mean: a pixel that started at 0.8 is now a bell curve centred at 0.504, with a standard deviation of sqrt(0.604) = 0.777. Whatever the original value, the noise has a spread of 0.777 around it, bigger than the shift of the mean.

   *Exercise 1.3:* predict first, then measure. For a pixel x_0 = 0.8 at step t = 600, write the mean and the variance that the closed form predicts in `theory_mean_600` and `theory_var_600`. Then make 10,000 samples with `q_sample` at t = 600 (use `rng = np.random.RandomState(SEED)`) and store their mean and variance in `emp_mean_600` and `emp_var_600`. Why it matters: the closed form is the line you will use inside the training loop in Part 4, so you should trust it and know what it predicts.

```python
theory_mean_600 = None   # the mean of x_600 for a pixel x0 = 0.8, from the formula
theory_var_600 = None    # the variance of x_600, from the formula
emp_mean_600 = None      # the mean of 10,000 samples made with q_sample at t = 600
emp_var_600 = None       # their variance
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if None in (theory_mean_600, theory_var_600, emp_mean_600, emp_var_600):
    print("Fill in the four variables above, then run this cell again.")
else:
    check("theory mean is right", abs(theory_mean_600 - np.sqrt(abar[600]) * 0.8) < 1e-3, "mean = sqrt(abar_t) * x0.")
    check("theory variance is right", abs(theory_var_600 - (1 - abar[600])) < 1e-3, "variance = 1 - abar_t, because eps has variance 1.")
    check("the samples agree with the theory (0.03)", abs(emp_mean_600 - theory_mean_600) < 0.03 and abs(emp_var_600 - theory_var_600) < 0.03, "Use q_sample(np.full(10000, 0.8), 600, rng.randn(10000), abar).")
    RESULTS["p1_closed"] = (round(float(theory_mean_600), 3), round(float(theory_var_600), 3))
```

   A possible answer, and what it means:

```python
theory_mean_600 = float(np.sqrt(abar[600]) * 0.8)
theory_var_600 = float(1 - abar[600])
rng = np.random.RandomState(SEED)
s = q_sample(np.full(10000, 0.8), 600, rng.randn(10000), abar)
emp_mean_600, emp_var_600 = float(s.mean()), float(s.var())
print(theory_mean_600, theory_var_600, emp_mean_600, emp_var_600)
```

   The theory gives a mean of 0.129 and a variance of 0.974; the samples give 0.127 and 0.981. At t = 600 the picture weight is only sqrt(abar_600) = 0.161, so the pixel value 0.8 has almost disappeared: the mean has shrunk from 0.8 to 0.129, and the variance is very close to 1.

4. **Step 1.4. Noising a real picture.** Now we apply the closed form to a real picture. We take the cross from `make_shapes` without any noise (`noise=0.0`), scale it from the range [0, 1] to [-1, 1] (the diffusion models of this week work with pictures between -1 and 1, so that black is -1 and white is +1, and the noise, which is centred at 0, sits in the middle of the range), and draw **one** noise picture eps. Using the *same* eps for every t makes the pictures a film of one picture disappearing: only the amount of noise changes. We show t = 0, 50, 100, 200, 300, 500, 800 and 1000, print abar_t for each, and measure with the **correlation** between x_t and x_0 how much of the shape is still there (1.0 means the same picture, 0 means unrelated).

```python
Xc, yc = make_shapes(5, noise=0.0, seed=SEED)            # five clean shapes, one of each kind
x0 = 2 * Xc[int(np.where(yc == 2)[0][0])] - 1             # the cross, scaled from [0, 1] to [-1, 1]; shape (1, 16, 16)
rng = np.random.RandomState(SEED)
eps = rng.randn(*x0.shape)                                # ONE noise picture, reused for every t
ts = [0, 50, 100, 200, 300, 500, 800, 1000]
imgs = np.array([q_sample(x0, t, eps, abar) for t in ts])
show_grid(imgs, [f"t={t}" for t in ts], cols=8)
print("  t    abar_t   correlation(x_t, x_0)")
for t_, im in zip(ts, imgs):
    print(f"{t_:5d}  {abar[t_]:.4f}   {np.corrcoef(im.ravel(), x0.ravel())[0, 1]:.3f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
  t    abar_t   correlation(x_t, x_0)
    0  1.0000   1.000
   50  0.9710   0.946
  100  0.8970   0.828
  200  0.6590   0.562
  300  0.3964   0.357
  500  0.0786   0.116
  800  0.0015   -0.012
 1000  0.0000   -0.029
```

   You should see a row of eight small pictures and a table. At t = 0 the cross is clean. At t = 50 (abar = 0.971) it is still sharp, with a faint speckle; at t = 100 (abar = 0.897) the speckle is clearly visible, but the cross is easy to see; at t = 200 (abar = 0.659) the cross is still recognisable, with a lot of grain; at t = 300 (abar = 0.396) you have to look for it; at t = 500 (abar = 0.079) you can at best guess the pattern; at t = 800 and t = 1000 there is only static. The correlation says the same in numbers: 1.000, 0.946, 0.828, 0.562, 0.357, 0.116, -0.012 and -0.029. So for this picture and this noise draw, a human eye gives up somewhere between t = 300 and t = 500, while the last 500 steps add nothing that a person could see. (Your eye is a detective, as in Week 12; a network will do better than you here, because it can use the tiny remaining correlation.) Answer to the second Imagine question: look at your own guess, and compare it with the step where the cross disappears in your figure.

   *Exercise 1.4:* find the first step t at which the correlation between x_t and x_0 (using the same `eps`) falls below 0.5 and store it in `t_corr_half`; then the first step at which it falls below 0.1, in `t_corr_tenth`. Loop over t = 0, 1, 2, ... 1000 and call `q_sample(x0, t, eps, abar)`. Why it matters: it turns "at which step is the shape gone?" from a matter of eyesight into a number.

```python
t_corr_half = None   # the first step t at which the correlation between x_t and x_0 falls below 0.5
t_corr_tenth = None  # the first step t at which it falls below 0.1
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if t_corr_half is None or t_corr_tenth is None:
    print("Fill in t_corr_half and t_corr_tenth above, then run this cell again.")
else:
    cs = [np.corrcoef(q_sample(x0, t, eps, abar).ravel(), x0.ravel())[0, 1] for t in range(1001)]
    check("t_corr_half is right", t_corr_half == next(t for t in range(1001) if cs[t] < 0.5), "Loop over t = 0, 1, 2, ... and stop at the first correlation below 0.5.")
    check("t_corr_tenth is right", t_corr_tenth == next(t for t in range(1001) if cs[t] < 0.1), "Same loop, threshold 0.1.")
    check("the order is right", t_corr_half < t_corr_tenth, "The picture fades: the correlation only goes down.")
    RESULTS["p1_fade"] = (int(t_corr_half), int(t_corr_tenth))
```

   A possible answer, and what it means:

```python
corr = [np.corrcoef(q_sample(x0, t, eps, abar).ravel(), x0.ravel())[0, 1] for t in range(1001)]
t_corr_half = next(t for t in range(1001) if corr[t] < 0.5)
t_corr_tenth = next(t for t in range(1001) if corr[t] < 0.1)
print(t_corr_half, t_corr_tenth)
```

   The correlation falls below 0.5 at t = 227 and below 0.1 at t = 522. So **about a fifth of the way through the process, half of the shape is gone, and at about half-way only a tenth is left**. Look at what this implies for training: the interesting part, where the picture changes from clear to gone, happens in the first 500 steps of 1,000.

5. **Step 1.5. Signal-to-noise ratio.** The **signal-to-noise ratio** (SNR) is the one number that says how hard denoising is at step t: SNR_t = abar_t / (1 - abar_t), the power of the picture divided by the power of the noise. If the SNR is 100 the picture is 100 times stronger than the noise (an easy case); if it is 1 they are equally strong; if it is 0.01 the noise is 100 times stronger (hopeless to the eye). Use it to compare different schedules fairly. The block prints the SNR at t = 1, 50, 100, 300, 500 and 1000, finds the step where the SNR crosses 1 (abar = 0.5), and plots the SNR on a *log axis*, because it falls over many orders of magnitude.

```python
def snr(t, abar):
    """Signal-to-noise ratio at step t: how much picture power for one unit of noise power."""
    return abar[t] / (1 - abar[t])

print("  t    abar_t    SNR_t")
for t_ in [1, 50, 100, 300, 500, 1000]:
    print(f"{t_:5d}  {abar[t_]:.5f}  {snr(t_, abar):12.4g}")
t_snr1 = int(np.argmax(abar <= 0.5))                    # the first step with abar <= 0.5, i.e. SNR <= 1
print("first t with SNR at most 1:", t_snr1, "  abar there:", round(abar[t_snr1], 4), "  SNR:", round(snr(t_snr1, abar), 4))
assert abar[t_snr1 - 1] > 0.5 >= abar[t_snr1]

tt = np.arange(1, 1001)
plt.figure(figsize=(5, 3)); plt.semilogy(tt, snr(tt, abar)); plt.axhline(1, color="grey", ls="--")
plt.xlabel("step t"); plt.ylabel("SNR (log axis)"); plt.title("signal-to-noise ratio, linear schedule"); plt.tight_layout(); plt.show()
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
  t    abar_t    SNR_t
    1  0.99990          9999
   50  0.97102          33.5
  100  0.89702          8.71
  300  0.39642        0.6568
  500  0.07859       0.08529
 1000  0.00004     4.036e-05
first t with SNR at most 1: 260   abar there: 0.4976   SNR: 0.9905
```

   You should see a table, one line about the crossing point and a falling curve. The SNR is 9,999 at t = 1 (almost no noise), 33.5 at t = 50, 8.71 at t = 100, **0.657 at t = 300** (the number of the lesson), 0.0853 at t = 500 and about 0.00004 at t = 1000. The curve on the log axis keeps sloping down without ever levelling off: the SNR falls by a large factor over the whole run, from thousands to almost nothing. The SNR first drops to 1 or below at **t = 260**, where abar = 0.4976: before that step the picture is stronger than the noise, after it the noise is stronger. That is only a quarter of the way through the process. The assertion confirms that abar is above 0.5 at step 259 and at most 0.5 at step 260.

   *Exercise 1.5:* find two more landmarks: the first step at which the SNR falls below 100 (store it in `t_snr_100`) and the first step at which it falls below 0.01 (store it in `t_snr_001`). Use `snr(np.arange(1, 1001), abar)` and remember that step numbers start at 1, not 0. Why it matters: the range of SNR that a network must handle, from hundreds down to a hundredth, is huge; this is one reason the network must be told t.

```python
t_snr_100 = None    # the first step t at which the SNR falls below 100 (the picture is still 100 times stronger than the noise)
t_snr_001 = None    # the first step t at which the SNR falls below 0.01 (the noise is 100 times stronger)
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if t_snr_100 is None or t_snr_001 is None:
    print("Fill in t_snr_100 and t_snr_001 above, then run this cell again.")
else:
    s_ = snr(np.arange(1, 1001), abar)
    check("t_snr_100 is right", t_snr_100 == int(np.argmax(s_ < 100)) + 1, "Compute snr for t = 1 .. 1000 and find the first value below 100 (remember that t starts at 1).")
    check("t_snr_001 is right", t_snr_001 == int(np.argmax(s_ < 0.01)) + 1, "Same, with the threshold 0.01.")
    RESULTS["p1_snr"] = (int(t_snr_100), int(t_snr_001))
```

   A possible answer, and what it means:

```python
s_ = snr(np.arange(1, 1001), abar)
t_snr_100 = int(np.argmax(s_ < 100)) + 1
t_snr_001 = int(np.argmax(s_ < 0.01)) + 1
print(t_snr_100, t_snr_001)
```

   The SNR falls below 100 at t = 28 already (almost immediately: a little noise is enough to make the picture only 100 times stronger) and below 0.01 at t = 675. So the SNR spans the range from 100 to 0.01 in about 650 steps, a factor of 10,000. The network has to denoise at all these levels with the same weights.

6. **Step 1.6. The end is pure noise.** The whole method depends on the end of the forward process being **the same for every picture**: pure N(0, 1) noise. If that is true we can start the reverse process from random numbers that we draw ourselves, without having any picture. The block checks it for the five different shapes (one of each kind): for each, it makes 50 noise draws at t = 1000 and prints the mean and the variance of all the pixel values (they should be 0 and 1). The assertion allows an error of 0.1.

```python
rng = np.random.RandomState(SEED)
print("shape              mean of x_1000   variance of x_1000")
for k in range(5):
    pic = 2 * Xc[int(np.where(yc == k)[0][0])] - 1
    xs = np.stack([q_sample(pic, 1000, rng.randn(*pic.shape), abar) for _ in range(50)])      # 50 noise draws
    print(f"{SHAPE_NAMES[k]:16s}  {xs.mean():8.3f}        {xs.var():8.3f}")
    assert abs(xs.mean()) < 0.1 and abs(xs.var() - 1) < 0.1
print("weight of the picture at t = 1000:", round(float(np.sqrt(abar[1000])), 4), "  weight of the noise:", round(float(np.sqrt(1 - abar[1000])), 4))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
shape              mean of x_1000   variance of x_1000
horizontal bar      -0.010           0.999
vertical bar        -0.000           0.993
cross               -0.022           1.008
square               0.008           1.000
diagonal             0.003           1.021
weight of the picture at t = 1000: 0.0064   weight of the noise: 1.0
```

   You should see five lines and a last line with the two weights. For all five shapes the mean of x_1000 is within 0.022 of 0 (the values are -0.010, -0.000, -0.022, 0.008 and 0.003) and the variance is between 0.993 and 1.021: very close to 1. The reason is in the last line: at t = 1000 the weight of the picture is only sqrt(4.04e-5) = 0.0064, so even a picture with pixel values of -1 or +1 shifts the mean by at most 0.0064, and the noise has the weight 1.0000. **Whatever you put in, a standard bell curve comes out.** That is the starting point of every sampler in the rest of the lab.

   *Exercise 1.6:* measure the "leak" of the picture into the final step. Store sqrt(abar_1000) in `leak_1000` and sqrt(abar_300) in `leak_300`. Then make 50 all-white pictures of shape 16 by 16 (every pixel +1), noise them to t = 300 with `q_sample` and store the mean of the result in `white_mean_300`. Why it matters: it shows that the leak at t = 300 is large (the picture is still there) and at t = 1000 negligible.

```python
leak_1000 = None   # the weight of the clean picture in x_1000
leak_300 = None    # the weight of the clean picture in x_300
white_mean_300 = None   # the mean of x_300 for a picture that is all white (every pixel +1), averaged over many noise draws
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if None in (leak_1000, leak_300, white_mean_300):
    print("Fill in leak_1000, leak_300 and white_mean_300 above, then run this cell again.")
else:
    check("leak_1000 is right", abs(leak_1000 - np.sqrt(abar[1000])) < 1e-4, "The weight of the picture is sqrt(abar_t).")
    check("leak_300 is right", abs(leak_300 - np.sqrt(abar[300])) < 1e-3, "sqrt(abar_300).")
    check("white_mean_300 is right (0.03)", abs(white_mean_300 - leak_300) < 0.03, "For a constant picture of +1 the mean of x_t is sqrt(abar_t) * 1.")
    RESULTS["p1_leak"] = (round(float(leak_1000), 4), round(float(leak_300), 3))
```

   A possible answer, and what it means:

```python
leak_1000 = float(np.sqrt(abar[1000]))
leak_300 = float(np.sqrt(abar[300]))
rng = np.random.RandomState(SEED)
white = np.ones((50, 16, 16))
white_mean_300 = float(q_sample(white, 300, rng.randn(*white.shape), abar).mean())
print(leak_1000, leak_300, white_mean_300)
```

   `leak_1000` is 0.0064 and `leak_300` is 0.630; the all-white picture, which has the mean +1, has a mean of 0.627 at t = 300, which is the same as sqrt(abar_300) (it is the weight of the picture, multiplied by 1). At t = 1000 the same white picture would have a mean of 0.006, which is nothing.

7. **Step 1.7. A recipe, not a learner.** In Week 12 the generator was a network with weights that were learned in a game against a detective. The forward process of this week is the opposite: **it has no weights at all**. It is a recipe, written in one line, and it needs no training. The block makes two noisy versions of the same clean cross at the same step t = 300, with two different noise draws, and measures how similar they are to the clean picture and to each other. It also prints the number of learned parameters in the forward process (zero).

```python
rng = np.random.RandomState(SEED)
xa = q_sample(x0, 300, rng.randn(*x0.shape), abar)       # two different noise draws, same clean picture, same t
xb = q_sample(x0, 300, rng.randn(*x0.shape), abar)
c = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
print(f"correlation of x_300 (draw 1) with x_0 : {c(xa, x0):.3f}")
print(f"correlation of x_300 (draw 2) with x_0 : {c(xb, x0):.3f}")
print(f"correlation of the two draws           : {c(xa, xb):.3f}")
print("numbers learned by the forward process :", 0)
show_grid(np.array([x0, xa, xb]), ["clean", "draw 1", "draw 2"], cols=3)
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
correlation of x_300 (draw 1) with x_0 : 0.357
correlation of x_300 (draw 2) with x_0 : 0.404
correlation of the two draws           : 0.134
numbers learned by the forward process : 0
```

   You should see three correlations, one line with a zero and three small pictures. Each noisy version is moderately similar to the clean picture (0.357 and 0.404), but the two noisy versions are much *less* similar to each other (0.134), because they share the picture but not the noise. The forward process has **0** learned numbers. Two noisy pictures of the same cross look different, and each of them is a perfectly good training example, with a different noise that we know.

   *Exercise 1.7:* write three to five sentences between the quotation marks in `answer_1_7`: how does the forward process differ from the generator of Week 12, and why is a fixed recipe useful for training? Use the words *learned*, *noise* and *known* if they help. Why it matters: this is the main difference between GANs and diffusion models, and you will meet it again in the reflection questions of Part 7.

```python
answer_1_7 = ""   # write 3 to 5 sentences between the quotation marks
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if len(answer_1_7.split()) < 25:
    print("Write at least 25 words in answer_1_7 above, then run this cell again.")
else:
    check("you wrote an answer of 25 words or more", True)
    RESULTS["p1_answer"] = answer_1_7[:200]
```

   A possible answer, and what it means:

```python
answer_1_7 = ("In Week 12 the generator had weights that were learned in a game, and nobody could say in advance what it would draw. "
              "The forward process has no weights: it is a fixed formula, so for every clean picture and every step t we can make a noisy picture "
              "AND we know exactly which noise we added. That known noise is a free, perfect answer key, which lets us train a network with an ordinary "
              "loss, without any opponent. The end of the recipe, pure N(0, 1) noise, is also a starting point that we can draw whenever we like.")
```

   The sample answer makes three points: (1) the forward process has nothing learned in it, (2) we know exactly which noise we added, so we have a free and perfect answer key, and (3) the end is pure noise that we can draw ourselves. Compare with your own answer. A strong answer mentions that the training of a diffusion model is an ordinary regression (predict a known noise with a mean squared error), without the unstable two-player game of the GAN.

### Part 2. Noise schedules (20 min)

*Lesson section: "Forward".* The linear schedule of Part 1 is not the only way to destroy a picture, and it is not even the best one. How fast the SNR falls decides which steps of the process are easy for the network, which are hard, and which are wasted. In this part you build the **cosine schedule** that is widely used today, compare it with the linear one on curves, on pictures and in numbers, count the steps that a linear schedule spends on pictures that are already pure noise, choose the number of steps T for the little experiments of the following parts, and finally design a schedule of your own.

**Imagine first.** Suppose that you destroy the picture *more slowly*, so that fewer steps are spent on pure noise and every step removes about the same amount of information. Does that give the denoising network an easier or a harder job? Before you run any code, write two predictions. (1) The word "easier" or "harder". (2) How many of the 1,000 steps of the linear schedule have abar below 0.001 (the picture is already gone)? You compare with Steps 2.2 and 2.3.

```python
# Imagine first: a slower schedule
my_guess_easier = None    # a schedule that destroys the picture more slowly: is the network's job "easier" or "harder"? (a string)
my_guess_wasted = None    # of the 1,000 steps of the linear schedule, how many have abar below 0.001 (the picture is already gone)? (a number)
```

1. **Step 2.1. The cosine schedule.** The cosine schedule, proposed by Nichol and Dhariwal in 2021, does not choose beta_t directly: it chooses the *curve* abar_t, namely f(t) / f(0) with f(t) = cos^2(((t / T + s) / (1 + s)) * pi / 2), and computes beta_t = 1 - abar_t / abar_(t-1) from it (clipped at 0.999 so that the last steps stay well defined). The small offset s = 0.008 keeps beta from being almost 0 at the start. The block builds it in the same format as `make_schedule`, so that all functions of Part 1 work with it, and prints abar at t = 300 and 500 next to the linear values.

```python
def cosine_schedule(T=1000, s=0.008):
    """Cosine schedule (Nichol and Dhariwal, 2021). Same output format as make_schedule: beta, alpha, abar of length T + 1."""
    tt = np.arange(T + 1)
    f = np.cos(((tt / T + s) / (1 + s)) * np.pi / 2) ** 2          # the smooth target shape
    abar_f = f / f[0]                                              # abar at t = 0 must be 1
    beta = np.zeros(T + 1)
    beta[1:] = np.minimum(1 - abar_f[1:] / abar_f[:-1], 0.999)     # the betas that give that curve; never above 0.999
    alpha = 1 - beta
    return beta, alpha, np.cumprod(alpha)

beta_c, alpha_c, abar_c = cosine_schedule()
print(f"cosine: abar[300] = {abar_c[300]:.3f}   abar[500] = {abar_c[500]:.3f}   abar[1000] = {abar_c[1000]:.2e}")
print(f"cosine: beta[1] = {beta_c[1]:.5f}   beta[500] = {beta_c[500]:.4f}   beta[999] = {beta_c[999]:.3f}   beta[1000] = {beta_c[1000]:.3f}")
print(f"linear: abar[300] = {abar[300]:.3f}   abar[500] = {abar[500]:.3f}")
assert abs(abar_c[300] - 0.787) < 0.005 and abs(abar_c[500] - 0.494) < 0.005
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
cosine: abar[300] = 0.787   abar[500] = 0.494   abar[1000] = 2.43e-09
cosine: beta[1] = 0.00004   beta[500] = 0.0031   beta[999] = 0.750   beta[1000] = 0.999
linear: abar[300] = 0.396   abar[500] = 0.079
```

   You should see three lines. The cosine schedule has abar_300 = 0.787 and abar_500 = 0.494 (the values of the lesson), while the linear one has abar_300 = 0.396 and abar_500 = 0.079. At t = 500 the linear schedule has already removed 92 percent of the picture's power; the cosine schedule has removed only about half. The second line shows how the betas look: beta_1 is 0.00004 (a tiny first step), beta_500 is 0.0031, and the last betas become very large (0.750 at t = 999, 0.999 at t = 1000, the clip): the cosine schedule saves its large steps for the end, when the picture is already almost gone. At the end abar_1000 is 2.4e-9, essentially zero. The assertion checks the two values to 0.005.

   *Exercise 2.1:* compute the SNR at t = 500 for both schedules with the function `snr(t, abar)` from Part 1 (the cosine schedule's array is `abar_c`) and store them in `snr_lin_500` and `snr_cos_500`. Why it matters: the SNR is the fair way to compare schedules, and these two numbers show how differently they behave half-way.

```python
snr_lin_500 = None   # SNR at t = 500, linear schedule
snr_cos_500 = None   # SNR at t = 500, cosine schedule
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if snr_lin_500 is None or snr_cos_500 is None:
    print("Fill in snr_lin_500 and snr_cos_500 above, then run this cell again.")
else:
    check("snr_lin_500 is right", abs(snr_lin_500 - snr(500, abar)) < 0.01, "snr(500, abar).")
    check("snr_cos_500 is right", abs(snr_cos_500 - snr(500, abar_c)) < 0.01, "snr(500, abar_c).")
    check("the cosine schedule keeps more signal at t = 500", snr_cos_500 > snr_lin_500, "Look at the numbers again.")
    RESULTS["p2_snr500"] = (round(float(snr_lin_500), 3), round(float(snr_cos_500), 3))
```

   A possible answer, and what it means:

```python
snr_lin_500 = float(snr(500, abar))
snr_cos_500 = float(snr(500, abar_c))
print(snr_lin_500, snr_cos_500)
```

   The linear schedule has an SNR of 0.0853 at t = 500 (the noise is 12 times stronger than the picture), the cosine schedule an SNR of 0.976 (picture and noise are almost equally strong). The difference is a factor of about 11.

2. **Step 2.2. Compare the two schedules.** Now we compare the schedules in three ways. First, the two curves abar_t and the two SNR curves (log axis). Second, the share of the steps for which the SNR is above 1 (the picture is stronger than the noise). Third, the same cross as in Part 1, with the same noise draw, at t = 200, 400, 600 and 800 with each schedule, in two rows, together with the correlation between x_t and x_0.

```python
t = np.arange(1001)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(t, abar, label="linear"); ax[0].plot(t, abar_c, label="cosine"); ax[0].set_title("abar_t"); ax[0].legend(); ax[0].set_xlabel("step t")
ax[1].semilogy(t[1:-1], snr(t[1:-1], abar), label="linear"); ax[1].semilogy(t[1:-1], snr(t[1:-1], abar_c), label="cosine")
ax[1].axhline(1, color="grey", ls="--"); ax[1].set_title("SNR_t (log axis)"); ax[1].legend(); ax[1].set_xlabel("step t")
plt.tight_layout(); plt.show()

frac_lin = float((snr(t[1:-1], abar) > 1).mean()); frac_cos = float((snr(t[1:-1], abar_c) > 1).mean())
print(f"share of steps with SNR above 1:  linear {frac_lin:.3f}   cosine {frac_cos:.3f}")

ts2 = [200, 400, 600, 800]
row_lin = [q_sample(x0, t_, eps, abar) for t_ in ts2]
row_cos = [q_sample(x0, t_, eps, abar_c) for t_ in ts2]
show_grid(np.array(row_lin + row_cos), [f"lin t={t_}" for t_ in ts2] + [f"cos t={t_}" for t_ in ts2], cols=4)
for name, ab in [("linear", abar), ("cosine", abar_c)]:
    print(name, "correlation with x_0 at t = 200, 400, 600, 800:", [round(float(np.corrcoef(q_sample(x0, t_, eps, ab).ravel(), x0.ravel())[0, 1]), 2) for t_ in ts2])
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
share of steps with SNR above 1:  linear 0.259   cosine 0.496
linear correlation with x_0 at t = 200, 400, 600, 800: [0.56, 0.21, 0.05, -0.01]
cosine correlation with x_0 at t = 200, 400, 600, 800: [0.83, 0.55, 0.32, 0.13]
```

   You should see two panels of curves, a line with two shares, a figure with two rows of four pictures and two lines of correlations. In the left panel the linear curve of abar_t falls faster and hits the floor early, the cosine curve stays high for longer, bends and falls steeply at the end. On the SNR panel the linear curve is a descending line that crosses the dashed line SNR = 1 early, at about step 260, and the cosine curve crosses it half-way, at about step 500. The shares say the same: **26 percent of the linear steps and 50 percent of the cosine steps have an SNR above 1.** In the pictures, the top row (linear) shows the cross at t = 200 with grain, at t = 400 barely, and then only static; the bottom row (cosine) still shows the cross at t = 200 and 400, and traces of it at 600. The correlations with x_0 are 0.56, 0.21, 0.05 and -0.01 for the linear schedule and 0.83, 0.55, 0.32 and 0.13 for the cosine schedule. The cosine schedule keeps the picture alive for much longer.

   *Exercise 2.2:* find the first step at which abar falls below 0.1 (90 percent of the power gone) for the linear schedule (`t_lin_01`, from `abar`) and for the cosine schedule (`t_cos_01`, from `abar_c`). Why it matters: it tells you how many of the T steps carry real information about the picture.

```python
t_lin_01 = None   # the first step t at which abar falls below 0.1 for the linear schedule
t_cos_01 = None   # the same for the cosine schedule
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if t_lin_01 is None or t_cos_01 is None:
    print("Fill in t_lin_01 and t_cos_01 above, then run this cell again.")
else:
    check("t_lin_01 is right", t_lin_01 == int(np.argmax(abar < 0.1)), "np.argmax(abar < 0.1) gives the first index where the test is True.")
    check("t_cos_01 is right", t_cos_01 == int(np.argmax(abar_c < 0.1)), "Same with abar_c.")
    RESULTS["p2_t01"] = (int(t_lin_01), int(t_cos_01))
```

   A possible answer, and what it means:

```python
t_lin_01 = int(np.argmax(abar < 0.1))
t_cos_01 = int(np.argmax(abar_c < 0.1))
print(t_lin_01, t_cos_01)
```

   The linear schedule reaches abar below 0.1 at t = 476, the cosine schedule only at t = 794. With the linear schedule, the last half of the steps (from 476 to 1,000) deals with pictures that have lost more than 90 percent of their power; with the cosine schedule that is only the last fifth.

3. **Step 2.3. Wasted steps.** When we train, we pick a random step t for every training picture, uniformly between 1 and T. If many steps are at pure noise, then many training examples are pure noise, where the network can do nothing clever (the best prediction of the noise is the input itself) and therefore learns little. The block counts, for four thresholds, how many of the 1,000 steps have an abar below the threshold, for both schedules, and the share of training examples that are already pure noise at the strictest of the thresholds that is used in this lab (abar below 0.001).

```python
print("threshold      linear: steps below   cosine: steps below   (out of 1,000)")
for thr in [0.1, 0.01, 0.001, 0.0001]:
    nl = int((abar[1:] < thr).sum()); nc = int((abar_c[1:] < thr).sum())
    print(f"abar < {thr:<8}   {nl:5d}  ({nl / 10:4.1f}%)          {nc:5d}  ({nc / 10:4.1f}%)")
wasted_lin = int((abar[1:] < 0.001).sum()); wasted_cos = int((abar_c[1:] < 0.001).sum())
print("\nIf t is drawn uniformly, the share of training examples that are already pure noise (abar < 0.001):")
print(f"  linear {wasted_lin / 1000:.1%}   cosine {wasted_cos / 1000:.1%}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
threshold      linear: steps below   cosine: steps below   (out of 1,000)
abar < 0.1          525  (52.5%)            207  (20.7%)
abar < 0.01         327  (32.7%)             65  ( 6.5%)
abar < 0.001        175  (17.5%)             21  ( 2.1%)
abar < 0.0001        46  ( 4.6%)              7  ( 0.7%)

If t is drawn uniformly, the share of training examples that are already pure noise (abar < 0.001):
  linear 17.5%   cosine 2.1%
```

   You should see a table with four rows and two lines. For the linear schedule 525 steps (52.5 percent) have abar below 0.1, 327 steps (32.7 percent) below 0.01, **175 steps (17.5 percent) below 0.001** and 46 below 0.0001. For the cosine schedule the numbers are 207 (20.7 percent), 65 (6.5 percent), 21 (2.1 percent) and 7. So if you draw t uniformly, **17.5 percent of the training examples of the linear schedule are already pure noise, against only 2.1 percent for the cosine schedule.** This is the answer to the second Imagine question: about 175 steps are wasted. Several of the other steps are almost wasted too: at the 0.01 threshold a third of the linear steps are lost.

   *Exercise 2.3:* count the steps with abar below 0.01 (instead of 0.001) for the linear schedule (`waste_lin_001`) and for the cosine schedule (`waste_cos_001`); count only t = 1 .. 1000 (`abar[1:]`). Why it matters: it shows that the conclusion does not depend on one particular threshold.

```python
waste_lin_001 = None   # how many of the 1,000 steps have abar below 0.01 (linear schedule)
waste_cos_001 = None   # the same for the cosine schedule
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if waste_lin_001 is None or waste_cos_001 is None:
    print("Fill in waste_lin_001 and waste_cos_001 above, then run this cell again.")
else:
    check("waste_lin_001 is right", waste_lin_001 == int((abar[1:] < 0.01).sum()), "Count over t = 1 .. 1000 (abar[1:]).")
    check("waste_cos_001 is right", waste_cos_001 == int((abar_c[1:] < 0.01).sum()), "Same with abar_c.")
    RESULTS["p2_waste"] = (int(waste_lin_001), int(waste_cos_001))
```

   A possible answer, and what it means:

```python
waste_lin_001 = int((abar[1:] < 0.01).sum())
waste_cos_001 = int((abar_c[1:] < 0.01).sum())
print(waste_lin_001, waste_cos_001)
```

   The counts are 327 and 65, in agreement with the table. In a real training run this means that, with the linear schedule, roughly every third example gives the network an almost-pure-noise problem, and with the cosine schedule only about one in fifteen. A good schedule spreads the *difficulty* evenly over the steps.

4. **Step 2.4. Choosing the number of steps T.** For the experiments of the following parts we do not want 1,000 network calls per picture: it would take too long on a CPU. We can use a shorter process, with fewer and larger steps, as long as two things stay true: the process must still end in (almost) pure noise, and no single step may add so much noise that it becomes hard to undo. The rule we use is to shorten T and to stretch the range of beta by the same factor: b0 = 0.0001 * 1000 / T and b1 = 0.02 * 1000 / T. The block prints the end value abar_T and the largest beta for several values of T, builds the schedule that we use in Part 6 (T = 100) and prints abar at four steps.

```python
print("   T    b0        b1       abar_T      largest beta")
for T_ in [50, 100, 200, 500, 1000]:
    b_, a_, ab_ = make_schedule(T_, 1e-4 * 1000 / T_, 0.02 * 1000 / T_)
    print(f"{T_:5d}  {1e-4 * 1000 / T_:.4f}   {0.02 * 1000 / T_:.4f}   {ab_[T_]:.3e}   {b_.max():.3f}")

T_lab = 100
beta_l, alpha_l, abar_l = make_schedule(T_lab, 1e-4 * 1000 / T_lab, 0.02 * 1000 / T_lab)
print("\nThe schedule used for the pictures in Part 6: T = 100, beta from", beta_l[1], "to", beta_l[-1])
print("abar at t = 25, 50, 75, 100:", [round(float(abar_l[t_]), 4) for t_ in [25, 50, 75, 100] ])
assert abar_l[T_lab] < 0.01
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
   T    b0        b1       abar_T      largest beta
   50  0.0020   0.4000   7.745e-06   0.400
  100  0.0010   0.2000   2.039e-05   0.200
  200  0.0005   0.1000   3.032e-05   0.100
  500  0.0002   0.0400   3.766e-05   0.040
 1000  0.0001   0.0200   4.036e-05   0.020

The schedule used for the pictures in Part 6: T = 100, beta from 0.001 to 0.2
abar at t = 25, 50, 75, 100: [0.5279, 0.0742, 0.0026, 0.0]
```

   You should see a table and two lines. With the scaled range every schedule ends at pure noise: abar_T is 7.7e-6 for T = 50, 2.0e-5 for T = 100, 3.0e-5 for T = 200, and 4.0e-5 for T = 1000, all far below 0.01 (the assertion for T = 100 checks it). What changes is the largest beta: 0.4 for T = 50, 0.2 for T = 100, 0.1 for T = 200 and 0.02 for T = 1000. A big beta means a big, hard step. We choose **T = 100** for the pictures of Part 6: it is ten times cheaper than T = 1000 and each step is still small (beta from 0.001 to 0.2). The abar values 0.5279, 0.0742, 0.0026 and 0.00002 at t = 25, 50, 75 and 100 show that the picture is half gone at about a quarter of the way and completely gone by three quarters. (The lab's 1-D experiments in Part 3 use T = 200 with a different range, from 0.001 to 0.06.)

   *Exercise 2.4:* scale the same rule to a longer process: T = 500 with b0 = 0.0001 * 1000 / 500 and b1 = 0.02 * 1000 / 500. Store abar_T in `abar_T_500` and the largest beta in `max_beta_500`. Why it matters: it confirms the rule that scaling the beta range with 1000 / T keeps the end at pure noise for any T.

```python
abar_T_500 = None    # abar_T for T = 500 with the scaled range (b0 = 1e-4 * 1000 / T, b1 = 0.02 * 1000 / T)
max_beta_500 = None  # the largest beta of that schedule
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if abar_T_500 is None or max_beta_500 is None:
    print("Fill in abar_T_500 and max_beta_500 above, then run this cell again.")
else:
    b5, _, a5 = make_schedule(500, 1e-4 * 1000 / 500, 0.02 * 1000 / 500)
    check("abar_T_500 is right", abs(abar_T_500 - a5[500]) < 1e-7, "make_schedule(500, 0.0002, 0.04)[2][500].")
    check("max_beta_500 is right", abs(max_beta_500 - 0.04) < 1e-9, "The last beta is b1 = 0.02 * 1000 / 500.")
    check("the end is pure noise (below 0.01)", abar_T_500 < 0.01, "Check that you used the scaled range.")
    RESULTS["p2_T500"] = (float(abar_T_500), float(max_beta_500))
```

   A possible answer, and what it means:

```python
b5, _, a5 = make_schedule(500, 1e-4 * 1000 / 500, 0.02 * 1000 / 500)
abar_T_500 = float(a5[500])
max_beta_500 = float(b5.max())
print(abar_T_500, max_beta_500)
```

   For T = 500 the final abar is 3.77e-05 (pure noise) and the largest beta is 0.04: smaller steps than T = 100, at the price of five times as many network calls. Choosing T always trades the **cost of sampling** against the **difficulty of every step**.

5. **Step 2.5. A schedule is a choice.** There is nothing sacred about the linear or the cosine curve. Any smooth curve that starts at abar_0 = 1, decreases and ends near 0 defines a valid forward process, because the closed form of Part 1.3 works for every abar. To see how different the choices can be, the block adds a third schedule, the quadratic one: abar_t = (1 - t / T) squared. It plots the three SNR curves on one log axis and prints, for each, the step at which the SNR passes 1 and the values of abar at t = 300 and t = 900.

```python
def quad_schedule(T=1000):
    """A third schedule: abar_t = (1 - t / T) squared; the picture is gone exactly at t = T."""
    ab = (1 - np.arange(T + 1) / T) ** 2
    beta = np.zeros(T + 1); beta[1:] = np.minimum(1 - ab[1:] / ab[:-1], 0.999)
    return beta, 1 - beta, np.cumprod(1 - beta)

abar_q = quad_schedule()[2]
plt.figure(figsize=(5, 3))
for name, ab in [("linear", abar), ("cosine", abar_c), ("quadratic", abar_q)]:
    plt.semilogy(np.arange(1, 1000), snr(np.arange(1, 1000), ab), label=name)
plt.axhline(1, color="grey", ls="--"); plt.legend(); plt.xlabel("step t"); plt.ylabel("SNR (log axis)"); plt.tight_layout(); plt.show()
for name, ab in [("linear", abar), ("cosine", abar_c), ("quadratic", abar_q)]:
    print(f"{name:10s} SNR = 1 at t = {int(np.argmax(ab <= 0.5)):4d}   abar[300] = {ab[300]:.3f}   abar[900] = {ab[900]:.5f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
linear     SNR = 1 at t =  260   abar[300] = 0.396   abar[900] = 0.00028
cosine     SNR = 1 at t =  497   abar[300] = 0.787   abar[900] = 0.02409
quadratic  SNR = 1 at t =  293   abar[300] = 0.490   abar[900] = 0.01000
```

   You should see a plot with three descending curves and three lines. The SNR crosses 1 at t = 260 (linear), t = 497 (cosine) and t = 293 (quadratic). At t = 300 the three schedules have abar = 0.396, 0.787 and 0.490; at t = 900 they have 0.00028, 0.024 and 0.0100. All three are valid forward processes, and they sit in a family of many others. They differ in how fast they remove the information, and in whether the end is really pure noise (the cosine schedule at t = 900 still has abar = 0.024, so its last 100 steps are used to finish the job). Researchers choose by trying them: the cosine schedule gave better pictures in the paper that introduced it, but this is an experimental finding, not a law.

   *Exercise 2.5:* design your own schedule. Write the function `my_abar(T)` so that it returns an array of T + 1 numbers (abar_0 = 1, strictly decreasing, abar_T below 0.01), and write two or three sentences in `answer_2_5` on why a schedule is a design choice and not a law. Why it matters: it shows that you understand which properties are required (a known recipe with a pure noise end) and which are free.

```python
def my_abar(T=1000):
    """Return an array of length T + 1: abar_0 = 1, strictly decreasing, abar_T below 0.01."""
    return None

answer_2_5 = ""   # two or three sentences: why is a schedule a design choice and not a law?
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
ab_mine = my_abar(1000)
if ab_mine is None or len(answer_2_5.split()) < 15:
    print("Write my_abar (return an array) and at least 15 words in answer_2_5 above, then run this cell again.")
else:
    ab_mine = np.asarray(ab_mine, dtype=float)
    check("length is T + 1 = 1001", len(ab_mine) == 1001, "np.arange(1001) has 1001 numbers.")
    check("abar_0 is 1", abs(ab_mine[0] - 1) < 1e-9, "At t = 0 nothing has been noised.")
    check("strictly decreasing", bool(np.all(np.diff(ab_mine) < 0)), "Each step must destroy a little more of the picture.")
    check("abar_T is below 0.01", ab_mine[-1] < 0.01, "The end must be (almost) pure noise.")
    RESULTS["p2_mine"] = round(float(ab_mine[300]), 3)
```

   A possible answer, and what it means:

```python
def my_abar(T=1000):
    tt = np.arange(T + 1) / T
    return np.exp(-7.0 * tt ** 1.5)          # slow start, then faster; ends at exp(-7) = 0.0009

answer_2_5 = ("The only things that the training needs are a known recipe and an end that is pure noise, so many curves work. "
              "The linear schedule is simple but wastes many late steps on pictures that are already noise, the cosine schedule spreads the "
              "difficulty more evenly, and my own curve is just another point in the same family. Which one is best depends on the data and "
              "is found by trying, not by a theorem.")
```

   The example uses abar_t = exp(-7 * (t / T) to the power 1.5): a slow start, then a faster fall, ending at exp(-7) = 0.0009. Any curve that passes the four checks works. In your answer, check that you said what is *required* (start at 1, strictly decreasing, end near 0, so that the end is pure noise) and what is *free* (the shape between). The best shape is found by experiments on your data.

### Part 3. The reverse step and a perfect denoiser (25 min)

*Lesson section: "Sampling".* Now we turn the film around. If someone gives us a picture at step t and tells us which noise it contains, we can take a small step back to step t-1. Repeat this step T times, starting from pure noise, and a picture comes out: the sculptor takes a block of fog and chips away until the statue appears. The network of the next parts will have the job of *telling* us the noise. In this part we do not need a network at all, because for a very simple data set (numbers that come from two bumps) the perfect noise predictor can be written down as a formula. You build the reverse step by hand, derive the formula of the perfect predictor, run the loop from noise to two clean bumps, and then break the loop on purpose (no fresh noise, a predictor that is wrong) to see what each ingredient does. Everything in this part is NumPy and runs in a second.

**Imagine first.** In this part the data are single numbers that come from two narrow bumps, at -2 and at 3, each with a spread of 0.4 and half of the numbers in each bump. We start from 600 random numbers from N(0, 1) (a single broad bell curve, no bumps at all) and apply 200 small denoising steps. Before you run any code, write two predictions. (1) Will the final 600 numbers form two separate bumps, "yes" or "no"? (2) What percentage of them will end in the left bump? You compare with Step 3.3.

```python
# Imagine first: noise to bumps
my_guess_two_bumps = None   # will 600 random numbers from N(0, 1), pushed through 200 small "denoise" steps, become two separate bumps? ("yes" or "no")
my_guess_left = None        # if yes: what percent of them end in the left bump (0 to 100)?
```

1. **Step 3.1. The reverse step by hand.** The reverse step turns x_t into x_(t-1) with the formula x_(t-1) = (x_t - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sigma_t * z, where eps_hat is the noise estimate (from the network, later), z is a fresh draw from N(0, 1) and sigma_t = sqrt(beta_t). Read it from the inside out. (1) beta_t / sqrt(1 - abar_t) * eps_hat is the part of the noise that this step is allowed to remove. (2) Dividing by sqrt(alpha_t) undoes the shrinking of the forward step. (3) The fresh noise sigma_t * z is added back because the true previous picture is not known exactly: the process must keep some randomness. At the very last step (t = 1) we use z = 0, since there is nothing left to explore. The block defines this schedule for the whole part: T = 200 with beta from 0.001 to 0.06 (ᾱ_T = 0.00198), then writes the function `reverse_step`, and works the one-pixel example of the lesson in detail: t = 300 on the schedule of Part 1, x_t = 1.4, with a perfect noise estimate for a clean pixel x_0 = 2.

```python
beta3, alpha3, abar3 = make_schedule(200, 0.001, 0.06)     # the schedule of this part: T = 200
print(f"schedule of Part 3: abar[200] = {abar3[200]:.5f}")
assert abs(abar3[200] - 0.00198) < 0.0001

def reverse_step(x_t, t, eps_hat, z, beta, abar):
    """One denoising step: x_(t-1) = (x_t - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sigma_t * z, with sigma_t = sqrt(beta_t)."""
    alpha_t = 1 - beta[t]
    mean = (x_t - beta[t] / np.sqrt(1 - abar[t]) * eps_hat) / np.sqrt(alpha_t)
    return mean + np.sqrt(beta[t]) * z

# a worked example on the schedule of Part 1: t = 300, x_t = 1.4, and the clean pixel was x_0 = 2
t_, x_t = 300, 1.4
eps_hat = (x_t - np.sqrt(abar[t_]) * 2.0) / np.sqrt(1 - abar[t_])      # the perfect noise estimate for x_0 = 2
step_a = beta[t_] / np.sqrt(1 - abar[t_]) * eps_hat                    # how much noise to take away
print(f"beta_300 = {beta[t_]:.5f}   abar_300 = {abar[t_]:.4f}   eps_hat = {eps_hat:.4f}")
print(f"subtract beta/sqrt(1-abar)*eps_hat = {step_a:.5f}   ->  {x_t - step_a:.5f}")
print(f"divide by sqrt(alpha_300) = {np.sqrt(1 - beta[t_]):.5f}    ->  {(x_t - step_a) / np.sqrt(1 - beta[t_]):.5f}")
print(f"add sigma * z with sigma = {np.sqrt(beta[t_]):.5f} and z = 1  ->  {reverse_step(x_t, t_, eps_hat, 1.0, beta, abar):.5f}")
print(f"the last step (z = 0), t = 1, x_1 = 0.1, eps_hat = 0.05  ->  {reverse_step(0.1, 1, 0.05, 0.0, beta, abar):.5f}")
assert abs(reverse_step(x_t, t_, eps_hat, 0.0, beta, abar) - (x_t - step_a) / np.sqrt(1 - beta[t_])) < 1e-12
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
schedule of Part 3: abar[200] = 0.00198
beta_300 = 0.00606   abar_300 = 0.3964   eps_hat = 0.1812
subtract beta/sqrt(1-abar)*eps_hat = 0.00141   ->  1.39859
divide by sqrt(alpha_300) = 0.99697    ->  1.40284
add sigma * z with sigma = 0.07782 and z = 1  ->  1.48066
the last step (z = 0), t = 1, x_1 = 0.1, eps_hat = 0.05  ->  0.09950
```

   You should see the schedule check and the example. For the schedule of Part 3, abar_200 is 0.00198: pure noise at the end. In the worked example, beta_300 = 0.00606, abar_300 = 0.3964 and the perfect noise estimate is eps_hat = 0.1812 (the noise that was mixed into x_0 = 2 to make 1.4). The step removes 0.00606 / 0.777 * 0.1812 = 0.00141, so 1.4 becomes 1.39859; dividing by sqrt(alpha_300) = 0.99697 gives **1.40284**, the mean of the previous step; adding the fresh noise with sigma = 0.07782 and z = 1 gives 1.48066. The pixel moved by only 0.003 in the deterministic part, and by up to 0.08 for the random part: **the random part is larger than the denoising part at each single step**, and the denoising only wins in the long run, over many steps. The last line is the last step (t = 1, z = 0): 0.1 becomes 0.0995. The assertion checks the formula.

   *Exercise 3.1:* compute two reverse steps on the schedule of Part 3 (`beta3`, `abar3`), both with z = 0: at t = 200 with x_t = 1.0 and eps_hat = 0.9, stored in `x_prev`; and at t = 1 with x_t = 0.1 and eps_hat = 0.05, stored in `x_prev_last`. Use `reverse_step(x_t, t, eps_hat, z, beta3, abar3)`. Why it matters: it shows how different the step is at the two ends of the process.

```python
x_prev = None        # t = 200, x_t = 1.0, eps_hat = 0.9, z = 0.0, on the schedule beta3, abar3
x_prev_last = None   # t = 1, x_t = 0.1, eps_hat = 0.05, z = 0.0, on the same schedule
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if x_prev is None or x_prev_last is None:
    print("Fill in x_prev and x_prev_last above, then run this cell again.")
else:
    e1 = (1.0 - beta3[200] / np.sqrt(1 - abar3[200]) * 0.9) / np.sqrt(1 - beta3[200])
    e2 = (0.1 - beta3[1] / np.sqrt(1 - abar3[1]) * 0.05) / np.sqrt(1 - beta3[1])
    check("x_prev is right", abs(x_prev - e1) < 1e-6, "reverse_step(1.0, 200, 0.9, 0.0, beta3, abar3).")
    check("x_prev_last is right", abs(x_prev_last - e2) < 1e-6, "reverse_step(0.1, 1, 0.05, 0.0, beta3, abar3).")
    RESULTS["p3_steps"] = (round(float(x_prev), 4), round(float(x_prev_last), 4))
```

   A possible answer, and what it means:

```python
x_prev = float(reverse_step(1.0, 200, 0.9, 0.0, beta3, abar3))
x_prev_last = float(reverse_step(0.1, 1, 0.05, 0.0, beta3, abar3))
print(x_prev, x_prev_last)
```

   The first step gives 0.9757 and the last step 0.0985. At t = 200 beta is large (0.06) and the step changes the number by about 0.024 for these inputs; at t = 1 beta is tiny (0.001) and the step changes it by 0.0015. In the early reverse steps (large t) the network makes big moves, in the last ones (small t) it only polishes.

2. **Step 3.2. An exact noise predictor.** To run the loop we need a noise predictor. For data that come from a mixture of bumps there is a **perfect** one, and we can write it down. Here is the derivation in plain words. (1) If the clean number x_0 comes from a bump with mean mu and spread sd, then x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps is also a bell curve, with mean sqrt(abar_t) * mu and variance v = abar_t * sd squared + 1 - abar_t. (2) So the whole density of x_t is again a mixture of two bumps, each wider than before. (3) A fact from statistics (Tweedie's formula) says that the best guess of the noise is minus sqrt(1 - abar_t) times the slope of the logarithm of that density. (4) For a mixture, the slope is an average of the slopes of the bumps, weighted by the probability r_i that x came from bump i. Together: eps_exact(x, t) = sqrt(1 - abar_t) * sum over i of r_i * (x - sqrt(abar_t) * mu_i) / v. The code below writes this formula down, and also writes the density itself (`log_marginal`) so that we can check the formula against a **numerical derivative** of the logarithm of the density at one point. Then it prints the predictor for a few values of x at three steps.

```python
MEANS, WEIGHTS, SD = (-2.0, 3.0), (0.5, 0.5), 0.4        # the data: two bumps at -2 and 3, each with standard deviation 0.4

def log_marginal(x, t, abar, means=MEANS, weights=WEIGHTS, sd=SD):
    """log of the density of x_t: each bump becomes a wider bump with mean sqrt(abar) * mu_i and variance v_i = abar * sd^2 + 1 - abar."""
    x = np.asarray(x, float); v = abar[t] * sd ** 2 + 1 - abar[t]
    terms = np.stack([np.log(w) - 0.5 * np.log(2 * np.pi * v) - (x - np.sqrt(abar[t]) * m) ** 2 / (2 * v) for m, w in zip(means, weights)])
    mx = terms.max(0)
    return mx + np.log(np.exp(terms - mx).sum(0))

def eps_exact(x, t, abar, means=MEANS, weights=WEIGHTS, sd=SD):
    """The best possible noise predictor: sqrt(1 - abar_t) * sum_i r_i * (x - sqrt(abar_t) * mu_i) / v_i, where r_i is the probability that x came from bump i."""
    x = np.asarray(x, float); v = abar[t] * sd ** 2 + 1 - abar[t]
    terms = np.stack([np.log(w) - (x - np.sqrt(abar[t]) * m) ** 2 / (2 * v) for m, w in zip(means, weights)])
    r = np.exp(terms - terms.max(0)); r = r / r.sum(0)                       # posterior weights r_i
    return np.sqrt(1 - abar[t]) * sum(ri * (x - np.sqrt(abar[t]) * m) / v for ri, m in zip(r, means))

# check against a numerical derivative: eps_exact = - sqrt(1 - abar_t) * d/dx log p_t(x)
t_, x_, h = 100, 0.3, 1e-5
num = -np.sqrt(1 - abar3[t_]) * (log_marginal(x_ + h, t_, abar3) - log_marginal(x_ - h, t_, abar3)) / (2 * h)
print(f"t = {t_}, x = {x_}:  formula {float(eps_exact(x_, t_, abar3)):.6f}   numerical derivative {float(num):.6f}")
assert abs(float(eps_exact(x_, t_, abar3)) - float(num)) < 1e-5
xs = np.array([-3.0, -2.0, 0.0, 1.0, 3.0, 5.0])
print("x                 ", xs)
for tt in [10, 100, 190]:
    print(f"eps_exact, t={tt:3d}  ", np.round(eps_exact(xs, tt, abar3), 3))
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
t = 100, x = 0.3:  formula -0.043103   numerical derivative -0.043103
x                  [-3. -2.  0.  1.  3.  5.]
eps_exact, t= 10   [-0.867 -0.02   1.675 -1.665  0.03   1.724]
eps_exact, t=100   [-2.256 -1.184  0.122 -0.124  1.769  3.922]
eps_exact, t=190   [-2.969 -1.988 -0.029  0.949  2.91   4.88 ]
```

   You should see a check line and a small table. At t = 100 and x = 0.3 the formula gives -0.043103 and the numerical derivative gives -0.043103: they agree to six decimals (the assertion allows 0.00001), so the formula is right. In the table, read each row as "if you see this value of x_t at step t, the noise in it is probably this much". At t = 190 the predicted noise is nearly x itself (-2.969 for x = -3, 2.910 for x = 3): **at large t almost everything you see is noise**. At t = 10 the answers are very different: x = -2 gives -0.020 (a value at the centre of a bump is almost noise-free, since there is nearly no noise at this step), but x = 0 gives 1.675 and x = 1 gives -1.665: the predictor flips its sign between the two bumps (a point at x = 0 belongs to the left bump, so its noise is positive; at x = 1 to the right bump, so its noise is negative). A network will have to learn exactly this kind of sharp structure.

   *Exercise 3.2:* evaluate the exact predictor at the point x = 3.0 for two steps: t = 10 (store it in `eps_x3_t10`) and t = 200 (store it in `eps_x3_t200`), with `eps_exact(x, t, abar3)`. Why it matters: x = 3 is the centre of the right bump; what the predictor says there at the two ends of the process shows how the job of the denoiser changes with t.

```python
eps_x3_t10 = None    # eps_exact at x = 3.0, t = 10 (schedule abar3)
eps_x3_t200 = None   # eps_exact at x = 3.0, t = 200
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if eps_x3_t10 is None or eps_x3_t200 is None:
    print("Fill in eps_x3_t10 and eps_x3_t200 above, then run this cell again.")
else:
    check("eps_x3_t10 is right", abs(eps_x3_t10 - float(eps_exact(3.0, 10, abar3))) < 1e-6, "eps_exact(3.0, 10, abar3).")
    check("eps_x3_t200 is right", abs(eps_x3_t200 - float(eps_exact(3.0, 200, abar3))) < 1e-6, "eps_exact(3.0, 200, abar3).")
    check("at t = 200 the predictor is almost the identity (within 0.1 of x)", abs(eps_x3_t200 - 3.0) < 0.1, "At the end almost everything in x_t is noise, so eps is almost x_t.")
    RESULTS["p3_eps"] = (round(eps_x3_t10, 3), round(eps_x3_t200, 3))
```

   A possible answer, and what it means:

```python
eps_x3_t10 = float(eps_exact(3.0, 10, abar3))
eps_x3_t200 = float(eps_exact(3.0, 200, abar3))
print(eps_x3_t10, eps_x3_t200)
```

   At t = 10 the answer is 0.030: x = 3 is the centre of a bump, and at this step there is almost no noise to remove. At t = 200 it is 2.944, close to x itself: x_200 is almost pure noise, so the best guess of the noise is the input itself, with a slight correction that depends on the position of the bumps.

3. **Step 3.3. The reverse loop.** Here is the sampler. The function `run_reverse` starts with n numbers from N(0, 1), applies `reverse_step` for t = T, T-1, ..., 1 with a predictor that you give it, and returns the final numbers and some snapshots. Run it with 600 numbers and the exact predictor. Draw the histogram at t = 200, 150, 100, 50 and 0 against the true density (the black curve), and measure three numbers: the **mean** of the final numbers (the data mean is 0.5), the **share** in the left bump (the data have 50 percent), and the average **spread inside a bump** (the data have 0.4). The lesson's figure `sampler1d` shows the same experiment.

```python
def run_reverse(n, eps_fn, beta, abar, T, z_scale=1.0, seed=0, record=()):
    """Start from n numbers of pure noise and apply reverse_step for t = T, T-1, ..., 1. Returns the final numbers and a dict of snapshots."""
    r = np.random.RandomState(seed)
    x = r.randn(n); snaps = {T: x.copy()}
    for t in range(T, 0, -1):
        z = z_scale * r.randn(n) if t > 1 else np.zeros(n)          # no fresh noise at the very last step
        x = reverse_step(x, t, eps_fn(x, t), z, beta, abar)
        if t - 1 in record: snaps[t - 1] = x.copy()
    return x, snaps

def bump_stats(x, split=0.5):
    """Mean, share in the left bump (x below the split), and the average spread inside the two bumps."""
    L, R = x[x < split], x[x >= split]
    return float(x.mean()), float(len(L) / len(x)), float((L.std() + R.std()) / 2)

exact3 = lambda x, t: eps_exact(x, t, abar3)
final, snaps = run_reverse(600, exact3, beta3, abar3, 200, record=(150, 100, 50, 0))
grid = np.linspace(-5, 6, 400)
dens = 0.5 * np.exp(-(grid + 2) ** 2 / (2 * 0.16)) / np.sqrt(2 * np.pi * 0.16) + 0.5 * np.exp(-(grid - 3) ** 2 / (2 * 0.16)) / np.sqrt(2 * np.pi * 0.16)
fig, ax = plt.subplots(1, 5, figsize=(14, 2.4), sharey=True)
for a, t_ in zip(ax, [200, 150, 100, 50, 0]):
    a.hist(snaps[t_], 40, range=(-5, 6), density=True, color="C0", alpha=0.7); a.plot(grid, dens, "k", lw=1); a.set_title(f"t = {t_}")
plt.tight_layout(); plt.show()
m, sl, sp = bump_stats(final)
print(f"final: mean {m:.2f}   share in the left bump {sl:.1%}   spread inside a bump {sp:.2f}   (data: mean 0.50, share 50%, spread 0.40)")
print("spread of the cloud at t = 200, 150, 100, 50, 0:", [round(float(snaps[t_].std()), 2) for t_ in [200, 150, 100, 50, 0] ])
assert abs(m - 0.5) < 0.3 and 0.40 < sl < 0.60 and 0.3 < sp < 0.55
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
final: mean 0.45   share in the left bump 51.8%   spread inside a bump 0.39   (data: mean 0.50, share 50%, spread 0.40)
spread of the cloud at t = 200, 150, 100, 50, 0: [1.0, 1.08, 1.43, 2.11, 2.48]
```

   You should see five histograms in a row and two lines. At t = 200 the histogram is one bell curve (the starting noise). At t = 150 it has widened slightly; at t = 100 it is broad and flat, with a first hint of two humps; at t = 50 two clear humps with a gap between them; at t = 0 the histogram follows the black curve of the data with two narrow bumps. The numbers: the final mean is 0.45 (data: 0.50), the share in the left bump is **51.8 percent** (data: 50), the spread inside a bump is **0.39** (data: 0.40). The assertion allows 0.3 for the mean, 40 to 60 percent for the share and 0.3 to 0.55 for the spread. The second line is a surprise: the standard deviation of the whole cloud is 1.00 at t = 200, then grows to 1.08, 1.43, 2.11 and 2.48. The cloud has to *grow* from a spread of 1 to the spread of the two bumps (which is about 2.5, because the bumps are 5 apart). **Pure noise became two bumps, with nothing but a small repeated step and the perfect predictor**: the answer to the Imagine questions is "yes", with close to half in each bump. With only 600 samples, the share fluctuates by a few percent.

   *Exercise 3.3:* change the data. Let the mixture have weights 0.8 for the left bump and 0.2 for the right bump: use `weights=(0.8, 0.2)` in `eps_exact`, run `run_reverse` with 2,000 numbers (seed=1), and store the share of the final numbers in the left bump in `share_left_80` and their mean in `mean_80`. Use the function `bump_stats` of the last block. Why it matters: the sampler has no knowledge of the bumps except through the noise predictor, so changing the predictor changes the result, as it should.

```python
share_left_80 = None   # share of samples in the left bump when the data are 80% left bump, 20% right bump
mean_80 = None         # the mean of the final samples
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if share_left_80 is None or mean_80 is None:
    print("Fill in share_left_80 and mean_80 above, then run this cell again.")
else:
    check("share_left_80 is close to 0.8 (0.06)", abs(share_left_80 - 0.8) < 0.06, "Pass weights=(0.8, 0.2) to eps_exact, run 2,000 particles.")
    check("mean_80 is close to the true mean -1.0 (0.15)", abs(mean_80 - (0.8 * -2 + 0.2 * 3)) < 0.15, "0.8 * (-2) + 0.2 * 3 = -1.0.")
    RESULTS["p3_lopsided"] = (round(float(share_left_80), 3), round(float(mean_80), 2))
```

   A possible answer, and what it means:

```python
eps80 = lambda x, t: eps_exact(x, t, abar3, weights=(0.8, 0.2))
x80, _ = run_reverse(2000, eps80, beta3, abar3, 200, seed=1)
mean_80, share_left_80, _ = bump_stats(x80)
print(share_left_80, mean_80)
```

   The share in the left bump is 0.794 (about 80 percent) and the mean is -0.97 (the true mean is 0.8 * (-2) + 0.2 * 3 = -1.0). The sampler followed the new predictor without any change in the loop: **all the knowledge about the data sits in the noise predictor.**

4. **Step 3.4. Fresh noise or not?.** The reverse step adds fresh noise sigma_t * z at every step. Is it necessary? Run the loop with the fresh noise multiplied by `z_scale` = 1.0 (the real sampler), 0.5 and 0.0 (no fresh noise), with the same starting numbers, and print the mean, the share in the left bump and the spread inside a bump. Then draw the final histograms for 1.0 and 0.0.

```python
print("fresh noise (z_scale)   mean    share left   spread inside a bump")
res = {}
for zs in [1.0, 0.5, 0.0]:
    xf, _ = run_reverse(600, exact3, beta3, abar3, 200, z_scale=zs, seed=0)
    res[zs] = xf; m, sl, sp = bump_stats(xf)
    print(f"{zs:12.1f}          {m:5.2f}   {sl:8.1%}     {sp:6.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
for a, zs in zip(ax, [1.0, 0.0]):
    a.hist(res[zs], 50, range=(-5, 6), density=True, alpha=0.7); a.plot(grid, dens, "k", lw=1); a.set_title(f"z_scale = {zs}")
plt.tight_layout(); plt.show()
assert bump_stats(res[0.0])[2] < bump_stats(res[1.0])[2]
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
fresh noise (z_scale)   mean    share left   spread inside a bump
         1.0           0.45      51.8%      0.390
         0.5           0.41      52.2%      0.203
         0.0           0.34      53.3%      0.072
```

   You should see a table with three rows and two histograms. With full fresh noise the spread inside a bump is 0.390, with half of it 0.203, and **without any fresh noise 0.072**: the numbers all end up in a tiny spike at the centre of each bump (the right histogram shows two thin needles instead of two bumps). The mean (0.45, 0.41, 0.34) and the share in the left bump (51.8, 52.2, 53.3 percent) change much less: the two bumps are in the right places, but without fresh noise each one is **collapsed to nearly one value**. What the fresh noise does: the predictor, the exact one included, always moves a particle towards the most likely clean value. Without randomness, every particle that starts close to another one is pulled to the same point, so you lose the *variety* of the data: this is similar in spirit to the loss of variety you saw in Week 12, but here the cause is clear and the remedy is built in. The fresh noise at every step keeps the cloud as wide as the data. (In Part 6 you will meet DDIM, a deterministic sampler that is not collapsed: it uses a different, matching update, not the DDPM step without noise.)

   *Exercise 3.4:* store the spread inside a bump (the third number of `bump_stats`) for `z_scale=0.5` in `spread_half` and for `z_scale=1.0` in `spread_full`, using 600 particles, seed 0 and the same schedule and predictor. Why it matters: it confirms that the spread grows with the fresh noise, between the needle and the real width.

```python
spread_half = None     # the spread inside a bump (bump_stats, third number) with z_scale = 0.5, 600 particles, seed 0
spread_full = None     # the same with z_scale = 1.0
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if spread_half is None or spread_full is None:
    print("Fill in spread_half and spread_full above, then run this cell again.")
else:
    spread_zero = bump_stats(run_reverse(600, exact3, beta3, abar3, 200, z_scale=0.0, seed=0)[0])[2]
    check("spread_full is close to the data spread 0.4 (0.08)", abs(spread_full - 0.4) < 0.08, "z_scale=1.0 is the real sampler.")
    check("the spread grows with the amount of fresh noise", spread_zero < spread_half < spread_full + 0.05, "Less fresh noise, tighter bumps.")
    RESULTS["p3_spread"] = (round(float(spread_half), 3), round(float(spread_full), 3))
```

   A possible answer, and what it means:

```python
spread_half = bump_stats(run_reverse(600, exact3, beta3, abar3, 200, z_scale=0.5, seed=0)[0])[2]
spread_full = bump_stats(run_reverse(600, exact3, beta3, abar3, 200, z_scale=1.0, seed=0)[0])[2]
print(spread_half, spread_full)
```

   `spread_half` is 0.203 and `spread_full` is 0.390, against 0.072 for no noise: half the noise gives about half of the spread. So the fresh noise is not decoration: it sets the width of the bumps.

5. **Step 3.5. Wrong denoisers.** What happens if the noise predictor is wrong? We test four predictors with the same loop: the exact one, a predictor that always says "there is no noise" (eps_hat = 0), a predictor that says "everything I see is noise" (eps_hat = x_t) and the exact one at half strength. For each, the block prints the mean and the standard deviation of the final numbers, their smallest and largest value, and the share that end within 1.2 (three standard deviations) of a bump centre.

```python
wrong = {
    "exact predictor":          exact3,
    "predict no noise (0)":     lambda x, t: np.zeros_like(x),
    "predict all noise (x_t)":  lambda x, t: x,
    "half-strength exact":      lambda x, t: 0.5 * exact3(x, t),
}
print("denoiser                  mean      std     min      max     in the bumps (within 3 sd = 1.2)")
for name, fn in wrong.items():
    xf, _ = run_reverse(600, fn, beta3, abar3, 200, seed=0)
    near = np.minimum(np.abs(xf + 2), np.abs(xf - 3)) < 1.2
    print(f"{name:24s} {xf.mean():7.2f} {xf.std():8.2f} {xf.min():8.2f} {xf.max():8.2f}   {near.mean():6.1%}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
denoiser                  mean      std     min      max     in the bumps (within 3 sd = 1.2)
exact predictor             0.45     2.48    -3.06     4.05    99.7%
predict no noise (0)       -1.90    31.86  -101.42   100.93     6.2%
predict all noise (x_t)     0.04     0.41    -1.26     1.37     2.3%
half-strength exact         0.36     3.13    -5.57     6.02    74.5%
```

   You should see a table with four rows. The exact predictor gives a mean of 0.45, a standard deviation of 2.48, values between -3.06 and 4.05, and 99.7 percent of the numbers within 1.2 of a bump centre. **Predicting no noise** (eps_hat = 0) removes nothing: each step only divides by sqrt(alpha_t), a number slightly below 1, and adds fresh noise, so the numbers grow at every step; after 200 steps they have a standard deviation of 31.9 and range from -101 to +101, and only 6.2 percent are near a bump. **Predicting that everything is noise** (eps_hat = x_t) takes away too much: all numbers are pulled to 0 (standard deviation 0.41, between -1.26 and 1.37), and only 2.3 percent are within 1.2 of a bump centre, so they land in the gap between the bumps. The **half-strength** predictor lands in between: a standard deviation of 3.13 and a range from -5.6 to 6.0, with 74.5 percent near a bump. The lesson: the loop is only as good as the predictor, and you can tell a bad predictor from the final samples.

   *Exercise 3.5:* try a predictor that is too strong: 1.5 times the exact one (`lambda x, t: 1.5 * exact3(x, t)`), with 600 numbers and seed 0. Store the share of the final numbers within 1.2 of a bump centre in `near_strong` and the standard deviation of the final numbers in `std_strong`. Why it matters: a bad network can be wrong in either direction; this is the opposite of the half-strength case.

```python
near_strong = None   # the share of final samples within 1.2 of a bump centre, when the predictor is 1.5 * eps_exact (600 particles, seed 0)
std_strong = None    # the standard deviation of those final samples
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if near_strong is None or std_strong is None:
    print("Fill in near_strong and std_strong above, then run this cell again.")
else:
    xs_ = run_reverse(600, lambda x, t: 1.5 * exact3(x, t), beta3, abar3, 200, seed=0)[0]
    chk = np.minimum(np.abs(xs_ + 2), np.abs(xs_ - 3)) < 1.2
    check("near_strong is right", abs(near_strong - chk.mean()) < 0.01, "Count the samples with min(|x + 2|, |x - 3|) below 1.2.")
    check("std_strong is right", abs(std_strong - xs_.std()) < 0.01, "xs_.std() of the final samples.")
    RESULTS["p3_strong"] = (round(float(near_strong), 3), round(float(std_strong), 2))
```

   A possible answer, and what it means:

```python
xs_ = run_reverse(600, lambda x, t: 1.5 * exact3(x, t), beta3, abar3, 200, seed=0)[0]
near_strong = float((np.minimum(np.abs(xs_ + 2), np.abs(xs_ - 3)) < 1.2).mean())
std_strong = float(xs_.std())
print(near_strong, std_strong)
```

   With a predictor that is 1.5 times too strong, `near_strong` is 1.000 and `std_strong` is 2.47: every number lands at a bump, so this looks perfect in the share, but look at the spread inside a bump, which is only 0.27 against 0.39 for the exact predictor. An over-confident predictor gives samples that look right but are **too sharp and too similar**: they have lost the variety of the data. A single number such as "share near a bump" cannot tell you this; that is why we measure diversity and quality separately in Part 6.

6. **Step 3.6. What must the network learn?.** The neural network of the next part has to *learn* the function that you just wrote as a formula. It will only see examples (a noisy number, the step t and the true noise) and not the formula. To get a feeling for how hard that is, the block draws the exact predictor eps_exact(x, t) as a function of x for three steps (t = 10, 100, 190), together with the line eps = x_t, and prints for four steps the largest gap to this line and the steepest slope.

```python
xg = np.linspace(-4, 4, 161)
fig, ax = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for a, tt in zip(ax, [10, 100, 190]):
    a.plot(xg, eps_exact(xg, tt, abar3), label="exact eps"); a.plot(xg, xg, "k:", label="x_t"); a.set_title(f"t = {tt}"); a.set_xlabel("x_t")
ax[0].legend(); plt.tight_layout(); plt.show()
for tt in [10, 100, 190, 200]:
    e = eps_exact(xg, tt, abar3)
    print(f"t = {tt:3d}: largest gap to the line eps = x_t is {np.abs(e - xg).max():.3f}   steepest slope {np.abs(np.diff(e) / np.diff(xg)).max():.2f}")
```

   The block prints (your numbers must be the same with `SEED = 42`):

```text
t =  10: largest gap to the line eps = x_t is 3.123   steepest slope 25.24
t = 100: largest gap to the line eps = x_t is 1.293   steepest slope 1.08
t = 190: largest gap to the line eps = x_t is 0.106   steepest slope 0.99
t = 200: largest gap to the line eps = x_t is 0.066   steepest slope 0.99
```

   You should see three panels and four lines. At t = 190 the blue curve nearly lies on the dotted line: the largest gap to eps = x_t is only 0.106 and the steepest slope is 0.99, so the function to learn is almost the identity. At t = 100 the curve already leaves the line (largest gap 1.293) but is still smooth (steepest slope 1.08). At t = 10 the curve is **steep and jumpy**: the largest gap is 3.123 and the steepest slope is 25.2. It rises and falls sharply near the border between the bumps. So the job gets harder as t gets smaller: for large t the network only has to copy its input; for small t it has to know where the bumps are, down to a fraction of 0.4. At t = 200, the last step of the forward process, the gap is just 0.066.

   *Exercise 3.6:* compute the largest gap between eps_exact(x, 200) and the line eps = x over the grid `xg` and store it in `gap_at_T`. Then write three to five sentences in `answer_3_6`: what must a neural network learn in the next part, and which steps will be easy and which hard? Why it matters: you now know exactly what a perfect network would compute, and that knowledge is the target of the training in Part 4.

```python
gap_at_T = None    # the largest gap |eps_exact(x, 200) - x| over xg
answer_3_6 = ""    # 3 to 5 sentences: what does a neural network have to learn in the next part, and which values of t will be easy or hard?
```

   Then test yourself with this block (it prints a tick or a cross, and keeps your answer for Part 7):

```python
if gap_at_T is None or len(answer_3_6.split()) < 25:
    print("Fill in gap_at_T and write at least 25 words in answer_3_6 above, then run this cell again.")
else:
    check("gap_at_T is right", abs(gap_at_T - np.abs(eps_exact(xg, 200, abar3) - xg).max()) < 1e-3, "np.abs(eps_exact(xg, 200, abar3) - xg).max().")
    check("answer is long enough", True)
    RESULTS["p3_gap"] = round(float(gap_at_T), 3)
```

   A possible answer, and what it means:

```python
gap_at_T = float(np.abs(eps_exact(xg, 200, abar3) - xg).max())
answer_3_6 = ("The network has to learn the function eps(x_t, t) that I computed here by hand: given a noisy number and the step, it must say which part of it is noise. "
              "For large t this is easy, the answer is nearly x_t itself. For small t it is a sharp, wiggly curve that depends on where the bumps are, "
              "so the network must learn the shape of the data. It sees only examples, not the formula, and it needs t as an input because the curve changes with t.")
print(gap_at_T)
```

   The gap at t = 200 is 0.066: the predictor at the end of the process is almost the identity. In your answer, check that you mention (1) the network must predict the noise from the noisy input and the step, (2) large t is easy, small t is hard because the curve is steep and depends on the data, and (3) the network needs to know t. Part 4 trains a small network on exactly this kind of function, now in two dimensions.

### Part 4. Train a denoiser on 2-D data (25 min)

*Lesson section: "Training".* In Parts 1 to 3 you built the forward process and the reverse step, and in Part 3 you gave the reverse step a perfect, hand-made noise predictor. Now a network has to learn that predictor from examples. ==You will build the training pairs (a noisy point, its step number, and the noise that was added), write a sinusoidal time embedding and a small network that takes the noisy point plus the embedding, train it for 4,000 steps on eight blobs arranged in a ring, sample with the reverse loop of Part 3, watch the paths the points take, and see what fewer training steps do.== The data are the same eight blobs as in Week 12, so you can compare with the GAN. This part defines its own noise schedule and samplers (the code of Parts 1 to 3 is not needed). All numbers are for `SEED = 42` (seed 1 is the second seed); training is not exactly the same on every machine, so each number comes with a tolerance.

**Imagine first.** A GAN has trouble with eight blobs: in Week 12 it needed careful tuning, and it often kept only a few of them (mode collapse). A diffusion model never plays a game against a second network. It only learns to guess the noise that was added to a point, over and over, for random noise levels. At the end you start from pure noise and apply the guess 100 times. Before you run anything, guess how many of the 8 blobs a trained diffusion model reaches (0 to 8), and what share of its points lands on a blob. Write your guess in the cell below. You compare it with Step 4.4.

```python
# Imagine first: how many of the 8 blobs will the diffusion model reach, and what share of its points will land on a blob?
my_guess_blobs = None      # a whole number from 0 to 8
my_guess_on_blob = None    # a share between 0 and 1, for example 0.5
```

1. **Step 4.1. The training pairs.** A training example for a diffusion model is made in three moves: take a real point x0, pick a random step number t between 1 and T, and add the amount of noise that belongs to step t. The network gets the noisy point x_t and the number t, and must return the noise ε that was added. Because we know ε (we drew it), we always have the right answer. The noise schedule is the one of Part 2, scaled for a short run: T = 100 steps with β rising linearly from 0.001 to 0.2, which leaves almost nothing of the picture at the end (ᾱ_100 = 2.0e-5). We store the schedule in arrays with T + 1 entries so that the step number t is the index itself (entry 0 means "no noise yet", ᾱ_0 = 1). The helper `q_sample_t` is the closed form of Part 1, written so that every example in a batch has its own t.

```python
# the data: 4,000 points from eight blobs on a ring (the same data as Week 12) and the noise schedule
X_mix, blob_id, centres = make_mixture(4000, seed=0)
data_mix = torch.tensor(X_mix)
print("data:", tuple(data_mix.shape), "  mean %.3f %.3f  std %.3f %.3f" % (*data_mix.mean(0), *data_mix.std(0)))

def make_sched(T=100, b0=0.001, b1=0.2):
    """Linear noise schedule. The arrays have T + 1 entries: the step number t (1..T) is the index; entry 0 means 'no noise yet' (abar = 1)."""
    beta = torch.zeros(T + 1); beta[1:] = torch.linspace(b0, b1, T)
    alpha = 1 - beta
    abar = torch.cumprod(alpha, 0)            # abar_t = alpha_1 * alpha_2 * ... * alpha_t
    return {"T": T, "beta": beta, "alpha": alpha, "abar": abar}

S = make_sched()
print("abar_1 = %.4f   abar_30 = %.4f   abar_100 = %.2e" % (S["abar"][1], S["abar"][30], S["abar"][100]))

def q_sample_t(x0, t, eps, S):
    """Closed form x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps for a batch; t is a LongTensor with one step number per example."""
    ab = S["abar"][t].view(-1, *([1] * (x0.dim() - 1)))     # one abar per example, reshaped so that it broadcasts over the other axes
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps
RESULTS["p4_abar_T"] = float(S["abar"][100])
```

It prints:

```text
data: (4000, 2)   mean -0.054 0.037  std 2.122 2.128
abar_1 = 0.9990   abar_30 = 0.3973   abar_100 = 2.04e-05
```

   *Exercise 4.1:* Write `make_batch_mine(x0, S)`. It receives a batch of clean points `x0` (shape (B, 2)) and must return three tensors: `xt` (the noisy points), `t` (one random integer from 1 to T for every example, a LongTensor of shape (B,)) and `eps` (the noise, standard normal, same shape as `x0`). Why it matters: this function is the whole "data set" of a diffusion model: the pairs (x_t and t in, ε out) are made fresh at every training step, so the network never sees the same noisy point twice. Use `torch.randint(1, T + 1, (B,))` (the upper end is excluded), `torch.randn_like(x0)` and the helper `q_sample_t`.

```python
def make_batch_mine(x0, S):
    """Return (xt, t, eps) for a batch of clean points x0."""
    return None   # YOUR CODE: t = random integers 1..T, eps = standard normal noise, xt = q_sample_t(x0, t, eps, S)

out_ = make_batch_mine(data_mix[:256], S)
if out_ is None:
    print("Fill in make_batch_mine above, then run this cell again.")
else:
    xt_, t_, eps_ = out_
    check("xt and eps have the shape of x0, t has shape (256,)", tuple(xt_.shape) == (256, 2) and tuple(eps_.shape) == (256, 2) and tuple(t_.shape) == (256,), "t needs one number per example")
    check("t holds integers between 1 and 100", t_.dtype == torch.long and int(t_.min()) >= 1 and int(t_.max()) <= 100, "torch.randint(1, T + 1, (B,)) excludes the upper end")
    check("xt follows the closed form for these t and eps", torch.allclose(xt_, q_sample_t(data_mix[:256], t_, eps_, S), atol=1e-6), "xt = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps")
    check("the steps are spread out (both small and large t occur)", int(t_.min()) < 15 and int(t_.max()) > 85, "t must be random for every example, not one number for the batch")
```

   A possible answer (try first, then compare):

```python
def make_batch(x0, S):
    t = torch.randint(1, S["T"] + 1, (len(x0),))     # one random step number per example, 1..T
    eps = torch.randn_like(x0)                       # the noise we will ask the network to guess
    xt = q_sample_t(x0, t, eps, S)                   # the noisy point, from the closed form
    return xt, t, eps
```

The three lines are the three moves of the step. Drawing a new t for every example (and not one t per batch) makes every gradient step see easy and hard noise levels together, which is steadier than training level by level.

```python
xb, tb, eb = make_batch(data_mix[:256], S)
print("one batch: xt", tuple(xb.shape), " t", tuple(tb.shape), " eps", tuple(eb.shape))
print("first three examples:")
for i in range(3):
    print("  x0 = (%5.2f, %5.2f)  t = %3d  eps = (%5.2f, %5.2f)  ->  xt = (%5.2f, %5.2f)" % (*data_mix[i], int(tb[i]), *eb[i], *xb[i]))
assert xb.shape == (256, 2) and int(tb.min()) >= 1 and int(tb.max()) <= S["T"]
```

It prints:

```text
one batch: xt (256, 2)  t (256,)  eps (256, 2)
first three examples:
  x0 = (-3.02,  0.21)  t =  43  eps = (-0.69,  0.56)  ->  xt = (-1.79,  0.60)
  x0 = ( 1.87, -2.10)  t =  68  eps = (-1.51, -1.61)  ->  xt = (-1.34, -1.79)
  x0 = (-2.36, -1.99)  t =  77  eps = (-1.48,  0.43)  ->  xt = (-1.58,  0.35)
```

Now a statistical check of the closed form: noise the same point (3, 0) 20,000 times with the same t. The mean must be √ᾱ_t·x0 and the standard deviation √(1 − ᾱ_t) in each direction.

```python
x_fixed = torch.tensor([ [3.0, 0.0] ]).repeat(20000, 1)
print(" t   abar_t   expected mean (x)  measured    expected std  measured")
for tt in (10, 30, 60, 100):
    tv = torch.full((20000,), tt, dtype=torch.long)
    xs = q_sample_t(x_fixed, tv, torch.randn_like(x_fixed), S)
    ab = float(S["abar"][tt])
    print("%3d  %.5f  %14.3f  %10.3f  %12.3f  %8.3f" % (tt, ab, math.sqrt(ab) * 3.0, xs[:, 0].mean(), math.sqrt(1 - ab), xs[:, 0].std()))
    assert abs(float(xs[:, 0].mean()) - math.sqrt(ab) * 3.0) < 0.03 and abs(float(xs[:, 0].std()) - math.sqrt(1 - ab)) < 0.03
```

It prints:

```text
 t   abar_t   expected mean (x)  measured    expected std  measured
 10  0.90381           2.852       2.851         0.310     0.313
 30  0.39727           1.891       1.886         0.776     0.785
 60  0.02299           0.455       0.455         0.988     0.985
100  0.00002           0.014       0.018         1.000     1.007
```

   **What you should see.**

   The check passes four ticks for your `make_batch_mine`, and the answer prints the shapes of one batch, for example `xt (256, 2)`, `t (256,)` and `eps (256, 2)`, and three examples (the numbers differ with the random generator: t is any integer from 1 to 100). The statistical table is the important part (20,000 copies of the point (3, 0)): at t = 10 (ᾱ = 0.904) the mean of the x-coordinate is 2.851 against the expected √ᾱ·3 = 2.852 and the standard deviation is 0.313 against √(1 − ᾱ) = 0.310; at t = 30 (ᾱ = 0.397) the mean is 1.886 against 1.891 and the spread 0.785 against 0.776; at t = 60 the mean is 0.455 against 0.455; at t = 100 (ᾱ = 2.0e-5) the mean is 0.018 against 0.014 and the standard deviation 1.007 against 1.000. The tolerance is 0.03 (a sample of 20,000 points has a sampling error of about 0.007). The schedule gives ᾱ_1 = 0.9990, ᾱ_30 = 0.3973 and ᾱ_100 = 2.04e-5.

   *What it means.* The closed form does what Part 1 promised: any noisy version of a point can be produced in one line, with no loop over the steps. At t = 100 the point (3, 0) has completely disappeared: the mean is 0.018 and the spread is 1, which is exactly standard normal noise. The network will therefore see the whole range, from a lightly blurred point (t = 10) to pure noise (t = 100).

   Tips and common mistakes:

   - `torch.randint(1, T + 1, ...)` excludes the upper end: without the `+ 1` the step T is never trained, and sampling starts exactly there.
   - `eps` must be drawn with `randn_like(x0)` (shape of x0), not one number for the whole batch.

   Try also (optional):

   - Print `S["abar"][ [1, 25, 50, 75, 100] ]` and find the step where half of the picture is gone (we get ᾱ_27 = 0.474, the first value below 0.5; Step 6.1 asks for it).

2. **Step 4.2. The denoiser: a time embedding and a small network.** The network needs to know how much noise it is looking at: the same noisy point means something different at t = 5 (hardly any noise) and at t = 90 (almost only noise). We give it the step number through a **sinusoidal time embedding**, the formula of Week 8 applied to t: a vector of `dim` numbers, the first half sines and the second half cosines, at frequencies from 1 down to 1/10,000 (the exponent is −ln(10000)·k/half for k = 0, 1, …, half − 1). Two nearby steps get similar vectors and distant steps get different ones, and the network can read the vector with ordinary layers. The embedding (size 16) is glued to the two coordinates of the point, so the network sees 18 numbers, passes them through three hidden layers of 128 units with SiLU (a smooth ReLU, x·sigmoid(x), used in almost every diffusion model) and returns 2 numbers: its guess ε̂ of the noise.

   *Exercise 4.2:* Write `time_embed_mine(t, dim)`. It receives a LongTensor `t` of shape (B,) and returns a float tensor of shape (B, dim): first `half = dim // 2` sines, then `half` cosines, of the angles t·f_k with f_k = exp(−ln(10000)·k/half) for k = 0 … half − 1. Why it matters: this is how the network "hears" the noise level; in Part 5 you will take it away and see what breaks. The check uses two values you can verify by hand: at t = 0 all sines are 0 and all cosines are 1, and at t = 1 the first entry is sin(1) = 0.8415.

```python
def time_embed_mine(t, dim=16):
    """Sinusoidal embedding of the step numbers t (shape (B,)) -> (B, dim)."""
    return None   # YOUR CODE: freqs = exp(-ln(10000) * arange(half) / half); angles = t[:, None] * freqs[None]; cat([sin(angles), cos(angles)], 1)

e0_ = time_embed_mine(torch.tensor([0, 1, 50]), 16)
if e0_ is None:
    print("Fill in time_embed_mine above, then run this cell again.")
else:
    check("the shape is (3, 16)", tuple(e0_.shape) == (3, 16), "dim numbers per step: dim // 2 sines then dim // 2 cosines")
    check("at t = 0: eight zeros (sines) then eight ones (cosines)", torch.allclose(e0_[0], torch.cat([torch.zeros(8), torch.ones(8)])), "sin(0) = 0 and cos(0) = 1")
    check("at t = 1 the first entry is sin(1) = 0.8415", abs(float(e0_[1, 0]) - math.sin(1.0)) < 1e-5, "the first frequency is exp(0) = 1")
    check("at t = 50 the second sine is sin(50 * exp(-ln(10000) / 8))", abs(float(e0_[2, 1]) - math.sin(50 * math.exp(-math.log(10000) / 8))) < 1e-4, "f_1 = exp(-ln(10000) * 1 / half) with half = 8")
```

   A possible answer (try first, then compare):

```python
def time_embed(t, dim=16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)    # f_k from 1 down to about 1/10000
    ang = t.float()[:, None] * freqs[None]                               # an angle for every (step, frequency) pair
    return torch.cat([torch.sin(ang), torch.cos(ang)], 1)                # first half sines, second half cosines
```

Every column is a wave in t: column 0 is the fastest (it changes with every step), the last sine column is the slowest (it hardly changes over 100 steps). Together they work like the digits of a clock: fast hands give the fine position, slow hands the rough one.

Now the network. It uses `time_embed` and lets you switch the time input off (`use_time=False`), which Part 5 needs. The count by hand: the first layer has 18 × 128 + 128 = 2,432 numbers, each 128-to-128 layer has 128 × 128 + 128 = 16,512 numbers (two of them), the output layer has 128 × 2 + 2 = 258: together 2,432 + 33,024 + 258 = 35,714.

```python
class MLPDenoiser(nn.Module):
    """Predicts the noise in a 2-D point x_t, given the step number t (through a sinusoidal time embedding)."""
    def __init__(self, hidden=128, tdim=16, use_time=True):
        super().__init__()
        self.tdim, self.use_time = tdim, use_time
        self.net = nn.Sequential(nn.Linear(2 + (tdim if use_time else 0), hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))
    def forward(self, x, t):
        if self.use_time:
            x = torch.cat([x, time_embed(t, self.tdim)], 1)       # the 2-D point and the 16 time numbers side by side
        return self.net(x)

torch.manual_seed(SEED)
net0 = MLPDenoiser()
print("parameters with time embedding   :", count_params(net0))
print("parameters without time embedding:", count_params(MLPDenoiser(use_time=False)))
print("output for a batch of 5 points   :", tuple(net0(torch.randn(5, 2), torch.randint(1, 101, (5,))).shape))
RESULTS["p4_params"] = count_params(net0)
assert count_params(net0) == 35714
```

It prints:

```text
parameters with time embedding   : 35714
parameters without time embedding: 33666
output for a batch of 5 points   : (5, 2)
```

   **What you should see.**

   The check passes four ticks. The network has **35,714** parameters (2,432 + 16,512 + 16,512 + 258) and the same network without the time embedding has **33,666** (its first layer has 2 × 128 + 128 = 384 numbers instead of 2,432). A batch of 5 points goes in and a batch of 5 noise guesses of shape (5, 2) comes out. The assertion makes sure that your reading of the architecture is the one used in the rest of the lab.

   *What it means.* The time embedding costs only 2,048 parameters (16 extra inputs × 128), under six percent of the network, but it carries one piece of information the network cannot get anywhere else: the step number. In Part 5 you remove it and see what that costs.

   Tips and common mistakes:

   - The two halves must be sines first and then cosines in **both** training and sampling; the network learns one fixed convention.
   - Pass `t` as a LongTensor of step numbers 1 to T, not as the fraction t/T: the frequencies are designed for whole steps.

3. **Step 4.3. The loss and the training loop.** The loss is the mean squared error between the network's guess ε̂ and the true noise ε. That is all: no discriminator, no second network, no adversarial game. One training step: draw a batch of 256 clean points, make the pairs with `make_batch`, compute the loss, take an Adam step. The loss will **not** go to 0: for most noisy points the added noise cannot be told apart exactly (a point near a blob could have come from this blob with a little noise or from the next one with a lot), so the best possible loss is clearly above 0.

   *Exercise 4.3:* Write `train_step(model, opt, x0, S)`: draw a training batch with `make_batch`, compute `F.mse_loss(model(xt, t), eps)`, zero the gradients, call `backward()`, take the optimiser step and return the loss as a Python float. Why it matters: every model in this lab, from the 2-D denoiser to the UNet of Part 6, is trained with exactly these five lines. The check trains a new network for 300 steps with your function: the loss must fall clearly (a fresh network starts near 1, the loss of always guessing 0 noise).

```python
def train_step_mine(model, opt, x0, S):
    """One optimisation step on a batch x0; returns the loss as a float."""
    return None   # YOUR CODE: xt, t, eps = make_batch(...); loss = F.mse_loss(model(xt, t), eps); opt.zero_grad(); loss.backward(); opt.step()

if train_step_mine(MLPDenoiser(), torch.optim.Adam(net0.parameters()), data_mix[:8], S) is None:
    print("Fill in train_step_mine above, then run this cell again.")
else:
    torch.manual_seed(0); m_ = MLPDenoiser(); o_ = torch.optim.Adam(m_.parameters(), lr=1e-3)
    ls_ = [train_step_mine(m_, o_, data_mix[torch.randint(0, 4000, (256,))], S) for _ in range(300)]
    check("the loss at the start is near 1 (the loss of guessing zero noise)", 0.7 < float(np.mean(ls_[:5])) < 1.6, "the mean of ls_[:5] was %.2f; expected about 1" % float(np.mean(ls_[:5])))
    check("after 300 steps the loss has fallen below 0.7", float(np.mean(ls_[-30:])) < 0.7, "the mean of the last 30 losses was %.2f; did you call opt.step()?" % float(np.mean(ls_[-30:])))
```

   A possible answer (try first, then compare):

```python
def train_step(model, opt, x0, S):
    xt, t, eps = make_batch(x0, S)                   # fresh noisy examples
    loss = F.mse_loss(model(xt, t), eps)             # how far is the guessed noise from the true noise?
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)
```

The loss compares the guess with the noise we ourselves drew, so every example has a correct answer: this is why training a diffusion model is as steady as training a classifier.

The full loop: 4,000 steps with batch 256, Adam with learning rate 1e-3. We keep copies of the weights after 500 and 1,500 steps (Step 4.6 uses them). On a CPU this takes about six to twelve seconds. With `QUICK` (the test mode that the Setup cell can switch on) the loop is shortened to 600 steps; the numbers in this text are for the full run.

```python
STEPS_MLP = 600 if QUICK else 4000
CK_MLP = (STEPS_MLP // 8, 3 * STEPS_MLP // 8, STEPS_MLP)          # 500, 1500, 4000 in the full run

def train_mlp(steps=STEPS_MLP, seed=42, use_time=True, ckpt=(), data=None, batch=256, lr=1e-3):
    """Train an MLPDenoiser; returns the model, the list of losses and a dict of weight copies at the steps in ckpt."""
    data = data_mix if data is None else data
    torch.manual_seed(seed)
    model = MLPDenoiser(use_time=use_time); opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, saved = [], {}
    for s in range(1, steps + 1):
        losses.append(train_step(model, opt, data[torch.randint(0, len(data), (batch,))], S))
        if s in ckpt: saved[s] = {k: v.clone() for k, v in model.state_dict().items()}
    return model, losses, saved

t0 = time.time()
model42, losses42, saved42 = train_mlp(seed=42, ckpt=CK_MLP)
print("training took %.1f seconds" % (time.time() - t0))
print("loss at steps 1-50: %.3f   steps 451-500: %.3f   last 200 steps: %.3f" % (np.mean(losses42[:50]), np.mean(losses42[CK_MLP[0] - 50:CK_MLP[0] ]), np.mean(losses42[-200:])))
RESULTS["p4_final_loss"] = round(float(np.mean(losses42[-200:])), 3)
sm = np.convolve(losses42, np.ones(50) / 50, mode="valid")
plt.figure(figsize=(6, 3)); plt.plot(losses42, alpha=0.25, label="loss per step"); plt.plot(np.arange(49, len(losses42)), sm, label="average of 50 steps")
plt.xlabel("training step"); plt.ylabel("MSE of the noise guess"); plt.ylim(0, 1.2); plt.legend(); plt.title("the loss falls fast and then stays above 0"); plt.show()
```

It prints:

```text
training took 6.1 seconds
loss at steps 1-50: 0.716   steps 451-500: 0.324   last 200 steps: 0.254
```

   **What you should see.**

   The check passes two ticks. The training loop takes about **6 seconds** on our CPU (up to about 12 on a slower machine) for 4,000 steps. The loss averaged over steps 1 to 50 is **0.716**, over steps 451 to 500 it is **0.324**, and over the last 200 steps it is **0.254** (tolerance: between 0.2 and 0.3 for seed 42). The plot shows a fast fall in the first 300 steps and then a slow, noisy decline: the faint blue line is the raw loss of every step, the orange line its average.

   *What it means.* The loss stops at about 0.25, not at 0. The part that cannot be predicted is real: at a step with a lot of noise a point lying between two blobs could have come from either of them, and the best guess of the noise is an average. A loss of 0.25 means that the squared error of the noise guess is a quarter of the noise's own variance (1). Do not use "the loss is low" as a quality criterion: the loss at 1,500 steps (0.31) is not much higher than at 4,000 steps (0.25), yet in Step 4.6 the samples are very different.

   Tips and common mistakes:

   - Call `opt.zero_grad()` before `loss.backward()`; without it the gradients pile up from step to step.
   - Do not average a loss over steps of very different t and then compare it with a single-step loss: Step 4.7 shows that the loss depends strongly on t.

4. **Step 4.4. Sample with the trained network.** Sampling is the reverse loop of Part 3 with the trained network in place of the exact predictor. Start from 2,000 points of pure noise N(0, I). For t = T, T − 1, …, 1 compute the guess ε̂ = model(x_t, t) and take the reverse step x_(t−1) = (x_t − β_t/√(1 − ᾱ_t)·ε̂)/√α_t + σ_t·z with σ_t² = β_t and a fresh standard normal z (at the last step, t = 1, no fresh noise: z = 0). First you write the one reverse step, then the loop calls it 100 times.

   *Exercise 4.4:* Write `reverse_step_mine(x, e, t, S, z)`: given the current points `x` (shape (B, 2)), the network's guess `e`, the step number `t` (a plain integer) and the fresh noise `z`, return x_(t−1). Use `S["beta"][t]`, `S["alpha"][t]` and `S["abar"][t]`. Why it matters: it is the same formula as Part 3.1, now with ε̂ from a network; a wrong factor here gives pictures that explode or fade, and Step 6.7 shows what happens when the 1/√α_t is forgotten. The check compares your step with the formula evaluated number by number at t = 50 and at t = 1.

```python
def reverse_step_mine(x, e, t, S, z):
    """One reverse step: from x_t to x_(t-1)."""
    return None   # YOUR CODE: mean = (x - beta_t / sqrt(1 - abar_t) * e) / sqrt(alpha_t); then add sqrt(beta_t) * z

x_, e_, z_ = torch.tensor([ [1.0, -2.0] ]), torch.tensor([ [0.5, 0.25] ]), torch.tensor([ [1.0, -1.0] ])
r_ = reverse_step_mine(x_, e_, 50, S, z_)
if r_ is None:
    print("Fill in reverse_step_mine above, then run this cell again.")
else:
    b_, a_, ab_ = float(S["beta"][50]), float(S["alpha"][50]), float(S["abar"][50])
    want_ = (x_ - b_ / math.sqrt(1 - ab_) * e_) / math.sqrt(a_) + math.sqrt(b_) * z_
    check("step 50 matches the formula (%.4f, %.4f)" % tuple(want_[0]), torch.allclose(r_, want_, atol=1e-5), "x_(t-1) = (x - beta/sqrt(1 - abar) * e) / sqrt(alpha) + sqrt(beta) * z")
    r1_ = reverse_step_mine(x_, e_, 1, S, torch.zeros_like(x_)); b1_, a1_, ab1_ = float(S["beta"][1]), float(S["alpha"][1]), float(S["abar"][1])
    check("step 1 with z = 0 is the mean alone", torch.allclose(r1_, (x_ - b1_ / math.sqrt(1 - ab1_) * e_) / math.sqrt(a1_), atol=1e-5), "no fresh noise at the last step")
```

   A possible answer (try first, then compare):

```python
def reverse_step(x, e, t, S, z):
    b, a, ab = S["beta"][t], S["alpha"][t], S["abar"][t]
    mean = (x - b / (1 - ab).sqrt() * e) / a.sqrt()       # the posterior mean: remove the guessed noise, then undo the shrinking of the step
    return mean + b.sqrt() * z                            # add fresh noise with variance beta_t
```

The mean takes out the part of x_t that the network says is noise and rescales by 1/√α_t (a forward step shrank the picture by √α_t); the fresh noise z keeps the particles spread out like real data instead of collapsing to a few points (Part 3.4).

Now the loop. `sample_ddpm` takes any function `eps_fn(x, t)` that returns the noise guess for a batch (so it works for the MLP now and for the UNet in Part 6), a shape, a schedule and a seed. With `keep` you can ask for copies of x_t at chosen steps (Step 4.5 uses this); the option `fix_scale=False` leaves out the 1/√α_t and is only there for Step 6.7.

```python
@torch.no_grad()
def sample_ddpm(eps_fn, shape, S, seed=0, keep=(), fix_scale=True):
    """DDPM sampling: start from noise and apply the reverse step T times. Returns x_0 (and a dict of copies of x_t for t in `keep`)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g); snaps = {S["T"]: x.clone()}
    for t in range(S["T"], 0, -1):
        e = eps_fn(x, torch.full((shape[0],), t, dtype=torch.long))
        z = torch.randn(shape, generator=g) if t > 1 else torch.zeros(shape)     # no fresh noise at the very last step
        if fix_scale: x = reverse_step(x, e, t, S, z)
        else:         x = (x - S["beta"][t] / (1 - S["abar"][t]).sqrt() * e) + S["beta"][t].sqrt() * z    # the broken sampler of Step 6.7
        if (t - 1) in keep: snaps[t - 1] = x.clone()
    return (x, {k: v for k, v in snaps.items() if k in keep}) if keep else x

model42.eval()
t0 = time.time()
gen42 = sample_ddpm(lambda x, t: model42(x, t), (2000, 2), S, seed=0).numpy()
print("sampling 2,000 points took %.1f seconds (%d network calls)" % (time.time() - t0, S["T"]))
cov_r, q_r = modes_covered(X_mix, centres); cov_g, q_g = modes_covered(gen42, centres)
print("real data     : %d of 8 blobs, %.3f of the points on a blob" % (cov_r, q_r))
print("diffusion s42 : %d of 8 blobs, %.3f of the points on a blob" % (cov_g, q_g))
RESULTS["p4_blobs_s42"] = cov_g; RESULTS["p4_on_blob_s42"] = round(q_g, 3)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a_, pts, ttl in ((ax[0], X_mix[:2000], "real data"), (ax[1], gen42, "diffusion model (seed 42)")):
    a_.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5); a_.scatter(centres[:, 0], centres[:, 1], c="r", marker="x"); a_.set_title(ttl); a_.set_aspect("equal"); a_.set_xlim(-4.5, 4.5); a_.set_ylim(-4.5, 4.5)
plt.show()
```

It prints:

```text
sampling 2,000 points took 0.1 seconds (100 network calls)
real data     : 8 of 8 blobs, 0.990 of the points on a blob
diffusion s42 : 8 of 8 blobs, 0.917 of the points on a blob
```

A second seed shows how much the result depends on luck. The cell trains a second network (seed 1) and samples from it.

```python
model1, losses1, _ = train_mlp(seed=1)
model1.eval()
gen1 = sample_ddpm(lambda x, t: model1(x, t), (2000, 2), S, seed=0).numpy()
cov_1, q_1 = modes_covered(gen1, centres)
print("diffusion s1  : %d of 8 blobs, %.3f of the points on a blob   (final loss %.3f)" % (cov_1, q_1, np.mean(losses1[-200:])))
RESULTS["p4_blobs_s1"] = cov_1; RESULTS["p4_on_blob_s1"] = round(q_1, 3)
if not QUICK:
    assert cov_g >= 7 and cov_1 >= 7 and q_g > 0.6 and q_1 > 0.6, "the diffusion model should cover (almost) all eight blobs with most points on a blob"
```

It prints:

```text
diffusion s1  : 8 of 8 blobs, 0.912 of the points on a blob   (final loss 0.261)
```

   **What you should see.**

   The check passes two ticks. Sampling 2,000 points takes **0.1 seconds** (100 network calls on a batch of 2,000). The real data (checked with the same function) cover **8 of 8** blobs with **0.990** of the points on a blob (a point counts as "on a blob" within three standard deviations of a centre, so 1 − 0.990 is the 1 percent of true points that fall outside by chance). The diffusion model with seed 42 covers **8 of 8** blobs with **0.917** of its points on a blob; the second seed gives **8 of 8** and **0.912** (final loss 0.261). Tolerance: 8 (at least 7) of 8 blobs and a share between 0.85 and 0.95; the assertion in the cell fails if fewer than 7 blobs are found or fewer than 60 percent of the points are on a blob. The scatter plot shows eight clean clusters with a few points in between the blobs (the 8 percent that are not on a blob).

   *What it means.* Compare with your guess from the beginning of the part, and with Week 12: the GAN needed a careful choice of learning rates and often collapsed onto a few blobs; the diffusion model, trained with an ordinary squared-error loss and no second network, found all eight blobs with both seeds, and with the same settings. The price is in sampling: 100 network calls per point instead of the GAN's one. The 8 percent of points that are off a blob lie between blobs or in their tails; Step 4.6 shows that this share is much larger for a briefly trained network.

   Tips and common mistakes:

   - Put the model in `eval()` mode before sampling (the MLP has no dropout, but it is a good habit) and use `torch.no_grad()`: the decorator on `sample_ddpm` does this.
   - No fresh noise at the last step (t = 1): otherwise you add noise to the finished point and lose sharpness.

5. **Step 4.5. Look at the trajectories.** Why does this work? Because each reverse step moves every point a little towards where data are more likely, and the fresh noise makes sure the points do not all fall into the nearest blob. A picture of the paths makes this visible: we run the sampler again with copies of x_t at all steps and draw the paths of 20 points as lines from noise (t = 100) to data (t = 0), and the whole cloud of 1,000 points at five moments.

```python
keep_all = set(range(0, S["T"] + 1))
_, snaps = sample_ddpm(lambda x, t: model42(x, t), (1000, 2), S, seed=3, keep=keep_all)
path = torch.stack([snaps[t] for t in range(S["T"], -1, -1)])          # shape (T + 1, 1000, 2): the position of every point at every step
print("path array:", tuple(path.shape), "  mean distance from the origin at t = 100, 50, 0: %.2f, %.2f, %.2f" % tuple(float(path[S['T'] - t].norm(dim=1).mean()) for t in (100, 50, 0)))
fig, ax = plt.subplots(1, 6, figsize=(16, 3))
for i in range(20): ax[0].plot(path[:, i, 0], path[:, i, 1], lw=0.7, alpha=0.8)
ax[0].scatter(centres[:, 0], centres[:, 1], c="k", marker="x"); ax[0].set_title("20 paths, t = 100 to 0")
for a_, t in zip(ax[1:], (100, 75, 50, 25, 0)):
    p_ = snaps[t].numpy(); a_.scatter(p_[:, 0], p_[:, 1], s=2, alpha=0.5); a_.scatter(centres[:, 0], centres[:, 1], c="r", marker="x", s=15); a_.set_title("t = %d" % t)
for a_ in ax: a_.set_aspect("equal"); a_.set_xlim(-4.5, 4.5); a_.set_ylim(-4.5, 4.5)
plt.tight_layout(); plt.show()
```

It prints:

```text
path array: (101, 1000, 2)   mean distance from the origin at t = 100, 50, 0: 1.27, 1.44, 3.02
```

   *Exercise 4.5:* Compute for each of the 1,000 paths the **straightness**: the straight-line distance from the start (t = 100) to the end (t = 0) divided by the total length of the path (the sum of the lengths of the 100 small steps). A value of 1 means a straight line, a small value means a winding path. Store the mean in `straightness`. Why it matters: it turns the picture into one number and shows that the fresh noise z makes the paths wiggle (the DDIM sampler of Part 6 removes that wiggle). Use `path`: its first axis is the step, the second the point, the third x and y.

```python
straightness = None   # YOUR CODE: total = (path[1:] - path[:-1]).norm(dim=2).sum(0); direct = (path[-1] - path[0]).norm(dim=1); straightness = float((direct / total).mean())

if straightness is None:
    print("Fill in straightness above, then run this cell again.")
else:
    check("straightness is between 0 and 1", 0 < straightness <= 1.0, "a path cannot be shorter than the straight line")
    check("the paths wind (straightness below 0.7)", straightness < 0.7, "total length = sum over the 100 steps of the distance between neighbouring positions")
```

   A possible answer (try first, then compare):

```python
total = (path[1:] - path[:-1]).norm(dim=2).sum(0)       # length of each path: the sum of its 100 small steps
direct = (path[-1] - path[0]).norm(dim=1)               # straight-line distance from the starting noise to the final point
straightness = float((direct / total).mean())
print("mean straightness: %.3f" % straightness)
RESULTS["p4_straightness"] = round(straightness, 3)
```

Most of the length of a path is wiggle caused by the fresh noise z at every step (each step adds noise with a standard deviation of √β_t, which is large for the last steps of the schedule). The wiggle is not a mistake: it is the randomness that lets the same starting region end up in different blobs.

It prints:

```text
mean straightness: 0.084
```

   **What you should see.**

   The check passes two ticks. The path array has the shape (101, 1000, 2). The mean distance of the points from the origin is **1.27** at t = 100 (the mean length of a 2-D standard normal vector is 1.25), **1.44** at t = 50 and **3.02** at t = 0 (the blobs lie on a circle of radius 3). The mean straightness is **0.084**: the length of the straight line from start to end is only about 8 percent of the length of the path (tolerance: between 0.05 and 0.15). In the left panel the 20 paths wander and finally settle in blobs on the ring; the five snapshots show a round cloud at t = 100, a smaller, ring-like cloud at t = 50 and eight clusters at t = 0.

   *What it means.* The points do not travel along straight lines from noise to data. Each step takes a small drift towards the data (from the network) plus a random kick (the fresh noise z) of size √β_t, which is large in the late steps of our schedule (β_100 = 0.2). The cloud grows only slowly at first (mean distance from the origin 1.27 at t = 100 and 1.44 at t = 50) and in the last 50 steps the points move out to the ring at distance 3. This is also why DDIM (Part 6), which has no random kick, can take much longer jumps.

6. **Step 4.6. Fewer training steps.** How much training does the model need? Because we saved the weights after 500 and 1,500 steps of the same run, we can sample from all three versions with the same starting noise and compare. We use the two numbers of `modes_covered`: the count of blobs reached and the share of points on a blob. Counting blobs alone would be too generous, since a cloud of points spread everywhere also "reaches" every blob; the second number tells you how many points really sit on one.

```python
rows46 = []
net_ck = MLPDenoiser()
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a_, st in zip(ax, CK_MLP):
    net_ck.load_state_dict(saved42[st]); net_ck.eval()
    pts = sample_ddpm(lambda x, t: net_ck(x, t), (2000, 2), S, seed=0).numpy()
    c_, q_ = modes_covered(pts, centres); l_ = float(np.mean(losses42[max(0, st - 100):st]))
    rows46.append((st, l_, c_, q_))
    a_.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5); a_.scatter(centres[:, 0], centres[:, 1], c="r", marker="x"); a_.set_title("%d steps" % st); a_.set_aspect("equal"); a_.set_xlim(-4.5, 4.5); a_.set_ylim(-4.5, 4.5)
plt.show()
print("steps   loss (last 100)   blobs covered   share on a blob")
for st, l_, c_, q_ in rows46: print("%5d   %15.3f   %13d   %15.3f" % (st, l_, c_, q_))
RESULTS["p4_on_blob_by_steps"] = [round(r[3], 3) for r in rows46]
```

It prints:

```text
steps   loss (last 100)   blobs covered   share on a blob
  500             0.320               8             0.307
 1500             0.307               8             0.379
 4000             0.255               8             0.917
```

   *Exercise 4.6:* From `rows46` find the first checkpoint at which the share on a blob is above 0.6, and store its step number in `first_good` (use `None` if there is none). Then compute `loss_drop`, the loss of the first row minus the loss of the last row. Why it matters: it shows that the loss and the quality of the samples do not move together evenly: the loss changes little in the second half of training while the samples change a lot.

```python
first_good = "unset"   # YOUR CODE: the step of the first row of rows46 whose share on a blob (the last entry of the row) is above 0.6, or None
loss_drop = None       # YOUR CODE: rows46[0][1] - rows46[-1][1]

if first_good == "unset" or loss_drop is None:
    print("Fill in first_good and loss_drop above, then run this cell again.")
else:
    check("first_good is a checkpoint step or None", first_good is None or first_good in CK_MLP, "it must be one of %s" % (CK_MLP,))
    check("loss_drop is the difference of the loss column", abs(loss_drop - (rows46[0][1] - rows46[-1][1])) < 1e-9, "rows46[0][1] - rows46[-1][1]")
    if not QUICK: check("the first good checkpoint is the last one in our run", first_good == CK_MLP[-1], "look at the share on a blob in the last column")
```

   A possible answer (try first, then compare):

```python
first_good = next((r[0] for r in rows46 if r[3] > 0.6), None)
loss_drop = rows46[0][1] - rows46[-1][1]
print("first checkpoint with more than 60 percent of the points on a blob:", first_good, "  loss drop: %.3f" % loss_drop)
```

`next(..., None)` takes the first row that satisfies the condition. The loss falls by only a little between the first and the last checkpoint, but the share of points on a blob rises from about a third to over 90 percent: the small differences in the noise guess are exactly what separates a blurry cloud from sharp blobs.

It prints:

```text
first checkpoint with more than 60 percent of the points on a blob: 4000   loss drop: 0.065
```

   **What you should see.**

   The scatter plots show the same starting noise with three versions of the network. The table: after **500** steps the loss over the last 100 steps is 0.320, all 8 blobs are "covered" but only **0.307** of the points are on a blob; after **1,500** steps the loss is 0.307 and the share is **0.379**; after **4,000** steps the loss is 0.255 and the share is **0.917**. The Check passes two ticks and, in a full run, the third (first_good is 4000); the answer prints `4000` and a loss drop of 0.065.

   *What it means.* The loss moved by 0.065 (from 0.320 to 0.255, about 20 percent) while the share of points on a blob tripled. The early versions know roughly where the data are (a ring) but not yet where the eight blobs lie: the points land on the ring, between blobs, and `modes_covered` still counts all 8 blobs since each blob receives the minimum 2 percent of the points. This is why we always report two numbers: the blob count alone would have said "8 of 8" for a network that has barely learned anything. (In this run the last 2,500 steps made the difference; a shortened run with 1,500 steps would give a poor picture of what the method can do.)

   Tips and common mistakes:

   - The checkpoints belong to one run (seed 42), so the three rows are comparable; do not compare them with the seed-1 run.
   - With `QUICK` (a 600-step run) the checkpoints are at 75, 225 and 600 steps and the third Check is skipped.

7. **Step 4.7. Imagine and write: easy and hard noise levels.** Which part of the denoiser's job is easy and which is hard? At large t (almost pure noise) the best guess is simple: x_t is nearly all noise, so ε̂ is about x_t. At small t (a little noise on a point that is almost clean) the network must tell the tiny noise apart from the fine structure of the data. We measure it: the loss of the trained network in five groups of t (1 to 20, 21 to 40, and so on), always with the same 20,000 evaluation pairs. The function `loss_by_bin` is reused in Part 5.

```python
def loss_by_bin(model, S, data, bins=5, n=20000, seed=123):
    """Average noise-prediction loss of a model in `bins` groups of step numbers (equal-sized groups of 1..T), on fixed evaluation pairs."""
    g = torch.Generator().manual_seed(seed)
    x0 = data[torch.randint(0, len(data), (n,), generator=g)]
    t = torch.randint(1, S["T"] + 1, (n,), generator=g); eps = torch.randn(x0.shape, generator=g)
    with torch.no_grad(): err = ((model(q_sample_t(x0, t, eps, S), t) - eps) ** 2).mean(1)
    edges = np.linspace(0, S["T"], bins + 1).astype(int)
    return [float(err[(t > lo) & (t <= hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]

lb42 = loss_by_bin(model42, S, data_mix)
print("t group        :", "  ".join("%3d-%3d" % (lo + 1, hi) for lo, hi in zip(range(0, 100, 20), range(20, 101, 20))))
print("loss (trained) :", "  ".join("%7.3f" % v for v in lb42))
RESULTS["p4_loss_by_bin"] = [round(v, 3) for v in lb42]
```

It prints:

```text
t group        :   1- 20   21- 40   41- 60   61- 80   81-100
loss (trained) :   0.416    0.562    0.271    0.034    0.003
```

   *Exercise 4.7:* Write two or three sentences in `answer_4_7`: in which group of t is the loss highest, in which lowest, and why, in your own words? Use the table above and what you know about the noise at those levels (Parts 1 and 2). Then say what this suggests for how much time a training run should spend on each noise level.

```python
answer_4_7 = ""   # YOUR TEXT HERE

if answer_4_7.strip() == "":
    print("Write your answer in answer_4_7 above, then run this cell again.")
else:
    hi_, lo_ = int(np.argmax(lb42)), int(np.argmin(lb42))
    print("Your answer has %d words. In our run the highest loss is in group %d (t = %d-%d) and the lowest in group %d (t = %d-%d)." % (len(answer_4_7.split()), hi_ + 1, 20 * hi_ + 1, 20 * hi_ + 20, lo_ + 1, 20 * lo_ + 1, 20 * lo_ + 20))
    check("you wrote at least 15 words", len(answer_4_7.split()) >= 15, "two or three sentences")
```

   **What you should see.**

   The table: in the five groups of t the loss is **0.416** (t from 1 to 20), **0.562** (21 to 40), **0.271** (41 to 60), **0.034** (61 to 80) and **0.003** (81 to 100). The cell prints the same table for you; with a different run the numbers differ in the third digit. The Check only counts your words and prints where the highest and lowest group lie in your run.

   *What it means.* The easy part is the late half of the noising process (t above 60): the point is nearly all noise, ε̂ is nearly x_t, and the loss is almost zero (0.003 above t = 80). The hardest groups are the early and middle steps, with the maximum between t = 21 and t = 40, where ᾱ_t falls from 0.64 to 0.19 (at t = 27 the noise already carries as much power as the signal). There the network must decide, from a noisy point, which of the eight blobs the point came from, and how far it is from there. At very small t (1 to 20) there is little noise but the blobs are narrow (spread 0.15), so the exact blob position still matters. The consequence for a training run is that equal numbers of examples for every t (our choice) spend a lot of the work on t above 60, where there is little to learn; some models draw the noise levels non-uniformly to focus on the hard region (this is the topic of the schedule comparison of Part 2).

   A possible answer (try first, then compare):

```python
answer_4_7 = "The loss is highest for t from 21 to 40 (0.56) and lowest above t = 80 (0.003). Late in the process x_t is almost only noise, so predicting epsilon as x_t is easy; in the middle the network must work out which blob the point came from. So a run should not spend most of its effort on large t."
print(answer_4_7)
```

The highest loss is in the middle of the noising process and the lowest at the end, where the guess "all of x_t is noise" is nearly right. A model gains little from many examples with t above 80.

It prints:

```text
The loss is highest for t from 21 to 40 (0.56) and lowest above t = 80 (0.003). Late in the process x_t is almost only noise, so predicting epsilon as x_t is easy; in the middle the network must work out which blob the point came from. So a run should not spend most of its effort on large t.
```

### Part 5. Time embeddings, the UNet and its parts (30 min)

*Lesson section: "UNet".* The denoiser of Part 4 was a small MLP on two numbers. Pictures have 256 numbers (even our tiny 16 × 16 ones) and the same network layout would be wasteful, so image diffusion models use a **UNet**: a convolutional network that shrinks the picture, works at several resolutions, grows it back, and receives the step number everywhere. ==You will compute the time embedding by hand and look at it, remove it from the 2-D denoiser to see what it was doing, write the building block that injects the time into a convolution, assemble a small UNet and count its parameters part by part, test what the skip connections do, and compute how much of the picture a neuron can see.== The UNet you build here has exactly 135,713 parameters, the number in the lesson's `unet` figure, and is the network of Part 6. This part continues in the same session as Part 4 (it uses `S`, `train_mlp`, `time_embed`, `loss_by_bin`, `model42` and the data of Part 4).

**Imagine first.** In Part 4 the network was told the step number t through a time embedding. Suppose we take that away: the network gets only the noisy point x_t and must guess the noise, without being told whether it is looking at a little noise (t = 5) or almost only noise (t = 90). Is the job impossible, harder but possible, or no different? Think about what x_t itself reveals about t. Write your guess in the cell below. You compare it with Step 5.2.

```python
# Imagine first: what happens to the denoiser when it is not told the step number?
my_guess_notime = None    # for example "it fails completely", "a bit worse" or "no different": write one of them
```

1. **Step 5.1. The time embedding by hand.** You wrote `time_embed` in Step 4.2. Here we compute it by hand for a tiny size (dim 4), to see exactly what the network receives. With dim 4 we have half = 2 frequencies: f_0 = exp(0) = 1 and f_1 = exp(−ln(10000)/2) = 0.01. Our convention is **first the sines, then the cosines**, so the embedding of step t is [sin(t), sin(0.01·t), cos(t), cos(0.01·t)]. (Some libraries interleave sine and cosine; the network does not care, as long as the convention stays the same during training and sampling.) For t = 1 this is [0.8415, 0.0100, 0.5403, 1.0000]: the fast wave has already turned by one radian, the slow wave has hardly moved.

   *Exercise 5.1:* Write `embed_by_hand(t, dim)` with plain Python (the `math` module, no PyTorch) that returns a list of `dim` numbers in our convention: first the `dim // 2` sines then the `dim // 2` cosines of t·f_k, with f_k = exp(−ln(10000)·k/(dim // 2)). Why it matters: if you can compute the embedding with a calculator, you understand what the network is given; the check compares your list with `time_embed` for t = 1, 100 and 300.

```python
def embed_by_hand(t, dim=4):
    """The sinusoidal embedding of one step number t as a plain Python list."""
    return None   # YOUR CODE: half = dim // 2; f = [math.exp(-math.log(10000) * k / half) for k in range(half)]; sines first, then cosines

if embed_by_hand(1) is None:
    print("Fill in embed_by_hand above, then run this cell again.")
else:
    for tt in (1, 100, 300):
        mine = np.array(embed_by_hand(tt, 4)); ref = time_embed(torch.tensor([tt]), 4)[0].numpy()
        check("t = %3d: %s" % (tt, np.round(mine, 4).tolist()), mine.shape == (4,) and np.allclose(mine, ref, atol=1e-4), "sines first: sin(t), sin(0.01 t), then cosines: cos(t), cos(0.01 t); the reference is %s" % np.round(ref, 4).tolist())
```

   A possible answer (try first, then compare):

```python
def embed_by_hand(t, dim=4):
    half = dim // 2
    f = [math.exp(-math.log(10000) * k / half) for k in range(half)]      # the frequencies, from 1 down to about 1/10000
    return [math.sin(t * fk) for fk in f] + [math.cos(t * fk) for fk in f]
for tt in (1, 100, 300):
    print("t = %3d:" % tt, [round(v, 4) for v in embed_by_hand(tt, 4)])
```

For t = 100 the fast wave sin(100) is far from the one of t = 101 (the wave has turned by a full radian between neighbours) while the slow wave sin(1) hardly differs: the pair together tells the network both "roughly where in the schedule" (slow waves) and "exactly which step" (fast waves).

It prints:

```text
t =   1: [0.8415, 0.01, 0.5403, 1.0]
t = 100: [-0.5064, 0.8415, 0.8623, 0.5403]
t = 300: [-0.9998, 0.1411, -0.0221, -0.99]
```

Now look at the whole embedding for t = 0 to 200 with dim 32 as a heat map: every row is one step, every column one wave. Fast columns (left, and the matching cosine columns) form a fine pattern, slow columns change gradually. Then compare embeddings of neighbouring and distant steps with the **cosine similarity** cos(a, b) = a·b/(|a||b|): 1 means the same direction, 0 unrelated.

```python
E = time_embed(torch.arange(0, 201), 32).numpy()
plt.figure(figsize=(7, 3.2)); plt.imshow(E.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1); plt.colorbar(label="value"); plt.xlabel("step t"); plt.ylabel("column (16 sines, then 16 cosines)"); plt.title("sinusoidal time embedding, dim 32"); plt.show()
def cos_sim(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
for other in (101, 110, 200):
    print("cosine similarity of the embeddings of t = 100 and t = %3d: %.3f" % (other, cos_sim(E[100], E[other])))
RESULTS["p5_cos_100_101"] = round(cos_sim(E[100], E[101]), 3); RESULTS["p5_cos_100_200"] = round(cos_sim(E[100], E[200]), 3)
```

It prints:

```text
cosine similarity of the embeddings of t = 100 and t = 101: 0.957
cosine similarity of the embeddings of t = 100 and t = 110: 0.629
cosine similarity of the embeddings of t = 100 and t = 200: 0.583
```

   **What you should see.**

   The check passes three ticks (one per step number). The embeddings are, in our convention [sin t, sin 0.01·t, cos t, cos 0.01·t]: for t = 1 **[0.8415, 0.0100, 0.5403, 1.0000]**, for t = 100 **[−0.5064, 0.8415, 0.8623, 0.5403]** and for t = 300 **[−0.9998, 0.1411, −0.0221, −0.9900]**. In the heat map (dim 32, t from 0 to 200) the first columns (the fast waves) form fine stripes, the last columns of each half change so slowly that they look like broad colour bands. The cosine similarity of the embeddings of t = 100 with t = 101 is **0.957**, with t = 110 it is **0.629** and with t = 200 it is **0.583** (tolerance: ±0.02).

   *What it means.* Neighbouring steps have very similar embeddings (0.957): the network gets a smooth signal and treats step 101 almost like step 100. Ten steps apart the similarity is already down to 0.63, so the network can tell them apart, but note that t = 200 is not much less similar to t = 100 (0.58) than t = 110 is: the slow waves are nearly the same, and the information that separates them is in the fast waves. A network that reads all columns gets a unique "fingerprint" for every step.

   Tips and common mistakes:

   - The slowest frequency in a size-32 embedding is exp(−ln(10000)·15/16) = 0.00018: it is practically constant over 100 steps. Larger T needs the same formula, which is why it was designed this way.

2. **Step 5.2. Ablation: no time input.** We now retrain the 2-D denoiser of Part 4 without the time embedding (`use_time=False`: the network sees only the two coordinates of x_t), with the same 4,000 steps, the same data and the same seed, and compare the final loss, the blobs covered and the share of points on a blob. We also compare the loss in the five groups of t of Step 4.7. A fair expectation is that the network will still learn *something*, because x_t reveals the noise level through its distance from the blobs (a point far outside the ring must have had a lot of noise), but only roughly.

   *Exercise 5.2:* Train a no-time denoiser with `train_mlp(seed=42, use_time=False)`, sample 2,000 points with `sample_ddpm` (seed 0, as in Step 4.4) and store the mean of its last 200 losses in `loss_no_time`, and the two numbers returned by `modes_covered` in `blobs_no_time` and `on_blob_no_time`. Why it matters: an ablation (removing one part and measuring) is how you find out which parts of a design earn their place. The check compares with the network of Part 4.

```python
loss_no_time = None; blobs_no_time = None; on_blob_no_time = None    # YOUR CODE: three numbers from a no-time run (see the exercise)

if loss_no_time is None:
    print("Fill in the three numbers above, then run this cell again.")
else:
    check("the loss of the no-time model is above the loss of the model with time", loss_no_time > float(np.mean(losses42[-200:])) + 0.03, "expected %.3f for the model with time; no-time should be clearly higher" % float(np.mean(losses42[-200:])))
    check("blobs_no_time is a whole number from 0 to 8", blobs_no_time in range(0, 9), "the first value returned by modes_covered")
    check("on_blob_no_time is a share between 0 and 1", 0 <= on_blob_no_time <= 1, "the second value returned by modes_covered")
```

   A possible answer (try first, then compare):

```python
model_nt, losses_nt, _ = train_mlp(seed=42, use_time=False)
model_nt.eval()
gen_nt = sample_ddpm(lambda x, t: model_nt(x, t), (2000, 2), S, seed=0).numpy()
loss_no_time = float(np.mean(losses_nt[-200:]))
blobs_no_time, on_blob_no_time = modes_covered(gen_nt, centres)
print("no time input: loss %.3f, %d of 8 blobs, %.3f of the points on a blob" % (loss_no_time, blobs_no_time, on_blob_no_time))
RESULTS["p5_notime_loss"] = round(loss_no_time, 3); RESULTS["p5_notime_on_blob"] = round(on_blob_no_time, 3)
```

The loss is much higher (it stays above 0.5 where the network with time goes down to about 0.25) and far fewer points land on a blob. The network could tell the noise level only roughly from the position of x_t.

It prints:

```text
no time input: loss 0.543, 8 of 8 blobs, 0.363 of the points on a blob
```

The table for both seeds and both networks, with the loss per group of t (Step 4.7). The second seed needs two more training runs of a few seconds each.

```python
model_nt1, losses_nt1, _ = train_mlp(seed=1, use_time=False); model_nt1.eval()
gen_nt1 = sample_ddpm(lambda x, t: model_nt1(x, t), (2000, 2), S, seed=0).numpy()
cov_nt1, q_nt1 = modes_covered(gen_nt1, centres)
rows52 = [("with time, s42", model42, losses42, cov_g, q_g), ("no time,   s42", model_nt, losses_nt, blobs_no_time, on_blob_no_time),
          ("with time, s1 ", model1, losses1, cov_1, q_1), ("no time,   s1 ", model_nt1, losses_nt1, cov_nt1, q_nt1)]
print("model            final loss  blobs  on blob |  loss in groups of t: 1-20  21-40  41-60  61-80  81-100")
for name, m_, l_, c_, q_ in rows52:
    lb = loss_by_bin(m_, S, data_mix)
    print("%-16s %10.3f  %5d  %7.3f |  %s" % (name, np.mean(l_[-200:]), c_, q_, "   ".join("%5.3f" % v for v in lb)))
RESULTS["p5_notime_on_blob_s1"] = round(q_nt1, 3)
```

It prints:

```text
model            final loss  blobs  on blob |  loss in groups of t: 1-20  21-40  41-60  61-80  81-100
with time, s42        0.254      8    0.917 |  0.416   0.562   0.271   0.034   0.003
no time,   s42        0.543      8    0.363 |  0.945   0.985   0.403   0.226   0.218
with time, s1         0.261      8    0.912 |  0.413   0.564   0.271   0.035   0.006
no time,   s1         0.556      8    0.412 |  0.916   0.963   0.413   0.244   0.239
```

   **What you should see.**

   The check passes three ticks. The no-time network ends with a loss of **0.543** (seed 42; **0.556** for seed 1) against **0.254** (**0.261**) for the network with time, covers **8 of 8** blobs but puts only **0.363** (seed 1: **0.412**) of its points on a blob, against **0.917** (**0.912**). The table with the loss in five groups of t: with time (seed 42) 0.416, 0.562, 0.271, 0.034, 0.003; without time 0.945, 0.985, 0.403, 0.226, 0.218; for seed 1 the numbers are 0.413, 0.564, 0.271, 0.035, 0.006 (with time) and 0.916, 0.963, 0.413, 0.244, 0.239 (without time). Tolerance: the no-time loss is between 0.45 and 0.65 and the share on a blob between 0.25 and 0.5.

   *What it means.* Compare with your guess: the job was **not impossible**, but much worse. The no-time network still learns something, as we expected: it finds all eight blobs, because x_t reveals roughly how far it is from a blob. But in the late steps (t above 60, the groups 4 and 5) its loss is 0.22 to 0.24 where the network with time has 0.03 and 0.003, a factor of 7 to 70. At t = 100 and t = 70 a noisy point looks almost the same (a standard normal cloud), so without t the network cannot know whether to trust the position of x_t (t = 60: ᾱ = 0.02, some signal left) or to ignore it (t = 100: nothing left); it learns a compromise that is wrong at both. In the early groups the loss without time is 0.95 to 0.99 (nearly the value of "always guess 0": 1.0) against 0.42 to 0.56. The lesson: **the network needs to be told the noise level**, even though a good part of it can be guessed from the data.

   Tips and common mistakes:

   - The training and the sampling must use the same switch: a no-time network is also sampled without time (the model object remembers `use_time`).

3. **Step 5.3. Where the time goes in: the TimeBlock.** In a UNet the step number must reach every resolution, not only the input. The standard way is a **block** with two 3 × 3 convolutions and a time projection between them: the embedding (32 numbers) goes through a linear layer that makes one number per channel, and this number is *added* to every pixel of that channel after the first convolution. So the block can behave differently for different noise levels. The formula for its parameters is 9·c_in·c_out + c_out (conv 1) + time_dim·c_out + c_out (the projection) + 9·c_out·c_out + c_out (conv 2).

   *Exercise 5.3:* Write the class `TimeBlock_mine(c_in, c_out, time_dim)`. Its `forward(x, emb)` does: `h = conv1(x)` (3 × 3, padding 1), then `h = h + proj(emb)[:, :, None, None]` (`proj` is a `Linear(time_dim, c_out)`; the two `None` axes let one number per channel be added to every pixel), then `h = silu(h)`, then `silu(conv2(h))` with a second 3 × 3 convolution with `c_out` channels. If `emb` is `None`, skip the addition (Step 6.7 needs this). Why it matters: this block is the repeated unit of the UNet. The check counts the parameters against the formula and tests the shapes.

```python
class TimeBlock_mine(nn.Module):
    def __init__(self, c_in, c_out, time_dim=32):
        super().__init__()
        # YOUR CODE: self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1); self.proj = nn.Linear(time_dim, c_out); self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
    def forward(self, x, emb=None):
        return x   # YOUR CODE: conv1, then add the projected embedding (unless emb is None), SiLU, conv2, SiLU

def timeblock_params(c_in, c_out, time_dim=32):
    return 9 * c_in * c_out + c_out + time_dim * c_out + c_out + 9 * c_out * c_out + c_out
tb_ = TimeBlock_mine(1, 16, 32)
check("the parameters of TimeBlock(1, 16) are %d by the formula" % timeblock_params(1, 16), count_params(tb_) == timeblock_params(1, 16), "you have %d; conv1 3x3, a Linear(32, 16) and conv2 3x3 are needed" % count_params(tb_))
try:
    y_ = tb_(torch.randn(2, 1, 16, 16), torch.randn(2, 32))
    check("the output has shape (2, 16, 16, 16)", tuple(y_.shape) == (2, 16, 16, 16), "padding=1 keeps the size, c_out channels")
except Exception as err:
    print("✗ the block does not run yet:", err)
```

   A possible answer (try first, then compare):

```python
class TimeBlock(nn.Module):
    """conv 3x3 -> add a projection of the time embedding -> SiLU -> conv 3x3 -> SiLU"""
    def __init__(self, c_in, c_out, time_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.proj = nn.Linear(time_dim, c_out)                # one number per channel
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
    def forward(self, x, emb=None):
        h = self.conv1(x)
        if emb is not None: h = h + self.proj(emb)[:, :, None, None]     # the same number is added at every pixel of a channel
        return F.silu(self.conv2(F.silu(h)))
```

The projection is how the noise level enters: a different embedding gives a different number per channel, so the block can switch its behaviour (smooth strongly at high noise, keep detail at low noise).

The five blocks of the UNet you build next are TimeBlock(1, 16), (16, 32), (32, 64), (64, 32) and (32, 16). We check the formula against `count_params` for all five and look at the shapes.

```python
def timeblock_params(c_in, c_out, time_dim=32):
    return 9 * c_in * c_out + c_out + time_dim * c_out + c_out + 9 * c_out * c_out + c_out
print("block (c_in, c_out)   by the formula   count_params")
for ci, co in ((1, 16), (16, 32), (32, 64), (64, 32), (32, 16)):
    b_ = TimeBlock(ci, co, 32)
    print("TimeBlock(%2d, %2d)   %14d   %12d" % (ci, co, timeblock_params(ci, co), count_params(b_)))
    assert count_params(b_) == timeblock_params(ci, co)
x_in = torch.randn(4, 16, 8, 8); emb_in = torch.randn(4, 32)
print("TimeBlock(16, 32) on a (4, 16, 8, 8) batch ->", tuple(TimeBlock(16, 32)(x_in, emb_in).shape), "  without time ->", tuple(TimeBlock(16, 32)(x_in).shape))
```

It prints:

```text
block (c_in, c_out)   by the formula   count_params
TimeBlock( 1, 16)             3008           3008
TimeBlock(16, 32)            14944          14944
TimeBlock(32, 64)            57536          57536
TimeBlock(64, 32)            28768          28768
TimeBlock(32, 16)             7472           7472
TimeBlock(16, 32) on a (4, 16, 8, 8) batch -> (4, 32, 8, 8)   without time -> (4, 32, 8, 8)
```

   **What you should see.**

   The first block of the check prints two crosses until you fill in the class: with the starter (no layers) the parameter count is 0 and the shape check fails; with your class both ticks appear. The table for the five blocks of the UNet: TimeBlock(1, 16) has **3,008** parameters by the formula and by `count_params`; TimeBlock(16, 32) **14,944**; TimeBlock(32, 64) **57,536**; TimeBlock(64, 32) **28,768**; TimeBlock(32, 16) **7,472**. A (4, 16, 8, 8) batch goes through TimeBlock(16, 32) and gives (4, 32, 8, 8), with and without the time input.

   *What it means.* The formula and the code agree, so you know where every number sits. For TimeBlock(1, 16): conv 1 has 9 × 1 × 16 + 16 = 160, the projection 32 × 16 + 16 = 528, conv 2 has 9 × 16 × 16 + 16 = 2,320, and 160 + 528 + 2,320 = 3,008. The time projection is the small part: in TimeBlock(32, 64) it costs 2,112 of 57,536 parameters (3.7 percent).

   Tips and common mistakes:

   - `proj(emb)` has shape (B, c_out); `[:, :, None, None]` makes it (B, c_out, 1, 1) so that it is added to every pixel. Without the two `None` axes PyTorch tries to add a (B, c_out) tensor to a (B, c_out, H, W) tensor and complains (or silently adds it to the wrong axis if H or W equals B).
   - Use `padding=1` with the 3 × 3 convolutions to keep the size.

4. **Step 5.4. The UNet by hand.** Now we assemble the blocks. The picture goes through three levels. **Down path:** a TimeBlock at 16 × 16 (16 channels), a 3 × 3 convolution with stride 2 that halves the size (8 × 8), a TimeBlock with 32 channels, another stride-2 convolution (4 × 4). **Bottleneck:** a TimeBlock with 64 channels at 4 × 4. **Up path:** a 2 × 2 transpose convolution with stride 2 (Week 12) doubles the size to 8 × 8, the result is **concatenated** with the 32-channel features saved on the way down (the skip connection; channels 32 + 32 = 64) and a TimeBlock brings it back to 32 channels; the same again for 16 × 16 with 16 + 16 = 32 channels in and 16 out. A final 1 × 1 convolution makes the output picture with one channel, the guess ε̂ of the noise. The step number goes through the sinusoidal embedding (size 32) and a small two-layer MLP, and the result is given to every block. The two switches `use_time` and `use_skips` are for Step 6.7, and `n_classes` is for Step 6.8; with their default values you get the plain network.

```python
class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, base=16, time_dim=32, use_time=True, use_skips=True, n_classes=0):
        super().__init__()
        c1, c2, c3 = base, 2 * base, 4 * base
        self.time_dim, self.use_time, self.use_skips = time_dim, use_time, use_skips
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.down1 = TimeBlock(in_ch, c1, time_dim); self.pool1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)      # 16 x 16 -> 8 x 8
        self.down2 = TimeBlock(c1, c2, time_dim);    self.pool2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)      # 8 x 8 -> 4 x 4
        self.mid = TimeBlock(c2, c3, time_dim)                                                                   # the bottleneck, 4 x 4
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2); self.dec2 = TimeBlock(2 * c2, c2, time_dim)        # 4 x 4 -> 8 x 8, then merge the skip
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2); self.dec1 = TimeBlock(2 * c1, c1, time_dim)        # 8 x 8 -> 16 x 16, then merge the skip
        self.out = nn.Conv2d(c1, in_ch, 1)                                                                       # 1 x 1 convolution: the noise guess
        if n_classes: self.label_emb = nn.Embedding(n_classes + 1, time_dim)     # Step 6.8 only: one vector per class plus one for "no label"
    def forward(self, x, t, y=None, trace=None):
        emb = self.time_mlp(time_embed(t, self.time_dim)) if self.use_time else None
        if y is not None: emb = (emb if emb is not None else 0) + self.label_emb(y)
        d1 = self.down1(x, emb); d2 = self.down2(self.pool1(d1), emb); m = self.mid(self.pool2(d2), emb)
        s2 = d2 if self.use_skips else torch.zeros_like(d2)            # the skip: features from the way down (or zeros, Steps 5.6 and 6.7)
        u2 = self.dec2(torch.cat([self.up2(m), s2], 1), emb)
        s1 = d1 if self.use_skips else torch.zeros_like(d1)
        u1 = self.dec1(torch.cat([self.up1(u2), s1], 1), emb)
        if trace is not None: trace.extend([("input", x), ("down1", d1), ("down2", d2), ("bottleneck", m), ("up2", u2), ("up1", u1)])
        return self.out(u1)
```

```python
torch.manual_seed(SEED)
unet0 = TinyUNet()
xb_, tb_ = torch.randn(8, 1, 16, 16), torch.randint(1, 101, (8,))
trace = []; yb_ = unet0(xb_, tb_, trace=trace)
for name, v in trace: print("%-11s -> %s" % (name, tuple(v.shape)))
print("output      ->", tuple(yb_.shape))
print("parameters  :", count_params(unet0))
assert yb_.shape == xb_.shape and count_params(unet0) == 135713
RESULTS["p5_unet_params"] = count_params(unet0)
```

It prints:

```text
input       -> (8, 1, 16, 16)
down1       -> (8, 16, 16, 16)
down2       -> (8, 32, 8, 8)
bottleneck  -> (8, 64, 4, 4)
up2         -> (8, 32, 8, 8)
up1         -> (8, 16, 16, 16)
output      -> (8, 1, 16, 16)
parameters  : 135713
```

   *Exercise 5.4:* Before looking at the trace, predict the shape (channels, height, width) of the features after each stage, for one picture, and fill in the dictionary. For example the input is (1, 16, 16). Why it matters: being able to predict shapes is how you find bugs in a UNet (a wrong size breaks the skip concatenation). The check compares each of your entries with the real trace.

```python
shape_guess = {"down1": None, "down2": None, "bottleneck": None, "up2": None, "up1": None}    # YOUR CODE: replace every None by a tuple (channels, height, width)

if all(v is None for v in shape_guess.values()):
    print("Fill in shape_guess above, then run this cell again.")
else:
    for name, v in trace:
        if name in shape_guess: check("%-10s %s" % (name, shape_guess[name]), shape_guess[name] == tuple(v.shape[1:]), "the real shape is %s: channels from the table in the step text, size 16, 8 or 4" % (tuple(v.shape[1:]),))
```

   A possible answer (try first, then compare):

```python
shape_guess = {"down1": (16, 16, 16),       # first block at full size, 16 channels
               "down2": (32, 8, 8),         # after the stride-2 convolution: half the size; the block doubles the channels
               "bottleneck": (64, 4, 4),    # half the size again, 64 channels
               "up2": (32, 8, 8),           # transpose convolution doubles the size; the block brings 64 channels (32 + 32 skip) back to 32
               "up1": (16, 16, 16)}         # back at full size with 16 channels, then the 1 x 1 convolution makes 1 channel
for name, v in trace:
    if name in shape_guess: assert shape_guess[name] == tuple(v.shape[1:])
print("all five shapes are right")
```

The size is halved twice on the way down (16, 8, 4) and doubled twice on the way up, so the skip features have the same height and width as the up-path features that they are glued to; only the channels differ (and double after the concatenation).

It prints:

```text
all five shapes are right
```

   **What you should see.**

   The trace prints `input -> (8, 1, 16, 16)`, `down1 -> (8, 16, 16, 16)`, `down2 -> (8, 32, 8, 8)`, `bottleneck -> (8, 64, 4, 4)`, `up2 -> (8, 32, 8, 8)`, `up1 -> (8, 16, 16, 16)` and `output -> (8, 1, 16, 16)`, and the total of **135,713** parameters (the assertion passes). Your five Check lines show a tick for each stage; the answer prints `all five shapes are right`.

   *What it means.* The output has exactly the shape of the input: the network maps a noisy picture to a noise picture of the same size, which is what the loss needs. The channels double on the way down (16, 32, 64) while the size halves, so each level holds fewer positions but richer features; the up path reverses this. The 135,713 parameters are the number in the lesson's `unet` figure: the blocks plus the strided convolutions, the transpose convolutions, the time MLP and the 1 × 1 output convolution.

   Tips and common mistakes:

   - If the input size is not divisible by 4 (for example 18 × 18) the up path is larger than the skip features and `torch.cat` fails; 16 is a good size.
   - The skip is concatenated **after** the transpose convolution: 32 channels from the up path plus 32 from the skip make the 64 inputs of `dec2`.

5. **Step 5.5. Parameters by part.** Where are the 135,713 numbers? We loop over the child modules of the UNet and count each. The lesson's figure lists the blocks only; here we also list the strided convolutions, the transpose convolutions and the output convolution, which the figure leaves out (they make up the remaining 21,873 numbers). The bottleneck is the biggest single part, though it works on the smallest picture: its 64 channels at 4 × 4 are cheap to compute but need many weights (9 × 32 × 64 and 9 × 64 × 64 in the two convolutions).

```python
tot = count_params(unet0)
print("%-10s %9s %7s" % ("part", "params", "share"))
for name, mod in unet0.named_children():
    n = count_params(mod)
    print("%-10s %9d %6.1f%%" % (name, n, 100 * n / tot))
print("%-10s %9d %6.1f%%" % ("total", tot, 100))
print("the five blocks together:", sum(count_params(getattr(unet0, k)) for k in ("down1", "down2", "mid", "dec2", "dec1")), "  (3,008 + 14,944 + 57,536 + 28,768 + 7,472)")
```

It prints:

```text
part          params   share
time_mlp        2112    1.6%
down1           3008    2.2%
pool1           2320    1.7%
down2          14944   11.0%
pool2           9248    6.8%
mid            57536   42.4%
up2             8224    6.1%
dec2           28768   21.2%
up1             2064    1.5%
dec1            7472    5.5%
out               17    0.0%
total         135713  100.0%
the five blocks together: 111728   (3,008 + 14,944 + 57,536 + 28,768 + 7,472)
```

   *Exercise 5.5:* Compute `share_mid`, the share of the parameters held by the bottleneck block (`unet0.mid`), as a number between 0 and 1, and `share_blocks`, the share held by all five TimeBlocks. Why it matters: it shows where a UNet's capacity sits: in the deepest, narrowest layers, which see the whole picture at low resolution.

```python
share_mid = None; share_blocks = None    # YOUR CODE: count_params(...) of the part divided by count_params(unet0)

if share_mid is None or share_blocks is None:
    print("Fill in share_mid and share_blocks above, then run this cell again.")
else:
    check("share_mid is 57,536 / 135,713 = 0.424", abs(share_mid - 57536 / 135713) < 0.002, "count_params(unet0.mid) / count_params(unet0)")
    check("share_blocks is 111,728 / 135,713 = 0.823", abs(share_blocks - 111728 / 135713) < 0.002, "add down1, down2, mid, dec2 and dec1")
```

   A possible answer (try first, then compare):

```python
share_mid = count_params(unet0.mid) / count_params(unet0)
share_blocks = sum(count_params(getattr(unet0, k)) for k in ("down1", "down2", "mid", "dec2", "dec1")) / count_params(unet0)
print("bottleneck %.3f   all five blocks %.3f" % (share_mid, share_blocks))
RESULTS["p5_share_mid"] = round(share_mid, 3)
```

The bottleneck holds 42 percent of all numbers although it handles only 4 × 4 positions: a convolution's cost in weights depends on the channels (9 × c_in × c_out), and the deepest layer has the most channels. The five blocks hold 82 percent; the rest are the strided and transpose convolutions, the time MLP and the output layer.

It prints:

```text
bottleneck 0.424   all five blocks 0.823
```

   **What you should see.**

   The table lists: time_mlp 2,112 (1.6%), down1 3,008 (2.2%), pool1 2,320 (1.7%), down2 14,944 (11.0%), pool2 9,248 (6.8%), **mid 57,536 (42.4%)**, up2 8,224 (6.1%), dec2 28,768 (21.2%), up1 2,064 (1.5%), dec1 7,472 (5.5%), out 17 (0.0%), total 135,713. The five blocks together have 111,728 parameters. The Check passes two ticks (0.424 and 0.823); the answer prints `bottleneck 0.424   all five blocks 0.823`.

   *What it means.* The bottleneck block has more than twice as many parameters as the next block (dec2, 28,768), although it works on the smallest picture. A convolution costs 9 × c_in × c_out weights regardless of the picture size, and the bottleneck has the most channels. Compute and weights are different things: the bottleneck uses few multiplications (16 positions) for its many weights, the first block uses many multiplications (256 positions) for few weights. Also notice the little output layer: 17 numbers (a 1 × 1 convolution from 16 channels to 1 channel).

6. **Step 5.6. Skip connections.** Why does the UNet carry features across (the "U")? On the way down the picture is squeezed to 4 × 4 positions; fine detail (where exactly a one-pixel line runs) is hard to carry through such a bottleneck. The skip hands the decoder the full-resolution features directly. A toy example shows the idea with the simplest possible "bottleneck": average the picture over 4 × 4 patches (leaving 16 numbers instead of 256) and expand it again. Then we test the real network: we replace the skip features by zeros in the **untrained** UNet and measure how much its output changes. (The same test on the **trained** network is Step 6.7; it tells you how much the trained network relies on the skips.)

   *Exercise 5.6:* Write `pool_up(img, k=4)`: average the picture over k × k patches with `F.avg_pool2d` and expand it again to the original size with `F.interpolate(..., scale_factor=k, mode="nearest")`. Why it matters: after this operation the picture can hold only (16 / k)² different values: that is what a decoder without a skip would have to work from if the bottleneck were one channel. The check uses a cross: pool_up(cross) must differ from the cross, and have at most 16 distinct values.

```python
def pool_up_mine(img, k=4):
    """Average over k x k patches, then expand back to the original size (nearest neighbour). img: tensor (B, C, H, W)."""
    return img   # YOUR CODE: F.interpolate(F.avg_pool2d(img, k), scale_factor=k, mode="nearest")
Xc, yc = make_shapes(5, noise=0.0, seed=0)
cross_img = torch.tensor(Xc[np.where(yc == 2)[0][:1] ])      # one clean cross, shape (1, 1, 16, 16)

r_ = pool_up_mine(cross_img, 4)
check("the output has the shape of the input", tuple(r_.shape) == tuple(cross_img.shape), "(1, 1, 16, 16)")
check("the cross is changed (a thin line does not survive averaging)", float((r_ - cross_img).abs().mean()) > 0.05, "this function must average and expand; right now it returns the picture itself")
check("at most 16 distinct values", len(torch.unique(r_)) <= 16, "a 4 x 4 grid of patches holds 16 numbers")
```

   A possible answer (try first, then compare):

```python
def pool_up(img, k=4):
    return F.interpolate(F.avg_pool2d(img, k), scale_factor=k, mode="nearest")
r = pool_up(cross_img, 4)
print("distinct values after pooling: %d (of %d pixels); mean absolute error: %.3f" % (len(torch.unique(r)), r.numel(), float((r - cross_img).abs().mean())))
```

Averaging smears a one-pixel line over its whole 4 × 4 patch. A skip connection hands over the original pixels, so the error of this toy goes to zero. In the real UNet the bottleneck has 64 channels (1,024 numbers, more than the 256 pixels), so the number of values is not the only limit, but the features there are built from large patches and do not know exactly where things are.

It prints:

```text
distinct values after pooling: 4 (of 256 pixels); mean absolute error: 0.092
```

The untrained UNet: how much does its output change when the skips are zeroed? We feed the same batch and the same step numbers through the network with and without skips (the switch `use_skips` only changes what is concatenated; the weights are the same).

```python
r_toy = pool_up(cross_img, 4)
print("toy: distinct values after pooling %d, mean absolute error %.3f" % (len(torch.unique(r_toy)), float((r_toy - cross_img).abs().mean())))
torch.manual_seed(SEED); u_skip = TinyUNet(use_skips=True)
u_zero = TinyUNet(use_skips=False); u_zero.load_state_dict(u_skip.state_dict())       # the same weights
with torch.no_grad():
    o1, o2 = u_skip(xb_, tb_), u_zero(xb_, tb_)
rel = float((o1 - o2).abs().mean() / o1.abs().mean())
print("untrained network: mean |output| %.4f, mean |change| %.4f, relative change %.2f" % (float(o1.abs().mean()), float((o1 - o2).abs().mean()), rel))
RESULTS["p5_skip_rel_change_untrained"] = round(rel, 2)
```

It prints:

```text
toy: distinct values after pooling 4, mean absolute error 0.092
untrained network: mean |output| 0.1950, mean |change| 0.0035, relative change 0.02
```

   **What you should see.**

   The Check shows three lines for the toy; with your function filled in they are ticks. The answer prints `distinct values after pooling: 4 (of 256 pixels); mean absolute error: 0.092`: the cross touches only a few of the 16 patches, so after pooling the picture has just 4 different values, and the lines have become faint grey squares. (The error is small on average because most pixels are background, but the lines, which are the only content, are destroyed.) For the untrained UNet the output changes only a little when its skips are zeroed: mean |output| 0.1950, mean |change| 0.0035, relative change **0.02**.

   *What it means.* Two different messages. The toy shows what a skip connection is **for**: the squeezed picture has lost where exactly the line ran. The second number looks as if the skips did not matter, but that is an artefact of the untrained network: with random weights the decoder blocks produce small outputs whatever they receive, and the skip is only one of two inputs, so the change is 2 percent. The question "does a trained network rely on its skips?" can only be asked of a trained network: Step 6.7 does it, and the answer is a relative change of 1.06 to 1.13: without its skips the trained network's output changes by as much as the output itself.

7. **Step 5.7. Receptive field.** How much of the picture can one output neuron see? The **receptive field** of a stack of convolutions grows with every layer: each layer with kernel k adds (k − 1) times the current **jump** (the distance in input pixels between neighbouring outputs), and a stride s multiplies the jump by s. Start with receptive field 1 and jump 1. Three 3 × 3 convolutions at full resolution give 1 + 2 + 2 + 2 = 7. Strides make it grow much faster: that is why a UNet sees the whole 16 × 16 picture in its bottleneck.

   *Exercise 5.7:* Write `receptive_field(layers)` for a list of (kernel, stride) pairs: keep `rf` (start 1) and `jump` (start 1); for each layer do `rf += (k - 1) * jump` and then `jump *= s`; return `rf`. Why it matters: it tells you whether the network can possibly use context far away, which the denoiser needs (a line's end tells you where the line is). The check uses three stacks of layers whose receptive fields you can compute by hand.

```python
def receptive_field_mine(layers):
    """layers: list of (kernel_size, stride). Returns the receptive field in input pixels."""
    return None   # YOUR CODE: rf = 1, jump = 1; for k, s in layers: rf += (k - 1) * jump; jump *= s

if receptive_field_mine([(3, 1)]) is None:
    print("Fill in receptive_field_mine above, then run this cell again.")
else:
    check("one 3x3 convolution: 3", receptive_field_mine([(3, 1)]) == 3, "rf = 1 + (3 - 1) * 1")
    check("three 3x3 convolutions at full resolution: 7", receptive_field_mine([(3, 1)] * 3) == 7, "1 + 2 + 2 + 2")
    check("3x3, then 3x3 with stride 2, then 3x3: 1 + 2 + 2 + 2 * 2 = 9", receptive_field_mine([(3, 1), (3, 2), (3, 1)]) == 9, "the jump doubles after the stride-2 layer, so the next 3x3 adds 2 * 2")
```

   A possible answer (try first, then compare):

```python
def receptive_field(layers):
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump        # a kernel of size k sees k - 1 more positions, each `jump` input pixels apart
        jump *= s                   # a stride s makes the outputs s times further apart
    return rf
print(receptive_field([(3, 1)] * 3), receptive_field([(3, 1), (3, 2), (3, 1)]))
```

The jump is the key: after a stride-2 layer neighbouring neurons are two input pixels apart, so the next 3 × 3 convolution reaches twice as far.

It prints:

```text
7 9
```

Now the encoder path of our UNet, in two versions: the five-layer sketch (3 × 3, stride 2, 3 × 3, stride 2, 3 × 3) and the real one, where every block has two convolutions: down1 (two convolutions), the stride-2 convolution, down2 (two), the second stride-2 convolution and the bottleneck block (two).

```python
sketch = [(3, 1), (3, 2), (3, 1), (3, 2), (3, 1)]
real = [(3, 1), (3, 1), (3, 2), (3, 1), (3, 1), (3, 2), (3, 1), (3, 1)]
for name, lay in (("three 3x3 at full resolution", [(3, 1)] * 3), ("the five-layer sketch", sketch), ("the real encoder of TinyUNet", real)):
    print("%-30s receptive field %2d of 16 pixels" % (name, receptive_field(lay)))
RESULTS["p5_rf_sketch"] = receptive_field(sketch); RESULTS["p5_rf_real"] = receptive_field(real)
```

It prints:

```text
three 3x3 at full resolution   receptive field  7 of 16 pixels
the five-layer sketch          receptive field 21 of 16 pixels
the real encoder of TinyUNet   receptive field 35 of 16 pixels
```

   **What you should see.**

   The check passes three ticks (3, 7 and 9 for the three stacks), and the answer prints `7 9`. For the encoder of our UNet: three 3 × 3 convolutions at full resolution have a receptive field of **7**; the five-layer sketch (3 × 3, stride 2, 3 × 3, stride 2, 3 × 3) has **21**; the real encoder with two convolutions per block has **35** pixels, against a picture that is only 16 pixels wide.

   *What it means.* A neuron in the bottleneck can see more than the whole picture (the borders add a zero padding that it also sees), so the bottleneck can use the full picture as context: the position and kind of the shape. At full resolution the receptive field is 5 for the two convolutions of `down1`: that part of the network sees only a small neighbourhood, which is right for "is there a line pixel here?". The skip connections let the decoder combine both: a wide context from the bottleneck and the exact positions from the encoder's first blocks.

### Part 6. A DDPM on tiny pictures (35 min)

*Lesson sections: "Training", "Sampling" and "UNet".* Now everything comes together. ==You will train the UNet of Part 5 from scratch on 2,000 noisy line shapes (16 × 16 pixels), generate pictures with the DDPM loop of Part 4, judge them with a classifier (the Week 12 idea), generate faster with the deterministic DDIM sampler, follow how the pictures improve during training, break the model on purpose in four ways, and, as an option, steer it with classifier-free guidance.== A UNet with 135,713 parameters learns to draw five kinds of shapes in about a minute and a half on a CPU. The training run of Step 6.2 is reused by Steps 6.3 to 6.7 (and Part 5.6 asked for its skip test here). All numbers are for `SEED = 42`, and the second seed is 1 for the key experiments. With `QUICK` (the test mode that the Setup cell can switch on) the training loops are much shorter and the pictures are poor; the text always describes the full run.

**Imagine first.** After 3,000 training steps the UNet draws shapes. But what does it draw after only 250 steps, when it has seen the equivalent of 8 passes over the data (250 × 64 pictures / 2,000 pictures)? Pick the best description: (a) pure noise, (b) smooth grey blurs without lines, (c) blurry strokes in the right places that are not yet clean shapes, (d) already clean shapes. Write the letter in the cell below. You compare it with Step 6.6.

```python
# Imagine first: what do the pictures look like after only 250 training steps?
my_guess_250 = None   # "a", "b", "c" or "d"
```

1. **Step 6.1. Data and schedule.** The training pictures are the five shapes of Week 4 on a 16 × 16 grid (`make_shapes`, 2,000 pictures with a little noise of 0.05, always at the centre). We bring them to the range [−1, 1] that the diffusion process expects (the noise has mean 0 and standard deviation 1, so the data should be on a matching scale): first clip to [0, 1] to remove the few pixels pushed outside by the noise, then 2·x − 1. The schedule is the one of Part 4 (T = 100, β from 0.001 to 0.2, ᾱ_100 = 2.0e-5), and `q_sample_t` already works for pictures: its `view` makes one ᾱ per picture. The cell noises one cross at six steps so that you can see what the network must undo.

```python
Xs_raw, ys_raw = make_shapes(2000, noise=0.05, seed=0)
data_img = torch.tensor(np.clip(Xs_raw, 0, 1) * 2 - 1)               # pictures in [-1, 1], shape (2000, 1, 16, 16)
labels_img = torch.tensor(ys_raw)
print("pictures:", tuple(data_img.shape), "  range [%.2f, %.2f]  mean %.3f" % (float(data_img.min()), float(data_img.max()), float(data_img.mean())))
assert data_img.min() >= -1 and data_img.max() <= 1 and abs(S["abar"][S["T"] ]) < 0.01          # range [-1, 1] and a schedule that ends in noise
cross_ = data_img[labels_img == 2][:1]
g_ = torch.Generator().manual_seed(0); eps_ = torch.randn(cross_.shape, generator=g_)
ts_show = [0, 5, 15, 30, 60, 100]
noised = torch.cat([q_sample_t(cross_, torch.tensor([t]), eps_, S) if t > 0 else cross_ for t in ts_show])
show_grid(noised.numpy(), titles=["t = %d, abar = %.3f" % (t, float(S["abar"][t])) for t in ts_show], cols=6)
```

It prints:

```text
pictures: (2000, 1, 16, 16)   range [-1.00, 1.00]  mean -0.846
```

   *Exercise 6.1:* Find `t_half`, the smallest step t for which ᾱ_t is below 0.5 (the signal-to-noise ratio ᾱ/(1 − ᾱ) falls below 1: more noise than picture), and `t_gone`, the smallest t with ᾱ_t below 0.01 (almost nothing of the picture is left). Use `S["abar"]` (index 0 is the unnoised picture; look only at t from 1 to T). Why it matters: a schedule that leaves the picture mostly intact for half the steps spends the network's effort on fine detail; one that destroys it quickly spends it on coarse structure. The check compares with the table printed above.

```python
t_half = None; t_gone = None    # YOUR CODE: for example int((S["abar"][1:] < 0.5).nonzero()[0]) + 1

if t_half is None or t_gone is None:
    print("Fill in t_half and t_gone above, then run this cell again.")
else:
    ab_ = S["abar"].numpy()
    check("t_half: abar_t is below 0.5 at t = %d but not at t = %d" % (t_half, t_half - 1), ab_[t_half] < 0.5 <= ab_[t_half - 1], "the first t with abar_t < 0.5")
    check("t_gone: abar_t is below 0.01 at t = %d but not at t = %d" % (t_gone, t_gone - 1), ab_[t_gone] < 0.01 <= ab_[t_gone - 1], "the first t with abar_t < 0.01")
```

   A possible answer (try first, then compare):

```python
t_half = int((S["abar"][1:] < 0.5).nonzero()[0]) + 1       # the first position (counting from step 1) where abar falls below 0.5
t_gone = int((S["abar"][1:] < 0.01).nonzero()[0]) + 1
print("abar < 0.5 from t = %d (SNR below 1), abar < 0.01 from t = %d, of T = %d" % (t_half, t_gone, S["T"]))
RESULTS["p6_t_half"] = t_half; RESULTS["p6_t_gone"] = t_gone
```

`nonzero()[0]` is the first index where the condition holds; we cut off the entry t = 0 with `[1:]` and add 1 to turn the position back into a step number.

It prints:

```text
abar < 0.5 from t = 27 (SNR below 1), abar < 0.01 from t = 67, of T = 100
```

   **What you should see.**

   The cell prints `pictures: (2000, 1, 16, 16)   range [-1.00, 1.00]  mean -0.846` (the mean is close to −1 because most pixels are background; a shape has 9 to 32 bright pixels out of 256). The six pictures of the cross show how the signal disappears: at t = 5 (ᾱ = 0.975) the cross is almost untouched, at t = 15 (ᾱ = 0.796) it is still clear but speckled, at t = 30 (ᾱ = 0.40) the lines are faint among the noise, at t = 60 (ᾱ = 0.02) you can only guess it, and at t = 100 it is pure noise. The check passes two ticks, and the answer prints `abar < 0.5 from t = 27 (SNR below 1), abar < 0.01 from t = 67, of T = 100`.

   *What it means.* The picture is "half gone" at t = 27 and practically gone at t = 67: about a third of our steps (68 to 100) are spent on pictures that contain essentially only noise. The network learns to place the shape coarsely between t = 100 and t = 60, to refine its edges around t = 30 and to clean up the background at the end. Compare this with Part 2: a cosine schedule would spread the work out more evenly.

   Tips and common mistakes:

   - `(S["abar"][1:] < 0.5).nonzero()[0]` is the first index in the shortened array: add 1 to get the step number, or the answer is off by one.

2. **Step 6.2. The model and the training loop.** The model is the `TinyUNet` of Part 5 (135,713 parameters, no change needed for 16 × 16 pictures). The loop is the one of Part 4 with three differences: pictures instead of points, batch 64 instead of 256, and a learning rate that starts at 1e-3 and decays smoothly to 0 following a cosine curve over the run (`CosineAnnealingLR`), which makes the last steps gentle and the final weights less noisy. The function also keeps copies of the weights at chosen steps (Step 6.6) and has two switches for Step 6.7 (`use_time`, `use_skips`) and one for Step 6.8 (`labels`, `n_classes`, `p_drop`: with labels, a fraction `p_drop` of the labels is replaced by the "no label" value, which is how one network learns both a conditional and an unconditional guess). It prints the average loss every 250 steps. The full run is 3,000 steps and takes about 75 seconds on our CPU; `QUICK` shortens it to 200.

```python
STEPS_UNET = 200 if QUICK else 3000
CK_UNET = (250, 1000) if not QUICK else (50, 100)

def train_unet(steps=STEPS_UNET, seed=42, use_time=True, use_skips=True, ckpt=(), labels=None, n_classes=0, p_drop=0.15, batch=64, lr=1e-3, log=250):
    """Train a TinyUNet to predict the noise in pictures. Returns the model, the loss list and a dict of weight copies at the steps in ckpt."""
    torch.manual_seed(seed)
    model = TinyUNet(use_time=use_time, use_skips=use_skips, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses, saved, t0 = [], {}, time.time()
    for s in range(1, steps + 1):
        idx = torch.randint(0, len(data_img), (batch,))
        xt, t, eps = make_batch(data_img[idx], S)
        y = None
        if labels is not None:
            y = labels[idx].clone(); y[torch.rand(batch) < p_drop] = n_classes            # the value n_classes means "no label"
        loss = F.mse_loss(model(xt, t, y), eps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); losses.append(float(loss))
        if s in ckpt: saved[s] = {k: v.clone() for k, v in model.state_dict().items()}
        if log and s % log == 0: print("step %4d   loss (last %d steps) %.4f   %.0f s" % (s, log, np.mean(losses[-log:]), time.time() - t0))
    model.eval()
    return model, losses, saved
```

```python
t0 = time.time()
unet42, losses_u42, saved_u42 = train_unet(seed=42, ckpt=CK_UNET)
time_unet = time.time() - t0
print("training took %.0f seconds for %d steps (%.0f ms per step)" % (time_unet, STEPS_UNET, 1000 * time_unet / STEPS_UNET))
RESULTS["p6_final_loss"] = round(float(np.mean(losses_u42[-100:])), 4); RESULTS["p6_train_seconds"] = round(time_unet)
plt.figure(figsize=(6, 3)); plt.plot(np.convolve(losses_u42, np.ones(25) / 25, mode="valid")); plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("noise-prediction loss"); plt.title("UNet training loss (average of 25 steps)"); plt.show()
```

It prints:

```text
step  250   loss (last 250 steps) 0.2670   5 s
step  500   loss (last 250 steps) 0.0642   11 s
step  750   loss (last 250 steps) 0.0430   17 s
step 1000   loss (last 250 steps) 0.0358   22 s
step 1250   loss (last 250 steps) 0.0316   28 s
step 1500   loss (last 250 steps) 0.0309   34 s
step 1750   loss (last 250 steps) 0.0283   39 s
step 2000   loss (last 250 steps) 0.0276   45 s
step 2250   loss (last 250 steps) 0.0262   51 s
step 2500   loss (last 250 steps) 0.0252   57 s
step 2750   loss (last 250 steps) 0.0267   62 s
step 3000   loss (last 250 steps) 0.0256   68 s
training took 68 seconds for 3000 steps (23 ms per step)
```

   *Exercise 6.2:* Compute `epochs_seen`: how many times the training set was passed through (steps × batch size divided by the number of pictures), and `ms_per_step`: the training time per step in milliseconds (use `time_unet` and `STEPS_UNET`). Why it matters: diffusion models are trained for many epochs; knowing the cost per step lets you plan a run, and the epoch count shows that 2,000 pictures are seen over and over (the random noise is different every time, so it is not the same example twice).

```python
epochs_seen = None; ms_per_step = None    # YOUR CODE: STEPS_UNET * 64 / len(data_img); 1000 * time_unet / STEPS_UNET

if epochs_seen is None or ms_per_step is None:
    print("Fill in epochs_seen and ms_per_step above, then run this cell again.")
else:
    check("epochs_seen = %d * 64 / 2000 = %.1f" % (STEPS_UNET, STEPS_UNET * 64 / 2000), abs(epochs_seen - STEPS_UNET * 64 / 2000) < 0.01, "steps times batch size divided by the number of pictures")
    check("ms_per_step is the training time per step", abs(ms_per_step - 1000 * time_unet / STEPS_UNET) < 1.0, "1000 * time_unet / STEPS_UNET")
```

   A possible answer (try first, then compare):

```python
epochs_seen = STEPS_UNET * 64 / len(data_img)
ms_per_step = 1000 * time_unet / STEPS_UNET
print("%.0f passes over the 2,000 pictures; %.0f ms per step" % (epochs_seen, ms_per_step))
```

3,000 steps of 64 pictures are 96 passes over the data. Each pass shows every picture with different noise at a different step number, so the network does not memorise noisy pictures; it can still memorise the five clean shapes, which for this tiny data set is what we want it to know.

It prints:

```text
96 passes over the 2,000 pictures; 23 ms per step
```

   **What you should see.**

   The loss, averaged over 250 steps, falls from **0.2670** (steps 1 to 250) to 0.0642 (500), 0.0358 (1,000), 0.0309 (1,500), 0.0276 (2,000) and **0.0256** at step 3,000; the average of the last 100 steps is about **0.027** (tolerance: between 0.02 and 0.04). The training takes about **68 seconds**, which is about **23 ms per step** (on a slower CPU up to two minutes). The check passes two ticks; the answer prints `96 passes over the 2,000 pictures; 23 ms per step`.

   *What it means.* Compare the loss with that of the 2-D denoiser (0.25): the picture denoiser reaches a loss of 0.027, ten times lower, because most of the 256 pixels are plain background that is easy to predict at every noise level. After about 1,000 steps the loss has almost levelled off, but the pictures of Step 6.6 show that the last 2,000 steps still matter: the loss falls from 0.036 to 0.026, a small number with a visible effect (cleaner background). The training log is on a log scale in the plot, and the curve is noisy because every step has random t.

   Tips and common mistakes:

   - `CosineAnnealingLR(opt, steps)` needs the **total** number of steps, and `sched.step()` once per step, not once per epoch.

3. **Step 6.3. DDPM sampling: pictures from noise.** Sampling is exactly the loop of Part 4, with the UNet as `eps_fn` and the shape (n, 1, 16, 16). It needs 100 network calls, one per step. We draw 32 pictures and show them, and then a "denoising film": the same 8 pictures at t = 100, 75, 50, 25 and 0, so that you can watch noise turn into shapes. The network works on batches, so 32 pictures cost the same number of calls as 1. Before judging pictures, we clip them to the range [−1, 1] and map them to [0, 1]; you write that small function first.

   *Exercise 6.3:* Write `to_unit(x)`: it takes a picture tensor in [−1, 1] (the output of the sampler may go a little outside) and returns `(x.clamp(-1, 1) + 1) / 2`, in [0, 1]. Why it matters: the judge of Step 6.4 and the human eye both expect pixel values between 0 and 1; a diffusion sampler is not forced to stay in range, and values outside are cut off rather than allowed to wrap. The check tests three values.

```python
def to_unit_mine(x):
    return None   # YOUR CODE: clip to [-1, 1], then shift and scale to [0, 1]

r_ = to_unit_mine(torch.tensor([-1.7, -1.0, 0.0, 1.0, 1.4]))
if r_ is None:
    print("Fill in to_unit_mine above, then run this cell again.")
else:
    check("-1 -> 0, 0 -> 0.5, 1 -> 1", torch.allclose(r_[1:4], torch.tensor([0.0, 0.5, 1.0])), "(x + 1) / 2")
    check("values outside [-1, 1] are clipped: -1.7 -> 0 and 1.4 -> 1", float(r_[0]) == 0.0 and float(r_[4]) == 1.0, "use x.clamp(-1, 1) first")
```

   A possible answer (try first, then compare):

```python
def to_unit(x):
    return (x.clamp(-1, 1) + 1) / 2
print(to_unit(torch.tensor([-1.7, -1.0, 0.0, 1.0, 1.4])))
```

The `clamp` is applied first, so a value of 1.4 becomes 1.0 and not 1.2.

It prints:

```text
tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000])
```

```python
t0 = time.time()
samples, film = sample_ddpm(lambda x, t: unet42(x, t), (32, 1, 16, 16), S, seed=1, keep={100, 75, 50, 25, 0})
print("DDPM sampling of 32 pictures: %.1f seconds, %d network calls" % (time.time() - t0, S["T"]))
print("range of the final pictures: [%.2f, %.2f]" % (float(samples.min()), float(samples.max())))
show_grid(to_unit(samples).numpy(), cols=8)
strip = torch.cat([to_unit(film[t][:8]) for t in (100, 75, 50, 25, 0)])            # the first 8 pictures at five moments
show_grid(strip.numpy(), titles=["t=100"] * 8 + ["t=75"] * 8 + ["t=50"] * 8 + ["t=25"] * 8 + ["t=0"] * 8, cols=8)
```

It prints:

```text
DDPM sampling of 32 pictures: 0.5 seconds, 100 network calls
range of the final pictures: [-1.31, 1.18]
```

   **What you should see.**

   The cell prints `DDPM sampling of 32 pictures: 0.5 seconds, 100 network calls` (times differ) and `range of the final pictures: [-1.31, 1.18]`: the sampler goes a little outside [−1, 1], which is why we clip before judging. The check passes two ticks (the answer prints `tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000])`). In the grid of 32 you see clean crosses, squares, bars and diagonals with a faintly speckled background; every one of the five shapes appears, none is a blend of two shapes. The denoising film (rows t = 100, 75, 50, 25, 0): the first three rows look like pure noise, in the row t = 25 faint outlines (a square's corner, the arm of a cross) show through the noise, and the last row shows the finished shapes.

   *What it means.* The shape is **decided late**: nothing is recognisable at t = 50, yet at t = 25 the outlines are there and at t = 0 the picture is clean. The network does not draw the picture top to bottom; it refines the whole picture at once, first the rough layout and then the details, each step a small correction. Notice also that the model makes new noise-free pictures that it has not copied from the data: the pictures are not exact copies of the training set (which has noise of 0.05 and a different noise pattern in every picture) but samples from what the model has learned.

4. **Step 6.4. The judge.** Looking at 32 pictures is not enough, and we need a number that does not depend on our eyes. We use the idea from Week 12: a **judge**, a small classifier that has been trained on clean pictures of the five shapes and is then frozen. For a batch of generated pictures it gives the *confidence* (the mean of its highest probability), the *entropy* of the histogram of predicted classes (maximum ln 5 = 1.609 if all five classes occur equally often) and the *coverage* (how many classes have at least 5 percent of the pictures). We add a fourth number for the pictures' cleanliness, which the classifier cannot see: the **template error**, the mean squared difference (in [0, 1] units) between a picture and the nearest of the five clean, noise-free shapes. Good pictures give a high confidence, an entropy near 1.609, a coverage of 5 and a small template error. Remember the two cautions of Week 12: the judge is confident on noise too, so confidence alone does not say a picture is good; and it tells you nothing about how new the pictures are. First the judge and the two helper functions are trained and defined (this takes a few seconds).

```python
class JudgeCNN(nn.Module):
    """A small CNN as in Week 4 (two convolutions with pooling, one linear layer) that tells the five shapes apart."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1); self.c2 = nn.Conv2d(8, 16, 3, padding=1); self.fc = nn.Linear(16 * 4 * 4, 5)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2); return self.fc(x.flatten(1))

Xj, yj = make_shapes(400, noise=0.1, seed=1); Xjt, yjt = make_shapes(200, noise=0.1, seed=2)            # 400 training and 200 test pictures
torch.manual_seed(0); judge_net = JudgeCNN()
Xt5, yt5 = make_shapes(5, noise=0.0, seed=0); templates = torch.tensor(Xt5[np.argsort(yt5)])         # one clean picture per class, in [0, 1]: used for the "template error" below
hist_j = train_classifier(judge_net, np.clip(Xj, 0, 1), yj, np.clip(Xjt, 0, 1), yjt, epochs=20, lr=0.01, seed=0)
judge_net.eval()
for p_ in judge_net.parameters(): p_.requires_grad = False        # frozen: the judge never learns from the generated pictures
print("judge: train accuracy %.3f  test accuracy %.3f" % (hist_j["train_acc"][-1], hist_j["test_acc"][-1]))
RESULTS["p6_judge_test_acc"] = round(hist_j["test_acc"][-1], 3)

def entropy_nats(hist):
    """Entropy of a class histogram (shares that sum to 1), in nats."""
    p = np.asarray(hist, dtype=float); p = p[p > 0]
    return float(-(p * np.log(p)).sum())
def coverage(hist, min_share=0.05):
    """The number of classes whose share is at least min_share."""
    return int((np.asarray(hist) >= min_share).sum())
assert abs(entropy_nats(np.ones(5) / 5) - math.log(5)) < 1e-9 and coverage(np.ones(5) / 5) == 5
```

It prints:

```text
judge: train accuracy 1.000  test accuracy 1.000
```

   *Exercise 6.4:* Write `judge_mine(images, unit=False)` (the template error is an extra in our version: you may leave it out). It receives a tensor or array of pictures with shape (N, 1, 16, 16), in [−1, 1] (the range of generated pictures; with `unit=True` they are already in [0, 1]), and returns a dictionary with `"pred"` (predicted classes), `"conf"` (the mean of the highest softmax probability), `"hist"` (the share of each of the five classes among the predictions), `"entropy"` and `"coverage"`. Use `judge_net`, `entropy_nats` and `coverage`, switch off the gradient with `torch.no_grad()`, and map to [0, 1] first (and clip) when `unit` is False. Why it matters: every experiment from here to Part 7 is scored with this function.

```python
def judge_mine(images, unit=False):
    """Score a batch of pictures with the frozen judge. Returns a dict with pred, conf, hist, entropy, coverage."""
    return None   # YOUR CODE: x = to_unit(...) unless unit; probabilities = softmax(judge_net(x)); conf, pred = probabilities.max(1); hist = np.bincount(pred, minlength=5) / N

Xr_, yr_ = make_shapes(300, noise=0.05, seed=5)
r_ = judge_mine(np.clip(Xr_, 0, 1) * 2 - 1)
if r_ is None:
    print("Fill in judge_mine above, then run this cell again.")
else:
    check("the confidence on real pictures is above 0.95", r_["conf"] > 0.95, "judge_net(x), then softmax over dimension 1, then the maximum; the pictures must be in [0, 1]")
    check("the histogram has five shares that add up to 1", len(r_["hist"]) == 5 and abs(float(np.sum(r_["hist"])) - 1) < 1e-6, "np.bincount(pred, minlength=5) / len(pred)")
    check("real pictures cover all 5 classes with entropy near 1.609", r_["coverage"] == 5 and r_["entropy"] > 1.55, "the classes are balanced in the data")
    check("the predictions agree with the labels (accuracy above 0.95)", float((r_["pred"] == yr_).mean()) > 0.95, "the judge was trained on these five shapes")
```

   A possible answer (try first, then compare):

```python
def judge(images, unit=False):
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    x = x.clamp(0, 1) if unit else to_unit(x)                        # back to [0, 1], the range the judge was trained on
    with torch.no_grad():
        pr = F.softmax(judge_net(x), 1)
    conf, pred = pr.max(1)
    hist = np.bincount(pred.numpy(), minlength=5) / len(pred)
    tmpl = float(((x[:, None] - templates[None]) ** 2).flatten(2).mean(2).min(1).values.mean())   # squared error to the nearest clean template: how clean are the pictures?
    return {"pred": pred.numpy(), "conf": float(conf.mean()), "hist": hist, "entropy": entropy_nats(hist), "coverage": coverage(hist), "tmpl": tmpl}
print("judge ready")
```

The judge sees pictures in [0, 1], like the ones it was trained on. We compute the histogram of predicted classes and read two numbers from it: the entropy (how evenly the classes occur) and the coverage (how many classes occur at all).

It prints:

```text
judge ready
```

Now judge three kinds of pictures: 500 real pictures, 500 pictures of pure noise (the same noise the sampler starts from, clipped to the range) and 500 pictures generated by the DDPM sampler with the UNet. Compare them. Sampling 500 pictures takes about 8 seconds.

```python
def show_j(name, j):
    print("%-22s confidence %.3f  entropy %.3f  coverage %d of 5  template error %.4f  histogram %s" % (name, j["conf"], j["entropy"], j["coverage"], j["tmpl"], np.round(j["hist"], 2)))
N_JUDGE = 100 if QUICK else 500
Xr, yr = make_shapes(N_JUDGE, noise=0.05, seed=5)
j_real = judge(np.clip(Xr, 0, 1) * 2 - 1)
j_noise = judge(torch.randn(N_JUDGE, 1, 16, 16).clamp(-1, 1))
t0 = time.time(); gen_u42 = sample_ddpm(lambda x, t: unet42(x, t), (N_JUDGE, 1, 16, 16), S, seed=0); t_ddpm = time.time() - t0
j_ddpm = judge(gen_u42)
show_j("real pictures", j_real); show_j("pure noise", j_noise); show_j("DDPM, 100 steps", j_ddpm)
print("sampling took %.1f seconds; accuracy of the judge on real: %.3f" % (t_ddpm, float((j_real["pred"] == yr).mean())))
RESULTS["p6_ddpm_conf"] = round(j_ddpm["conf"], 3); RESULTS["p6_ddpm_entropy"] = round(j_ddpm["entropy"], 3); RESULTS["p6_ddpm_coverage"] = j_ddpm["coverage"]
RESULTS["p6_noise_conf"] = round(j_noise["conf"], 3); RESULTS["p6_noise_entropy"] = round(j_noise["entropy"], 3); RESULTS["p6_ddpm_tmpl"] = round(j_ddpm["tmpl"], 4); RESULTS["p6_real_tmpl"] = round(j_real["tmpl"], 4)
```

It prints:

```text
real pictures          confidence 1.000  entropy 1.609  coverage 5 of 5  template error 0.0013  histogram [0.2 0.2 0.2 0.2 0.2]
pure noise             confidence 0.881  entropy 0.549  coverage 2 of 5  template error 0.3669  histogram [0.   0.   0.   0.76 0.24]
DDPM, 100 steps        confidence 1.000  entropy 1.601  coverage 5 of 5  template error 0.0014  histogram [0.23 0.21 0.16 0.18 0.22]
sampling took 5.9 seconds; accuracy of the judge on real: 1.000
```

   **What you should see.**

   The judge reaches a train and test accuracy of **1.000** (tolerance: test accuracy above 0.95; the five shapes are easy), the check passes four ticks for your function and the answer prints `judge ready`. The three rows (500 pictures each, seed 42): **real pictures**: confidence 1.000, entropy 1.609, coverage 5 of 5, template error 0.0013, histogram 0.2 for every class. **Pure noise**: confidence **0.881**, entropy 0.549, coverage 2 of 5, template error 0.3669, and 76 percent of the noise pictures are called the same class. **DDPM, 100 steps**: confidence **1.000**, entropy **1.601**, coverage **5 of 5**, template error 0.0014, histogram 0.23, 0.21, 0.16, 0.18, 0.22. Sampling 500 pictures takes about 6 seconds; the judge's accuracy on the real pictures is 1.000. Tolerance for the generated row: confidence above 0.95, entropy above 1.45, coverage 5.

   *What it means.* Two warnings from Week 12 show again. **Confidence alone says little**: the noise gets a confidence of 0.881 (the classifier must choose a class), what gives it away is the entropy (0.549, a spike on one class) and the template error (0.367, 280 times that of real pictures). **The generated pictures look like the real ones in all four numbers**: every class appears (coverage 5), the entropy is 1.601 of the maximum 1.609, and the template error is 0.0014, as low as real data (0.0013). The histogram shows a slightly lower share of one class (0.16 against 0.20); with 500 pictures a share has a sampling error of about 0.018, so this is a mild preference at most. The judge cannot tell you that the pictures are new rather than copies, and it does not see small defects: that is what the template error and your eyes are for. Compare with the DCGAN of Week 12 (confidence about 0.97, entropy 1.61, coverage 5, after much more tuning): the same judge numbers, reached without a second network.

5. **Step 6.5. Faster sampling with DDIM.** The DDPM loop needs 100 network calls, and for large models this is the main cost of diffusion (1,000 calls for the original DDPM). **DDIM** (denoising diffusion implicit models) is a sampler for the *same* trained network that uses a shorter list of steps, for example 100 → 50, 20, 10 or 5, and takes no fresh noise (η = 0, deterministic: the same starting noise always gives the same picture). In each step it first estimates the clean picture, x̂0 = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t (clipped to [−1, 1]), and then moves to the earlier step t′ by keeping that estimate and the predicted noise: x_t′ = √ᾱ_t′·x̂0 + √(1 − ᾱ_t′)·ε̂. The steps can jump (100, 95, 90 … or 100, 80, 60 …).

   *Exercise 6.5:* Write `ddim_step_mine(x, e, t, t_prev, S)` that implements the two formulas above (clip x̂0 to [−1, 1] before the second formula). The check uses a neat property: if `e` is the *true* noise that made x_t, then x̂0 is exactly the clean picture and the step must land exactly on the closed-form x_t′ for the same noise: `q_sample_t(x0, t_prev, e)`. Why it matters: it shows that DDIM is consistent with the forward process, and it lets you test your formula without any network.

```python
def ddim_step_mine(x, e, t, t_prev, S):
    """From x_t to x_(t_prev) with the deterministic DDIM rule. t and t_prev are plain integers, t_prev < t."""
    return None   # YOUR CODE: x0_hat = ((x - sqrt(1 - abar_t) * e) / sqrt(abar_t)).clamp(-1, 1); return sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev) * e

x0_c = data_img[:4]; e_c = torch.randn(x0_c.shape); t_c, tp_c = 60, 40
x_c = q_sample_t(x0_c, torch.full((4,), t_c), e_c, S)
r_ = ddim_step_mine(x_c, e_c, t_c, tp_c, S)
if r_ is None:
    print("Fill in ddim_step_mine above, then run this cell again.")
else:
    check("with the true noise the DDIM step lands on the closed form at t' = 40", torch.allclose(r_, q_sample_t(x0_c, torch.full((4,), tp_c), e_c, S), atol=1e-4), "x0_hat is then the clean picture; x_prev = sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev) * e")
    check("the last step (t' = 0) returns the estimate of the clean picture", torch.allclose(ddim_step_mine(x_c, e_c, t_c, 0, S), x0_c.clamp(-1, 1), atol=1e-4), "abar_0 = 1, so x_0 = x0_hat")
```

   A possible answer (try first, then compare):

```python
def ddim_step(x, e, t, t_prev, S):
    ab, ab_prev = S["abar"][t], S["abar"][t_prev]
    x0_hat = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)       # the network's estimate of the clean picture
    return ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * e           # re-noise that estimate to the earlier level, with the same noise guess
```

The step does not add fresh noise, so the whole path is determined by the starting noise. At the last step t_prev = 0 and ᾱ_0 = 1, so the output is the estimate x̂0 itself.

Now the sampler: it builds the list of steps with `np.linspace(T, 0, steps + 1)` (for 20 steps: 100, 95, 90, …, 0), calls the network once per step and applies `ddim_step`. We judge 500 pictures for each number of steps with the same starting noise (seed 0), measure the time, and plot the judge's numbers against the number of network calls. The DDPM row is the 100-step run from Step 6.4.

```python
@torch.no_grad()
def sample_ddim(eps_fn, shape, S, steps=20, seed=0):
    """Deterministic DDIM sampling with `steps` network calls."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    ts = np.linspace(S["T"], 0, steps + 1).round().astype(int)           # for example 100, 95, ..., 5, 0
    for t, t_prev in zip(ts[:-1], ts[1:]):
        e = eps_fn(x, torch.full((shape[0],), int(t), dtype=torch.long))
        x = ddim_step(x, e, int(t), int(t_prev), S)
    return x

rows65 = [("DDPM", 100, t_ddpm, j_ddpm)]
for K in (50, 20, 10, 5, 2, 1):
    t0 = time.time(); xg = sample_ddim(lambda x, t: unet42(x, t), (N_JUDGE, 1, 16, 16), S, steps=K, seed=0); dt = time.time() - t0
    rows65.append(("DDIM", K, dt, judge(xg)))
print("sampler  calls  seconds  confidence  entropy  coverage  template error")
for name, K, dt, j in rows65: print("%-7s %6d %8.1f %11.3f %8.3f %9d %15.4f" % (name, K, dt, j["conf"], j["entropy"], j["coverage"], j["tmpl"]))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot([r[1] for r in rows65], [r[3]["conf"] for r in rows65], "o-", label="confidence"); ax[0].plot([r[1] for r in rows65], [r[3]["entropy"] / math.log(5) for r in rows65], "s-", label="entropy / ln 5")
ax[0].set_xscale("log"); ax[0].set_xlabel("network calls"); ax[0].set_ylim(0, 1.05); ax[0].legend(); ax[0].set_title("judge's numbers against the number of steps")
ax[1].plot([r[2] for r in rows65], [r[3]["conf"] for r in rows65], "o-"); ax[1].set_xlabel("seconds for 500 pictures"); ax[1].set_ylabel("confidence"); ax[1].set_title("quality against time"); plt.tight_layout(); plt.show()
RESULTS["p6_ddim20_conf"] = round(rows65[2][3]["conf"], 3); RESULTS["p6_ddim20_entropy"] = round(rows65[2][3]["entropy"], 3); RESULTS["p6_ddim20_seconds"] = round(rows65[2][2], 1)
RESULTS["p6_ddim1_conf"] = round(rows65[-1][3]["conf"], 3); RESULTS["p6_ddim1_entropy"] = round(rows65[-1][3]["entropy"], 3)
```

It prints:

```text
sampler  calls  seconds  confidence  entropy  coverage  template error
DDPM       100      5.9       1.000    1.601         5          0.0014
DDIM        50      2.9       0.982    1.523         5          0.0228
DDIM        20      1.1       0.997    1.576         5          0.0023
DDIM        10      0.6       0.991    1.575         5          0.0031
DDIM         5      0.3       0.979    1.581         5          0.0059
DDIM         2      0.1       0.915    1.594         5          0.0199
DDIM         1      0.1       0.899    0.687         2          0.3930
```

Look at the pictures at 20, 5, 2 and 1 steps with the same starting noise: where does the quality go?

```python
fig_rows = []
for K in (20, 5, 2, 1):
    xg = sample_ddim(lambda x, t: unet42(x, t), (8, 1, 16, 16), S, steps=K, seed=1); fig_rows.append(to_unit(xg))
show_grid(torch.cat(fig_rows).numpy(), titles=["%d steps" % K for K in (20, 5, 2, 1) for _ in range(8)], cols=8)
print("mean absolute difference between the 20-step and the 1-step picture: %.3f" % float((fig_rows[0] - fig_rows[3]).abs().mean()))
```

It prints:

```text
mean absolute difference between the 20-step and the 1-step picture: 0.445
```

   **What you should see.**

   The check passes two ticks. The table for 500 pictures with the same starting noise (the times are rounded values from our CPU and differ from machine to machine; the other numbers should match within the stated tolerance):

   - DDPM, 100 calls: about 6 s, confidence 1.000, entropy 1.601, coverage 5, template error 0.0014
   - DDIM, 50 calls: about 3 s, 0.982, 1.523, 5, 0.0228
   - DDIM, 20 calls: about 1 s, 0.997, 1.576, 5, 0.0023
   - DDIM, 10 calls: about 0.6 s, 0.991, 1.575, 5, 0.0031
   - DDIM, 5 calls: about 0.3 s, 0.979, 1.581, 5, 0.0059
   - DDIM, 2 calls: about 0.1 s, 0.915, 1.594, 5, 0.0199
   - DDIM, 1 call: about 0.1 s, 0.899, 0.687, 2, 0.3930

   In the grid of 8 pictures: 20 steps gives clean shapes; 5 steps gives mostly clean shapes, with some pictures that show a faint second copy of a line ("ghost" strokes); 2 steps gives faint, overlaid shapes (a cross with a square across it, several bars at once); 1 step gives a black-and-white speckle pattern with no shape (the mean absolute difference from the 20-step picture is 0.445). The plot shows the confidence flat near 1 down to 5 calls and falling at 2 and 1 calls, while the entropy only falls at 1 call. Tolerance: 20 calls within 0.03 of the DDPM confidence; 1 call clearly broken (coverage below 4).

   *What it means.* DDIM with **20 calls is five times faster than DDPM with 100 and almost as good** in every number (confidence 0.997, template error 0.0023 against 0.0014); even 5 calls give recognisable shapes in the judge's eyes (0.979, coverage 5), though the template error has already quadrupled (0.0059) and ghost strokes appear. At 2 calls the picture is the network's best average guess of the clean picture, a blend of what the shapes might be, and the judge still sees five classes because blends are still mostly recognisable (0.915); at 1 call the sampler jumps directly from pure noise to a clipped estimate and fails (coverage 2). The result is not monotone: 50 calls has a worse template error (0.0228) than 20 calls (0.0023) in our run, because a few pictures ended up as blends; we have one starting-noise seed here, so do not over-read differences between 10 and 100 steps. The robust finding: **quality is flat from 100 down to about 10 calls, degrades at 5 and 2, and breaks at 1.**

   Tips and common mistakes:

   - The list of steps must start at T and end at 0, with the last step landing on t = 0 (ᾱ_0 = 1), otherwise the output is not a clean picture.
   - DDIM here is deterministic: the same seed always gives the same picture, in contrast to DDPM where every random z changes the result.

6. **Step 6.6. Training progress.** Now the checkpoints of Step 6.2: after 250 and 1,000 steps and at the end. We sample 500 pictures from each (DDPM, the same starting noise), judge them and show 8 pictures of each. This is the film of learning. Compare with your Imagine-first letter.

```python
rows66 = []; net_c = TinyUNet(); grids66 = []
for st in list(CK_UNET) + [STEPS_UNET]:
    if st == STEPS_UNET: net_c.load_state_dict(unet42.state_dict())
    else: net_c.load_state_dict(saved_u42[st])
    net_c.eval()
    xg = sample_ddpm(lambda x, t: net_c(x, t), (N_JUDGE, 1, 16, 16), S, seed=0); j = judge(xg); rows66.append((st, j)); grids66.append(to_unit(xg[:8]))
print("training step  confidence  entropy  coverage  template error  histogram")
for st, j in rows66: print("%13d %11.3f %8.3f %9d %15.4f  %s" % (st, j["conf"], j["entropy"], j["coverage"], j["tmpl"], np.round(j["hist"], 2)))
show_grid(torch.cat(grids66).numpy(), titles=["%d steps" % st for st, _ in rows66 for _ in range(8)], cols=8)
RESULTS["p6_conf_250"] = round(rows66[0][1]["conf"], 3); RESULTS["p6_cov_250"] = rows66[0][1]["coverage"]; RESULTS["p6_conf_1000"] = round(rows66[1][1]["conf"], 3)
```

It prints:

```text
training step  confidence  entropy  coverage  template error  histogram
          250       0.880    1.365         4          0.0433  [0.28 0.16 0.04 0.09 0.42]
         1000       0.997    1.545         5          0.0031  [0.34 0.2  0.14 0.14 0.18]
         3000       1.000    1.601         5          0.0014  [0.23 0.21 0.16 0.18 0.22]
```

   *Exercise 6.6:* Store in `earliest_full_coverage` the earliest checkpoint step at which the judge finds all 5 classes (coverage 5), or `None`, and in `my_description_250` one sentence describing what the 250-step pictures look like (look at the first row of the grid). Why it matters: the judge measures *what* the pictures are, your eyes tell you *how* they look; both are needed (a blurry stroke can be judged "horizontal bar" with high confidence).

```python
earliest_full_coverage = "unset"    # YOUR CODE: the first step in rows66 with coverage 5, or None
my_description_250 = ""             # YOUR TEXT: one sentence

if earliest_full_coverage == "unset" or my_description_250.strip() == "":
    print("Fill in both variables above, then run this cell again.")
else:
    want_ = next((st for st, j in rows66 if j["coverage"] == 5), None)
    check("earliest_full_coverage is %s" % want_, earliest_full_coverage == want_, "the first row of rows66 whose coverage is 5")
    check("your sentence has at least 6 words", len(my_description_250.split()) >= 6, "describe the strokes, their sharpness and the background")
```

   A possible answer (try first, then compare):

```python
earliest_full_coverage = next((st for st, j in rows66 if j["coverage"] == 5), None)
my_description_250 = "Scattered dark fragments of strokes around the centre on a speckled grey background; no complete shape yet."
print(earliest_full_coverage, "|", my_description_250)
```

Your sentence is yours, but it should say that there are fragments of strokes near the centre while the shapes are not complete and the background is speckled; the network learns *where* shapes go early and *how to complete and clean* them later.

It prints:

```text
1000 | Scattered dark fragments of strokes around the centre on a speckled grey background; no complete shape yet.
```

   **What you should see.**

   The table (500 pictures, DDPM, the same starting noise): after **250** steps confidence **0.880**, entropy 1.365, coverage **4**, template error 0.0433, histogram 0.28, 0.16, 0.04, 0.09, 0.42; after **1,000** steps confidence 0.997, entropy 1.545, coverage 5, template error 0.0031, histogram 0.34, 0.20, 0.14, 0.14, 0.18; after **3,000** steps confidence 1.000, entropy 1.601, coverage 5, template error 0.0014. In the grid the 250-step pictures are scattered dark fragments around the centre on a speckled background: pieces of strokes, here and there a short line, no complete shape. At 1,000 steps the shapes are clean and complete (a few have a slightly uneven line), at 3,000 steps the background is quieter. The Check passes two ticks; `earliest_full_coverage` is 1000.

   *What it means.* Compare with your Imagine-first letter. The 250-step network is **between (b) and (c)**: it has not produced smooth grey blurs; it draws fragments of strokes in roughly the right region but not complete shapes (so (c) comes closest, with "fragments" instead of "blurry strokes"). Its judge numbers are not bad (confidence 0.88, the judge always finds something) but the histogram is badly skewed (42 percent of the pictures are called one class, 4 percent another) and the template error is 30 times that of the final network. Between 250 and 1,000 steps the network learns **what the shapes are**; between 1,000 and 3,000 it learns **to draw them cleanly and in equal numbers** (entropy from 1.545 to 1.601, template error halved). The loss (0.0358 at 1,000 steps, 0.0256 at 3,000) hardly shows this.

7. **Step 6.7. Break it on purpose.** A diffusion model has several parts that look small: the time input, the skip connections, the number of sampling steps and the factor 1/√α_t in the reverse step. We break each of them in turn and read the judge's numbers against the healthy baseline. There are four experiments: **(a)** no time input (the UNet with `use_time=False`, retrained: 75 seconds); **(b)** no skip connections (`use_skips=False`, the skip features are replaced by zeros, retrained: 75 seconds), plus the test of Step 5.6 on the healthy trained network: zero the skips only at sampling time; **(c)** too few sampling steps: DDIM with 5, 2 and 1 steps (Step 6.5); **(d)** a wrong sampler: the reverse step without the factor 1/√α_t (`fix_scale=False`). We run everything for seed 42 first and then for seed 1 (three more trainings). The training cells are separate so that each one stays under two minutes.

```python
t0 = time.time(); unet_nt42, _, _ = train_unet(seed=42, use_time=False, log=0)
print("(a) no time input, seed 42: trained in %.0f seconds" % (time.time() - t0))
```

It prints:

```text
(a) no time input, seed 42: trained in 64 seconds
```

```python
t0 = time.time(); unet_ns42, _, _ = train_unet(seed=42, use_skips=False, log=0)
print("(b) no skip connections, seed 42: trained in %.0f seconds" % (time.time() - t0))
```

It prints:

```text
(b) no skip connections, seed 42: trained in 71 seconds
```

The evaluation function: for one seed it needs the healthy network and the two broken ones, and returns the judge's numbers for the healthy baseline, (a), (b), the post-hoc zero-skip test, (c) with 5, 2 and 1 steps and (d). It also measures, for the healthy network, how much its output changes when the skips are set to zero at sampling time (the follow-up of Step 5.6) on 256 real pictures noised to t = 50.

```python
def evaluate_breakages(net, net_nt, net_ns, n=N_JUDGE, seed=0):
    """The judge's numbers for the healthy network and for the four breakages. Returns a dict {name: judge dict} and the relative output change when the skips are zeroed."""
    res = {}
    f = lambda m: (lambda x, t: m(x, t))
    shape = (n, 1, 16, 16)
    res["healthy (DDPM 100)"] = judge(sample_ddpm(f(net), shape, S, seed=seed))
    res["(a) no time input"] = judge(sample_ddpm(f(net_nt), shape, S, seed=seed))
    res["(b) no skips, retrained"] = judge(sample_ddpm(f(net_ns), shape, S, seed=seed))
    net.use_skips = False                                          # zero the skips of the healthy network only at sampling time
    res["(b') skips zeroed after training"] = judge(sample_ddpm(f(net), shape, S, seed=seed))
    net.use_skips = True
    for K in (5, 2, 1): res["(c) DDIM %d steps" % K] = judge(sample_ddim(f(net), shape, S, steps=K, seed=seed))
    res["(d) no 1/sqrt(alpha) factor"] = judge(sample_ddpm(f(net), shape, S, seed=seed, fix_scale=False))
    x0 = data_img[:256]; tt = torch.full((256,), 50, dtype=torch.long); xt = q_sample_t(x0, tt, torch.randn(x0.shape), S)
    with torch.no_grad():
        o1 = net(xt, tt); net.use_skips = False; o2 = net(xt, tt); net.use_skips = True
    return res, float((o1 - o2).abs().mean() / o1.abs().mean())
```

```python
t0 = time.time()
res67 = {}; rel67 = {}
res67[42], rel67[42] = evaluate_breakages(unet42, unet_nt42, unet_ns42)
print("evaluated in %.0f seconds" % (time.time() - t0))
print("%-34s confidence  entropy  coverage  template error" % "seed 42")
for name, j in res67[42].items(): print("%-34s %10.3f %8.3f %9d %15.4f" % (name, j["conf"], j["entropy"], j["coverage"], j["tmpl"]))
print("relative change of the healthy network's output when its skips are zeroed (t = 50): %.2f" % rel67[42])
RESULTS["p6_rel_skip_change"] = round(rel67[42], 2)
```

It prints:

```text
evaluated in 31 seconds
seed 42                            confidence  entropy  coverage  template error
healthy (DDPM 100)                      1.000    1.601         5          0.0014
(a) no time input                       0.988    1.583         5          0.0170
(b) no skips, retrained                 0.881    0.524         2          0.5005
(b') skips zeroed after training        0.906    0.702         2          0.3999
(c) DDIM 5 steps                        0.979    1.581         5          0.0059
(c) DDIM 2 steps                        0.915    1.594         5          0.0199
(c) DDIM 1 steps                        0.899    0.687         2          0.3930
(d) no 1/sqrt(alpha) factor             1.000    1.529         5          0.0018
relative change of the healthy network's output when its skips are zeroed (t = 50): 1.06
```

The second seed: three more trainings (the healthy network, the one without time, the one without skips), each about 75 seconds, in separate cells.

```python
t0 = time.time(); unet1, _, _ = train_unet(seed=1, log=0)
print("healthy network, seed 1: trained in %.0f seconds" % (time.time() - t0))
```

It prints:

```text
healthy network, seed 1: trained in 70 seconds
```

```python
t0 = time.time(); unet_nt1, _, _ = train_unet(seed=1, use_time=False, log=0)
print("(a) no time input, seed 1: trained in %.0f seconds" % (time.time() - t0))
```

It prints:

```text
(a) no time input, seed 1: trained in 64 seconds
```

```python
t0 = time.time(); unet_ns1, _, _ = train_unet(seed=1, use_skips=False, log=0)
print("(b) no skip connections, seed 1: trained in %.0f seconds" % (time.time() - t0))
```

It prints:

```text
(b) no skip connections, seed 1: trained in 71 seconds
```

```python
res67[1], rel67[1] = evaluate_breakages(unet1, unet_nt1, unet_ns1)
print("%-34s %s    %s" % ("", "seed 42: conf  entropy  cov  tmpl", "seed 1: conf  entropy  cov  tmpl"))
for name in res67[42]:
    a_, b_ = res67[42][name], res67[1][name]
    print("%-34s %14.3f %8.3f %4d %7.4f %12.3f %8.3f %4d %7.4f" % (name, a_["conf"], a_["entropy"], a_["coverage"], a_["tmpl"], b_["conf"], b_["entropy"], b_["coverage"], b_["tmpl"]))
print("relative change when the skips are zeroed: seed 42 %.2f, seed 1 %.2f" % (rel67[42], rel67[1]))
```

It prints:

```text
                                   seed 42: conf  entropy  cov  tmpl    seed 1: conf  entropy  cov  tmpl
healthy (DDPM 100)                          1.000    1.601    5  0.0014        0.997    1.535    5  0.0063
(a) no time input                           0.988    1.583    5  0.0170        0.980    1.549    5  0.0231
(b) no skips, retrained                     0.881    0.524    2  0.5005        0.908    0.489    2  0.5283
(b') skips zeroed after training            0.906    0.702    2  0.3999        0.985    0.378    2  0.6934
(c) DDIM 5 steps                            0.979    1.581    5  0.0059        0.973    1.534    5  0.0104
(c) DDIM 2 steps                            0.915    1.594    5  0.0199        0.911    1.555    5  0.0212
(c) DDIM 1 steps                            0.899    0.687    2  0.3930        0.927    0.603    2  0.2935
(d) no 1/sqrt(alpha) factor                 1.000    1.529    5  0.0018        1.000    1.465    5  0.0040
relative change when the skips are zeroed: seed 42 1.06, seed 1 1.13
```

   *Exercise 6.7:* Find `worst_breakage`: the name (a key of `res67[42]`) of the row with the lowest judge confidence for seed 42, excluding the healthy baseline. Then write in `answer_6_7` four sentences, one per breakage (a) to (d): did it change the judge's numbers, and in which direction? Be honest: report a breakage that did **not** show an effect, and say that the judge cannot see everything (look at the pictures too: draw 16 pictures from the broken model with `show_grid(to_unit(sample_ddpm(...)).numpy())`). Why it matters: a good experimenter reports what happened, not what the theory predicts.

```python
worst_breakage = None     # YOUR CODE: min over the keys of res67[42] except "healthy (DDPM 100)", by res67[42][name]["conf"]
answer_6_7 = ""           # YOUR TEXT: four sentences, one per breakage

if worst_breakage is None or answer_6_7.strip() == "":
    print("Fill in worst_breakage and answer_6_7 above, then run this cell again.")
else:
    want_ = min((k for k in res67[42] if not k.startswith("healthy")), key=lambda k: res67[42][k]["conf"])
    check("worst_breakage is %r" % want_, worst_breakage == want_, "the key with the smallest 'conf' among the broken rows")
    check("your answer has at least 40 words", len(answer_6_7.split()) >= 40, "one sentence for each of (a), (b), (c) and (d)")
```

   A possible answer (try first, then compare):

```python
worst_breakage = min((k for k in res67[42] if not k.startswith("healthy")), key=lambda k: res67[42][k]["conf"])
print("lowest confidence among the broken rows:", worst_breakage, " (%.3f)" % res67[42][worst_breakage]["conf"])
RESULTS["p6_worst_breakage"] = worst_breakage
x_bad = sample_ddpm(lambda x, t: unet42(x, t), (16, 1, 16, 16), S, seed=1, fix_scale=False)
show_grid(to_unit(x_bad).numpy(), cols=8)                     # what the wrong sampler (d) draws
```

The row with the lowest confidence is the retrained network without skips (0.881), which is also the most broken one by every other measure. A sample answer for `answer_6_7`: "(a) Without the time input the judge's numbers hardly changed (confidence 0.988, coverage 5) but the pictures are noisier (template error 0.017 against 0.0014) and one picture in eight stayed noise. (b) Without skips the network produced no shapes at all (confidence 0.881, coverage 2, template error 0.50), the same numbers as pure noise; zeroing the skips after training does the same. (c) Five sampling steps were still fine (0.979, coverage 5) but two gave blended shapes and one step gave noise. (d) The sampler without 1/sqrt(alpha) showed no visible effect in the judge's numbers or in the pictures, so the experiment did not confirm that it matters here."

It prints:

```text
lowest confidence among the broken rows: (b) no skips, retrained  (0.881)
```

   **What you should see.**

   The check passes two ticks. Healthy network, seed 42: confidence 1.000, entropy 1.601, coverage 5, template error 0.0014; seed 1: 0.997, 1.535, 5, 0.0063. The breakages (seed 42, then seed 1):

   - **(a) no time input:** 0.988, 1.583, coverage 5, template error 0.0170; seed 1: 0.980, 1.549, 5, 0.0231.
   - **(b) no skips, retrained:** 0.881, 0.524, **coverage 2**, template error 0.5005; seed 1: 0.908, 0.489, 2, 0.5283.
   - **(b') skips zeroed after training:** 0.906, 0.702, **coverage 2**, 0.3999; seed 1: 0.985, 0.378, 2, 0.6934.
   - **(c) DDIM with 5, 2 and 1 steps:** 5 steps: 0.979, 1.581, 5, 0.0059; 2 steps: 0.915, 1.594, 5, 0.0199; 1 step: 0.899, 0.687, **coverage 2**, 0.3930 (seed 1: 0.973, 0.911, 0.927 for the three; coverage 5, 5, 2).
   - **(d) no 1/√α factor:** 1.000, 1.529, 5, 0.0018; seed 1: 1.000, 1.465, 5, 0.0040.

   The relative change of the healthy network's output when its skips are zeroed (at t = 50) is **1.06** for seed 42 and **1.13** for seed 1. Looking at 8 pictures: (a) shows, in the 8 pictures we drew, mostly clean shapes with a speckled background and one picture that stayed a noise pattern; (b) and (b') show black-and-white speckle, no shape at all; (c) at 1 step is the same speckle; (d) shows clean shapes, as good as the healthy model. The answer to `worst_breakage` is `(b) no skips, retrained`. Tolerance: (b), (b') and the 1-step row have coverage 2 in both seeds (below 4); the healthy rows have confidence above 0.99 and coverage 5.

   *What it means.* An honest reading of the table:

   - **Skip connections are essential here.** Without them (retrained or removed after training) the model draws no shapes: the numbers are those of pure noise (confidence 0.881, entropy 0.549, coverage 2 in Step 6.4). The change of the output by 106 to 113 percent when the skips are zeroed says that the trained network gets its fine detail from the skips and not from the 4 × 4 bottleneck, as the toy of Step 5.6 suggested.
   - **The time input matters less to the judge than to the pictures.** The judge does not notice (0.988, coverage 5), but the template error is 12 times larger for seed 42 and 3.7 times for seed 1, and some pictures remain noise. This matches Step 5.2: without t the network is a compromise at every noise level. The judge alone would have told us "no problem": the second number was needed.
   - **Too few steps degrade gradually and then break.** 5 steps: fine; 2 steps: blends; 1 step: no shape.
   - **The missing 1/√α factor did not show.** We expected a failure and found none: the factor is between 1.0005 and 1.118 (for β up to 0.2), so the error per step is small, the next network call corrects part of it and the loop is run many times. The judge's entropy is a little lower (1.529 and 1.465 against 1.601 and 1.535), the template error is no worse. It is a correct finding, not a successful ablation: with other schedules (larger steps) the factor matters more, and the formula is the right one regardless.
   - **Judge and eyes see different things.** Pure noise gets a confidence of 0.881, so confidence alone says little; the entropy, the template error and the pictures themselves complete the picture, and in this lab each of them caught something that another one missed.

8. **Step 6.8. Optional stretch: classifier-free guidance.** So far the model draws a random shape. **Classifier-free guidance** lets you ask for one. During training the network receives the class of the picture (through an embedding added to the time embedding), and in 15 percent of the steps the label is replaced by "no label"; so the same network learns two guesses: ε(c), the noise guess given the class, and ε(∅), the guess without it. At sampling time we mix the two: ε̃ = ε(∅) + w·(ε(c) − ε(∅)). With w = 0 the label is ignored, with w = 1 you get the plain conditional model and with w above 1 the class is *exaggerated*. This step trains a conditional network (about 75 seconds) and samples 60 pictures per class for w in 0, 1, 3 and 6.

   *Exercise 6.8:* Write `cfg_eps(model, x, t, y, w)`: compute the guess with the label (`model(x, t, y)`) and the guess without it (`model(x, t, null)`, where `null` is a tensor filled with the value 5, the "no label" index), and return `e_uncond + w * (e_cond - e_uncond)`. Why it matters: it is the whole trick; the rest of the sampler is unchanged. The check tests that w = 0 returns the unconditional guess and w = 1 the conditional guess, using a dummy model.

```python
def cfg_eps_mine(model, x, t, y, w):
    """Classifier-free guidance: e_uncond + w * (e_cond - e_uncond). The 'no label' index is 5."""
    return None   # YOUR CODE: two calls of the model (with the label y and with torch.full_like(y, 5)), then combine

class _Dummy:
    def __call__(self, x, t, y): return x * 0 + (1.0 if int(y[0]) < 5 else 3.0)        # label -> 1.0, no label -> 3.0
xd_, td_, yd_ = torch.zeros(2, 1, 4, 4), torch.ones(2, dtype=torch.long), torch.tensor([2, 2])
r0_ = cfg_eps_mine(_Dummy(), xd_, td_, yd_, 0.0)
if r0_ is None:
    print("Fill in cfg_eps_mine above, then run this cell again.")
else:
    check("w = 0 gives the unconditional guess (3.0)", torch.allclose(r0_, torch.full_like(xd_, 3.0)), "e_uncond + 0 * (...)")
    check("w = 1 gives the conditional guess (1.0)", torch.allclose(cfg_eps_mine(_Dummy(), xd_, td_, yd_, 1.0), torch.full_like(xd_, 1.0)), "e_uncond + 1 * (e_cond - e_uncond) = e_cond")
    check("w = 3 goes past the conditional guess (3 + 3 * (1 - 3) = -3)", torch.allclose(cfg_eps_mine(_Dummy(), xd_, td_, yd_, 3.0), torch.full_like(xd_, -3.0)), "e_uncond + w * (e_cond - e_uncond)")
```

   A possible answer (try first, then compare):

```python
def cfg_eps(model, x, t, y, w):
    e_c = model(x, t, y)                                         # the guess given the requested class
    e_u = model(x, t, torch.full_like(y, 5))                     # the guess without a class (the 'no label' index)
    return e_u + w * (e_c - e_u)                                 # w = 0: ignore the class, w = 1: plain conditional, w > 1: exaggerate it
```

For w = 1 the two terms collapse to e_c, the plain conditional guess; for w = 3 the guess is pushed three times as far from the unconditional one in the direction the class points, which draws stronger and more typical shapes at the price of variety.

```python
t0 = time.time(); unet_cond, _, _ = train_unet(seed=42, labels=labels_img, n_classes=5, log=0)
print("conditional network trained in %.0f seconds (%d parameters)" % (time.time() - t0, count_params(unet_cond)))
```

It prints:

```text
conditional network trained in 70 seconds (135905 parameters)
```

```python
PER = 12 if QUICK else 60
y_req = torch.arange(5).repeat_interleave(PER)                                  # 60 requests for class 0, then 60 for class 1, ...
rows68 = []; show68 = {}
for w in (0.0, 1.0, 3.0, 6.0):
    xg = sample_ddpm(lambda x, t: cfg_eps(unet_cond, x, t, y_req, w), (5 * PER, 1, 16, 16), S, seed=0)
    j = judge(xg)
    acc = float((j["pred"] == y_req.numpy()).mean())
    spread = float(np.mean([xg[y_req == c].std(0).mean() for c in range(5)]))        # how much do pictures of the same class differ from each other?
    ink = float((to_unit(xg) > 0.5).float().flatten(1).sum(1).mean())                  # dark pixels per picture: how thick are the lines?
    rows68.append((w, acc, j["conf"], j["entropy"], spread, j["tmpl"], ink)); show68[w] = to_unit(torch.stack([xg[y_req == c][0] for c in range(5)]))
ink_ref = float((to_unit(data_img) > 0.5).float().flatten(1).sum(1).mean()); print("dark pixels per picture in the training data: %.1f" % ink_ref)
print("   w   requested equals judged   confidence   class entropy   spread within a class   template error   dark pixels")
for w, acc, cf, en, sp, te, ik in rows68: print("%4.1f %18.3f %12.3f %15.3f %23.3f %16.4f %13.1f" % (w, acc, cf, en, sp, te, ik))
show_grid(torch.cat([show68[w] for w in (0.0, 1.0, 3.0, 6.0)]).numpy(), titles=["w=%g: %s" % (w, SHAPE_NAMES[c]) for w in (0.0, 1.0, 3.0, 6.0) for c in range(5)], cols=5)
RESULTS["p6_cfg_acc_w0"] = round(rows68[0][1], 3); RESULTS["p6_cfg_acc_w1"] = round(rows68[1][1], 3); RESULTS["p6_cfg_acc_w6"] = round(rows68[3][1], 3); RESULTS["p6_cfg_ink_w1_w6"] = (round(rows68[1][6], 1), round(rows68[3][6], 1))
```

It prints:

```text
dark pixels per picture in the training data: 15.2
   w   requested equals judged   confidence   class entropy   spread within a class   template error   dark pixels
 0.0              0.213        0.999           1.583                   0.217           0.0017          14.2
 1.0              1.000        1.000           1.609                   0.073           0.0016          15.2
 3.0              1.000        1.000           1.609                   0.071           0.0016          15.2
 6.0              1.000        1.000           1.609                   0.070           0.0019          15.2
```

   **What you should see.**

   The training takes about 69 seconds; the conditional network has **135,905** parameters (135,713 plus 6 × 32 = 192 for the label embeddings). The check passes three ticks. The table for 300 requested pictures per w (60 per class), with the training data's dark pixels per picture as 15.2:

   - **w = 0:** requested class equals the judged class for **0.213** of the pictures (chance is 0.2), confidence 0.999, class entropy 1.583, spread within a class 0.217, template error 0.0017, 14.2 dark pixels.
   - **w = 1:** accuracy **1.000**, confidence 1.000, entropy 1.609, spread 0.073, template error 0.0016, 15.2 dark pixels.
   - **w = 3:** 1.000, 1.000, 1.609, 0.071, 0.0016, 15.2.
   - **w = 6:** 1.000, 1.000, 1.609, 0.070, 0.0019, 15.2.

   Tolerance: w = 0 accuracy below 0.4, w = 1 accuracy above 0.95. The grid shows at w = 0 five shapes that have nothing to do with the requested class (the first row), and at w = 1, 3 and 6 the five requested shapes, correctly, in every row.

   *What it means.* The label works: with w = 0 it is ignored (accuracy at chance), with w = 1 the conditional network always draws the requested shape. **Guidance beyond w = 1 changed almost nothing in our tiny task**, which is an honest finding and not a failure of the method: the usual trade-off (more accurate but less varied for higher w, and harsher images for very high w) shows when the conditional guess is uncertain, as with the thousands of classes and free-text prompts of Stable Diffusion. Here the five shapes are so easy that w = 1 is already certain; the spread within a class falls only from 0.073 to 0.070, the ink does not change (15.2 dark pixels at every w from 1 to 6), and the template error rises only from 0.0016 to 0.0019 at w = 6. (The shapes are always in the centre, so there is little variety for guidance to take away.) In Step 7.3 the guidance scale of a real model makes a visible difference.

### Part 7. Guidance, latent space, responsible use and the capstone (10 min)

*Lesson sections: "Guidance" and "Capstone".* You have built a diffusion model that works. ==This last part puts it in proportion: you compute why real image models work in a compressed "latent" space, measure what sampling costs, (optionally, with a GPU) try the settings of Stable Diffusion, write a plan for how a generative model could, and could not, help the capstone, and look back on the lab.== The tiny UNet of Part 6 is used for the cost measurements.

1. **Step 7.1. Latent diffusion arithmetic.** Stable Diffusion does not run the denoising loop on pixels. A separate autoencoder compresses a 512 × 512 × 3 picture into a 64 × 64 × 4 **latent**, the UNet denoises the latent, and the autoencoder's decoder turns the result into a picture. This is **latent diffusion**. The saving is large and easy to compute. The UNet also contains attention layers (Week 8), whose cost grows with the square of the number of positions, so the saving there is larger still.

   *Exercise 7.1:* Compute `values_pixel` (512 × 512 × 3), `values_latent` (64 × 64 × 4), `ratio_values` (the first divided by the second), `positions_pixel` (512 × 512), `positions_latent` (64 × 64), `ratio_positions`, and `ratio_attention` (the square of `ratio_positions`: the attention table has one entry for every pair of positions). Why it matters: these three ratios (48, 64 and 4,096) are why a model that draws 512 × 512 pictures runs on one graphics card. The check tests the first and the last.

```python
values_pixel = values_latent = ratio_values = None
positions_pixel = positions_latent = ratio_positions = ratio_attention = None      # YOUR CODE: replace every None by the computed number

if values_pixel is None or ratio_attention is None:
    print("Fill in the seven numbers above, then run this cell again.")
else:
    check("values: 786,432 against 16,384, ratio 48", (values_pixel, values_latent) == (786432, 16384) and abs(ratio_values - 48) < 1e-9, "512 * 512 * 3 and 64 * 64 * 4")
    check("positions: ratio 64", abs(ratio_positions - 64) < 1e-9, "512 * 512 = 262,144 and 64 * 64 = 4,096")
    check("attention table: ratio 4,096", abs(ratio_attention - 4096) < 1e-6, "ratio_positions squared")
```

   A possible answer (try first, then compare):

```python
values_pixel, values_latent = 512 * 512 * 3, 64 * 64 * 4
ratio_values = values_pixel / values_latent
positions_pixel, positions_latent = 512 * 512, 64 * 64
ratio_positions = positions_pixel / positions_latent
ratio_attention = ratio_positions ** 2
print("values: %d against %d, ratio %.0f" % (values_pixel, values_latent, ratio_values))
print("positions: %d against %d, ratio %.0f; attention table entries: ratio %.0f" % (positions_pixel, positions_latent, ratio_positions, ratio_attention))
assert ratio_values == 48 and ratio_positions == 64 and ratio_attention == 4096
RESULTS["p7_ratios"] = (int(ratio_values), int(ratio_positions), int(ratio_attention))
```

The latent has 48 times fewer numbers, 64 times fewer positions, and the attention table (positions squared) is 4,096 times smaller. These savings are paid for with an autoencoder that must reconstruct pictures well; the denoising loop then runs on the small latent.

It prints:

```text
values: 786432 against 16384, ratio 48
positions: 262144 against 4096, ratio 64; attention table entries: ratio 4096
```

   **What you should see.**

   The check passes three ticks and the answer prints `values: 786432 against 16384, ratio 48` and `positions: 262144 against 4096, ratio 64; attention table entries: ratio 4096`; the assertion passes. `RESULTS["p7_ratios"]` is (48, 64, 4096).

   *What it means.* The loop of 20 to 50 network calls that makes a Stable Diffusion picture runs on a tensor of 16,384 numbers, not on 786,432; the attention layers handle 4,096 positions, not 262,144, and their cost grows with the square of the number of positions: 64² = 4,096 times less in the attention table. That is why a 512 × 512 model is practical on one graphics card. The price is the autoencoder: any detail it cannot reconstruct (very small text, fine textures) is lost for good, and the diffusion model never sees pixels.

2. **Step 7.2. The cost of sampling.** A GAN draws a picture with one forward pass. A diffusion model needs one pass per step. We measure how long one pass of the tiny UNet takes (a batch of 32 pictures, averaged over 20 calls) and build the table of costs for the samplers of Part 6. For a large model we only extrapolate, under an assumption that we state: one network call takes 0.05 seconds (a big UNet for one picture on a good GPU; a made-up round number, not a measurement; real values depend on model, resolution and hardware).

```python
x32 = torch.randn(32, 1, 16, 16); t32 = torch.randint(1, 101, (32,))
with torch.no_grad():
    for _ in range(3): unet42(x32, t32)                                  # warm up
    t0 = time.time()
    for _ in range(20): unet42(x32, t32)
sec_call = (time.time() - t0) / 20
print("one pass of the tiny UNet on a batch of 32 pictures: %.1f ms" % (1000 * sec_call))
ASSUMED = 0.05          # assumed seconds per call for a large model (an assumption, not a measurement)
print("%-16s %6s %20s %24s" % ("sampler", "calls", "tiny UNet (32 pics)", "large model at 0.05 s/call"))
for name, calls in (("GAN", 1), ("DDPM, 1000", 1000), ("DDPM, 100", 100), ("DDIM, 50", 50), ("DDIM, 20", 20)):
    print("%-16s %6d %17.2f s %21.1f s" % (name, calls, calls * sec_call, calls * ASSUMED))
RESULTS["p7_ms_per_call"] = round(1000 * sec_call, 1)
```

It prints:

```text
one pass of the tiny UNet on a batch of 32 pictures: 4.6 ms
sampler           calls  tiny UNet (32 pics) large model at 0.05 s/call
GAN                   1              0.00 s                   0.1 s
DDPM, 1000         1000              4.60 s                  50.0 s
DDPM, 100           100              0.46 s                   5.0 s
DDIM, 50             50              0.23 s                   2.5 s
DDIM, 20             20              0.09 s                   1.0 s
```

   *Exercise 7.2:* You want to draw 1,000 pictures with the large model (batch size 1, so 1,000 separate runs of the loop) in one minute (60 seconds). Store in `max_calls` the largest number of network calls per picture that fits, at 0.05 seconds per call, and in `ddim_steps_fit` the largest number of DDIM steps from the list [100, 50, 20, 10, 5] that fits (or `None`). Why it matters: the budget decides the sampler. The check recomputes both.

```python
max_calls = None; ddim_steps_fit = "unset"    # YOUR CODE: 60 / (1000 * 0.05) calls per picture; then the largest entry of [100, 50, 20, 10, 5] that is at most max_calls

if max_calls is None or ddim_steps_fit == "unset":
    print("Fill in max_calls and ddim_steps_fit above, then run this cell again.")
else:
    want_calls = 60 / (1000 * 0.05); want_steps = max([s for s in (100, 50, 20, 10, 5) if s <= want_calls] or [None], key=lambda s: -1 if s is None else s)
    check("max_calls is %.1f" % want_calls, abs(max_calls - want_calls) < 1e-9, "60 seconds divided by (1,000 pictures times 0.05 seconds)")
    check("ddim_steps_fit is %s" % want_steps, ddim_steps_fit == want_steps, "the largest step count that does not exceed max_calls; None if even 5 is too many")
```

   A possible answer (try first, then compare):

```python
max_calls = 60 / (1000 * 0.05)                                           # 1.2 calls per picture
ddim_steps_fit = max([s for s in (100, 50, 20, 10, 5) if s <= max_calls] or [None], key=lambda s: -1 if s is None else s)
print("calls per picture that fit: %.1f; DDIM steps that fit: %s" % (max_calls, ddim_steps_fit))
```

Only 1.2 calls per picture fit, so even 5 DDIM steps are too many for one minute: you would need a faster model (smaller, distilled to one or a few steps), a faster graphics card, or more time. This is why research on few-step samplers is so active.

It prints:

```text
calls per picture that fit: 1.2; DDIM steps that fit: None
```

   **What you should see.**

   One forward pass of the tiny UNet on a batch of 32 pictures takes about **4.6 ms** on our CPU (the number depends on the machine; expect between 3 and 20 ms). The table (tiny-UNet times are about the per-call time multiplied by the number of calls and differ from machine to machine; the large-model column does not): GAN, 1 call, about 0.005 s for the tiny UNet (the table prints 0.00) and 0.1 s (0.05 rounded) for the large model; DDPM with 1,000 calls about 4.6 s and **50.0 s**; DDPM with 100 calls about 0.46 s and 5.0 s; DDIM with 50 calls about 0.23 s and 2.5 s; DDIM with 20 calls about 0.09 s and 1.0 s. The check passes two ticks; the answer prints `calls per picture that fit: 1.2; DDIM steps that fit: None`.

   *What it means.* The large-model column is an assumption (0.05 s per call), so read it as a comparison, not a measurement: relative to a GAN, a 1,000-step DDPM costs 1,000 times more per picture and a 20-step DDIM 20 times. For the budget question 60 / (1,000 × 0.05) = 1.2 calls per picture, so even the 5-step DDIM does not fit: you would need a batch (several pictures per call, which lowers the cost per picture a lot on a GPU), a faster model, or a few-step sampler. Real systems do all three.

3. **Step 7.3. Optional GPU stretch: Stable Diffusion.** This step needs a GPU and the `diffusers` library; it is **not required** for the grade, and the cell does nothing on a CPU. If your instructor has stored the Stable Diffusion v1.5 weights (model ID `stable-diffusion-v1-5/stable-diffusion-v1-5`, about 4 GB; the older ID `runwayml/stable-diffusion-v1-5` no longer exists), you can repeat the two experiments of this lab on a real model: change the number of steps (5, 15, 30) with the same prompt and seed, and change the guidance scale (1, 7.5, 15), which is the w of Step 6.8. What to look for: with 5 steps the picture is blurry and unfinished, from about 15 steps it settles; with guidance 1 the picture ignores the prompt more, with 7.5 it follows it, with 15 colours become harsh and saturated and details look forced (the same price as w = 6 in the toy). Free the GPU at the end.

```python
if device == "cuda":
    try:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
        prompt = "a photograph of a cracked concrete wall, close up"
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        for i, steps in enumerate((5, 15, 30)):                     # same prompt, same seed, different number of steps
            img = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, generator=torch.Generator("cuda").manual_seed(0)).images[0]
            ax[0, i].imshow(img); ax[0, i].set_title("%d steps" % steps); ax[0, i].axis("off")
        for i, gs in enumerate((1.0, 7.5, 15.0)):                   # same prompt, same seed, different guidance scale
            img = pipe(prompt, num_inference_steps=30, guidance_scale=gs, generator=torch.Generator("cuda").manual_seed(0)).images[0]
            ax[1, i].imshow(img); ax[1, i].set_title("guidance %g" % gs); ax[1, i].axis("off")
        plt.show()
        del pipe; gc.collect(); torch.cuda.empty_cache()           # free the GPU
    except Exception as err:
        print("Skipped: the Stable Diffusion weights are not available here (%s)." % type(err).__name__)
else:
    print("No GPU: this optional step is skipped.")
```

4. **Step 7.4. Capstone planning.** In Week 19 your capstone is the **Structural Inspection Assistant**: a photo of a structure goes in; the pipeline classifies it (crack, spalling, corrosion or none), looks up the matching checklist items with retrieval (RAG), and drafts a summary for the inspector; **a licensed engineer makes every pass or fail decision**. The Capstone page of the course app (`/capstone`) describes the project; it opens this week. Could a generative model help? Possibly, for example by making extra training pictures of rare damage. But it can also do harm. Write a short plan, as a Python dictionary that prints as a card, with five fields: `where_it_helps` (where could synthetic images help the pipeline?), `what_could_go_wrong` (name at least two risks: think of unrealistic cracks, wrong labels, mixing synthetic with real evidence), `how_i_would_evaluate` (use the judge idea of Part 6 and a human check), `what_i_will_not_do` (for example, never use synthetic pictures as evidence in a report), and `human_check` (who looks at the result and what they decide).

   *Exercise 7.4:* Fill in the five fields of `plan` with one to three sentences each, in your own words and about this project (not generic). Why it matters: planning how a tool could fail before you build it is the habit of responsible engineering; the evaluation field should name a number you would measure. The check only tests that every field has at least 12 words and that the evaluation field mentions a measurement.

```python
plan = {"where_it_helps": "", "what_could_go_wrong": "", "how_i_would_evaluate": "", "what_i_will_not_do": "", "human_check": ""}    # YOUR TEXT in every field

empty_ = [k for k, v in plan.items() if len(v.split()) < 12]
if len(empty_) == len(plan):
    print("Fill in the five fields of plan above, then run this cell again.")
else:
    for k, v in plan.items(): check("%-22s %d words" % (k, len(v.split())), len(v.split()) >= 12, "write at least one full sentence")
    check("the evaluation names something you would measure (accuracy, confidence, coverage, percent, ...)", any(w in plan["how_i_would_evaluate"].lower() for w in ("accuracy", "confidence", "coverage", "entropy", "percent", "error", "recall", "precision", "%")), "for example the judge's confidence and coverage, or the accuracy of the classifier on real pictures only")
```

   A possible answer (try first, then compare):

```python
plan = {"where_it_helps": "Spalling and corrosion photos are rare in a small data set, so a diffusion model trained on real examples could add varied extra training pictures for the classifier, but only for training.",
        "what_could_go_wrong": "Generated cracks may look unrealistic or too regular, the classifier may learn the artefacts of the generator instead of real damage, and a synthetic picture could be mistaken for a real inspection photo if the files are mixed.",
        "how_i_would_evaluate": "Train the classifier with and without synthetic pictures and compare accuracy and recall on real, held-out photos only; use a judge's confidence and coverage on the generated set, and let an engineer rate 50 generated pictures as realistic or not.",
        "what_i_will_not_do": "I will never add synthetic pictures to the test set or to a report, never present them as evidence of a real structure, and I will label every generated file with its origin.",
        "human_check": "A licensed engineer reviews the data set and the evaluation, and makes every pass or fail decision on real photos; the model only suggests and summarises."}
print("CAPSTONE CARD: generative model for the Structural Inspection Assistant")
for k, v in plan.items(): print("- %s: %s" % (k.replace("_", " "), v))
```

Notice the pattern: the generated pictures are used only where a mistake is cheap (training), the decision is human, and the evaluation uses real data. Your own plan can differ; what matters is that every field is specific to this project.

It prints:

```text
CAPSTONE CARD: generative model for the Structural Inspection Assistant
- where it helps: Spalling and corrosion photos are rare in a small data set, so a diffusion model trained on real examples could add varied extra training pictures for the classifier, but only for training.
- what could go wrong: Generated cracks may look unrealistic or too regular, the classifier may learn the artefacts of the generator instead of real damage, and a synthetic picture could be mistaken for a real inspection photo if the files are mixed.
- how i would evaluate: Train the classifier with and without synthetic pictures and compare accuracy and recall on real, held-out photos only; use a judge's confidence and coverage on the generated set, and let an engineer rate 50 generated pictures as realistic or not.
- what i will not do: I will never add synthetic pictures to the test set or to a report, never present them as evidence of a real structure, and I will label every generated file with its origin.
- human check: A licensed engineer reviews the data set and the evaluation, and makes every pass or fail decision on real photos; the model only suggests and summarises.
```

   **What you should see.**

   The check passes a tick for each field of your plan and one for the evaluation (it counts words and looks for a word like accuracy, confidence, coverage or percent). The answer shows an example card. What a good plan has in common: **(1)** synthetic pictures only where a mistake is cheap (training, never the test set or a report); **(2)** an evaluation on real photos, with a number (the classifier's recall for spalling on real held-out photos, with and without synthetic training pictures) and a human check; **(3)** a named risk specific to this project (unrealistic cracks that teach the classifier the generator's artefacts, or synthetic and real files mixed); **(4)** a licensed engineer making every pass or fail decision. A plan that says "the model could also write the report and decide" fails the last point: in the capstone the model suggests and a licensed engineer decides.

5. **Step 7.5. Your results and four questions.** Everything you saved in `RESULTS` is printed here, so that you can paste it into your report. Then answer four questions in your own words, each with at least one number from this lab.

```python
for k, v in RESULTS.items(): print("%-34s %s" % (k, v))
```

It prints:

```text
p4_abar_T                          2.0390087229316123e-05
p4_params                          35714
p4_final_loss                      0.254
p4_blobs_s42                       8
p4_on_blob_s42                     0.917
p4_blobs_s1                        8
p4_on_blob_s1                      0.912
p4_straightness                    0.084
p4_on_blob_by_steps                [0.307, 0.379, 0.917]
p4_loss_by_bin                     [0.416, 0.562, 0.271, 0.034, 0.003]
p5_cos_100_101                     0.957
p5_cos_100_200                     0.583
p5_notime_loss                     0.543
p5_notime_on_blob                  0.363
p5_notime_on_blob_s1               0.412
p5_unet_params                     135713
p5_share_mid                       0.424
p5_skip_rel_change_untrained       0.02
p5_rf_sketch                       21
p5_rf_real                         35
p6_t_half                          27
p6_t_gone                          67
p6_final_loss                      0.027
p6_train_seconds                   68
p6_judge_test_acc                  1.0
p6_ddpm_conf                       1.0
p6_ddpm_entropy                    1.601
p6_ddpm_coverage                   5
p6_noise_conf                      0.881
p6_noise_entropy                   0.549
p6_ddpm_tmpl                       0.0014
p6_real_tmpl                       0.0013
p6_ddim20_conf                     0.997
p6_ddim20_entropy                  1.576
p6_ddim20_seconds                  1.1
p6_ddim1_conf                      0.899
p6_ddim1_entropy                   0.687
p6_conf_250                        0.88
p6_cov_250                         4
p6_conf_1000                       0.997
p6_rel_skip_change                 1.06
p6_worst_breakage                  (b) no skips, retrained
p6_cfg_acc_w0                      0.213
p6_cfg_acc_w1                      1.0
p6_cfg_acc_w6                      1.0
p6_cfg_ink_w1_w6                   (15.2, 15.2)
p7_ratios                          (48, 64, 4096)
p7_ms_per_call                     4.6
```

   *Exercise 7.5:* Answer in `reflection`: **q1** What are the forward and the reverse process, and which of them involves learning? **q2** Why does the network predict the noise (and not the picture), and what loss does it use? **q3** Why does the denoiser need the step t, and why a UNet for pictures (cite one number from Part 5 or 6.7)? **q4** Compare diffusion with the GAN of Week 12 on training, sampling and coverage (cite numbers: the blobs and the share on a blob, the number of network calls). Why it matters: being able to explain the method in a few sentences with numbers is the real test of understanding. Each answer needs at least 25 words and at least one digit.

```python
reflection = {"q1": "", "q2": "", "q3": "", "q4": ""}    # YOUR TEXT

if all(v.strip() == "" for v in reflection.values()):
    print("Write your four answers in reflection above, then run this cell again.")
else:
    for k, v in reflection.items():
        check("%s: at least 25 words and at least one number" % k, len(v.split()) >= 25 and any(ch.isdigit() for ch in v), "write two or three sentences and cite a number from the lab")
```

   A possible answer (try first, then compare):

```python
reflection = {
    "q1": "The forward process adds Gaussian noise to a picture in T = 100 small steps until only noise is left (abar_100 = 2e-5); it is a fixed recipe and nothing is learned. The reverse process removes noise step by step, and only this part is learned, by the network.",
    "q2": "Predicting the noise gives every training example a known right answer (the noise we drew), the loss is a plain mean squared error and training is as steady as for a classifier; the loss stayed above 0 (0.25 on the blobs) because part of the noise cannot be told apart from the data.",
    "q3": "The same noisy picture means different things at t = 5 and t = 90, so the network must be told t; without it the share of points on a blob fell from 0.92 to about 0.36. A UNet works at several resolutions, so one neuron sees the whole 16 x 16 picture (receptive field 35) while the skips keep the exact positions.",
    "q4": "The GAN needed careful tuning and often lost blobs, while diffusion covered 8 of 8 blobs with about 91 percent of its points on a blob with plain mean squared error training and two seeds; the price is sampling: 100 network calls instead of 1, reduced to 20 by DDIM with little loss in the judge's numbers.",
}
for k, v in reflection.items(): print(k, len(v.split()), "words")
```

These are examples; yours should use your own numbers and words.

It prints:

```text
q1 49 words
q2 53 words
q3 63 words
q4 57 words
```

   **What you should see.**

   The printed `RESULTS` list has about 45 entries. The important ones, all for seed 42: p4_params 35714; p4_final_loss 0.254; p4_blobs_s42 8 with p4_on_blob_s42 0.917 (seed 1: 8 and 0.912); p5_unet_params 135713; p5_share_mid 0.424; p5_notime_on_blob 0.363; p6_ddpm_conf 1.0 with p6_ddpm_entropy 1.601 and p6_ddpm_coverage 5; p6_ddim20_conf 0.997; p6_worst_breakage '(b) no skips, retrained'; p7_ratios (48, 64, 4096). The check passes four ticks when each of your answers has at least 25 words and a number.

   *What it means.* Your four answers are the real test of the lab. A good answer to q4, for example, says that diffusion covered all 8 blobs with 91 to 92 percent of its points on a blob for two seeds (with an ordinary squared-error loss), that the GAN of Week 12 needed tuning and could lose blobs, and that the price of diffusion is 100 network calls per sample against 1 for the GAN (20 with DDIM, with almost no loss in the judge's numbers).

### Deliverables

A short checklist before you submit: every exercise has a written answer under its block; every `None` and every empty string in a starter has been replaced; the code blocks ran from the first to the last without an error (the 2-D denoiser trainings of Part 4 take about 6 to 12 seconds each, a UNet training of Part 6 about 65 to 70 seconds, and the breakage table of Step 6.7 with both seeds about 8 minutes in total); and the printed `RESULTS` summary of Step 7.5 is included. If a run fails (a network does not learn, or your numbers differ from the ones in this lesson), do not hide it: report the numbers you got, the seed, and what you think happened. That is a finding.

Submit your work the way your instructor asks (the notebook with outputs, or your JupyterLab files with the printed output and your written answers). For Parts 4 to 7 the submission must include:

- **Part 4 (16 points):** `make_batch` (4.1) with the closed-form table at t = 10, 30, 60 and 100; `time_embed` (4.2) and the MLP with 35,714 parameters; `train_step` and the loss (about 0.25 at the end, and why it is not 0) (4.3); `reverse_step` and the sampled points, with 8 of 8 blobs and the share on a blob (about 0.92) for two seeds and a sentence on the GAN of Week 12 (4.4); the paths, the five snapshots and the straightness (about 0.08) (4.5); the checkpoint table with 0.31, 0.38 and 0.92 on a blob and what it says about counting blobs (4.6); the loss by group of t and your reading (4.7). Your Imagine-first guess compared with what you measured.
- **Part 5 (16 points):** `embed_by_hand` and the cosine similarities (5.1); the no-time ablation with the loss, the blobs and the share on a blob for two seeds and the loss by group of t (5.2); `TimeBlock` with the formula checked for five blocks (5.3); the UNet with the shapes and 135,713 parameters (5.4); the parameter table and the bottleneck's 42.4 percent (5.5); `pool_up` and the untrained skip test with its honest reading (5.6); `receptive_field` with 7, 21 and 35 (5.7). Your Imagine-first guess compared with what you measured.
- **Part 6 (26 points):** the data in [−1, 1] and `t_half`, `t_gone` (6.1); the trained UNet with its loss curve, the epochs and the time per step (6.2); `to_unit`, the grid of 32 pictures and the denoising film (6.3); the judge, the three rows (real, noise, generated) with the template error (6.4); `ddim_step`, the table for 100, 50, 20, 10, 5, 2 and 1 steps with times, and the grid (6.5); the checkpoint table and your description of the 250-step pictures (6.6); the breakage table for two seeds, `worst_breakage` and your four-sentence answer, including the breakages that did **not** show (6.7); optionally the guidance table (6.8). Your Imagine-first letter compared with Step 6.6.
- **Part 7 (10 points):** the three ratios 48, 64 and 4,096 (7.1); the cost table and the budget answer (7.2); optionally the Stable Diffusion comparison (7.3); your capstone card with five specific fields (7.4); the printed `RESULTS` and your four reflection answers, each citing a number from the lab (7.5).

Together with Parts 1 to 3 the lab is marked out of **100 points**: Part 1 is worth 10, Part 2 is worth 8, Part 3 is worth 14, Part 4 is worth 16, Part 5 is worth 16, Part 6 is worth 26 and Part 7 is worth 10. The rubric is in the section "How Week 13 is assessed".

**Stretch goals.** Train the 2-D denoiser with a cosine schedule (T = 100) and compare the share on a blob with the linear one; sample the 2-D model with DDIM and count blobs at 50, 20 and 10 steps; train the UNet with base 32 channels (about four times the parameters) and compare the judge's numbers and the template error; find the smallest number of training steps at which the 16 × 16 UNet reaches coverage 5; and make the `plan` of Step 7.4 into a one-page proposal for the capstone.
<!-- LAB-END -->

## Check your understanding

{sub} Formative · Not graded · Unlimited attempts · Your best score is saved to your account

Fifteen questions cover the six outcomes; ten of them need a calculation, so keep a calculator or Python open.

```quiz
{
  "id": "week-13-check",
  "title": "Week 13 quiz: Diffusion models",
  "passMark": 70,
  "questions": [
    {
      "q": "Which statement about the two processes of a diffusion model is correct?",
      "options": [
        "Both the noising and the denoising processes are learned by the network",
        "The forward process is a fixed recipe that adds noise, and the reverse process, which removes it, is done by a trained network",
        "The forward process is trained, and the reverse process is a fixed recipe",
        "The forward process makes new pictures, and the reverse process destroys them"
      ],
      "answer": 1,
      "why": "The forward process follows the noise schedule and has no parameters to learn. The learned part is the denoiser that runs the reverse process, so nothing in the forward direction is trained."
    },
    {
      "q": "With the linear schedule ᾱ_300 = 0.3964, so √ᾱ_300 = 0.6296 and √(1 − ᾱ_300) = 0.7769. A pixel has x₀ = 0.5 and the noise draw is ε = −1. What is x_300?",
      "options": [
        "1.0917",
        "−0.4054",
        "−0.4621",
        "−0.7769"
      ],
      "answer": 2,
      "why": "x_300 = 0.6296 × 0.5 + 0.7769 × (−1) = 0.3148 − 0.7769 = −0.4621. The value 1.0917 has a sign error, −0.4054 uses ᾱ instead of its square root, and −0.7769 forgets the picture part."
    },
    {
      "q": "A toy noise schedule has β₁ = 0.02, β₂ = 0.05 and β₃ = 0.1. What is ᾱ_3?",
      "options": [
        "0.8379",
        "0.8300",
        "0.9000",
        "0.1621"
      ],
      "answer": 0,
      "why": "ᾱ_3 is the product (1 − 0.02)(1 − 0.05)(1 − 0.1) = 0.98 × 0.95 × 0.9 = 0.8379. Subtracting the sum of the β (0.83) ignores that the losses multiply, 0.9 is only α_3, and 0.1621 is 1 − ᾱ_3."
    },
    {
      "q": "At some step ᾱ_t = 0.2. What is the signal-to-noise ratio SNR_t = ᾱ_t/(1 − ᾱ_t), and which part of x_t is stronger?",
      "options": [
        "4, so the picture part is stronger",
        "0.2, so the noise part is stronger",
        "0.8, so the picture part is stronger",
        "0.25, so the noise part is stronger"
      ],
      "answer": 3,
      "why": "SNR = 0.2/0.8 = 0.25. Since it is below 1, the noise part is stronger. The value 4 is the inverse ratio, 0.2 is ᾱ itself, and 0.8 is 1 − ᾱ."
    },
    {
      "q": "Compared with a GAN, which description of a diffusion model is the most accurate (typically)?",
      "options": [
        "It makes a picture in one pass but is trained with an unstable game",
        "It is trained with a plain squared error and covers the data well, but needs many network calls to make one picture",
        "It needs a discriminator to tell real from fake",
        "It cannot be conditioned on a text prompt"
      ],
      "answer": 1,
      "why": "Diffusion trains one network on the noise with a squared error and covers the data well, but sampling is a loop of many calls. The first description fits a GAN, a discriminator belongs to GANs, and diffusion is routinely conditioned on text."
    },
    {
      "q": "Why is predicting the noise ε a good training target?",
      "options": [
        "The noise has fewer numbers than the picture, so the network needs fewer weights",
        "A discriminator can then judge the noise guess against real noise",
        "We drew the noise ourselves, so the target is known exactly and training is an ordinary squared error; knowing it also gives the clean picture through the closed form",
        "It removes the need for the forward process"
      ],
      "answer": 2,
      "why": "Since ε is generated by us, the answer is known for every training example, and the closed form turns a noise guess into a picture guess. The noise has as many numbers as the picture, there is no discriminator, and the forward process is still needed to build x_t."
    },
    {
      "q": "The real noise is ε = (1, −1) and the network guesses (0.5, −0.5). What is the mean squared error over the two numbers?",
      "options": [
        "0.50",
        "0.25",
        "1.00",
        "0.00"
      ],
      "answer": 1,
      "why": "The differences are 0.5 and −0.5, with squares 0.25 and 0.25. The sum is 0.50 and the mean over the two numbers is 0.25. The sum of squares, 0.50, is a common slip."
    },
    {
      "q": "One DDPM reverse step has α_t = 0.99, β_t = 0.01, ᾱ_t = 0.25 (so √(1 − ᾱ_t) = 0.8660), x_t = 1.0 and ε̂ = 0.6. With z = 0 (the last step), what is x_(t−1)?",
      "options": [
        "0.4000",
        "0.9931",
        "1.0481",
        "0.9981"
      ],
      "answer": 3,
      "why": "The coefficient is 0.01/0.8660 = 0.011547; 1.0 − 0.011547 × 0.6 = 0.993072; dividing by √0.99 = 0.994987 gives 0.9981. The value 0.9931 forgets to divide by √α_t, 1.0481 wrongly adds fresh noise, and 0.4000 is x_t − ε̂."
    },
    {
      "q": "A DDIM sampler takes 25 steps, and classifier-free guidance needs two network evaluations per step (with and without the prompt). How many network evaluations does one picture need?",
      "options": [
        "50",
        "25",
        "75",
        "1,000"
      ],
      "answer": 0,
      "why": "25 steps × 2 evaluations per step = 50. Counting only 25 forgets the second evaluation, 75 is arithmetic that matches nothing, and 1,000 is the count for a full DDPM run without guidance."
    },
    {
      "q": "A DDIM step has x_t = 1.5, ᾱ_t = 0.25 (so √ᾱ_t = 0.5 and √(1 − ᾱ_t) = 0.8660) and ε̂ = 0.5. What is the estimated clean value x̂₀ = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t?",
      "options": [
        "1.0670",
        "1.0000",
        "2.1340",
        "2.0000"
      ],
      "answer": 2,
      "why": "The numerator is 1.5 − 0.8660 × 0.5 = 1.0670, and dividing by 0.5 gives 2.1340. The value 1.0670 forgets the division, 1.0000 is x_t − ε̂, and 2.0000 forgets to multiply ε̂ by the noise weight."
    },
    {
      "q": "Why does a UNet denoiser have skip connections?",
      "options": [
        "To make the time embedding longer",
        "To reduce the number of channels in the bottleneck",
        "To let the network skip the decoder when the noise is small",
        "To let the decoder reuse the encoder’s full-resolution detail, which shrinking the picture would otherwise lose"
      ],
      "answer": 3,
      "why": "The encoder shrinks the picture to gain context but loses where the fine edges were; the skip copies those early feature maps to the decoder at the same resolution. Skips do not touch the time embedding and they increase, not reduce, the channels the decoder sees."
    },
    {
      "q": "A UNet block has a 3 × 3 convolution from 8 to 16 channels (weights and biases). How many parameters does that convolution have?",
      "options": [
        "1,168",
        "1,152",
        "128",
        "2,320"
      ],
      "answer": 0,
      "why": "9 × 8 × 16 + 16 = 1,152 + 16 = 1,168. The value 1,152 leaves out the 16 biases, 128 forgets the 3 × 3 kernel area, and 2,320 is the count for 16 input channels instead of 8."
    },
    {
      "q": "Why is the step number t given to the denoiser through a time embedding?",
      "options": [
        "So that the network can count how many layers it has",
        "Because the same-looking noisy input needs different handling at t = 20 and at t = 900, and the picture alone does not tell the noise level reliably",
        "Because the sinusoidal formula makes the network deterministic",
        "Because the forward process cannot be written without an embedding"
      ],
      "answer": 1,
      "why": "The right correction at a light dusting differs from the right correction at heavy snow, and one network must serve every level, so it is told t. The embedding has nothing to do with counting layers, determinism or the forward formula."
    },
    {
      "q": "Classifier-free guidance combines ε(∅) = 0.1 and ε(c) = 0.4 with guidance scale w = 5 as ε̃ = ε(∅) + w·(ε(c) − ε(∅)). What is ε̃?",
      "options": [
        "0.50",
        "2.0",
        "1.5",
        "1.6"
      ],
      "answer": 3,
      "why": "ε(c) − ε(∅) = 0.3, times 5 is 1.5, plus 0.1 gives 1.6. The value 0.50 adds the two guesses, 2.0 is w × ε(c), and 1.5 forgets to add the unconditional guess back."
    },
    {
      "q": "A 256 × 256 colour picture is compressed to a 32 × 32 × 4 latent (a factor of 8 on each side, as in Stable Diffusion v1). By what factor does the number of values shrink?",
      "options": [
        "48",
        "8",
        "64",
        "12"
      ],
      "answer": 0,
      "why": "256 × 256 × 3 = 196,608 values against 32 × 32 × 4 = 4,096, and 196,608/4,096 = 48. The factor 8 is the ratio of the sides, 64 is the ratio of the pixel counts only (it ignores that the latent has 4 channels, not 3), and 12 is a different ratio."
    }
  ]
}
```

## How Week 13 is assessed

**Your lab (100 points).**

| Criterion | Points | Excellent | Satisfactory | Needs work |
|---|---|---|---|---|
| Part 1: The forward process by hand (schedule, one step, closed form, noising a picture, SNR, the end of the process) | 10 | `make_schedule` asserted at ᾱ_300 = 0.3964 and ᾱ_1000 = 4.04e-5 with the four curves plotted; the one-pixel step computed and asserted; the closed form compared with 300 iterated steps by mean and variance within 0.02 and the reason (a sum of independent Gaussians) stated; the shape noised at eight steps with ᾱ_t and the measured correlation printed and the point where it becomes unrecognisable named; the SNR table and the step where SNR is 1 found exactly; the end of the process checked as mean near 0 and variance near 1 for five shapes, with the link to starting sampling from N(0, I) | Everything runs and the asserts pass; the explanation of the closed form or of the fixed recipe against the learned generator of Week 12 is brief | The closed form checked against the iterated steps with a different noise sequence and expected to match exactly; ᾱ_t and β_t confused; the correlation not measured |
| Part 2: Noise schedules (cosine schedule, comparison, wasted steps, choice of T) | 8 | The cosine schedule asserted at ᾱ_300 = 0.787 and ᾱ_500 = 0.494; both schedules plotted together with the share of steps with SNR above 1 measured; the same picture noised with both and the difference described; the number of steps with ᾱ_t below 0.001 counted for each and linked to wasted training; ᾱ_T reported for T = 50, 100 and 200 and one T chosen with the rule ᾱ_T below 0.01; a clear sentence on why a schedule is a design choice | Everything runs and the asserts pass; the sentence on wasted steps or on the choice of T is brief | The cosine formula without the division by f(0); β_t not clipped; a T chosen without looking at ᾱ_T |
| Part 3: The reverse step and a perfect denoiser (posterior mean, exact noise predictor, the reverse loop, fresh noise, wrong denoisers) | 14 | The one-pixel reverse step asserted with z = 0 at the last step; `eps_exact` checked against a numerical derivative of the log density; the 200-step loop from N(0, 1) with histograms at five moments, a final mean near 0.5, about half the particles in the left bump and a spread near 0.42; the run without fresh noise with its smaller spread measured and explained; the two wrong predictors (zero and x_t) with measured mean and spread and the reason for each failure; a clear statement of what the network will have to learn | Everything runs and the asserts pass; the explanation of the fresh noise or of the wrong denoisers is brief | The fresh noise added at the last step; the exact predictor taken from the data instead of from the formula; the failure of a wrong predictor described without a number |
| Part 4: Train a denoiser on 2-D data (training pairs, time embedding, MLP, loss and loop, sampling, trajectories, fewer steps, easy and hard noise levels) | 16 | `make_batch` passing the check, with the closed-form table at t = 10, 30, 60 and 100; `time_embed` verified at t = 0 and t = 1; the MLP with 35,714 parameters and a `train_step` that makes the loss fall (about 0.25 after 4,000 steps, not 0); `reverse_step` checked against the formula; a trained model covering 8 of 8 blobs with about 0.92 of the points on a blob for seed 42 and seed 1 (0.917 and 0.912) next to the GAN of Week 12 in one sentence; the paths and five snapshots with the straightness (about 0.08) read as winding; the checkpoint table (0.31, 0.38, 0.92 on a blob) with the lesson that blob counts alone are too generous; the loss by group of t read correctly (highest at t from 21 to 40, almost 0 above t = 80) | Everything runs and the checks pass; the reading of the checkpoint table or of the loss groups goes little beyond the numbers | One t for the whole batch; the 1/√α_t left out of the reverse step; blobs covered quoted without the share on a blob; numbers from a shortened run quoted; one seed trusted |
| Part 5: Time embeddings, the UNet and its parts (embedding by hand, ablation, TimeBlock, UNet shapes, parameters, skips, receptive field) | 16 | `embed_by_hand` equal to `time_embed` at t = 1, 100 and 300 with the heat map and the cosine similarities (0.957 for t = 100 and 101, 0.583 for 100 and 200); the no-time retraining reported with its loss (about 0.54 against 0.25) and the share on a blob (about 0.36 and 0.41 for two seeds against 0.92 and 0.91) and the loss by group of t read honestly; a `TimeBlock` whose parameters equal the formula for all five blocks; the UNet with the right shapes and exactly 135,713 parameters; the parameter table with the bottleneck at 42.4 percent; `pool_up` and the untrained skip test with the relative change reported as small and explained as untrained; `receptive_field` giving 7, 21 and 35 | Everything runs and the checks pass; the explanation of the ablation or of the skip toy is brief | The time embedding described as unnecessary because the network "still works"; shapes guessed without the trace; the block parameter formula not checked; the receptive field computed without the jump |
| Part 6: A DDPM on tiny pictures (data and schedule, the UNet loop, DDPM sampling, the judge, DDIM, training progress, four breakages, optional guidance) | 26 | Pictures scaled to [−1, 1] with the range asserted and `t_half` 27 and `t_gone` 67 found; a UNet trained for 3,000 steps (about 68 seconds, 96 epochs) and a loss near 0.027; a grid of 32 shapes and the denoising film; the judge (test accuracy 1.0) with the real, noise and generated rows (generated: confidence 1.0, entropy 1.601, coverage 5; noise: 0.881, 0.549, 2) and the template error as a number the classifier cannot give; the DDIM step passing the closed-form check and the table for 100, 50, 20, 10, 5, 2 and 1 steps with times, read honestly (still good at 5 steps, breaking at 1); the checkpoints (250: coverage 4, 1,000, 3,000) with pictures described; the breakage table for two seeds with an honest reading: the one that changed the judge (no skips), the ones that did not (no time input, wrong sampler, 5 steps), what the template error and the pictures add, and the relative output change of about 1.1 when skips are zeroed; optionally guidance with the accuracy at w = 0 and w = 1 | Everything runs and the checks pass; the reading of the breakage table or of the DDIM table is brief | Pictures not scaled to [−1, 1]; the judge not frozen; judge confidence read as quality; a breakage claimed to hurt when the table does not show it (or the reverse); one seed in the breakage table; numbers from a shortened run quoted |
| Part 7: Guidance, latent space, responsible use and the capstone (latent arithmetic, cost of sampling, optional GPU stretch, capstone plan, reflection) | 10 | Ratios 48, 64 and 4,096 computed and asserted; the cost table with the measured time per call and the budget answer (1.2 calls per picture, so no DDIM step count fits) with the 0.05 s assumption stated as an assumption; a capstone card with all five fields specific to the Structural Inspection Assistant, a measurable evaluation on real photos only, a named risk of synthetic images mixed with evidence and a licensed engineer who decides; the printed `RESULTS`; four reflection answers that each cite a number from the lab | Everything runs and the checks pass; the card or the reflection is generic | A card that gives the model any decision a licensed engineer must make; synthetic pictures proposed for the test set; reflection answers without a number from the lab |

**Across the course.** Laboratories and knowledge checks run through the term, with exams in Weeks 5, 10, 15 and 20, plus a capstone. Your instructor announces the component weights in the syllabus. The long-term goal is the NVIDIA-Certified Associate exam (NCA-GENL).

## Exit ticket and homework

Answer these three questions in one or two sentences each before you leave:
1. Describe the forward process and the reverse process of a diffusion model in your own words. Which one is learned, and what does the network output?
2. At t = 300 of the linear schedule, ᾱ is 0.3964. Which has the bigger weight in x_300, the picture or the noise, and by how much (give both weights)?
3. Why does the denoiser need the step t as an input, and what two things about a UNet make it a good fit for predicting the noise?

**Homework / self-study**
- Finish the lab exercises, the Part 7 capstone plan and the reflection.
- Re-read the `equation` figures of this lesson (one forward step, ᾱ, the closed form, the SNR, the loss, the reverse step, the DDIM step, the time embedding, classifier-free guidance, the latent ratio) and redo each worked example with different numbers.
- Read the [Capstone](/capstone) page and write down one question about how your team might use a generative model, or why it might not.
- Preview Week 14: fine-tuning, RLHF and parameter-efficient methods (PEFT, such as LoRA), which show how a pre-trained model is adapted into a helpful assistant.
