# %% [markdown]
# # Week 12 — GANs and Image Generation: lab notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive**, so your changes are kept in your own copy.
#
# **Runtime:** CPU is enough (no GPU needed, no downloads). **Time:** about 3 hours; the computing itself takes about 2 minutes.
#
# **What you will do**
# 1. Predict the output size of a transpose convolution by hand, then check it with PyTorch.
# 2. Trace the shapes of a small image generator that grows a 4 × 4 grid into 32 × 32.
# 3. Train a toy GAN on a two-hump distribution and plot its samples every few hundred steps.
# 4. Break the balance between the two networks and measure what goes wrong.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.

# %%
import os, math, random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)    # the toy GAN below is tiny, so it runs on the CPU either way
QUICK = bool(os.environ.get("NB_TEST"))

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1 — Transpose-convolution sizes
#
# From the lesson section "Transpose convolution": with input side n, kernel k, stride s and padding p, PyTorch's `ConvTranspose2d` (no dilation) gives an output side of **(n − 1) × s − 2p + k**, plus `output_padding` if you set it. We write the formula as a function and compare it with PyTorch for the four settings in the lesson's table.

# %%
def out_size(n, k, s, p=0, op=0):
    return (n - 1) * s - 2 * p + k + op

for n, k, s, p in [(2, 3, 2, 0), (4, 4, 2, 1), (8, 4, 2, 1), (7, 3, 2, 1)]:
    layer = nn.ConvTranspose2d(1, 1, kernel_size=k, stride=s, padding=p)
    y = layer(torch.zeros(1, 1, n, n))
    print(n, k, s, p, "formula:", out_size(n, k, s, p), "PyTorch:", y.shape[-1])
    assert y.shape[-1] == out_size(n, k, s, p)

# %% [markdown]
# ### ✏️ Your turn: predict, then check
#
# Predict the output side for a transpose convolution with input side `n = 5`, kernel `k = 3`, stride `s = 2`, padding `p = 1` and `output_padding = 1`. Write the prediction first, then the Check runs PyTorch to see if you were right.

# %%
prediction = None   # YOUR CODE HERE: an integer

# %%
# Check
if prediction is None:
    print("Fill in your answer above, then run this cell again.")
else:
    layer = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
    real = layer(torch.zeros(1, 1, 5, 5)).shape[-1]
    check(f"the output side is {real}", prediction == real, "(n-1)*s - 2p + k + output_padding")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# prediction = out_size(5, 3, 2, 1, 1)   # (5-1)*2 - 2 + 3 + 1 = 10
# ```
#
# `output_padding` just adds extra rows and columns on one side, which is useful when you need an exact even size.
# </details>

# %% [markdown]
# ### The 2 × 2 example by hand
#
# The lesson's example: input `[[1, 2], [3, 4]]`, a 3 × 3 kernel of all ones, stride 2. Each input value multiplies the kernel and the 3 × 3 copies are placed 2 steps apart and **added where they overlap**. The output is 5 × 5. We reproduce it with `ConvTranspose2d` and fixed weights.

# %%
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])[None, None]
tc = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, bias=False)
with torch.no_grad():
    tc.weight.fill_(1.0)                    # the kernel of all ones
y = tc(x)[0, 0]
print(y)
assert y.shape == (5, 5) and y[2, 2].item() == 10.0   # the centre cell receives 1 + 2 + 3 + 4

# %% [markdown]
# ### ✏️ Your turn: read the corners and the middle row
#
# Set `corner` to the value in the top-left cell, and `middle_row` to the list of five values in the middle row (row index 2). Use the printed matrix, then say in a comment why the corner differs from the centre.

# %%
corner = None        # YOUR CODE HERE
middle_row = None    # YOUR CODE HERE, e.g. [a, b, c, d, e]

# %%
# Check
if corner is None or middle_row is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("top-left corner", float(corner) == y[0, 0].item(), "only the input value 1 reaches the top-left corner")
    check("middle row", [float(v) for v in middle_row] == y[2].tolist(), "copy the third row of the printed matrix")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# corner = 1.0                       # only input 1 touches that cell
# middle_row = [4.0, 4.0, 10.0, 6.0, 6.0]
# ```
#
# The corner comes from a single input (1), while the centre cell is covered by all four copies, so it holds 1 + 2 + 3 + 4 = 10. Uneven overlap like this is one reason some generators show a checkerboard pattern.
# </details>

# %% [markdown]
# ## Step 2 — A small image generator, shapes only
#
# A generator that grows a 16-number noise vector into a 32 × 32 image. Each `ConvTranspose2d(…, 4, 2, 1)` doubles the side (k = 4, s = 2, p = 1 gives exactly 2n). We trace every shape.

# %%
gen = nn.Sequential(
    nn.Linear(16, 64 * 4 * 4), nn.ReLU(),
    nn.Unflatten(1, (64, 4, 4)),
    nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
    nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
    nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Tanh(),
)
z = torch.randn(5, 16)
for layer in gen:
    z = layer(z)
    print(layer.__class__.__name__, tuple(z.shape))
assert tuple(z.shape) == (5, 1, 32, 32)
print("trainable parameters:", sum(q.numel() for q in gen.parameters()))

# %% [markdown]
# ### ✏️ Your turn: count the last layer
#
# Count the parameters of the last transpose convolution by hand: weights `16 × 1 × 4 × 4` plus one bias per output channel. Set `answer`.

# %%
answer = None   # YOUR CODE HERE

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    real = sum(q.numel() for q in gen[-2].parameters())
    check(f"the last layer has {real} parameters", answer == real, "16*1*4*4 weights + 1 bias")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = 16 * 1 * 4 * 4 + 1    # 257
# ```
#
# A transpose convolution stores its weights as (in_channels, out_channels, k, k), so the count is 16 × 1 × 4 × 4 = 256, plus 1 bias.
# </details>

# %% [markdown]
# ## Step 3 — Set up the toy GAN
#
# The real data are two Gaussian humps centred at −2 and +2 with standard deviation 0.5, so about half of the real samples are positive. The generator G turns random noise into one number; the discriminator D scores a number as real or fake. `D` has no sigmoid at the end because the loss we use, `BCEWithLogitsLoss`, applies the sigmoid itself.

# %%
Z = 8   # size of the noise vector

def real_batch(n):
    centers = torch.randint(0, 2, (n, 1)).float() * 4 - 2     # -2 or +2, each half the time
    return centers + 0.5 * torch.randn(n, 1)

def make_G():
    return nn.Sequential(nn.Linear(Z, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

def make_D():
    return nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

r = real_batch(5000)
print("real mean", round(r.mean().item(), 3), "fraction > 0:", round((r > 0).float().mean().item(), 3))
assert abs(r.mean().item()) < 0.15 and abs((r > 0).float().mean().item() - 0.5) < 0.05

# %% [markdown]
# ## Step 4 — Train it and watch the samples
#
# This is the loop from the lesson: update D on real and fake batches, then update G through D. Every `every` steps we save 2000 samples from G so we can plot how they change. The `.detach()` in the D step stops D's loss from sending gradients into G.

# %%
def train(steps=3000, lr_g=1e-3, lr_d=1e-3, batch=128, every=500):
    torch.manual_seed(SEED)
    G, D = make_G(), make_D()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ones, zeros = torch.ones(batch, 1), torch.zeros(batch, 1)
    snaps, dfake = {}, []
    for step in range(1, steps + 1):
        fake = G(torch.randn(batch, Z)).detach()                     # detach: do not train G here
        loss_d = bce(D(real_batch(batch)), ones) + bce(D(fake), zeros)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()
        fake = G(torch.randn(batch, Z))
        loss_g = bce(D(fake), ones)                                  # G wants D to say "real" (label 1)
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        dfake.append(torch.sigmoid(D(fake)).mean().item())
        if step == 1 or step % every == 0:
            with torch.no_grad():
                snaps[step] = G(torch.randn(2000, Z)).squeeze().numpy()
    return snaps, dfake

steps = 300 if QUICK else 3000
samples, dfake = train(steps=steps, every=steps // 6)
fig, ax = plt.subplots(1, len(samples), figsize=(15, 2.3), sharex=True, sharey=True)
for a_, (step, s) in zip(ax, samples.items()):
    a_.hist(s, bins=40, range=(-4, 4)); a_.set_title(f"step {step}", fontsize=9)
plt.tight_layout(); plt.show()
print("D(fake) over the last 200 steps:", round(float(np.mean(dfake[-200:])), 3))
final = samples[steps]
print("fraction of samples > 0:", round(float((final > 0).mean()), 3), " std:", round(float(final.std()), 3))
assert np.isfinite(final).all()

# %% [markdown]
# ### ✏️ Your turn: read the histograms
#
# Set `final_dfake` to the average D(fake) shown above (rounded to 2 decimals) and `both_humps` to `True` if the last histogram has mass near both −2 and +2, otherwise `False`. In a comment, say what a D(fake) of 0.5 would mean.

# %%
final_dfake = None   # YOUR CODE HERE
both_humps = None    # YOUR CODE HERE: True or False

# %%
# Check
if final_dfake is None or both_humps is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true_d = float(np.mean(dfake[-200:]))
    true_both = bool(((final > 1).mean() > 0.1) and ((final < -1).mean() > 0.1))
    check(f"D(fake) is about {true_d:.2f}", abs(final_dfake - true_d) < 0.02, "average the last 200 values of dfake")
    check(f"both humps covered: {true_both}", both_humps == true_both, "look at the last panel: is there mass at both -2 and +2?")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# final_dfake = round(float(np.mean(dfake[-200:])), 2)
# both_humps = bool(((final > 1).mean() > 0.1) and ((final < -1).mean() > 0.1))
# ```
#
# If D(fake) were exactly 0.5, D could not tell real from fake: it is guessing. GAN runs are noisy, so your numbers may differ slightly from a classmate's.
# </details>

# %% [markdown]
# ## Step 5 — Break the balance
#
# A GAN works only if neither network wins too easily (lesson section "When training goes wrong"). We run the same loop with a much stronger D, a much stronger G, and the baseline. For each run we record the fraction of samples above 0 (the real value is about 0.5), the spread of the samples, and the final D(fake). Results vary, so we report what happens rather than what we expect.

# %%
runs = {
    "D much stronger": dict(lr_d=1e-2, lr_g=1e-4),
    "G much stronger": dict(lr_d=1e-4, lr_g=1e-2),
    "Baseline": dict(),
}
summary = {}
fig, ax = plt.subplots(1, 3, figsize=(11, 2.4), sharex=True, sharey=True)
for a_, (name, kw) in zip(ax, runs.items()):
    s, d = train(steps=steps, every=steps, **kw)
    last = s[steps]
    summary[name] = dict(frac=float((last > 0).mean()), std=float(last.std()), dfake=float(np.mean(d[-200:])))
    a_.hist(last, bins=40, range=(-4, 4)); a_.set_title(name, fontsize=9)
    print(f"{name:16} fraction>0: {summary[name]['frac']:.2f}  std: {summary[name]['std']:.2f}  D(fake): {summary[name]['dfake']:.2f}")
plt.tight_layout(); plt.show()
assert all(np.isfinite(v["frac"]) for v in summary.values())

# %% [markdown]
# ### ✏️ Your turn: which run is furthest from the real data?
#
# The real data have a fraction above 0 of about 0.5. Set `worst_run` to the name of the run whose fraction is **furthest from 0.5** (this is a sign of everything piling into one hump). Write in a comment one change you would try to fix it.

# %%
worst_run = None   # YOUR CODE HERE: one of the three run names

# %%
# Check
if worst_run is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true = max(summary, key=lambda k: abs(summary[k]["frac"] - 0.5))
    check(f"furthest from 0.5: {true}", worst_run == true, "compute abs(fraction - 0.5) for each run and take the largest")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# worst_run = max(summary, key=lambda k: abs(summary[k]["frac"] - 0.5))
# ```
#
# A fraction far from 0.5 means one hump is missing, which is the signature of mode collapse; a D(fake) near 0 would instead point to a discriminator far ahead of G. Things people often try: lower or balance the learning rates, train D a little less or more evenly, or use Adam with beta1 around 0.5 as here. These are not guarantees, so look at the samples, not only at the losses.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - A transpose convolution has output side (n − 1) × s − 2p + k, and k = 4, s = 2, p = 1 doubles the size: it is learnable upsampling.
# - A GAN is two networks trained against each other; you judge it by looking at samples, not only at the losses.
# - Training can go wrong when one network gets ahead of the other: the samples can lose a hump (mode collapse) or the discriminator can leave G no useful gradient.
#
# **Reflect:** double-click to write two sentences: why does the generator need transpose convolutions but the discriminator does not, and why can a GAN with good-looking losses still be a poor generator?
#
# Files in `/content` vanish when the session ends, so download anything you want to keep. Submit this notebook the way your instructor asks.
#
# **Go further:** UvA Deep Learning Tutorial 9, [Autoencoders](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html), by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence). It builds an encoder with strided convolutions and a decoder with `ConvTranspose2d` that grows a 4×4 grid back to 32×32 on CIFAR10. It downloads data and checkpoints from the internet, so run it in its own Colab or on a laptop.
