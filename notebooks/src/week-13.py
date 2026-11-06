# %% [markdown]
# # Week 13 — Diffusion Models: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is fine for everything; no GPU is needed (a GPU is used only by one optional experiment at the end). Nothing is downloaded: every data set is drawn with NumPy. **Time:** about 3 hours for this notebook. Every cell finishes in seconds to about two minutes.
#
# This lab is called **Imagine the sculptor and the fog**. Week 13 asks how a network can make a picture by *removing noise*. You answer by building the story from the bottom: the forward process that destroys a picture step by step, noise schedules compared, the reverse step and an exact denoiser that turns noise into two bumps with no network at all, a first trained denoiser, the time embedding and the UNet, a complete diffusion model (DDPM) on small pictures with fast sampling (DDIM), and a plan for the capstone.
#
# ### What you will do
#
# 1. **Part 1. The forward process by hand** (25 min): the noise schedule, one step, the closed form, noising a real picture, the signal-to-noise ratio and the pure-noise end.
# 2. **Part 2. Noise schedules** (20 min): the cosine schedule, linear versus cosine, wasted steps, the number of steps T, and a schedule of your own.
# 3. **Part 3. The reverse step and a perfect denoiser** (25 min): the reverse step by hand, the exact noise predictor for two bumps, the reverse loop, fresh noise, and wrong denoisers.
# 4. **Part 4. Train a denoiser on 2-D data** (25 min): training pairs, a small network with a time input, the loss, sampling, trajectories and fewer training steps.
# 5. **Part 5. Time embeddings, the UNet and its parts** (30 min): the sinusoidal time embedding, an ablation without time, a block with a time input, a small UNet, parameters by part, skip connections and the receptive field.
# 6. **Part 6. A DDPM on tiny pictures** (35 min): train a UNet on the line shapes, sample with DDPM and DDIM, judge the pictures with a classifier, break the model on purpose, and (optional) classifier-free guidance.
# 7. **Part 7. Guidance, latent space, responsible use and the capstone** (10 min): latent diffusion arithmetic, the cost of sampling, an optional Stable Diffusion experiment, a capstone plan and reflection.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells (the exercises) are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried.

# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed so your results match the ones in the lesson, and defines helpers (the same ones you used in Week 12, so nothing is new here): `check(...)` to test your answers; data sets drawn with NumPy so that nothing is downloaded: `sample_real_1d(...)` (numbers from a bell curve), `make_mixture(...)` (points from eight Gaussian blobs arranged in a ring) and `make_shapes(...)` (the five line shapes of Week 4, the "pictures" the diffusion model will learn to draw); `modes_covered(...)` to count how many of the blobs a generator reaches and how many of its points land on a blob; `show_grid(...)` to look at a grid of small pictures; `count_params(...)` and `train_classifier(...)` (Week 4) to train the picture classifier that we use as a judge of generated pictures. The noise schedule, the denoising networks and the samplers are **not** given: you will build them in the lab. Everything runs on a CPU in a few minutes.

# %%
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
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone
print("Running on:", device)

RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

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

# %% [markdown]
# ## Part 1. The forward process by hand (25 min)
#
# *Lesson section: "Forward".* Everything in this week rests on one fixed recipe: **take a picture and destroy it slowly, in many tiny steps, by adding a little random noise each time.** This is the *forward process*. Think of a photograph on a window in heavy rain, or a barangay mural slowly washed by floods: at first you can still read every detail, after a while only the big shapes are left, and in the end there is only grey fog. Nothing in this recipe is learned. It is a formula, and because it is a formula, we can compute exactly what the picture looks like after any number of steps, and we know exactly which noise was added. That known noise is the answer key that the network of the next parts will be trained on. In this part you build the recipe by hand: the schedule that says how much noise each step adds, one step, the shortcut that jumps straight to step t, a real picture disappearing into fog, the signal-to-noise ratio that measures how much of the picture is left, and the proof by experiment that the end of the process is pure noise.

# %% [markdown]
# ### ✏️ Imagine first
#
# Picture a clean 16 by 16 picture of a cross. We add noise in 1,000 tiny steps, a little more each time. After 300 steps, what is left of the original? Before you run any code, write two predictions. (1) How much of the picture's *strength* is left after 300 of the 1,000 steps, as a percentage between 0 and 100 (100 means nothing has changed, 0 means nothing is left)? (2) At which step t do you think a person can no longer recognise the cross at all? You compare with Steps 1.1 and 1.4.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: the fog
my_guess_left = None   # percent of the original picture's strength that is left after 300 of 1,000 noising steps (0 to 100)
my_guess_gone = None   # the step t (1 to 1000) at which you think a shape can no longer be recognised

# %% [markdown]
# ### Step 1.1. The noise schedule
#
# The **noise schedule** is the list of numbers beta_1, beta_2, ..., beta_T that says how much noise step t adds. We use the **linear schedule** of the original paper: beta grows in a straight line from 0.0001 (a whisper) to 0.02 over T = 1,000 steps. From beta we compute alpha_t = 1 - beta_t (how much of the picture survives step t) and the running product **abar_t** (written ᾱ_t in the lesson), the product alpha_1 * alpha_2 * ... * alpha_t: how much of the original picture survives *all t steps together*. We store each list with index 0 meaning "no step yet" (beta_0 = 0, abar_0 = 1), so that `abar[300]` really is abar_300. The block builds the schedule, prints the numbers of the lesson and draws the four curves that matter: beta_t, abar_t, the weight of the picture sqrt(abar_t) and the weight of the noise sqrt(1 - abar_t).

# %%
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

# %% [markdown]
# You should see three printed lines and a figure with two panels. The first line shows that beta_1 is 0.0001, beta_300 is 0.00606 and beta_1000 is 0.02: each single step adds very little noise. The second line is the answer to the first Imagine question: abar_300 is 0.3964, so the weight of the clean picture in x_300 is sqrt(abar_300) = 0.630 and the weight of the noise is sqrt(1 - abar_300) = 0.777. In words: after 300 of the 1,000 steps, **63 percent of the picture's strength is left** (but only 39.6 percent of its power, which is the square of the strength), and the noise already has the bigger weight. At the end abar_1000 is 4.04e-5, which is 0.00004: nothing of the picture remains. In the figure, the left panel shows beta climbing in a straight line, and the right panel shows abar_t falling like a slide: slowly at first, quickest in the middle, flat near zero at the end. The picture weight sqrt(abar_t) and the noise weight sqrt(1 - abar_t) cross where abar = 0.5, at about step 260. The silent assertion checks the two numbers of the lesson (0.3964 and 4.04e-5).

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.1. A faster schedule
#
# Your task: a *faster* schedule: let beta grow from 0.0001 to 0.04 instead of 0.02 (call `make_schedule(1000, 1e-4, 0.04)`). Store the value of abar at t = 300 in `abar_fast_300`, and the first step at which abar is below 0.001 (the picture is gone for all practical purposes) in `t_gone_fast`. Why it matters: the schedule is a choice, and you can see right away how a more aggressive choice changes how many steps are really needed.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
abar_fast_300 = None   # abar at t = 300 for the faster schedule
t_gone_fast = None     # the first step t at which abar is below 0.001

# %%
if abar_fast_300 is None or t_gone_fast is None:
    print("Fill in abar_fast_300 and t_gone_fast above, then run this cell again.")
else:
    _, _, ab2 = make_schedule(1000, 1e-4, 0.04)
    check("abar_fast_300 is right", abs(abar_fast_300 - ab2[300]) < 1e-4, "Call make_schedule(1000, 1e-4, 0.04) and read abar[300].")
    check("abar_fast_300 is smaller than for the slow schedule", abar_fast_300 < abar[300], "More noise per step leaves less of the picture.")
    check("t_gone_fast is right", t_gone_fast == int(np.argmax(ab2 < 0.001)), "Find the first index where abar is below 0.001 (np.argmax of a boolean array gives the first True).")
    RESULTS["p1_fast"] = (round(float(abar_fast_300), 4), int(t_gone_fast))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# _, _, abar_fast = make_schedule(1000, 1e-4, 0.04)
# abar_fast_300 = float(abar_fast[300])
# t_gone_fast = int(np.argmax(abar_fast < 0.001))
# print(abar_fast_300, t_gone_fast)
# ```
#
# With beta ending at 0.04, abar_300 drops to 0.1606 (against 0.3964), and the picture is gone (abar below 0.001) at step 584 instead of step 826. Doubling the final beta does not halve the time: the effect compounds, because abar is a product.
#
# </details>

# %% [markdown]
# ### Step 1.2. One forward step
#
# One step of the forward process is: x_t = sqrt(1 - beta_t) * x_(t-1) + sqrt(beta_t) * eps, where eps is a fresh random number from the standard bell curve N(0, 1). The first part *shrinks* the old picture a tiny bit; the second part *adds* a tiny bit of fresh noise. The two weights are chosen so that the total variance stays the same (we check this in the exercise). We use the example of the lesson, one pixel with the value 0.8, beta = 0.02 and a noise draw of eps = 0.5, and then apply five steps one after the other with a hand-written loop to a picture with four pixels, so that you see the pixels drift.

# %%
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

# %% [markdown]
# You should see the three pieces of the worked example and then five lines for the four-pixel picture. The signal part is sqrt(0.98) * 0.8 = 0.7920 (the pixel shrank from 0.8 to 0.792), the noise part is sqrt(0.02) * 0.5 = 0.0707, and the new pixel is **0.8627**, the number in the lesson. In the five-step table the four pixels start at 0.8, -0.5, 0.0 and 1.0 and wander away from these values in random directions: after five steps the first pixel has moved from 0.8 to 0.622, the third from 0.0 to -0.129. After only five steps the picture is still easy to recognise. Each step is a small push, and 1,000 small pushes add up. The assertion checks 0.8627 to three decimals.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.2. Variance preserving
#
# Your task: check the claim that a forward step *preserves the variance*. Take 100,000 pixels that all start at exactly 1.0 (so they have no spread at all), apply one step with beta = 0.02 and store the variance of the result in `var_const`. Then take 100,000 pixels drawn from N(0, 1) (variance 1), apply one step and store the variance of the result in `var_normal`. Use a `RandomState(SEED)` for the noise. Why it matters: a process that kept growing or shrinking in size would be impossible to train a network on; because the variance stays at 1, the picture and the noise always live on the same scale.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
var_const = None    # variance after one step (beta = 0.02) of 100,000 pixels that all start at exactly 1.0
var_normal = None   # variance after one step (beta = 0.02) of 100,000 pixels that start as N(0, 1) numbers

# %%
if var_const is None or var_normal is None:
    print("Fill in var_const and var_normal above, then run this cell again.")
else:
    check("var_const is close to beta = 0.02", abs(var_const - 0.02) < 0.003, "All pixels start equal, so the only spread is the new noise: sqrt(beta) * eps has variance beta.")
    check("var_normal is close to 1", abs(var_normal - 1.0) < 0.03, "(1 - beta) * 1 + beta * 1 = 1: the step keeps the variance.")
    RESULTS["p1_var"] = (round(float(var_const), 4), round(float(var_normal), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# rng = np.random.RandomState(SEED)
# n = 100000
# var_const = float(q_step(np.ones(n), 0.02, rng.randn(n)).var())
# var_normal = float(q_step(rng.randn(n), 0.02, rng.randn(n)).var())
# print(var_const, var_normal)
# ```
#
# `var_const` is 0.0200, almost exactly beta = 0.02: all the spread comes from the new noise, with weight sqrt(0.02), whose variance is 0.02. `var_normal` is 0.997, close to 1: (1 - beta) * 1 + beta * 1 = 1. The small differences are sampling noise from using 100,000 numbers instead of infinitely many. This is why the process is called *variance preserving*: if the picture has variance 1 and the noise has variance 1, so does the result.
#
# </details>

# %% [markdown]
# ### Step 1.3. The closed form: jump to step t
#
# Running 1,000 steps one after the other is slow, and there is a shortcut. A pixel after t steps is a sum of the original pixel (shrunk many times) and many independent bell-curve noises (each shrunk by the later steps). **A sum of independent bell curves is again a bell curve, with the variances added**, so all those noises together are just *one* bell-curve noise. The result is the **closed form**: x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps, with a single eps from N(0, 1). It lets us jump from the clean picture to *any* step t in one move, which is what makes training fast. The block checks this claim by experiment. It starts 10,000 copies of one pixel (x_0 = 0.8) and (a) runs 300 single steps with fresh noise each time, and (b) makes the same number of samples with one jump to t = 300. The two groups do not use the same random numbers, so we cannot compare them number by number: we compare their mean and their variance, and the theory (mean sqrt(abar_300) * 0.8, variance 1 - abar_300).

# %%
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

# %% [markdown]
# You should see three lines. With 300 single steps the 10,000 samples have a mean of 0.500 and a variance of 0.604; with the closed form they have a mean of 0.505 and a variance of 0.601; the theory says 0.504 and 0.604. The three agree within the sampling error (the assertion allows 0.02 in the mean and in the variance). So 300 little pushes and one big push give the same bell curve: **the closed form is exact, not an approximation.** Notice also what the numbers mean: a pixel that started at 0.8 is now a bell curve centred at 0.504, with a standard deviation of sqrt(0.604) = 0.777. Whatever the original value, the noise has a spread of 0.777 around it, bigger than the shift of the mean.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.3. Predict, then measure
#
# Your task: predict first, then measure. For a pixel x_0 = 0.8 at step t = 600, write the mean and the variance that the closed form predicts in `theory_mean_600` and `theory_var_600`. Then make 10,000 samples with `q_sample` at t = 600 (use `rng = np.random.RandomState(SEED)`) and store their mean and variance in `emp_mean_600` and `emp_var_600`. Why it matters: the closed form is the line you will use inside the training loop in Part 4, so you should trust it and know what it predicts.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
theory_mean_600 = None   # the mean of x_600 for a pixel x0 = 0.8, from the formula
theory_var_600 = None    # the variance of x_600, from the formula
emp_mean_600 = None      # the mean of 10,000 samples made with q_sample at t = 600
emp_var_600 = None       # their variance

# %%
if None in (theory_mean_600, theory_var_600, emp_mean_600, emp_var_600):
    print("Fill in the four variables above, then run this cell again.")
else:
    check("theory mean is right", abs(theory_mean_600 - np.sqrt(abar[600]) * 0.8) < 1e-3, "mean = sqrt(abar_t) * x0.")
    check("theory variance is right", abs(theory_var_600 - (1 - abar[600])) < 1e-3, "variance = 1 - abar_t, because eps has variance 1.")
    check("the samples agree with the theory (0.03)", abs(emp_mean_600 - theory_mean_600) < 0.03 and abs(emp_var_600 - theory_var_600) < 0.03, "Use q_sample(np.full(10000, 0.8), 600, rng.randn(10000), abar).")
    RESULTS["p1_closed"] = (round(float(theory_mean_600), 3), round(float(theory_var_600), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# theory_mean_600 = float(np.sqrt(abar[600]) * 0.8)
# theory_var_600 = float(1 - abar[600])
# rng = np.random.RandomState(SEED)
# s = q_sample(np.full(10000, 0.8), 600, rng.randn(10000), abar)
# emp_mean_600, emp_var_600 = float(s.mean()), float(s.var())
# print(theory_mean_600, theory_var_600, emp_mean_600, emp_var_600)
# ```
#
# The theory gives a mean of 0.129 and a variance of 0.974; the samples give 0.127 and 0.981. At t = 600 the picture weight is only sqrt(abar_600) = 0.161, so the pixel value 0.8 has almost disappeared: the mean has shrunk from 0.8 to 0.129, and the variance is very close to 1.
#
# </details>

# %% [markdown]
# ### Step 1.4. Noising a real picture
#
# Now we apply the closed form to a real picture. We take the cross from `make_shapes` without any noise (`noise=0.0`), scale it from the range [0, 1] to [-1, 1] (the diffusion models of this week work with pictures between -1 and 1, so that black is -1 and white is +1, and the noise, which is centred at 0, sits in the middle of the range), and draw **one** noise picture eps. Using the *same* eps for every t makes the pictures a film of one picture disappearing: only the amount of noise changes. We show t = 0, 50, 100, 200, 300, 500, 800 and 1000, print abar_t for each, and measure with the **correlation** between x_t and x_0 how much of the shape is still there (1.0 means the same picture, 0 means unrelated).

# %%
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

# %% [markdown]
# You should see a row of eight small pictures and a table. At t = 0 the cross is clean. At t = 50 (abar = 0.971) it is still sharp, with a faint speckle; at t = 100 (abar = 0.897) the speckle is clearly visible, but the cross is easy to see; at t = 200 (abar = 0.659) the cross is still recognisable, with a lot of grain; at t = 300 (abar = 0.396) you have to look for it; at t = 500 (abar = 0.079) you can at best guess the pattern; at t = 800 and t = 1000 there is only static. The correlation says the same in numbers: 1.000, 0.946, 0.828, 0.562, 0.357, 0.116, -0.012 and -0.029. So for this picture and this noise draw, a human eye gives up somewhere between t = 300 and t = 500, while the last 500 steps add nothing that a person could see. (Your eye is a detective, as in Week 12; a network will do better than you here, because it can use the tiny remaining correlation.) Answer to the second Imagine question: look at your own guess, and compare it with the step where the cross disappears in your figure.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.4. When does the shape fade?
#
# Your task: find the first step t at which the correlation between x_t and x_0 (using the same `eps`) falls below 0.5 and store it in `t_corr_half`; then the first step at which it falls below 0.1, in `t_corr_tenth`. Loop over t = 0, 1, 2, ... 1000 and call `q_sample(x0, t, eps, abar)`. Why it matters: it turns "at which step is the shape gone?" from a matter of eyesight into a number.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
t_corr_half = None   # the first step t at which the correlation between x_t and x_0 falls below 0.5
t_corr_tenth = None  # the first step t at which it falls below 0.1

# %%
if t_corr_half is None or t_corr_tenth is None:
    print("Fill in t_corr_half and t_corr_tenth above, then run this cell again.")
else:
    cs = [np.corrcoef(q_sample(x0, t, eps, abar).ravel(), x0.ravel())[0, 1] for t in range(1001)]
    check("t_corr_half is right", t_corr_half == next(t for t in range(1001) if cs[t] < 0.5), "Loop over t = 0, 1, 2, ... and stop at the first correlation below 0.5.")
    check("t_corr_tenth is right", t_corr_tenth == next(t for t in range(1001) if cs[t] < 0.1), "Same loop, threshold 0.1.")
    check("the order is right", t_corr_half < t_corr_tenth, "The picture fades: the correlation only goes down.")
    RESULTS["p1_fade"] = (int(t_corr_half), int(t_corr_tenth))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# corr = [np.corrcoef(q_sample(x0, t, eps, abar).ravel(), x0.ravel())[0, 1] for t in range(1001)]
# t_corr_half = next(t for t in range(1001) if corr[t] < 0.5)
# t_corr_tenth = next(t for t in range(1001) if corr[t] < 0.1)
# print(t_corr_half, t_corr_tenth)
# ```
#
# The correlation falls below 0.5 at t = 227 and below 0.1 at t = 522. So **about a fifth of the way through the process, half of the shape is gone, and at about half-way only a tenth is left**. Look at what this implies for training: the interesting part, where the picture changes from clear to gone, happens in the first 500 steps of 1,000.
#
# </details>

# %% [markdown]
# ### Step 1.5. Signal-to-noise ratio
#
# The **signal-to-noise ratio** (SNR) is the one number that says how hard denoising is at step t: SNR_t = abar_t / (1 - abar_t), the power of the picture divided by the power of the noise. If the SNR is 100 the picture is 100 times stronger than the noise (an easy case); if it is 1 they are equally strong; if it is 0.01 the noise is 100 times stronger (hopeless to the eye). Use it to compare different schedules fairly. The block prints the SNR at t = 1, 50, 100, 300, 500 and 1000, finds the step where the SNR crosses 1 (abar = 0.5), and plots the SNR on a *log axis*, because it falls over many orders of magnitude.

# %%
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

# %% [markdown]
# You should see a table, one line about the crossing point and a falling curve. The SNR is 9,999 at t = 1 (almost no noise), 33.5 at t = 50, 8.71 at t = 100, **0.657 at t = 300** (the number of the lesson), 0.0853 at t = 500 and about 0.00004 at t = 1000. The curve on the log axis keeps sloping down without ever levelling off: the SNR falls by a large factor over the whole run, from thousands to almost nothing. The SNR first drops to 1 or below at **t = 260**, where abar = 0.4976: before that step the picture is stronger than the noise, after it the noise is stronger. That is only a quarter of the way through the process. The assertion confirms that abar is above 0.5 at step 259 and at most 0.5 at step 260.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.5. Two more landmarks
#
# Your task: find two more landmarks: the first step at which the SNR falls below 100 (store it in `t_snr_100`) and the first step at which it falls below 0.01 (store it in `t_snr_001`). Use `snr(np.arange(1, 1001), abar)` and remember that step numbers start at 1, not 0. Why it matters: the range of SNR that a network must handle, from hundreds down to a hundredth, is huge; this is one reason the network must be told t.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
t_snr_100 = None    # the first step t at which the SNR falls below 100 (the picture is still 100 times stronger than the noise)
t_snr_001 = None    # the first step t at which the SNR falls below 0.01 (the noise is 100 times stronger)

# %%
if t_snr_100 is None or t_snr_001 is None:
    print("Fill in t_snr_100 and t_snr_001 above, then run this cell again.")
else:
    s_ = snr(np.arange(1, 1001), abar)
    check("t_snr_100 is right", t_snr_100 == int(np.argmax(s_ < 100)) + 1, "Compute snr for t = 1 .. 1000 and find the first value below 100 (remember that t starts at 1).")
    check("t_snr_001 is right", t_snr_001 == int(np.argmax(s_ < 0.01)) + 1, "Same, with the threshold 0.01.")
    RESULTS["p1_snr"] = (int(t_snr_100), int(t_snr_001))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# s_ = snr(np.arange(1, 1001), abar)
# t_snr_100 = int(np.argmax(s_ < 100)) + 1
# t_snr_001 = int(np.argmax(s_ < 0.01)) + 1
# print(t_snr_100, t_snr_001)
# ```
#
# The SNR falls below 100 at t = 28 already (almost immediately: a little noise is enough to make the picture only 100 times stronger) and below 0.01 at t = 675. So the SNR spans the range from 100 to 0.01 in about 650 steps, a factor of 10,000. The network has to denoise at all these levels with the same weights.
#
# </details>

# %% [markdown]
# ### Step 1.6. The end is pure noise
#
# The whole method depends on the end of the forward process being **the same for every picture**: pure N(0, 1) noise. If that is true we can start the reverse process from random numbers that we draw ourselves, without having any picture. The block checks it for the five different shapes (one of each kind): for each, it makes 50 noise draws at t = 1000 and prints the mean and the variance of all the pixel values (they should be 0 and 1). The assertion allows an error of 0.1.

# %%
rng = np.random.RandomState(SEED)
print("shape              mean of x_1000   variance of x_1000")
for k in range(5):
    pic = 2 * Xc[int(np.where(yc == k)[0][0])] - 1
    xs = np.stack([q_sample(pic, 1000, rng.randn(*pic.shape), abar) for _ in range(50)])      # 50 noise draws
    print(f"{SHAPE_NAMES[k]:16s}  {xs.mean():8.3f}        {xs.var():8.3f}")
    assert abs(xs.mean()) < 0.1 and abs(xs.var() - 1) < 0.1
print("weight of the picture at t = 1000:", round(float(np.sqrt(abar[1000])), 4), "  weight of the noise:", round(float(np.sqrt(1 - abar[1000])), 4))

# %% [markdown]
# You should see five lines and a last line with the two weights. For all five shapes the mean of x_1000 is within 0.022 of 0 (the values are -0.010, -0.000, -0.022, 0.008 and 0.003) and the variance is between 0.993 and 1.021: very close to 1. The reason is in the last line: at t = 1000 the weight of the picture is only sqrt(4.04e-5) = 0.0064, so even a picture with pixel values of -1 or +1 shifts the mean by at most 0.0064, and the noise has the weight 1.0000. **Whatever you put in, a standard bell curve comes out.** That is the starting point of every sampler in the rest of the lab.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.6. How much of the picture leaks?
#
# Your task: measure the "leak" of the picture into the final step. Store sqrt(abar_1000) in `leak_1000` and sqrt(abar_300) in `leak_300`. Then make 50 all-white pictures of shape 16 by 16 (every pixel +1), noise them to t = 300 with `q_sample` and store the mean of the result in `white_mean_300`. Why it matters: it shows that the leak at t = 300 is large (the picture is still there) and at t = 1000 negligible.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
leak_1000 = None   # the weight of the clean picture in x_1000
leak_300 = None    # the weight of the clean picture in x_300
white_mean_300 = None   # the mean of x_300 for a picture that is all white (every pixel +1), averaged over many noise draws

# %%
if None in (leak_1000, leak_300, white_mean_300):
    print("Fill in leak_1000, leak_300 and white_mean_300 above, then run this cell again.")
else:
    check("leak_1000 is right", abs(leak_1000 - np.sqrt(abar[1000])) < 1e-4, "The weight of the picture is sqrt(abar_t).")
    check("leak_300 is right", abs(leak_300 - np.sqrt(abar[300])) < 1e-3, "sqrt(abar_300).")
    check("white_mean_300 is right (0.03)", abs(white_mean_300 - leak_300) < 0.03, "For a constant picture of +1 the mean of x_t is sqrt(abar_t) * 1.")
    RESULTS["p1_leak"] = (round(float(leak_1000), 4), round(float(leak_300), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# leak_1000 = float(np.sqrt(abar[1000]))
# leak_300 = float(np.sqrt(abar[300]))
# rng = np.random.RandomState(SEED)
# white = np.ones((50, 16, 16))
# white_mean_300 = float(q_sample(white, 300, rng.randn(*white.shape), abar).mean())
# print(leak_1000, leak_300, white_mean_300)
# ```
#
# `leak_1000` is 0.0064 and `leak_300` is 0.630; the all-white picture, which has the mean +1, has a mean of 0.627 at t = 300, which is the same as sqrt(abar_300) (it is the weight of the picture, multiplied by 1). At t = 1000 the same white picture would have a mean of 0.006, which is nothing.
#
# </details>

# %% [markdown]
# ### Step 1.7. A recipe, not a learner
#
# In Week 12 the generator was a network with weights that were learned in a game against a detective. The forward process of this week is the opposite: **it has no weights at all**. It is a recipe, written in one line, and it needs no training. The block makes two noisy versions of the same clean cross at the same step t = 300, with two different noise draws, and measures how similar they are to the clean picture and to each other. It also prints the number of learned parameters in the forward process (zero).

# %%
rng = np.random.RandomState(SEED)
xa = q_sample(x0, 300, rng.randn(*x0.shape), abar)       # two different noise draws, same clean picture, same t
xb = q_sample(x0, 300, rng.randn(*x0.shape), abar)
c = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
print(f"correlation of x_300 (draw 1) with x_0 : {c(xa, x0):.3f}")
print(f"correlation of x_300 (draw 2) with x_0 : {c(xb, x0):.3f}")
print(f"correlation of the two draws           : {c(xa, xb):.3f}")
print("numbers learned by the forward process :", 0)
show_grid(np.array([x0, xa, xb]), ["clean", "draw 1", "draw 2"], cols=3)

# %% [markdown]
# You should see three correlations, one line with a zero and three small pictures. Each noisy version is moderately similar to the clean picture (0.357 and 0.404), but the two noisy versions are much *less* similar to each other (0.134), because they share the picture but not the noise. The forward process has **0** learned numbers. Two noisy pictures of the same cross look different, and each of them is a perfectly good training example, with a different noise that we know.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.7. Why is a fixed recipe useful?
#
# Your task: write three to five sentences between the quotation marks in `answer_1_7`: how does the forward process differ from the generator of Week 12, and why is a fixed recipe useful for training? Use the words *learned*, *noise* and *known* if they help. Why it matters: this is the main difference between GANs and diffusion models, and you will meet it again in the reflection questions of Part 7.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
answer_1_7 = ""   # write 3 to 5 sentences between the quotation marks

# %%
if len(answer_1_7.split()) < 25:
    print("Write at least 25 words in answer_1_7 above, then run this cell again.")
else:
    check("you wrote an answer of 25 words or more", True)
    RESULTS["p1_answer"] = answer_1_7[:200]

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_1_7 = ("In Week 12 the generator had weights that were learned in a game, and nobody could say in advance what it would draw. "
#               "The forward process has no weights: it is a fixed formula, so for every clean picture and every step t we can make a noisy picture "
#               "AND we know exactly which noise we added. That known noise is a free, perfect answer key, which lets us train a network with an ordinary "
#               "loss, without any opponent. The end of the recipe, pure N(0, 1) noise, is also a starting point that we can draw whenever we like.")
# ```
#
# The sample answer makes three points: (1) the forward process has nothing learned in it, (2) we know exactly which noise we added, so we have a free and perfect answer key, and (3) the end is pure noise that we can draw ourselves. Compare with your own answer. A strong answer mentions that the training of a diffusion model is an ordinary regression (predict a known noise with a mean squared error), without the unstable two-player game of the GAN.
#
# </details>

# %% [markdown]
# ## Part 2. Noise schedules (20 min)
#
# *Lesson section: "Forward".* The linear schedule of Part 1 is not the only way to destroy a picture, and it is not even the best one. How fast the SNR falls decides which steps of the process are easy for the network, which are hard, and which are wasted. In this part you build the **cosine schedule** that is widely used today, compare it with the linear one on curves, on pictures and in numbers, count the steps that a linear schedule spends on pictures that are already pure noise, choose the number of steps T for the little experiments of the following parts, and finally design a schedule of your own.

# %% [markdown]
# ### ✏️ Imagine first
#
# Suppose that you destroy the picture *more slowly*, so that fewer steps are spent on pure noise and every step removes about the same amount of information. Does that give the denoising network an easier or a harder job? Before you run any code, write two predictions. (1) The word "easier" or "harder". (2) How many of the 1,000 steps of the linear schedule have abar below 0.001 (the picture is already gone)? You compare with Steps 2.2 and 2.3.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: a slower schedule
my_guess_easier = None    # a schedule that destroys the picture more slowly: is the network's job "easier" or "harder"? (a string)
my_guess_wasted = None    # of the 1,000 steps of the linear schedule, how many have abar below 0.001 (the picture is already gone)? (a number)

# %% [markdown]
# ### Step 2.1. The cosine schedule
#
# The cosine schedule, proposed by Nichol and Dhariwal in 2021, does not choose beta_t directly: it chooses the *curve* abar_t, namely f(t) / f(0) with f(t) = cos^2(((t / T + s) / (1 + s)) * pi / 2), and computes beta_t = 1 - abar_t / abar_(t-1) from it (clipped at 0.999 so that the last steps stay well defined). The small offset s = 0.008 keeps beta from being almost 0 at the start. The block builds it in the same format as `make_schedule`, so that all functions of Part 1 work with it, and prints abar at t = 300 and 500 next to the linear values.

# %%
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

# %% [markdown]
# You should see three lines. The cosine schedule has abar_300 = 0.787 and abar_500 = 0.494 (the values of the lesson), while the linear one has abar_300 = 0.396 and abar_500 = 0.079. At t = 500 the linear schedule has already removed 92 percent of the picture's power; the cosine schedule has removed only about half. The second line shows how the betas look: beta_1 is 0.00004 (a tiny first step), beta_500 is 0.0031, and the last betas become very large (0.750 at t = 999, 0.999 at t = 1000, the clip): the cosine schedule saves its large steps for the end, when the picture is already almost gone. At the end abar_1000 is 2.4e-9, essentially zero. The assertion checks the two values to 0.005.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.1. Compare the SNR at t = 500
#
# Your task: compute the SNR at t = 500 for both schedules with the function `snr(t, abar)` from Part 1 (the cosine schedule's array is `abar_c`) and store them in `snr_lin_500` and `snr_cos_500`. Why it matters: the SNR is the fair way to compare schedules, and these two numbers show how differently they behave half-way.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
snr_lin_500 = None   # SNR at t = 500, linear schedule
snr_cos_500 = None   # SNR at t = 500, cosine schedule

# %%
if snr_lin_500 is None or snr_cos_500 is None:
    print("Fill in snr_lin_500 and snr_cos_500 above, then run this cell again.")
else:
    check("snr_lin_500 is right", abs(snr_lin_500 - snr(500, abar)) < 0.01, "snr(500, abar).")
    check("snr_cos_500 is right", abs(snr_cos_500 - snr(500, abar_c)) < 0.01, "snr(500, abar_c).")
    check("the cosine schedule keeps more signal at t = 500", snr_cos_500 > snr_lin_500, "Look at the numbers again.")
    RESULTS["p2_snr500"] = (round(float(snr_lin_500), 3), round(float(snr_cos_500), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# snr_lin_500 = float(snr(500, abar))
# snr_cos_500 = float(snr(500, abar_c))
# print(snr_lin_500, snr_cos_500)
# ```
#
# The linear schedule has an SNR of 0.0853 at t = 500 (the noise is 12 times stronger than the picture), the cosine schedule an SNR of 0.976 (picture and noise are almost equally strong). The difference is a factor of about 11.
#
# </details>

# %% [markdown]
# ### Step 2.2. Compare the two schedules
#
# Now we compare the schedules in three ways. First, the two curves abar_t and the two SNR curves (log axis). Second, the share of the steps for which the SNR is above 1 (the picture is stronger than the noise). Third, the same cross as in Part 1, with the same noise draw, at t = 200, 400, 600 and 800 with each schedule, in two rows, together with the correlation between x_t and x_0.

# %%
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

# %% [markdown]
# You should see two panels of curves, a line with two shares, a figure with two rows of four pictures and two lines of correlations. In the left panel the linear curve of abar_t falls faster and hits the floor early, the cosine curve stays high for longer, bends and falls steeply at the end. On the SNR panel the linear curve is a descending line that crosses the dashed line SNR = 1 early, at about step 260, and the cosine curve crosses it half-way, at about step 500. The shares say the same: **26 percent of the linear steps and 50 percent of the cosine steps have an SNR above 1.** In the pictures, the top row (linear) shows the cross at t = 200 with grain, at t = 400 barely, and then only static; the bottom row (cosine) still shows the cross at t = 200 and 400, and traces of it at 600. The correlations with x_0 are 0.56, 0.21, 0.05 and -0.01 for the linear schedule and 0.83, 0.55, 0.32 and 0.13 for the cosine schedule. The cosine schedule keeps the picture alive for much longer.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.2. When is 90 percent of the picture gone?
#
# Your task: find the first step at which abar falls below 0.1 (90 percent of the power gone) for the linear schedule (`t_lin_01`, from `abar`) and for the cosine schedule (`t_cos_01`, from `abar_c`). Why it matters: it tells you how many of the T steps carry real information about the picture.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
t_lin_01 = None   # the first step t at which abar falls below 0.1 for the linear schedule
t_cos_01 = None   # the same for the cosine schedule

# %%
if t_lin_01 is None or t_cos_01 is None:
    print("Fill in t_lin_01 and t_cos_01 above, then run this cell again.")
else:
    check("t_lin_01 is right", t_lin_01 == int(np.argmax(abar < 0.1)), "np.argmax(abar < 0.1) gives the first index where the test is True.")
    check("t_cos_01 is right", t_cos_01 == int(np.argmax(abar_c < 0.1)), "Same with abar_c.")
    RESULTS["p2_t01"] = (int(t_lin_01), int(t_cos_01))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# t_lin_01 = int(np.argmax(abar < 0.1))
# t_cos_01 = int(np.argmax(abar_c < 0.1))
# print(t_lin_01, t_cos_01)
# ```
#
# The linear schedule reaches abar below 0.1 at t = 476, the cosine schedule only at t = 794. With the linear schedule, the last half of the steps (from 476 to 1,000) deals with pictures that have lost more than 90 percent of their power; with the cosine schedule that is only the last fifth.
#
# </details>

# %% [markdown]
# ### Step 2.3. Wasted steps
#
# When we train, we pick a random step t for every training picture, uniformly between 1 and T. If many steps are at pure noise, then many training examples are pure noise, where the network can do nothing clever (the best prediction of the noise is the input itself) and therefore learns little. The block counts, for four thresholds, how many of the 1,000 steps have an abar below the threshold, for both schedules, and the share of training examples that are already pure noise at the strictest of the thresholds that is used in this lab (abar below 0.001).

# %%
print("threshold      linear: steps below   cosine: steps below   (out of 1,000)")
for thr in [0.1, 0.01, 0.001, 0.0001]:
    nl = int((abar[1:] < thr).sum()); nc = int((abar_c[1:] < thr).sum())
    print(f"abar < {thr:<8}   {nl:5d}  ({nl / 10:4.1f}%)          {nc:5d}  ({nc / 10:4.1f}%)")
wasted_lin = int((abar[1:] < 0.001).sum()); wasted_cos = int((abar_c[1:] < 0.001).sum())
print("\nIf t is drawn uniformly, the share of training examples that are already pure noise (abar < 0.001):")
print(f"  linear {wasted_lin / 1000:.1%}   cosine {wasted_cos / 1000:.1%}")

# %% [markdown]
# You should see a table with four rows and two lines. For the linear schedule 525 steps (52.5 percent) have abar below 0.1, 327 steps (32.7 percent) below 0.01, **175 steps (17.5 percent) below 0.001** and 46 below 0.0001. For the cosine schedule the numbers are 207 (20.7 percent), 65 (6.5 percent), 21 (2.1 percent) and 7. So if you draw t uniformly, **17.5 percent of the training examples of the linear schedule are already pure noise, against only 2.1 percent for the cosine schedule.** This is the answer to the second Imagine question: about 175 steps are wasted. Several of the other steps are almost wasted too: at the 0.01 threshold a third of the linear steps are lost.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.3. A stricter threshold
#
# Your task: count the steps with abar below 0.01 (instead of 0.001) for the linear schedule (`waste_lin_001`) and for the cosine schedule (`waste_cos_001`); count only t = 1 .. 1000 (`abar[1:]`). Why it matters: it shows that the conclusion does not depend on one particular threshold.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
waste_lin_001 = None   # how many of the 1,000 steps have abar below 0.01 (linear schedule)
waste_cos_001 = None   # the same for the cosine schedule

# %%
if waste_lin_001 is None or waste_cos_001 is None:
    print("Fill in waste_lin_001 and waste_cos_001 above, then run this cell again.")
else:
    check("waste_lin_001 is right", waste_lin_001 == int((abar[1:] < 0.01).sum()), "Count over t = 1 .. 1000 (abar[1:]).")
    check("waste_cos_001 is right", waste_cos_001 == int((abar_c[1:] < 0.01).sum()), "Same with abar_c.")
    RESULTS["p2_waste"] = (int(waste_lin_001), int(waste_cos_001))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# waste_lin_001 = int((abar[1:] < 0.01).sum())
# waste_cos_001 = int((abar_c[1:] < 0.01).sum())
# print(waste_lin_001, waste_cos_001)
# ```
#
# The counts are 327 and 65, in agreement with the table. In a real training run this means that, with the linear schedule, roughly every third example gives the network an almost-pure-noise problem, and with the cosine schedule only about one in fifteen. A good schedule spreads the *difficulty* evenly over the steps.
#
# </details>

# %% [markdown]
# ### Step 2.4. Choosing the number of steps T
#
# For the experiments of the following parts we do not want 1,000 network calls per picture: it would take too long on a CPU. We can use a shorter process, with fewer and larger steps, as long as two things stay true: the process must still end in (almost) pure noise, and no single step may add so much noise that it becomes hard to undo. The rule we use is to shorten T and to stretch the range of beta by the same factor: b0 = 0.0001 * 1000 / T and b1 = 0.02 * 1000 / T. The block prints the end value abar_T and the largest beta for several values of T, builds the schedule that we use in Part 6 (T = 100) and prints abar at four steps.

# %%
print("   T    b0        b1       abar_T      largest beta")
for T_ in [50, 100, 200, 500, 1000]:
    b_, a_, ab_ = make_schedule(T_, 1e-4 * 1000 / T_, 0.02 * 1000 / T_)
    print(f"{T_:5d}  {1e-4 * 1000 / T_:.4f}   {0.02 * 1000 / T_:.4f}   {ab_[T_]:.3e}   {b_.max():.3f}")

T_lab = 100
beta_l, alpha_l, abar_l = make_schedule(T_lab, 1e-4 * 1000 / T_lab, 0.02 * 1000 / T_lab)
print("\nThe schedule used for the pictures in Part 6: T = 100, beta from", beta_l[1], "to", beta_l[-1])
print("abar at t = 25, 50, 75, 100:", [round(float(abar_l[t_]), 4) for t_ in [25, 50, 75, 100] ])
assert abar_l[T_lab] < 0.01

# %% [markdown]
# You should see a table and two lines. With the scaled range every schedule ends at pure noise: abar_T is 7.7e-6 for T = 50, 2.0e-5 for T = 100, 3.0e-5 for T = 200, and 4.0e-5 for T = 1000, all far below 0.01 (the assertion for T = 100 checks it). What changes is the largest beta: 0.4 for T = 50, 0.2 for T = 100, 0.1 for T = 200 and 0.02 for T = 1000. A big beta means a big, hard step. We choose **T = 100** for the pictures of Part 6: it is ten times cheaper than T = 1000 and each step is still small (beta from 0.001 to 0.2). The abar values 0.5279, 0.0742, 0.0026 and 0.00002 at t = 25, 50, 75 and 100 show that the picture is half gone at about a quarter of the way and completely gone by three quarters. (The lab's 1-D experiments in Part 3 use T = 200 with a different range, from 0.001 to 0.06.)

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.4. Scale the schedule to T = 500
#
# Your task: scale the same rule to a longer process: T = 500 with b0 = 0.0001 * 1000 / 500 and b1 = 0.02 * 1000 / 500. Store abar_T in `abar_T_500` and the largest beta in `max_beta_500`. Why it matters: it confirms the rule that scaling the beta range with 1000 / T keeps the end at pure noise for any T.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
abar_T_500 = None    # abar_T for T = 500 with the scaled range (b0 = 1e-4 * 1000 / T, b1 = 0.02 * 1000 / T)
max_beta_500 = None  # the largest beta of that schedule

# %%
if abar_T_500 is None or max_beta_500 is None:
    print("Fill in abar_T_500 and max_beta_500 above, then run this cell again.")
else:
    b5, _, a5 = make_schedule(500, 1e-4 * 1000 / 500, 0.02 * 1000 / 500)
    check("abar_T_500 is right", abs(abar_T_500 - a5[500]) < 1e-7, "make_schedule(500, 0.0002, 0.04)[2][500].")
    check("max_beta_500 is right", abs(max_beta_500 - 0.04) < 1e-9, "The last beta is b1 = 0.02 * 1000 / 500.")
    check("the end is pure noise (below 0.01)", abar_T_500 < 0.01, "Check that you used the scaled range.")
    RESULTS["p2_T500"] = (float(abar_T_500), float(max_beta_500))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# b5, _, a5 = make_schedule(500, 1e-4 * 1000 / 500, 0.02 * 1000 / 500)
# abar_T_500 = float(a5[500])
# max_beta_500 = float(b5.max())
# print(abar_T_500, max_beta_500)
# ```
#
# For T = 500 the final abar is 3.77e-05 (pure noise) and the largest beta is 0.04: smaller steps than T = 100, at the price of five times as many network calls. Choosing T always trades the **cost of sampling** against the **difficulty of every step**.
#
# </details>

# %% [markdown]
# ### Step 2.5. A schedule is a choice
#
# There is nothing sacred about the linear or the cosine curve. Any smooth curve that starts at abar_0 = 1, decreases and ends near 0 defines a valid forward process, because the closed form of Part 1.3 works for every abar. To see how different the choices can be, the block adds a third schedule, the quadratic one: abar_t = (1 - t / T) squared. It plots the three SNR curves on one log axis and prints, for each, the step at which the SNR passes 1 and the values of abar at t = 300 and t = 900.

# %%
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

# %% [markdown]
# You should see a plot with three descending curves and three lines. The SNR crosses 1 at t = 260 (linear), t = 497 (cosine) and t = 293 (quadratic). At t = 300 the three schedules have abar = 0.396, 0.787 and 0.490; at t = 900 they have 0.00028, 0.024 and 0.0100. All three are valid forward processes, and they sit in a family of many others. They differ in how fast they remove the information, and in whether the end is really pure noise (the cosine schedule at t = 900 still has abar = 0.024, so its last 100 steps are used to finish the job). Researchers choose by trying them: the cosine schedule gave better pictures in the paper that introduced it, but this is an experimental finding, not a law.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.5. Design your own schedule
#
# Your task: design your own schedule. Write the function `my_abar(T)` so that it returns an array of T + 1 numbers (abar_0 = 1, strictly decreasing, abar_T below 0.01), and write two or three sentences in `answer_2_5` on why a schedule is a design choice and not a law. Why it matters: it shows that you understand which properties are required (a known recipe with a pure noise end) and which are free.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
def my_abar(T=1000):
    """Return an array of length T + 1: abar_0 = 1, strictly decreasing, abar_T below 0.01."""
    return None

answer_2_5 = ""   # two or three sentences: why is a schedule a design choice and not a law?

# %%
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

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def my_abar(T=1000):
#     tt = np.arange(T + 1) / T
#     return np.exp(-7.0 * tt ** 1.5)          # slow start, then faster; ends at exp(-7) = 0.0009
#
# answer_2_5 = ("The only things that the training needs are a known recipe and an end that is pure noise, so many curves work. "
#               "The linear schedule is simple but wastes many late steps on pictures that are already noise, the cosine schedule spreads the "
#               "difficulty more evenly, and my own curve is just another point in the same family. Which one is best depends on the data and "
#               "is found by trying, not by a theorem.")
# ```
#
# The example uses abar_t = exp(-7 * (t / T) to the power 1.5): a slow start, then a faster fall, ending at exp(-7) = 0.0009. Any curve that passes the four checks works. In your answer, check that you said what is *required* (start at 1, strictly decreasing, end near 0, so that the end is pure noise) and what is *free* (the shape between). The best shape is found by experiments on your data.
#
# </details>

# %% [markdown]
# ## Part 3. The reverse step and a perfect denoiser (25 min)
#
# *Lesson section: "Sampling".* Now we turn the film around. If someone gives us a picture at step t and tells us which noise it contains, we can take a small step back to step t-1. Repeat this step T times, starting from pure noise, and a picture comes out: the sculptor takes a block of fog and chips away until the statue appears. The network of the next parts will have the job of *telling* us the noise. In this part we do not need a network at all, because for a very simple data set (numbers that come from two bumps) the perfect noise predictor can be written down as a formula. You build the reverse step by hand, derive the formula of the perfect predictor, run the loop from noise to two clean bumps, and then break the loop on purpose (no fresh noise, a predictor that is wrong) to see what each ingredient does. Everything in this part is NumPy and runs in a second.

# %% [markdown]
# ### ✏️ Imagine first
#
# In this part the data are single numbers that come from two narrow bumps, at -2 and at 3, each with a spread of 0.4 and half of the numbers in each bump. We start from 600 random numbers from N(0, 1) (a single broad bell curve, no bumps at all) and apply 200 small denoising steps. Before you run any code, write two predictions. (1) Will the final 600 numbers form two separate bumps, "yes" or "no"? (2) What percentage of them will end in the left bump? You compare with Step 3.3.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: noise to bumps
my_guess_two_bumps = None   # will 600 random numbers from N(0, 1), pushed through 200 small "denoise" steps, become two separate bumps? ("yes" or "no")
my_guess_left = None        # if yes: what percent of them end in the left bump (0 to 100)?

# %% [markdown]
# ### Step 3.1. The reverse step by hand
#
# The reverse step turns x_t into x_(t-1) with the formula x_(t-1) = (x_t - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sigma_t * z, where eps_hat is the noise estimate (from the network, later), z is a fresh draw from N(0, 1) and sigma_t = sqrt(beta_t). Read it from the inside out. (1) beta_t / sqrt(1 - abar_t) * eps_hat is the part of the noise that this step is allowed to remove. (2) Dividing by sqrt(alpha_t) undoes the shrinking of the forward step. (3) The fresh noise sigma_t * z is added back because the true previous picture is not known exactly: the process must keep some randomness. At the very last step (t = 1) we use z = 0, since there is nothing left to explore. The block defines this schedule for the whole part: T = 200 with beta from 0.001 to 0.06 (ᾱ_T = 0.00198), then writes the function `reverse_step`, and works the one-pixel example of the lesson in detail: t = 300 on the schedule of Part 1, x_t = 1.4, with a perfect noise estimate for a clean pixel x_0 = 2.

# %%
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

# %% [markdown]
# You should see the schedule check and the example. For the schedule of Part 3, abar_200 is 0.00198: pure noise at the end. In the worked example, beta_300 = 0.00606, abar_300 = 0.3964 and the perfect noise estimate is eps_hat = 0.1812 (the noise that was mixed into x_0 = 2 to make 1.4). The step removes 0.00606 / 0.777 * 0.1812 = 0.00141, so 1.4 becomes 1.39859; dividing by sqrt(alpha_300) = 0.99697 gives **1.40284**, the mean of the previous step; adding the fresh noise with sigma = 0.07782 and z = 1 gives 1.48066. The pixel moved by only 0.003 in the deterministic part, and by up to 0.08 for the random part: **the random part is larger than the denoising part at each single step**, and the denoising only wins in the long run, over many steps. The last line is the last step (t = 1, z = 0): 0.1 becomes 0.0995. The assertion checks the formula.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.1. Two steps on the schedule of Part 3
#
# Your task: compute two reverse steps on the schedule of Part 3 (`beta3`, `abar3`), both with z = 0: at t = 200 with x_t = 1.0 and eps_hat = 0.9, stored in `x_prev`; and at t = 1 with x_t = 0.1 and eps_hat = 0.05, stored in `x_prev_last`. Use `reverse_step(x_t, t, eps_hat, z, beta3, abar3)`. Why it matters: it shows how different the step is at the two ends of the process.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
x_prev = None        # t = 200, x_t = 1.0, eps_hat = 0.9, z = 0.0, on the schedule beta3, abar3
x_prev_last = None   # t = 1, x_t = 0.1, eps_hat = 0.05, z = 0.0, on the same schedule

# %%
if x_prev is None or x_prev_last is None:
    print("Fill in x_prev and x_prev_last above, then run this cell again.")
else:
    e1 = (1.0 - beta3[200] / np.sqrt(1 - abar3[200]) * 0.9) / np.sqrt(1 - beta3[200])
    e2 = (0.1 - beta3[1] / np.sqrt(1 - abar3[1]) * 0.05) / np.sqrt(1 - beta3[1])
    check("x_prev is right", abs(x_prev - e1) < 1e-6, "reverse_step(1.0, 200, 0.9, 0.0, beta3, abar3).")
    check("x_prev_last is right", abs(x_prev_last - e2) < 1e-6, "reverse_step(0.1, 1, 0.05, 0.0, beta3, abar3).")
    RESULTS["p3_steps"] = (round(float(x_prev), 4), round(float(x_prev_last), 4))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# x_prev = float(reverse_step(1.0, 200, 0.9, 0.0, beta3, abar3))
# x_prev_last = float(reverse_step(0.1, 1, 0.05, 0.0, beta3, abar3))
# print(x_prev, x_prev_last)
# ```
#
# The first step gives 0.9757 and the last step 0.0985. At t = 200 beta is large (0.06) and the step changes the number by about 0.024 for these inputs; at t = 1 beta is tiny (0.001) and the step changes it by 0.0015. In the early reverse steps (large t) the network makes big moves, in the last ones (small t) it only polishes.
#
# </details>

# %% [markdown]
# ### Step 3.2. An exact noise predictor
#
# To run the loop we need a noise predictor. For data that come from a mixture of bumps there is a **perfect** one, and we can write it down. Here is the derivation in plain words. (1) If the clean number x_0 comes from a bump with mean mu and spread sd, then x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps is also a bell curve, with mean sqrt(abar_t) * mu and variance v = abar_t * sd squared + 1 - abar_t. (2) So the whole density of x_t is again a mixture of two bumps, each wider than before. (3) A fact from statistics (Tweedie's formula) says that the best guess of the noise is minus sqrt(1 - abar_t) times the slope of the logarithm of that density. (4) For a mixture, the slope is an average of the slopes of the bumps, weighted by the probability r_i that x came from bump i. Together: eps_exact(x, t) = sqrt(1 - abar_t) * sum over i of r_i * (x - sqrt(abar_t) * mu_i) / v. The code below writes this formula down, and also writes the density itself (`log_marginal`) so that we can check the formula against a **numerical derivative** of the logarithm of the density at one point. Then it prints the predictor for a few values of x at three steps.

# %%
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

# %% [markdown]
# You should see a check line and a small table. At t = 100 and x = 0.3 the formula gives -0.043103 and the numerical derivative gives -0.043103: they agree to six decimals (the assertion allows 0.00001), so the formula is right. In the table, read each row as "if you see this value of x_t at step t, the noise in it is probably this much". At t = 190 the predicted noise is nearly x itself (-2.969 for x = -3, 2.910 for x = 3): **at large t almost everything you see is noise**. At t = 10 the answers are very different: x = -2 gives -0.020 (a value at the centre of a bump is almost noise-free, since there is nearly no noise at this step), but x = 0 gives 1.675 and x = 1 gives -1.665: the predictor flips its sign between the two bumps (a point at x = 0 belongs to the left bump, so its noise is positive; at x = 1 to the right bump, so its noise is negative). A network will have to learn exactly this kind of sharp structure.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.2. What does the exact predictor say at the end?
#
# Your task: evaluate the exact predictor at the point x = 3.0 for two steps: t = 10 (store it in `eps_x3_t10`) and t = 200 (store it in `eps_x3_t200`), with `eps_exact(x, t, abar3)`. Why it matters: x = 3 is the centre of the right bump; what the predictor says there at the two ends of the process shows how the job of the denoiser changes with t.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
eps_x3_t10 = None    # eps_exact at x = 3.0, t = 10 (schedule abar3)
eps_x3_t200 = None   # eps_exact at x = 3.0, t = 200

# %%
if eps_x3_t10 is None or eps_x3_t200 is None:
    print("Fill in eps_x3_t10 and eps_x3_t200 above, then run this cell again.")
else:
    check("eps_x3_t10 is right", abs(eps_x3_t10 - float(eps_exact(3.0, 10, abar3))) < 1e-6, "eps_exact(3.0, 10, abar3).")
    check("eps_x3_t200 is right", abs(eps_x3_t200 - float(eps_exact(3.0, 200, abar3))) < 1e-6, "eps_exact(3.0, 200, abar3).")
    check("at t = 200 the predictor is almost the identity (within 0.1 of x)", abs(eps_x3_t200 - 3.0) < 0.1, "At the end almost everything in x_t is noise, so eps is almost x_t.")
    RESULTS["p3_eps"] = (round(eps_x3_t10, 3), round(eps_x3_t200, 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# eps_x3_t10 = float(eps_exact(3.0, 10, abar3))
# eps_x3_t200 = float(eps_exact(3.0, 200, abar3))
# print(eps_x3_t10, eps_x3_t200)
# ```
#
# At t = 10 the answer is 0.030: x = 3 is the centre of a bump, and at this step there is almost no noise to remove. At t = 200 it is 2.944, close to x itself: x_200 is almost pure noise, so the best guess of the noise is the input itself, with a slight correction that depends on the position of the bumps.
#
# </details>

# %% [markdown]
# ### Step 3.3. The reverse loop
#
# Here is the sampler. The function `run_reverse` starts with n numbers from N(0, 1), applies `reverse_step` for t = T, T-1, ..., 1 with a predictor that you give it, and returns the final numbers and some snapshots. Run it with 600 numbers and the exact predictor. Draw the histogram at t = 200, 150, 100, 50 and 0 against the true density (the black curve), and measure three numbers: the **mean** of the final numbers (the data mean is 0.5), the **share** in the left bump (the data have 50 percent), and the average **spread inside a bump** (the data have 0.4). The lesson's figure `sampler1d` shows the same experiment.

# %%
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

# %% [markdown]
# You should see five histograms in a row and two lines. At t = 200 the histogram is one bell curve (the starting noise). At t = 150 it has widened slightly; at t = 100 it is broad and flat, with a first hint of two humps; at t = 50 two clear humps with a gap between them; at t = 0 the histogram follows the black curve of the data with two narrow bumps. The numbers: the final mean is 0.45 (data: 0.50), the share in the left bump is **51.8 percent** (data: 50), the spread inside a bump is **0.39** (data: 0.40). The assertion allows 0.3 for the mean, 40 to 60 percent for the share and 0.3 to 0.55 for the spread. The second line is a surprise: the standard deviation of the whole cloud is 1.00 at t = 200, then grows to 1.08, 1.43, 2.11 and 2.48. The cloud has to *grow* from a spread of 1 to the spread of the two bumps (which is about 2.5, because the bumps are 5 apart). **Pure noise became two bumps, with nothing but a small repeated step and the perfect predictor**: the answer to the Imagine questions is "yes", with close to half in each bump. With only 600 samples, the share fluctuates by a few percent.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.3. A lopsided mixture
#
# Your task: change the data. Let the mixture have weights 0.8 for the left bump and 0.2 for the right bump: use `weights=(0.8, 0.2)` in `eps_exact`, run `run_reverse` with 2,000 numbers (seed=1), and store the share of the final numbers in the left bump in `share_left_80` and their mean in `mean_80`. Use the function `bump_stats` of the last block. Why it matters: the sampler has no knowledge of the bumps except through the noise predictor, so changing the predictor changes the result, as it should.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
share_left_80 = None   # share of samples in the left bump when the data are 80% left bump, 20% right bump
mean_80 = None         # the mean of the final samples

# %%
if share_left_80 is None or mean_80 is None:
    print("Fill in share_left_80 and mean_80 above, then run this cell again.")
else:
    check("share_left_80 is close to 0.8 (0.06)", abs(share_left_80 - 0.8) < 0.06, "Pass weights=(0.8, 0.2) to eps_exact, run 2,000 particles.")
    check("mean_80 is close to the true mean -1.0 (0.15)", abs(mean_80 - (0.8 * -2 + 0.2 * 3)) < 0.15, "0.8 * (-2) + 0.2 * 3 = -1.0.")
    RESULTS["p3_lopsided"] = (round(float(share_left_80), 3), round(float(mean_80), 2))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# eps80 = lambda x, t: eps_exact(x, t, abar3, weights=(0.8, 0.2))
# x80, _ = run_reverse(2000, eps80, beta3, abar3, 200, seed=1)
# mean_80, share_left_80, _ = bump_stats(x80)
# print(share_left_80, mean_80)
# ```
#
# The share in the left bump is 0.794 (about 80 percent) and the mean is -0.97 (the true mean is 0.8 * (-2) + 0.2 * 3 = -1.0). The sampler followed the new predictor without any change in the loop: **all the knowledge about the data sits in the noise predictor.**
#
# </details>

# %% [markdown]
# ### Step 3.4. Fresh noise or not?
#
# The reverse step adds fresh noise sigma_t * z at every step. Is it necessary? Run the loop with the fresh noise multiplied by `z_scale` = 1.0 (the real sampler), 0.5 and 0.0 (no fresh noise), with the same starting numbers, and print the mean, the share in the left bump and the spread inside a bump. Then draw the final histograms for 1.0 and 0.0.

# %%
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

# %% [markdown]
# You should see a table with three rows and two histograms. With full fresh noise the spread inside a bump is 0.390, with half of it 0.203, and **without any fresh noise 0.072**: the numbers all end up in a tiny spike at the centre of each bump (the right histogram shows two thin needles instead of two bumps). The mean (0.45, 0.41, 0.34) and the share in the left bump (51.8, 52.2, 53.3 percent) change much less: the two bumps are in the right places, but without fresh noise each one is **collapsed to nearly one value**. What the fresh noise does: the predictor, the exact one included, always moves a particle towards the most likely clean value. Without randomness, every particle that starts close to another one is pulled to the same point, so you lose the *variety* of the data: this is similar in spirit to the loss of variety you saw in Week 12, but here the cause is clear and the remedy is built in. The fresh noise at every step keeps the cloud as wide as the data. (In Part 6 you will meet DDIM, a deterministic sampler that is not collapsed: it uses a different, matching update, not the DDPM step without noise.)

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.4. Half the fresh noise
#
# Your task: store the spread inside a bump (the third number of `bump_stats`) for `z_scale=0.5` in `spread_half` and for `z_scale=1.0` in `spread_full`, using 600 particles, seed 0 and the same schedule and predictor. Why it matters: it confirms that the spread grows with the fresh noise, between the needle and the real width.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
spread_half = None     # the spread inside a bump (bump_stats, third number) with z_scale = 0.5, 600 particles, seed 0
spread_full = None     # the same with z_scale = 1.0

# %%
if spread_half is None or spread_full is None:
    print("Fill in spread_half and spread_full above, then run this cell again.")
else:
    spread_zero = bump_stats(run_reverse(600, exact3, beta3, abar3, 200, z_scale=0.0, seed=0)[0])[2]
    check("spread_full is close to the data spread 0.4 (0.08)", abs(spread_full - 0.4) < 0.08, "z_scale=1.0 is the real sampler.")
    check("the spread grows with the amount of fresh noise", spread_zero < spread_half < spread_full + 0.05, "Less fresh noise, tighter bumps.")
    RESULTS["p3_spread"] = (round(float(spread_half), 3), round(float(spread_full), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# spread_half = bump_stats(run_reverse(600, exact3, beta3, abar3, 200, z_scale=0.5, seed=0)[0])[2]
# spread_full = bump_stats(run_reverse(600, exact3, beta3, abar3, 200, z_scale=1.0, seed=0)[0])[2]
# print(spread_half, spread_full)
# ```
#
# `spread_half` is 0.203 and `spread_full` is 0.390, against 0.072 for no noise: half the noise gives about half of the spread. So the fresh noise is not decoration: it sets the width of the bumps.
#
# </details>

# %% [markdown]
# ### Step 3.5. Wrong denoisers
#
# What happens if the noise predictor is wrong? We test four predictors with the same loop: the exact one, a predictor that always says "there is no noise" (eps_hat = 0), a predictor that says "everything I see is noise" (eps_hat = x_t) and the exact one at half strength. For each, the block prints the mean and the standard deviation of the final numbers, their smallest and largest value, and the share that end within 1.2 (three standard deviations) of a bump centre.

# %%
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

# %% [markdown]
# You should see a table with four rows. The exact predictor gives a mean of 0.45, a standard deviation of 2.48, values between -3.06 and 4.05, and 99.7 percent of the numbers within 1.2 of a bump centre. **Predicting no noise** (eps_hat = 0) removes nothing: each step only divides by sqrt(alpha_t), a number slightly below 1, and adds fresh noise, so the numbers grow at every step; after 200 steps they have a standard deviation of 31.9 and range from -101 to +101, and only 6.2 percent are near a bump. **Predicting that everything is noise** (eps_hat = x_t) takes away too much: all numbers are pulled to 0 (standard deviation 0.41, between -1.26 and 1.37), and only 2.3 percent are within 1.2 of a bump centre, so they land in the gap between the bumps. The **half-strength** predictor lands in between: a standard deviation of 3.13 and a range from -5.6 to 6.0, with 74.5 percent near a bump. The lesson: the loop is only as good as the predictor, and you can tell a bad predictor from the final samples.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.5. A denoiser that is too strong
#
# Your task: try a predictor that is too strong: 1.5 times the exact one (`lambda x, t: 1.5 * exact3(x, t)`), with 600 numbers and seed 0. Store the share of the final numbers within 1.2 of a bump centre in `near_strong` and the standard deviation of the final numbers in `std_strong`. Why it matters: a bad network can be wrong in either direction; this is the opposite of the half-strength case.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
near_strong = None   # the share of final samples within 1.2 of a bump centre, when the predictor is 1.5 * eps_exact (600 particles, seed 0)
std_strong = None    # the standard deviation of those final samples

# %%
if near_strong is None or std_strong is None:
    print("Fill in near_strong and std_strong above, then run this cell again.")
else:
    xs_ = run_reverse(600, lambda x, t: 1.5 * exact3(x, t), beta3, abar3, 200, seed=0)[0]
    chk = np.minimum(np.abs(xs_ + 2), np.abs(xs_ - 3)) < 1.2
    check("near_strong is right", abs(near_strong - chk.mean()) < 0.01, "Count the samples with min(|x + 2|, |x - 3|) below 1.2.")
    check("std_strong is right", abs(std_strong - xs_.std()) < 0.01, "xs_.std() of the final samples.")
    RESULTS["p3_strong"] = (round(float(near_strong), 3), round(float(std_strong), 2))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# xs_ = run_reverse(600, lambda x, t: 1.5 * exact3(x, t), beta3, abar3, 200, seed=0)[0]
# near_strong = float((np.minimum(np.abs(xs_ + 2), np.abs(xs_ - 3)) < 1.2).mean())
# std_strong = float(xs_.std())
# print(near_strong, std_strong)
# ```
#
# With a predictor that is 1.5 times too strong, `near_strong` is 1.000 and `std_strong` is 2.47: every number lands at a bump, so this looks perfect in the share, but look at the spread inside a bump, which is only 0.27 against 0.39 for the exact predictor. An over-confident predictor gives samples that look right but are **too sharp and too similar**: they have lost the variety of the data. A single number such as "share near a bump" cannot tell you this; that is why we measure diversity and quality separately in Part 6.
#
# </details>

# %% [markdown]
# ### Step 3.6. What must the network learn?
#
# The neural network of the next part has to *learn* the function that you just wrote as a formula. It will only see examples (a noisy number, the step t and the true noise) and not the formula. To get a feeling for how hard that is, the block draws the exact predictor eps_exact(x, t) as a function of x for three steps (t = 10, 100, 190), together with the line eps = x_t, and prints for four steps the largest gap to this line and the steepest slope.

# %%
xg = np.linspace(-4, 4, 161)
fig, ax = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for a, tt in zip(ax, [10, 100, 190]):
    a.plot(xg, eps_exact(xg, tt, abar3), label="exact eps"); a.plot(xg, xg, "k:", label="x_t"); a.set_title(f"t = {tt}"); a.set_xlabel("x_t")
ax[0].legend(); plt.tight_layout(); plt.show()
for tt in [10, 100, 190, 200]:
    e = eps_exact(xg, tt, abar3)
    print(f"t = {tt:3d}: largest gap to the line eps = x_t is {np.abs(e - xg).max():.3f}   steepest slope {np.abs(np.diff(e) / np.diff(xg)).max():.2f}")

# %% [markdown]
# You should see three panels and four lines. At t = 190 the blue curve nearly lies on the dotted line: the largest gap to eps = x_t is only 0.106 and the steepest slope is 0.99, so the function to learn is almost the identity. At t = 100 the curve already leaves the line (largest gap 1.293) but is still smooth (steepest slope 1.08). At t = 10 the curve is **steep and jumpy**: the largest gap is 3.123 and the steepest slope is 25.2. It rises and falls sharply near the border between the bumps. So the job gets harder as t gets smaller: for large t the network only has to copy its input; for small t it has to know where the bumps are, down to a fraction of 0.4. At t = 200, the last step of the forward process, the gap is just 0.066.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.6. In your own words
#
# Your task: compute the largest gap between eps_exact(x, 200) and the line eps = x over the grid `xg` and store it in `gap_at_T`. Then write three to five sentences in `answer_3_6`: what must a neural network learn in the next part, and which steps will be easy and which hard? Why it matters: you now know exactly what a perfect network would compute, and that knowledge is the target of the training in Part 4.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
gap_at_T = None    # the largest gap |eps_exact(x, 200) - x| over xg
answer_3_6 = ""    # 3 to 5 sentences: what does a neural network have to learn in the next part, and which values of t will be easy or hard?

# %%
if gap_at_T is None or len(answer_3_6.split()) < 25:
    print("Fill in gap_at_T and write at least 25 words in answer_3_6 above, then run this cell again.")
else:
    check("gap_at_T is right", abs(gap_at_T - np.abs(eps_exact(xg, 200, abar3) - xg).max()) < 1e-3, "np.abs(eps_exact(xg, 200, abar3) - xg).max().")
    check("answer is long enough", True)
    RESULTS["p3_gap"] = round(float(gap_at_T), 3)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# gap_at_T = float(np.abs(eps_exact(xg, 200, abar3) - xg).max())
# answer_3_6 = ("The network has to learn the function eps(x_t, t) that I computed here by hand: given a noisy number and the step, it must say which part of it is noise. "
#               "For large t this is easy, the answer is nearly x_t itself. For small t it is a sharp, wiggly curve that depends on where the bumps are, "
#               "so the network must learn the shape of the data. It sees only examples, not the formula, and it needs t as an input because the curve changes with t.")
# print(gap_at_T)
# ```
#
# The gap at t = 200 is 0.066: the predictor at the end of the process is almost the identity. In your answer, check that you mention (1) the network must predict the noise from the noisy input and the step, (2) large t is easy, small t is hard because the curve is steep and depends on the data, and (3) the network needs to know t. Part 4 trains a small network on exactly this kind of function, now in two dimensions.
#
# </details>

# %% [markdown]
# ## Part 4. Train a denoiser on 2-D data (25 min)
#
# *Lesson section: "Training".* In Parts 1 to 3 you built the forward process and the reverse step, and in Part 3 you gave the reverse step a perfect, hand-made noise predictor. Now a network has to learn that predictor from examples. ==You will build the training pairs (a noisy point, its step number, and the noise that was added), write a sinusoidal time embedding and a small network that takes the noisy point plus the embedding, train it for 4,000 steps on eight blobs arranged in a ring, sample with the reverse loop of Part 3, watch the paths the points take, and see what fewer training steps do.== The data are the same eight blobs as in Week 12, so you can compare with the GAN. This part defines its own noise schedule and samplers (the code of Parts 1 to 3 is not needed). All numbers are for `SEED = 42` (seed 1 is the second seed); training is not exactly the same on every machine, so each number comes with a tolerance.

# %% [markdown]
# ### ✏️ Imagine first
#
# A GAN has trouble with eight blobs: in Week 12 it needed careful tuning, and it often kept only a few of them (mode collapse). A diffusion model never plays a game against a second network. It only learns to guess the noise that was added to a point, over and over, for random noise levels. At the end you start from pure noise and apply the guess 100 times. Before you run anything, guess how many of the 8 blobs a trained diffusion model reaches (0 to 8), and what share of its points lands on a blob. Write your guess in the cell below. You compare it with Step 4.4.

# %%
# Imagine first: how many of the 8 blobs will the diffusion model reach, and what share of its points will land on a blob?
my_guess_blobs = None      # a whole number from 0 to 8
my_guess_on_blob = None    # a share between 0 and 1, for example 0.5

# %% [markdown]
# ### Step 4.1. The training pairs
#
# A training example for a diffusion model is made in three moves: take a real point x0, pick a random step number t between 1 and T, and add the amount of noise that belongs to step t. The network gets the noisy point x_t and the number t, and must return the noise ε that was added. Because we know ε (we drew it), we always have the right answer. The noise schedule is the one of Part 2, scaled for a short run: T = 100 steps with β rising linearly from 0.001 to 0.2, which leaves almost nothing of the picture at the end (ᾱ_100 = 2.0e-5). We store the schedule in arrays with T + 1 entries so that the step number t is the index itself (entry 0 means "no noise yet", ᾱ_0 = 1). The helper `q_sample_t` is the closed form of Part 1, written so that every example in a batch has its own t.

# %%
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

# %% [markdown]
# ### ✏️ Your turn: The training pairs
#
# *Exercise 4.1:* Write `make_batch_mine(x0, S)`. It receives a batch of clean points `x0` (shape (B, 2)) and must return three tensors: `xt` (the noisy points), `t` (one random integer from 1 to T for every example, a LongTensor of shape (B,)) and `eps` (the noise, standard normal, same shape as `x0`). Why it matters: this function is the whole "data set" of a diffusion model: the pairs (x_t and t in, ε out) are made fresh at every training step, so the network never sees the same noisy point twice. Use `torch.randint(1, T + 1, (B,))` (the upper end is excluded), `torch.randn_like(x0)` and the helper `q_sample_t`.

# %%
def make_batch_mine(x0, S):
    """Return (xt, t, eps) for a batch of clean points x0."""
    return None   # YOUR CODE: t = random integers 1..T, eps = standard normal noise, xt = q_sample_t(x0, t, eps, S)

# %%
out_ = make_batch_mine(data_mix[:256], S)
if out_ is None:
    print("Fill in make_batch_mine above, then run this cell again.")
else:
    xt_, t_, eps_ = out_
    check("xt and eps have the shape of x0, t has shape (256,)", tuple(xt_.shape) == (256, 2) and tuple(eps_.shape) == (256, 2) and tuple(t_.shape) == (256,), "t needs one number per example")
    check("t holds integers between 1 and 100", t_.dtype == torch.long and int(t_.min()) >= 1 and int(t_.max()) <= 100, "torch.randint(1, T + 1, (B,)) excludes the upper end")
    check("xt follows the closed form for these t and eps", torch.allclose(xt_, q_sample_t(data_mix[:256], t_, eps_, S), atol=1e-6), "xt = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps")
    check("the steps are spread out (both small and large t occur)", int(t_.min()) < 15 and int(t_.max()) > 85, "t must be random for every example, not one number for the batch")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def make_batch(x0, S):
#     t = torch.randint(1, S["T"] + 1, (len(x0),))     # one random step number per example, 1..T
#     eps = torch.randn_like(x0)                       # the noise we will ask the network to guess
#     xt = q_sample_t(x0, t, eps, S)                   # the noisy point, from the closed form
#     return xt, t, eps
# ```
#
# The three lines are the three moves of the step. Drawing a new t for every example (and not one t per batch) makes every gradient step see easy and hard noise levels together, which is steadier than training level by level.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def make_batch(x0, S):
    t = torch.randint(1, S["T"] + 1, (len(x0),))     # one random step number per example, 1..T
    eps = torch.randn_like(x0)                       # the noise we will ask the network to guess
    xt = q_sample_t(x0, t, eps, S)                   # the noisy point, from the closed form
    return xt, t, eps

# %%
xb, tb, eb = make_batch(data_mix[:256], S)
print("one batch: xt", tuple(xb.shape), " t", tuple(tb.shape), " eps", tuple(eb.shape))
print("first three examples:")
for i in range(3):
    print("  x0 = (%5.2f, %5.2f)  t = %3d  eps = (%5.2f, %5.2f)  ->  xt = (%5.2f, %5.2f)" % (*data_mix[i], int(tb[i]), *eb[i], *xb[i]))
assert xb.shape == (256, 2) and int(tb.min()) >= 1 and int(tb.max()) <= S["T"]

# %% [markdown]
# Now a statistical check of the closed form: noise the same point (3, 0) 20,000 times with the same t. The mean must be √ᾱ_t·x0 and the standard deviation √(1 − ᾱ_t) in each direction.

# %%
x_fixed = torch.tensor([[3.0, 0.0]]).repeat(20000, 1)
print(" t   abar_t   expected mean (x)  measured    expected std  measured")
for tt in (10, 30, 60, 100):
    tv = torch.full((20000,), tt, dtype=torch.long)
    xs = q_sample_t(x_fixed, tv, torch.randn_like(x_fixed), S)
    ab = float(S["abar"][tt])
    print("%3d  %.5f  %14.3f  %10.3f  %12.3f  %8.3f" % (tt, ab, math.sqrt(ab) * 3.0, xs[:, 0].mean(), math.sqrt(1 - ab), xs[:, 0].std()))
    assert abs(float(xs[:, 0].mean()) - math.sqrt(ab) * 3.0) < 0.03 and abs(float(xs[:, 0].std()) - math.sqrt(1 - ab)) < 0.03

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes four ticks for your `make_batch_mine`, and the answer prints the shapes of one batch, for example `xt (256, 2)`, `t (256,)` and `eps (256, 2)`, and three examples (the numbers differ with the random generator: t is any integer from 1 to 100). The statistical table is the important part (20,000 copies of the point (3, 0)): at t = 10 (ᾱ = 0.904) the mean of the x-coordinate is 2.851 against the expected √ᾱ·3 = 2.852 and the standard deviation is 0.313 against √(1 − ᾱ) = 0.310; at t = 30 (ᾱ = 0.397) the mean is 1.886 against 1.891 and the spread 0.785 against 0.776; at t = 60 the mean is 0.455 against 0.455; at t = 100 (ᾱ = 2.0e-5) the mean is 0.018 against 0.014 and the standard deviation 1.007 against 1.000. The tolerance is 0.03 (a sample of 20,000 points has a sampling error of about 0.007). The schedule gives ᾱ_1 = 0.9990, ᾱ_30 = 0.3973 and ᾱ_100 = 2.04e-5.
#
# *What it means.* The closed form does what Part 1 promised: any noisy version of a point can be produced in one line, with no loop over the steps. At t = 100 the point (3, 0) has completely disappeared: the mean is 0.018 and the spread is 1, which is exactly standard normal noise. The network will therefore see the whole range, from a lightly blurred point (t = 10) to pure noise (t = 100).
#
# Tips and common mistakes:
#
# - `torch.randint(1, T + 1, ...)` excludes the upper end: without the `+ 1` the step T is never trained, and sampling starts exactly there.
# - `eps` must be drawn with `randn_like(x0)` (shape of x0), not one number for the whole batch.
#
# Try also (optional):
#
# - Print `S["abar"][[1, 25, 50, 75, 100]]` and find the step where half of the picture is gone (we get ᾱ_27 = 0.474, the first value below 0.5; Step 6.1 asks for it).

# %% [markdown]
# ### Step 4.2. The denoiser: a time embedding and a small network
#
# The network needs to know how much noise it is looking at: the same noisy point means something different at t = 5 (hardly any noise) and at t = 90 (almost only noise). We give it the step number through a **sinusoidal time embedding**, the formula of Week 8 applied to t: a vector of `dim` numbers, the first half sines and the second half cosines, at frequencies from 1 down to 1/10,000 (the exponent is −ln(10000)·k/half for k = 0, 1, …, half − 1). Two nearby steps get similar vectors and distant steps get different ones, and the network can read the vector with ordinary layers. The embedding (size 16) is glued to the two coordinates of the point, so the network sees 18 numbers, passes them through three hidden layers of 128 units with SiLU (a smooth ReLU, x·sigmoid(x), used in almost every diffusion model) and returns 2 numbers: its guess ε̂ of the noise.

# %% [markdown]
# ### ✏️ Your turn: The time embedding
#
# *Exercise 4.2:* Write `time_embed_mine(t, dim)`. It receives a LongTensor `t` of shape (B,) and returns a float tensor of shape (B, dim): first `half = dim // 2` sines, then `half` cosines, of the angles t·f_k with f_k = exp(−ln(10000)·k/half) for k = 0 … half − 1. Why it matters: this is how the network "hears" the noise level; in Part 5 you will take it away and see what breaks. The check uses two values you can verify by hand: at t = 0 all sines are 0 and all cosines are 1, and at t = 1 the first entry is sin(1) = 0.8415.

# %%
def time_embed_mine(t, dim=16):
    """Sinusoidal embedding of the step numbers t (shape (B,)) -> (B, dim)."""
    return None   # YOUR CODE: freqs = exp(-ln(10000) * arange(half) / half); angles = t[:, None] * freqs[None]; cat([sin(angles), cos(angles)], 1)

# %%
e0_ = time_embed_mine(torch.tensor([0, 1, 50]), 16)
if e0_ is None:
    print("Fill in time_embed_mine above, then run this cell again.")
else:
    check("the shape is (3, 16)", tuple(e0_.shape) == (3, 16), "dim numbers per step: dim // 2 sines then dim // 2 cosines")
    check("at t = 0: eight zeros (sines) then eight ones (cosines)", torch.allclose(e0_[0], torch.cat([torch.zeros(8), torch.ones(8)])), "sin(0) = 0 and cos(0) = 1")
    check("at t = 1 the first entry is sin(1) = 0.8415", abs(float(e0_[1, 0]) - math.sin(1.0)) < 1e-5, "the first frequency is exp(0) = 1")
    check("at t = 50 the second sine is sin(50 * exp(-ln(10000) / 8))", abs(float(e0_[2, 1]) - math.sin(50 * math.exp(-math.log(10000) / 8))) < 1e-4, "f_1 = exp(-ln(10000) * 1 / half) with half = 8")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def time_embed(t, dim=16):
#     half = dim // 2
#     freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)    # f_k from 1 down to about 1/10000
#     ang = t.float()[:, None] * freqs[None]                               # an angle for every (step, frequency) pair
#     return torch.cat([torch.sin(ang), torch.cos(ang)], 1)                # first half sines, second half cosines
# ```
#
# Every column is a wave in t: column 0 is the fastest (it changes with every step), the last sine column is the slowest (it hardly changes over 100 steps). Together they work like the digits of a clock: fast hands give the fine position, slow hands the rough one.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def time_embed(t, dim=16):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)    # f_k from 1 down to about 1/10000
    ang = t.float()[:, None] * freqs[None]                               # an angle for every (step, frequency) pair
    return torch.cat([torch.sin(ang), torch.cos(ang)], 1)                # first half sines, second half cosines

# %% [markdown]
# Now the network. It uses `time_embed` and lets you switch the time input off (`use_time=False`), which Part 5 needs. The count by hand: the first layer has 18 × 128 + 128 = 2,432 numbers, each 128-to-128 layer has 128 × 128 + 128 = 16,512 numbers (two of them), the output layer has 128 × 2 + 2 = 258: together 2,432 + 33,024 + 258 = 35,714.

# %%
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

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes four ticks. The network has **35,714** parameters (2,432 + 16,512 + 16,512 + 258) and the same network without the time embedding has **33,666** (its first layer has 2 × 128 + 128 = 384 numbers instead of 2,432). A batch of 5 points goes in and a batch of 5 noise guesses of shape (5, 2) comes out. The assertion makes sure that your reading of the architecture is the one used in the rest of the lab.
#
# *What it means.* The time embedding costs only 2,048 parameters (16 extra inputs × 128), under six percent of the network, but it carries one piece of information the network cannot get anywhere else: the step number. In Part 5 you remove it and see what that costs.
#
# Tips and common mistakes:
#
# - The two halves must be sines first and then cosines in **both** training and sampling; the network learns one fixed convention.
# - Pass `t` as a LongTensor of step numbers 1 to T, not as the fraction t/T: the frequencies are designed for whole steps.

# %% [markdown]
# ### Step 4.3. The loss and the training loop
#
# The loss is the mean squared error between the network's guess ε̂ and the true noise ε. That is all: no discriminator, no second network, no adversarial game. One training step: draw a batch of 256 clean points, make the pairs with `make_batch`, compute the loss, take an Adam step. The loss will **not** go to 0: for most noisy points the added noise cannot be told apart exactly (a point near a blob could have come from this blob with a little noise or from the next one with a lot), so the best possible loss is clearly above 0.

# %% [markdown]
# ### ✏️ Your turn: The training step
#
# *Exercise 4.3:* Write `train_step(model, opt, x0, S)`: draw a training batch with `make_batch`, compute `F.mse_loss(model(xt, t), eps)`, zero the gradients, call `backward()`, take the optimiser step and return the loss as a Python float. Why it matters: every model in this lab, from the 2-D denoiser to the UNet of Part 6, is trained with exactly these five lines. The check trains a new network for 300 steps with your function: the loss must fall clearly (a fresh network starts near 1, the loss of always guessing 0 noise).

# %%
def train_step_mine(model, opt, x0, S):
    """One optimisation step on a batch x0; returns the loss as a float."""
    return None   # YOUR CODE: xt, t, eps = make_batch(...); loss = F.mse_loss(model(xt, t), eps); opt.zero_grad(); loss.backward(); opt.step()

# %%
if train_step_mine(MLPDenoiser(), torch.optim.Adam(net0.parameters()), data_mix[:8], S) is None:
    print("Fill in train_step_mine above, then run this cell again.")
else:
    torch.manual_seed(0); m_ = MLPDenoiser(); o_ = torch.optim.Adam(m_.parameters(), lr=1e-3)
    ls_ = [train_step_mine(m_, o_, data_mix[torch.randint(0, 4000, (256,))], S) for _ in range(300)]
    check("the loss at the start is near 1 (the loss of guessing zero noise)", 0.7 < float(np.mean(ls_[:5])) < 1.6, "the mean of ls_[:5] was %.2f; expected about 1" % float(np.mean(ls_[:5])))
    check("after 300 steps the loss has fallen below 0.7", float(np.mean(ls_[-30:])) < 0.7, "the mean of the last 30 losses was %.2f; did you call opt.step()?" % float(np.mean(ls_[-30:])))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def train_step(model, opt, x0, S):
#     xt, t, eps = make_batch(x0, S)                   # fresh noisy examples
#     loss = F.mse_loss(model(xt, t), eps)             # how far is the guessed noise from the true noise?
#     opt.zero_grad(); loss.backward(); opt.step()
#     return float(loss)
# ```
#
# The loss compares the guess with the noise we ourselves drew, so every example has a correct answer: this is why training a diffusion model is as steady as training a classifier.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def train_step(model, opt, x0, S):
    xt, t, eps = make_batch(x0, S)                   # fresh noisy examples
    loss = F.mse_loss(model(xt, t), eps)             # how far is the guessed noise from the true noise?
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)

# %% [markdown]
# The full loop: 4,000 steps with batch 256, Adam with learning rate 1e-3. We keep copies of the weights after 500 and 1,500 steps (Step 4.6 uses them). On a CPU this takes about six to twelve seconds. With `QUICK` (the test mode that the Setup cell can switch on) the loop is shortened to 600 steps; the numbers in this text are for the full run.

# %%
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
print("loss at steps 1-50: %.3f   steps 451-500: %.3f   last 200 steps: %.3f" % (np.mean(losses42[:50]), np.mean(losses42[CK_MLP[0] - 50:CK_MLP[0]]), np.mean(losses42[-200:])))
RESULTS["p4_final_loss"] = round(float(np.mean(losses42[-200:])), 3)
sm = np.convolve(losses42, np.ones(50) / 50, mode="valid")
plt.figure(figsize=(6, 3)); plt.plot(losses42, alpha=0.25, label="loss per step"); plt.plot(np.arange(49, len(losses42)), sm, label="average of 50 steps")
plt.xlabel("training step"); plt.ylabel("MSE of the noise guess"); plt.ylim(0, 1.2); plt.legend(); plt.title("the loss falls fast and then stays above 0"); plt.show()

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes two ticks. The training loop takes about **6 seconds** on our CPU (up to about 12 on a slower machine) for 4,000 steps. The loss averaged over steps 1 to 50 is **0.716**, over steps 451 to 500 it is **0.324**, and over the last 200 steps it is **0.254** (tolerance: between 0.2 and 0.3 for seed 42). The plot shows a fast fall in the first 300 steps and then a slow, noisy decline: the faint blue line is the raw loss of every step, the orange line its average.
#
# *What it means.* The loss stops at about 0.25, not at 0. The part that cannot be predicted is real: at a step with a lot of noise a point lying between two blobs could have come from either of them, and the best guess of the noise is an average. A loss of 0.25 means that the squared error of the noise guess is a quarter of the noise's own variance (1). Do not use "the loss is low" as a quality criterion: the loss at 1,500 steps (0.31) is not much higher than at 4,000 steps (0.25), yet in Step 4.6 the samples are very different.
#
# Tips and common mistakes:
#
# - Call `opt.zero_grad()` before `loss.backward()`; without it the gradients pile up from step to step.
# - Do not average a loss over steps of very different t and then compare it with a single-step loss: Step 4.7 shows that the loss depends strongly on t.

# %% [markdown]
# ### Step 4.4. Sample with the trained network
#
# Sampling is the reverse loop of Part 3 with the trained network in place of the exact predictor. Start from 2,000 points of pure noise N(0, I). For t = T, T − 1, …, 1 compute the guess ε̂ = model(x_t, t) and take the reverse step x_(t−1) = (x_t − β_t/√(1 − ᾱ_t)·ε̂)/√α_t + σ_t·z with σ_t² = β_t and a fresh standard normal z (at the last step, t = 1, no fresh noise: z = 0). First you write the one reverse step, then the loop calls it 100 times.

# %% [markdown]
# ### ✏️ Your turn: One reverse step
#
# *Exercise 4.4:* Write `reverse_step_mine(x, e, t, S, z)`: given the current points `x` (shape (B, 2)), the network's guess `e`, the step number `t` (a plain integer) and the fresh noise `z`, return x_(t−1). Use `S["beta"][t]`, `S["alpha"][t]` and `S["abar"][t]`. Why it matters: it is the same formula as Part 3.1, now with ε̂ from a network; a wrong factor here gives pictures that explode or fade, and Step 6.7 shows what happens when the 1/√α_t is forgotten. The check compares your step with the formula evaluated number by number at t = 50 and at t = 1.

# %%
def reverse_step_mine(x, e, t, S, z):
    """One reverse step: from x_t to x_(t-1)."""
    return None   # YOUR CODE: mean = (x - beta_t / sqrt(1 - abar_t) * e) / sqrt(alpha_t); then add sqrt(beta_t) * z

# %%
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

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def reverse_step(x, e, t, S, z):
#     b, a, ab = S["beta"][t], S["alpha"][t], S["abar"][t]
#     mean = (x - b / (1 - ab).sqrt() * e) / a.sqrt()       # the posterior mean: remove the guessed noise, then undo the shrinking of the step
#     return mean + b.sqrt() * z                            # add fresh noise with variance beta_t
# ```
#
# The mean takes out the part of x_t that the network says is noise and rescales by 1/√α_t (a forward step shrank the picture by √α_t); the fresh noise z keeps the particles spread out like real data instead of collapsing to a few points (Part 3.4).
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def reverse_step(x, e, t, S, z):
    b, a, ab = S["beta"][t], S["alpha"][t], S["abar"][t]
    mean = (x - b / (1 - ab).sqrt() * e) / a.sqrt()       # the posterior mean: remove the guessed noise, then undo the shrinking of the step
    return mean + b.sqrt() * z                            # add fresh noise with variance beta_t

# %% [markdown]
# Now the loop. `sample_ddpm` takes any function `eps_fn(x, t)` that returns the noise guess for a batch (so it works for the MLP now and for the UNet in Part 6), a shape, a schedule and a seed. With `keep` you can ask for copies of x_t at chosen steps (Step 4.5 uses this); the option `fix_scale=False` leaves out the 1/√α_t and is only there for Step 6.7.

# %%
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

# %% [markdown]
# A second seed shows how much the result depends on luck. The cell trains a second network (seed 1) and samples from it.

# %%
model1, losses1, _ = train_mlp(seed=1)
model1.eval()
gen1 = sample_ddpm(lambda x, t: model1(x, t), (2000, 2), S, seed=0).numpy()
cov_1, q_1 = modes_covered(gen1, centres)
print("diffusion s1  : %d of 8 blobs, %.3f of the points on a blob   (final loss %.3f)" % (cov_1, q_1, np.mean(losses1[-200:])))
RESULTS["p4_blobs_s1"] = cov_1; RESULTS["p4_on_blob_s1"] = round(q_1, 3)
if not QUICK:
    assert cov_g >= 7 and cov_1 >= 7 and q_g > 0.6 and q_1 > 0.6, "the diffusion model should cover (almost) all eight blobs with most points on a blob"

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes two ticks. Sampling 2,000 points takes **0.1 seconds** (100 network calls on a batch of 2,000). The real data (checked with the same function) cover **8 of 8** blobs with **0.990** of the points on a blob (a point counts as "on a blob" within three standard deviations of a centre, so 1 − 0.990 is the 1 percent of true points that fall outside by chance). The diffusion model with seed 42 covers **8 of 8** blobs with **0.917** of its points on a blob; the second seed gives **8 of 8** and **0.912** (final loss 0.261). Tolerance: 8 (at least 7) of 8 blobs and a share between 0.85 and 0.95; the assertion in the cell fails if fewer than 7 blobs are found or fewer than 60 percent of the points are on a blob. The scatter plot shows eight clean clusters with a few points in between the blobs (the 8 percent that are not on a blob).
#
# *What it means.* Compare with your guess from the beginning of the part, and with Week 12: the GAN needed a careful choice of learning rates and often collapsed onto a few blobs; the diffusion model, trained with an ordinary squared-error loss and no second network, found all eight blobs with both seeds, and with the same settings. The price is in sampling: 100 network calls per point instead of the GAN's one. The 8 percent of points that are off a blob lie between blobs or in their tails; Step 4.6 shows that this share is much larger for a briefly trained network.
#
# Tips and common mistakes:
#
# - Put the model in `eval()` mode before sampling (the MLP has no dropout, but it is a good habit) and use `torch.no_grad()`: the decorator on `sample_ddpm` does this.
# - No fresh noise at the last step (t = 1): otherwise you add noise to the finished point and lose sharpness.

# %% [markdown]
# ### Step 4.5. Look at the trajectories
#
# Why does this work? Because each reverse step moves every point a little towards where data are more likely, and the fresh noise makes sure the points do not all fall into the nearest blob. A picture of the paths makes this visible: we run the sampler again with copies of x_t at all steps and draw the paths of 20 points as lines from noise (t = 100) to data (t = 0), and the whole cloud of 1,000 points at five moments.

# %%
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

# %% [markdown]
# ### ✏️ Your turn: Straight or winding?
#
# *Exercise 4.5:* Compute for each of the 1,000 paths the **straightness**: the straight-line distance from the start (t = 100) to the end (t = 0) divided by the total length of the path (the sum of the lengths of the 100 small steps). A value of 1 means a straight line, a small value means a winding path. Store the mean in `straightness`. Why it matters: it turns the picture into one number and shows that the fresh noise z makes the paths wiggle (the DDIM sampler of Part 6 removes that wiggle). Use `path`: its first axis is the step, the second the point, the third x and y.

# %%
straightness = None   # YOUR CODE: total = (path[1:] - path[:-1]).norm(dim=2).sum(0); direct = (path[-1] - path[0]).norm(dim=1); straightness = float((direct / total).mean())

# %%
if straightness is None:
    print("Fill in straightness above, then run this cell again.")
else:
    check("straightness is between 0 and 1", 0 < straightness <= 1.0, "a path cannot be shorter than the straight line")
    check("the paths wind (straightness below 0.7)", straightness < 0.7, "total length = sum over the 100 steps of the distance between neighbouring positions")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# total = (path[1:] - path[:-1]).norm(dim=2).sum(0)       # length of each path: the sum of its 100 small steps
# direct = (path[-1] - path[0]).norm(dim=1)               # straight-line distance from the starting noise to the final point
# straightness = float((direct / total).mean())
# print("mean straightness: %.3f" % straightness)
# RESULTS["p4_straightness"] = round(straightness, 3)
# ```
#
# Most of the length of a path is wiggle caused by the fresh noise z at every step (each step adds noise with a standard deviation of √β_t, which is large for the last steps of the schedule). The wiggle is not a mistake: it is the randomness that lets the same starting region end up in different blobs.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
total = (path[1:] - path[:-1]).norm(dim=2).sum(0)       # length of each path: the sum of its 100 small steps
direct = (path[-1] - path[0]).norm(dim=1)               # straight-line distance from the starting noise to the final point
straightness = float((direct / total).mean())
print("mean straightness: %.3f" % straightness)
RESULTS["p4_straightness"] = round(straightness, 3)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes two ticks. The path array has the shape (101, 1000, 2). The mean distance of the points from the origin is **1.27** at t = 100 (the mean length of a 2-D standard normal vector is 1.25), **1.44** at t = 50 and **3.02** at t = 0 (the blobs lie on a circle of radius 3). The mean straightness is **0.084**: the length of the straight line from start to end is only about 8 percent of the length of the path (tolerance: between 0.05 and 0.15). In the left panel the 20 paths wander and finally settle in blobs on the ring; the five snapshots show a round cloud at t = 100, a smaller, ring-like cloud at t = 50 and eight clusters at t = 0.
#
# *What it means.* The points do not travel along straight lines from noise to data. Each step takes a small drift towards the data (from the network) plus a random kick (the fresh noise z) of size √β_t, which is large in the late steps of our schedule (β_100 = 0.2). The cloud grows only slowly at first (mean distance from the origin 1.27 at t = 100 and 1.44 at t = 50) and in the last 50 steps the points move out to the ring at distance 3. This is also why DDIM (Part 6), which has no random kick, can take much longer jumps.

# %% [markdown]
# ### Step 4.6. Fewer training steps
#
# How much training does the model need? Because we saved the weights after 500 and 1,500 steps of the same run, we can sample from all three versions with the same starting noise and compare. We use the two numbers of `modes_covered`: the count of blobs reached and the share of points on a blob. Counting blobs alone would be too generous, since a cloud of points spread everywhere also "reaches" every blob; the second number tells you how many points really sit on one.

# %%
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

# %% [markdown]
# ### ✏️ Your turn: Reading the table
#
# *Exercise 4.6:* From `rows46` find the first checkpoint at which the share on a blob is above 0.6, and store its step number in `first_good` (use `None` if there is none). Then compute `loss_drop`, the loss of the first row minus the loss of the last row. Why it matters: it shows that the loss and the quality of the samples do not move together evenly: the loss changes little in the second half of training while the samples change a lot.

# %%
first_good = "unset"   # YOUR CODE: the step of the first row of rows46 whose share on a blob (the last entry of the row) is above 0.6, or None
loss_drop = None       # YOUR CODE: rows46[0][1] - rows46[-1][1]

# %%
if first_good == "unset" or loss_drop is None:
    print("Fill in first_good and loss_drop above, then run this cell again.")
else:
    check("first_good is a checkpoint step or None", first_good is None or first_good in CK_MLP, "it must be one of %s" % (CK_MLP,))
    check("loss_drop is the difference of the loss column", abs(loss_drop - (rows46[0][1] - rows46[-1][1])) < 1e-9, "rows46[0][1] - rows46[-1][1]")
    if not QUICK: check("the first good checkpoint is the last one in our run", first_good == CK_MLP[-1], "look at the share on a blob in the last column")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# first_good = next((r[0] for r in rows46 if r[3] > 0.6), None)
# loss_drop = rows46[0][1] - rows46[-1][1]
# print("first checkpoint with more than 60 percent of the points on a blob:", first_good, "  loss drop: %.3f" % loss_drop)
# ```
#
# `next(..., None)` takes the first row that satisfies the condition. The loss falls by only a little between the first and the last checkpoint, but the share of points on a blob rises from about a third to over 90 percent: the small differences in the noise guess are exactly what separates a blurry cloud from sharp blobs.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
first_good = next((r[0] for r in rows46 if r[3] > 0.6), None)
loss_drop = rows46[0][1] - rows46[-1][1]
print("first checkpoint with more than 60 percent of the points on a blob:", first_good, "  loss drop: %.3f" % loss_drop)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The scatter plots show the same starting noise with three versions of the network. The table: after **500** steps the loss over the last 100 steps is 0.320, all 8 blobs are "covered" but only **0.307** of the points are on a blob; after **1,500** steps the loss is 0.307 and the share is **0.379**; after **4,000** steps the loss is 0.255 and the share is **0.917**. The Check passes two ticks and, in a full run, the third (first_good is 4000); the answer prints `4000` and a loss drop of 0.065.
#
# *What it means.* The loss moved by 0.065 (from 0.320 to 0.255, about 20 percent) while the share of points on a blob tripled. The early versions know roughly where the data are (a ring) but not yet where the eight blobs lie: the points land on the ring, between blobs, and `modes_covered` still counts all 8 blobs since each blob receives the minimum 2 percent of the points. This is why we always report two numbers: the blob count alone would have said "8 of 8" for a network that has barely learned anything. (In this run the last 2,500 steps made the difference; a shortened run with 1,500 steps would give a poor picture of what the method can do.)
#
# Tips and common mistakes:
#
# - The checkpoints belong to one run (seed 42), so the three rows are comparable; do not compare them with the seed-1 run.
# - With `QUICK` (a 600-step run) the checkpoints are at 75, 225 and 600 steps and the third Check is skipped.

# %% [markdown]
# ### Step 4.7. Imagine and write: easy and hard noise levels
#
# Which part of the denoiser's job is easy and which is hard? At large t (almost pure noise) the best guess is simple: x_t is nearly all noise, so ε̂ is about x_t. At small t (a little noise on a point that is almost clean) the network must tell the tiny noise apart from the fine structure of the data. We measure it: the loss of the trained network in five groups of t (1 to 20, 21 to 40, and so on), always with the same 20,000 evaluation pairs. The function `loss_by_bin` is reused in Part 5.

# %%
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

# %% [markdown]
# ### ✏️ Your turn: Write your reading
#
# *Exercise 4.7:* Write two or three sentences in `answer_4_7`: in which group of t is the loss highest, in which lowest, and why, in your own words? Use the table above and what you know about the noise at those levels (Parts 1 and 2). Then say what this suggests for how much time a training run should spend on each noise level.

# %%
answer_4_7 = ""   # YOUR TEXT HERE

# %%
if answer_4_7.strip() == "":
    print("Write your answer in answer_4_7 above, then run this cell again.")
else:
    hi_, lo_ = int(np.argmax(lb42)), int(np.argmin(lb42))
    print("Your answer has %d words. In our run the highest loss is in group %d (t = %d-%d) and the lowest in group %d (t = %d-%d)." % (len(answer_4_7.split()), hi_ + 1, 20 * hi_ + 1, 20 * hi_ + 20, lo_ + 1, 20 * lo_ + 1, 20 * lo_ + 20))
    check("you wrote at least 15 words", len(answer_4_7.split()) >= 15, "two or three sentences")

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The table: in the five groups of t the loss is **0.416** (t from 1 to 20), **0.562** (21 to 40), **0.271** (41 to 60), **0.034** (61 to 80) and **0.003** (81 to 100). The cell prints the same table for you; with a different run the numbers differ in the third digit. The Check only counts your words and prints where the highest and lowest group lie in your run.
#
# *What it means.* The easy part is the late half of the noising process (t above 60): the point is nearly all noise, ε̂ is nearly x_t, and the loss is almost zero (0.003 above t = 80). The hardest groups are the early and middle steps, with the maximum between t = 21 and t = 40, where ᾱ_t falls from 0.64 to 0.19 (at t = 27 the noise already carries as much power as the signal). There the network must decide, from a noisy point, which of the eight blobs the point came from, and how far it is from there. At very small t (1 to 20) there is little noise but the blobs are narrow (spread 0.15), so the exact blob position still matters. The consequence for a training run is that equal numbers of examples for every t (our choice) spend a lot of the work on t above 60, where there is little to learn; some models draw the noise levels non-uniformly to focus on the hard region (this is the topic of the schedule comparison of Part 2).

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_4_7 = "The loss is highest for t from 21 to 40 (0.56) and lowest above t = 80 (0.003). Late in the process x_t is almost only noise, so predicting epsilon as x_t is easy; in the middle the network must work out which blob the point came from. So a run should not spend most of its effort on large t."
# print(answer_4_7)
# ```
#
# The highest loss is in the middle of the noising process and the lowest at the end, where the guess "all of x_t is noise" is nearly right. A model gains little from many examples with t above 80.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
answer_4_7 = "The loss is highest for t from 21 to 40 (0.56) and lowest above t = 80 (0.003). Late in the process x_t is almost only noise, so predicting epsilon as x_t is easy; in the middle the network must work out which blob the point came from. So a run should not spend most of its effort on large t."
print(answer_4_7)

# %% [markdown]
# ## Part 5. Time embeddings, the UNet and its parts (30 min)
#
# *Lesson section: "UNet".* The denoiser of Part 4 was a small MLP on two numbers. Pictures have 256 numbers (even our tiny 16 × 16 ones) and the same network layout would be wasteful, so image diffusion models use a **UNet**: a convolutional network that shrinks the picture, works at several resolutions, grows it back, and receives the step number everywhere. ==You will compute the time embedding by hand and look at it, remove it from the 2-D denoiser to see what it was doing, write the building block that injects the time into a convolution, assemble a small UNet and count its parameters part by part, test what the skip connections do, and compute how much of the picture a neuron can see.== The UNet you build here has exactly 135,713 parameters, the number in the lesson's `unet` figure, and is the network of Part 6. This part continues in the same session as Part 4 (it uses `S`, `train_mlp`, `time_embed`, `loss_by_bin`, `model42` and the data of Part 4).

# %% [markdown]
# ### ✏️ Imagine first
#
# In Part 4 the network was told the step number t through a time embedding. Suppose we take that away: the network gets only the noisy point x_t and must guess the noise, without being told whether it is looking at a little noise (t = 5) or almost only noise (t = 90). Is the job impossible, harder but possible, or no different? Think about what x_t itself reveals about t. Write your guess in the cell below. You compare it with Step 5.2.

# %%
# Imagine first: what happens to the denoiser when it is not told the step number?
my_guess_notime = None    # for example "it fails completely", "a bit worse" or "no different": write one of them

# %% [markdown]
# ### Step 5.1. The time embedding by hand
#
# You wrote `time_embed` in Step 4.2. Here we compute it by hand for a tiny size (dim 4), to see exactly what the network receives. With dim 4 we have half = 2 frequencies: f_0 = exp(0) = 1 and f_1 = exp(−ln(10000)/2) = 0.01. Our convention is **first the sines, then the cosines**, so the embedding of step t is [sin(t), sin(0.01·t), cos(t), cos(0.01·t)]. (Some libraries interleave sine and cosine; the network does not care, as long as the convention stays the same during training and sampling.) For t = 1 this is [0.8415, 0.0100, 0.5403, 1.0000]: the fast wave has already turned by one radian, the slow wave has hardly moved.

# %% [markdown]
# ### ✏️ Your turn: The embedding by hand
#
# *Exercise 5.1:* Write `embed_by_hand(t, dim)` with plain Python (the `math` module, no PyTorch) that returns a list of `dim` numbers in our convention: first the `dim // 2` sines then the `dim // 2` cosines of t·f_k, with f_k = exp(−ln(10000)·k/(dim // 2)). Why it matters: if you can compute the embedding with a calculator, you understand what the network is given; the check compares your list with `time_embed` for t = 1, 100 and 300.

# %%
def embed_by_hand(t, dim=4):
    """The sinusoidal embedding of one step number t as a plain Python list."""
    return None   # YOUR CODE: half = dim // 2; f = [math.exp(-math.log(10000) * k / half) for k in range(half)]; sines first, then cosines

# %%
if embed_by_hand(1) is None:
    print("Fill in embed_by_hand above, then run this cell again.")
else:
    for tt in (1, 100, 300):
        mine = np.array(embed_by_hand(tt, 4)); ref = time_embed(torch.tensor([tt]), 4)[0].numpy()
        check("t = %3d: %s" % (tt, np.round(mine, 4).tolist()), mine.shape == (4,) and np.allclose(mine, ref, atol=1e-4), "sines first: sin(t), sin(0.01 t), then cosines: cos(t), cos(0.01 t); the reference is %s" % np.round(ref, 4).tolist())

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def embed_by_hand(t, dim=4):
#     half = dim // 2
#     f = [math.exp(-math.log(10000) * k / half) for k in range(half)]      # the frequencies, from 1 down to about 1/10000
#     return [math.sin(t * fk) for fk in f] + [math.cos(t * fk) for fk in f]
# for tt in (1, 100, 300):
#     print("t = %3d:" % tt, [round(v, 4) for v in embed_by_hand(tt, 4)])
# ```
#
# For t = 100 the fast wave sin(100) is far from the one of t = 101 (the wave has turned by a full radian between neighbours) while the slow wave sin(1) hardly differs: the pair together tells the network both "roughly where in the schedule" (slow waves) and "exactly which step" (fast waves).
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def embed_by_hand(t, dim=4):
    half = dim // 2
    f = [math.exp(-math.log(10000) * k / half) for k in range(half)]      # the frequencies, from 1 down to about 1/10000
    return [math.sin(t * fk) for fk in f] + [math.cos(t * fk) for fk in f]
for tt in (1, 100, 300):
    print("t = %3d:" % tt, [round(v, 4) for v in embed_by_hand(tt, 4)])

# %% [markdown]
# Now look at the whole embedding for t = 0 to 200 with dim 32 as a heat map: every row is one step, every column one wave. Fast columns (left, and the matching cosine columns) form a fine pattern, slow columns change gradually. Then compare embeddings of neighbouring and distant steps with the **cosine similarity** cos(a, b) = a·b/(|a||b|): 1 means the same direction, 0 unrelated.

# %%
E = time_embed(torch.arange(0, 201), 32).numpy()
plt.figure(figsize=(7, 3.2)); plt.imshow(E.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1); plt.colorbar(label="value"); plt.xlabel("step t"); plt.ylabel("column (16 sines, then 16 cosines)"); plt.title("sinusoidal time embedding, dim 32"); plt.show()
def cos_sim(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
for other in (101, 110, 200):
    print("cosine similarity of the embeddings of t = 100 and t = %3d: %.3f" % (other, cos_sim(E[100], E[other])))
RESULTS["p5_cos_100_101"] = round(cos_sim(E[100], E[101]), 3); RESULTS["p5_cos_100_200"] = round(cos_sim(E[100], E[200]), 3)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes three ticks (one per step number). The embeddings are, in our convention [sin t, sin 0.01·t, cos t, cos 0.01·t]: for t = 1 **[0.8415, 0.0100, 0.5403, 1.0000]**, for t = 100 **[−0.5064, 0.8415, 0.8623, 0.5403]** and for t = 300 **[−0.9998, 0.1411, −0.0221, −0.9900]**. In the heat map (dim 32, t from 0 to 200) the first columns (the fast waves) form fine stripes, the last columns of each half change so slowly that they look like broad colour bands. The cosine similarity of the embeddings of t = 100 with t = 101 is **0.957**, with t = 110 it is **0.629** and with t = 200 it is **0.583** (tolerance: ±0.02).
#
# *What it means.* Neighbouring steps have very similar embeddings (0.957): the network gets a smooth signal and treats step 101 almost like step 100. Ten steps apart the similarity is already down to 0.63, so the network can tell them apart, but note that t = 200 is not much less similar to t = 100 (0.58) than t = 110 is: the slow waves are nearly the same, and the information that separates them is in the fast waves. A network that reads all columns gets a unique "fingerprint" for every step.
#
# Tips and common mistakes:
#
# - The slowest frequency in a size-32 embedding is exp(−ln(10000)·15/16) = 0.00018: it is practically constant over 100 steps. Larger T needs the same formula, which is why it was designed this way.

# %% [markdown]
# ### Step 5.2. Ablation: no time input
#
# We now retrain the 2-D denoiser of Part 4 without the time embedding (`use_time=False`: the network sees only the two coordinates of x_t), with the same 4,000 steps, the same data and the same seed, and compare the final loss, the blobs covered and the share of points on a blob. We also compare the loss in the five groups of t of Step 4.7. A fair expectation is that the network will still learn *something*, because x_t reveals the noise level through its distance from the blobs (a point far outside the ring must have had a lot of noise), but only roughly.

# %% [markdown]
# ### ✏️ Your turn: Train without time
#
# *Exercise 5.2:* Train a no-time denoiser with `train_mlp(seed=42, use_time=False)`, sample 2,000 points with `sample_ddpm` (seed 0, as in Step 4.4) and store the mean of its last 200 losses in `loss_no_time`, and the two numbers returned by `modes_covered` in `blobs_no_time` and `on_blob_no_time`. Why it matters: an ablation (removing one part and measuring) is how you find out which parts of a design earn their place. The check compares with the network of Part 4.

# %%
loss_no_time = None; blobs_no_time = None; on_blob_no_time = None    # YOUR CODE: three numbers from a no-time run (see the exercise)

# %%
if loss_no_time is None:
    print("Fill in the three numbers above, then run this cell again.")
else:
    check("the loss of the no-time model is above the loss of the model with time", loss_no_time > float(np.mean(losses42[-200:])) + 0.03, "expected %.3f for the model with time; no-time should be clearly higher" % float(np.mean(losses42[-200:])))
    check("blobs_no_time is a whole number from 0 to 8", blobs_no_time in range(0, 9), "the first value returned by modes_covered")
    check("on_blob_no_time is a share between 0 and 1", 0 <= on_blob_no_time <= 1, "the second value returned by modes_covered")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# model_nt, losses_nt, _ = train_mlp(seed=42, use_time=False)
# model_nt.eval()
# gen_nt = sample_ddpm(lambda x, t: model_nt(x, t), (2000, 2), S, seed=0).numpy()
# loss_no_time = float(np.mean(losses_nt[-200:]))
# blobs_no_time, on_blob_no_time = modes_covered(gen_nt, centres)
# print("no time input: loss %.3f, %d of 8 blobs, %.3f of the points on a blob" % (loss_no_time, blobs_no_time, on_blob_no_time))
# RESULTS["p5_notime_loss"] = round(loss_no_time, 3); RESULTS["p5_notime_on_blob"] = round(on_blob_no_time, 3)
# ```
#
# The loss is much higher (it stays above 0.5 where the network with time goes down to about 0.25) and far fewer points land on a blob. The network could tell the noise level only roughly from the position of x_t.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
model_nt, losses_nt, _ = train_mlp(seed=42, use_time=False)
model_nt.eval()
gen_nt = sample_ddpm(lambda x, t: model_nt(x, t), (2000, 2), S, seed=0).numpy()
loss_no_time = float(np.mean(losses_nt[-200:]))
blobs_no_time, on_blob_no_time = modes_covered(gen_nt, centres)
print("no time input: loss %.3f, %d of 8 blobs, %.3f of the points on a blob" % (loss_no_time, blobs_no_time, on_blob_no_time))
RESULTS["p5_notime_loss"] = round(loss_no_time, 3); RESULTS["p5_notime_on_blob"] = round(on_blob_no_time, 3)

# %% [markdown]
# The table for both seeds and both networks, with the loss per group of t (Step 4.7). The second seed needs two more training runs of a few seconds each.

# %%
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

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes three ticks. The no-time network ends with a loss of **0.543** (seed 42; **0.556** for seed 1) against **0.254** (**0.261**) for the network with time, covers **8 of 8** blobs but puts only **0.363** (seed 1: **0.412**) of its points on a blob, against **0.917** (**0.912**). The table with the loss in five groups of t: with time (seed 42) 0.416, 0.562, 0.271, 0.034, 0.003; without time 0.945, 0.985, 0.403, 0.226, 0.218; for seed 1 the numbers are 0.413, 0.564, 0.271, 0.035, 0.006 (with time) and 0.916, 0.963, 0.413, 0.244, 0.239 (without time). Tolerance: the no-time loss is between 0.45 and 0.65 and the share on a blob between 0.25 and 0.5.
#
# *What it means.* Compare with your guess: the job was **not impossible**, but much worse. The no-time network still learns something, as we expected: it finds all eight blobs, because x_t reveals roughly how far it is from a blob. But in the late steps (t above 60, the groups 4 and 5) its loss is 0.22 to 0.24 where the network with time has 0.03 and 0.003, a factor of 7 to 70. At t = 100 and t = 70 a noisy point looks almost the same (a standard normal cloud), so without t the network cannot know whether to trust the position of x_t (t = 60: ᾱ = 0.02, some signal left) or to ignore it (t = 100: nothing left); it learns a compromise that is wrong at both. In the early groups the loss without time is 0.95 to 0.99 (nearly the value of "always guess 0": 1.0) against 0.42 to 0.56. The lesson: **the network needs to be told the noise level**, even though a good part of it can be guessed from the data.
#
# Tips and common mistakes:
#
# - The training and the sampling must use the same switch: a no-time network is also sampled without time (the model object remembers `use_time`).

# %% [markdown]
# ### Step 5.3. Where the time goes in: the TimeBlock
#
# In a UNet the step number must reach every resolution, not only the input. The standard way is a **block** with two 3 × 3 convolutions and a time projection between them: the embedding (32 numbers) goes through a linear layer that makes one number per channel, and this number is *added* to every pixel of that channel after the first convolution. So the block can behave differently for different noise levels. The formula for its parameters is 9·c_in·c_out + c_out (conv 1) + time_dim·c_out + c_out (the projection) + 9·c_out·c_out + c_out (conv 2).

# %% [markdown]
# ### ✏️ Your turn: The block
#
# *Exercise 5.3:* Write the class `TimeBlock_mine(c_in, c_out, time_dim)`. Its `forward(x, emb)` does: `h = conv1(x)` (3 × 3, padding 1), then `h = h + proj(emb)[:, :, None, None]` (`proj` is a `Linear(time_dim, c_out)`; the two `None` axes let one number per channel be added to every pixel), then `h = silu(h)`, then `silu(conv2(h))` with a second 3 × 3 convolution with `c_out` channels. If `emb` is `None`, skip the addition (Step 6.7 needs this). Why it matters: this block is the repeated unit of the UNet. The check counts the parameters against the formula and tests the shapes.

# %%
class TimeBlock_mine(nn.Module):
    def __init__(self, c_in, c_out, time_dim=32):
        super().__init__()
        # YOUR CODE: self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1); self.proj = nn.Linear(time_dim, c_out); self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
    def forward(self, x, emb=None):
        return x   # YOUR CODE: conv1, then add the projected embedding (unless emb is None), SiLU, conv2, SiLU

# %%
def timeblock_params(c_in, c_out, time_dim=32):
    return 9 * c_in * c_out + c_out + time_dim * c_out + c_out + 9 * c_out * c_out + c_out
tb_ = TimeBlock_mine(1, 16, 32)
check("the parameters of TimeBlock(1, 16) are %d by the formula" % timeblock_params(1, 16), count_params(tb_) == timeblock_params(1, 16), "you have %d; conv1 3x3, a Linear(32, 16) and conv2 3x3 are needed" % count_params(tb_))
try:
    y_ = tb_(torch.randn(2, 1, 16, 16), torch.randn(2, 32))
    check("the output has shape (2, 16, 16, 16)", tuple(y_.shape) == (2, 16, 16, 16), "padding=1 keeps the size, c_out channels")
except Exception as err:
    print("✗ the block does not run yet:", err)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# class TimeBlock(nn.Module):
#     """conv 3x3 -> add a projection of the time embedding -> SiLU -> conv 3x3 -> SiLU"""
#     def __init__(self, c_in, c_out, time_dim=32):
#         super().__init__()
#         self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
#         self.proj = nn.Linear(time_dim, c_out)                # one number per channel
#         self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
#     def forward(self, x, emb=None):
#         h = self.conv1(x)
#         if emb is not None: h = h + self.proj(emb)[:, :, None, None]     # the same number is added at every pixel of a channel
#         return F.silu(self.conv2(F.silu(h)))
# ```
#
# The projection is how the noise level enters: a different embedding gives a different number per channel, so the block can switch its behaviour (smooth strongly at high noise, keep detail at low noise).
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
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

# %% [markdown]
# The five blocks of the UNet you build next are TimeBlock(1, 16), (16, 32), (32, 64), (64, 32) and (32, 16). We check the formula against `count_params` for all five and look at the shapes.

# %%
def timeblock_params(c_in, c_out, time_dim=32):
    return 9 * c_in * c_out + c_out + time_dim * c_out + c_out + 9 * c_out * c_out + c_out
print("block (c_in, c_out)   by the formula   count_params")
for ci, co in ((1, 16), (16, 32), (32, 64), (64, 32), (32, 16)):
    b_ = TimeBlock(ci, co, 32)
    print("TimeBlock(%2d, %2d)   %14d   %12d" % (ci, co, timeblock_params(ci, co), count_params(b_)))
    assert count_params(b_) == timeblock_params(ci, co)
x_in = torch.randn(4, 16, 8, 8); emb_in = torch.randn(4, 32)
print("TimeBlock(16, 32) on a (4, 16, 8, 8) batch ->", tuple(TimeBlock(16, 32)(x_in, emb_in).shape), "  without time ->", tuple(TimeBlock(16, 32)(x_in).shape))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The first block of the check prints two crosses until you fill in the class: with the starter (no layers) the parameter count is 0 and the shape check fails; with your class both ticks appear. The table for the five blocks of the UNet: TimeBlock(1, 16) has **3,008** parameters by the formula and by `count_params`; TimeBlock(16, 32) **14,944**; TimeBlock(32, 64) **57,536**; TimeBlock(64, 32) **28,768**; TimeBlock(32, 16) **7,472**. A (4, 16, 8, 8) batch goes through TimeBlock(16, 32) and gives (4, 32, 8, 8), with and without the time input.
#
# *What it means.* The formula and the code agree, so you know where every number sits. For TimeBlock(1, 16): conv 1 has 9 × 1 × 16 + 16 = 160, the projection 32 × 16 + 16 = 528, conv 2 has 9 × 16 × 16 + 16 = 2,320, and 160 + 528 + 2,320 = 3,008. The time projection is the small part: in TimeBlock(32, 64) it costs 2,112 of 57,536 parameters (3.7 percent).
#
# Tips and common mistakes:
#
# - `proj(emb)` has shape (B, c_out); `[:, :, None, None]` makes it (B, c_out, 1, 1) so that it is added to every pixel. Without the two `None` axes PyTorch tries to add a (B, c_out) tensor to a (B, c_out, H, W) tensor and complains (or silently adds it to the wrong axis if H or W equals B).
# - Use `padding=1` with the 3 × 3 convolutions to keep the size.

# %% [markdown]
# ### Step 5.4. The UNet by hand
#
# Now we assemble the blocks. The picture goes through three levels. **Down path:** a TimeBlock at 16 × 16 (16 channels), a 3 × 3 convolution with stride 2 that halves the size (8 × 8), a TimeBlock with 32 channels, another stride-2 convolution (4 × 4). **Bottleneck:** a TimeBlock with 64 channels at 4 × 4. **Up path:** a 2 × 2 transpose convolution with stride 2 (Week 12) doubles the size to 8 × 8, the result is **concatenated** with the 32-channel features saved on the way down (the skip connection; channels 32 + 32 = 64) and a TimeBlock brings it back to 32 channels; the same again for 16 × 16 with 16 + 16 = 32 channels in and 16 out. A final 1 × 1 convolution makes the output picture with one channel, the guess ε̂ of the noise. The step number goes through the sinusoidal embedding (size 32) and a small two-layer MLP, and the result is given to every block. The two switches `use_time` and `use_skips` are for Step 6.7, and `n_classes` is for Step 6.8; with their default values you get the plain network.

# %%
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

# %%
torch.manual_seed(SEED)
unet0 = TinyUNet()
xb_, tb_ = torch.randn(8, 1, 16, 16), torch.randint(1, 101, (8,))
trace = []; yb_ = unet0(xb_, tb_, trace=trace)
for name, v in trace: print("%-11s -> %s" % (name, tuple(v.shape)))
print("output      ->", tuple(yb_.shape))
print("parameters  :", count_params(unet0))
assert yb_.shape == xb_.shape and count_params(unet0) == 135713
RESULTS["p5_unet_params"] = count_params(unet0)

# %% [markdown]
# ### ✏️ Your turn: Shapes by thought
#
# *Exercise 5.4:* Before looking at the trace, predict the shape (channels, height, width) of the features after each stage, for one picture, and fill in the dictionary. For example the input is (1, 16, 16). Why it matters: being able to predict shapes is how you find bugs in a UNet (a wrong size breaks the skip concatenation). The check compares each of your entries with the real trace.

# %%
shape_guess = {"down1": None, "down2": None, "bottleneck": None, "up2": None, "up1": None}    # YOUR CODE: replace every None by a tuple (channels, height, width)

# %%
if all(v is None for v in shape_guess.values()):
    print("Fill in shape_guess above, then run this cell again.")
else:
    for name, v in trace:
        if name in shape_guess: check("%-10s %s" % (name, shape_guess[name]), shape_guess[name] == tuple(v.shape[1:]), "the real shape is %s: channels from the table in the step text, size 16, 8 or 4" % (tuple(v.shape[1:]),))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# shape_guess = {"down1": (16, 16, 16),       # first block at full size, 16 channels
#                "down2": (32, 8, 8),         # after the stride-2 convolution: half the size; the block doubles the channels
#                "bottleneck": (64, 4, 4),    # half the size again, 64 channels
#                "up2": (32, 8, 8),           # transpose convolution doubles the size; the block brings 64 channels (32 + 32 skip) back to 32
#                "up1": (16, 16, 16)}         # back at full size with 16 channels, then the 1 x 1 convolution makes 1 channel
# for name, v in trace:
#     if name in shape_guess: assert shape_guess[name] == tuple(v.shape[1:])
# print("all five shapes are right")
# ```
#
# The size is halved twice on the way down (16, 8, 4) and doubled twice on the way up, so the skip features have the same height and width as the up-path features that they are glued to; only the channels differ (and double after the concatenation).
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
shape_guess = {"down1": (16, 16, 16),       # first block at full size, 16 channels
               "down2": (32, 8, 8),         # after the stride-2 convolution: half the size; the block doubles the channels
               "bottleneck": (64, 4, 4),    # half the size again, 64 channels
               "up2": (32, 8, 8),           # transpose convolution doubles the size; the block brings 64 channels (32 + 32 skip) back to 32
               "up1": (16, 16, 16)}         # back at full size with 16 channels, then the 1 x 1 convolution makes 1 channel
for name, v in trace:
    if name in shape_guess: assert shape_guess[name] == tuple(v.shape[1:])
print("all five shapes are right")

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The trace prints `input -> (8, 1, 16, 16)`, `down1 -> (8, 16, 16, 16)`, `down2 -> (8, 32, 8, 8)`, `bottleneck -> (8, 64, 4, 4)`, `up2 -> (8, 32, 8, 8)`, `up1 -> (8, 16, 16, 16)` and `output -> (8, 1, 16, 16)`, and the total of **135,713** parameters (the assertion passes). Your five Check lines show a tick for each stage; the answer prints `all five shapes are right`.
#
# *What it means.* The output has exactly the shape of the input: the network maps a noisy picture to a noise picture of the same size, which is what the loss needs. The channels double on the way down (16, 32, 64) while the size halves, so each level holds fewer positions but richer features; the up path reverses this. The 135,713 parameters are the number in the lesson's `unet` figure: the blocks plus the strided convolutions, the transpose convolutions, the time MLP and the 1 × 1 output convolution.
#
# Tips and common mistakes:
#
# - If the input size is not divisible by 4 (for example 18 × 18) the up path is larger than the skip features and `torch.cat` fails; 16 is a good size.
# - The skip is concatenated **after** the transpose convolution: 32 channels from the up path plus 32 from the skip make the 64 inputs of `dec2`.

# %% [markdown]
# ### Step 5.5. Parameters by part
#
# Where are the 135,713 numbers? We loop over the child modules of the UNet and count each. The lesson's figure lists the blocks only; here we also list the strided convolutions, the transpose convolutions and the output convolution, which the figure leaves out (they make up the remaining 21,873 numbers). The bottleneck is the biggest single part, though it works on the smallest picture: its 64 channels at 4 × 4 are cheap to compute but need many weights (9 × 32 × 64 and 9 × 64 × 64 in the two convolutions).

# %%
tot = count_params(unet0)
print("%-10s %9s %7s" % ("part", "params", "share"))
for name, mod in unet0.named_children():
    n = count_params(mod)
    print("%-10s %9d %6.1f%%" % (name, n, 100 * n / tot))
print("%-10s %9d %6.1f%%" % ("total", tot, 100))
print("the five blocks together:", sum(count_params(getattr(unet0, k)) for k in ("down1", "down2", "mid", "dec2", "dec1")), "  (3,008 + 14,944 + 57,536 + 28,768 + 7,472)")

# %% [markdown]
# ### ✏️ Your turn: The bottleneck's share
#
# *Exercise 5.5:* Compute `share_mid`, the share of the parameters held by the bottleneck block (`unet0.mid`), as a number between 0 and 1, and `share_blocks`, the share held by all five TimeBlocks. Why it matters: it shows where a UNet's capacity sits: in the deepest, narrowest layers, which see the whole picture at low resolution.

# %%
share_mid = None; share_blocks = None    # YOUR CODE: count_params(...) of the part divided by count_params(unet0)

# %%
if share_mid is None or share_blocks is None:
    print("Fill in share_mid and share_blocks above, then run this cell again.")
else:
    check("share_mid is 57,536 / 135,713 = 0.424", abs(share_mid - 57536 / 135713) < 0.002, "count_params(unet0.mid) / count_params(unet0)")
    check("share_blocks is 111,728 / 135,713 = 0.823", abs(share_blocks - 111728 / 135713) < 0.002, "add down1, down2, mid, dec2 and dec1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# share_mid = count_params(unet0.mid) / count_params(unet0)
# share_blocks = sum(count_params(getattr(unet0, k)) for k in ("down1", "down2", "mid", "dec2", "dec1")) / count_params(unet0)
# print("bottleneck %.3f   all five blocks %.3f" % (share_mid, share_blocks))
# RESULTS["p5_share_mid"] = round(share_mid, 3)
# ```
#
# The bottleneck holds 42 percent of all numbers although it handles only 4 × 4 positions: a convolution's cost in weights depends on the channels (9 × c_in × c_out), and the deepest layer has the most channels. The five blocks hold 82 percent; the rest are the strided and transpose convolutions, the time MLP and the output layer.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
share_mid = count_params(unet0.mid) / count_params(unet0)
share_blocks = sum(count_params(getattr(unet0, k)) for k in ("down1", "down2", "mid", "dec2", "dec1")) / count_params(unet0)
print("bottleneck %.3f   all five blocks %.3f" % (share_mid, share_blocks))
RESULTS["p5_share_mid"] = round(share_mid, 3)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The table lists: time_mlp 2,112 (1.6%), down1 3,008 (2.2%), pool1 2,320 (1.7%), down2 14,944 (11.0%), pool2 9,248 (6.8%), **mid 57,536 (42.4%)**, up2 8,224 (6.1%), dec2 28,768 (21.2%), up1 2,064 (1.5%), dec1 7,472 (5.5%), out 17 (0.0%), total 135,713. The five blocks together have 111,728 parameters. The Check passes two ticks (0.424 and 0.823); the answer prints `bottleneck 0.424   all five blocks 0.823`.
#
# *What it means.* The bottleneck block has more than twice as many parameters as the next block (dec2, 28,768), although it works on the smallest picture. A convolution costs 9 × c_in × c_out weights regardless of the picture size, and the bottleneck has the most channels. Compute and weights are different things: the bottleneck uses few multiplications (16 positions) for its many weights, the first block uses many multiplications (256 positions) for few weights. Also notice the little output layer: 17 numbers (a 1 × 1 convolution from 16 channels to 1 channel).

# %% [markdown]
# ### Step 5.6. Skip connections
#
# Why does the UNet carry features across (the "U")? On the way down the picture is squeezed to 4 × 4 positions; fine detail (where exactly a one-pixel line runs) is hard to carry through such a bottleneck. The skip hands the decoder the full-resolution features directly. A toy example shows the idea with the simplest possible "bottleneck": average the picture over 4 × 4 patches (leaving 16 numbers instead of 256) and expand it again. Then we test the real network: we replace the skip features by zeros in the **untrained** UNet and measure how much its output changes. (The same test on the **trained** network is Step 6.7; it tells you how much the trained network relies on the skips.)

# %% [markdown]
# ### ✏️ Your turn: A bottleneck without a skip
#
# *Exercise 5.6:* Write `pool_up(img, k=4)`: average the picture over k × k patches with `F.avg_pool2d` and expand it again to the original size with `F.interpolate(..., scale_factor=k, mode="nearest")`. Why it matters: after this operation the picture can hold only (16 / k)² different values: that is what a decoder without a skip would have to work from if the bottleneck were one channel. The check uses a cross: pool_up(cross) must differ from the cross, and have at most 16 distinct values.

# %%
def pool_up_mine(img, k=4):
    """Average over k x k patches, then expand back to the original size (nearest neighbour). img: tensor (B, C, H, W)."""
    return img   # YOUR CODE: F.interpolate(F.avg_pool2d(img, k), scale_factor=k, mode="nearest")
Xc, yc = make_shapes(5, noise=0.0, seed=0)
cross_img = torch.tensor(Xc[np.where(yc == 2)[0][:1]])      # one clean cross, shape (1, 1, 16, 16)

# %%
r_ = pool_up_mine(cross_img, 4)
check("the output has the shape of the input", tuple(r_.shape) == tuple(cross_img.shape), "(1, 1, 16, 16)")
check("the cross is changed (a thin line does not survive averaging)", float((r_ - cross_img).abs().mean()) > 0.05, "this function must average and expand; right now it returns the picture itself")
check("at most 16 distinct values", len(torch.unique(r_)) <= 16, "a 4 x 4 grid of patches holds 16 numbers")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def pool_up(img, k=4):
#     return F.interpolate(F.avg_pool2d(img, k), scale_factor=k, mode="nearest")
# r = pool_up(cross_img, 4)
# print("distinct values after pooling: %d (of %d pixels); mean absolute error: %.3f" % (len(torch.unique(r)), r.numel(), float((r - cross_img).abs().mean())))
# ```
#
# Averaging smears a one-pixel line over its whole 4 × 4 patch. A skip connection hands over the original pixels, so the error of this toy goes to zero. In the real UNet the bottleneck has 64 channels (1,024 numbers, more than the 256 pixels), so the number of values is not the only limit, but the features there are built from large patches and do not know exactly where things are.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def pool_up(img, k=4):
    return F.interpolate(F.avg_pool2d(img, k), scale_factor=k, mode="nearest")
r = pool_up(cross_img, 4)
print("distinct values after pooling: %d (of %d pixels); mean absolute error: %.3f" % (len(torch.unique(r)), r.numel(), float((r - cross_img).abs().mean())))

# %% [markdown]
# The untrained UNet: how much does its output change when the skips are zeroed? We feed the same batch and the same step numbers through the network with and without skips (the switch `use_skips` only changes what is concatenated; the weights are the same).

# %%
r_toy = pool_up(cross_img, 4)
print("toy: distinct values after pooling %d, mean absolute error %.3f" % (len(torch.unique(r_toy)), float((r_toy - cross_img).abs().mean())))
torch.manual_seed(SEED); u_skip = TinyUNet(use_skips=True)
u_zero = TinyUNet(use_skips=False); u_zero.load_state_dict(u_skip.state_dict())       # the same weights
with torch.no_grad():
    o1, o2 = u_skip(xb_, tb_), u_zero(xb_, tb_)
rel = float((o1 - o2).abs().mean() / o1.abs().mean())
print("untrained network: mean |output| %.4f, mean |change| %.4f, relative change %.2f" % (float(o1.abs().mean()), float((o1 - o2).abs().mean()), rel))
RESULTS["p5_skip_rel_change_untrained"] = round(rel, 2)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The Check shows three lines for the toy; with your function filled in they are ticks. The answer prints `distinct values after pooling: 4 (of 256 pixels); mean absolute error: 0.092`: the cross touches only a few of the 16 patches, so after pooling the picture has just 4 different values, and the lines have become faint grey squares. (The error is small on average because most pixels are background, but the lines, which are the only content, are destroyed.) For the untrained UNet the output changes only a little when its skips are zeroed: mean |output| 0.1950, mean |change| 0.0035, relative change **0.02**.
#
# *What it means.* Two different messages. The toy shows what a skip connection is **for**: the squeezed picture has lost where exactly the line ran. The second number looks as if the skips did not matter, but that is an artefact of the untrained network: with random weights the decoder blocks produce small outputs whatever they receive, and the skip is only one of two inputs, so the change is 2 percent. The question "does a trained network rely on its skips?" can only be asked of a trained network: Step 6.7 does it, and the answer is a relative change of 1.06 to 1.13: without its skips the trained network's output changes by as much as the output itself.

# %% [markdown]
# ### Step 5.7. Receptive field
#
# How much of the picture can one output neuron see? The **receptive field** of a stack of convolutions grows with every layer: each layer with kernel k adds (k − 1) times the current **jump** (the distance in input pixels between neighbouring outputs), and a stride s multiplies the jump by s. Start with receptive field 1 and jump 1. Three 3 × 3 convolutions at full resolution give 1 + 2 + 2 + 2 = 7. Strides make it grow much faster: that is why a UNet sees the whole 16 × 16 picture in its bottleneck.

# %% [markdown]
# ### ✏️ Your turn: The receptive field
#
# *Exercise 5.7:* Write `receptive_field(layers)` for a list of (kernel, stride) pairs: keep `rf` (start 1) and `jump` (start 1); for each layer do `rf += (k - 1) * jump` and then `jump *= s`; return `rf`. Why it matters: it tells you whether the network can possibly use context far away, which the denoiser needs (a line's end tells you where the line is). The check uses three stacks of layers whose receptive fields you can compute by hand.

# %%
def receptive_field_mine(layers):
    """layers: list of (kernel_size, stride). Returns the receptive field in input pixels."""
    return None   # YOUR CODE: rf = 1, jump = 1; for k, s in layers: rf += (k - 1) * jump; jump *= s

# %%
if receptive_field_mine([(3, 1)]) is None:
    print("Fill in receptive_field_mine above, then run this cell again.")
else:
    check("one 3x3 convolution: 3", receptive_field_mine([(3, 1)]) == 3, "rf = 1 + (3 - 1) * 1")
    check("three 3x3 convolutions at full resolution: 7", receptive_field_mine([(3, 1)] * 3) == 7, "1 + 2 + 2 + 2")
    check("3x3, then 3x3 with stride 2, then 3x3: 1 + 2 + 2 + 2 * 2 = 9", receptive_field_mine([(3, 1), (3, 2), (3, 1)]) == 9, "the jump doubles after the stride-2 layer, so the next 3x3 adds 2 * 2")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def receptive_field(layers):
#     rf, jump = 1, 1
#     for k, s in layers:
#         rf += (k - 1) * jump        # a kernel of size k sees k - 1 more positions, each `jump` input pixels apart
#         jump *= s                   # a stride s makes the outputs s times further apart
#     return rf
# print(receptive_field([(3, 1)] * 3), receptive_field([(3, 1), (3, 2), (3, 1)]))
# ```
#
# The jump is the key: after a stride-2 layer neighbouring neurons are two input pixels apart, so the next 3 × 3 convolution reaches twice as far.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def receptive_field(layers):
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump        # a kernel of size k sees k - 1 more positions, each `jump` input pixels apart
        jump *= s                   # a stride s makes the outputs s times further apart
    return rf
print(receptive_field([(3, 1)] * 3), receptive_field([(3, 1), (3, 2), (3, 1)]))

# %% [markdown]
# Now the encoder path of our UNet, in two versions: the five-layer sketch (3 × 3, stride 2, 3 × 3, stride 2, 3 × 3) and the real one, where every block has two convolutions: down1 (two convolutions), the stride-2 convolution, down2 (two), the second stride-2 convolution and the bottleneck block (two).

# %%
sketch = [(3, 1), (3, 2), (3, 1), (3, 2), (3, 1)]
real = [(3, 1), (3, 1), (3, 2), (3, 1), (3, 1), (3, 2), (3, 1), (3, 1)]
for name, lay in (("three 3x3 at full resolution", [(3, 1)] * 3), ("the five-layer sketch", sketch), ("the real encoder of TinyUNet", real)):
    print("%-30s receptive field %2d of 16 pixels" % (name, receptive_field(lay)))
RESULTS["p5_rf_sketch"] = receptive_field(sketch); RESULTS["p5_rf_real"] = receptive_field(real)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes three ticks (3, 7 and 9 for the three stacks), and the answer prints `7 9`. For the encoder of our UNet: three 3 × 3 convolutions at full resolution have a receptive field of **7**; the five-layer sketch (3 × 3, stride 2, 3 × 3, stride 2, 3 × 3) has **21**; the real encoder with two convolutions per block has **35** pixels, against a picture that is only 16 pixels wide.
#
# *What it means.* A neuron in the bottleneck can see more than the whole picture (the borders add a zero padding that it also sees), so the bottleneck can use the full picture as context: the position and kind of the shape. At full resolution the receptive field is 5 for the two convolutions of `down1`: that part of the network sees only a small neighbourhood, which is right for "is there a line pixel here?". The skip connections let the decoder combine both: a wide context from the bottleneck and the exact positions from the encoder's first blocks.

# %% [markdown]
# ## Part 6. A DDPM on tiny pictures (35 min)
#
# *Lesson sections: "Training", "Sampling" and "UNet".* Now everything comes together. ==You will train the UNet of Part 5 from scratch on 2,000 noisy line shapes (16 × 16 pixels), generate pictures with the DDPM loop of Part 4, judge them with a classifier (the Week 12 idea), generate faster with the deterministic DDIM sampler, follow how the pictures improve during training, break the model on purpose in four ways, and, as an option, steer it with classifier-free guidance.== A UNet with 135,713 parameters learns to draw five kinds of shapes in about a minute and a half on a CPU. The training run of Step 6.2 is reused by Steps 6.3 to 6.7 (and Part 5.6 asked for its skip test here). All numbers are for `SEED = 42`, and the second seed is 1 for the key experiments. With `QUICK` (the test mode that the Setup cell can switch on) the training loops are much shorter and the pictures are poor; the text always describes the full run.

# %% [markdown]
# ### ✏️ Imagine first
#
# After 3,000 training steps the UNet draws shapes. But what does it draw after only 250 steps, when it has seen the equivalent of 8 passes over the data (250 × 64 pictures / 2,000 pictures)? Pick the best description: (a) pure noise, (b) smooth grey blurs without lines, (c) blurry strokes in the right places that are not yet clean shapes, (d) already clean shapes. Write the letter in the cell below. You compare it with Step 6.6.

# %%
# Imagine first: what do the pictures look like after only 250 training steps?
my_guess_250 = None   # "a", "b", "c" or "d"

# %% [markdown]
# ### Step 6.1. Data and schedule
#
# The training pictures are the five shapes of Week 4 on a 16 × 16 grid (`make_shapes`, 2,000 pictures with a little noise of 0.05, always at the centre). We bring them to the range [−1, 1] that the diffusion process expects (the noise has mean 0 and standard deviation 1, so the data should be on a matching scale): first clip to [0, 1] to remove the few pixels pushed outside by the noise, then 2·x − 1. The schedule is the one of Part 4 (T = 100, β from 0.001 to 0.2, ᾱ_100 = 2.0e-5), and `q_sample_t` already works for pictures: its `view` makes one ᾱ per picture. The cell noises one cross at six steps so that you can see what the network must undo.

# %%
Xs_raw, ys_raw = make_shapes(2000, noise=0.05, seed=0)
data_img = torch.tensor(np.clip(Xs_raw, 0, 1) * 2 - 1)               # pictures in [-1, 1], shape (2000, 1, 16, 16)
labels_img = torch.tensor(ys_raw)
print("pictures:", tuple(data_img.shape), "  range [%.2f, %.2f]  mean %.3f" % (float(data_img.min()), float(data_img.max()), float(data_img.mean())))
assert data_img.min() >= -1 and data_img.max() <= 1 and abs(S["abar"][S["T"]]) < 0.01          # range [-1, 1] and a schedule that ends in noise
cross_ = data_img[labels_img == 2][:1]
g_ = torch.Generator().manual_seed(0); eps_ = torch.randn(cross_.shape, generator=g_)
ts_show = [0, 5, 15, 30, 60, 100]
noised = torch.cat([q_sample_t(cross_, torch.tensor([t]), eps_, S) if t > 0 else cross_ for t in ts_show])
show_grid(noised.numpy(), titles=["t = %d, abar = %.3f" % (t, float(S["abar"][t])) for t in ts_show], cols=6)

# %% [markdown]
# ### ✏️ Your turn: Where does the picture vanish?
#
# *Exercise 6.1:* Find `t_half`, the smallest step t for which ᾱ_t is below 0.5 (the signal-to-noise ratio ᾱ/(1 − ᾱ) falls below 1: more noise than picture), and `t_gone`, the smallest t with ᾱ_t below 0.01 (almost nothing of the picture is left). Use `S["abar"]` (index 0 is the unnoised picture; look only at t from 1 to T). Why it matters: a schedule that leaves the picture mostly intact for half the steps spends the network's effort on fine detail; one that destroys it quickly spends it on coarse structure. The check compares with the table printed above.

# %%
t_half = None; t_gone = None    # YOUR CODE: for example int((S["abar"][1:] < 0.5).nonzero()[0]) + 1

# %%
if t_half is None or t_gone is None:
    print("Fill in t_half and t_gone above, then run this cell again.")
else:
    ab_ = S["abar"].numpy()
    check("t_half: abar_t is below 0.5 at t = %d but not at t = %d" % (t_half, t_half - 1), ab_[t_half] < 0.5 <= ab_[t_half - 1], "the first t with abar_t < 0.5")
    check("t_gone: abar_t is below 0.01 at t = %d but not at t = %d" % (t_gone, t_gone - 1), ab_[t_gone] < 0.01 <= ab_[t_gone - 1], "the first t with abar_t < 0.01")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# t_half = int((S["abar"][1:] < 0.5).nonzero()[0]) + 1       # the first position (counting from step 1) where abar falls below 0.5
# t_gone = int((S["abar"][1:] < 0.01).nonzero()[0]) + 1
# print("abar < 0.5 from t = %d (SNR below 1), abar < 0.01 from t = %d, of T = %d" % (t_half, t_gone, S["T"]))
# RESULTS["p6_t_half"] = t_half; RESULTS["p6_t_gone"] = t_gone
# ```
#
# `nonzero()[0]` is the first index where the condition holds; we cut off the entry t = 0 with `[1:]` and add 1 to turn the position back into a step number.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
t_half = int((S["abar"][1:] < 0.5).nonzero()[0]) + 1       # the first position (counting from step 1) where abar falls below 0.5
t_gone = int((S["abar"][1:] < 0.01).nonzero()[0]) + 1
print("abar < 0.5 from t = %d (SNR below 1), abar < 0.01 from t = %d, of T = %d" % (t_half, t_gone, S["T"]))
RESULTS["p6_t_half"] = t_half; RESULTS["p6_t_gone"] = t_gone

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The cell prints `pictures: (2000, 1, 16, 16)   range [-1.00, 1.00]  mean -0.846` (the mean is close to −1 because most pixels are background; a shape has 9 to 32 bright pixels out of 256). The six pictures of the cross show how the signal disappears: at t = 5 (ᾱ = 0.975) the cross is almost untouched, at t = 15 (ᾱ = 0.796) it is still clear but speckled, at t = 30 (ᾱ = 0.40) the lines are faint among the noise, at t = 60 (ᾱ = 0.02) you can only guess it, and at t = 100 it is pure noise. The check passes two ticks, and the answer prints `abar < 0.5 from t = 27 (SNR below 1), abar < 0.01 from t = 67, of T = 100`.
#
# *What it means.* The picture is "half gone" at t = 27 and practically gone at t = 67: about a third of our steps (68 to 100) are spent on pictures that contain essentially only noise. The network learns to place the shape coarsely between t = 100 and t = 60, to refine its edges around t = 30 and to clean up the background at the end. Compare this with Part 2: a cosine schedule would spread the work out more evenly.
#
# Tips and common mistakes:
#
# - `(S["abar"][1:] < 0.5).nonzero()[0]` is the first index in the shortened array: add 1 to get the step number, or the answer is off by one.

# %% [markdown]
# ### Step 6.2. The model and the training loop
#
# The model is the `TinyUNet` of Part 5 (135,713 parameters, no change needed for 16 × 16 pictures). The loop is the one of Part 4 with three differences: pictures instead of points, batch 64 instead of 256, and a learning rate that starts at 1e-3 and decays smoothly to 0 following a cosine curve over the run (`CosineAnnealingLR`), which makes the last steps gentle and the final weights less noisy. The function also keeps copies of the weights at chosen steps (Step 6.6) and has two switches for Step 6.7 (`use_time`, `use_skips`) and one for Step 6.8 (`labels`, `n_classes`, `p_drop`: with labels, a fraction `p_drop` of the labels is replaced by the "no label" value, which is how one network learns both a conditional and an unconditional guess). It prints the average loss every 250 steps. The full run is 3,000 steps and takes about 75 seconds on our CPU; `QUICK` shortens it to 200.

# %%
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

# %%
t0 = time.time()
unet42, losses_u42, saved_u42 = train_unet(seed=42, ckpt=CK_UNET)
time_unet = time.time() - t0
print("training took %.0f seconds for %d steps (%.0f ms per step)" % (time_unet, STEPS_UNET, 1000 * time_unet / STEPS_UNET))
RESULTS["p6_final_loss"] = round(float(np.mean(losses_u42[-100:])), 4); RESULTS["p6_train_seconds"] = round(time_unet)
plt.figure(figsize=(6, 3)); plt.plot(np.convolve(losses_u42, np.ones(25) / 25, mode="valid")); plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("noise-prediction loss"); plt.title("UNet training loss (average of 25 steps)"); plt.show()

# %% [markdown]
# ### ✏️ Your turn: Epochs and cost
#
# *Exercise 6.2:* Compute `epochs_seen`: how many times the training set was passed through (steps × batch size divided by the number of pictures), and `ms_per_step`: the training time per step in milliseconds (use `time_unet` and `STEPS_UNET`). Why it matters: diffusion models are trained for many epochs; knowing the cost per step lets you plan a run, and the epoch count shows that 2,000 pictures are seen over and over (the random noise is different every time, so it is not the same example twice).

# %%
epochs_seen = None; ms_per_step = None    # YOUR CODE: STEPS_UNET * 64 / len(data_img); 1000 * time_unet / STEPS_UNET

# %%
if epochs_seen is None or ms_per_step is None:
    print("Fill in epochs_seen and ms_per_step above, then run this cell again.")
else:
    check("epochs_seen = %d * 64 / 2000 = %.1f" % (STEPS_UNET, STEPS_UNET * 64 / 2000), abs(epochs_seen - STEPS_UNET * 64 / 2000) < 0.01, "steps times batch size divided by the number of pictures")
    check("ms_per_step is the training time per step", abs(ms_per_step - 1000 * time_unet / STEPS_UNET) < 1.0, "1000 * time_unet / STEPS_UNET")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# epochs_seen = STEPS_UNET * 64 / len(data_img)
# ms_per_step = 1000 * time_unet / STEPS_UNET
# print("%.0f passes over the 2,000 pictures; %.0f ms per step" % (epochs_seen, ms_per_step))
# ```
#
# 3,000 steps of 64 pictures are 96 passes over the data. Each pass shows every picture with different noise at a different step number, so the network does not memorise noisy pictures; it can still memorise the five clean shapes, which for this tiny data set is what we want it to know.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
epochs_seen = STEPS_UNET * 64 / len(data_img)
ms_per_step = 1000 * time_unet / STEPS_UNET
print("%.0f passes over the 2,000 pictures; %.0f ms per step" % (epochs_seen, ms_per_step))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The loss, averaged over 250 steps, falls from **0.2670** (steps 1 to 250) to 0.0642 (500), 0.0358 (1,000), 0.0309 (1,500), 0.0276 (2,000) and **0.0256** at step 3,000; the average of the last 100 steps is about **0.027** (tolerance: between 0.02 and 0.04). The training takes about **68 seconds**, which is about **23 ms per step** (on a slower CPU up to two minutes). The check passes two ticks; the answer prints `96 passes over the 2,000 pictures; 23 ms per step`.
#
# *What it means.* Compare the loss with that of the 2-D denoiser (0.25): the picture denoiser reaches a loss of 0.027, ten times lower, because most of the 256 pixels are plain background that is easy to predict at every noise level. After about 1,000 steps the loss has almost levelled off, but the pictures of Step 6.6 show that the last 2,000 steps still matter: the loss falls from 0.036 to 0.026, a small number with a visible effect (cleaner background). The training log is on a log scale in the plot, and the curve is noisy because every step has random t.
#
# Tips and common mistakes:
#
# - `CosineAnnealingLR(opt, steps)` needs the **total** number of steps, and `sched.step()` once per step, not once per epoch.

# %% [markdown]
# ### Step 6.3. DDPM sampling: pictures from noise
#
# Sampling is exactly the loop of Part 4, with the UNet as `eps_fn` and the shape (n, 1, 16, 16). It needs 100 network calls, one per step. We draw 32 pictures and show them, and then a "denoising film": the same 8 pictures at t = 100, 75, 50, 25 and 0, so that you can watch noise turn into shapes. The network works on batches, so 32 pictures cost the same number of calls as 1. Before judging pictures, we clip them to the range [−1, 1] and map them to [0, 1]; you write that small function first.

# %% [markdown]
# ### ✏️ Your turn: From [-1, 1] to [0, 1]
#
# *Exercise 6.3:* Write `to_unit(x)`: it takes a picture tensor in [−1, 1] (the output of the sampler may go a little outside) and returns `(x.clamp(-1, 1) + 1) / 2`, in [0, 1]. Why it matters: the judge of Step 6.4 and the human eye both expect pixel values between 0 and 1; a diffusion sampler is not forced to stay in range, and values outside are cut off rather than allowed to wrap. The check tests three values.

# %%
def to_unit_mine(x):
    return None   # YOUR CODE: clip to [-1, 1], then shift and scale to [0, 1]

# %%
r_ = to_unit_mine(torch.tensor([-1.7, -1.0, 0.0, 1.0, 1.4]))
if r_ is None:
    print("Fill in to_unit_mine above, then run this cell again.")
else:
    check("-1 -> 0, 0 -> 0.5, 1 -> 1", torch.allclose(r_[1:4], torch.tensor([0.0, 0.5, 1.0])), "(x + 1) / 2")
    check("values outside [-1, 1] are clipped: -1.7 -> 0 and 1.4 -> 1", float(r_[0]) == 0.0 and float(r_[4]) == 1.0, "use x.clamp(-1, 1) first")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def to_unit(x):
#     return (x.clamp(-1, 1) + 1) / 2
# print(to_unit(torch.tensor([-1.7, -1.0, 0.0, 1.0, 1.4])))
# ```
#
# The `clamp` is applied first, so a value of 1.4 becomes 1.0 and not 1.2.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def to_unit(x):
    return (x.clamp(-1, 1) + 1) / 2
print(to_unit(torch.tensor([-1.7, -1.0, 0.0, 1.0, 1.4])))

# %%
t0 = time.time()
samples, film = sample_ddpm(lambda x, t: unet42(x, t), (32, 1, 16, 16), S, seed=1, keep={100, 75, 50, 25, 0})
print("DDPM sampling of 32 pictures: %.1f seconds, %d network calls" % (time.time() - t0, S["T"]))
print("range of the final pictures: [%.2f, %.2f]" % (float(samples.min()), float(samples.max())))
show_grid(to_unit(samples).numpy(), cols=8)
strip = torch.cat([to_unit(film[t][:8]) for t in (100, 75, 50, 25, 0)])            # the first 8 pictures at five moments
show_grid(strip.numpy(), titles=["t=100"] * 8 + ["t=75"] * 8 + ["t=50"] * 8 + ["t=25"] * 8 + ["t=0"] * 8, cols=8)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The cell prints `DDPM sampling of 32 pictures: 0.5 seconds, 100 network calls` (times differ) and `range of the final pictures: [-1.31, 1.18]`: the sampler goes a little outside [−1, 1], which is why we clip before judging. The check passes two ticks (the answer prints `tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000])`). In the grid of 32 you see clean crosses, squares, bars and diagonals with a faintly speckled background; every one of the five shapes appears, none is a blend of two shapes. The denoising film (rows t = 100, 75, 50, 25, 0): the first three rows look like pure noise, in the row t = 25 faint outlines (a square's corner, the arm of a cross) show through the noise, and the last row shows the finished shapes.
#
# *What it means.* The shape is **decided late**: nothing is recognisable at t = 50, yet at t = 25 the outlines are there and at t = 0 the picture is clean. The network does not draw the picture top to bottom; it refines the whole picture at once, first the rough layout and then the details, each step a small correction. Notice also that the model makes new noise-free pictures that it has not copied from the data: the pictures are not exact copies of the training set (which has noise of 0.05 and a different noise pattern in every picture) but samples from what the model has learned.

# %% [markdown]
# ### Step 6.4. The judge
#
# Looking at 32 pictures is not enough, and we need a number that does not depend on our eyes. We use the idea from Week 12: a **judge**, a small classifier that has been trained on clean pictures of the five shapes and is then frozen. For a batch of generated pictures it gives the *confidence* (the mean of its highest probability), the *entropy* of the histogram of predicted classes (maximum ln 5 = 1.609 if all five classes occur equally often) and the *coverage* (how many classes have at least 5 percent of the pictures). We add a fourth number for the pictures' cleanliness, which the classifier cannot see: the **template error**, the mean squared difference (in [0, 1] units) between a picture and the nearest of the five clean, noise-free shapes. Good pictures give a high confidence, an entropy near 1.609, a coverage of 5 and a small template error. Remember the two cautions of Week 12: the judge is confident on noise too, so confidence alone does not say a picture is good; and it tells you nothing about how new the pictures are. First the judge and the two helper functions are trained and defined (this takes a few seconds).

# %%
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

# %% [markdown]
# ### ✏️ Your turn: The judge function
#
# *Exercise 6.4:* Write `judge_mine(images, unit=False)` (the template error is an extra in our version: you may leave it out). It receives a tensor or array of pictures with shape (N, 1, 16, 16), in [−1, 1] (the range of generated pictures; with `unit=True` they are already in [0, 1]), and returns a dictionary with `"pred"` (predicted classes), `"conf"` (the mean of the highest softmax probability), `"hist"` (the share of each of the five classes among the predictions), `"entropy"` and `"coverage"`. Use `judge_net`, `entropy_nats` and `coverage`, switch off the gradient with `torch.no_grad()`, and map to [0, 1] first (and clip) when `unit` is False. Why it matters: every experiment from here to Part 7 is scored with this function.

# %%
def judge_mine(images, unit=False):
    """Score a batch of pictures with the frozen judge. Returns a dict with pred, conf, hist, entropy, coverage."""
    return None   # YOUR CODE: x = to_unit(...) unless unit; probabilities = softmax(judge_net(x)); conf, pred = probabilities.max(1); hist = np.bincount(pred, minlength=5) / N

# %%
Xr_, yr_ = make_shapes(300, noise=0.05, seed=5)
r_ = judge_mine(np.clip(Xr_, 0, 1) * 2 - 1)
if r_ is None:
    print("Fill in judge_mine above, then run this cell again.")
else:
    check("the confidence on real pictures is above 0.95", r_["conf"] > 0.95, "judge_net(x), then softmax over dimension 1, then the maximum; the pictures must be in [0, 1]")
    check("the histogram has five shares that add up to 1", len(r_["hist"]) == 5 and abs(float(np.sum(r_["hist"])) - 1) < 1e-6, "np.bincount(pred, minlength=5) / len(pred)")
    check("real pictures cover all 5 classes with entropy near 1.609", r_["coverage"] == 5 and r_["entropy"] > 1.55, "the classes are balanced in the data")
    check("the predictions agree with the labels (accuracy above 0.95)", float((r_["pred"] == yr_).mean()) > 0.95, "the judge was trained on these five shapes")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def judge(images, unit=False):
#     x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
#     x = x.clamp(0, 1) if unit else to_unit(x)                        # back to [0, 1], the range the judge was trained on
#     with torch.no_grad():
#         pr = F.softmax(judge_net(x), 1)
#     conf, pred = pr.max(1)
#     hist = np.bincount(pred.numpy(), minlength=5) / len(pred)
#     tmpl = float(((x[:, None] - templates[None]) ** 2).flatten(2).mean(2).min(1).values.mean())   # squared error to the nearest clean template: how clean are the pictures?
#     return {"pred": pred.numpy(), "conf": float(conf.mean()), "hist": hist, "entropy": entropy_nats(hist), "coverage": coverage(hist), "tmpl": tmpl}
# print("judge ready")
# ```
#
# The judge sees pictures in [0, 1], like the ones it was trained on. We compute the histogram of predicted classes and read two numbers from it: the entropy (how evenly the classes occur) and the coverage (how many classes occur at all).
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
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

# %% [markdown]
# Now judge three kinds of pictures: 500 real pictures, 500 pictures of pure noise (the same noise the sampler starts from, clipped to the range) and 500 pictures generated by the DDPM sampler with the UNet. Compare them. Sampling 500 pictures takes about 8 seconds.

# %%
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

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The judge reaches a train and test accuracy of **1.000** (tolerance: test accuracy above 0.95; the five shapes are easy), the check passes four ticks for your function and the answer prints `judge ready`. The three rows (500 pictures each, seed 42): **real pictures**: confidence 1.000, entropy 1.609, coverage 5 of 5, template error 0.0013, histogram 0.2 for every class. **Pure noise**: confidence **0.881**, entropy 0.549, coverage 2 of 5, template error 0.3669, and 76 percent of the noise pictures are called the same class. **DDPM, 100 steps**: confidence **1.000**, entropy **1.601**, coverage **5 of 5**, template error 0.0014, histogram 0.23, 0.21, 0.16, 0.18, 0.22. Sampling 500 pictures takes about 6 seconds; the judge's accuracy on the real pictures is 1.000. Tolerance for the generated row: confidence above 0.95, entropy above 1.45, coverage 5.
#
# *What it means.* Two warnings from Week 12 show again. **Confidence alone says little**: the noise gets a confidence of 0.881 (the classifier must choose a class), what gives it away is the entropy (0.549, a spike on one class) and the template error (0.367, 280 times that of real pictures). **The generated pictures look like the real ones in all four numbers**: every class appears (coverage 5), the entropy is 1.601 of the maximum 1.609, and the template error is 0.0014, as low as real data (0.0013). The histogram shows a slightly lower share of one class (0.16 against 0.20); with 500 pictures a share has a sampling error of about 0.018, so this is a mild preference at most. The judge cannot tell you that the pictures are new rather than copies, and it does not see small defects: that is what the template error and your eyes are for. Compare with the DCGAN of Week 12 (confidence about 0.97, entropy 1.61, coverage 5, after much more tuning): the same judge numbers, reached without a second network.

# %% [markdown]
# ### Step 6.5. Faster sampling with DDIM
#
# The DDPM loop needs 100 network calls, and for large models this is the main cost of diffusion (1,000 calls for the original DDPM). **DDIM** (denoising diffusion implicit models) is a sampler for the *same* trained network that uses a shorter list of steps, for example 100 → 50, 20, 10 or 5, and takes no fresh noise (η = 0, deterministic: the same starting noise always gives the same picture). In each step it first estimates the clean picture, x̂0 = (x_t − √(1 − ᾱ_t)·ε̂)/√ᾱ_t (clipped to [−1, 1]), and then moves to the earlier step t′ by keeping that estimate and the predicted noise: x_t′ = √ᾱ_t′·x̂0 + √(1 − ᾱ_t′)·ε̂. The steps can jump (100, 95, 90 … or 100, 80, 60 …).

# %% [markdown]
# ### ✏️ Your turn: One DDIM step
#
# *Exercise 6.5:* Write `ddim_step_mine(x, e, t, t_prev, S)` that implements the two formulas above (clip x̂0 to [−1, 1] before the second formula). The check uses a neat property: if `e` is the *true* noise that made x_t, then x̂0 is exactly the clean picture and the step must land exactly on the closed-form x_t′ for the same noise: `q_sample_t(x0, t_prev, e)`. Why it matters: it shows that DDIM is consistent with the forward process, and it lets you test your formula without any network.

# %%
def ddim_step_mine(x, e, t, t_prev, S):
    """From x_t to x_(t_prev) with the deterministic DDIM rule. t and t_prev are plain integers, t_prev < t."""
    return None   # YOUR CODE: x0_hat = ((x - sqrt(1 - abar_t) * e) / sqrt(abar_t)).clamp(-1, 1); return sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev) * e

# %%
x0_c = data_img[:4]; e_c = torch.randn(x0_c.shape); t_c, tp_c = 60, 40
x_c = q_sample_t(x0_c, torch.full((4,), t_c), e_c, S)
r_ = ddim_step_mine(x_c, e_c, t_c, tp_c, S)
if r_ is None:
    print("Fill in ddim_step_mine above, then run this cell again.")
else:
    check("with the true noise the DDIM step lands on the closed form at t' = 40", torch.allclose(r_, q_sample_t(x0_c, torch.full((4,), tp_c), e_c, S), atol=1e-4), "x0_hat is then the clean picture; x_prev = sqrt(abar_prev) * x0_hat + sqrt(1 - abar_prev) * e")
    check("the last step (t' = 0) returns the estimate of the clean picture", torch.allclose(ddim_step_mine(x_c, e_c, t_c, 0, S), x0_c.clamp(-1, 1), atol=1e-4), "abar_0 = 1, so x_0 = x0_hat")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def ddim_step(x, e, t, t_prev, S):
#     ab, ab_prev = S["abar"][t], S["abar"][t_prev]
#     x0_hat = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)       # the network's estimate of the clean picture
#     return ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * e           # re-noise that estimate to the earlier level, with the same noise guess
# ```
#
# The step does not add fresh noise, so the whole path is determined by the starting noise. At the last step t_prev = 0 and ᾱ_0 = 1, so the output is the estimate x̂0 itself.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def ddim_step(x, e, t, t_prev, S):
    ab, ab_prev = S["abar"][t], S["abar"][t_prev]
    x0_hat = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)       # the network's estimate of the clean picture
    return ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * e           # re-noise that estimate to the earlier level, with the same noise guess

# %% [markdown]
# Now the sampler: it builds the list of steps with `np.linspace(T, 0, steps + 1)` (for 20 steps: 100, 95, 90, …, 0), calls the network once per step and applies `ddim_step`. We judge 500 pictures for each number of steps with the same starting noise (seed 0), measure the time, and plot the judge's numbers against the number of network calls. The DDPM row is the 100-step run from Step 6.4.

# %%
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

# %% [markdown]
# Look at the pictures at 20, 5, 2 and 1 steps with the same starting noise: where does the quality go?

# %%
fig_rows = []
for K in (20, 5, 2, 1):
    xg = sample_ddim(lambda x, t: unet42(x, t), (8, 1, 16, 16), S, steps=K, seed=1); fig_rows.append(to_unit(xg))
show_grid(torch.cat(fig_rows).numpy(), titles=["%d steps" % K for K in (20, 5, 2, 1) for _ in range(8)], cols=8)
print("mean absolute difference between the 20-step and the 1-step picture: %.3f" % float((fig_rows[0] - fig_rows[3]).abs().mean()))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes two ticks. The table for 500 pictures with the same starting noise (the times are rounded values from our CPU and differ from machine to machine; the other numbers should match within the stated tolerance):
#
# - DDPM, 100 calls: about 6 s, confidence 1.000, entropy 1.601, coverage 5, template error 0.0014
# - DDIM, 50 calls: about 3 s, 0.982, 1.523, 5, 0.0228
# - DDIM, 20 calls: about 1 s, 0.997, 1.576, 5, 0.0023
# - DDIM, 10 calls: about 0.6 s, 0.991, 1.575, 5, 0.0031
# - DDIM, 5 calls: about 0.3 s, 0.979, 1.581, 5, 0.0059
# - DDIM, 2 calls: about 0.1 s, 0.915, 1.594, 5, 0.0199
# - DDIM, 1 call: about 0.1 s, 0.899, 0.687, 2, 0.3930
#
# In the grid of 8 pictures: 20 steps gives clean shapes; 5 steps gives mostly clean shapes, with some pictures that show a faint second copy of a line ("ghost" strokes); 2 steps gives faint, overlaid shapes (a cross with a square across it, several bars at once); 1 step gives a black-and-white speckle pattern with no shape (the mean absolute difference from the 20-step picture is 0.445). The plot shows the confidence flat near 1 down to 5 calls and falling at 2 and 1 calls, while the entropy only falls at 1 call. Tolerance: 20 calls within 0.03 of the DDPM confidence; 1 call clearly broken (coverage below 4).
#
# *What it means.* DDIM with **20 calls is five times faster than DDPM with 100 and almost as good** in every number (confidence 0.997, template error 0.0023 against 0.0014); even 5 calls give recognisable shapes in the judge's eyes (0.979, coverage 5), though the template error has already quadrupled (0.0059) and ghost strokes appear. At 2 calls the picture is the network's best average guess of the clean picture, a blend of what the shapes might be, and the judge still sees five classes because blends are still mostly recognisable (0.915); at 1 call the sampler jumps directly from pure noise to a clipped estimate and fails (coverage 2). The result is not monotone: 50 calls has a worse template error (0.0228) than 20 calls (0.0023) in our run, because a few pictures ended up as blends; we have one starting-noise seed here, so do not over-read differences between 10 and 100 steps. The robust finding: **quality is flat from 100 down to about 10 calls, degrades at 5 and 2, and breaks at 1.**
#
# Tips and common mistakes:
#
# - The list of steps must start at T and end at 0, with the last step landing on t = 0 (ᾱ_0 = 1), otherwise the output is not a clean picture.
# - DDIM here is deterministic: the same seed always gives the same picture, in contrast to DDPM where every random z changes the result.

# %% [markdown]
# ### Step 6.6. Training progress
#
# Now the checkpoints of Step 6.2: after 250 and 1,000 steps and at the end. We sample 500 pictures from each (DDPM, the same starting noise), judge them and show 8 pictures of each. This is the film of learning. Compare with your Imagine-first letter.

# %%
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

# %% [markdown]
# ### ✏️ Your turn: Read the film
#
# *Exercise 6.6:* Store in `earliest_full_coverage` the earliest checkpoint step at which the judge finds all 5 classes (coverage 5), or `None`, and in `my_description_250` one sentence describing what the 250-step pictures look like (look at the first row of the grid). Why it matters: the judge measures *what* the pictures are, your eyes tell you *how* they look; both are needed (a blurry stroke can be judged "horizontal bar" with high confidence).

# %%
earliest_full_coverage = "unset"    # YOUR CODE: the first step in rows66 with coverage 5, or None
my_description_250 = ""             # YOUR TEXT: one sentence

# %%
if earliest_full_coverage == "unset" or my_description_250.strip() == "":
    print("Fill in both variables above, then run this cell again.")
else:
    want_ = next((st for st, j in rows66 if j["coverage"] == 5), None)
    check("earliest_full_coverage is %s" % want_, earliest_full_coverage == want_, "the first row of rows66 whose coverage is 5")
    check("your sentence has at least 6 words", len(my_description_250.split()) >= 6, "describe the strokes, their sharpness and the background")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# earliest_full_coverage = next((st for st, j in rows66 if j["coverage"] == 5), None)
# my_description_250 = "Scattered dark fragments of strokes around the centre on a speckled grey background; no complete shape yet."
# print(earliest_full_coverage, "|", my_description_250)
# ```
#
# Your sentence is yours, but it should say that there are fragments of strokes near the centre while the shapes are not complete and the background is speckled; the network learns *where* shapes go early and *how to complete and clean* them later.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
earliest_full_coverage = next((st for st, j in rows66 if j["coverage"] == 5), None)
my_description_250 = "Scattered dark fragments of strokes around the centre on a speckled grey background; no complete shape yet."
print(earliest_full_coverage, "|", my_description_250)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The table (500 pictures, DDPM, the same starting noise): after **250** steps confidence **0.880**, entropy 1.365, coverage **4**, template error 0.0433, histogram 0.28, 0.16, 0.04, 0.09, 0.42; after **1,000** steps confidence 0.997, entropy 1.545, coverage 5, template error 0.0031, histogram 0.34, 0.20, 0.14, 0.14, 0.18; after **3,000** steps confidence 1.000, entropy 1.601, coverage 5, template error 0.0014. In the grid the 250-step pictures are scattered dark fragments around the centre on a speckled background: pieces of strokes, here and there a short line, no complete shape. At 1,000 steps the shapes are clean and complete (a few have a slightly uneven line), at 3,000 steps the background is quieter. The Check passes two ticks; `earliest_full_coverage` is 1000.
#
# *What it means.* Compare with your Imagine-first letter. The 250-step network is **between (b) and (c)**: it has not produced smooth grey blurs; it draws fragments of strokes in roughly the right region but not complete shapes (so (c) comes closest, with "fragments" instead of "blurry strokes"). Its judge numbers are not bad (confidence 0.88, the judge always finds something) but the histogram is badly skewed (42 percent of the pictures are called one class, 4 percent another) and the template error is 30 times that of the final network. Between 250 and 1,000 steps the network learns **what the shapes are**; between 1,000 and 3,000 it learns **to draw them cleanly and in equal numbers** (entropy from 1.545 to 1.601, template error halved). The loss (0.0358 at 1,000 steps, 0.0256 at 3,000) hardly shows this.

# %% [markdown]
# ### Step 6.7. Break it on purpose
#
# A diffusion model has several parts that look small: the time input, the skip connections, the number of sampling steps and the factor 1/√α_t in the reverse step. We break each of them in turn and read the judge's numbers against the healthy baseline. There are four experiments: **(a)** no time input (the UNet with `use_time=False`, retrained: 75 seconds); **(b)** no skip connections (`use_skips=False`, the skip features are replaced by zeros, retrained: 75 seconds), plus the test of Step 5.6 on the healthy trained network: zero the skips only at sampling time; **(c)** too few sampling steps: DDIM with 5, 2 and 1 steps (Step 6.5); **(d)** a wrong sampler: the reverse step without the factor 1/√α_t (`fix_scale=False`). We run everything for seed 42 first and then for seed 1 (three more trainings). The training cells are separate so that each one stays under two minutes.

# %%
t0 = time.time(); unet_nt42, _, _ = train_unet(seed=42, use_time=False, log=0)
print("(a) no time input, seed 42: trained in %.0f seconds" % (time.time() - t0))

# %%
t0 = time.time(); unet_ns42, _, _ = train_unet(seed=42, use_skips=False, log=0)
print("(b) no skip connections, seed 42: trained in %.0f seconds" % (time.time() - t0))

# %% [markdown]
# The evaluation function: for one seed it needs the healthy network and the two broken ones, and returns the judge's numbers for the healthy baseline, (a), (b), the post-hoc zero-skip test, (c) with 5, 2 and 1 steps and (d). It also measures, for the healthy network, how much its output changes when the skips are set to zero at sampling time (the follow-up of Step 5.6) on 256 real pictures noised to t = 50.

# %%
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

# %%
t0 = time.time()
res67 = {}; rel67 = {}
res67[42], rel67[42] = evaluate_breakages(unet42, unet_nt42, unet_ns42)
print("evaluated in %.0f seconds" % (time.time() - t0))
print("%-34s confidence  entropy  coverage  template error" % "seed 42")
for name, j in res67[42].items(): print("%-34s %10.3f %8.3f %9d %15.4f" % (name, j["conf"], j["entropy"], j["coverage"], j["tmpl"]))
print("relative change of the healthy network's output when its skips are zeroed (t = 50): %.2f" % rel67[42])
RESULTS["p6_rel_skip_change"] = round(rel67[42], 2)

# %% [markdown]
# The second seed: three more trainings (the healthy network, the one without time, the one without skips), each about 75 seconds, in separate cells.

# %%
t0 = time.time(); unet1, _, _ = train_unet(seed=1, log=0)
print("healthy network, seed 1: trained in %.0f seconds" % (time.time() - t0))

# %%
t0 = time.time(); unet_nt1, _, _ = train_unet(seed=1, use_time=False, log=0)
print("(a) no time input, seed 1: trained in %.0f seconds" % (time.time() - t0))

# %%
t0 = time.time(); unet_ns1, _, _ = train_unet(seed=1, use_skips=False, log=0)
print("(b) no skip connections, seed 1: trained in %.0f seconds" % (time.time() - t0))

# %%
res67[1], rel67[1] = evaluate_breakages(unet1, unet_nt1, unet_ns1)
print("%-34s %s    %s" % ("", "seed 42: conf  entropy  cov  tmpl", "seed 1: conf  entropy  cov  tmpl"))
for name in res67[42]:
    a_, b_ = res67[42][name], res67[1][name]
    print("%-34s %14.3f %8.3f %4d %7.4f %12.3f %8.3f %4d %7.4f" % (name, a_["conf"], a_["entropy"], a_["coverage"], a_["tmpl"], b_["conf"], b_["entropy"], b_["coverage"], b_["tmpl"]))
print("relative change when the skips are zeroed: seed 42 %.2f, seed 1 %.2f" % (rel67[42], rel67[1]))

# %% [markdown]
# ### ✏️ Your turn: Which breakage hurts most?
#
# *Exercise 6.7:* Find `worst_breakage`: the name (a key of `res67[42]`) of the row with the lowest judge confidence for seed 42, excluding the healthy baseline. Then write in `answer_6_7` four sentences, one per breakage (a) to (d): did it change the judge's numbers, and in which direction? Be honest: report a breakage that did **not** show an effect, and say that the judge cannot see everything (look at the pictures too: draw 16 pictures from the broken model with `show_grid(to_unit(sample_ddpm(...)).numpy())`). Why it matters: a good experimenter reports what happened, not what the theory predicts.

# %%
worst_breakage = None     # YOUR CODE: min over the keys of res67[42] except "healthy (DDPM 100)", by res67[42][name]["conf"]
answer_6_7 = ""           # YOUR TEXT: four sentences, one per breakage

# %%
if worst_breakage is None or answer_6_7.strip() == "":
    print("Fill in worst_breakage and answer_6_7 above, then run this cell again.")
else:
    want_ = min((k for k in res67[42] if not k.startswith("healthy")), key=lambda k: res67[42][k]["conf"])
    check("worst_breakage is %r" % want_, worst_breakage == want_, "the key with the smallest 'conf' among the broken rows")
    check("your answer has at least 40 words", len(answer_6_7.split()) >= 40, "one sentence for each of (a), (b), (c) and (d)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# worst_breakage = min((k for k in res67[42] if not k.startswith("healthy")), key=lambda k: res67[42][k]["conf"])
# print("lowest confidence among the broken rows:", worst_breakage, " (%.3f)" % res67[42][worst_breakage]["conf"])
# RESULTS["p6_worst_breakage"] = worst_breakage
# x_bad = sample_ddpm(lambda x, t: unet42(x, t), (16, 1, 16, 16), S, seed=1, fix_scale=False)
# show_grid(to_unit(x_bad).numpy(), cols=8)                     # what the wrong sampler (d) draws
# ```
#
# The row with the lowest confidence is the retrained network without skips (0.881), which is also the most broken one by every other measure. A sample answer for `answer_6_7`: "(a) Without the time input the judge's numbers hardly changed (confidence 0.988, coverage 5) but the pictures are noisier (template error 0.017 against 0.0014) and one picture in eight stayed noise. (b) Without skips the network produced no shapes at all (confidence 0.881, coverage 2, template error 0.50), the same numbers as pure noise; zeroing the skips after training does the same. (c) Five sampling steps were still fine (0.979, coverage 5) but two gave blended shapes and one step gave noise. (d) The sampler without 1/sqrt(alpha) showed no visible effect in the judge's numbers or in the pictures, so the experiment did not confirm that it matters here."
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
worst_breakage = min((k for k in res67[42] if not k.startswith("healthy")), key=lambda k: res67[42][k]["conf"])
print("lowest confidence among the broken rows:", worst_breakage, " (%.3f)" % res67[42][worst_breakage]["conf"])
RESULTS["p6_worst_breakage"] = worst_breakage
x_bad = sample_ddpm(lambda x, t: unet42(x, t), (16, 1, 16, 16), S, seed=1, fix_scale=False)
show_grid(to_unit(x_bad).numpy(), cols=8)                     # what the wrong sampler (d) draws

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes two ticks. Healthy network, seed 42: confidence 1.000, entropy 1.601, coverage 5, template error 0.0014; seed 1: 0.997, 1.535, 5, 0.0063. The breakages (seed 42, then seed 1):
#
# - **(a) no time input:** 0.988, 1.583, coverage 5, template error 0.0170; seed 1: 0.980, 1.549, 5, 0.0231.
# - **(b) no skips, retrained:** 0.881, 0.524, **coverage 2**, template error 0.5005; seed 1: 0.908, 0.489, 2, 0.5283.
# - **(b') skips zeroed after training:** 0.906, 0.702, **coverage 2**, 0.3999; seed 1: 0.985, 0.378, 2, 0.6934.
# - **(c) DDIM with 5, 2 and 1 steps:** 5 steps: 0.979, 1.581, 5, 0.0059; 2 steps: 0.915, 1.594, 5, 0.0199; 1 step: 0.899, 0.687, **coverage 2**, 0.3930 (seed 1: 0.973, 0.911, 0.927 for the three; coverage 5, 5, 2).
# - **(d) no 1/√α factor:** 1.000, 1.529, 5, 0.0018; seed 1: 1.000, 1.465, 5, 0.0040.
#
# The relative change of the healthy network's output when its skips are zeroed (at t = 50) is **1.06** for seed 42 and **1.13** for seed 1. Looking at 8 pictures: (a) shows, in the 8 pictures we drew, mostly clean shapes with a speckled background and one picture that stayed a noise pattern; (b) and (b') show black-and-white speckle, no shape at all; (c) at 1 step is the same speckle; (d) shows clean shapes, as good as the healthy model. The answer to `worst_breakage` is `(b) no skips, retrained`. Tolerance: (b), (b') and the 1-step row have coverage 2 in both seeds (below 4); the healthy rows have confidence above 0.99 and coverage 5.
#
# *What it means.* An honest reading of the table:
#
# - **Skip connections are essential here.** Without them (retrained or removed after training) the model draws no shapes: the numbers are those of pure noise (confidence 0.881, entropy 0.549, coverage 2 in Step 6.4). The change of the output by 106 to 113 percent when the skips are zeroed says that the trained network gets its fine detail from the skips and not from the 4 × 4 bottleneck, as the toy of Step 5.6 suggested.
# - **The time input matters less to the judge than to the pictures.** The judge does not notice (0.988, coverage 5), but the template error is 12 times larger for seed 42 and 3.7 times for seed 1, and some pictures remain noise. This matches Step 5.2: without t the network is a compromise at every noise level. The judge alone would have told us "no problem": the second number was needed.
# - **Too few steps degrade gradually and then break.** 5 steps: fine; 2 steps: blends; 1 step: no shape.
# - **The missing 1/√α factor did not show.** We expected a failure and found none: the factor is between 1.0005 and 1.118 (for β up to 0.2), so the error per step is small, the next network call corrects part of it and the loop is run many times. The judge's entropy is a little lower (1.529 and 1.465 against 1.601 and 1.535), the template error is no worse. It is a correct finding, not a successful ablation: with other schedules (larger steps) the factor matters more, and the formula is the right one regardless.
# - **Judge and eyes see different things.** Pure noise gets a confidence of 0.881, so confidence alone says little; the entropy, the template error and the pictures themselves complete the picture, and in this lab each of them caught something that another one missed.

# %% [markdown]
# ### Step 6.8. Optional stretch: classifier-free guidance
#
# So far the model draws a random shape. **Classifier-free guidance** lets you ask for one. During training the network receives the class of the picture (through an embedding added to the time embedding), and in 15 percent of the steps the label is replaced by "no label"; so the same network learns two guesses: ε(c), the noise guess given the class, and ε(∅), the guess without it. At sampling time we mix the two: ε̃ = ε(∅) + w·(ε(c) − ε(∅)). With w = 0 the label is ignored, with w = 1 you get the plain conditional model and with w above 1 the class is *exaggerated*. This step trains a conditional network (about 75 seconds) and samples 60 pictures per class for w in 0, 1, 3 and 6.

# %% [markdown]
# ### ✏️ Your turn: The guided guess
#
# *Exercise 6.8:* Write `cfg_eps(model, x, t, y, w)`: compute the guess with the label (`model(x, t, y)`) and the guess without it (`model(x, t, null)`, where `null` is a tensor filled with the value 5, the "no label" index), and return `e_uncond + w * (e_cond - e_uncond)`. Why it matters: it is the whole trick; the rest of the sampler is unchanged. The check tests that w = 0 returns the unconditional guess and w = 1 the conditional guess, using a dummy model.

# %%
def cfg_eps_mine(model, x, t, y, w):
    """Classifier-free guidance: e_uncond + w * (e_cond - e_uncond). The 'no label' index is 5."""
    return None   # YOUR CODE: two calls of the model (with the label y and with torch.full_like(y, 5)), then combine

# %%
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

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def cfg_eps(model, x, t, y, w):
#     e_c = model(x, t, y)                                         # the guess given the requested class
#     e_u = model(x, t, torch.full_like(y, 5))                     # the guess without a class (the 'no label' index)
#     return e_u + w * (e_c - e_u)                                 # w = 0: ignore the class, w = 1: plain conditional, w > 1: exaggerate it
# ```
#
# For w = 1 the two terms collapse to e_c, the plain conditional guess; for w = 3 the guess is pushed three times as far from the unconditional one in the direction the class points, which draws stronger and more typical shapes at the price of variety.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def cfg_eps(model, x, t, y, w):
    e_c = model(x, t, y)                                         # the guess given the requested class
    e_u = model(x, t, torch.full_like(y, 5))                     # the guess without a class (the 'no label' index)
    return e_u + w * (e_c - e_u)                                 # w = 0: ignore the class, w = 1: plain conditional, w > 1: exaggerate it

# %%
t0 = time.time(); unet_cond, _, _ = train_unet(seed=42, labels=labels_img, n_classes=5, log=0)
print("conditional network trained in %.0f seconds (%d parameters)" % (time.time() - t0, count_params(unet_cond)))

# %%
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

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The training takes about 69 seconds; the conditional network has **135,905** parameters (135,713 plus 6 × 32 = 192 for the label embeddings). The check passes three ticks. The table for 300 requested pictures per w (60 per class), with the training data's dark pixels per picture as 15.2:
#
# - **w = 0:** requested class equals the judged class for **0.213** of the pictures (chance is 0.2), confidence 0.999, class entropy 1.583, spread within a class 0.217, template error 0.0017, 14.2 dark pixels.
# - **w = 1:** accuracy **1.000**, confidence 1.000, entropy 1.609, spread 0.073, template error 0.0016, 15.2 dark pixels.
# - **w = 3:** 1.000, 1.000, 1.609, 0.071, 0.0016, 15.2.
# - **w = 6:** 1.000, 1.000, 1.609, 0.070, 0.0019, 15.2.
#
# Tolerance: w = 0 accuracy below 0.4, w = 1 accuracy above 0.95. The grid shows at w = 0 five shapes that have nothing to do with the requested class (the first row), and at w = 1, 3 and 6 the five requested shapes, correctly, in every row.
#
# *What it means.* The label works: with w = 0 it is ignored (accuracy at chance), with w = 1 the conditional network always draws the requested shape. **Guidance beyond w = 1 changed almost nothing in our tiny task**, which is an honest finding and not a failure of the method: the usual trade-off (more accurate but less varied for higher w, and harsher images for very high w) shows when the conditional guess is uncertain, as with the thousands of classes and free-text prompts of Stable Diffusion. Here the five shapes are so easy that w = 1 is already certain; the spread within a class falls only from 0.073 to 0.070, the ink does not change (15.2 dark pixels at every w from 1 to 6), and the template error rises only from 0.0016 to 0.0019 at w = 6. (The shapes are always in the centre, so there is little variety for guidance to take away.) In Step 7.3 the guidance scale of a real model makes a visible difference.

# %% [markdown]
# ## Part 7. Guidance, latent space, responsible use and the capstone (10 min)
#
# *Lesson sections: "Guidance" and "Capstone".* You have built a diffusion model that works. ==This last part puts it in proportion: you compute why real image models work in a compressed "latent" space, measure what sampling costs, (optionally, with a GPU) try the settings of Stable Diffusion, write a plan for how a generative model could, and could not, help the capstone, and look back on the lab.== The tiny UNet of Part 6 is used for the cost measurements.

# %% [markdown]
# ### Step 7.1. Latent diffusion arithmetic
#
# Stable Diffusion does not run the denoising loop on pixels. A separate autoencoder compresses a 512 × 512 × 3 picture into a 64 × 64 × 4 **latent**, the UNet denoises the latent, and the autoencoder's decoder turns the result into a picture. This is **latent diffusion**. The saving is large and easy to compute. The UNet also contains attention layers (Week 8), whose cost grows with the square of the number of positions, so the saving there is larger still.

# %% [markdown]
# ### ✏️ Your turn: Count the savings
#
# *Exercise 7.1:* Compute `values_pixel` (512 × 512 × 3), `values_latent` (64 × 64 × 4), `ratio_values` (the first divided by the second), `positions_pixel` (512 × 512), `positions_latent` (64 × 64), `ratio_positions`, and `ratio_attention` (the square of `ratio_positions`: the attention table has one entry for every pair of positions). Why it matters: these three ratios (48, 64 and 4,096) are why a model that draws 512 × 512 pictures runs on one graphics card. The check tests the first and the last.

# %%
values_pixel = values_latent = ratio_values = None
positions_pixel = positions_latent = ratio_positions = ratio_attention = None      # YOUR CODE: replace every None by the computed number

# %%
if values_pixel is None or ratio_attention is None:
    print("Fill in the seven numbers above, then run this cell again.")
else:
    check("values: 786,432 against 16,384, ratio 48", (values_pixel, values_latent) == (786432, 16384) and abs(ratio_values - 48) < 1e-9, "512 * 512 * 3 and 64 * 64 * 4")
    check("positions: ratio 64", abs(ratio_positions - 64) < 1e-9, "512 * 512 = 262,144 and 64 * 64 = 4,096")
    check("attention table: ratio 4,096", abs(ratio_attention - 4096) < 1e-6, "ratio_positions squared")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# values_pixel, values_latent = 512 * 512 * 3, 64 * 64 * 4
# ratio_values = values_pixel / values_latent
# positions_pixel, positions_latent = 512 * 512, 64 * 64
# ratio_positions = positions_pixel / positions_latent
# ratio_attention = ratio_positions ** 2
# print("values: %d against %d, ratio %.0f" % (values_pixel, values_latent, ratio_values))
# print("positions: %d against %d, ratio %.0f; attention table entries: ratio %.0f" % (positions_pixel, positions_latent, ratio_positions, ratio_attention))
# assert ratio_values == 48 and ratio_positions == 64 and ratio_attention == 4096
# RESULTS["p7_ratios"] = (int(ratio_values), int(ratio_positions), int(ratio_attention))
# ```
#
# The latent has 48 times fewer numbers, 64 times fewer positions, and the attention table (positions squared) is 4,096 times smaller. These savings are paid for with an autoencoder that must reconstruct pictures well; the denoising loop then runs on the small latent.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
values_pixel, values_latent = 512 * 512 * 3, 64 * 64 * 4
ratio_values = values_pixel / values_latent
positions_pixel, positions_latent = 512 * 512, 64 * 64
ratio_positions = positions_pixel / positions_latent
ratio_attention = ratio_positions ** 2
print("values: %d against %d, ratio %.0f" % (values_pixel, values_latent, ratio_values))
print("positions: %d against %d, ratio %.0f; attention table entries: ratio %.0f" % (positions_pixel, positions_latent, ratio_positions, ratio_attention))
assert ratio_values == 48 and ratio_positions == 64 and ratio_attention == 4096
RESULTS["p7_ratios"] = (int(ratio_values), int(ratio_positions), int(ratio_attention))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes three ticks and the answer prints `values: 786432 against 16384, ratio 48` and `positions: 262144 against 4096, ratio 64; attention table entries: ratio 4096`; the assertion passes. `RESULTS["p7_ratios"]` is (48, 64, 4096).
#
# *What it means.* The loop of 20 to 50 network calls that makes a Stable Diffusion picture runs on a tensor of 16,384 numbers, not on 786,432; the attention layers handle 4,096 positions, not 262,144, and their cost grows with the square of the number of positions: 64² = 4,096 times less in the attention table. That is why a 512 × 512 model is practical on one graphics card. The price is the autoencoder: any detail it cannot reconstruct (very small text, fine textures) is lost for good, and the diffusion model never sees pixels.

# %% [markdown]
# ### Step 7.2. The cost of sampling
#
# A GAN draws a picture with one forward pass. A diffusion model needs one pass per step. We measure how long one pass of the tiny UNet takes (a batch of 32 pictures, averaged over 20 calls) and build the table of costs for the samplers of Part 6. For a large model we only extrapolate, under an assumption that we state: one network call takes 0.05 seconds (a big UNet for one picture on a good GPU; a made-up round number, not a measurement; real values depend on model, resolution and hardware).

# %%
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

# %% [markdown]
# ### ✏️ Your turn: Plan a budget
#
# *Exercise 7.2:* You want to draw 1,000 pictures with the large model (batch size 1, so 1,000 separate runs of the loop) in one minute (60 seconds). Store in `max_calls` the largest number of network calls per picture that fits, at 0.05 seconds per call, and in `ddim_steps_fit` the largest number of DDIM steps from the list [100, 50, 20, 10, 5] that fits (or `None`). Why it matters: the budget decides the sampler. The check recomputes both.

# %%
max_calls = None; ddim_steps_fit = "unset"    # YOUR CODE: 60 / (1000 * 0.05) calls per picture; then the largest entry of [100, 50, 20, 10, 5] that is at most max_calls

# %%
if max_calls is None or ddim_steps_fit == "unset":
    print("Fill in max_calls and ddim_steps_fit above, then run this cell again.")
else:
    want_calls = 60 / (1000 * 0.05); want_steps = max([s for s in (100, 50, 20, 10, 5) if s <= want_calls] or [None], key=lambda s: -1 if s is None else s)
    check("max_calls is %.1f" % want_calls, abs(max_calls - want_calls) < 1e-9, "60 seconds divided by (1,000 pictures times 0.05 seconds)")
    check("ddim_steps_fit is %s" % want_steps, ddim_steps_fit == want_steps, "the largest step count that does not exceed max_calls; None if even 5 is too many")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# max_calls = 60 / (1000 * 0.05)                                           # 1.2 calls per picture
# ddim_steps_fit = max([s for s in (100, 50, 20, 10, 5) if s <= max_calls] or [None], key=lambda s: -1 if s is None else s)
# print("calls per picture that fit: %.1f; DDIM steps that fit: %s" % (max_calls, ddim_steps_fit))
# ```
#
# Only 1.2 calls per picture fit, so even 5 DDIM steps are too many for one minute: you would need a faster model (smaller, distilled to one or a few steps), a faster graphics card, or more time. This is why research on few-step samplers is so active.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
max_calls = 60 / (1000 * 0.05)                                           # 1.2 calls per picture
ddim_steps_fit = max([s for s in (100, 50, 20, 10, 5) if s <= max_calls] or [None], key=lambda s: -1 if s is None else s)
print("calls per picture that fit: %.1f; DDIM steps that fit: %s" % (max_calls, ddim_steps_fit))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# One forward pass of the tiny UNet on a batch of 32 pictures takes about **4.6 ms** on our CPU (the number depends on the machine; expect between 3 and 20 ms). The table (tiny-UNet times are about the per-call time multiplied by the number of calls and differ from machine to machine; the large-model column does not): GAN, 1 call, about 0.005 s for the tiny UNet (the table prints 0.00) and 0.1 s (0.05 rounded) for the large model; DDPM with 1,000 calls about 4.6 s and **50.0 s**; DDPM with 100 calls about 0.46 s and 5.0 s; DDIM with 50 calls about 0.23 s and 2.5 s; DDIM with 20 calls about 0.09 s and 1.0 s. The check passes two ticks; the answer prints `calls per picture that fit: 1.2; DDIM steps that fit: None`.
#
# *What it means.* The large-model column is an assumption (0.05 s per call), so read it as a comparison, not a measurement: relative to a GAN, a 1,000-step DDPM costs 1,000 times more per picture and a 20-step DDIM 20 times. For the budget question 60 / (1,000 × 0.05) = 1.2 calls per picture, so even the 5-step DDIM does not fit: you would need a batch (several pictures per call, which lowers the cost per picture a lot on a GPU), a faster model, or a few-step sampler. Real systems do all three.

# %% [markdown]
# ### Step 7.3. Optional GPU stretch: Stable Diffusion
#
# This step needs a GPU and the `diffusers` library; it is **not required** for the grade, and the cell does nothing on a CPU. If your instructor has stored the Stable Diffusion v1.5 weights (model ID `stable-diffusion-v1-5/stable-diffusion-v1-5`, about 4 GB; the older ID `runwayml/stable-diffusion-v1-5` no longer exists), you can repeat the two experiments of this lab on a real model: change the number of steps (5, 15, 30) with the same prompt and seed, and change the guidance scale (1, 7.5, 15), which is the w of Step 6.8. What to look for: with 5 steps the picture is blurry and unfinished, from about 15 steps it settles; with guidance 1 the picture ignores the prompt more, with 7.5 it follows it, with 15 colours become harsh and saturated and details look forced (the same price as w = 6 in the toy). Free the GPU at the end.

# %%
#!pip install -q diffusers transformers accelerate

# %%
# nb-test: skip
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

# %% [markdown]
# ### Step 7.4. Capstone planning
#
# In Week 19 your capstone is the **Structural Inspection Assistant**: a photo of a structure goes in; the pipeline classifies it (crack, spalling, corrosion or none), looks up the matching checklist items with retrieval (RAG), and drafts a summary for the inspector; **a licensed engineer makes every pass or fail decision**. The Capstone page of the course app (`/capstone`) describes the project; it opens this week. Could a generative model help? Possibly, for example by making extra training pictures of rare damage. But it can also do harm. Write a short plan, as a Python dictionary that prints as a card, with five fields: `where_it_helps` (where could synthetic images help the pipeline?), `what_could_go_wrong` (name at least two risks: think of unrealistic cracks, wrong labels, mixing synthetic with real evidence), `how_i_would_evaluate` (use the judge idea of Part 6 and a human check), `what_i_will_not_do` (for example, never use synthetic pictures as evidence in a report), and `human_check` (who looks at the result and what they decide).

# %% [markdown]
# ### ✏️ Your turn: Write the plan
#
# *Exercise 7.4:* Fill in the five fields of `plan` with one to three sentences each, in your own words and about this project (not generic). Why it matters: planning how a tool could fail before you build it is the habit of responsible engineering; the evaluation field should name a number you would measure. The check only tests that every field has at least 12 words and that the evaluation field mentions a measurement.

# %%
plan = {"where_it_helps": "", "what_could_go_wrong": "", "how_i_would_evaluate": "", "what_i_will_not_do": "", "human_check": ""}    # YOUR TEXT in every field

# %%
empty_ = [k for k, v in plan.items() if len(v.split()) < 12]
if len(empty_) == len(plan):
    print("Fill in the five fields of plan above, then run this cell again.")
else:
    for k, v in plan.items(): check("%-22s %d words" % (k, len(v.split())), len(v.split()) >= 12, "write at least one full sentence")
    check("the evaluation names something you would measure (accuracy, confidence, coverage, percent, ...)", any(w in plan["how_i_would_evaluate"].lower() for w in ("accuracy", "confidence", "coverage", "entropy", "percent", "error", "recall", "precision", "%")), "for example the judge's confidence and coverage, or the accuracy of the classifier on real pictures only")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# plan = {"where_it_helps": "Spalling and corrosion photos are rare in a small data set, so a diffusion model trained on real examples could add varied extra training pictures for the classifier, but only for training.",
#         "what_could_go_wrong": "Generated cracks may look unrealistic or too regular, the classifier may learn the artefacts of the generator instead of real damage, and a synthetic picture could be mistaken for a real inspection photo if the files are mixed.",
#         "how_i_would_evaluate": "Train the classifier with and without synthetic pictures and compare accuracy and recall on real, held-out photos only; use a judge's confidence and coverage on the generated set, and let an engineer rate 50 generated pictures as realistic or not.",
#         "what_i_will_not_do": "I will never add synthetic pictures to the test set or to a report, never present them as evidence of a real structure, and I will label every generated file with its origin.",
#         "human_check": "A licensed engineer reviews the data set and the evaluation, and makes every pass or fail decision on real photos; the model only suggests and summarises."}
# print("CAPSTONE CARD: generative model for the Structural Inspection Assistant")
# for k, v in plan.items(): print("- %s: %s" % (k.replace("_", " "), v))
# ```
#
# Notice the pattern: the generated pictures are used only where a mistake is cheap (training), the decision is human, and the evaluation uses real data. Your own plan can differ; what matters is that every field is specific to this project.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
plan = {"where_it_helps": "Spalling and corrosion photos are rare in a small data set, so a diffusion model trained on real examples could add varied extra training pictures for the classifier, but only for training.",
        "what_could_go_wrong": "Generated cracks may look unrealistic or too regular, the classifier may learn the artefacts of the generator instead of real damage, and a synthetic picture could be mistaken for a real inspection photo if the files are mixed.",
        "how_i_would_evaluate": "Train the classifier with and without synthetic pictures and compare accuracy and recall on real, held-out photos only; use a judge's confidence and coverage on the generated set, and let an engineer rate 50 generated pictures as realistic or not.",
        "what_i_will_not_do": "I will never add synthetic pictures to the test set or to a report, never present them as evidence of a real structure, and I will label every generated file with its origin.",
        "human_check": "A licensed engineer reviews the data set and the evaluation, and makes every pass or fail decision on real photos; the model only suggests and summarises."}
print("CAPSTONE CARD: generative model for the Structural Inspection Assistant")
for k, v in plan.items(): print("- %s: %s" % (k.replace("_", " "), v))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The check passes a tick for each field of your plan and one for the evaluation (it counts words and looks for a word like accuracy, confidence, coverage or percent). The answer shows an example card. What a good plan has in common: **(1)** synthetic pictures only where a mistake is cheap (training, never the test set or a report); **(2)** an evaluation on real photos, with a number (the classifier's recall for spalling on real held-out photos, with and without synthetic training pictures) and a human check; **(3)** a named risk specific to this project (unrealistic cracks that teach the classifier the generator's artefacts, or synthetic and real files mixed); **(4)** a licensed engineer making every pass or fail decision. A plan that says "the model could also write the report and decide" fails the last point: in the capstone the model suggests and a licensed engineer decides.

# %% [markdown]
# ### Step 7.5. Your results and four questions
#
# Everything you saved in `RESULTS` is printed here, so that you can paste it into your report. Then answer four questions in your own words, each with at least one number from this lab.

# %%
for k, v in RESULTS.items(): print("%-34s %s" % (k, v))

# %% [markdown]
# ### ✏️ Your turn: Reflect
#
# *Exercise 7.5:* Answer in `reflection`: **q1** What are the forward and the reverse process, and which of them involves learning? **q2** Why does the network predict the noise (and not the picture), and what loss does it use? **q3** Why does the denoiser need the step t, and why a UNet for pictures (cite one number from Part 5 or 6.7)? **q4** Compare diffusion with the GAN of Week 12 on training, sampling and coverage (cite numbers: the blobs and the share on a blob, the number of network calls). Why it matters: being able to explain the method in a few sentences with numbers is the real test of understanding. Each answer needs at least 25 words and at least one digit.

# %%
reflection = {"q1": "", "q2": "", "q3": "", "q4": ""}    # YOUR TEXT

# %%
if all(v.strip() == "" for v in reflection.values()):
    print("Write your four answers in reflection above, then run this cell again.")
else:
    for k, v in reflection.items():
        check("%s: at least 25 words and at least one number" % k, len(v.split()) >= 25 and any(ch.isdigit() for ch in v), "write two or three sentences and cite a number from the lab")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reflection = {
#     "q1": "The forward process adds Gaussian noise to a picture in T = 100 small steps until only noise is left (abar_100 = 2e-5); it is a fixed recipe and nothing is learned. The reverse process removes noise step by step, and only this part is learned, by the network.",
#     "q2": "Predicting the noise gives every training example a known right answer (the noise we drew), the loss is a plain mean squared error and training is as steady as for a classifier; the loss stayed above 0 (0.25 on the blobs) because part of the noise cannot be told apart from the data.",
#     "q3": "The same noisy picture means different things at t = 5 and t = 90, so the network must be told t; without it the share of points on a blob fell from 0.92 to about 0.36. A UNet works at several resolutions, so one neuron sees the whole 16 x 16 picture (receptive field 35) while the skips keep the exact positions.",
#     "q4": "The GAN needed careful tuning and often lost blobs, while diffusion covered 8 of 8 blobs with about 91 percent of its points on a blob with plain mean squared error training and two seeds; the price is sampling: 100 network calls instead of 1, reduced to 20 by DDIM with little loss in the judge's numbers.",
# }
# for k, v in reflection.items(): print(k, len(v.split()), "words")
# ```
#
# These are examples; yours should use your own numbers and words.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
reflection = {
    "q1": "The forward process adds Gaussian noise to a picture in T = 100 small steps until only noise is left (abar_100 = 2e-5); it is a fixed recipe and nothing is learned. The reverse process removes noise step by step, and only this part is learned, by the network.",
    "q2": "Predicting the noise gives every training example a known right answer (the noise we drew), the loss is a plain mean squared error and training is as steady as for a classifier; the loss stayed above 0 (0.25 on the blobs) because part of the noise cannot be told apart from the data.",
    "q3": "The same noisy picture means different things at t = 5 and t = 90, so the network must be told t; without it the share of points on a blob fell from 0.92 to about 0.36. A UNet works at several resolutions, so one neuron sees the whole 16 x 16 picture (receptive field 35) while the skips keep the exact positions.",
    "q4": "The GAN needed careful tuning and often lost blobs, while diffusion covered 8 of 8 blobs with about 91 percent of its points on a blob with plain mean squared error training and two seeds; the price is sampling: 100 network calls instead of 1, reduced to 20 by DDIM with little loss in the judge's numbers.",
}
for k, v in reflection.items(): print(k, len(v.split()), "words")

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; training differs a little between machines):
#
# The printed `RESULTS` list has about 45 entries. The important ones, all for seed 42: p4_params 35714; p4_final_loss 0.254; p4_blobs_s42 8 with p4_on_blob_s42 0.917 (seed 1: 8 and 0.912); p5_unet_params 135713; p5_share_mid 0.424; p5_notime_on_blob 0.363; p6_ddpm_conf 1.0 with p6_ddpm_entropy 1.601 and p6_ddpm_coverage 5; p6_ddim20_conf 0.997; p6_worst_breakage '(b) no skips, retrained'; p7_ratios (48, 64, 4096). The check passes four ticks when each of your answers has at least 25 words and a number.
#
# *What it means.* Your four answers are the real test of the lab. A good answer to q4, for example, says that diffusion covered all 8 blobs with 91 to 92 percent of its points on a blob for two seeds (with an ordinary squared-error loss), that the GAN of Week 12 needed tuning and could lose blobs, and that the price of diffusion is 100 network calls per sample against 1 for the GAN (20 with DDIM, with almost no loss in the judge's numbers).

# %% [markdown]
# ## Wrap-up
#
# What you learned in this notebook:
#
# - **Diffusion is a fixed forward recipe and a learned reverse.** You added noise with a schedule (T = 100, ᾱ_100 = 2e-5), trained a network to predict the noise with a plain squared-error loss, and sampled by repeating the reverse step: 8 of 8 blobs with about 92 percent of the points on a blob, for two seeds, with no discriminator.
# - **The network needs the noise level and a UNet.** The time embedding cut the loss from about 0.54 to 0.25 on the 2-D data; the UNet (135,713 parameters) learned to draw five shapes in about a minute (confidence 1.0, entropy 1.601, coverage 5) and drew nothing at all without its skip connections. Faster DDIM sampling gave almost the same quality with 20 calls instead of 100.
# - **Measure, then trust.** The judge, the template error and your eyes caught different things; two of the four breakages showed no effect on the judge, and guidance beyond w = 1 changed nothing on this tiny task. Reporting that honestly is part of the method. For the capstone you now have a plan with limits: generated pictures only for training, real photos for evaluation, and a licensed engineer who decides.
#
# **Next week (Week 14)** turns from building models to adapting big ones: fine-tuning, RLHF and parameter-efficient methods (PEFT, such as LoRA). **The capstone opens this week:** read the Capstone page of the course app (`/capstone`) and start your plan (the card of Step 7.4 is a first piece).
#
# Double-click here to write two sentences: what surprised you most in this lab, and which experiment would you run next with more time?
#
# Remember: **files in `/content` vanish when your Colab session ends.** Use **File → Save a copy in Drive** and download anything you want to keep (File → Download .ipynb). Submit the notebook the way your instructor asks.
