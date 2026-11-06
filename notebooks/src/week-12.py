# %% [markdown]
# # Week 12 — Image Generation and GANs: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is fine for everything; no GPU is needed. Nothing is downloaded: every data set is drawn with NumPy. **Time:** about 3 hours for this notebook. Every cell finishes in seconds to about two minutes.
#
# This lab is called **Imagine the forger and the detective**. Week 12 asks how a network can *invent* pictures it has never seen when the only teacher is a game against a critic. You answer by building the story from the bottom: a generator built by hand and an autoencoder that fails to invent, the transpose convolution taken apart, the GAN game computed by hand, a one-dimensional GAN, a two-dimensional GAN where you provoke and measure mode collapse, and a small DCGAN that draws line shapes, judged by a classifier.
#
# ### What you will do
#
# 1. **Part 1. Generating is sampling** (25 min): pictures as points, a hand-built generator, noise turned into a ring, a walk in latent space, and an autoencoder that is not a generative model.
# 2. **Part 2. Transpose convolution by hand** (25 min): the output-size formula, the stamp view in NumPy, the transposed matrix, the link to the input gradient of Week 4, checkerboards, and a layer that learns to upsample.
# 3. **Part 3. The GAN game by hand** (20 min): the value of the game, the discriminator loss, saturating versus non-saturating generator loss, the optimal discriminator and one training step.
# 4. **Part 4. A one-dimensional GAN from scratch** (25 min): train a forger and a detective on a bell curve, watch the histogram form, and compare the two generator losses.
# 5. **Part 5. A two-dimensional GAN and mode collapse** (25 min): eight blobs, a healthy run, a collapsed run, three remedies and the lesson that losses lie.
# 6. **Part 6. A DCGAN on tiny pictures** (35 min): shapes and parameters, a classifier as judge, training with `ConvTranspose2d`, latent walks and breaking the model on purpose.
# 7. **Part 7. Judge, design and reflect** (15 min): diversity and quality as numbers, a responsible-use design, a misuse pre-mortem and reflection questions.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells (the exercises) are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried. Your key numbers are collected in a dictionary called `RESULTS` that Part 7 prints.
# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed so your results match the ones in the lesson, and defines helpers: `check(...)` to test your answers; data sets drawn with NumPy so that nothing is downloaded: `sample_real_1d(...)` (numbers from a bell curve, the "real data" of the first GAN), `make_mixture(...)` (points from eight Gaussian blobs arranged in a ring: a data set with several **modes**) and `make_shapes(...)` (the five line shapes of Week 4, the "real pictures" of the DCGAN); `modes_covered(...)` to count how many of the blobs a generator reaches and how many of its points land on a blob; `show_grid(...)` to look at a grid of small pictures; `count_params(...)` and `train_classifier(...)` (Week 4) to train the picture classifier that we later use as a judge of generated pictures. Everything runs on a CPU in a few minutes.

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
# ## Part 1. Generating is sampling (25 min)
#
# *Lesson section: "Generating".* Until now the models of this course *answered* questions: which class is this picture, what is the next word. A **generative model** does something stranger: it *invents* a new example that looks as if it came from the same source as the training data. The key idea of the whole week fits in one sentence: **generating is sampling: you draw random numbers that are easy to draw, and a function turns them into a picture (or a number, or a point) that follows the distribution of the real data.** In this part you build the smallest honest examples by hand. A function that turns a bell curve into another bell curve, a function that turns random angles into a ring, a "decoder" that you walk along a straight line, and finally a trained autoencoder that, as you will measure, fails to invent anything useful. At the end you will know exactly what is missing, which is the question that the rest of the week answers.

# %% [markdown]
# ### ✏️ Imagine first
#
# The pictures of this week are tiny 16 by 16 greyscale pictures of line shapes (the data of Week 4). Suppose that you fill every pixel with a random number and look at the result. Then suppose that you train an autoencoder (Week 6) on 600 real shape pictures, throw away its encoder, feed its decoder random numbers and look at the result. Before you run any code, write two predictions. (1) In one short sentence: what does a picture made of random pixels look like? (2) Out of 8 pictures made by the decoder from random numbers, how many will look like one of the five shapes (a number from 0 to 8)? You compare with Steps 1.1 and 1.5.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: noise pictures and an autoencoder that tries to invent
my_noise_look = None     # your sentence, e.g. "grey snow" (a string)
my_ae_good = None        # your guess: how many of 8 decoded random vectors look like a shape (0 to 8)

# %% [markdown]
# ### Step 1.1. Pictures are points
#
# A 16 by 16 greyscale picture has 256 pixels, so it is **one point in a 256-dimensional space**: one number per axis. A 64 by 64 colour picture is a point in a space with 64 x 64 x 3 = 12,288 axes. Most points of that space are noise, and the real pictures sit on a tiny, thin, structured part of it. The block below makes 8 pictures with uniform random pixels, 8 with Gaussian random pixels (clipped to the range 0 to 1) and 8 real shape pictures, shows them side by side, and measures the mean and the standard deviation of the pixels in each family, plus the share of bright pixels (above 0.5).

# %%
rng = np.random.RandomState(SEED)
print("numbers in a 16 x 16 greyscale picture:", 16 * 16)
print("numbers in a 64 x 64 RGB picture:      ", 64 * 64 * 3)

noise_u = rng.rand(8, 1, 16, 16).astype(np.float32)                                     # every pixel uniform in [0, 1]
noise_g = np.clip(0.5 + 0.25 * rng.randn(8, 1, 16, 16), 0, 1).astype(np.float32)        # Gaussian around 0.5, clipped
Xreal, yreal = make_shapes(300, seed=SEED)                                              # 300 real pictures (noise 0.15)
show_grid(noise_u, ["uniform noise"] * 8)
show_grid(noise_g, ["Gaussian noise"] * 8)
show_grid(Xreal[:8], [SHAPE_NAMES[k] for k in yreal[:8]])

for name, A in [("uniform noise", noise_u), ("Gaussian noise", noise_g), ("real shapes", Xreal)]:
    print(f"{name:15s} mean {A.mean():.3f}  std {A.std():.3f}  share of pixels above 0.5: {(A > 0.5).mean():.3f}")
assert 16 * 16 == 256 and 64 * 64 * 3 == 12288

# %% [markdown]
# **What you should see.** You should see three figures and a small table of numbers. The first line prints 256 and 12,288 (the number of axes of the two spaces). The first figure (uniform noise) and the second (Gaussian noise) are grey static, like an untuned television: every picture looks the same as every other, nothing in them is a shape. The third figure shows eight real pictures: a thin bright line (a bar, a cross, a square outline or a diagonal) on a dark background, with a little speckle. The table says the same in numbers: the uniform noise has a pixel mean of 0.498 and a standard deviation of 0.292, and 0.505 of its pixels are bright; the Gaussian noise has mean 0.503, standard deviation 0.241 and 0.503 bright pixels; the real pictures have mean 0.060, standard deviation 0.280 and only 0.060 bright pixels. So a real picture is mostly dark with about 15 bright pixels out of 256, while noise has half of its pixels bright. Notice that the standard deviation alone is not a good detector (0.280 for real pictures is as large as for noise): what matters is the *structure*, which pixels are bright together. Real pictures occupy a tiny, thin part of the 256-dimensional space; a generator has to learn where it is.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.1. Dimension and ink
#
# Your task: compute the number of axes of a 224 by 224 RGB picture (`dim_224`) and the average number of *bright pixels* (above 0.5) in one real picture of `Xreal` (`bright_per_picture`, a number). Why it matters: the "dimension of the data" is the number of numbers a generator must produce for one example, and the thin structure of real pictures (few bright pixels) is what a generator must learn.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
dim_224 = None             # number of numbers in a 224 x 224 RGB picture
bright_per_picture = None  # average number of pixels above 0.5 in one picture of Xreal

# %%
if dim_224 is None or bright_per_picture is None:
    print("Fill in dim_224 and bright_per_picture above, then run this cell again.")
else:
    check("dim_224 is right", dim_224 == 224 * 224 * 3, "Height times width times 3 channels.")
    check("bright_per_picture is right", abs(bright_per_picture - float((Xreal > 0.5).sum(axis=(1, 2, 3)).mean())) < 0.01, "Count the pixels above 0.5 in each picture, then average over the 300 pictures.")
    RESULTS["p1_dim"] = (int(dim_224), round(float(bright_per_picture), 2))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# dim_224 = 224 * 224 * 3                                   # 150,528
# bright_per_picture = float((Xreal > 0.5).sum(axis=(1, 2, 3)).mean())
# ```
#
# `dim_224` is 150,528 and each real picture has on average 15.3 pixels above 0.5 (6 percent of 256). A generator for 224 by 224 pictures has to produce 150,528 numbers that are jointly structured, which is why the networks of the next parts are built with convolutions and not with one number per pixel.
#
# </details>

# %% [markdown]
# ### Step 1.2. A hand-built generator
#
# Here is the smallest generator there is. Draw 10,000 random numbers z from a standard bell curve N(0, 1) (the **latent** numbers: *latent* means hidden, the input the user never sees) and apply a function G. If G(z) = a * z + b, the output is again a bell curve, with mean b and standard deviation |a|. So choosing a = 1.5 and b = 4 turns N(0, 1) into N(4, 1.5 squared). A **non-linear** G changes the shape: with G(z) = 4 + 2 * tanh(z) the bell curve is squashed into the range 2 to 6. The block builds both, checks the numbers and draws the histograms.

# %%
rng = np.random.RandomState(SEED)
z = rng.randn(10000)                       # the latent numbers: N(0, 1)

a, b = 1.5, 4.0
g_lin = a * z + b                          # a linear generator
print(f"linear   G(z) = {a} z + {b}:  mean {g_lin.mean():.3f}  std {g_lin.std():.3f}")
assert abs(g_lin.mean() - 4.0) < 0.05 and abs(g_lin.std() - 1.5) < 0.05

g_tanh = 4 + 2 * np.tanh(z)                # a non-linear generator
print(f"tanh     G(z) = 4 + 2 tanh(z):  mean {g_tanh.mean():.3f}  std {g_tanh.std():.3f}  min {g_tanh.min():.3f}  max {g_tanh.max():.3f}")

fig, ax = plt.subplots(1, 3, figsize=(10, 2.6), sharey=False)
ax[0].hist(z, 50, color="grey");        ax[0].set_title("latent z ~ N(0, 1)")
ax[1].hist(g_lin, 50, color="C0");      ax[1].set_title("a z + b  (a = 1.5, b = 4)")
ax[2].hist(g_tanh, 50, color="C1");     ax[2].set_title("4 + 2 tanh(z)")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** You should see two printed lines and three histograms in a row. The first histogram is the latent bell curve centred at 0. The linear generator gives a mean of 3.997 and a standard deviation of 1.505 (the targets were 4 and 1.5, and the silent assertion allows an error of 0.05; the small gap is sampling noise, because 10,000 samples are not infinitely many). Its histogram is the same bell, shifted and stretched. The `tanh` generator is different: the mean is again 3.997 (by symmetry, the squashing is the same on both sides), but the standard deviation is only 1.256, and the outputs lie between 2.002 and 5.998: they can never leave the range 2 to 6, whatever z is. The histogram is a squashed bell with its tails pushed against the walls. The lesson: the shape of the output distribution is decided entirely by G and by the distribution of z. Training a generator means choosing G so that this shape is the shape of the data.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.2. Choose a and b
#
# Your task: choose `a_new` and `b_new` so that `a_new * z + b_new` follows N(-2, 0.5 squared), that is a mean of -2 and a standard deviation of 0.5, and store the generated numbers in `g_new`. Why it matters: this is, in miniature, what training a generator means: finding the parameters of G so that the distribution of G(z) matches the data. Here you do it by thinking; later the discriminator will do it for you.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
a_new = None   # the multiplier
b_new = None   # the shift
g_new = None   # a_new * z + b_new

# %%
if a_new is None or b_new is None:
    print("Fill in a_new and b_new above (and compute g_new = a_new * z + b_new), then run this cell again.")
else:
    g_new = a_new * z + b_new
    check("mean is -2 (within 0.05)", abs(g_new.mean() + 2.0) < 0.05, "The shift b_new is the mean, because z has mean 0.")
    check("std is 0.5 (within 0.05)", abs(g_new.std() - 0.5) < 0.05, "The multiplier a_new is the standard deviation, because z has std 1.")
    RESULTS["p1_ab"] = (float(a_new), float(b_new))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# a_new, b_new = 0.5, -2.0
# g_new = a_new * z + b_new
# ```
#
# With a = 0.5 and b = -2 the generated numbers have mean -2.001 and standard deviation 0.502. A negative a would work too (the bell curve is symmetric), which shows that many different generators produce the same distribution.
#
# </details>

# %% [markdown]
# ### Step 1.3. From uniform noise to a ring
#
# Now the output is a point in the plane, and the noise is *uniform*, not a bell curve: (z1, z2) is uniform in the unit square. Take G(z) = (r cos(2 pi z1), r sin(2 pi z1)) with r = 3 + 0.3 (z2 - 0.5), plus a little Gaussian noise (0.1). The first latent number picks an angle, the second wiggles the radius a little, so the points form a ring. Compare it with the real data of the week: `make_mixture` gives 2,000 points from **eight** small round blobs (Gaussian clusters, each called a *mode*) whose centres lie on a circle of radius 3. The helper `modes_covered(points, centres)` counts how many blobs receive at least 2 percent of the points and what fraction of all points lie within 3 standard deviations (0.45) of some blob centre.

# %%
rng = np.random.RandomState(SEED)
z12 = rng.rand(2000, 2)                                              # uniform noise in the unit square
r = 3 + 0.3 * (z12[:, 1] - 0.5)
ring = np.stack([r * np.cos(2 * np.pi * z12[:, 0]), r * np.sin(2 * np.pi * z12[:, 0])], 1) + 0.1 * rng.randn(2000, 2)

Xmix, kmix, centres = make_mixture(2000, seed=SEED)                  # the real data: 8 blobs
fig, ax = plt.subplots(1, 2, figsize=(8, 3.8))
ax[0].scatter(ring[:, 0], ring[:, 1], s=4, alpha=0.5);  ax[0].set_title("generator: a ring")
ax[1].scatter(Xmix[:, 0], Xmix[:, 1], s=4, alpha=0.5);  ax[1].set_title("real data: 8 blobs")
for a_ in ax:
    a_.scatter(centres[:, 0], centres[:, 1], marker="x", c="k"); a_.set_aspect("equal"); a_.set_xlim(-4.5, 4.5); a_.set_ylim(-4.5, 4.5)
plt.tight_layout(); plt.show()

print("ring generator :", modes_covered(ring, centres))
print("real mixture   :", modes_covered(Xmix, centres))

# %% [markdown]
# **What you should see.** You should see two scatter plots with a black cross at each of the eight blob centres. On the left the generator draws a **smooth ring** of points that passes through all eight centres; on the right the real data has **eight tight clumps** on the same circle with empty space between them. The numbers agree: `modes_covered` gives (8, 0.366) for the ring and (8, 0.988) for the real mixture. Read the pair as (blobs reached, fraction of points that land on a blob). The ring reaches all 8 blobs (every blob gets more than 2 percent of the points), but only 37 percent of its points are near a centre: the rest sit in the gaps where the real data never goes. So the ring has the right *support* (the circle) but the wrong *lumpiness*. Keep these two numbers in mind, they will be the yardstick of Part 5: **diversity** (how many blobs) and **quality** (how many points on a blob).

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.3. A generator with blobs
#
# Your task: change the generator so that it produces **blobs**: replace the continuous angle `2 pi z1` by one of eight fixed angles, `2 pi floor(8 z1) / 8`, and use a noise of 0.15. Store the points in `blobs` (an array of shape (2000, 2)) and the result of `modes_covered` in `cov_blobs`. Why it matters: the ring generator was almost right, but its output is smooth where the data is lumpy. A non-linear function (here a staircase) can build lumps from smooth noise; a trained generator has to discover such a function by itself.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
blobs = None       # array of shape (2000, 2): your blob generator applied to z12
cov_blobs = None   # the result of modes_covered(blobs, centres)

# %%
if blobs is None or cov_blobs is None:
    print("Fill in blobs and cov_blobs above, then run this cell again.")
else:
    check("shape is (2000, 2)", np.shape(blobs) == (2000, 2), "One (x, y) point per latent pair.")
    check("all 8 blobs are covered", cov_blobs[0] == 8, "The angle must take 8 different values 2 pi k / 8.")
    check("most points sit on a blob (above 0.9)", cov_blobs[1] > 0.9, "Use noise 0.15 and keep the radius at 3.")
    RESULTS["p1_blobs"] = (int(cov_blobs[0]), round(float(cov_blobs[1]), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ang = 2 * np.pi * np.floor(8 * z12[:, 0]) / 8
# blobs = np.stack([3 * np.cos(ang), 3 * np.sin(ang)], 1) + 0.15 * rng.randn(2000, 2)
# cov_blobs = modes_covered(blobs, centres)
# ```
#
# The staircase generator covers 8 blobs with 98.9 percent of its points on a blob: almost the real data. A smooth input became lumps because a *non-linear* function bent the line of angles into eight plateaus. Real generators are not told this function; they learn it from a signal that says "your points in the gaps look fake".
#
# </details>

# %% [markdown]
# ### Step 1.4. Latent interpolation by hand
#
# A generator maps a **latent space** (here 16 numbers) to the data space (here 256 pixels). One of the nicest things to do with it is to take two latent vectors z_a and z_b and walk along the straight line between them: z(alpha) = (1 - alpha) z_a + alpha z_b, for alpha from 0 to 1. To see the idea without any training, the block uses a **toy linear decoder**: a random matrix W of shape 256 x 16 followed by tanh. The seven weights alpha are 0, 1/6, 2/6, ..., 1. Because W is linear, the pre-activation of the middle picture is exactly the average of the two end pictures; the assertion below checks that.

# %%
rng = np.random.RandomState(SEED)
W = rng.randn(256, 16) / 4                       # the toy decoder: 16 latent numbers -> 256 pixels
z_a, z_b = rng.randn(16), rng.randn(16)          # two latent vectors

alphas = np.linspace(0, 1, 7)
print("alpha:", np.round(alphas, 4))
Z = np.array([(1 - al) * z_a + al * z_b for al in alphas])        # 7 latent vectors on the line
pre = Z @ W.T                                                     # (7, 256) pre-activations
pics = np.tanh(pre).reshape(7, 1, 16, 16)                         # pictures in [-1, 1]
show_grid(pics, [f"alpha {al:.2f}" for al in alphas], cols=7)

mid = pre[3]; ends = 0.5 * (pre[0] + pre[6])
print("middle picture equals the average of the ends (before tanh):", np.allclose(mid, ends))
assert np.allclose(mid, ends) and len(alphas) == 7 and abs(alphas[1] - 1 / 6) < 1e-12

# %% [markdown]
# **What you should see.** You should see the printed weights `[0. 0.1667 0.3333 0.5 0.6667 0.8333 1.]` (seven numbers, equally spaced, 1/6 apart), a strip of seven pictures labelled alpha 0.00 to alpha 1.00, and the line that says `True`. The pictures are random patterns (the decoder is random, not trained), but they change **gradually**: the first picture slowly turns into the last one, each step being a little more like the end and a little less like the start. This gradual change is the signature of a good latent space: nearby latent vectors give similar pictures. The assertion confirms the linearity: for this decoder the middle picture (before the tanh) is exactly the average of the two end pictures.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.4. A third of the way
#
# Your task: decode the latent vector at alpha = 1/3 and store the 256 pre-activations in `pre_third`. Then compute `expected` = two thirds of the pre-activation of z_a plus one third of that of z_b, without decoding a new latent vector. Why it matters: interpolating in latent space and mixing the outputs are the same thing for a linear decoder. For a deep non-linear generator they are not, and that is why a smooth walk in latent space is such a strong test of what a generator has learned.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
pre_third = None   # W applied to the latent vector at alpha = 1/3 (shape (256,))
expected = None    # 2/3 of W z_a plus 1/3 of W z_b

# %%
if pre_third is None or expected is None:
    print("Fill in pre_third and expected above, then run this cell again.")
else:
    check("pre_third has 256 numbers", np.shape(pre_third) == (256,), "W has shape (256, 16) and the latent vector has 16 numbers.")
    check("the two agree", np.allclose(pre_third, expected), "Both are linear in alpha.")
    RESULTS["p1_third"] = float(np.abs(np.asarray(pre_third) - np.asarray(expected)).max())

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pre_third = W @ ((2 / 3) * z_a + (1 / 3) * z_b)
# expected = (2 / 3) * (W @ z_a) + (1 / 3) * (W @ z_b)
# ```
#
# The largest difference between the two is 6.66e-16, which is rounding error: a linear map keeps straight lines straight.
#
# </details>

# %% [markdown]
# ### Step 1.5. A vanilla autoencoder is not a generative model
#
# In Week 6 you met the autoencoder: an encoder squeezes a picture into a few numbers and a decoder rebuilds it. The decoder *is* a function from a small latent space to pictures, so why not feed it random numbers and call it a generator? Try. The block trains an MLP autoencoder (256 to 32 to 256) on 600 shape pictures with a little jitter (`shift=1`), shows 8 reconstructions, then decodes 8 **random** latent vectors drawn from a standard bell curve, and measures two distances: how far a reconstruction is from its original, and how far a decoded random vector is from the **nearest** of the 600 training pictures.

# %%
torch.manual_seed(SEED)
Xtr_ae, _ = make_shapes(600, shift=1, seed=SEED)
Xt = torch.tensor(Xtr_ae.reshape(600, 256))
enc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 32))
dec = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 256), nn.Sigmoid())
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
for epoch in range(150):
    perm = torch.randperm(600)
    for i in range(0, 600, 60):
        idx = perm[i:i + 60]
        loss = F.mse_loss(dec(enc(Xt[idx])), Xt[idx])
        opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    rec = dec(enc(Xt))
    mse_rec = float(F.mse_loss(rec, Xt))
    z_rand = torch.randn(8, 32)
    fake = dec(z_rand)
show_grid(Xtr_ae[:8], ["original"] * 8)
show_grid(rec[:8].reshape(8, 1, 16, 16).numpy(), ["reconstruction"] * 8)
show_grid(fake.reshape(8, 1, 16, 16).numpy(), ["random z"] * 8)

def dist_to_nearest(P, train):                    # distance of every row of P to its nearest row of train
    d = torch.cdist(P, train)
    return d.min(dim=1).values
d_rec = float(torch.linalg.norm(rec - Xt, dim=1).mean())      # reconstruction vs its own original
d_fake = float(dist_to_nearest(fake, Xt).mean())              # decoded random vector vs nearest training picture
Xnew, _ = make_shapes(8, shift=1, seed=SEED + 1)             # 8 FRESH real pictures, not in the training set
d_new = float(dist_to_nearest(torch.tensor(Xnew.reshape(8, 256)), Xt).mean())
print(f"reconstruction MSE per pixel: {mse_rec:.4f}")
print(f"mean distance, reconstruction to its original   : {d_rec:.3f}")
print(f"mean distance, decoded random z to nearest train: {d_fake:.3f}")
print(f"mean distance, a fresh REAL picture to nearest train: {d_new:.3f}")
assert mse_rec < 0.05

# %% [markdown]
# **What you should see.** You should see three strips of pictures and three numbers. The originals are crisp shapes; the reconstructions (the decoder fed with the **encoded** pictures) look like clean copies of them, with most of the speckle removed: the autoencoder works, with a reconstruction error of 0.0393 per pixel (for comparison, the speckle noise that we added has a variance of 0.15 squared, 0.0225, which no autoencoder can reproduce). The third strip is the test of this step: eight pictures decoded from random latent vectors. They are grey, noisy smears: you may recognise a faint cross or bar in one or two of them (my honest count is at most 2 of 8), none of them is crisp, and none is as clean as a reconstruction. The distances say the same. A reconstruction is 3.01 away from its original, and a fresh *real* picture (never seen in training) is 3.19 away from its nearest training picture: that is what real data looks like from the training set. A decoded random vector is 5.20 away from its nearest training picture, much farther. What is missing? Nobody told the decoder **where in the 32-dimensional latent space the real data live**. The encoder puts the 600 pictures into some small region of that space; a standard bell curve covers a much larger region, and most random vectors land where the decoder has never been trained. A plain autoencoder learns to decode only the codes that its own encoder produces. It is a compression machine, not a generative model.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.5. Sample where the data live
#
# Your task: encode all 600 training pictures, compute the mean and the standard deviation of each of the 32 latent coordinates (over the 600 pictures), draw 8 new latent vectors from a normal distribution with **that** mean and standard deviation, decode them and store the mean distance to the nearest training picture in `d_fit`. Why it matters: the decoder was never told where the real data live in its latent space. If you tell it, by fitting a simple distribution to the encoded training pictures, the decoded pictures get closer to real ones. This is the first step towards the variational autoencoder, and a hint that a good generator must learn the *distribution* of the latent space too.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
d_fit = None   # mean distance of 8 decoded, data-fitted random vectors to the nearest training picture

# %%
if d_fit is None:
    print("Fill in d_fit above, then run this cell again.")
else:
    check("d_fit is a plausible distance (between 0 and 10)", 0 < d_fit < 10, "Use torch.cdist as in dist_to_nearest.")
    check("sampling where the data live is closer than N(0, 1) latents", d_fit < d_fake, "Use the mean and std of the encoded training pictures, not 0 and 1.")
    RESULTS["p1_dfit"] = round(float(d_fit), 3)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# with torch.no_grad():
#     codes = enc(Xt)                                   # (600, 32)
#     mu, sd = codes.mean(0), codes.std(0)
#     g = torch.Generator().manual_seed(SEED)
#     z_fit = mu + sd * torch.randn(8, 32, generator=g)
#     fake_fit = dec(z_fit)
# d_fit = float(dist_to_nearest(fake_fit, Xt).mean())
# show_grid(fake_fit.reshape(8, 1, 16, 16).numpy(), ["fitted z"] * 8)
# ```
#
# Sampling the latent vector where the encoded pictures live reduces the distance to the nearest training picture from 5.20 to 3.25 (a fresh real picture has 3.19). Look at the figure it draws: the pictures are sparser and sharper than before, but still clumsy (a few bright pixels, not clean lines), because a single Gaussian is a poor description of the encoded data. This idea, giving the decoder a latent distribution that it was trained on, is the variational autoencoder; the GAN does something different: it does not ask for a reconstruction at all.
#
# </details>

# %% [markdown]
# ### Step 1.6. Imagine and write: no correct picture
#
# The autoencoder had it easy: for every input there was a correct output (the same picture), so the loss was a plain reconstruction error. A generator has no such luxury. You hand it random numbers and there is **no correct picture** for them: nobody has a drawing that "belongs" to z = (0.3, -1.2, ...). So no pixel-by-pixel error can be computed. This step has no code to run, only a small piece of thinking: use the cell below to show again what the generator receives and what it has to produce, then write your own idea.

# %%
z_demo = np.random.RandomState(SEED).randn(1, 8)
print("what the generator receives :", np.round(z_demo[0], 2))
print("what the data set contains  : 600 pictures, but none of them is labelled with a z")
print("so the question is: how can G be told that its output 'looks real'?")

# %% [markdown]
# **What you should see.** The cell prints one latent vector of 8 numbers, and the sentence that sums up the difficulty: there are real pictures and there are random vectors, but no pairs. Possible ideas that you may have written: compare the *whole set* of generated pictures with the whole set of real ones; ask a second network to tell real from generated and use its opinion as the training signal; measure the distance between two distributions instead of two pictures. The last two are the adversarial idea of Part 3.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.6. Your idea in words
#
# Your task: write two or three sentences in the string `idea`: how would you train a generator when no picture is the correct answer for a given noise vector? Mention what the generator receives, what it produces and where the training signal could come from. Why it matters: you will meet the answer (a discriminator) in Part 3. Writing your own guess first makes the answer stick, and a wrong guess is useful: it shows what you thought a generator needs.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
idea = ""   # your two or three sentences (a string)

# %%
if len(idea.strip()) < 40:
    print("Write at least two sentences in idea above, then run this cell again.")
else:
    check("you wrote an idea", True)
    check("you mention a second network, a judge or a comparison", any(w in idea.lower() for w in ["discriminat", "judge", "critic", "detect", "compare", "classif", "real"]), "Think: who or what could say 'this looks real'?")
    RESULTS["p1_idea"] = idea.strip()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# idea = ("The generator turns random noise into a picture. Since no picture is the right one for a given noise, "
#         "I would train a second network, a judge, to tell real pictures from generated ones, and use its score "
#         "to tell the generator how to change its output so that it looks more real.")
# ```
#
# That is the adversarial idea in one paragraph: the judge replaces the missing correct picture. Part 3 turns it into a loss function, and Parts 4 to 6 train it.
#
# </details>

# %% [markdown]
# ## Part 2. Transpose convolution by hand (25 min)
#
# *Lesson section: "Upsampling".* A generator has to turn a small tensor (a few numbers, or a 4 by 4 map of features) into a big picture. A convolution makes pictures **smaller** (or keeps their size); we need the opposite operation, one that makes them **bigger** and has weights that can be learned. That operation is the **transpose convolution** (also called a *fractionally strided* convolution, or, wrongly, a "deconvolution"). It is the layer that every DCGAN generator is made of, and in Part 6 you will build one. Here you take it apart with NumPy and small pieces of PyTorch until nothing about it is a mystery: its output size, the picture of "stamps", its link to the matrix of a convolution and to the input gradient of Week 4, the uneven overlap that causes checkerboard patterns, and the fact that it can *learn* to upsample.

# %% [markdown]
# ### ✏️ Imagine first
#
# Take a 4 by 4 input and a transpose convolution with a 4 by 4 kernel, stride 2 and padding 1. A normal convolution with stride 2 would **halve** the size. What size will this layer's output have: 2 by 2, 4 by 4, 8 by 8 or 10 by 10? Write your guess, with a one-line reason. You will compute it in Step 2.1.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: the output size of a transpose convolution (n = 4, k = 4, stride 2, padding 1)
my_guess_size = None    # a number: the side of the output (2, 4, 8 or 10?)
my_reason = ""          # one line: why?

# %% [markdown]
# ### Step 2.1. The output-size formula
#
# For a normal convolution the output side is (n + 2p - k) / s + 1, rounded down. The transpose convolution **undoes the size change**: its output side is out = (n - 1) * s - 2p + k + output_padding. Here n is the input side, k the kernel side, s the stride, p the padding, and *output_padding* (default 0) is a small extra amount (smaller than the stride) that resolves an ambiguity: several input sizes give the same convolution output size, so the transpose needs a hint about which one you want back. The block writes `out_size_T`, prints the four cases of the lesson and checks each one against a real `nn.ConvTranspose2d` fed with a dummy tensor.

# %%
def out_size_T(n, k, s, p, op=0):
    """Output side of a transpose convolution: input side n, kernel k, stride s, padding p, output_padding op."""
    return (n - 1) * s - 2 * p + k + op

cases = [  # (name, n, k, s, p, output_padding)
    ("DCGAN stage",          4, 4, 2, 1, 0),
    ("no padding, k = 3",    3, 3, 2, 0, 0),
    ("k = s (stamps tile)",  4, 2, 2, 0, 0),
    ("with output_padding",  7, 3, 2, 1, 1),
]
print(f"{'case':22s} {'n':>2s} {'k':>2s} {'s':>2s} {'p':>2s} {'op':>2s}  formula  PyTorch")
for name, n, k, s, p, op in cases:
    layer = nn.ConvTranspose2d(1, 1, k, stride=s, padding=p, output_padding=op)
    real = layer(torch.zeros(1, 1, n, n)).shape[-1]
    print(f"{name:22s} {n:2d} {k:2d} {s:2d} {p:2d} {op:2d}  {out_size_T(n, k, s, p, op):6d}  {real:7d}")
    assert real == out_size_T(n, k, s, p, op)

# %% [markdown]
# **What you should see.** You should see four rows, and in each row the formula and PyTorch agree: (4, 4, 2, 1) gives **8**, (3, 3, 2, 0) gives **7**, (4, 2, 2, 0) gives **8** and (7, 3, 2, 1, 1) gives **14**. So the answer to the Imagine question is 8: with stride 2 the layer **doubles** the side (the padding of 1 trims one cell from each side of the full result, and the kernel adds back 2: (4 - 1) * 2 - 2 + 4 = 8). That is exactly the "k = 4, s = 2, p = 1" layer that doubles the picture in the DCGAN, 4 to 8 to 16 to 32 to 64. Notice the third row: when the kernel equals the stride, the output is exactly s times as big, no overlap and no padding needed.

# %% [markdown]
# **Tips and common mistakes**
#
# - A normal convolution *divides* the side by the stride and the transpose convolution *multiplies* by it. If your output side is off by one, `output_padding` is the knob.
# - `output_padding` must be smaller than the stride (or the dilation). It only adds extra cells on one side of the output; it adds no new weights.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.1. Sizes in a generator
#
# Your task: use `out_size_T` to compute (a) `out_16_32`, the output side for n = 16, k = 4, s = 2, p = 1; (b) `out_1_4`, the side for n = 1, k = 4, s = 1, p = 0 (a 1 by 1 input becoming 4 by 4); (c) `k_needed`, the kernel side that turns n = 7 into 14 with s = 2, p = 1 and no output padding. Why it matters: you will use all three when you design a generator: the same formula tells you which kernel, stride and padding give the doubling you want.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
out_16_32 = None   # output side for n = 16, k = 4, s = 2, p = 1
out_1_4 = None     # output side for n = 1, k = 4, s = 1, p = 0
k_needed = None    # kernel side that gives 14 from n = 7 with s = 2, p = 1, op = 0

# %%
if out_16_32 is None or out_1_4 is None or k_needed is None:
    print("Fill in out_16_32, out_1_4 and k_needed above, then run this cell again.")
else:
    check("out_16_32 is 32", out_16_32 == 32, "(16 - 1) * 2 - 2 + 4.")
    check("out_1_4 is 4", out_1_4 == 4, "(1 - 1) * 1 - 0 + 4.")
    check("k_needed gives 14", out_size_T(7, k_needed, 2, 1) == 14, "Solve (7 - 1) * 2 - 2 + k = 14 for k.")
    RESULTS["p2_sizes"] = (out_16_32, out_1_4, k_needed)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# out_16_32 = out_size_T(16, 4, 2, 1)   # 32
# out_1_4 = out_size_T(1, 4, 1, 0)      # 4
# k_needed = 14 - (7 - 1) * 2 + 2 * 1   # k = out - (n - 1) * s + 2p = 4
# ```
#
# The results are 32, 4 and 4. The second one is the first layer of the original DCGAN: a 1 by 1 by 100 noise tensor becomes a 4 by 4 map with a kernel of 4, stride 1 and no padding (our small generator in Part 6 uses a linear layer for this step instead).
#
# </details>

# %% [markdown]
# ### Step 2.2. The stamp view in NumPy
#
# The easiest way to *see* a transpose convolution is as stamping. Every input value multiplies the whole kernel and the scaled kernel is **added** into the output at a position that moves by `stride` cells per input cell: one stamp per input cell, overlapping stamps add up. Padding is a crop of the finished result. The block writes this with two loops, tests it on a 2 by 2 input and a 2 by 2 kernel at stride 2, and compares it with PyTorch's `F.conv_transpose2d` on three random cases.

# %%
def transpose_conv2d(x, k, stride=1, padding=0):
    """Transpose convolution of a 2-D array x with a 2-D kernel k by stamping; padding crops the result."""
    H, W = x.shape; kh, kw = k.shape
    out = np.zeros(((H - 1) * stride + kh, (W - 1) * stride + kw))
    for i in range(H):
        for j in range(W):
            out[i * stride:i * stride + kh, j * stride:j * stride + kw] += x[i, j] * k      # one stamp per input cell
    if padding:
        out = out[padding:out.shape[0] - padding, padding:out.shape[1] - padding]
    return out

x_small = np.array([ [1., 2.], [3., 4.] ])
k_small = np.array([ [1., 0.], [0., 1.] ])
print(transpose_conv2d(x_small, k_small, stride=2))

rng = np.random.RandomState(SEED)
for n, k, s, p in [(5, 3, 2, 0), (4, 4, 2, 1), (6, 3, 1, 1)]:
    x = rng.randn(n, n); ker = rng.randn(k, k)
    mine = transpose_conv2d(x, ker, s, p)
    ref = F.conv_transpose2d(torch.tensor(x)[None, None], torch.tensor(ker)[None, None], stride=s, padding=p)[0, 0].numpy()
    print(f"n={n} k={k} s={s} p={p}: output {mine.shape}, largest difference to PyTorch {np.abs(mine - ref).max():.1e}")
    assert mine.shape == ref.shape and np.allclose(mine, ref)

# %% [markdown]
# **What you should see.** You should see a 4 by 4 array first: the four input values 1, 2, 3, 4 appear at the corners of their 2 by 2 blocks, each followed by a zero because the identity-like kernel [ [1, 0], [0, 1] ] stamps its diagonal: rows `1 0 2 0 / 0 1 0 2 / 3 0 4 0 / 0 3 0 4`. With stride 2 and a 2 by 2 kernel the stamps tile the output without overlap. Then three lines, one per random case, with output sizes (11, 11), (8, 8) and (6, 6) (check them with the formula of Step 2.1) and differences around 1e-16: your two loops give the same numbers as PyTorch, up to rounding. Note that `conv_transpose2d` does not flip the kernel: the stamp is the kernel as it is.

# %% [markdown]
# **Tips and common mistakes**
#
# - Stamps *add*: forget the `+=` (write `=`) and the overlapping parts overwrite each other. That is the most common mistake.
# - With padding, PyTorch first computes the full output and then removes `padding` rows and columns on every side.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.2. Stamp with a block of ones
#
# Your task: predict by hand, then compute with `transpose_conv2d`, the result of stamping the all-ones 2 by 2 kernel on `x_small` with **stride 1**. Store the array in `out_ones`. Why it matters: with stride 1 the stamps overlap heavily, and you see how the overlaps add. Compare your hand prediction with the program.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
out_ones = None   # a 3 x 3 array: transpose_conv2d(x_small, ones, stride=1)

# %%
if out_ones is None:
    print("Fill in out_ones above, then run this cell again.")
else:
    expected = np.array([ [1, 3, 2], [4, 10, 6], [3, 7, 4] ])
    check("out_ones has shape (3, 3)", np.shape(out_ones) == (3, 3), "(2 - 1) * 1 + 2 = 3.")
    check("out_ones has the right values", np.allclose(out_ones, expected), "The centre cell receives all four stamps: 1 + 2 + 3 + 4 = 10.")
    RESULTS["p2_ones_centre"] = float(np.asarray(out_ones)[1, 1])

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# out_ones = transpose_conv2d(x_small, np.ones((2, 2)), stride=1)
# print(out_ones)
# ```
#
# The array is `1 3 2 / 4 10 6 / 3 7 4`. The centre cell is touched by all four stamps and holds 1 + 2 + 3 + 4 = 10; the corners are touched by one stamp each. Overlap is where the "adding" of the transpose convolution happens: remember it for the checkerboard of Step 2.5.
#
# </details>

# %% [markdown]
# ### Step 2.3. The transpose of the connectivity
#
# Why is it called a *transpose* convolution? Because a convolution is a **matrix multiplication**: y = C x, where C is a sparse matrix whose rows contain the kernel, shifted by one stride at each row. The transpose convolution multiplies by the **transpose** matrix C^T, which maps the short vector back to the long one. A one-dimensional example shows it without clutter: kernel (1, 2, 3), input of length 5, stride 1, no padding, so the output has length 3. Build the 3 by 5 matrix C, check that it reproduces `F.conv1d`, print C^T (5 by 3) and check that `F.conv_transpose1d` equals C^T y.

# %%
kern = np.array([1., 2., 3.])
x5 = np.array([1., 2., 3., 4., 5.])
C = np.zeros((3, 5))
for i in range(3):
    C[i, i:i + 3] = kern                         # row i holds the kernel, shifted by i
y3 = C @ x5
ref = F.conv1d(torch.tensor(x5)[None, None], torch.tensor(kern)[None, None])[0, 0].numpy()
print("C (3 x 5):\n", C)
print("conv(x) = C x =", y3, "  PyTorch conv1d:", ref)
print("C transposed (5 x 3):\n", C.T)

up = F.conv_transpose1d(torch.tensor(y3)[None, None], torch.tensor(kern)[None, None])[0, 0].numpy()
print("C^T y =", C.T @ y3, "  PyTorch conv_transpose1d:", up)
assert np.allclose(y3, ref) and np.allclose(up, C.T @ y3) and len(up) == 5

# %% [markdown]
# **What you should see.** You should see C as three rows `1 2 3 0 0 / 0 1 2 3 0 / 0 0 1 2 3`, the convolution result `[14. 20. 26.]` (identical to PyTorch's `conv1d`), and C^T with five rows of three numbers: `1 0 0 / 2 1 0 / 3 2 1 / 0 3 2 / 0 0 3`. Multiplying C^T by the length-3 vector gives a length-5 vector, `[14. 48. 108. 112. 78.]`, and `conv_transpose1d` returns exactly the same. Two things to remember. First, C^T y is **not** the original x = (1, 2, 3, 4, 5) (it is not an inverse): it only has the right *size* and the same connectivity, with the same kernel weights on the same connections. Second, the direction of the connections is reversed: the forward convolution connects 3 neighbouring inputs to each output, and the transpose connects each input to 3 neighbouring outputs, which is exactly the stamp of Step 2.2.

# %% [markdown]
# **Tips and common mistakes**
#
# - A transpose convolution has the **same weights** as the convolution whose matrix it transposes; only the direction of the connections changes. A convolution layer and the transpose layer that undoes its size change are *two different layers* with separate weights in a network.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.3. The matrix for stride 2
#
# Your task: build the matrix `C2` of the convolution with the same kernel, an input of length 7 and **stride 2** (it has 3 rows and 7 columns; row i starts at column 2i). Then store the vector `back = C2.T @ y2` for `y2 = C2 @ x7`, where `x7` is given. Why it matters: the transposed stride-2 matrix turns a vector of length 3 into one of length 7: it upsamples by a factor of about two, which is the whole point of the layer in a generator.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
x7 = np.arange(1., 8.)
C2 = None     # a 3 x 7 array
back = None   # C2.T @ (C2 @ x7), a vector of length 7

# %%
if C2 is None or back is None:
    print("Fill in C2 and back above, then run this cell again.")
else:
    y_ref = F.conv1d(torch.tensor(x7)[None, None], torch.tensor(kern)[None, None], stride=2)[0, 0].numpy()
    up_ref = F.conv_transpose1d(torch.tensor(y_ref)[None, None], torch.tensor(kern)[None, None], stride=2)[0, 0].numpy()
    check("C2 has shape (3, 7)", np.shape(C2) == (3, 7), "Three outputs, seven inputs.")
    check("C2 x7 equals the stride-2 convolution", np.allclose(np.asarray(C2) @ x7, y_ref), "Row i holds the kernel starting at column 2 * i.")
    check("C2 transposed equals conv_transpose1d", np.allclose(back, up_ref) and len(back) == 7, "Multiply the transposed matrix with the output of the convolution.")
    RESULTS["p2_back"] = [float(v) for v in np.asarray(back)]

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# C2 = np.zeros((3, 7))
# for i in range(3):
#     C2[i, 2 * i:2 * i + 3] = kern
# y2 = C2 @ x7
# back = C2.T @ y2
# print(C2); print(back)
# ```
#
# The vector `back` is [14.0, 28.0, 68.0, 52.0, 116.0, 76.0, 114.0]. Cells 2 and 4 (counting from 0) are touched by two connections each (they are where two neighbouring kernels overlap), the others by one. That uneven overlap is the seed of the checkerboard problem.
#
# </details>

# %% [markdown]
# ### Step 2.4. The link to Week 4: the input gradient
#
# In Week 4 you computed how a convolution y = conv(x, K) passes the gradient back to its input. Let G be the gradient that arrives from above (the same shape as y). The gradient with respect to the input, dx, is the **transpose convolution of G with the same kernel K**. This is the real reason for the name: the backward pass of a convolution is a transpose convolution. The block checks it for the 4 by 4 input and 2 by 2 kernel of Week 4: dx from PyTorch's autograd against `F.conv_transpose2d(G, K)`.

# %%
x4 = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4).requires_grad_(True)
K = torch.tensor([ [1., 2.], [3., 4.] ], dtype=torch.float64).reshape(1, 1, 2, 2)
y = F.conv2d(x4, K)                                        # 3 x 3 output
G = torch.tensor([ [1., 0., 2.], [0., 1., 0.], [3., 0., 1.] ], dtype=torch.float64).reshape(1, 1, 3, 3)   # an upstream gradient
y.backward(G)                                              # autograd computes dx
dx_auto = x4.grad[0, 0]
dx_T = F.conv_transpose2d(G, K)[0, 0]                      # transpose convolution of G with K
print("dx from autograd:\n", dx_auto.numpy())
print("dx from conv_transpose2d:\n", dx_T.numpy())
print("equal:", torch.allclose(dx_auto, dx_T))
assert torch.allclose(dx_auto, dx_T) and dx_T.shape == (4, 4)

# %% [markdown]
# **What you should see.** You should see two 4 by 4 arrays that are identical, row for row (`1 2 2 4 / 3 5 8 8 / 3 9 5 2 / 9 12 3 4`), and `equal: True`. The shape of the gradient is 4 by 4, the shape of the *input*: the transpose convolution maps the 3 by 3 gradient back to the input size. So the layers you meet in a generator are not exotic: you used this operation every time a convolution layer was trained. The "transpose convolution" that upsamples in the forward pass is the same operation that carries gradients backward through a downsampling convolution.

# %% [markdown]
# **Tips and common mistakes**
#
# - `y.backward(G)` is the same as `(y * G).sum().backward()`: it starts the backward pass with the gradient G instead of the number 1.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.4. A size that does not fit
#
# Your task: take a 6 by 6 input and a 3 by 3 kernel with **stride 2** (the convolution output is 2 by 2). Compute `dx_hand` with `F.conv_transpose2d` so that it has the input's shape (6, 6) and equals the autograd gradient. Which `output_padding` do you need? Store it in `op_needed`. Why it matters: stride 2 loses information about the size: inputs of side 5 and 6 both give a 2 by 2 output. This is the ambiguity that `output_padding` resolves, and you meet it whenever you invert a strided convolution.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
op_needed = None   # the output_padding that gives a 6 x 6 result
dx_hand = None     # F.conv_transpose2d(G2, K2, stride=2, output_padding=op_needed)

# %%
g2 = torch.Generator().manual_seed(SEED)
x6 = torch.randn(1, 1, 6, 6, generator=g2, dtype=torch.float64).requires_grad_(True)
K2 = torch.randn(1, 1, 3, 3, generator=g2, dtype=torch.float64)
G2 = torch.randn(1, 1, 2, 2, generator=g2, dtype=torch.float64)
F.conv2d(x6, K2, stride=2).backward(G2)
if op_needed is None or dx_hand is None:
    print("Fill in op_needed and dx_hand above (use G2 and K2 from this cell), then run this cell again.")
else:
    check("op_needed is 1", op_needed == 1, "Without it the result is 5 x 5: (2 - 1) * 2 + 3 = 5. You need 6.")
    check("dx_hand equals the autograd gradient", torch.allclose(torch.as_tensor(dx_hand).reshape(6, 6), x6.grad[0, 0]), "Use the same kernel K2 and stride 2.")
    RESULTS["p2_op"] = int(op_needed)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# op_needed = 1
# dx_hand = F.conv_transpose2d(G2, K2, stride=2, output_padding=op_needed)[0, 0]
# ```
#
# With `output_padding=1` the result is 6 by 6 and equals the autograd gradient (last row and column of the 6 by 6 input receive no contribution from the convolution: they were never used because the last window ended earlier, so their gradient is zero). Without it you would get a 5 by 5 array.
#
# </details>

# %% [markdown]
# ### Step 2.5. Overlap and checkerboards
#
# The stamps overlap, and overlap adds. If every output cell is covered by the **same number** of stamps, the output is smooth. If some cells are covered twice and neighbours once, you see a regular pattern of stripes or a **checkerboard**, even when the input is perfectly flat. The block counts, for each output cell, how many kernel copies reach it (feed a 3 by 3 input of ones through an all-ones kernel: each output cell then holds exactly the number of overlapping stamps) for two layers: (k = 3, s = 2, p = 0) and (k = 4, s = 2, p = 1). Then it feeds a *constant* 8 by 8 picture of ones through both layers and draws heat maps.

# %%
def overlap(n, k, s, p):
    """Number of kernel copies that reach each output cell (n x n input of ones, k x k kernel of ones)."""
    return F.conv_transpose2d(torch.ones(1, 1, n, n), torch.ones(1, 1, k, k), stride=s, padding=p)[0, 0].int().numpy()

c3 = overlap(3, 3, 2, 0)      # 7 x 7
c4 = overlap(3, 4, 2, 1)      # 6 x 6
print("k = 3, s = 2, p = 0:"); [print("  " + "".join(str(v) for v in row)) for row in c3]
print("k = 4, s = 2, p = 1:"); [print("  " + "".join(str(v) for v in row)) for row in c4]
assert ["".join(str(v) for v in r) for r in c3] == ["1121211", "1121211", "2242422", "1121211", "2242422", "1121211", "1121211"]
assert (c4[1:-1, 1:-1] == 4).all() and c4.min() == 1

flat3 = overlap(8, 3, 2, 0); flat4 = overlap(8, 4, 2, 1)       # a constant picture through the same layers
print("middle row, k = 3:", flat3[8])
print("middle row, k = 4:", flat4[8])
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a_, A, t in [(ax[0], flat3, "k = 3, s = 2: stripes"), (ax[1], flat4, "k = 4, s = 2, p = 1: even inside")]:
    im = a_.imshow(A, cmap="viridis"); a_.set_title(t, fontsize=9); a_.axis("off"); plt.colorbar(im, ax=a_, fraction=0.046)
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** You should see the two overlap maps. For k = 3, s = 2 the 7 by 7 map is `1121211 / 1121211 / 2242422 / 1121211 / 2242422 / 1121211 / 1121211`: the counts alternate 1, 2, 1, 2 along every row and column and reach 4 where a double row meets a double column. That is **uneven overlap**, and the assertion confirms the lesson's numbers. For k = 4, s = 2, p = 1 the 6 by 6 map has a block of 4s in the middle (all 16 inner cells are 4), edge cells with 2 and corners with 1. The second half of the block shows what this means in practice: a perfectly *flat* picture of ones leaves the first layer as the pattern `[np.int32(2), np.int32(2), np.int32(4), np.int32(2), np.int32(4), np.int32(2), np.int32(4), np.int32(2), np.int32(4), np.int32(2), np.int32(4), np.int32(2), np.int32(4), np.int32(2), np.int32(4), np.int32(2), np.int32(2)]` along the middle row (a regular 2, 4, 2, 4 beat: stripes, in both directions: a checkerboard of high and low), while the second layer gives `[np.int32(2), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(4), np.int32(2)]` (a constant 4 in the interior, with 2 at the two ends). The heat maps show the stripes on the left and a smooth plateau on the right. The rule of thumb: **the kernel size should be divisible by the stride** (k = 4 with s = 2), or the stamps overlap unevenly. This is why the DCGAN uses 4 by 4 kernels with stride 2, and why a generator with 3 by 3 kernels and stride 2 often shows a faint checkerboard, as you will measure in Part 6.

# %% [markdown]
# **Tips and common mistakes**
#
# - The checkerboard comes from the *layer*, not from the data: it appears even when the input is constant. Training can learn to cancel it, but it is better not to create it.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.5. Which kernels are even?
#
# Your task: for the stride 2 and no padding, loop over the kernel sides k = 2, 3, 4, 5, 6, build `overlap(8, k, 2, 0)` and keep the k for which the **interior** of the map (cut off k cells on every side) has the same count everywhere. Store the list in `even_ks`. Why it matters: the pattern you find (the even kernels are exactly the multiples of the stride) is the design rule used for upsampling layers.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
even_ks = None   # a list of kernel sides, e.g. [2, 3]

# %%
if even_ks is None:
    print("Fill in even_ks above, then run this cell again.")
else:
    check("even_ks is [2, 4, 6]", list(even_ks) == [2, 4, 6], "A kernel is even when k is divisible by the stride. Look at the min and the max of the interior.")
    RESULTS["p2_even_ks"] = list(even_ks)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# even_ks = []
# for k in [2, 3, 4, 5, 6]:
#     m = overlap(8, k, 2, 0)[k:-k, k:-k]            # interior only
#     if m.min() == m.max():
#         even_ks.append(k)
# ```
#
# The list is [2, 4, 6]: exactly the multiples of the stride 2. For k = 3 and k = 5 the interior alternates between 2 values.
#
# </details>

# %% [markdown]
# ### Step 2.6. Learnable upsampling
#
# A transpose convolution has weights, so it can *learn* an upsampler instead of using a fixed one. As a test, train a single `nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)` to imitate bilinear x2 upsampling: the targets come from `F.interpolate(mode="bilinear", align_corners=False)` applied to random 8 by 8 inputs, the loss is the mean squared error **inside the picture** (the outer frame of 2 pixels is cut off: at the frame, bilinear interpolation copies the edge pixels while a transpose convolution sees zeros outside the picture, so the two cannot agree there, and we do not want that disagreement to distort the kernel), the optimiser is Adam with a learning rate of 0.05, 400 steps. The exact bilinear kernel for this layer is the outer product of (1, 3, 3, 1) with itself, divided by 16. The block prints the learned 4 by 4 kernel next to it and measures the final error on a fresh batch, overall and with the outer frame of 2 pixels cut off.

# %%
torch.manual_seed(SEED)
up = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)
opt = torch.optim.Adam(up.parameters(), lr=0.05)
for step in range(400):
    x = torch.rand(32, 1, 8, 8)
    target = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    loss = F.mse_loss(up(x)[:, :, 2:-2, 2:-2], target[:, :, 2:-2, 2:-2])   # compare the inside only (see the text about the frame)
    opt.zero_grad(); loss.backward(); opt.step()

bil = np.outer([1, 3, 3, 1], [1, 3, 3, 1]) / 16                    # the exact bilinear kernel
learned = up.weight.detach()[0, 0].numpy()
print("learned kernel:\n", np.round(learned, 3))
print("bilinear kernel (outer product of 1 3 3 1, divided by 16):\n", np.round(bil, 3))

g = torch.Generator().manual_seed(123)
xb = torch.rand(64, 1, 8, 8, generator=g)                           # a fresh batch
tb = F.interpolate(xb, scale_factor=2, mode="bilinear", align_corners=False)
with torch.no_grad():
    pb = up(xb)
mse_all = float(F.mse_loss(pb, tb)); mse_in = float(F.mse_loss(pb[:, :, 2:-2, 2:-2], tb[:, :, 2:-2, 2:-2]))
print(f"final MSE, whole picture: {mse_all:.2e}   inside (frame of 2 cut off): {mse_in:.2e}   kernel difference (largest): {np.abs(learned - bil).max():.3f}")
assert np.abs(learned - bil).max() < 0.05 and mse_in < 1e-3

# %% [markdown]
# **What you should see.** You should see the learned kernel and the bilinear kernel printed one under the other, and they are the same to three decimals: `0.062 0.188 0.188 0.062` in the first row, `0.188 0.562 0.562 0.188` in the second (1/16 = 0.0625, 3/16 = 0.1875, 9/16 = 0.5625). The largest difference is 6.0e-08. Nobody told the layer what bilinear interpolation is: it found the 16 weights by gradient descent, from random starting values, in 400 steps of a tiny regression problem. That is what "learnable upsampling" means: the layer starts as a random upsampler and training turns it into a useful one. The error inside the picture is 9.6e-16 (zero up to rounding). On the whole picture the error is 5.03e-03 and it all comes from the 2-pixel frame, where the transpose convolution has no neighbours outside the picture and bilinear interpolation repeats the edge pixels. In a GAN this kind of border effect is why generated pictures sometimes have odd edges.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.6. The hand-set kernel
#
# Your task: copy the exact bilinear kernel `bil` into a fresh `nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)` (name it `up_hand`) and measure its MSE on the fresh batch `xb`, `tb` of the step: store it in `mse_hand`. Why it matters: a learned kernel is only worth the trouble if it reaches what a hand-designed one achieves. Here you set the best possible kernel by hand and see how close the learned one came.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
mse_hand = None   # MSE of the hand-set bilinear kernel on (xb, tb)

# %%
if mse_hand is None:
    print("Fill in mse_hand above, then run this cell again.")
else:
    check("mse_hand is tiny (inside the frame it is exact)", 0 < mse_hand < 0.01, "Copy bil into up_hand.weight with torch.no_grad() and .copy_().")
    check("the learned kernel is close to the hand-set one", abs(mse_hand - mse_all) < 0.002, "Both should have about the same error: the frame is the difference to the true bilinear.")
    RESULTS["p2_mse_hand"] = float(mse_hand)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# up_hand = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)
# with torch.no_grad():
#     up_hand.weight.copy_(torch.tensor(bil, dtype=torch.float32).reshape(1, 1, 4, 4))
#     mse_hand = float(F.mse_loss(up_hand(xb), tb))
# ```
#
# The hand-set kernel gives an MSE of 5.03e-03 on the whole picture, the same as the learned one (5.03e-03): both are exact inside and both are off at the frame. The learned kernel therefore reached what a human expert would have set by hand. The difference in a GAN is that nobody knows the best kernel for *generating* pictures, so it has to be learned, and in a GAN the kernels are learned from the discriminator's feedback, not from a bilinear target.
#
# </details>

# %% [markdown]
# ### Step 2.7. Three upsamplers on a real picture
#
# Finally compare three ways to get from 8 by 8 to 16 by 16 on a real shape picture: **nearest neighbour** (copy each pixel into a 2 by 2 block), **bilinear** interpolation, and the **transpose convolution with the bilinear kernel** that you set by hand. First downsample a clean 16 by 16 picture by averaging 2 by 2 blocks, then upsample it back and measure the mean squared error to the original.

# %%
Xp, yp = make_shapes(1, noise=0.0, seed=SEED)                  # one clean picture
orig = torch.tensor(Xp)                                        # (1, 1, 16, 16)
small = F.avg_pool2d(orig, 2)                                  # 8 x 8
up_near = F.interpolate(small, scale_factor=2, mode="nearest")
up_bil = F.interpolate(small, scale_factor=2, mode="bilinear", align_corners=False)
convT_hand = nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)     # the layer, with the kernel set by hand
with torch.no_grad():
    convT_hand.weight.copy_(torch.tensor(bil, dtype=torch.float32).reshape(1, 1, 4, 4))
    up_t = convT_hand(small)
for name, U in [("nearest", up_near), ("bilinear", up_bil), ("transpose conv (bilinear kernel)", up_t)]:
    print(f"{name:34s} MSE to the original: {float(F.mse_loss(U, orig)):.4f}")
show_grid(torch.cat([orig, small.repeat_interleave(2, 2).repeat_interleave(2, 3), up_near, up_bil, up_t]).numpy(),
          [f"original ({SHAPE_NAMES[yp[0]]})", "8 x 8 (enlarged)", "nearest", "bilinear", "transpose conv"], cols=5)

# %% [markdown]
# **What you should see.** You should see three lines and a row of five pictures: the clean original, its 8 by 8 version (enlarged so that you can see the blocks), and the three upsampled results. The MSE to the original is 0.0186 for nearest neighbour, 0.0207 for bilinear and 0.0207 for the transpose convolution with the bilinear kernel. Two lessons. First, the last two are the **same number** (up to the fourth decimal): a transpose convolution with the bilinear kernel *is* bilinear upsampling, and it differs only at the frame, where this picture is black anyway. Second, no upsampler recovers the original: the thin one-pixel line was averaged away when we downsampled, and upsampling cannot bring back information that is gone. In this example nearest neighbour even has the lowest MSE, although it looks blocky, because the MSE compares pixel by pixel and rewards sharp (if misplaced) blocks over smooth blur: MSE is not a good judge of how a picture *looks*, a warning that returns in Part 7. The point of the transpose convolution is not that it beats bilinear on a fixed task: it can **learn** to make something better than any fixed rule for the task at hand.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.7. Average over many pictures
#
# Your task: repeat the comparison for 100 clean pictures (`make_shapes(100, noise=0.0, seed=SEED)`) and store the mean MSE of each method in the dictionary `mean_mse` with the keys `"nearest"`, `"bilinear"` and `"convT"`. Why it matters: one picture can be lucky. The average over 100 pictures of five different shapes gives a fairer ranking, which is the basis of the choice you make when you design a generator.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
mean_mse = {"nearest": None, "bilinear": None, "convT": None}   # three numbers

# %%
if any(v is None for v in mean_mse.values()):
    print("Fill in the three values of mean_mse above, then run this cell again.")
else:
    check("all three are positive numbers", all(0 < v < 0.2 for v in mean_mse.values()), "Average F.mse_loss over the 100 pictures, with the originals as targets.")
    check("bilinear and transpose conv (same kernel) agree inside, so they are close", abs(mean_mse["bilinear"] - mean_mse["convT"]) < 0.01, "The only difference between them is the frame of the picture.")
    RESULTS["p2_upsamplers"] = {k: round(float(v), 4) for k, v in mean_mse.items()}

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# X100, _ = make_shapes(100, noise=0.0, seed=SEED)
# o100 = torch.tensor(X100); s100 = F.avg_pool2d(o100, 2)
# with torch.no_grad():
#     outs = {"nearest": F.interpolate(s100, scale_factor=2, mode="nearest"),
#             "bilinear": F.interpolate(s100, scale_factor=2, mode="bilinear", align_corners=False),
#             "convT": convT_hand(s100)}
# mean_mse = {k: float(F.mse_loss(v, o100)) for k, v in outs.items()}
# print({k: round(v, 4) for k, v in mean_mse.items()})
# ```
#
# Averaged over 100 pictures the MSE is 0.0297 for nearest, 0.0333 for bilinear and 0.0333 for the transpose convolution: the ranking of the single picture holds, and the last two are almost identical because they are the same operation. A learned kernel can be better than any of them for the task it is trained on, which is why generators use transpose convolutions.
#
# </details>

# %% [markdown]
# ## Part 3. The GAN game by hand (20 min)
#
# *Lesson section: "GAN game".* A **GAN** (generative adversarial network) is a game between two players. The **generator** G is a forger: it turns noise into fake examples. The **discriminator** D is a detective: it receives an example, real or fake, and answers with a number between 0 and 1, its belief that the example is real. The detective is trained to tell them apart, the forger is trained to fool the detective. There is no "correct picture" anywhere: the detective's opinion *is* the training signal that Part 1 was missing. In this part you compute everything by hand, with a few numbers and a few lines of NumPy and autograd, before Part 4 trains real networks: the value of the game, the detective's loss, why the forger's loss can be written in two ways (one of which stops learning), the best possible detective, and one real step of training.

# %% [markdown]
# ### ✏️ Imagine first
#
# The value of the game is V = ln D(real) + ln(1 - D(fake)): the detective wants it large, the forger wants it small. Suppose the forger is so good that the detective is completely fooled and answers 0.5 for **every** example, real or fake. Before you compute anything, guess the value of V (a number: 0? -1? something else?). You compute it in Step 3.1.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: the value of the game when the detective says 0.5 for everything
my_guess_V = None   # a number

# %% [markdown]
# ### Step 3.1. The value function
#
# For one real example x and one fake example G(z) the **value** of the game is V = ln D(x) + ln(1 - D(G(z))). Here D(x) is the detective's belief that the real example is real, and D(G(z)) its belief that the fake is real. The detective wants V as **large** as possible (D(real) near 1, D(fake) near 0: both logarithms near 0). The forger wants it as **small** as possible: it can only act on D(fake), and ln(1 - D(fake)) becomes very negative when D(fake) is near 1. The block writes `value`, checks the two numbers of the lesson and draws V as a heat map over all pairs (D(real), D(fake)).

# %%
def value(d_real, d_fake):
    """The GAN value V = ln D(real) + ln(1 - D(fake)) for one pair of detective outputs."""
    return np.log(d_real) + np.log(1 - d_fake)

print("D(real) = 0.9, D(fake) = 0.2 :", round(float(value(0.9, 0.2)), 4))
print("D(real) = 0.5, D(fake) = 0.5 :", round(float(value(0.5, 0.5)), 4), "   minus ln 4 =", round(-math.log(4), 4))
assert abs(value(0.9, 0.2) - (-0.3285)) < 1e-4 and abs(value(0.5, 0.5) + math.log(4)) < 1e-12

g = np.linspace(0.01, 0.99, 99)
DR, DF = np.meshgrid(g, g)                             # D(real) along x, D(fake) along y
V = value(DR, DF)
plt.figure(figsize=(5.2, 4.2))
plt.imshow(V, origin="lower", extent=[0.01, 0.99, 0.01, 0.99], cmap="viridis", vmin=-5, aspect="auto")
plt.colorbar(label="V")
plt.contour(DR, DF, V, levels=[-3, -2, -1.386, -1, -0.5], colors="w", linewidths=0.7)
plt.scatter([0.9, 0.5], [0.2, 0.5], c=["r", "k"]); plt.xlabel("D(real)"); plt.ylabel("D(fake)"); plt.title("the value of the game")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** You should see two numbers, **-0.3285** for the pair (0.9, 0.2) and **-1.3863** for (0.5, 0.5), which equals minus ln 4 (the last assertion passes). So the answer to the Imagine question is -1.386, not 0: when the detective is fooled completely it says 0.5 for everything, ln 0.5 is -0.693, and two such terms add up to -1.386. The heat map shows the landscape on which the game is played. The bright corner at the bottom right (D(real) near 1, D(fake) near 0) is where the detective wins (V near 0); the dark band at the top (D(fake) near 1) is where the forger wins (V very negative). The white contour marked -1.386 passes through the black dot at (0.5, 0.5): the **equilibrium**, the point where the detective can no longer tell the two apart. It is a *minimax* game: the detective tries to push V up, the forger tries to push it down, and a perfect forger holds the value at -1.386, no lower and no higher.

# %% [markdown]
# **Tips and common mistakes**
#
# - The value is always negative or zero because probabilities are at most 1 and logarithms of numbers below 1 are negative. The best the detective can do is a value near 0.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.1. Who is winning?
#
# Your task: compute `v_detective_wins`, the value when D(real) = 0.99 and D(fake) = 0.01, and `v_forger_wins`, the value when D(real) = 0.9 and D(fake) = 0.99. Why it matters: reading a value is a skill you need in Part 4, where the training curves show these numbers. A value near 0 means the detective is winning; a very negative one means that the forger is.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
v_detective_wins = None   # D(real) = 0.99, D(fake) = 0.01
v_forger_wins = None      # D(real) = 0.9, D(fake) = 0.99

# %%
if v_detective_wins is None or v_forger_wins is None:
    print("Fill in v_detective_wins and v_forger_wins above, then run this cell again.")
else:
    check("v_detective_wins is about -0.0201", abs(v_detective_wins - 2 * math.log(0.99)) < 1e-6, "value(0.99, 0.01).")
    check("v_forger_wins is about -4.71", abs(v_forger_wins - (math.log(0.9) + math.log(0.01))) < 1e-6, "value(0.9, 0.99).")
    RESULTS["p3_values"] = (round(float(v_detective_wins), 4), round(float(v_forger_wins), 4))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# v_detective_wins = float(value(0.99, 0.01))
# v_forger_wins = float(value(0.9, 0.99))
# ```
#
# The values are -0.0201 (almost the maximum 0: the detective wins) and -4.711 (the forger wins: ln 0.01 alone is -4.6). Compare both with the equilibrium -1.386.
#
# </details>

# %% [markdown]
# ### Step 3.2. The detective's loss is binary cross-entropy
#
# In practice nobody maximises V; we **minimise a loss**. The detective's loss is the *binary cross-entropy* (Week 2) of a two-class problem: real examples have the label 1, fake ones the label 0. Written out, L_D = -mean(ln D(real)) - mean(ln(1 - D(fake))): minus the value, with averages over the batch. The block takes four real examples with detective outputs (0.9, 0.8, 0.6, 0.95) and four fakes with outputs (0.1, 0.3, 0.5, 0.05), computes the loss by hand and with `F.binary_cross_entropy` on all eight (which takes one mean over the eight, so it is half the sum of the two means), and the accuracy of the detective at the threshold 0.5.

# %%
d_real = np.array([0.9, 0.8, 0.6, 0.95])
d_fake = np.array([0.1, 0.3, 0.5, 0.05])
loss_real = -np.log(d_real).mean()                 # the detective should say 1 on real examples
loss_fake = -np.log(1 - d_fake).mean()             # ... and 0 on fake ones
L_D = loss_real + loss_fake
print(f"mean -ln D(real) = {loss_real:.4f}   mean -ln(1 - D(fake)) = {loss_fake:.4f}   L_D = {L_D:.4f}")

outputs = torch.tensor(np.concatenate([d_real, d_fake]))
labels = torch.tensor([1., 1., 1., 1., 0., 0., 0., 0.], dtype=torch.float64)
bce = float(F.binary_cross_entropy(outputs, labels))
print(f"F.binary_cross_entropy over 8 examples = {bce:.4f}   (= L_D / 2 = {L_D / 2:.4f})")
acc = float(((outputs > 0.5).double() == labels).double().mean())
print("accuracy at threshold 0.5:", acc)
assert abs(bce - L_D / 2) < 1e-9 and acc == 1.0

# %% [markdown]
# **What you should see.** You should see the two parts of the loss, 0.2227 for the real examples and 0.3016 for the fake ones, adding up to L_D = 0.5243. PyTorch's `binary_cross_entropy` prints 0.2621, exactly half of it: the only difference is that it averages over the eight examples at once instead of two separate means. Both conventions are used in the literature, and the factor of two never changes where the minimum is. The accuracy at the threshold 0.5 (the rule "above 0.5 means real") is **1.0**: all four real examples got a value above 0.5, and all four fakes are at or below 0.5 (the fake with the output 0.5 exactly is a coin flip that the strict rule happens to count as "fake"). Notice that the accuracy is perfect while the loss is still 0.52: the loss also sees *how sure* the detective is, and it is only moderately sure on the examples 0.6 and 0.5. This is why we train on the loss and not on the accuracy. Keep in mind the loss of a lazy detective that answers 0.5 for everything: its binary cross-entropy is ln 2 = 0.693. In Part 4 you will see the detective's loss sit near 0.69 when the game is balanced.

# %% [markdown]
# **Tips and common mistakes**
#
# - `F.binary_cross_entropy` expects probabilities between 0 and 1. If your network returns *logits* (numbers before the sigmoid), use `F.binary_cross_entropy_with_logits`, which is also numerically safer. Part 4 uses that one.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.2. The lazy and the confidently wrong detective
#
# Your task: with `F.binary_cross_entropy` and the labels of this step compute the loss `loss_lazy` of a detective that answers 0.5 on all eight examples, and the loss `loss_wrong` of a detective that is confidently wrong: it answers 0.1 on the real examples and 0.9 on the fake ones. Why it matters: these two numbers frame every discriminator loss curve that you will see in Part 4: 0.69 means "cannot tell", and a value above 2 means "sure and wrong".
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
loss_lazy = None    # the detective says 0.5 everywhere
loss_wrong = None   # the detective says 0.1 on real examples and 0.9 on fake ones

# %%
if loss_lazy is None or loss_wrong is None:
    print("Fill in loss_lazy and loss_wrong above, then run this cell again.")
else:
    check("loss_lazy is ln 2 = 0.6931", abs(loss_lazy - math.log(2)) < 1e-6, "Every term is -ln 0.5.")
    check("loss_wrong is -ln 0.1 = 2.3026", abs(loss_wrong - (-math.log(0.1))) < 1e-6, "Every example is wrong with probability 0.9 on the right label 0.1.")
    RESULTS["p3_losses"] = (round(float(loss_lazy), 4), round(float(loss_wrong), 4))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# loss_lazy = float(F.binary_cross_entropy(torch.full((8,), 0.5, dtype=torch.float64), labels))
# wrong = torch.tensor([0.1] * 4 + [0.9] * 4, dtype=torch.float64)
# loss_wrong = float(F.binary_cross_entropy(wrong, labels))
# ```
#
# The losses are 0.6931 (ln 2) and 2.3026 (-ln 0.1). A detective that merely guesses scores 0.69; one that is sure and wrong is punished more than three times harder.
#
# </details>

# %% [markdown]
# ### Step 3.3. Saturating versus non-saturating
#
# Now the forger. The original game says: minimise ln(1 - D(G(z))). Let u be the detective's **logit** for a fake example (the number before the sigmoid), so D = sigma(u). The original **saturating** generator loss is ln(1 - sigma(u)), and the alternative proposed in the original paper, the **non-saturating** loss, is -ln sigma(u): "make the detective say real" instead of "make it not say fake". Both have the same goal, but their gradients with respect to u differ: -sigma(u) for the first, sigma(u) - 1 for the second. The block lets **autograd** compute both gradients for six values of u and asserts that they equal these formulas, then plots the two gradients.

# %%
u = torch.tensor([-8., -6., -4., -2., 0., 2.], dtype=torch.float64, requires_grad=True)
s = torch.sigmoid(u)
loss_sat = torch.log(1 - s)            # the original loss for the forger (to be minimised)
loss_ns = -torch.log(s)                # the non-saturating loss
g_sat = torch.autograd.grad(loss_sat.sum(), u, retain_graph=True)[0]
g_ns = torch.autograd.grad(loss_ns.sum(), u)[0]
print(f"{'u':>4s} {'sigma(u)':>9s} {'ln(1-s)':>9s} {'-ln s':>8s} {'grad sat':>10s} {'grad non-sat':>13s}")
for i in range(len(u)):
    print(f"{float(u[i]):4.0f} {float(s[i]):9.4f} {float(loss_sat[i]):9.4f} {float(loss_ns[i]):8.4f} {float(g_sat[i]):10.4f} {float(g_ns[i]):13.4f}")
assert torch.allclose(g_sat, -s.detach()) and torch.allclose(g_ns, s.detach() - 1)

uu = np.linspace(-8, 8, 200); ss = 1 / (1 + np.exp(-uu))
plt.figure(figsize=(6, 3.4))
plt.plot(uu, -ss, label="saturating: -sigma(u)"); plt.plot(uu, ss - 1, label="non-saturating: sigma(u) - 1")
plt.axvline(-6, color="grey", ls=":"); plt.xlabel("u = detective's logit for a fake"); plt.ylabel("gradient of the forger's loss w.r.t. u")
plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** You should see a table with six rows and one plot. Look at the row u = -6, where the detective is already sure that the example is fake (sigma(u) = 0.0025): the gradient of the original loss is **-0.0025**, almost zero, while the gradient of the non-saturating loss is **-0.9975**, nearly the maximum. The assertion confirms the formulas -sigma(u) and sigma(u) - 1 with autograd. The plot shows the whole story: the gradient of the original loss (the blue curve) is flat near zero on the left, where u is very negative, which is exactly where a **young** forger finds itself, because at the start of training its fakes are bad and the detective is sure about them. A gradient near zero means that the forger **cannot learn**: the loss has *saturated*. The non-saturating loss has its strongest gradient exactly there. In the middle (u = 0) both are about 0.5 in size. On the right the roles swap, but a forger that is already winning needs a strong push less. In practice, nearly every GAN is trained with the non-saturating loss, and you will compare the two on a real training run in Part 4.

# %% [markdown]
# **Tips and common mistakes**
#
# - Do not confuse the two losses: in the code of Part 4, the non-saturating loss is simply the binary cross-entropy of the fake examples with the label **1** ("pretend that they are real").

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.3. Gradients at two other places
#
# Your task: compute with autograd the gradients of both losses at u = -10 (store them in `g_sat_m10` and `g_ns_m10`, as floats) and at u = +2 decide which of the two gradients is larger in size: store the string `"sat"` or `"ns"` in `bigger_at_2`. Why it matters: at u = -10 you see how extreme the vanishing gradient can get; at u = +2 you see that the non-saturating loss is gentler when the forger is already winning.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
g_sat_m10 = None    # gradient of ln(1 - sigma(u)) at u = -10
g_ns_m10 = None     # gradient of -ln sigma(u) at u = -10
bigger_at_2 = None  # "sat" or "ns": which gradient is larger in size at u = +2

# %%
if g_sat_m10 is None or g_ns_m10 is None or bigger_at_2 is None:
    print("Fill in g_sat_m10, g_ns_m10 and bigger_at_2 above, then run this cell again.")
else:
    sg = 1 / (1 + math.exp(10))
    check("g_sat_m10 is about -0.0000454", abs(g_sat_m10 + sg) < 1e-7, "The gradient is -sigma(u).")
    check("g_ns_m10 is about -0.99995", abs(g_ns_m10 - (sg - 1)) < 1e-6, "The gradient is sigma(u) - 1.")
    check("bigger_at_2 is correct", bigger_at_2 == "sat", "At u = 2: sigma(2) = 0.88, so the sizes are 0.88 and 0.12.")
    RESULTS["p3_grads"] = (float(g_sat_m10), float(g_ns_m10), bigger_at_2)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# u10 = torch.tensor(-10., dtype=torch.float64, requires_grad=True)
# g_sat_m10 = float(torch.autograd.grad(torch.log(1 - torch.sigmoid(u10)), u10)[0])
# u10 = torch.tensor(-10., dtype=torch.float64, requires_grad=True)
# g_ns_m10 = float(torch.autograd.grad(-torch.log(torch.sigmoid(u10)), u10)[0])
# bigger_at_2 = "sat" if 1 / (1 + math.exp(-2)) > 1 - 1 / (1 + math.exp(-2)) else "ns"
# ```
#
# At u = -10 the original gradient is -4.54e-05 (it has all but vanished) and the non-saturating one is -0.99995. At u = +2 the sizes are 0.881 for the saturating loss and 0.119 for the non-saturating loss, so the answer is `sat`: the non-saturating loss pushes hardest when the forger is *losing* and softly when it is *winning*, which is the behaviour you want from a learning signal.
#
# </details>

# %% [markdown]
# ### Step 3.4. The optimal discriminator
#
# If the forger is fixed, what is the *best* possible detective? The answer is a formula: at every point x the best detective says D*(x) = p_data(x) / (p_data(x) + p_g(x)), the share of the real density in the sum of both densities. Take p_data = N(4, 1) (the real bell curve) and p_g = N(2, 1) (a forger that is two units off). The block computes D* on a grid and plots it; D*(3) = 1/2 by symmetry (the point halfway between the two means) and D* = 0.5 everywhere if p_g = p_data. Then it computes the value V(D*, G) numerically, by averaging over 200,000 samples, for p_g = N(2, 1), N(3, 1) and N(4, 1).

# %%
def normal_pdf(x, mu, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

def d_star(x, mu_g, mu_data=4.0):
    p, q = normal_pdf(x, mu_data), normal_pdf(x, mu_g)
    return p / (p + q)

x = np.linspace(-2, 8, 400)
print("D*(3) when p_g = N(2, 1):", d_star(3.0, 2.0))
print("largest |D* - 0.5| when p_g = p_data:", float(np.abs(d_star(x, 4.0) - 0.5).max()))
assert abs(d_star(3.0, 2.0) - 0.5) < 1e-12 and np.abs(d_star(x, 4.0) - 0.5).max() < 1e-12

rng = np.random.RandomState(SEED)
xr = rng.randn(200000) + 4.0                                  # real samples
Vs = {}
for mu_g in [2.0, 3.0, 4.0]:
    xf = rng.randn(200000) + mu_g                             # fake samples
    Vs[mu_g] = float(np.log(d_star(xr, mu_g)).mean() + np.log(1 - d_star(xf, mu_g)).mean())
    print(f"p_g = N({mu_g:.0f}, 1):  V(D*, G) = {Vs[mu_g]:.4f}")
print("-ln 4 =", round(-math.log(4), 4))

plt.figure(figsize=(6.2, 3.4))
plt.plot(x, normal_pdf(x, 4.0), "C0", label="p_data = N(4, 1)"); plt.plot(x, normal_pdf(x, 2.0), "C1", label="p_g = N(2, 1)")
plt.plot(x, d_star(x, 2.0), "k", label="D*(x)"); plt.axhline(0.5, color="grey", ls=":"); plt.axvline(3, color="grey", ls=":")
plt.xlabel("x"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** You should see that D*(3) is exactly 0.5 and that D* is flat at 0.5 when the forger is perfect (the largest difference is 0.0; the assertions pass). The plot shows two bells (real in blue, fake in orange) and the detective's curve D*, an S-shaped function: near 0 on the left where only the forger puts mass, rising through 0.5 at x = 3, and near 1 on the right where only real data live. The detective is the *ratio* of the two densities, so it knows the forger's mistakes exactly. The three numbers are V = -0.7122 for p_g = N(2, 1), -1.1617 for N(3, 1) and -1.3863 for N(4, 1) (the perfect forger), against -ln 4 = -1.3863. **The value approaches -ln 4 from above** as the forger improves: V(D*, G) is -ln 4 plus twice a measure of the difference between the two distributions (the Jensen-Shannon divergence, which is never negative), so the closer the forger is, the smaller the surplus, and for the perfect forger it vanishes. Seen from the side of the loss, the detective's binary cross-entropy (Step 3.2) is minus half of V, so a perfect forger drives it to ln 2 = 0.693: a detective that is no better than chance. Note also the warning hidden here: the *optimal* detective only exists if it is trained to perfection before every forger step, and real detectives are not.

# %% [markdown]
# **Tips and common mistakes**
#
# - The formula for D* assumes both densities are known. In a real GAN we never know p_data, we only have samples: that is why a neural network has to learn D from examples.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.4. A forger that is very wrong
#
# Your task: compute `v_far`, the value V(D*, G) for p_g = N(-4, 1) with the same method (200,000 samples, the generator `rng2 = np.random.RandomState(7)`; real samples `xr2 = 4 + rng2.randn(200000)`, fake samples `xf2 = -4 + rng2.randn(200000)`). Why it matters: when the forger is hopeless, the detective wins easily and V is close to its maximum, 0: the opposite end of the scale from the equilibrium -1.386.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
v_far = None   # V(D*, G) for p_g = N(-4, 1)

# %%
if v_far is None:
    print("Fill in v_far above, then run this cell again.")
else:
    check("v_far is close to 0 (above -0.01)", v_far > -0.01, "D* is almost 1 on real samples and almost 0 on fakes.")
    check("v_far is higher than for the forger N(2, 1)", v_far > Vs[2.0], "A worse forger gives the detective an easier job.")
    RESULTS["p3_vfar"] = round(float(v_far), 6)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# rng2 = np.random.RandomState(7)
# xr2 = 4 + rng2.randn(200000); xf2 = -4 + rng2.randn(200000)
# v_far = float(np.log(d_star(xr2, -4.0)).mean() + np.log(1 - d_star(xf2, -4.0)).mean())
# ```
#
# The value is -0.000173: essentially 0, the maximum. Between the hopeless forger (V near 0) and the perfect one (V = -1.386) lies the whole training run of a GAN.
#
# </details>

# %% [markdown]
# ### Step 3.5. One training iteration by hand
#
# Time for one honest training step, with the smallest detective that exists: D(x) = sigma(w x + b) with two scalar parameters. The real data are four numbers near 4 (3.8, 4.1, 4.3, 3.9) and the fake data four numbers near 0 (0.2, -0.1, 0.3, 0.0). Start with w = 0.1 and b = 0. The block computes the detective's loss L_D (the mean cross-entropy of both groups), asks **autograd** for the gradients with respect to w and b, makes one step of gradient descent with a learning rate of 0.1, and computes the loss again.

# %%
real = torch.tensor([3.8, 4.1, 4.3, 3.9]); fake = torch.tensor([0.2, -0.1, 0.3, 0.0])
w = torch.tensor(0.1, requires_grad=True); b = torch.tensor(0.0, requires_grad=True)

def loss_D(w, b):
    # -mean ln D(real) - mean ln(1 - D(fake)), with D(x) = sigmoid(w x + b), written with logsigmoid for stability
    return -(F.logsigmoid(w * real + b).mean() + F.logsigmoid(-(w * fake + b)).mean())

L0 = loss_D(w, b)
L0.backward()
print(f"before: w = {float(w):.3f}  b = {float(b):.3f}  loss = {float(L0):.4f}")
print(f"gradients: dL/dw = {float(w.grad):.4f}   dL/db = {float(b.grad):.4f}")
with torch.no_grad():
    w -= 0.1 * w.grad; b -= 0.1 * b.grad                    # one step of gradient descent
L1 = loss_D(w, b)
print(f"after : w = {float(w):.3f}  b = {float(b):.3f}  loss = {float(L1):.4f}")
assert float(L1) < float(L0)

# %% [markdown]
# **What you should see.** You should see the loss before the step, 1.2102 (well above ln 2 = 0.693: with w = 0.1 the detective gives the real examples only about 0.6 and the fakes about 0.5, so it is not yet sure of anything), the two gradients, -1.5611 for w and 0.1018 for b (a **negative** gradient means that increasing the parameter lowers the loss, a positive one that decreasing it does), and the weights after the step: w grew to 0.256 and b went to -0.010. The new loss is 1.0091, lower, so the assertion passes. In words: the detective learned that bigger numbers are real, by raising w, and nudged b down a little because the fakes sit near 0 and should get outputs below one half. This is the whole of the discriminator update of a GAN: compute the loss of a batch of real and fake examples, take the gradients, step. In Part 4 the detective is a small network instead of two numbers, and the fake numbers are made by the forger.

# %% [markdown]
# **Tips and common mistakes**
#
# - `w.grad` is set by `backward()`, and *adds up* if you call `backward()` again without clearing it. In a loop, set the gradients to zero before every `backward()`, or create fresh tensors, as the answer below does.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.5. Fifty steps
#
# Your task: repeat the update 50 times from the same starting point (w = 0.1, b = 0, learning rate 0.1), store the loss after every step in the list `losses50` (50 numbers) and the final weight in `w_final`. Why it matters: you will see a detective that trains on a *fixed* forger: its loss keeps falling towards 0, it becomes perfect. That is the starting point for the thinking exercise at the end of this part.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
losses50 = None   # a list of 50 numbers
w_final = None    # the weight after 50 steps

# %%
if losses50 is None or w_final is None:
    print("Fill in losses50 and w_final above, then run this cell again.")
else:
    check("50 losses", len(losses50) == 50, "One loss after every step.")
    check("the loss goes down (last below first)", losses50[-1] < losses50[0], "Take the step in the direction opposite to the gradient.")
    check("w is positive and larger than before", w_final > 0.1, "Bigger numbers are real, so w should grow.")
    RESULTS["p3_fifty"] = (round(float(losses50[0]), 4), round(float(losses50[-1]), 4), round(float(w_final), 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# w2 = torch.tensor(0.1, requires_grad=True); b2 = torch.tensor(0.0, requires_grad=True)
# losses50 = []
# for step in range(50):
#     L = loss_D(w2, b2)
#     g_w, g_b = torch.autograd.grad(L, [w2, b2])
#     with torch.no_grad():
#         w2 -= 0.1 * g_w; b2 -= 0.1 * g_b
#     losses50.append(float(loss_D(w2, b2)))
# w_final = float(w2)
# ```
#
# After 50 steps the loss fell from 1.009 (after the first step) to 0.277 and w grew from 0.1 to 1.238. The detective is almost perfect on this fixed forger. What does that mean for the forger? The next step shows it.
#
# </details>

# %% [markdown]
# ### Step 3.6. Imagine and write: why train both together?
#
# Suppose that the detective of the last step is trained to perfection *before* the forger starts to learn. The block continues the training for 300 steps with a larger learning rate (0.5), and then asks what the forger would feel: for each of the four fake examples it prints the detective's logit u, and the gradient that the forger's loss sends to that example's logit under the two losses of Step 3.3.

# %%
w3 = torch.tensor(0.1, requires_grad=True); b3 = torch.tensor(0.0, requires_grad=True)
for step in range(300):
    g_w, g_b = torch.autograd.grad(loss_D(w3, b3), [w3, b3])
    with torch.no_grad():
        w3 -= 0.5 * g_w; b3 -= 0.5 * g_b
u_f = (w3 * fake + b3).detach()
print(f"after 300 steps: w = {float(w3):.2f}  b = {float(b3):.2f}  detective loss = {float(loss_D(w3, b3)):.6f}")
print("logits for the fake examples:     ", np.round(u_f.numpy(), 1))
print("saturating gradient  -sigma(u):   ", np.round((-torch.sigmoid(u_f)).numpy(), 6))
print("non-saturating gradient sigma(u)-1:", np.round((torch.sigmoid(u_f) - 1).numpy(), 4))
assert float(loss_D(w3, b3)) < 0.01

# %% [markdown]
# **What you should see.** You should see a detective that is nearly perfect: after 300 steps its loss is 0.0099, with w = 2.97 and b = -5.15. For the fake examples its logits are very negative ([-4.6, -5.5, -4.3, -5.2]), so the detective is sure that all four are fake. And the forger's gradient? With the **saturating** loss it is [-0.0104, -0.0043, -0.0139, -0.0057]: about a hundred times smaller than the non-saturating gradient, which is close to -1 for all four. The more the detective improves, the smaller the saturating gradient gets (at a logit of -10 it would be 0.00005, as you computed in Step 3.3). This is the lesson of Part 3 in one picture. A forger can only improve if the detective gives it a slope to follow; with the original loss, a detective that is too good gives a nearly flat landscape. The two networks have to grow **together**: the detective must stay just ahead of the forger, not infinitely far ahead.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.6. Your reasoning in words
#
# Your task: write three or four sentences in the string `reason`: why must the forger and the detective improve together, and what would happen if the detective were trained to perfection before the forger started? Use your numbers from this step. Why it matters: this is the core idea of GAN training and the reason why the training of Part 4 alternates between the two networks, step by step. Writing it down in your own words is the best preparation for the reflection questions of Part 7.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
reason = ""   # your three or four sentences (a string)

# %%
if len(reason.strip()) < 60:
    print("Write three or four sentences in reason above, then run this cell again.")
else:
    check("you wrote an answer", True)
    check("you mention the gradient or the signal", any(w in reason.lower() for w in ["gradient", "signal", "slope", "learn"]), "What does the forger receive from the detective?")
    RESULTS["p3_reason"] = reason.strip()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reason = ("The forger only learns from the gradient that the detective gives it. If the detective is perfect from the start, "
#           "it is sure about every fake (the logits are about -5), and with the original loss the gradient is a hundred times smaller than it could be, "
#           "so the forger learns very little. If the forger is perfect, the detective cannot learn either. So both must improve together, "
#           "each one a little ahead of the other, which is why we alternate small steps.")
# ```
#
# The key sentences are about the *signal*: the forger's only teacher is the detective's gradient, and a detective that is too good, or a forger that is too weak, makes that signal vanish. That balance is the hard part of GAN training, and Parts 4 and 5 will show it in the training curves.
#
# </details>

# %% [markdown]
# ## Part 4. A one-dimensional GAN from scratch (25 min)
#
# *Lesson sections: "GAN game" and "Training".* In Part 3 you played the GAN game with pencil and autograd on a few numbers. Now you let two real networks play it. ==You will build a generator and a discriminator for numbers on a line, write the two loss functions yourself, train the pair for 3,000 steps, watch a bell curve appear out of noise, look through the discriminator's eyes, compare the original (saturating) generator loss with the non-saturating one, and test which settings keep the game stable.== The "pictures" are single numbers drawn from a bell curve with mean 4 and standard deviation 1, so you can see the whole distribution in one histogram. All numbers below are for `SEED = 42` (seed 1 is the second seed). A GAN is not deterministic across machines, so every number comes with a tolerance.
#
# ### ✏️ Imagine first
#
# Picture the forger as a tiny machine that gets one random number and must turn it into a number that looks like it came from the real bell curve centred at 4. The detective sees one number at a time and says "real" or "fake". At the start the forger writes numbers around 0.5, far from 4. Before you run any code, guess how many training steps (one step is one update of the detective and one of the forger) the pair needs until the forger's histogram looks like the real bell curve: a handful, hundreds, or tens of thousands? Write your guess in the cell below. You compare it with Step 4.3.

# %%
# Imagine first: how many training steps does a tiny GAN need before its output looks like the real bell curve?
my_guess_steps = None   # write a number, for example 100, 1000 or 10000

# %% [markdown]
# ### Step 4.1. The pieces: data, generator, discriminator
#
# A GAN needs three things: a source of real data, a generator G that turns noise z into fake data, and a discriminator D that scores a number. The real data are numbers from N(4, 1), made by `sample_real_1d`. The generator gets one noise number z from N(0, 1) and must output one number; its hidden layer has 16 units with ReLU. The discriminator gets one number and must output a **logit** (a score before the sigmoid; the loss function `BCEWithLogitsLoss` applies the sigmoid inside, which is numerically safer). Its job is harder than the generator's: it must carve a "bump" around 4 out of the number line, which needs bends in its curve, so we give it two hidden layers of 32 units with LeakyReLU. (A smaller discriminator with one hidden layer also works, but in our tests it made the generated spread wobble between 0.8 and 2.0, so we use the deeper one.) The cell below builds both networks, counts parameters and shows what the untrained generator does.

# %%
real = sample_real_1d(1000, seed=0)       # the "real data": numbers from a bell curve with mean 4 and standard deviation 1
print("real data: mean %.3f  std %.3f  min %.2f  max %.2f" % (real.mean(), real.std(), real.min(), real.max()))

torch.manual_seed(SEED)
G1 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))      # one noise number -> 16 hidden units -> one fake number
D1 = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))   # a number -> a real/fake score (a logit)
print("generator parameters    :", count_params(G1))
print("discriminator parameters:", count_params(D1))

z = torch.randn(1000, 1)                  # the noise z comes from N(0, 1)
with torch.no_grad():
    fake = G1(z)
print("untrained generator: mean %.3f  std %.3f" % (fake.mean(), fake.std()))
RESULTS["p4_params_G"] = count_params(G1); RESULTS["p4_params_D"] = count_params(D1)

# %% [markdown]
# ### ✏️ Your turn: The pieces: data, generator, discriminator
#
# *Exercise 4.1:* Write the two functions `make_G1()` and `make_D1()` that return fresh networks exactly as described. Why it matters: you will create new networks for every experiment of this part, so these two functions are the factory. Count by hand first: the generator has (1 x 16 + 16) + (16 x 1 + 1) = 49 numbers and the discriminator has (1 x 32 + 32) + (32 x 32 + 32) + (32 x 1 + 1) = 1,153.

# %%
def make_G1():
    return None   # YOUR CODE: nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
def make_D1():
    return None   # YOUR CODE: nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

# %%
g_, d_ = make_G1(), make_D1()
if g_ is None or d_ is None:
    print("Fill in make_G1 and make_D1 above, then run this cell again.")
else:
    check("the generator has 49 parameters", count_params(g_) == 49, "(1*16 + 16) + (16*1 + 1) = 49")
    check("the discriminator has 1,153 parameters", count_params(d_) == 1153, "(1*32 + 32) + (32*32 + 32) + (32*1 + 1) = 1,153")
    check("both map a batch of 5 numbers to 5 numbers", tuple(g_(torch.randn(5, 1)).shape) == (5, 1) and tuple(d_(torch.randn(5, 1)).shape) == (5, 1), "one number in, one number out")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def make_G1():
#     return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
# def make_D1():
#     return nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
# print(count_params(make_G1()), count_params(make_D1()))
# assert count_params(make_G1()) == 49 and count_params(make_D1()) == 1153
# ```
#
# Your functions return networks with exactly 49 and 1,153 parameters and the right shapes. Layer 1 of G has 16 weights and 16 biases, layer 2 has 16 weights and 1 bias. For D the middle layer is the big one: 32 x 32 weights and 32 biases. From here on we use these factories.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def make_G1():
    return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
def make_D1():
    return nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
print(count_params(make_G1()), count_params(make_D1()))
assert count_params(make_G1()) == 49 and count_params(make_D1()) == 1153

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The real sample has mean 3.955 and standard deviation 0.987 (tolerance: mean within 0.15 of 4, std within 0.1 of 1; it is a sample of 1,000). The generator has **49** parameters and the discriminator **1,153**. The untrained generator outputs numbers with mean about 0.5 and standard deviation about 0.4: all fakes sit in a small cloud far to the left of the data, so a discriminator finds them easily.
#
# Tips and common mistakes:
#
#    - A `Linear(in, out)` layer has `in * out + out` numbers; ReLU and LeakyReLU have none.
#    - The last layer of D has no sigmoid: the loss function adds it.
#
# Try also (optional):
#
#    - Make G wider (64 hidden units) and count again. We get 193.

# %% [markdown]
# ### Step 4.2. The two losses and the training loop
#
# One training step has three parts: (a) draw a batch of real numbers and a batch of noise; (b) update D so that it says "real" (label 1) for real numbers and "fake" (label 0) for generated ones, with the generated numbers **detached** so that this update does not change G; (c) update G so that D says "real" for its fakes. For (c) we use the **non-saturating** loss from Part 3: G is trained with the label 1 on its own fakes (the loss minus ln D(G(z))) instead of minimising ln(1 - D(G(z))). Both losses are binary cross-entropy with the logits as input. First you write the two losses, then the loop below uses our versions.

# %% [markdown]
# ### ✏️ Your turn: The two losses and the training loop
#
# *Exercise 4.2:* Write `d_loss_mine(D, xr, xf)` (the average of the real part and the fake part) and `g_loss_mine(D, xf)` (the non-saturating generator loss) with `F.binary_cross_entropy_with_logits`. Why it matters: these two functions are the whole game; everything else in a GAN is bookkeeping. The check uses a "discriminator" that just returns its input, so you can verify by hand: for a real logit of 2 the loss is minus ln sigmoid(2) = 0.1269, for a fake logit of -1 it is minus ln(1 - sigmoid(-1)) = 0.3133, their average is 0.2201, and the generator loss for the fake logit -1 is minus ln sigmoid(-1) = 1.3133.

# %%
def d_loss_mine(D, xr, xf):
    """Discriminator loss: the average of the 'real' part and the 'fake' part."""
    return None   # YOUR CODE: BCE-with-logits of D(xr) against label 1 plus BCE-with-logits of D(xf) against label 0, divided by 2
def g_loss_mine(D, xf):
    """Non-saturating generator loss."""
    return None   # YOUR CODE: BCE-with-logits of D(xf) against label 1 (the generator wants D to say "real")

# %%
ident = lambda x: x                         # a "discriminator" that returns its input as the logit, so we can test with numbers we choose
xr_t, xf_t = torch.tensor([[2.0]]), torch.tensor([[-1.0]])
want_d = 0.5 * (math.log(1 + math.exp(-2.0)) + math.log(1 + math.exp(-1.0)))   # -ln(sigmoid(2)) and -ln(1 - sigmoid(-1)), averaged: 0.2201
want_g = math.log(1 + math.exp(1.0))                                           # -ln(sigmoid(-1)) = 1.3133
a_, b_ = d_loss_mine(ident, xr_t, xf_t), g_loss_mine(ident, xf_t)
if a_ is None or b_ is None:
    print("Fill in both loss functions above, then run this cell again.")
else:
    check("d_loss is %.4f for logits 2 (real) and -1 (fake)" % want_d, abs(float(a_) - want_d) < 1e-5, "real part -ln sigmoid(2), fake part -ln(1 - sigmoid(-1)), then the average")
    check("g_loss is %.4f for the fake logit -1" % want_g, abs(float(b_) - want_g) < 1e-5, "the target label for the generator is 1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def d_loss(D, xr, xf):
#     lr_, lf_ = D(xr), D(xf)
#     return 0.5 * (F.binary_cross_entropy_with_logits(lr_, torch.ones_like(lr_)) +
#                   F.binary_cross_entropy_with_logits(lf_, torch.zeros_like(lf_)))
# def g_loss(D, xf, saturating=False):
#     lo = D(xf)
#     if saturating:                                           # the original loss ln(1 - D(G(z))), see Step 4.5
#         return -F.binary_cross_entropy_with_logits(lo, torch.zeros_like(lo))
#     return F.binary_cross_entropy_with_logits(lo, torch.ones_like(lo))   # non-saturating: the labels of the fakes are flipped to 1
# ident = lambda x: x
# assert abs(float(d_loss(ident, torch.tensor([[2.0]]), torch.tensor([[-1.0]]))) - 0.2201) < 1e-3
# assert abs(float(g_loss(ident, torch.tensor([[-1.0]]))) - 1.3133) < 1e-3
# print("d_loss: %.4f   g_loss: %.4f" % (float(d_loss(ident, torch.tensor([[2.0]]), torch.tensor([[-1.0]]))), float(g_loss(ident, torch.tensor([[-1.0]])))))
# ```
#
# `d_loss` is the average of minus ln D(x) on reals and minus ln(1 - D(G(z))) on fakes. `g_loss` flips the label of the fakes to 1. The `detach()` in the loop is what keeps step (b) from changing G; G's own update in step (c) does not detach, so the gradient flows through D into G.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def d_loss(D, xr, xf):
    lr_, lf_ = D(xr), D(xf)
    return 0.5 * (F.binary_cross_entropy_with_logits(lr_, torch.ones_like(lr_)) +
                  F.binary_cross_entropy_with_logits(lf_, torch.zeros_like(lf_)))
def g_loss(D, xf, saturating=False):
    lo = D(xf)
    if saturating:                                           # the original loss ln(1 - D(G(z))), see Step 4.5
        return -F.binary_cross_entropy_with_logits(lo, torch.zeros_like(lo))
    return F.binary_cross_entropy_with_logits(lo, torch.ones_like(lo))   # non-saturating: the labels of the fakes are flipped to 1
ident = lambda x: x
assert abs(float(d_loss(ident, torch.tensor([[2.0]]), torch.tensor([[-1.0]]))) - 0.2201) < 1e-3
assert abs(float(g_loss(ident, torch.tensor([[-1.0]]))) - 1.3133) < 1e-3
print("d_loss: %.4f   g_loss: %.4f" % (float(d_loss(ident, torch.tensor([[2.0]]), torch.tensor([[-1.0]]))), float(g_loss(ident, torch.tensor([[-1.0]])))))

# %% [markdown]
# The training loop. It records the generated mean and standard deviation every 100 steps, the two losses at every step, 1,000 generated numbers at the steps in `snaps`, and the discriminator's output on a grid of x values at the same moments. The arguments `saturating` and `g_shift` are for Step 4.5.

# %%
def train_gan_1d(steps=3000, seed=42, lr_g=1e-3, lr_d=1e-3, d_steps=1, saturating=False, g_shift=0.0, batch=128, snaps=(0, 300, 1000, 3000), rec=100):
    """Train the 1-D GAN and return everything we want to plot."""
    torch.manual_seed(seed)
    G, D = make_G1(), make_D1()
    G[2].bias.data += g_shift                                             # g_shift moves the untrained generator away from the data (Step 4.5)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    data = torch.tensor(sample_real_1d(20000, seed=seed))                 # a pool of real numbers to draw batches from
    grid = torch.linspace(-2, 10, 121).unsqueeze(1)
    snaps = set(snaps) | {steps}
    h = {"step": [], "mean": [], "std": [], "ld": [], "lg": [], "samples": {}, "dgrid": {}, "grid": grid.squeeze(1).numpy(), "G": G, "D": D}
    for s in range(steps + 1):
        if s % rec == 0 or s in snaps:                                    # look at the generator now (no learning here)
            with torch.no_grad():
                g = G(torch.randn(1000, 1))
                if s in snaps:
                    h["samples"][s] = g.squeeze(1).numpy(); h["dgrid"][s] = torch.sigmoid(D(grid)).squeeze(1).numpy()
            if s % rec == 0:
                h["step"].append(s); h["mean"].append(float(g.mean())); h["std"].append(float(g.std()))
        if s == steps: break
        for _ in range(d_steps):                                          # (a)+(b) real batch, noise batch, update D
            xr = data[torch.randint(0, len(data), (batch,))]
            xf = G(torch.randn(batch, 1)).detach()                        # detach: the D update must not change G
            ld = d_loss(D, xr, xf); opt_d.zero_grad(); ld.backward(); opt_d.step()
        lg = g_loss(D, G(torch.randn(batch, 1)), saturating)              # (c) update G (the fakes are NOT detached now)
        opt_g.zero_grad(); lg.backward(); opt_g.step()
        h["ld"].append(float(ld)); h["lg"].append(float(lg))
    return h

t0 = time.time(); h_test = train_gan_1d(steps=200, snaps=())
print("200 steps took %.1f seconds; the loss values are finite:" % (time.time() - t0), np.isfinite(h_test["ld"]).all() and np.isfinite(h_test["lg"]).all())

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The check prints two ticks, the answer prints `d_loss: 0.2201   g_loss: 1.3133`, and 200 training steps take about one to two seconds on a CPU (so 3,000 steps take 15 to 30 seconds). The loss values are finite.
#
# Tips and common mistakes:
#
#    - Forgetting `detach()` in the D update does not crash; it quietly computes useless gradients for G. Always detach.
#    - Call `opt.zero_grad()` before `backward()` for each optimiser.

# %% [markdown]
# ### Step 4.3. Watch the bell curve appear
#
# Now we train for 3,000 steps with Adam (learning rate 1e-3 for both networks, betas 0.5 and 0.999, batch 128) and look at the result in two figures: four histograms of the generated numbers against the real bell curve (steps 0, 300, 1,000, 3,000), and the curves of the generated mean, the generated standard deviation and the two losses. The dotted line in the loss plot is ln 2 = 0.693: a discriminator that cannot tell, and says 0.5 for everything, has exactly this loss.

# %%
steps1 = 300 if QUICK else 3000
h = train_gan_1d(steps=steps1, seed=SEED)                                 # the main 1-D run
for s in (0, 100, 300, 1000, 2000, 3000):
    if s in h["step"]:
        i = h["step"].index(s); print("step %4d: generated mean %.2f  std %.2f" % (s, h["mean"][i], h["std"][i]))

xs = np.linspace(-2, 10, 200); bell = np.exp(-(xs - 4) ** 2 / 2) / np.sqrt(2 * np.pi)
snap_steps = sorted(h["samples"])
fig, axes = plt.subplots(1, len(snap_steps), figsize=(3.0 * len(snap_steps), 2.6), sharey=True)
for ax, s in zip(np.ravel(axes), snap_steps):
    ax.hist(h["samples"][s], bins=40, range=(-2, 10), density=True, alpha=0.6, color="tab:orange", label="generated")
    ax.plot(xs, bell, "k--", lw=1, label="real N(4, 1)"); ax.set_title("step %d" % s, fontsize=9)
np.ravel(axes)[0].legend(fontsize=7); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(12, 2.8))
ax[0].plot(h["step"], h["mean"], label="mean"); ax[0].axhline(4, color="k", ls="--", lw=0.8); ax[0].set_title("generated mean (target 4)")
ax[1].plot(h["step"], h["std"], color="tab:green"); ax[1].axhline(1, color="k", ls="--", lw=0.8); ax[1].set_title("generated std (target 1)")
sm = lambda v, w=50: np.convolve(v, np.ones(w) / w, mode="valid")
ax[2].plot(sm(h["ld"]), label="D loss"); ax[2].plot(sm(h["lg"]), label="G loss"); ax[2].axhline(math.log(2), color="k", ls=":", lw=0.8); ax[2].legend(); ax[2].set_title("losses (smoothed; dotted = ln 2)")
for a in ax: a.set_xlabel("step") if a is not ax[2] else None
plt.tight_layout(); plt.show()
print("average D loss over the last 500 steps: %.3f   (ln 2 = %.3f)" % (np.mean(h["ld"][-500:]), math.log(2)))
print("average G loss over the last 500 steps: %.3f" % np.mean(h["lg"][-500:]))

# %% [markdown]
# ### ✏️ Your turn: Watch the bell curve appear
#
# *Exercise 4.3:* Compute `final_mean` and `final_std` as the average of the last 5 recorded values (so you smooth the wobble), and set `verdict` to "healthy" if the mean is within 0.4 of 4 and the standard deviation between 0.5 and 1.5, else "not healthy". Why it matters: a GAN's last single number is noisy, and a rule you decide on beforehand keeps you honest.

# %%
final_mean = None    # YOUR CODE: the average of the last 5 recorded values of h["mean"]
final_std = None     # YOUR CODE: the average of the last 5 recorded values of h["std"]
verdict = None       # YOUR CODE: "healthy" if the mean is within 0.4 of 4 and the std between 0.5 and 1.5, else "not healthy"

# %%
if final_mean is None or final_std is None or verdict is None:
    print("Fill in the three variables above, then run this cell again.")
else:
    check("final mean within 0.4 of 4", abs(final_mean - 4) < 0.4, "average the last five numbers: np.mean(h['mean'][-5:])")
    check("final std between 0.5 and 1.5", 0.5 < final_std < 1.5, "np.mean(h['std'][-5:]); a GAN's spread can wobble, so the check is tolerant")
    check("the verdict agrees with your two numbers", verdict == ("healthy" if abs(final_mean - 4) < 0.4 and 0.5 < final_std < 1.5 else "not healthy"), "apply the rule from the exercise")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# final_mean = float(np.mean(h["mean"][-5:]))
# final_std = float(np.mean(h["std"][-5:]))
# verdict = "healthy" if abs(final_mean - 4) < 0.4 and 0.5 < final_std < 1.5 else "not healthy"
# print("final mean %.3f  final std %.3f  -> %s" % (final_mean, final_std, verdict))
# RESULTS["p4_final_mean"] = round(final_mean, 3); RESULTS["p4_final_std"] = round(final_std, 3)
# RESULTS["p4_dloss_end"] = round(float(np.mean(h["ld"][-500:])), 3)
# ```
#
# Mean and spread both match the real data, and the two losses hover near ln 2: the detective is guessing at about 50-50. This is what "equilibrium" looks like. Note how unexciting the loss curves are: they do not tell you that the generator got good, the histograms do (Part 5 comes back to this).
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The mean runs 0.52 (step 0), 1.64 (100), 4.44 (300), 3.96 (1,000), 4.01 (2,000), 4.00 (3,000) and the standard deviation 0.39, 1.21, 2.15, 0.93, 0.96, 0.94. In words: in about 300 steps the forger's cloud has travelled from 0.5 to the data, overshooting in width (std 2.15, a smear), and by 1,000 steps the histogram sits on the bell curve. The final values are mean 4.006 and std 0.94 (tolerance: mean within 0.4 of 4, std between 0.7 and 1.3 for seed 42; other seeds gave 0.75 to 1.4 in our tests), so the verdict is "healthy". The average discriminator loss over the last 500 steps is 0.681, close to ln 2 = 0.693 (tolerance: 0.60 to 0.70), and the generator loss is about 0.72, also close to ln 2.
#
# So the answer to the Imagine-first question is: **about 300 steps to get close, about 1,000 to look right**. If you guessed tens of thousands, this is a one-number problem; for pictures it takes far longer.
#
# Tips and common mistakes:
#
#    - The std overshoots early (2.15 at step 300) because G first stretches its cloud across the data; do not judge a GAN at step 300.
#    - If your final std is near 2 or near 0.5, try another seed: a GAN can settle on a wrong width.
#
# Try also (optional):
#
#    - Run with `seed=1`. We got a healthy result with mean about 4.0 (the std wobbles; see Step 4.6).

# %% [markdown]
# ### Step 4.4. The discriminator's view
#
# The discriminator is a function D(x) between 0 and 1. If the game is nearly won, D(x) should be near 0.5 wherever the real data live, because fakes and reals look alike there. The figure plots D(x) on a grid from -2 to 10 at the snapshot steps, together with the real density (the dashed bell). At step 0 the generator's cloud is near 0.5, so D should start to say "fake" there and "real" further right.

# %%
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(h["grid"], np.exp(-(h["grid"] - 4) ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="real density N(4, 1) (height scaled to 0.4)")
for s, c in zip(snap_steps, ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red"]):
    ax.plot(h["grid"], h["dgrid"][s], color=c, label="D(x) at step %d" % s)
ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xlabel("x"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7); plt.tight_layout(); plt.show()
last = snap_steps[-1]
for x in (0, 2, 4, 6, 8):
    j = int(np.argmin(np.abs(h["grid"] - x))); print("D(%d) at the end = %.3f   at step 0 = %.3f" % (x, h["dgrid"][last][j], h["dgrid"][0][j]))

# %% [markdown]
# ### ✏️ Your turn: The discriminator's view
#
# *Exercise 4.4:* Read the discriminator's output at x = 4 and at x = 9 at the end of training from `h["dgrid"][last]`, using the grid point closest to each x. Why it matters: it is the cleanest test that the discriminator is confused where the data live.

# %%
d_at_4 = None        # YOUR CODE: the discriminator's output D(4) at the end of training, read from h["dgrid"][last] at the grid point closest to x = 4
d_far = None         # YOUR CODE: the same at x = 9 (far away from the real data)

# %%
if d_at_4 is None or d_far is None:
    print("Fill in both variables above, then run this cell again.")
else:
    check("D(4) is between 0.25 and 0.75 (it cannot tell)", 0.25 < d_at_4 < 0.75, "use j = np.argmin(np.abs(h['grid'] - 4)) and h['dgrid'][last][j]")
    check("D(9) is a number between 0 and 1", 0 <= d_far <= 1, "the output of a sigmoid")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# j4 = int(np.argmin(np.abs(h["grid"] - 4))); j9 = int(np.argmin(np.abs(h["grid"] - 9)))
# d_at_4 = float(h["dgrid"][last][j4]); d_far = float(h["dgrid"][last][j9])
# print("D(4) = %.3f   D(9) = %.3f" % (d_at_4, d_far))
# RESULTS["p4_D_at_4"] = round(d_at_4, 3)
# ```
#
# The optimal discriminator of Part 3 is p_data / (p_data + p_g). Where p_g equals p_data it is 0.5, and your trained D shows that near 4. Outside the data it can say anything, because no real or fake number reaches that region and D gets no training signal there.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# At the end D(4) = 0.501 (tolerance: between 0.3 and 0.7) and D(6) = 0.507: the detective is undecided on the data. Away from the data D says other things: D(0) is 1.000 and D(2) = 0.877 on the left, D(8) = 0.388 and D(9) = 0.283 on the right. At step 0 the curve was flat (D(4) = 0.408). Only the region where fakes and reals both occur has to be 0.5.
#
# Tips and common mistakes:
#
#    - D(x) far from the data is not meaningful. Judge the discriminator only where the data live.

# %% [markdown]
# ### Step 4.5. The original (saturating) loss
#
# In Part 3 you computed that the original generator loss ln(1 - D(G(z))) has a gradient that vanishes when D is sure a fake is fake (about -0.0025 at logit -6), while the non-saturating loss has a gradient of about -0.9975 there. Does it matter in practice? We test it with the same seed and the same number of steps, first from the normal start (the generator's cloud near 0.5) and then from a **hard start**, where we shift the generator's last bias by -8 so that its fakes sit near -8, where the discriminator is certain they are fake.

# %% [markdown]
# ### ✏️ Your turn: The original (saturating) loss
#
# *Exercise 4.5:* Write the saturating generator loss `g_loss_sat_mine(D, xf)` stably as minus the BCE-with-logits of D(xf) against label 0 (this equals ln(1 - D(G(z)))). The check uses the logit -1: ln(1 - sigmoid(-1)) = -0.3133.

# %%
def g_loss_sat_mine(D, xf):
    """The ORIGINAL generator loss ln(1 - D(G(z))), written stably from the logit."""
    return None   # YOUR CODE: minus the BCE-with-logits of D(xf) against label 0

# %%
v_ = g_loss_sat_mine(ident, torch.tensor([[-1.0]]))
want_s = math.log(1 - 1 / (1 + math.exp(1.0)))        # ln(1 - sigmoid(-1)) = -0.3133
if v_ is None:
    print("Fill in g_loss_sat_mine above, then run this cell again.")
else:
    check("ln(1 - D) is %.4f for the logit -1" % want_s, abs(float(v_) - want_s) < 1e-5, "-F.binary_cross_entropy_with_logits(D(xf), zeros)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def g_loss_sat_mine(D, xf):
#     lo = D(xf)
#     return -F.binary_cross_entropy_with_logits(lo, torch.zeros_like(lo))
# assert abs(float(g_loss_sat_mine(ident, torch.tensor([[-1.0]]))) - (-0.3133)) < 1e-3
# print("ln(1 - D) at logit -1: %.4f (equal to g_loss(..., saturating=True): %.4f)" % (float(g_loss_sat_mine(ident, torch.tensor([[-1.0]]))), float(g_loss(ident, torch.tensor([[-1.0]]), True))))
# ```
#
# The saturating loss is not broken, it is slow when the generator is bad and the discriminator is strong, which is the situation at the beginning of training in most real GANs. In this tiny one-dimensional problem the start is easy and the problem hides. For pictures it does not hide, which is why most GANs since 2014 use the non-saturating loss.
#
# </details>

# %%
h_sat = train_gan_1d(steps=steps1, seed=SEED, saturating=True)              # same seed, same steps, the original loss
for s in (0, 100, 300, 1000, 2000, 3000):
    if s in h_sat["step"]:
        i = h_sat["step"].index(s); print("step %4d: saturating mean %.2f std %.2f   |  non-saturating mean %.2f std %.2f" % (s, h_sat["mean"][i], h_sat["std"][i], h["mean"][i], h["std"][i]))
fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
ax[0].plot(h["step"], h["mean"], label="non-saturating"); ax[0].plot(h_sat["step"], h_sat["mean"], label="saturating"); ax[0].axhline(4, color="k", ls="--", lw=0.8); ax[0].set_title("generated mean"); ax[0].legend(); ax[0].set_xlabel("step")
ax[1].plot(h["step"], h["std"], label="non-saturating"); ax[1].plot(h_sat["step"], h_sat["std"], label="saturating"); ax[1].axhline(1, color="k", ls="--", lw=0.8); ax[1].set_title("generated std"); ax[1].set_xlabel("step")
plt.tight_layout(); plt.show()
def first_step_near4(hh, tol=0.5):
    for s_, m_ in zip(hh["step"], hh["mean"]):
        if abs(m_ - 4) < tol: return s_
    return None
print("first recorded step with the mean within 0.5 of 4: non-saturating", first_step_near4(h), " saturating", first_step_near4(h_sat))
print("final (mean of last 5 records): saturating mean %.2f std %.2f" % (np.mean(h_sat["mean"][-5:]), np.mean(h_sat["std"][-5:])))
RESULTS["p4_sat_first_near4"] = first_step_near4(h_sat); RESULTS["p4_nonsat_first_near4"] = first_step_near4(h)
RESULTS["p4_sat_final_mean"] = round(float(np.mean(h_sat["mean"][-5:])), 3)

# a harder start: shift the untrained generator 8 units to the LEFT of the data, where the discriminator is sure that its output is fake
steps5 = 300 if QUICK else 1500
far = {}
for sat in (False, True):
    for seed in (42, 1):
        far[(sat, seed)] = train_gan_1d(steps=steps5, seed=seed, saturating=sat, g_shift=-8.0, snaps=())
print("generated mean every 300 steps, generator started near -8:")
for (sat, seed), hh in far.items():
    print("  %-14s seed %2d:" % ("saturating" if sat else "non-saturating", seed), " ".join("%5.1f" % hh["mean"][i] for i in range(0, len(hh["mean"]), 3)))
for (sat, seed), hh in far.items():
    RESULTS["p4_far_%s_%d" % ("sat" if sat else "nonsat", seed)] = first_step_near4(hh)
print("first recorded step with the mean within 0.5 of 4:", {("sat" if k[0] else "nonsat", k[1]): first_step_near4(v) for k, v in far.items()})

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# From the normal start both losses learn equally well: the mean is within 0.5 of 4 at step 300 for both, and the final values (average of the last 5 records) are mean 3.97, std 0.96 for the saturating loss against 4.00 and 0.94 for the non-saturating one. **No visible difference.** From the hard start the difference appears: the non-saturating loss first reaches the data at step 700 (seed 42) and 500 (seed 1), the saturating loss at step 1,300 (seed 42) and 1,100 (seed 1), about twice as slowly. The saturating generator with seed 1 sits at -8.1, -7.6, -7.3 for the first 600 steps: it barely moves, which is the vanishing gradient. Tolerance: the order (non-saturating first) held for both seeds; the steps themselves can differ by 200 to 300 on your machine. Both end at the data.
#
# Tips and common mistakes:
#
#    - Run both losses with the same seed and the same settings: only one thing may change at a time.
#    - "Learns more slowly" is a finding. We did not see a failure here.
#
# Try also (optional):
#
#    - Shift the generator by +8 (to the right of the data) instead of -8 and compare again. We saw both losses recover (about 300 to 600 steps).

# %% [markdown]
# ### Step 4.6. Balance: how many D steps, how fast a discriminator
#
# A GAN works only if the two players stay roughly in balance. We test twelve runs of 1,500 steps: the discriminator takes 1 or 3 steps per generator step, and its learning rate is 3e-4, 1e-3 or 3e-3 (the generator's is always 1e-3), each with seeds 42 and 1. The table reports the final mean, the final standard deviation (both averaged over the last 5 records) and the average discriminator loss over the last 300 steps. This cell takes about 45 seconds.

# %%
def grid_run(d_steps, lr_d, seed, lr_g=1e-3, steps=1500):
    hh = train_gan_1d(steps=steps, seed=seed, lr_g=lr_g, lr_d=lr_d, d_steps=d_steps, snaps=())
    return float(np.mean(hh["mean"][-5:])), float(np.mean(hh["std"][-5:])), float(np.mean(hh["ld"][-300:]))

steps6 = 200 if QUICK else 1500
rows = []
print("d_steps  lr_d    seed |  mean    std   D loss")
for d_steps in (1, 3):
    for lr_d in (3e-4, 1e-3, 3e-3):
        for seed in (42, 1):
            m, sd, ld = grid_run(d_steps, lr_d, seed, steps=steps6)
            rows.append((d_steps, lr_d, seed, m, sd, ld))
            print("%5d   %.0e   %3d | %5.2f  %5.2f   %.3f" % (d_steps, lr_d, seed, m, sd, ld))

# %% [markdown]
# ### ✏️ Your turn: Balance: how many D steps, how fast a discriminator
#
# *Exercise 4.6:* Fill in `stable`, the list of (d_steps, lr_d) pairs for which both seeds have a mean within 0.5 of 4 and a standard deviation between 0.5 and 1.5, and `wobbliest`, the pair whose two seeds disagree most about the standard deviation. Why it matters: a setting that works for one seed only is luck.

# %%
stable = None   # YOUR CODE: a list of (d_steps, lr_d) pairs for which BOTH seeds have |mean - 4| < 0.5 and std between 0.5 and 1.5
wobbliest = None   # YOUR CODE: the (d_steps, lr_d) pair whose two seeds have the most different std (the biggest gap)

# %%
want_stable, gaps = [], {}
for ds in (1, 3):
    for lrd in (3e-4, 1e-3, 3e-3):
        rr = [r for r in rows if r[0] == ds and r[1] == lrd]
        if all(abs(r[3] - 4) < 0.5 and 0.5 < r[4] < 1.5 for r in rr): want_stable.append((ds, lrd))
        gaps[(ds, lrd)] = abs(rr[0][4] - rr[1][4])
want_wob = max(gaps, key=gaps.get)
if stable is None or wobbliest is None:
    print("Fill in both variables above, then run this cell again.")
else:
    check("the list of stable settings matches the table", sorted(stable) == sorted(want_stable), "use the rows list: keep a pair only if both seeds pass")
    check("the wobbliest setting is the one with the biggest std gap", tuple(wobbliest) == want_wob, "compare abs(std of seed 42 - std of seed 1) for every pair")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# stable, gaps = [], {}
# for ds in (1, 3):
#     for lrd in (3e-4, 1e-3, 3e-3):
#         rr = [r for r in rows if r[0] == ds and r[1] == lrd]
#         if all(abs(r[3] - 4) < 0.5 and 0.5 < r[4] < 1.5 for r in rr): stable.append((ds, lrd))
#         gaps[(ds, lrd)] = abs(rr[0][4] - rr[1][4])
# wobbliest = max(gaps, key=gaps.get)
# print("stable settings (both seeds):", len(stable), "of 6")
# print("wobbliest setting:", wobbliest, " std gap between the two seeds: %.2f" % gaps[wobbliest])
# RESULTS["p4_stable_settings"] = len(stable)
# ```
#
# A one-dimensional GAN is forgiving: all these settings reach the bell curve. The lesson is not that balance does not matter, but that the standard deviation (the width of the forger's distribution) wobbles first, and that you need two seeds before you trust a setting. Part 5 shows what an imbalance does when there are eight blobs to find.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# All six settings are stable by this rule: the means lie between 3.80 and 4.09 and the standard deviations between 0.75 and 1.42, and the discriminator loss stays between 0.645 and 0.693, never far from ln 2. The wobbliest pair is 1 D step with lr_d 3e-4: std 1.42 for seed 42 against 0.79 for seed 1, a gap of 0.63 (tolerance: the pair can change on your machine; the wobbliest gap should be above 0.3). The weakest discriminator gives the loosest result; faster discriminators give a D loss a bit below ln 2 (0.645 to 0.665), which means D is slightly winning, and the generator still keeps up.
#
# Tips and common mistakes:
#
#    - Always report both seeds. With one seed you cannot tell a stable setting from a lucky one.
#
# Try also (optional):
#
#    - Add lr_d values of 1e-2 and 3e-2 (use `grid_run`). In our tests both were still stable (mean 3.8 to 4.0, std 0.92 to 1.03).

# %% [markdown]
# ### Part 4 checkpoint: A one-dimensional GAN from scratch
#
# You should now have:
#
# - a generator with 49 parameters and a discriminator with 1,153, and the two loss functions written by you (4.1 and 4.2)
# - a trained 1-D GAN with a generated mean within 0.4 of 4 and a spread near 1, the four histograms and the loss curves (4.3)
# - the discriminator's D(4) near 0.5 at the end (4.4)
# - the saturating loss compared with the non-saturating one, from two starting points (4.5)
# - a table of balance settings with two seeds and your reading of which are stable (4.6)

# %% [markdown]
# ## Part 5. A two-dimensional GAN and mode collapse (25 min)
#
# *Lesson section: "Training".* One number was easy. Now the real data are 2,000 points in the plane that sit in **eight small blobs on a circle** (the data of `make_mixture`): a distribution with eight **modes**. ==You will train a GAN that must find all eight, measure how well it does with `modes_covered`, then provoke **mode collapse** on purpose, try three remedies with the same budget, and see that the loss curves do not tell you which run is healthy.== Every run uses 3,000 steps of batch 128 with small networks and takes 10 to 25 seconds on a CPU. To find a healthy baseline we tried several settings first (a lower learning rate of 5e-4 covered only 7 of the 8 blobs in 5,000 steps; with 1e-3, 128 hidden units and Adam betas 0.5 and 0.9 the generator found all eight within 500 steps), so the baseline below is a tuned one. All numbers are for `SEED = 42` unless a seed is named.
#
# ### ✏️ Imagine first
#
# Think of the forger who has to learn eight different kinds of banknote at once. A lazy forger could produce only the easiest one, perfectly, and the detective might be fooled for a while. How many of the eight blobs will a healthy GAN find, and what happens if the forger learns much faster than the detective? Write a number and a sentence in the cell below. You compare them with Steps 5.3 and 5.4.

# %%
# Imagine first: how many of the 8 blobs will a GAN find, and what happens when the generator learns much faster than the discriminator?
my_guess_modes = None       # a number from 0 to 8
my_guess_fast_g = None      # write one sentence, for example "it finds ... because ..."

# %% [markdown]
# ### Step 5.1. The data and how to measure success
#
# The function `modes_covered(points, centres)` returns two numbers: how many of the eight blobs receive at least 2 percent of the points within three standard deviations (0.45) of their centre, and the fraction of all points that land within 0.45 of any centre ("quality"). Together they say whether a generator found all the modes (diversity) and whether its points are accurate (quality). First we test the function on the real data, where the right answer is known.

# %%
X2, blob2, C2 = make_mixture(2000, seed=0)            # 2000 real points from 8 Gaussian blobs on a circle of radius 3
cov_real, q_real = modes_covered(X2, C2)
print("real data: blobs covered %d of 8, fraction of points on a blob %.3f" % (cov_real, q_real))
plt.figure(figsize=(3.8, 3.8)); plt.scatter(X2[:, 0], X2[:, 1], s=3, alpha=0.4); plt.scatter(C2[:, 0], C2[:, 1], c="r", marker="x", s=60)
plt.gca().set_aspect("equal"); plt.title("the real data and the 8 centres"); plt.show()
assert cov_real == 8 and q_real > 0.98
RESULTS["p5_real_quality"] = round(q_real, 3)

# %% [markdown]
# ### ✏️ Your turn: The data and how to measure success
#
# *Exercise 5.1:* Apply `modes_covered` to 2,000 points spread evenly over the square from -4 to 4 in both directions (the code is in the starter). Why it matters: a measuring tool must give a bad score to a bad generator, and noise is the simplest bad generator.

# %%
noise_pts = np.random.RandomState(0).uniform(-4, 4, (2000, 2)).astype(np.float32)   # 2000 points spread evenly over a square
cov_noise, q_noise = None, None     # YOUR CODE: modes_covered(noise_pts, C2)

# %%
if cov_noise is None:
    print("Fill in the two numbers above, then run this cell again.")
else:
    check("uniform noise covers no blob (each blob gets less than 2 percent of the points)", cov_noise == 0, "the 8 small discs fill only a few percent of the square")
    check("fewer than 15 percent of the noise points land on a blob", q_noise < 0.15, "quality is the fraction of points within 3 sigma of ANY centre")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# cov_noise, q_noise = modes_covered(noise_pts, C2)
# print("uniform noise: blobs covered %d of 8, fraction on a blob %.3f" % (cov_noise, q_noise))
# RESULTS["p5_noise_covered"] = cov_noise; RESULTS["p5_noise_quality"] = round(q_noise, 3)
# ```
#
# The tool gives 8 and 0.99 to the real data and 0 and 0.07 to noise, so it can tell a good generator from a bad one. Note that "0 blobs covered" does not mean "no points on blobs": it means no blob has 2 percent of the points.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The real data cover 8 of 8 blobs with 0.99 of the points on a blob (the other 1 percent are the far tails; tolerance: above 0.98). Uniform noise covers 0 of 8 blobs and only 0.07 of its points land on a blob (tolerance: below 0.15), because the eight small discs fill only about 8 percent of the square and each blob gets about 1 percent of the points, under the 2 percent threshold.
#
# Tips and common mistakes:
#
#    - Quality alone is not enough: a generator that puts all points on one blob has a high quality. That is why we also count blobs.

# %% [markdown]
# ### Step 5.2. The networks and the training function
#
# Both networks are the same shape of MLP: two hidden layers of 128 units with LeakyReLU(0.2). The generator maps 4 noise numbers to a point (2 numbers); the discriminator maps a point to a logit. The training function `train_gan_2d` has the settings we will vary: the learning rates, how many D steps and G steps per iteration, the label of the real points (for label smoothing), the start spread of instance noise, the seed. It records `modes_covered` every 250 steps, the two losses at every step and 1,000 generated points at four moments.

# %% [markdown]
# ### ✏️ Your turn: The networks and the training function
#
# *Exercise 5.2:* Write `mlp(n_in, hidden, n_out)` and check the parameter counts. By hand: G has (4 x 128 + 128) + (128 x 128 + 128) + (128 x 2 + 2) = 640 + 16,512 + 258 = 17,410 and D has 384 + 16,512 + 129 = 17,025.

# %%
def mlp(n_in, hidden, n_out):
    """Two hidden layers of the given width with LeakyReLU(0.2) between the layers."""
    return None   # YOUR CODE: nn.Sequential(Linear(n_in, hidden), LeakyReLU(0.2), Linear(hidden, hidden), LeakyReLU(0.2), Linear(hidden, n_out))

# %%
g_, d_ = mlp(4, 128, 2), mlp(2, 128, 1)
if g_ is None:
    print("Fill in mlp above, then run this cell again.")
else:
    check("the generator (4 -> 128 -> 128 -> 2) has 17,410 parameters", count_params(g_) == 17410, "(4*128+128) + (128*128+128) + (128*2+2)")
    check("the discriminator (2 -> 128 -> 128 -> 1) has 17,025 parameters", count_params(d_) == 17025, "(2*128+128) + (128*128+128) + (128*1+1)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def mlp(n_in, hidden, n_out):
#     return nn.Sequential(nn.Linear(n_in, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, n_out))
# print("generator parameters    :", count_params(mlp(4, 128, 2)))
# print("discriminator parameters:", count_params(mlp(2, 128, 1)))
# assert count_params(mlp(4, 128, 2)) == 17410 and count_params(mlp(2, 128, 1)) == 17025
# RESULTS["p5_params_G"] = count_params(mlp(4, 128, 2)); RESULTS["p5_params_D"] = count_params(mlp(2, 128, 1))
# ```
#
# `mlp` builds the same stack for both players. Tiny as they are, 17 thousand parameters each are enough to map noise onto eight blobs.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, n_out))
print("generator parameters    :", count_params(mlp(4, 128, 2)))
print("discriminator parameters:", count_params(mlp(2, 128, 1)))
assert count_params(mlp(4, 128, 2)) == 17410 and count_params(mlp(2, 128, 1)) == 17025
RESULTS["p5_params_G"] = count_params(mlp(4, 128, 2)); RESULTS["p5_params_D"] = count_params(mlp(2, 128, 1))

# %% [markdown]
# The training function. Adam uses betas 0.5 and 0.9 (a lower second-moment memory than the default, which helped in our tests). The discriminator target for real points is `smooth` (1.0 normally, 0.9 for label smoothing); instance noise is added to the inputs of D and decays linearly to zero.

# %%
def train_gan_2d(g_lr=1e-3, d_lr=1e-3, d_steps=1, g_steps=1, steps=3000, noise_dim=4, hidden=128, smooth=1.0, inst_noise=0.0,
                 seed=42, batch=128, rec=250, snaps=(0, 500, 1500, 3000)):
    """Train a GAN on the ring of 8 blobs. smooth: the label of the real points (1.0 = no smoothing). inst_noise: start spread of the
    noise added to the inputs of D, decaying linearly to 0. Returns the models, the history of modes_covered and the losses."""
    torch.manual_seed(seed)
    G, D = mlp(noise_dim, hidden, 2), mlp(2, hidden, 1)
    opt_g = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss(); real = torch.tensor(X2)
    snaps = set(snaps) | {steps}; rec = min(rec, max(steps // 4, 1))
    h = {"step": [], "covered": [], "quality": [], "ld": [], "lg": [], "snap": {}, "G": G, "D": D}
    for s in range(steps + 1):
        if s % rec == 0 or s in snaps:
            with torch.no_grad(): pts = G(torch.randn(2000, noise_dim)).numpy()
            if s in snaps: h["snap"][s] = pts[:1000]
            if s % rec == 0:
                c, q = modes_covered(pts, C2); h["step"].append(s); h["covered"].append(c); h["quality"].append(q)
        if s == steps: break
        sig = inst_noise * (1 - s / steps)                                           # instance noise: large at first, zero at the end
        for _ in range(d_steps):
            xr = real[torch.randint(0, len(real), (batch,))]; xf = G(torch.randn(batch, noise_dim)).detach()
            if sig > 0: xr = xr + sig * torch.randn_like(xr); xf = xf + sig * torch.randn_like(xf)
            ld = bce(D(xr), torch.full((batch, 1), smooth)) + bce(D(xf), torch.zeros(batch, 1))
            opt_d.zero_grad(); ld.backward(); opt_d.step()
        for _ in range(g_steps):
            xf = G(torch.randn(batch, noise_dim))
            if sig > 0: xf = xf + sig * torch.randn_like(xf)
            lg = bce(D(xf), torch.ones(batch, 1)); opt_g.zero_grad(); lg.backward(); opt_g.step()
        h["ld"].append(float(ld) / 2); h["lg"].append(float(lg))
    return h

t0 = time.time(); h_t = train_gan_2d(steps=100, snaps=())
print("100 steps took %.1f seconds, so 3,000 steps take about %.0f seconds" % (time.time() - t0, 30 * (time.time() - t0)))

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The answer prints 17410 and 17025, and the timing line says that 100 steps take about one second, so 3,000 steps take about 10 to 30 seconds (the estimate printed is a rough upper bound).
#
# Tips and common mistakes:
#
#    - `nn.LeakyReLU(0.2)` keeps a small slope for negative inputs, so the discriminator's gradient never dies completely.

# %% [markdown]
# ### Step 5.3. A healthy run
#
# We train with the tuned baseline (both learning rates 1e-3, one step each, noise dimension 4) for 3,000 steps. Four scatter panels show the generated points at steps 0, 500, 1,500 and 3,000 against the eight centres; the curves show the blobs covered and the quality over time.

# %%
runs = {}                                                  # a cache of finished runs: (name, seed) -> history
steps2 = 300 if QUICK else 3000
runs[("healthy", 42)] = hb = train_gan_2d(steps=steps2, seed=SEED)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for ax, s in zip(axes, sorted(hb["snap"])):
    p = hb["snap"][s]; c, q = modes_covered(p, C2)
    ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.4); ax.scatter(C2[:, 0], C2[:, 1], c="r", marker="x", s=30)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.set_title("step %d: %d blobs, %.2f on a blob" % (s, c, q), fontsize=8)
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
ax[0].plot(hb["step"], hb["covered"], marker="o", ms=3); ax[0].set_ylim(-0.2, 8.5); ax[0].set_title("blobs covered (of 8)")
ax[1].plot(hb["step"], hb["quality"], color="tab:green", marker="o", ms=3); ax[1].set_ylim(0, 1); ax[1].set_title("fraction of points on a blob")
for a in ax: a.set_xlabel("step")
plt.tight_layout(); plt.show()
print("healthy run, seed 42: covered by step:", dict(zip(hb["step"][::2], hb["covered"][::2])))
print("final: %d of 8 blobs, %.3f of the points on a blob" % (hb["covered"][-1], hb["quality"][-1]))
RESULTS["p5_healthy_covered"] = hb["covered"][-1]; RESULTS["p5_healthy_quality"] = round(hb["quality"][-1], 3)

# %% [markdown]
# ### ✏️ Your turn: A healthy run
#
# *Exercise 5.3:* From the history `hb`, find the first recorded step at which all 8 blobs were covered, and the largest quality value in the whole history. Why it matters: "how fast" and "how good" are the two questions about any training run.

# %%
first_all = None    # YOUR CODE: the first recorded step at which all 8 blobs were covered (hb["step"] and hb["covered"]), or None if never
best_q = None       # YOUR CODE: the largest "fraction on a blob" in the whole history

# %%
if best_q is None:
    print("Fill in the two variables above, then run this cell again.")
else:
    check("best quality is the maximum of the quality list", abs(best_q - max(hb["quality"])) < 1e-9, "max(hb['quality'])")
    ok_ = [s for s, c in zip(hb["step"], hb["covered"]) if c == 8]
    check("first step with 8 blobs is right", first_all == (ok_[0] if ok_ else None), "loop over zip(hb['step'], hb['covered'])")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ok_ = [s for s, c in zip(hb["step"], hb["covered"]) if c == 8]
# first_all = ok_[0] if ok_ else None
# best_q = max(hb["quality"])
# print("all 8 blobs first covered at step", first_all, "; best quality %.3f" % best_q)
# RESULTS["p5_first_all8"] = first_all
# ```
#
# The generator found every mode without ever being told that eight modes exist: the only signal was the discriminator saying "real" in some regions and "fake" in others. Look at the scatter at step 500: eight clusters already.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Covered by step: 0 at step 0, then 8 from step 500 on (recorded every 250 steps; the first record with all eight is step 500, tolerance: any step up to 1,500). The final result is **8 of 8 blobs with 0.849 of the points on a blob** (tolerance: all 8 blobs, quality between 0.7 and 0.95). The quality does not reach the 0.99 of the real data because the generator's blobs are slightly wider than the real ones and a few points lie between blobs. With seed 1 we also got 8 of 8 (quality 0.83). So the answer to the Imagine-first question is: all eight, in this tuned setting.
#
# Tips and common mistakes:
#
#    - If your run covers 6 or 7 blobs, run longer or change the seed; a healthy run in our tests covered 8 for seeds 42 and 1.
#
# Try also (optional):
#
#    - Change `noise_dim` to 2 or 8 and see whether the result changes. We saw 8 of 8 with noise_dim 2 as well.

# %% [markdown]
# ### Step 5.4. Provoke mode collapse
#
# Now we break the balance in three ways: a fast generator (learning rate 3e-3 and 3 generator steps per iteration), a weak discriminator (learning rate 1e-4) and 5 generator steps per discriminator step. Same budget, same seed. For each run we print the number of blobs covered, the quality, the **spread** (the mean standard deviation of the generated points; the real data have 2.13) and the number of blobs covered at the last four records.

# %%
settings = {
    "fast generator (g_lr 3e-3, 3 G steps)": dict(g_lr=3e-3, d_lr=1e-3, g_steps=3),
    "weak discriminator (d_lr 1e-4)": dict(d_lr=1e-4),
    "5 G steps per D step": dict(g_steps=5),
}
print("seed 42, %d steps:" % steps2)
for name, cfg in settings.items():
    runs[(name, 42)] = hh_ = train_gan_2d(steps=steps2, seed=SEED, **cfg)
    spread_ = hh_["snap"][max(hh_["snap"])].std(axis=0).mean()
    print("  %-40s covered %d of 8, on a blob %.3f, spread of the points %.2f   (covered at the last 4 records: %s)" % (name, hh_["covered"][-1], hh_["quality"][-1], spread_, hh_["covered"][-4:]))
print("  (the real data has a spread of %.2f)" % X2.std(axis=0).mean())
worst_name = "fast generator (g_lr 3e-3, 3 G steps)"             # the clearest collapse: it sits on one blob; the other two settings wander between places
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2)); hw = runs[(worst_name, 42)]
for ax, s in zip(axes, sorted(hw["snap"])):
    p = hw["snap"][s]; c, q = modes_covered(p, C2)
    ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.4, color="tab:red"); ax.scatter(C2[:, 0], C2[:, 1], c="k", marker="x", s=30)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.set_title("step %d: %d blobs" % (s, c), fontsize=8)
plt.suptitle("worst case: " + worst_name, fontsize=9); plt.tight_layout(); plt.show()
RESULTS["p5_real_spread"] = round(float(X2.std(axis=0).mean()), 2)
RESULTS["p5_collapse_fastG"] = runs[("fast generator (g_lr 3e-3, 3 G steps)", 42)]["covered"][-1]
RESULTS["p5_collapse_weakD"] = runs[("weak discriminator (d_lr 1e-4)", 42)]["covered"][-1]
RESULTS["p5_collapse_5G"] = runs[("5 G steps per D step", 42)]["covered"][-1]

# %% [markdown]
# ### ✏️ Your turn: Provoke mode collapse
#
# *Exercise 5.4:* For the fast-generator run, compute `blob_shares`: for each of the 8 centres the fraction of the 1,000 snapshot points within 0.45 of it. Why it matters: it shows with numbers where the collapsed generator put its points.

# %%
blob_shares = None   # YOUR CODE: for the fast-generator run (worst_name), an array of 8 numbers: the fraction of its 1000 snapshot points that lie within 3 sigma (0.45) of each centre

# %%
if blob_shares is None:
    print("Fill in blob_shares above, then run this cell again.")
else:
    pw = runs[(worst_name, 42)]["snap"][max(runs[(worst_name, 42)]["snap"])]
    want_ = (np.linalg.norm(pw[:, None, :] - C2[None], axis=2) <= 0.45).mean(0)
    check("blob_shares has 8 numbers equal to the distance rule", np.shape(blob_shares) == (8,) and np.allclose(blob_shares, want_), "d = np.linalg.norm(p[:, None, :] - C2[None], axis=2); (d <= 0.45).mean(0)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pw = runs[(worst_name, 42)]["snap"][max(runs[(worst_name, 42)]["snap"])]
# blob_shares = (np.linalg.norm(pw[:, None, :] - C2[None], axis=2) <= 0.45).mean(0)
# print("share of the points on each blob:", np.round(blob_shares, 3), " total on a blob: %.3f" % blob_shares.sum())
# print("(a healthy GAN gives about 0.105 to each blob, because 8 blobs share the points)")
# ```
#
# This is **mode collapse**. The generator finds one place where the discriminator is fooled and puts everything there. The discriminator then learns that this place is fake, pushing the generator to another place, and so on: the cloud hops between modes instead of covering them, because the generator never gets a reason to spread out. A weak discriminator gives a bad signal; a too-fast generator moves before the discriminator can answer; either way the game is out of balance.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# All three settings fail (seed 42, 3,000 steps): the fast generator covers 1 of 8 blobs with 0.729 of its points on a blob and a spread of 0.20 (the whole cloud is smaller than one blob); the weak discriminator covers 0 of 8 (quality 0.00, spread 0.39) and the 5-steps setting covers 0 of 8 (quality 0.00, spread 0.36). The last four records show the motion: [1, 0, 1, 1], [0, 0, 1, 0] and [1, 0, 0, 0]: the generator sits on one blob, then leaves. In the blob shares of the fast-generator run one blob has 0.723 of all points and the other seven have 0 (tolerance: one blob above 0.5, the others near 0; which blob it is, and the exact share, differ with the seed). A healthy generator would give each blob about 0.1.
#
# Tips and common mistakes:
#
#    - Collapse in the plane is easy to see in a scatter plot. For pictures you only see it as repeated or nearly identical images, so count a diversity number.
#    - The words "covered 0" for the weak discriminator do not mean the points are random: they are in a tight cloud, between blobs (spread 0.39).

# %% [markdown]
# ### Step 5.5. Fight it: three remedies
#
# We take the collapsed setting (fast generator) and try three remedies with the same budget: (a) a lower generator learning rate (3e-4) with 2 D steps per iteration, (b) one-sided label smoothing (the real label is 0.9 instead of 1) and (c) instance noise (Gaussian noise with standard deviation 0.3, decaying to 0, added to the inputs of D). The cell prints the table for seed 42, the next cell the table for seed 1 and the averages. The second cell takes about a minute.

# %%
remedies = {
    "baseline (healthy)": None,
    "collapsed (fast generator)": dict(g_lr=3e-3, d_lr=1e-3, g_steps=3),
    "(a) g_lr 3e-4 and 2 D steps": dict(g_lr=3e-4, d_lr=1e-3, d_steps=2, g_steps=3),
    "(b) label smoothing 0.9": dict(g_lr=3e-3, d_lr=1e-3, g_steps=3, smooth=0.9),
    "(c) instance noise 0.3": dict(g_lr=3e-3, d_lr=1e-3, g_steps=3, inst_noise=0.3),
}
def get_run(name, seed):
    if (name, seed) not in runs:
        cfg = remedies[name] or {}
        runs[(name, seed)] = train_gan_2d(steps=steps2, seed=seed, **cfg)
    return runs[(name, seed)]
runs[("collapsed (fast generator)", 42)] = runs[("fast generator (g_lr 3e-3, 3 G steps)", 42)]        # the same setting as in Step 5.4
runs[("baseline (healthy)", 42)] = runs[("healthy", 42)]
print("seed 42:    covered   on a blob")
for name in remedies:
    r_ = get_run(name, 42); print("  %-30s %d   %.3f" % (name, r_["covered"][-1], r_["quality"][-1]))

# %%
print("seed 1:     covered   on a blob")
for name in remedies:
    r_ = get_run(name, 1); print("  %-30s %d   %.3f" % (name, r_["covered"][-1], r_["quality"][-1]))
print("average over the two seeds:")
mean_cov = {}
for name in remedies:
    mean_cov[name] = np.mean([get_run(name, s)["covered"][-1] for s in (42, 1)])
    print("  %-30s covered %.1f  on a blob %.3f" % (name, mean_cov[name], np.mean([get_run(name, s)["quality"][-1] for s in (42, 1)])))
RESULTS["p5_remedy_a_cov"] = [get_run(remedies and "(a) g_lr 3e-4 and 2 D steps", s)["covered"][-1] for s in (42, 1)]
RESULTS["p5_remedy_b_cov"] = [get_run("(b) label smoothing 0.9", s)["covered"][-1] for s in (42, 1)]
RESULTS["p5_remedy_c_cov"] = [get_run("(c) instance noise 0.3", s)["covered"][-1] for s in (42, 1)]
RESULTS["p5_collapsed_cov_2seeds"] = [get_run("collapsed (fast generator)", s)["covered"][-1] for s in (42, 1)]
RESULTS["p5_baseline_cov_2seeds"] = [get_run("baseline (healthy)", s)["covered"][-1] for s in (42, 1)]

# %% [markdown]
# ### ✏️ Your turn: Fight it: three remedies
#
# *Exercise 5.5:* Find the name of the remedy with the highest average number of blobs covered over the two seeds. Why it matters: it makes you read the averages instead of a single lucky run.

# %%
best_remedy = None    # YOUR CODE: the name (a string key of remedies) of the REMEDY, (a), (b) or (c), with the highest average covered over the two seeds

# %%
rem_ = [n for n in remedies if n.startswith("(")]
want_best = max(rem_, key=lambda n: mean_cov[n])
if best_remedy is None:
    print("Fill in best_remedy above, then run this cell again.")
else:
    check("best remedy = " + want_best, best_remedy == want_best, "compare mean_cov for the three names that start with '('")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# rem_ = [n for n in remedies if n.startswith("(")]
# best_remedy = max(rem_, key=lambda n: mean_cov[n])
# print("best remedy:", best_remedy, " average blobs covered: %.1f" % mean_cov[best_remedy])
# ```
#
# Label smoothing and instance noise are regularisers for the discriminator: they stop it from becoming overconfident. Here the problem is the opposite: the discriminator is too slow, and making it even less sure does not help. Reducing the generator's advantage does. A remedy has to fit the cause, and "common tricks" can do nothing: report it.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Seed 42: baseline 8 blobs (0.849), collapsed 1 (0.729), remedy (a) 6 (0.837), (b) 0 (0.001), (c) 0 (0.017). Seed 1: baseline 8 (0.825), collapsed 1 (0.566), (a) 6 (0.858), (b) 1 (0.041), (c) 1 (0.274). Averages over the two seeds: baseline 8.0, collapsed 1.0, **(a) 6.0**, (b) 0.5, (c) 0.5. So the best remedy is (a), the one that restores the balance between the players; label smoothing and instance noise **did not help** in this setting (tolerance: (a) 5 to 8 blobs for both seeds; (b) and (c) at most 2). Remedy (a) does not reach the baseline's 8 either.
#
# Tips and common mistakes:
#
#    - Change one thing at a time. Remedy (a) changes two (learning rate and D steps) and we do not know which one matters; the optional exercise tests that.
#
# Try also (optional):
#
#    - Run remedy (a) with only `g_lr=3e-4` (keep `d_steps=1`, `g_steps=3`) and with only `d_steps=2`.

# %% [markdown]
# ### Step 5.6. Losses lie
#
# In classification, a falling loss means a better model. Is it the same in a GAN? We plot the smoothed discriminator and generator losses of the healthy and the collapsed run next to their blobs-covered curves, and print the average of each loss over the last third of training.

# %%
sm_ = lambda v, w=100: np.convolve(v, np.ones(w) / w, mode="valid")
hh_h, hh_c = get_run("baseline (healthy)", 42), get_run("collapsed (fast generator)", 42)
fig, ax = plt.subplots(1, 3, figsize=(12, 2.8))
for hh_, lab, col in ((hh_h, "healthy", "tab:blue"), (hh_c, "collapsed", "tab:red")):
    ax[0].plot(sm_(hh_["ld"]), color=col, label=lab); ax[1].plot(sm_(hh_["lg"]), color=col, label=lab); ax[2].plot(hh_["step"], hh_["covered"], color=col, marker="o", ms=3, label=lab)
ax[0].set_title("D loss (smoothed)"); ax[1].set_title("G loss (smoothed)"); ax[2].set_title("blobs covered (of 8)"); ax[0].legend()
for a in ax: a.set_xlabel("step")
plt.tight_layout(); plt.show()
n_ = len(hh_h["ld"]) // 3
for lab, hh_ in (("healthy", hh_h), ("collapsed", hh_c)):
    print("%-10s D loss (last third) %.3f   G loss (last third) %.3f   blobs covered %d   on a blob %.3f" % (lab, np.mean(hh_["ld"][-n_:]), np.mean(hh_["lg"][-n_:]), hh_["covered"][-1], hh_["quality"][-1]))
RESULTS["p5_ld_healthy"] = round(float(np.mean(hh_h["ld"][-n_:])), 3); RESULTS["p5_ld_collapsed"] = round(float(np.mean(hh_c["ld"][-n_:])), 3)
RESULTS["p5_lg_healthy"] = round(float(np.mean(hh_h["lg"][-n_:])), 3); RESULTS["p5_lg_collapsed"] = round(float(np.mean(hh_c["lg"][-n_:])), 3)

# %% [markdown]
# ### ✏️ Your turn: Losses lie
#
# *Exercise 5.6:* Set `can_tell` to True if the discriminator loss of the collapsed run differs from the healthy one by more than 0.2, else False. Why it matters: it makes you decide, with a number, whether the loss alone could have warned you.

# %%
can_tell = None    # YOUR CODE: True if the D loss of the collapsed run is clearly different from the healthy one (a gap above 0.2), else False

# %%
gap_ = abs(np.mean(hh_h["ld"][-n_:]) - np.mean(hh_c["ld"][-n_:]))
if can_tell is None:
    print("Fill in can_tell above, then run this cell again.")
else:
    check("can_tell agrees with the gap of %.3f" % gap_, bool(can_tell) == bool(gap_ > 0.2), "compare the two last-third means of 'ld'")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# gap_ = abs(np.mean(hh_h["ld"][-n_:]) - np.mean(hh_c["ld"][-n_:]))
# can_tell = bool(gap_ > 0.2)
# print("D loss gap between healthy and collapsed: %.3f -> can_tell = %s" % (gap_, can_tell))
# RESULTS["p5_dloss_gap"] = round(float(gap_), 3)
# ```
#
# In a GAN the loss of each player depends on the other player, so the two numbers measure a game, not progress. Lesson: **always look at samples** and measure diversity and quality with something outside the game (the blobs, the classifier of Part 6).
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Healthy: D loss 0.648, G loss 0.953, 8 blobs, 0.849 on a blob. Collapsed: D loss 0.709, G loss 0.706, 1 blob, 0.729 on a blob. The D losses differ by only 0.061 (tolerance: gap below 0.2) and both are near ln 2 = 0.693, so `can_tell` is False. The generator losses differ more (0.95 against 0.71) but that is no useful alarm either: a lower generator loss here belongs to the worse run. The quality and diversity numbers, and the scatter plots, show the difference clearly.
#
# Tips and common mistakes:
#
#    - Do not stop a GAN because "the loss is flat", or keep it because "the loss looks fine".

# %% [markdown]
# ### Part 5 checkpoint: A two-dimensional GAN and mode collapse
#
# You should now have:
#
# - `modes_covered` applied to the real data (8 of 8, about 0.99 on a blob) and to uniform noise (0 of 8) (5.1)
# - generator and discriminator with 17,410 and 17,025 parameters and the training function (5.2)
# - a healthy run that covers all 8 blobs, with four scatter panels and the curves (5.3)
# - three provoked failures, the share of points per blob of the collapsed run (5.4)
# - a table of remedies for two seeds, with the honest finding of what helps and what does not (5.5)
# - the loss curves of a healthy and a collapsed run and your answer to "can you tell from the losses?" (5.6)

# %% [markdown]
# ## Part 6. A DCGAN on tiny pictures (35 min)
#
# *Lesson sections: "DCGAN" and "Upsampling".* Now the real thing in miniature: a **DCGAN** that draws pictures with transpose convolutions. ==You will do the shape and parameter arithmetic of the lesson's 64 x 64 generator, build a small generator and discriminator for 16 x 16 pictures, train a classifier to act as the judge, train the DCGAN on the five line shapes of Week 4, judge its pictures with numbers (confidence, class entropy, coverage), walk through its latent space, break it in three ways on purpose, and, as a stretch, make it conditional.== The training takes about 20 to 25 seconds per run on a CPU (1,500 steps, batch 64). All numbers are for `SEED = 42`; failures are reported as they are.
#
# ### ✏️ Imagine first
#
# A generator starts from a small 4 x 4 grid of feature maps and must produce a 16 x 16 picture. Each transpose convolution with stride 2 doubles the size. How many such layers does it need? And a poorly trained generator that has seen few steps: will it draw grey blur, random dots, or something that almost looks like the shapes? Write a number and a sentence in the cell below. You compare them with Steps 6.1 and 6.4.

# %%
# Imagine first: how many transpose convolutions turn a 4 x 4 tensor into a 16 x 16 picture, and what will a poorly trained generator draw?
my_guess_layers = None      # a whole number
my_guess_drawing = None     # one sentence, for example "grey blur" or "random dots" or "almost the shapes"

# %% [markdown]
# ### Step 6.1. Shapes and parameters of a generator
#
# The transpose convolution formula from Part 2 is n_out = (n - 1) x s - 2p + k (plus output_padding). The lesson's DCGAN generator turns a vector of 100 numbers into a 64 x 64 x 3 picture: a Linear layer to 4 x 4 x 512 numbers (8,192, so 100 x 8,192 + 8,192 = 827,392 parameters), then four transpose convolutions with k = 4, s = 2, p = 1 that go 512 to 256 to 128 to 64 to 3 channels. The first cell prints the sizes and parameter counts of those layers. After your exercise we build the **small** generator of this lab: 16 noise numbers, a Linear layer to 32 x 4 x 4, `ConvTranspose2d(32, 16, 4, 2, 1)` with BatchNorm and ReLU to 16 x 8 x 8, and `ConvTranspose2d(16, 1, 4, 2, 1)` with Tanh to 1 x 16 x 16.

# %%
def convT_out(n, k, s, p, op=0):
    """Output size of a transpose convolution: (n - 1) * s - 2p + k + op."""
    return (n - 1) * s - 2 * p + k + op

n = 4
for cin, cout in [(512, 256), (256, 128), (128, 64), (64, 3)]:      # the lesson's DCGAN generator: four layers with k = 4, s = 2, p = 1
    n_new = convT_out(n, 4, 2, 1); params = 4 * 4 * cin * cout + cout
    print("convT %3d -> %3d : %2d x %2d -> %2d x %2d   parameters %9s" % (cin, cout, n, n, n_new, n_new, format(params, ",")))
    n = n_new
print("the first layer, a Linear(100, 4*4*512):", format(100 * 8192 + 8192, ","), "parameters")

# %% [markdown]
# ### ✏️ Your turn: Shapes and parameters of a generator
#
# *Exercise 6.1:* Write `convT_shapes(n_in, layers)` that, for a list of layers (c_in, c_out, k, s, p), returns for each layer the output size and the number of parameters k x k x c_in x c_out + c_out. Why it matters: counting the parameters of a design before you build it is how you find out whether it fits in memory and how many pictures it needs.

# %%
def convT_shapes(n_in, layers):
    """layers: list of (c_in, c_out, k, s, p). Returns a list of (output_size, parameters), one per layer, for a square input of size n_in."""
    out = []
    return out   # YOUR CODE: for every layer compute the new size with the formula and the parameters k*k*c_in*c_out + c_out, and append both

# %%
lay_ = [(512, 256, 4, 2, 1), (256, 128, 4, 2, 1), (128, 64, 4, 2, 1), (64, 3, 4, 2, 1)]
r_ = convT_shapes(4, lay_)
if len(r_) == 0:
    print("Fill in convT_shapes above, then run this cell again.")
else:
    check("sizes 8, 16, 32, 64", [a for a, b in r_] == [8, 16, 32, 64], "n_new = (n - 1)*s - 2*p + k")
    check("parameters 2,097,408 / 524,416 / 131,136 / 3,075", [b for a, b in r_] == [2097408, 524416, 131136, 3075], "k*k*c_in*c_out + c_out")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def convT_shapes(n_in, layers):
#     out, n = [], n_in
#     for cin, cout, k, s, p in layers:
#         n = (n - 1) * s - 2 * p + k
#         out.append((n, k * k * cin * cout + cout))
#     return out
# lay_ = [(512, 256, 4, 2, 1), (256, 128, 4, 2, 1), (128, 64, 4, 2, 1), (64, 3, 4, 2, 1)]
# print(convT_shapes(4, lay_))
# assert [a for a, b in convT_shapes(4, lay_)] == [8, 16, 32, 64] and [b for a, b in convT_shapes(4, lay_)] == [2097408, 524416, 131136, 3075]
# print("total of the four transpose convolutions:", format(sum(b for a, b in convT_shapes(4, lay_)), ","))
# RESULTS["p6_lesson_convT_total"] = sum(b for a, b in convT_shapes(4, lay_))
# ```
#
# Each stride-2 transpose convolution doubles the size, so the number of layers is log2 of the growth factor. Most of the lesson generator's parameters sit in the Linear layer and the first transpose convolution, where there are 512 channels. The small generator has 17 thousand parameters in total, about 160 times fewer than the lesson's 3.6 million.
#
# </details>

# %% [markdown]
# The small generator. `verbose=True` prints the shape after each stage. For the option `k=3` we add `output_padding=1`, because a 3 x 3 kernel with stride 2 and padding 1 would otherwise give 7 instead of 8 (Part 2).

# %%
class SmallGen(nn.Module):
    """z (16) -> Linear -> (32, 4, 4) -> ConvTranspose2d -> (16, 8, 8) -> ConvTranspose2d -> Tanh -> (1, 16, 16)."""
    def __init__(self, nz=16, k=4, batchnorm=True):
        super().__init__()
        self.nz = nz
        extra = {"output_padding": 1} if k == 3 else {}              # a 3 x 3 kernel needs output_padding=1 to double the size (Step 6.7)
        self.fc = nn.Linear(nz, 32 * 4 * 4)
        self.up1 = nn.ConvTranspose2d(32, 16, k, 2, 1, **extra)
        self.bn1 = nn.BatchNorm2d(16) if batchnorm else nn.Identity()
        self.up2 = nn.ConvTranspose2d(16, 1, k, 2, 1, **extra)
    def forward(self, z, verbose=False):
        x = F.relu(self.fc(z)).view(-1, 32, 4, 4)
        if verbose: print("after Linear + reshape :", tuple(x.shape))
        x = F.relu(self.bn1(self.up1(x)))
        if verbose: print("after convT 32 -> 16   :", tuple(x.shape))
        x = torch.tanh(self.up2(x))
        if verbose: print("after convT 16 -> 1    :", tuple(x.shape))
        return x

G6 = SmallGen()
_ = G6(torch.randn(2, 16), verbose=True)
by_formula = (16 * 512 + 512) + (4 * 4 * 32 * 16 + 16) + (2 * 16) + (4 * 4 * 16 * 1 + 1)       # Linear + convT + BatchNorm (scale and shift) + convT
print("parameters by formula:", format(by_formula, ","), "  with count_params:", format(count_params(G6), ","))
assert by_formula == count_params(G6)
RESULTS["p6_params_G"] = count_params(G6)

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The four lesson layers give sizes 8, 16, 32, 64 and parameters 2,097,408 / 524,416 / 131,136 / 3,075 (total 2,756,035 for the four transpose convolutions, plus 827,392 for the Linear layer). The small generator prints the shapes (2, 32, 4, 4), (2, 16, 8, 8), (2, 1, 16, 16) and has **17,201 parameters**: Linear 16 x 512 + 512 = 8,704, first transpose convolution 4 x 4 x 32 x 16 + 16 = 8,208, BatchNorm 2 x 16 = 32 (scale and shift), second transpose convolution 4 x 4 x 16 x 1 + 1 = 257. The formula and `count_params` agree.
# So the answer to the Imagine-first question is **two transpose convolutions** for 4 to 8 to 16 (and four for 4 to 64).
#
# Tips and common mistakes:
#
#    - BatchNorm counts 2 parameters per channel (scale and shift). The running averages are buffers, not parameters.
#    - `output_padding` only resolves the ambiguity of the output size; it adds zeros on one side.

# %% [markdown]
# ### Step 6.2. The discriminator and the range of the pictures
#
# The discriminator mirrors the generator with ordinary convolutions of stride 2: `Conv2d(1, 16, 4, 2, 1)` with LeakyReLU to 16 x 8 x 8, `Conv2d(16, 32, 4, 2, 1)` with BatchNorm and LeakyReLU to 32 x 4 x 4, flatten to 512 numbers and a Linear layer to one logit. One more detail: the generator ends in Tanh, so its pictures lie in [-1, 1], while `make_shapes` gives values around 0 and 1. The real pictures must be scaled to the same range or the discriminator would tell real from fake by the range alone. We clip to [0, 1] first (the noise of `make_shapes` can leave the range) and then map to [-1, 1].

# %%
class SmallDisc(nn.Module):
    """(1, 16, 16) -> Conv2d -> (16, 8, 8) -> Conv2d + BatchNorm -> (32, 4, 4) -> flatten (512) -> Linear -> one logit."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 4, 2, 1); self.c2 = nn.Conv2d(16, 32, 4, 2, 1); self.bn = nn.BatchNorm2d(32); self.fc = nn.Linear(32 * 4 * 4, 1)
    def forward(self, x, verbose=False):
        x = F.leaky_relu(self.c1(x), 0.2)
        if verbose: print("after Conv2d 1 -> 16   :", tuple(x.shape))
        x = F.leaky_relu(self.bn(self.c2(x)), 0.2)
        if verbose: print("after Conv2d 16 -> 32  :", tuple(x.shape))
        x = x.flatten(1)
        if verbose: print("after flatten          :", tuple(x.shape))
        return self.fc(x)

Xreal, yreal = make_shapes(1500, noise=0.1, seed=3)          # the real training pictures of the DCGAN: 1,500 noisy shapes
Rt = torch.tensor(np.clip(Xreal, 0, 1) * 2 - 1)               # scaled from [0, 1] to [-1, 1] to match the generator's tanh (Exercise 6.2 writes this as a function)
assert float(Rt.min()) >= -1 and float(Rt.max()) <= 1
print("real pictures:", tuple(Rt.shape), " range [%.2f, %.2f]  mean pixel %.3f" % (float(Rt.min()), float(Rt.max()), float(Rt.mean())))
RESULTS["p6_real_mean_pixel"] = round(float(Rt.mean()), 3)
D6 = SmallDisc(); _ = D6(torch.randn(2, 1, 16, 16), verbose=True)
d_formula = (1 * 4 * 4 * 16 + 16) + (16 * 4 * 4 * 32 + 32) + (2 * 32) + (512 + 1)
print("discriminator parameters by formula:", format(d_formula, ","), "  with count_params:", format(count_params(D6), ","))
assert d_formula == count_params(D6)
RESULTS["p6_params_D"] = count_params(D6)

# %% [markdown]
# ### ✏️ Your turn: The discriminator and the range of the pictures
#
# *Exercise 6.2:* Write `to_tanh(X)`, which clips to [0, 1] and maps to [-1, 1]. Why it matters: a mismatch of ranges is the most common silent bug in a first GAN.

# %%
def to_tanh(X):
    """Map real pictures from [0, 1] to [-1, 1], the range of the generator's tanh."""
    return None   # YOUR CODE: first clip X to [0, 1] (the noise in make_shapes can leave the range), then 2 * X - 1

# %%
Xtmp, _y = make_shapes(100, noise=0.1, seed=3)
T_ = to_tanh(Xtmp)
if T_ is None:
    print("Fill in to_tanh above, then run this cell again.")
else:
    check("the values lie in [-1, 1]", float(np.min(T_)) >= -1.0 and float(np.max(T_)) <= 1.0, "np.clip(X, 0, 1) * 2 - 1")
    check("the shape is unchanged", T_.shape == Xtmp.shape, "only the values change")
    check("black (0) became -1 and white (1) became +1", abs(float(np.min(T_)) + 1.0) < 1e-6 and abs(float(np.max(T_)) - 1.0) < 1e-6, "2 * 0 - 1 = -1 and 2 * 1 - 1 = 1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def to_tanh(X):
#     return np.clip(X, 0, 1) * 2 - 1
# print("range after to_tanh: [%.1f, %.1f]" % (to_tanh(Xtmp).min(), to_tanh(Xtmp).max()))
# assert to_tanh(Xtmp).min() >= -1 and to_tanh(Xtmp).max() <= 1
# ```
#
# The discriminator is a small CNN like the one of Week 4, only with strided convolutions instead of pooling. A mean pixel of -0.81 means that the pictures are mostly black background in the [-1, 1] range: the generator must learn to produce mostly -1.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The 1,500 real training pictures have shape (1500, 1, 16, 16), range [-1.00, 1.00] and a mean pixel of -0.811 (most pixels are dark: only 5 to 9 of the 256 are on a line). The discriminator prints the shapes (2, 16, 8, 8), (2, 32, 4, 4), (2, 512) and has **9,073 parameters**: 1 x 4 x 4 x 16 + 16 = 272, 16 x 4 x 4 x 32 + 32 = 8,224, BatchNorm 64, Linear 512 + 1 = 513.
#
# Tips and common mistakes:
#
#    - Check `min` and `max` of your real batch before training. It takes one line.

# %% [markdown]
# ### Step 6.3. The judge: a classifier that scores generated pictures
#
# How do you know whether generated pictures are good? Looking at 32 of them is one answer, but we also want numbers. We train a small CNN classifier on `make_shapes` (400 training and 200 test pictures, noise 0.1) and **freeze** it. The `judge` function takes a batch of pictures and returns the predicted class, the mean confidence (the largest softmax probability), the histogram of predicted classes, its entropy in nats (the maximum for five classes is ln 5 = 1.609) and the **coverage**: the number of classes that get at least 5 percent of the pictures. A good generator has a high confidence (clear shapes) and a high entropy and coverage 5 (all shapes appear).

# %%
class JudgeCNN(nn.Module):
    """A small CNN as in Week 4 (two convolutions with pooling, one linear layer) that tells the five shapes apart."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1); self.c2 = nn.Conv2d(8, 16, 3, padding=1); self.fc = nn.Linear(16 * 4 * 4, 5)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2); x = F.max_pool2d(F.relu(self.c2(x)), 2); return self.fc(x.flatten(1))

Xj, yj = make_shapes(400, noise=0.1, seed=1); Xjt, yjt = make_shapes(200, noise=0.1, seed=2)
torch.manual_seed(0); judge_net = JudgeCNN()
hist_j = train_classifier(judge_net, np.clip(Xj, 0, 1), yj, np.clip(Xjt, 0, 1), yjt, epochs=20, lr=0.01, seed=0)
judge_net.eval()
for p_ in judge_net.parameters(): p_.requires_grad = False       # frozen: the judge never learns from the generated pictures
print("judge: train accuracy %.3f  test accuracy %.3f" % (hist_j["train_acc"][-1], hist_j["test_acc"][-1]))
RESULTS["p6_judge_test_acc"] = round(hist_j["test_acc"][-1], 3)

# %% [markdown]
# ### ✏️ Your turn: The judge: a classifier that scores generated pictures
#
# *Exercise 6.3:* Write `entropy_nats(hist)` and `coverage(hist, min_share=0.05)`. Why it matters: these two numbers are the diversity side of the judge; confidence alone cannot see mode collapse (Step 7.1 shows this).

# %%
def entropy_nats(hist):
    """Entropy of a class histogram (shares that sum to 1), in nats. Use 0 * ln 0 = 0."""
    return None   # YOUR CODE: minus the sum of p * ln p over the classes with p > 0
def coverage(hist, min_share=0.05):
    """How many classes receive at least min_share of the pictures."""
    return None   # YOUR CODE

# %%
if entropy_nats(np.ones(5) / 5) is None or coverage(np.ones(5) / 5) is None:
    print("Fill in both functions above, then run this cell again.")
else:
    check("entropy of a uniform histogram is ln 5 = 1.609", abs(entropy_nats(np.ones(5) / 5) - math.log(5)) < 1e-6, "-sum(p * log(p))")
    check("entropy of 'always the same class' is 0", abs(entropy_nats(np.array([1.0, 0, 0, 0, 0]))) < 1e-9, "skip the classes with p = 0")
    check("coverage of a uniform histogram is 5", coverage(np.ones(5) / 5) == 5, "count the shares that are >= 0.05")
    check("coverage of [0.9, 0.04, 0.04, 0.02, 0] is 1", coverage(np.array([0.9, 0.04, 0.04, 0.02, 0.0])) == 1, "only the first reaches 5 percent")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def entropy_nats(hist):
#     p = np.asarray(hist, dtype=float); p = p[p > 0]
#     return float(-(p * np.log(p)).sum()) + 0.0
# def coverage(hist, min_share=0.05):
#     return int((np.asarray(hist) >= min_share).sum())
# assert abs(entropy_nats(np.ones(5) / 5) - math.log(5)) < 1e-9 and coverage(np.ones(5) / 5) == 5
# def judge(images, unit=False):
#     """Judge a batch of pictures. images: array (N, 1, 16, 16) in [-1, 1] (generator output) or, with unit=True, in [0, 1]. Returns a dict."""
#     x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
#     if not unit: x = (x + 1) / 2                                      # back to [0, 1], the range the judge was trained on
#     with torch.no_grad(): pr = F.softmax(judge_net(x.clamp(0, 1)), 1)
#     conf, pred = pr.max(1); hist = np.bincount(pred.numpy(), minlength=5) / len(pred)
#     return {"pred": pred.numpy(), "conf": float(conf.mean()), "hist": hist, "entropy": entropy_nats(hist), "coverage": coverage(hist)}
# Xtest_real, ytest_real = make_shapes(500, noise=0.1, seed=4)
# jr = judge(np.clip(Xtest_real, 0, 1), unit=True)
# jn = judge(torch.rand(500, 1, 16, 16) * 2 - 1)                         # uniform noise pictures
# print("%-14s confidence %.3f  entropy %.3f  coverage %d of 5  histogram %s" % ("real pictures", jr["conf"], jr["entropy"], jr["coverage"], np.round(jr["hist"], 2)))
# print("%-14s confidence %.3f  entropy %.3f  coverage %d of 5  histogram %s" % ("uniform noise", jn["conf"], jn["entropy"], jn["coverage"], np.round(jn["hist"], 2)))
# print("accuracy of the judge on these real pictures: %.3f" % float((jr["pred"] == ytest_real).mean()))
# RESULTS["p6_judge_real_conf"] = round(jr["conf"], 3); RESULTS["p6_judge_real_entropy"] = round(jr["entropy"], 3)
# RESULTS["p6_judge_noise_conf"] = round(jn["conf"], 3); RESULTS["p6_judge_noise_entropy"] = round(jn["entropy"], 3); RESULTS["p6_judge_noise_cov"] = jn["coverage"]
# ```
#
# Confidence is not a measure of realism. A frozen classifier tells you what class a picture most resembles, not whether it is a good picture. That is why we use confidence together with entropy and coverage, and why we also look at the pictures. Real image-generation work uses the same idea in more refined form (the Inception Score and FID use a classifier's features), with the same caveat.
#
# </details>

# %% [markdown]
# Below the exercise we run our version of the code above, so that the next cells work even if you skipped the exercise.

# %%
def entropy_nats(hist):
    p = np.asarray(hist, dtype=float); p = p[p > 0]
    return float(-(p * np.log(p)).sum()) + 0.0
def coverage(hist, min_share=0.05):
    return int((np.asarray(hist) >= min_share).sum())
assert abs(entropy_nats(np.ones(5) / 5) - math.log(5)) < 1e-9 and coverage(np.ones(5) / 5) == 5
def judge(images, unit=False):
    """Judge a batch of pictures. images: array (N, 1, 16, 16) in [-1, 1] (generator output) or, with unit=True, in [0, 1]. Returns a dict."""
    x = torch.as_tensor(np.asarray(images), dtype=torch.float32)
    if not unit: x = (x + 1) / 2                                      # back to [0, 1], the range the judge was trained on
    with torch.no_grad(): pr = F.softmax(judge_net(x.clamp(0, 1)), 1)
    conf, pred = pr.max(1); hist = np.bincount(pred.numpy(), minlength=5) / len(pred)
    return {"pred": pred.numpy(), "conf": float(conf.mean()), "hist": hist, "entropy": entropy_nats(hist), "coverage": coverage(hist)}
Xtest_real, ytest_real = make_shapes(500, noise=0.1, seed=4)
jr = judge(np.clip(Xtest_real, 0, 1), unit=True)
jn = judge(torch.rand(500, 1, 16, 16) * 2 - 1)                         # uniform noise pictures
print("%-14s confidence %.3f  entropy %.3f  coverage %d of 5  histogram %s" % ("real pictures", jr["conf"], jr["entropy"], jr["coverage"], np.round(jr["hist"], 2)))
print("%-14s confidence %.3f  entropy %.3f  coverage %d of 5  histogram %s" % ("uniform noise", jn["conf"], jn["entropy"], jn["coverage"], np.round(jn["hist"], 2)))
print("accuracy of the judge on these real pictures: %.3f" % float((jr["pred"] == ytest_real).mean()))
RESULTS["p6_judge_real_conf"] = round(jr["conf"], 3); RESULTS["p6_judge_real_entropy"] = round(jr["entropy"], 3)
RESULTS["p6_judge_noise_conf"] = round(jn["conf"], 3); RESULTS["p6_judge_noise_entropy"] = round(jn["entropy"], 3); RESULTS["p6_judge_noise_cov"] = jn["coverage"]

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The judge reaches a test accuracy of 1.0 (tolerance: above 0.95; the five shapes are easy). On 500 real test pictures: confidence 1.0, entropy 1.609 (the maximum, 1.609), coverage 5 of 5, histogram 0.2 for every class. On 500 uniform-noise pictures: confidence 0.887, entropy 0.363 and coverage 2 of 5. **Surprise:** the judge is confident on noise (0.887). A classifier is forced to choose one of its five classes and it often chooses with a high softmax probability for inputs unlike anything it saw. What gives the noise away is the **entropy**: 88 percent of the noise pictures get the same class (the histogram has a spike at class 3, 'square'). Tolerance: noise confidence above 0.5 is possible; noise entropy below 1.0 and coverage of at most 3.
#
# Tips and common mistakes:
#
#    - Freeze the judge (`requires_grad = False`, `eval()`): it must never learn from the generated pictures.
#    - Do not call a picture real because the judge is sure about it.

# %% [markdown]
# ### Step 6.4. Train the DCGAN
#
# Time to train. Adam with learning rate 2e-4 and betas 0.5 and 0.999 (the DCGAN recipe), batch 64, the non-saturating loss, 1,500 steps on the 1,500 real pictures. Every 250 steps the judge scores 500 generated pictures. The grids below show the same 32 noise vectors (fixed noise) at steps 0, 250, 500 and 1,500, so you see the same "draft" improve. The cell prints the training time.

# %% [markdown]
# First the training function, which also records the judge's verdict during training.

# %%
def train_dcgan(steps=1500, seed=42, g_lr=2e-4, d_lr=2e-4, d_steps=1, k=4, batchnorm=True, nz=16, batch=64, rec=250, snaps=(0, 250, 500, 1500)):
    """Train the small DCGAN on the real shapes. Records the judge's verdict on 500 pictures every `rec` steps."""
    torch.manual_seed(seed)
    G, D = SmallGen(nz, k, batchnorm), SmallDisc()
    opt_g = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999)); opt_d = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); fixed = torch.randn(32, nz, generator=torch.Generator().manual_seed(123))
    snaps = set(snaps) | {steps}; rec = min(rec, max(steps // 4, 1))
    h = {"step": [], "conf": [], "entropy": [], "coverage": [], "ld": [], "lg": [], "grids": {}, "G": G, "D": D}
    for s in range(steps + 1):
        if s % rec == 0 or s in snaps:
            G.eval()                                                      # BatchNorm in evaluation mode for the pictures we look at
            with torch.no_grad():
                if s in snaps: h["grids"][s] = G(fixed).numpy()
                if s % rec == 0:
                    j = judge(G(torch.randn(500, nz)).numpy()); h["step"].append(s); h["conf"].append(j["conf"]); h["entropy"].append(j["entropy"]); h["coverage"].append(j["coverage"])
            G.train()
        if s == steps: break
        for _ in range(d_steps):
            xr = Rt[torch.randint(0, len(Rt), (batch,))]; xf = G(torch.randn(batch, nz)).detach()
            ld = 0.5 * (bce(D(xr), torch.ones(batch, 1)) + bce(D(xf), torch.zeros(batch, 1))); opt_d.zero_grad(); ld.backward(); opt_d.step()
        lg = bce(D(G(torch.randn(batch, nz))), torch.ones(batch, 1)); opt_g.zero_grad(); lg.backward(); opt_g.step()
        h["ld"].append(float(ld)); h["lg"].append(float(lg))
    G.eval()
    return h

t0 = time.time(); _ = train_dcgan(steps=50, snaps=())
print("50 steps took %.1f seconds, so 1,500 steps take about %.0f seconds" % (time.time() - t0, 30 * (time.time() - t0)))

# %%
steps6 = 150 if QUICK else 1500
t0 = time.time(); hd = train_dcgan(steps=steps6, seed=SEED); secs_dc = time.time() - t0
print("training took %.0f seconds" % secs_dc)
for s in sorted(hd["grids"]):
    print("step %d:" % s); show_grid(hd["grids"][s], cols=16, scale=0.75)
fig, ax = plt.subplots(figsize=(6, 2.6)); sm_ = lambda v, w=30: np.convolve(v, np.ones(w) / w, mode="valid")
ax.plot(sm_(hd["ld"]), label="D loss"); ax.plot(sm_(hd["lg"]), label="G loss"); ax.axhline(math.log(2), color="k", ls=":", lw=0.8); ax.legend(); ax.set_xlabel("step"); plt.tight_layout(); plt.show()
print("average D loss over the last 300 steps: %.3f; G loss %.3f" % (np.mean(hd["ld"][-300:]), np.mean(hd["lg"][-300:])))
RESULTS["p6_dloss_end"] = round(float(np.mean(hd["ld"][-300:])), 3)

# %% [markdown]
# ### ✏️ Your turn: Train the DCGAN
#
# *Exercise 6.4:* Train for 500 steps with both learning rates set to 1e-3 and store the judge's last confidence and entropy in a dictionary. Compare with the numbers of the 2e-4 run at step 500. Why it matters: the learning rate is the setting you will change most often, so you should know what a larger one does.

# %%
lr_trial = None    # YOUR CODE: train for 500 steps with g_lr = d_lr = 1e-3 (train_dcgan(steps=500, seed=SEED, g_lr=1e-3, d_lr=1e-3)) and store a dict
                   # {"conf": the last value of its history "conf", "entropy": the last value of "entropy"}

# %%
if lr_trial is None:
    print("Fill in lr_trial above, then run this cell again.")
else:
    check("lr_trial has a confidence between 0 and 1", 0 < lr_trial.get("conf", -1) <= 1, "take h['conf'][-1]")
    check("lr_trial has an entropy between 0 and ln 5", 0 <= lr_trial.get("entropy", -1) <= math.log(5) + 1e-9, "take h['entropy'][-1]")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# h_lr = train_dcgan(steps=100 if QUICK else 500, seed=SEED, g_lr=1e-3, d_lr=1e-3, snaps=())
# lr_trial = {"conf": h_lr["conf"][-1], "entropy": h_lr["entropy"][-1]}
# i_500 = hd["step"].index(500) if 500 in hd["step"] else -1
# print("lr 1e-3 after 500 steps : confidence %.3f  entropy %.3f" % (lr_trial["conf"], lr_trial["entropy"]))
# print("lr 2e-4 after 500 steps : confidence %.3f  entropy %.3f" % (hd["conf"][i_500], hd["entropy"][i_500]))
# RESULTS["p6_lr1e3_500_conf"] = round(lr_trial["conf"], 3); RESULTS["p6_lr1e3_500_entropy"] = round(lr_trial["entropy"], 3)
# ```
#
# The DCGAN learned the five shapes from 1,500 noisy examples in under half a minute. The larger learning rate reaches the same quality earlier but the training can be less stable over longer runs; in this small problem it was fine.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Training takes about 18 to 25 seconds on our CPU (tolerance: up to a minute on a slow machine). The grid at step 0 shows grey-black patches with a faint pattern, the grid at step 250 shows blurry blobs and bars, the grid at step 500 shows lines with the right orientation but ragged, and the grid at step 1,500 shows clean bars, crosses, squares and diagonals. The average D loss over the last 300 steps is 0.148 and the generator loss is about 2.5: the discriminator is winning clearly, and the pictures are still good (again, losses do not say). With learning rate 1e-3 the judge's confidence after 500 steps is 0.982 and the entropy 1.588, against 0.946 and 1.358 for 2e-4 at the same step: the larger rate was faster here (tolerance: both entropies between 1.1 and 1.61 at step 500).
# So the Imagine-first answer: a poorly trained generator (step 0 to 250) draws grey blobs and smears, not random dots; after about 500 steps the shapes are recognisable.
#
# Tips and common mistakes:
#
#    - Put the generator in `eval()` mode before generating pictures to look at (BatchNorm uses its running averages), and back in `train()` mode before training. The training function does this for you.
#    - Use the same fixed noise for every grid, otherwise you compare different pictures.
#
# Try also (optional):
#
#    - Train for 3,000 steps instead of 1,500 and see whether the judge's confidence improves (we expect little change).

# %% [markdown]
# ### Step 6.5. Judge the result
#
# The three plots show the judge's confidence, the class entropy and the coverage at every 250 steps of the run, and the table compares the final generator with the real pictures and with noise. The bar chart shows the class histogram of 500 generated pictures against the real one.

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 2.8))
ax[0].plot(hd["step"], hd["conf"], marker="o", ms=3); ax[0].set_title("judge confidence"); ax[0].set_ylim(0, 1.02)
ax[1].plot(hd["step"], hd["entropy"], marker="o", ms=3, color="tab:green"); ax[1].axhline(math.log(5), color="k", ls="--", lw=0.8); ax[1].set_title("class entropy (max ln 5 = 1.609)")
ax[2].plot(hd["step"], hd["coverage"], marker="o", ms=3, color="tab:red"); ax[2].set_ylim(0, 5.5); ax[2].set_title("classes covered (of 5)")
for a in ax: a.set_xlabel("step")
plt.tight_layout(); plt.show()
print("step   confidence  entropy  coverage")
for s, c, e, v in zip(hd["step"], hd["conf"], hd["entropy"], hd["coverage"]): print("%4d     %.3f      %.3f      %d" % (s, c, e, v))
Gd = hd["G"]; Gd.eval()
with torch.no_grad(): gen500 = Gd(torch.randn(500, 16)).numpy()
jg = judge(gen500)
print("\n%-16s confidence %.3f  entropy %.3f  coverage %d" % ("real pictures", jr["conf"], jr["entropy"], jr["coverage"]))
print("%-16s confidence %.3f  entropy %.3f  coverage %d" % ("generated (end)", jg["conf"], jg["entropy"], jg["coverage"]))
print("%-16s confidence %.3f  entropy %.3f  coverage %d" % ("uniform noise", jn["conf"], jn["entropy"], jn["coverage"]))
print("class histogram of the generated pictures:", dict(zip(SHAPE_NAMES, np.round(jg["hist"], 2))))
plt.figure(figsize=(5, 2.4)); plt.bar(SHAPE_NAMES, jg["hist"], alpha=0.7, label="generated"); plt.bar(SHAPE_NAMES, jr["hist"], alpha=0.4, label="real"); plt.xticks(rotation=20, fontsize=7); plt.legend(); plt.tight_layout(); plt.show()
RESULTS["p6_gen_conf"] = round(jg["conf"], 3); RESULTS["p6_gen_entropy"] = round(jg["entropy"], 3); RESULTS["p6_gen_coverage"] = jg["coverage"]

# %% [markdown]
# ### ✏️ Your turn: Judge the result
#
# *Exercise 6.5:* Find the name of the shape the generator draws least often and the confidence gap between real and generated pictures. Why it matters: a small imbalance between classes is an early sign of partial mode collapse.

# %%
rarest = None    # YOUR CODE: the NAME (from SHAPE_NAMES) of the shape the generator draws least often according to jg["hist"]
gap_conf = None  # YOUR CODE: the judge's confidence on real pictures minus its confidence on the generated pictures (jr["conf"] - jg["conf"])

# %%
if rarest is None or gap_conf is None:
    print("Fill in both variables above, then run this cell again.")
else:
    check("rarest = " + SHAPE_NAMES[int(np.argmin(jg["hist"]))], rarest == SHAPE_NAMES[int(np.argmin(jg["hist"]))], "SHAPE_NAMES[np.argmin(jg['hist'])]")
    check("gap_conf = %.3f" % (jr["conf"] - jg["conf"]), abs(gap_conf - (jr["conf"] - jg["conf"])) < 1e-6, "subtract the generated confidence from the real one")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# rarest = SHAPE_NAMES[int(np.argmin(jg["hist"]))]
# gap_conf = jr["conf"] - jg["conf"]
# print("rarest shape: %s (share %.2f); confidence gap real minus generated: %.3f" % (rarest, jg["hist"].min(), gap_conf))
# RESULTS["p6_rarest_share"] = round(float(jg["hist"].min()), 3)
# ```
#
# The generator covers all five classes in nearly equal shares (a diagonal-biased 0.23 and a bar-shy 0.18 are small deviations with 500 samples; the standard error of a share near 0.2 is 0.018). The numbers say the DCGAN did the job. Whether the pictures are also good to a human eye is for the grid above to say.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Over the run: step 0 confidence 0.994 with entropy 0.00 and coverage 1 (a flat grey picture that the judge calls one class with confidence: again, confidence is not realism); step 250: confidence 0.763, entropy 1.280, coverage 4; step 500: 0.946, 1.358, 4; step 1,000: 0.979, 1.586, 5; step 1,500: 0.969 confidence, entropy 1.606, coverage 5. (Tolerance for the final line: confidence above 0.9, entropy above 1.5, coverage 5.) Compared with real pictures (confidence 1.000, entropy 1.609, coverage 5) and noise (0.887, 0.363, 2) the DCGAN is close to real on all three. The class histogram of the generated pictures is 0.18 horizontal bar, 0.19 vertical bar, 0.19 cross, 0.20 square, 0.23 diagonal against 0.2 each: the rarest shape is the horizontal bar with a share of 0.184 and the confidence gap is 0.031.
#
# Tips and common mistakes:
#
#    - 500 samples give a share with an error of about plus or minus 0.02. Do not read differences smaller than that.

# %% [markdown]
# ### Step 6.6. Walk through the latent space
#
# The 16 noise numbers z are the generator's **latent space**. Two points in it, z_a and z_b, give two pictures. The straight line between them gives 8 more. If the generator learned a smooth map, the pictures should change gradually; if it memorised, they would jump. We also print what the judge says for each of the 8 pictures.

# %%
Gd.eval(); gz = torch.Generator().manual_seed(7)
za, zb = torch.randn(1, 16, generator=gz), torch.randn(1, 16, generator=gz)
alphas = np.linspace(0, 1, 8)
with torch.no_grad(): strip = Gd(torch.cat([(1 - a) * za + a * zb for a in alphas])).numpy()
show_grid(strip, titles=["a=%.2f" % a for a in alphas], cols=8)
pj = judge(strip)
print("judge's class for the 8 steps:", [SHAPE_NAMES[c] for c in pj["pred"]])
print("judge's confidence per step   :", np.round([float(F.softmax(judge_net(((torch.tensor(strip[i:i+1]) + 1) / 2).clamp(0, 1)), 1).max()) for i in range(8)], 2))

# %% [markdown]
# ### ✏️ Your turn: Walk through the latent space
#
# *Exercise 6.6:* Write `lerp(za, zb, n=8)` that returns the n latent vectors on the straight line from za (alpha = 0) to zb (alpha = 1) as a tensor of shape (n, 16). Why it matters: latent interpolation is the standard way to look at what a generator knows.

# %%
def lerp(za, zb, n=8):
    """n latent vectors on the straight line from za (alpha = 0) to zb (alpha = 1)."""
    return None   # YOUR CODE: a tensor of shape (n, 16): (1 - a) * za + a * zb for a = np.linspace(0, 1, n)

# %%
L_ = lerp(za, zb)
if L_ is None:
    print("Fill in lerp above, then run this cell again.")
else:
    check("shape is (8, 16)", tuple(L_.shape) == (8, 16), "stack the 8 vectors with torch.cat")
    check("the first row is za and the last row is zb", torch.allclose(L_[0], za[0]) and torch.allclose(L_[-1], zb[0], atol=1e-6), "alpha = 0 and alpha = 1")
    check("the middle is the average (alpha = 3/7 and 4/7 around the centre)", torch.allclose(L_[3] + L_[4], za[0] + zb[0], atol=1e-5), "rows 3 and 4 are symmetric around the midpoint")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def lerp(za, zb, n=8):
#     return torch.cat([(1 - a) * za + a * zb for a in np.linspace(0, 1, n)])
# L_ = lerp(za, zb)
# with torch.no_grad(): same_ = Gd(L_).numpy()
# print("lerp shape", tuple(L_.shape), "; same pictures as the strip above:", np.allclose(same_, strip, atol=1e-5))
# steps_changed = int((np.diff(pj["pred"]) != 0).sum())
# print("the judge's class changes %d times along the walk" % steps_changed)
# RESULTS["p6_walk_class_changes"] = steps_changed
# ```
#
# Nearby points in latent space give similar pictures and the generator moves between shapes through in-between pictures, which is what a smooth mapping does. This is also the reason a vanilla autoencoder fails to generate (Part 1.5): its latent space has holes between the training points, a GAN is trained so that pictures from every region of the noise space fool the discriminator.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The strip with seed 7 for the latent vectors shows a vertical bar that stays a vertical bar for 5 pictures, then a blend, then a square: judge's classes: vertical bar for alpha = 0 to 4/7, diagonal at 5/7, square at 6/7 and 1. Confidence per step: 1.00, 1.00, 1.00, 0.99, 0.96, 0.52, 0.81, 1.00. The judge's class changes 2 times along the walk (tolerance: 1 to 3 changes for other pairs of z). The least sure picture (0.52, the diagonal) is the one halfway between two shapes. The step from vertical bar to square is not a leap but a short morph through an in-between picture that the judge cannot classify with certainty.
#
# Tips and common mistakes:
#
#    - A straight line between two Gaussian vectors is shorter than a typical vector (the midpoint has a smaller norm); some people interpolate on a sphere instead. For a first look the straight line is fine.

# %% [markdown]
# ### Step 6.7. Break it on purpose
#
# Three breakages, each compared with the healthy run, for the seeds 42 and 1: (a) kernel size 3 instead of 4 in both transpose convolutions (with `output_padding=1` to get the sizes right), the classic cause of checkerboard patterns; (b) no BatchNorm in the generator; (c) an overly strong discriminator (3 D steps per G step and a 5 times larger learning rate, 1e-3 against 2e-4). We report the judge's confidence, entropy and coverage and a new number for checkerboards, the **parity gap**: the absolute mean of the column means weighted by +1, -1, +1, -1 and so on (and the same for the pixel means with a checkerboard weighting); it is 0 for a picture without a two-pixel pattern. A controlled probe of the transpose convolution itself is also printed: a constant picture of ones through a transpose convolution with all weights 1. The seed-42 table takes about 75 seconds, the seed-1 table about 90.

# %%
def parity_gap(imgs):
    """How strongly do the pictures prefer even over odd columns? Returns (column gap, checkerboard gap): the absolute mean of the
    column means (or of the pixel means) weighted by +1, -1, +1, -1, ... A picture without a 2-pixel pattern gives a number near 0."""
    x = np.asarray(imgs)[:, 0]; s = (-1.0) ** np.arange(x.shape[-1])
    return float(abs((x.mean(axis=(0, 1)) * s).mean())), float(abs((x.mean(axis=0) * np.outer(s, s)).mean()))

def gen_pictures(r_, n=500, nz=16):
    r_["G"].eval()
    with torch.no_grad(): return r_["G"](torch.randn(n, nz, generator=torch.Generator().manual_seed(5))).numpy()

breakages = {
    "healthy (k 4, BatchNorm, 1:1)": {},
    "(a) kernel 3 (output_padding 1)": dict(k=3),
    "(b) no BatchNorm in G": dict(batchnorm=False),
    "(c) strong D (3 steps, 5x lr)": dict(d_steps=3, d_lr=1e-3),
}
dc_runs = {("healthy (k 4, BatchNorm, 1:1)", 42): hd}              # the run of Step 6.4 is the healthy baseline for seed 42
def dc_run(name, seed):
    if (name, seed) not in dc_runs: dc_runs[(name, seed)] = train_dcgan(steps=steps6, seed=seed, snaps=(0, steps6), **breakages[name])
    return dc_runs[(name, seed)]
def dc_table(seed):
    print("seed %d:%s confidence  entropy  covered | column gap  checker gap" % (seed, " " * 28))
    for name in breakages:
        r_ = dc_run(name, seed); cg, kg = parity_gap(gen_pictures(r_))
        print("  %-34s    %.3f     %.3f      %d   |   %.4f      %.4f" % (name, r_["conf"][-1], r_["entropy"][-1], r_["coverage"][-1], cg, kg))
dc_table(42)
cg_r, kg_r = parity_gap(np.clip(Xtest_real, 0, 1) * 2 - 1)
print("  %-34s                                |   %.4f      %.4f" % ("(real pictures for comparison)", cg_r, kg_r))
# a controlled probe of the transpose convolution itself: a constant picture of ones through ConvTranspose2d with all weights 1
for k_ in (4, 3):
    ct = nn.ConvTranspose2d(1, 1, k_, 2, 1, bias=False, **({"output_padding": 1} if k_ == 3 else {})); nn.init.constant_(ct.weight, 1.0)
    y_ = ct(torch.ones(1, 1, 8, 8)).detach().numpy()
    print("constant input, kernel %d: row 4 of the output = %s   column gap %.3f" % (k_, y_[0, 0, 4].astype(int).tolist(), parity_gap(y_)[0]))
RESULTS["p6_const_gap_k3"] = round(parity_gap(ct(torch.ones(1, 1, 8, 8)).detach().numpy())[0], 3)

# %% [markdown]
# ### ✏️ Your turn: Break it on purpose
#
# *Exercise 6.7:* Write `parity_ratio(imgs)`: the column gap of the pictures divided by the column gap of the real test pictures. Why it matters: a ratio makes the number comparable (1.0 means the same amount of two-pixel pattern as real pictures).

# %%
def parity_ratio(imgs):
    """The column gap of the pictures divided by the column gap of the real test pictures (1.0 = as much 2-pixel pattern as real pictures)."""
    return None   # YOUR CODE: parity_gap(imgs)[0] divided by parity_gap(np.clip(Xtest_real, 0, 1) * 2 - 1)[0]

# %%
stripes_ = np.zeros((4, 1, 16, 16), np.float32); stripes_[:, :, :, 0::2] = 1          # bright even columns, dark odd columns
g_ = parity_gap(stripes_)
r2_ = parity_ratio(np.random.RandomState(0).rand(50, 1, 16, 16) * 2 - 1)
check("stripes: the column gap is 0.5 (mean of +1 and 0 over the columns)", abs(g_[0] - 0.5) < 1e-6, "mean of colmean * (+1, -1, +1, ...) for colmeans 1, 0, 1, 0, ...")
if r2_ is None:
    print("Fill in parity_ratio above, then run this cell again.")
else:
    check("random noise has almost no 2-pixel pattern (ratio below 1)", r2_ < 1.0, "noise columns average out to about 0")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def parity_ratio(imgs):
#     return parity_gap(imgs)[0] / parity_gap(np.clip(Xtest_real, 0, 1) * 2 - 1)[0]
# for name in breakages:
#     print("%-34s parity ratio %.2f" % (name, parity_ratio(gen_pictures(dc_run(name, 42)))))
# RESULTS["p6_gap_healthy"] = round(parity_gap(gen_pictures(dc_run("healthy (k 4, BatchNorm, 1:1)", 42)))[0], 4)
# RESULTS["p6_gap_k3"] = round(parity_gap(gen_pictures(dc_run("(a) kernel 3 (output_padding 1)", 42)))[0], 4)
# RESULTS["p6_gap_real"] = round(parity_gap(np.clip(Xtest_real, 0, 1) * 2 - 1)[0], 4)
# ```
#
# Two of the three breakages left no measurable trace on this small problem, and only the missing BatchNorm cost some diversity. That is the honest result, and it is useful: a bug that does not show in the numbers of a toy problem can still hurt on a real one, so you test the layer itself (the constant-input probe) as well as the final pictures. The probe confirms the overlap counts of Part 2: with k = 3 and stride 2 the cells alternate between 1 and 2 stamps, with k = 4 and stride 2 every interior cell gets 4.
#
# </details>

# %%
dc_table(1)
for key_, name in zip(["healthy", "a_k3", "b_nobn", "c_strongD"], breakages):
    for sd in (42, 1):
        r_ = dc_run(name, sd); RESULTS["p6_brk_%s_%d" % (key_, sd)] = [round(r_["conf"][-1], 3), round(r_["entropy"][-1], 3), r_["coverage"][-1]]

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Seed 42 (judge confidence, entropy, classes covered, column gap; the real pictures have a column gap of 0.0522): healthy 0.977, 1.600, 5, 0.0485; (a) kernel 3: 0.967, 1.558, 5, 0.0469; (b) no BatchNorm: 0.946, 1.435, 4, 0.0500; (c) strong D: 0.976, 1.608, 5, 0.0430. Seed 1: healthy 0.979, 1.593, 5; (a) 0.962, 1.569, 5; (b) 0.968, 1.325, 4; (c) 0.978, 1.602, 5. The `parity_ratio` values for seed 42 are 0.93, 0.90, 0.96 and 0.82.
#
# What shows and what does not (tolerances: the entropy and coverage of the healthy run as in Step 6.5; the other numbers can move by about 0.1 in entropy between machines):
#
# - **(a) Kernel 3: no visible checkerboard in the trained generator.** Its column gap (0.047) is the same as the healthy generator's (0.049) and the real pictures' (0.052). The controlled probe shows that the layer itself is guilty: a constant picture of ones through the k = 3 layer gives the row 1, 2, 1, 2, 1, 2 ... (column gap 0.629), while k = 4 gives 2, 4, 4, 4 ... 4, 2 (gap 0.000, only the borders are lower). A trained generator can learn weights that cancel the uneven overlap, and the following BatchNorm and ReLU help, so for 16 x 16 line drawings the artefact is learned away. On natural pictures with fine texture it is much harder to hide. We did **not** measure a checkerboard in the trained network; we measured it in the layer.
# - **(b) No BatchNorm: a small, repeatable loss of diversity.** Coverage drops from 5 to 4 classes for both seeds and the entropy from 1.60 and 1.59 to 1.435 and 1.325: one shape fell under the 5 percent threshold. The confidence stays high (0.95 to 0.97).
# - **(c) A strong discriminator (3 D steps, 5 times the rate): no harm here.** Entropy 1.608 and 1.602, coverage 5. In Part 5 the imbalance in favour of the generator was fatal; in favour of the discriminator it was harmless at this size (the lesson warns that a very strong D can stop G from learning in larger models).
#
# Tips and common mistakes:
#
#    - Always check the sizes after changing a kernel size: with k = 3, stride 2 and padding 1 the output of a 4 x 4 input is 7 x 7, not 8 x 8, unless you add `output_padding=1`.
#    - A breakage you cannot measure is still worth reporting: say what you looked at.

# %% [markdown]
# ### Step 6.8. Stretch: a conditional DCGAN
#
# Optional. A conditional GAN gets a label: "draw a cross". The class enters the generator through an embedding (16 numbers) that is added in the first Linear layer, and the discriminator gets the class as five extra constant channels, so it can ask "is this a real picture of **this class**?" We train 1,500 steps, generate 8 pictures for each class and let the judge check whether the requested class is the judged class.

# %%
class CondGen(nn.Module):
    """Like SmallGen, but the class gets its own embedding (16 numbers) that is added to the first Linear layer's input path."""
    def __init__(self, nz=16):
        super().__init__()
        self.emb = nn.Embedding(5, 16); self.fc = nn.Linear(nz, 512); self.fc_y = nn.Linear(16, 512)
        self.up1 = nn.ConvTranspose2d(32, 16, 4, 2, 1); self.bn1 = nn.BatchNorm2d(16); self.up2 = nn.ConvTranspose2d(16, 1, 4, 2, 1)
    def forward(self, z, y):
        x = F.relu(self.fc(z) + self.fc_y(self.emb(y))).view(-1, 32, 4, 4)
        return torch.tanh(self.up2(F.relu(self.bn1(self.up1(x)))))
class CondDisc(nn.Module):
    """Like SmallDisc, but the class is attached to the picture as 5 extra constant channels (one-hot)."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1 + 5, 16, 4, 2, 1); self.c2 = nn.Conv2d(16, 32, 4, 2, 1); self.bn = nn.BatchNorm2d(32); self.fc = nn.Linear(512, 1)
    def forward(self, x, y):
        m = F.one_hot(y, 5).float()[:, :, None, None].expand(-1, -1, 16, 16)
        x = F.leaky_relu(self.c1(torch.cat([x, m], 1)), 0.2); x = F.leaky_relu(self.bn(self.c2(x)), 0.2)
        return self.fc(x.flatten(1))

def train_cgan(steps=1500, seed=42, batch=64):
    torch.manual_seed(seed); G, D = CondGen(), CondDisc(); yt = torch.tensor(yreal)
    og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999)); od = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999)); bce = nn.BCEWithLogitsLoss()
    for s in range(steps):
        i = torch.randint(0, len(Rt), (batch,)); xr, yr = Rt[i], yt[i]
        yf = torch.randint(0, 5, (batch,)); xf = G(torch.randn(batch, 16), yf).detach()
        ld = 0.5 * (bce(D(xr, yr), torch.ones(batch, 1)) + bce(D(xf, yf), torch.zeros(batch, 1))); od.zero_grad(); ld.backward(); od.step()
        yf = torch.randint(0, 5, (batch,)); lg = bce(D(G(torch.randn(batch, 16), yf), yf), torch.ones(batch, 1)); og.zero_grad(); lg.backward(); og.step()
    G.eval(); return G

Gc = train_cgan(steps=150 if QUICK else 1500)
req = torch.arange(5).repeat_interleave(8)                       # 8 pictures for each of the 5 classes
with torch.no_grad(): cp = Gc(torch.randn(40, 16), req).numpy()
show_grid(cp, titles=[SHAPE_NAMES[c][:5] for c in req.tolist()], cols=8, scale=0.8)
judged = judge(cp)["pred"]
print("requested -> judged:")
for c in range(5): print("  %-15s judged as %s" % (SHAPE_NAMES[c], np.bincount(judged[req.numpy() == c], minlength=5)))

# %% [markdown]
# ### ✏️ Your turn: Stretch: a conditional DCGAN
#
# *Exercise 6.8:* Compute `cond_acc`, the share of the 40 generated pictures whose judged class equals the requested class. Why it matters: it turns "the generator obeys the label" into a number.

# %%
cond_acc = None    # YOUR CODE: the share of the 40 pictures whose judged class (judged) equals the requested class (req.numpy())

# %%
if cond_acc is None:
    print("Fill in cond_acc above, then run this cell again.")
else:
    check("cond_acc is the share of matches", abs(cond_acc - float((judged == req.numpy()).mean())) < 1e-9, "np.mean(judged == req.numpy())")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# cond_acc = float((judged == req.numpy()).mean())
# with torch.no_grad(): cp2 = Gc(torch.randn(500, 16), torch.arange(5).repeat(100)).numpy()
# acc500 = float((judge(cp2)["pred"] == torch.arange(5).repeat(100).numpy()).mean())
# print("accuracy of 'requested class = judged class': %.3f on 40 pictures, %.3f on 500 pictures" % (cond_acc, acc500))
# RESULTS["p6_cond_acc"] = round(acc500, 3)
# ```
#
# The accuracy shows whether the generator uses the label. A conditional GAN is the first step towards "text to image", where the label is replaced by a sentence (Week 13 and the lesson on diffusion).
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The grid shows the five rows of 8 pictures, one class per row, each with the requested shape. The judge agrees with every one of the 40 pictures (the table of requested against judged is diagonal: 8 of 8 in each class), so `cond_acc` = 1.000, and on 500 pictures the accuracy is 1.0 (tolerance: above 0.9; chance is 0.2).
#
# Tips and common mistakes:
#
#    - In our first attempt we attached the one-hot vector to the noise and to the flattened features of D; it gave an accuracy of only 0.2, the chance level for five classes: the label was ignored. The version above works. If the numbers in your own variants stay near 0.2, the label has no influence.

# %% [markdown]
# ### Part 6 checkpoint: A DCGAN on tiny pictures
#
# You should now have:
#
# - `convT_shapes` with 8, 16, 32, 64 and the lesson's parameter counts; the small generator (17,201 parameters) and discriminator (9,073) with their shapes (6.1, 6.2)
# - the frozen judge with its accuracy on real pictures, the entropy and coverage functions, and the table real against noise (6.3)
# - a trained DCGAN with grids at four moments, loss curves and the timing (6.4)
# - the judge's confidence, entropy and coverage over the training, and the class histogram (6.5)
# - a latent walk of 8 pictures with the judge's predictions (6.6)
# - the table of three breakages with two seeds and the honest reading (6.7)
# - the optional conditional GAN and its accuracy (6.8)

# %% [markdown]
# ## Part 7. Judge, design and reflect (15 min)
#
# You have now built, trained and broken generators on a line, in a plane and on small pictures. ==This last part asks the question that matters outside the lab: how do you judge a generator with numbers, and what is a responsible way to use one?== You will put diversity and quality side by side, design a use of image generation for your school or community with a design card, write a pre-mortem of the misuse of a face generator, collect your results, and answer four reflection questions.
#
# ### ✏️ Imagine first
#
# Suppose a generator draws exactly the same perfect horizontal bar every time, with no noise. Is it a good generator? Would the judge of Part 6 (confidence, entropy, coverage) notice? Write a yes or no and a few words in the cell below. You compare it with Step 7.1.

# %%
# Imagine first: a generator draws exactly the same perfect horizontal bar every time. Is it a good generator? What would the judge say?
my_guess_good = None      # "yes" or "no", and a few words why

# %% [markdown]
# ### Step 7.1. Diversity and quality as numbers
#
# A good generator needs two things: its pictures must be **good** (quality) and **varied** (diversity). The table puts the two measures of this week side by side: for the 2-D GAN the number of blobs covered (diversity) and the fraction of points on a blob (quality); for the DCGAN the class entropy relative to its maximum ln 5 (diversity) and the judge's confidence (quality). Then you test the claim of the Imagine-first cell.

# %%
rows7 = [
    ("2-D GAN (Part 5), healthy run", "blobs covered %d of 8" % hb["covered"][-1], hb["covered"][-1] / 8, "points on a blob", hb["quality"][-1]),
    ("2-D GAN (Part 5), fast generator", "blobs covered %d of 8" % runs[("collapsed (fast generator)", 42)]["covered"][-1], runs[("collapsed (fast generator)", 42)]["covered"][-1] / 8, "points on a blob", runs[("collapsed (fast generator)", 42)]["quality"][-1]),
    ("DCGAN (Part 6)", "class entropy %.2f of 1.61" % jg["entropy"], jg["entropy"] / math.log(5), "judge confidence", jg["conf"]),
    ("real pictures (reference)", "class entropy %.2f of 1.61" % jr["entropy"], jr["entropy"] / math.log(5), "judge confidence", jr["conf"]),
]
print("%-36s %-28s %-9s %-18s %s" % ("generator", "diversity", "(scaled)", "quality", "value"))
for name, dtxt, dsc, qtxt, q in rows7:
    print("%-36s %-28s %5.2f     %-18s %.3f" % (name, dtxt, dsc, qtxt, q))
RESULTS["p7_dcgan_div"] = round(jg["entropy"] / math.log(5), 3); RESULTS["p7_dcgan_quality"] = round(jg["conf"], 3)

# %% [markdown]
# ### ✏️ Your turn: Diversity and quality as numbers
#
# *Exercise 7.1:* Judge 500 copies of one clean horizontal bar (the starter builds the picture) with `judge(..., unit=True)`. Why it matters: it is the extreme case of mode collapse in the DCGAN world, and shows why one number is never enough.

# %%
clean, _y = make_shapes(5, noise=0.0, seed=0)               # five clean pictures, one of each shape (labels in _y)
bar = clean[_y == 0][:1]                                     # one clean horizontal bar, shape (1, 1, 16, 16)
coll = None    # YOUR CODE: judge(np.repeat(bar, 500, axis=0), unit=True): 500 copies of the same picture

# %%
if coll is None:
    print("Fill in coll above, then run this cell again.")
else:
    check("the class entropy is (almost) 0: only one class appears", coll["entropy"] < 0.01, "all 500 pictures get the same class")
    check("the coverage is 1 of 5", coll["coverage"] == 1, "one class has all the pictures")
    check("the confidence is high (above 0.8): quality alone looks perfect", coll["conf"] > 0.8, "the picture is a clean horizontal bar")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# coll = judge(np.repeat(bar, 500, axis=0), unit=True)
# print("500 copies of one clean bar: confidence %.3f  entropy %.3f  coverage %d of 5" % (coll["conf"], coll["entropy"], coll["coverage"]))
# RESULTS["p7_collapse_conf"] = round(coll["conf"], 3); RESULTS["p7_collapse_entropy"] = round(coll["entropy"], 3)
# ```
#
# Quality without diversity is a forger who knows one banknote; diversity without quality is a forger who draws anything. A report must have both, plus a look at the samples. This is also why research uses pairs of metrics, such as precision and recall of generated distributions.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The table: healthy 2-D GAN 8 of 8 blobs (scaled 1.00) and 0.849 on a blob; fast-generator 2-D GAN 1 of 8 (0.12) and 0.729; DCGAN entropy 1.61 (scaled 0.998) and confidence 0.969; real pictures entropy 1.61 (1.00) and confidence 1.000 (tolerance: DCGAN scaled entropy above 0.95 and confidence above 0.9). The collapsed 2-D generator has a decent quality (0.729) and a poor diversity (0.12). The repeating bar gets a confidence of 1.0 and an entropy of 0.0 (zero) with coverage 1: perfect quality, no diversity. The Imagine-first answer is therefore "no": a generator that repeats one perfect picture is not good, and only the diversity number sees it.
#
# Tips and common mistakes:
#
#    - Keep one threshold per metric fixed before you look at the result (for example coverage at least 5 percent per class), otherwise it is easy to fool yourself.

# %% [markdown]
# ### Step 7.2. Imagination design: a responsible use
#
# Design a use of image generation for your school or community. Some ideas: illustrations for the school newsletter, synthetic training images for a flood-gauge classifier (when real examples of floods are rare), restoring old class photos, sprites for a school game. Fill in a **design card** with seven fields: the purpose, the data and the consent behind it, the architecture (a GAN now, or a diffusion model, which you meet next week), how you will evaluate it (a judge for numbers and a human check), the failure modes, the misuse risks, and how you will tell people that the pictures are generated. The cell prints the card at the end. Think small and concrete: a real purpose, a real list of data.

# %% [markdown]
# ### ✏️ Your turn: Imagination design: a responsible use
#
# *Exercise 7.2:* Fill in all seven fields of `card` with at least one specific sentence each. Why it matters: a generator you cannot describe in seven sentences is not ready to be used.

# %%
card = {
    "purpose": "",             # what will the pictures be used for, and for whom?
    "data_and_consent": "",    # what training pictures do you use, who made them, did they agree?
    "architecture": "",        # GAN now, or a diffusion model (Week 13)? why?
    "evaluation": "",          # which numbers (a judge for quality and diversity) and which human check?
    "failure_modes": "",       # what can go wrong (mode collapse, artefacts, a class that never appears)?
    "misuse_risks": "",        # how could the output or the tool be misused, and by whom?
    "disclosure": "",          # how will you label the pictures as generated?
}

# %%
empty_ = [k for k, v in card.items() if len(v.strip()) < 15]
if empty_:
    print("Fill in these fields (at least a short sentence each), then run this cell again:", empty_)
else:
    check("all 7 fields of the card are filled in", len(card) == 7, "purpose, data_and_consent, architecture, evaluation, failure_modes, misuse_risks, disclosure")
    check("the evaluation field mentions a judge or a human check", any(w in card["evaluation"].lower() for w in ("judge", "classifier", "human", "teacher", "student", "review")), "say how quality and diversity are measured and who looks at the pictures")
    check("the disclosure field says how the pictures are labelled", any(w in card["disclosure"].lower() for w in ("label", "caption", "mark", "tell", "say", "watermark", "note")), "for example a caption 'made by a generator'")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# card = {
#     "purpose": "Simple illustrations of weather symbols for the school newsletter, so that the editors do not need to search for pictures.",
#     "data_and_consent": "Only pictures that the art club drew for this project and agreed in writing to share; no photos of people; the list of files is kept.",
#     "architecture": "A small DCGAN is enough for 16 x 16 symbols; a diffusion model would be for bigger, more detailed pictures (Week 13).",
#     "evaluation": "A classifier trained on the real symbols acts as the judge (confidence for quality, class entropy and coverage for diversity); two students and a teacher look at 100 samples before anything is printed.",
#     "failure_modes": "Mode collapse (one symbol drawn over and over), checkerboard artefacts, a symbol that never appears, a wrong symbol that looks right at a glance.",
#     "misuse_risks": "Someone could present generated pictures as photos of real weather events; pictures could be reused outside the school without the artists' consent.",
#     "disclosure": "Every printed picture gets a caption 'made with an image generator, based on the art club drawings', and the sample sheet is published with the issue.",
# }
# for k, v in card.items(): print("%-18s %s" % (k.upper() + ":", v))
# assert all(len(v) > 15 for v in card.values())
# RESULTS["p7_card_fields"] = len(card)
# ```
#
# The card turns an idea into questions you can answer. The most common gap is the second field (where do the training pictures come from, and did their makers agree?) and the sixth (what could others do with the output?).
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Your card prints seven lines. Our example card is for weather-symbol illustrations in a school newsletter, with consent from the art club, a classifier-judge plus a human review, mode collapse and checkerboards as failure modes, misrepresentation as photographs as the main misuse risk, and a caption on every picture. A good card names the data, the consent, a measurable evaluation and an actual labelling plan; a weak card says "AI pictures for fun" in every field.
#
# Tips and common mistakes:
#
#    - Write what you will do, not what is nice in general. "Respect privacy" is not a plan; "no photos of people" is.

# %% [markdown]
# ### Step 7.3. A misuse pre-mortem
#
# A pre-mortem imagines the failure before it happens. Suppose someone releases a face generator that makes realistic faces of people who do not exist. Write three ways it could be misused, who would be harmed and one mitigation for each.

# %% [markdown]
# ### ✏️ Your turn: A misuse pre-mortem
#
# *Exercise 7.3:* Fill in the three rows of `premortem` (misuse, who is harmed, mitigation). Why it matters: mitigations are cheaper before release than after.

# %%
premortem = [
    {"misuse": "", "who_is_harmed": "", "mitigation": ""},
    {"misuse": "", "who_is_harmed": "", "mitigation": ""},
    {"misuse": "", "who_is_harmed": "", "mitigation": ""},
]

# %%
bad_ = [i + 1 for i, r in enumerate(premortem) if min(len(r["misuse"].strip()), len(r["who_is_harmed"].strip()), len(r["mitigation"].strip())) < 10]
if bad_:
    print("Fill in every field of row(s)", bad_, "(a short phrase is enough), then run this cell again.")
else:
    check("three different misuses", len({r["misuse"].strip().lower() for r in premortem}) == 3, "the three rows must describe three different ways to misuse a face generator")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# premortem = [
#     {"misuse": "Fake profile pictures for scam or bot accounts that look like real people", "who_is_harmed": "People who are cheated or manipulated by the fake accounts", "mitigation": "Invisible and visible watermarks on generated faces; platforms check for them"},
#     {"misuse": "Synthetic faces placed in a made-up news photo or a fake testimony", "who_is_harmed": "Readers who are misled, and real people who are blamed or mimicked", "mitigation": "Disclosure rules and provenance labels on published images; verify with a second source"},
#     {"misuse": "A generator trained on faces scraped without consent can reproduce a real person's face", "who_is_harmed": "The people in the training photos, whose faces and privacy were used without asking", "mitigation": "Train only on consented, documented data; test for memorised faces before release"},
# ]
# print("%-3s %-60s | %-45s | %s" % ("#", "misuse", "who is harmed", "mitigation"))
# for i, r in enumerate(premortem, 1): print("%-3d %-60s | %-45s | %s" % (i, r["misuse"][:58], r["who_is_harmed"][:44], r["mitigation"]))
# ```
#
# Misuse comes from the output (fakes), from the training data (privacy) and from the lack of disclosure. A mitigation can be technical (a watermark), a rule (a label) or a decision not to build.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# Your table has three different rows. Ours: fake profile pictures for scams and bots (harm to people who are cheated; mitigation: watermarks and detection), synthetic faces in made-up news or testimony (harm to readers and to real people who are mimicked; mitigation: provenance labels, verification), and a generator trained on faces taken without consent that reproduces a real person (harm to the people in the data; mitigation: train only on consented data and test for memorisation). A strong answer names a person who is harmed, not only "society".
#
# Tips and common mistakes:
#
#    - A mitigation that depends on every user being honest is not a mitigation.

# %% [markdown]
# ### Step 7.4. Collect your results
#
# The notebook kept your key numbers in the dictionary `RESULTS`. Print the entries of Parts 4 to 7 and check that none of the important ones is missing, so that you can include the output in your submission.

# %%
import json
print("RESULTS collected in Parts 4 to 7 (%d entries):" % len([k for k in RESULTS if k.startswith(("p4", "p5", "p6", "p7"))]))
for k in sorted(RESULTS):
    if k.startswith(("p4", "p5", "p6", "p7")): print("  %-28s %s" % (k, json.dumps(RESULTS[k], default=str)))

# %% [markdown]
# ### ✏️ Your turn: Collect your results
#
# *Exercise 7.4:* Compute `missing`, the list of the four names in `needed` that are not keys of `RESULTS`. Why it matters: a missing key means a cell was skipped.

# %%
needed = ["p4_params_G", "p5_healthy_covered", "p6_gen_entropy", "p7_dcgan_div"]
missing = None    # YOUR CODE: the list of names in `needed` that are NOT keys of RESULTS (an empty list is the goal)

# %%
if missing is None:
    print("Fill in missing above, then run this cell again.")
else:
    check("nothing is missing", missing == [k for k in needed if k not in RESULTS], "[k for k in needed if k not in RESULTS]")
    if missing: print("Run the earlier cells that fill:", missing)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# missing = [k for k in needed if k not in RESULTS]
# print("missing keys:", missing)
# assert not missing, "run the cells of Parts 4 to 7 from the top"
# ```
#
# Having all the numbers in one place is the first step of a report. Next you add what they mean.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# About 70 entries are printed (Part 4: parameters, final mean and std, D(4), the saturating comparison; Part 5: the covered counts and qualities for all runs; Part 6: parameter counts, judge numbers, the breakage table; Part 7: the card). `missing` is the empty list `[]`.
#
# Tips and common mistakes:
#
#    - If a key is missing, run the cells above it, in order, from the top.

# %% [markdown]
# ### Step 7.5. Reflect
#
# Four questions. Write two or three sentences each and cite at least one number you measured in this lab. (1) How do a generator and a discriminator train each other? (2) Why does the original generator loss saturate, and what does the non-saturating loss change? (3) What does a transpose convolution do, and why can it make checkerboard patterns? (4) What did you learn about judging generated images, and why do the losses not tell you the quality?

# %% [markdown]
# ### ✏️ Your turn: Reflect
#
# *Exercise 7.5:* Write your four answers in the dictionary `reflection`. Why it matters: explaining it in your own words is the best test of whether you understood.

# %%
reflection = {
    "q1_two_networks": "",      # how do a generator and a discriminator train each other?
    "q2_saturating": "",        # why does the original generator loss saturate, and what does the non-saturating loss change?
    "q3_transpose_conv": "",    # what does a transpose convolution do, and why can it make checkerboard patterns?
    "q4_judging": "",           # what did you learn about judging generated images, and why do the losses not tell you the quality?
}

# %%
short_ = [k for k, v in reflection.items() if len(v.strip()) < 80]
if short_:
    print("Write two or three sentences (at least 80 characters) for:", short_)
else:
    check("four answers written", len(reflection) == 4, "four questions")
    check("each answer cites a number from the lab", all(any(ch.isdigit() for ch in v) for v in reflection.values()), "mention a measured number, for example the blobs covered or the confidence")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reflection = {
#     "q1_two_networks": "The discriminator learns to separate real from generated points, and its score is the only signal the generator gets: the generator moves its samples to where D says 'real'. When one side stops improving the other has nothing to learn from, which is why 5 generator steps per discriminator step collapsed to 1 blob of 8.",
#     "q2_saturating": "When D is sure a fake is fake, sigmoid(u) is tiny and the gradient of ln(1 - D) is only -0.0025 at u = -6, so G learns nothing. The non-saturating loss -ln D has gradient -0.9975 at the same point, so G learns fastest exactly when it is losing. In 1-D from a start at -8 it reached the data in about 500 to 700 steps, the saturating one in 1,100 to 1,300.",
#     "q3_transpose_conv": "A transpose convolution stamps a copy of the kernel, scaled by each input value, into a larger output, with stride steps between the stamps, and adds the overlaps. With k 3 and stride 2 some cells get 2 stamps and others 1, so the output gets a regular uneven pattern (a checkerboard); k 4 with stride 2 overlaps evenly in the interior.",
#     "q4_judging": "Samples and a judge tell the quality: the D loss of the healthy and the collapsed 2-D GAN were both near 0.7 while one covered 8 blobs and the other 1. I need a diversity number (blobs covered, class entropy) and a quality number (points on a blob, judge confidence), and I must look at the samples too.",
# }
# for k, v in reflection.items(): print(k, "->", v[:100] + "...")
# ```
#
# You can now explain, with measured numbers, why GANs are hard to train and hard to evaluate. The same two problems (balance and measurement) motivate the move to diffusion models next week.
#
# </details>

# %% [markdown]
# **What you should see** (all numbers for `SEED = 42`; a GAN differs a little between machines):
#
# The check passes when each answer has at least 80 characters and mentions a digit. A good answer to (1) uses the 2-D collapse (the fast generator covered 1 of 8 blobs while the D loss stayed near ln 2); to (2) uses the gradients -0.0025 and -0.9975 and the hard start of Step 4.5 (about twice as slow); to (3) the stamp picture and the overlap counts, plus the controlled probe of Step 6.7; to (4) the D loss gap of 0.06 between a healthy and a collapsed run and the entropy and confidence of the judge.
#
# Tips and common mistakes:
#
#    - Quote the number with its tolerance when you can: "8 of 8 blobs for seeds 42 and 1".

# %% [markdown]
# ### Step 7.6. Go further
#
# The University of Amsterdam Deep Learning tutorials contain a notebook on autoencoders, Tutorial 9, "Autoencoders" (`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html`). Its decoder is built from `ConvTranspose2d` layers, the same building block as your generator, and it reconstructs CIFAR-10 pictures from a small latent code. It also shows what you saw in Step 1.5: a vanilla autoencoder is not a generative model, because nothing tells the decoder where the real data live in latent space; a GAN (or a diffusion model) adds that missing signal. The tutorial is by Phillip Lippe, it is free to read, it is not part of the marked work, and no code from it is copied here.

# %% [markdown]
# ### Part 7 checkpoint: Judge, design and reflect
#
# You should now have:
#
# - a table of diversity and quality for the 2-D GAN, the collapsed 2-D GAN and the DCGAN, and the numbers of a generator that repeats one perfect picture (7.1)
# - a design card with seven fields for a responsible use of image generation (7.2)
# - a pre-mortem table with three misuses of a face generator, who is harmed and one mitigation each (7.3)
# - the printed RESULTS (7.4)
# - four reflection answers that cite numbers from the lab (7.5), and a look at one more tutorial (7.6)

# %% [markdown]
# ## Wrap-up
#
# **What you learned in Parts 4 to 7**
#
# - A GAN is two networks playing a game: the discriminator is trained with binary cross-entropy on real and generated data, the generator with the non-saturating loss through the discriminator. A one-dimensional GAN with 49 and 1,153 parameters found the bell curve (mean about 4.0, std about 0.9) in about 1,000 steps, and the original loss was about twice as slow when the generator started far from the data.
# - Balance is everything: the same 2-D GAN covered all 8 blobs with equal learning rates and 1 blob when the generator got 3 steps and a 3 times larger learning rate; restoring the balance helped (6 blobs), label smoothing and instance noise did not, and the D loss of the healthy and the collapsed run was almost the same (0.65 against 0.71).
# - A DCGAN with `ConvTranspose2d` (17,201 and 9,073 parameters) learned the five line shapes in about 25 seconds; a frozen classifier judged it (confidence about 0.97, entropy 1.61 of 1.61, all 5 classes) and showed that confidence alone is not realism. Quality and diversity are two numbers and you need both, plus a look at the samples.
#
# **Reflect.** Double-click this cell and write two sentences: which result in this lab was most different from what you guessed in an Imagine-first cell, and what would you measure next?
#
# **Next week (Week 13)** is diffusion models: another way to invent pictures, with a training that is stabler than a game, and the **capstone** opens.
#
# **Before you close this tab.** Files on the Colab runtime disappear when the session ends. Use **File → Save a copy in Drive** (and download the notebook with its outputs if you want a copy on your computer). Submit your work the way your instructor asks. If you want to go further, read the University of Amsterdam Tutorial 9 of Step 7.6 (credit: Phillip Lippe), and revisit the lesson text for this week.
