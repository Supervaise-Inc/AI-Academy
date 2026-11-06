# %% [markdown]
# # Week 3 — Deep Learning and Backpropagation: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is fine for everything; no GPU is needed and nothing is downloaded (all the small data sets are generated inside the notebook). **Time:** about 3 hours.
#
# This lab is called **Imagine a deep network, then trace its gradients**. First you picture a deep network as a relay of messengers: the signal runs forward, the blame runs backward. Then you build the engine yourself: a tiny automatic-differentiation engine from scratch, backpropagation on a small network by hand and in NumPy (checked against PyTorch and finite differences), and finally you watch gradients vanish and explode in deep networks and repair them with ReLU, initialization, normalization, residual connections and clipping.
#
# ### What you will do
#
# 1. **Part 1. Why depth? Folding, pieces and hierarchy** (25 min): fold a tent to count straight pieces, train shallow and deep networks on a zig-zag, and see layers re-describe the data.
# 2. **Part 2. The forward pass and the computation graph** (20 min): run the lesson's small network by hand and with NumPy, draw its computation graph and count the memory it needs.
# 3. **Part 3. The chain rule and a mini autograd engine** (35 min): check a gradient by hand and with finite differences, then build your own small autograd engine and train with it.
# 4. **Part 4. Backpropagation on layers: the four equations in NumPy** (30 min): code the four backpropagation equations for a whole network and check them against PyTorch.
# 5. **Part 5. Watching gradients vanish and explode** (30 min): measure the gradient in every layer of deep sigmoid, tanh and ReLU networks and watch training stall or blow up.
# 6. **Part 6. Repairing deep networks** (25 min): try ReLU, initialization, normalization, residual connections and gradient clipping, and compare them.
# 7. **Part 7. Design and reflect** (25 min): design a deep network, monitor its gradients, break it, repair it and write your reflection.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells (the exercises) are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried. Your key numbers are collected in a dictionary called `RESULTS` that Part 7 prints.

# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed so your results match the ones in the lesson, and creates a few helpers: `check(...)` to test your answers, `make_xor / make_moons / make_circles` to create small labelled datasets, `plot_decision(...)` to draw what a model has learned, `make_deep(...)` to build a deep network, `layer_grad_norms(...)` to measure how big the gradient is in every layer, and `train_model(...)` to train with a loss curve. Everything in this notebook runs on a CPU in under a few minutes; no downloads are needed.

# %%
import os, gc, math, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
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

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def make_xor(n=200, noise=0.12, seed=0):
    """Four clusters at the corners of a square. Label 1 where exactly one of the two coordinates is positive (XOR)."""
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
    """Colour the plane by `predict` (a function taking an (N, 2) float array and returning probabilities in 0..1) and draw the points."""
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
    """Draw circles for units and lines for weights. layers = e.g. [2, 4, 1]. Returns the number of parameters."""
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

def make_deep(depth=6, width=16, act="relu", init="default", n_in=2, n_out=1, norm=False, residual=False):
    """A stack of `depth` hidden Linear layers (plus an output layer). act: relu, tanh, sigmoid. init: default, xavier, he, small, large.
    Returns an nn.Sequential, or (when residual=True) a small residual network. Layers are reachable with `linear_layers(model)`."""
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    class Res(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Linear(n_in, width)
            s.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
            s.out = nn.Linear(width, n_out)
            s.f = acts[act]()
        def forward(s, x):
            h = s.f(s.inp(x))
            for b in s.blocks:
                h = h + s.f(b(h))
            return s.out(h)
    if residual:
        m = Res()
    else:
        mods, d = [], n_in
        for _ in range(depth):
            mods += [nn.Linear(d, width)] + ([nn.LayerNorm(width)] if norm else []) + [acts[act]()]
            d = width
        mods.append(nn.Linear(d, n_out))
        m = nn.Sequential(*mods)
    for lin in linear_layers(m):
        if init == "xavier": nn.init.xavier_normal_(lin.weight, gain=1.0 if act != "relu" else 1.0)
        elif init == "he": nn.init.kaiming_normal_(lin.weight, nonlinearity="relu" if act == "relu" else "tanh")
        elif init == "small": nn.init.normal_(lin.weight, 0, 0.05)
        elif init == "large": nn.init.normal_(lin.weight, 0, 1.0)
        if init != "default": nn.init.zeros_(lin.bias)
    return m

def linear_layers(model):
    """All nn.Linear layers of a model, in order from the input to the output."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]

def layer_grad_norms(model, X, y):
    """One forward and backward pass on (X, y) with binary cross-entropy. Returns the gradient norm of each Linear layer's weights, input side first."""
    X = torch.as_tensor(X, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    model.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(model(X), y).backward()
    return [float(l.weight.grad.norm()) for l in linear_layers(model)]

def train_model(model, X, y, epochs=300, lr=0.05, opt="sgd", clip=None, Xv=None, yv=None, seed=0):
    """Full-batch training with binary cross-entropy. Returns a dict with the loss curve ("loss"), accuracy ("acc"), and, if validation data are given, "val_loss" and "val_acc"."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    o = torch.optim.Adam(model.parameters(), lr=lr) if opt == "adam" else torch.optim.SGD(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    hist = {"loss": [], "val_loss": [], "grad_first": [], "grad_last": []}
    for _ in range(epochs):
        o.zero_grad(); loss = lossf(model(X), y); loss.backward()
        ls = linear_layers(model)
        hist["grad_first"].append(float(ls[0].weight.grad.norm())); hist["grad_last"].append(float(ls[-1].weight.grad.norm()))
        if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
        o.step(); hist["loss"].append(float(loss))
        if Xv is not None:
            with torch.no_grad(): hist["val_loss"].append(float(lossf(model(torch.as_tensor(Xv, dtype=torch.float32)), torch.as_tensor(yv, dtype=torch.float32).reshape(-1, 1))))
    with torch.no_grad():
        hist["acc"] = float(((model(X) > 0).float() == y).float().mean())
        if Xv is not None: hist["val_acc"] = float(((model(torch.as_tensor(Xv, dtype=torch.float32)) > 0).float() == torch.as_tensor(yv, dtype=torch.float32).reshape(-1, 1)).float().mean())
    return hist

# %% [markdown]
# ## Part 1. Why depth? Folding, pieces and hierarchy (25 min)
#
# *Lesson section: "Why depth".*
#
# Why would anyone stack many layers when one wide layer can, in theory, draw any curve? In this part you answer with your own hands. You fold a simple tent shape on top of itself, count how many straight pieces appear, and see that every extra layer doubles the count, while a single wide layer needs a unit for every piece. Then you train a shallow and a deep network on the same zig-zag and report honestly what happened (it is more subtle than the slogan "deeper is better"), and you look at a deep network as a machine that re-describes its input layer after layer.

# %% [markdown]
# ### ✏️ Imagine first
#
# A deep network folds its input space again and again. Before you run anything, predict two numbers for a network that folds the unit interval **6 times**: how many straight pieces can its output have at most, and how many ReLU units would a network with only **one** hidden layer need to draw the same zig-zag? (Hint: each fold is made of two ReLU units, and one hidden layer with m units can bend the line at most m times.) Write your guesses before you continue; you will compare them with the real numbers in Step 1.2.

# %%
# Imagine first: a tent folded 6 times. How many straight pieces? How many units in ONE hidden layer?
my_pieces = None          # replace None with your guess for the number of straight pieces
my_shallow_units = None   # replace None with your guess for the units one hidden layer needs

# %% [markdown]
# ### Step 1.1. Build the tent from two ReLUs and fold it
#
# The **tent** is the simplest zig-zag: it rises from 0 at x = 0 to 1 at x = 0.5 and falls back to 0 at x = 1, so it has two straight pieces. It can be written with two ReLU units, `tent(x) = 2·relu(x) − 4·relu(x − 0.5)`, which is one tiny hidden layer with two units. A **fold** means feeding the output of a tent back into another tent. The first block builds the tent both ways and asserts that they agree. The second block folds it k = 1, 2, … 6 times, draws the six curves in a 2 × 3 grid and counts the straight pieces numerically: it looks at the sign of the slope between neighbouring points (`np.diff`) and counts how often the sign changes. That count is the number of rising and falling pieces.

# %%
def relu_np(x):
    return np.maximum(0.0, x)

def tent(x):
    """The tent: rises from 0 (at x = 0) to 1 (at x = 0.5) and falls back to 0 (at x = 1)."""
    return 2 * np.minimum(x, 1 - x)

def tent_relu(x):
    """The same tent built from two ReLU units: 2*relu(x) - 4*relu(x - 0.5)."""
    return 2 * relu_np(x) - 4 * relu_np(x - 0.5)

xs = np.linspace(0, 1, 1001)
assert np.allclose(tent(xs), tent_relu(xs)), "the ReLU tent should equal the min-form tent"
print("largest difference between the two tents:", float(np.abs(tent(xs) - tent_relu(xs)).max()))
print("tent at x = 0, 0.25, 0.5, 0.75, 1:", tent_relu(np.array([0, 0.25, 0.5, 0.75, 1.0])).tolist())

# %%
def fold(x, k):
    """Apply the ReLU tent k times, one after the other (k layers of two units)."""
    for _ in range(k):
        x = tent_relu(x)
    return x

def count_pieces(y):
    """Count the rising and falling pieces: 1 + the number of times the slope changes sign."""
    slopes = np.sign(np.diff(y))
    slopes = slopes[slopes != 0]
    return 1 + int((slopes[1:] != slopes[:-1]).sum())

fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True, sharey=True)
pieces = {}
for k, ax in zip(range(1, 7), axes.ravel()):
    y = fold(xs, k)
    pieces[k] = count_pieces(y)
    ax.plot(xs, y, c="tab:blue")
    ax.set_title(f"folded {k} time(s): {pieces[k]} pieces")
plt.tight_layout(); plt.show()

for k, n in pieces.items():
    print(f"k = {k}: {n:>3} pieces   (2**k = {2 ** k})")
assert all(pieces[k] == 2 ** k for k in pieces), "every fold should double the number of pieces"

# %% [markdown]
# **What you should see:** You should see `largest difference between the two tents: 0.0`, the five tent values `[0.0, 0.5, 1.0, 0.5, 0.0]`, and a 2 × 3 figure in which the curve becomes more and more jagged: one tent, two tents, four tents, and so on up to 32 teeth in the last panel. The counts are exactly 2, 4, 8, 16, 32, 64 pieces for k = 1 … 6, which is 2**k. Six layers of two units each (12 units in total) produce a curve with 64 straight pieces. The assert at the end passes silently; it would stop the cell with an error if a count were wrong.

# %% [markdown]
# **Tips and common mistakes**
#
# - `np.diff(y)` gives the slope times the step size between neighbouring points; `np.sign` keeps only whether the curve goes up (+1), down (−1) or stays flat (0). Counting sign changes is a cheap way to count the pieces of a zig-zag.
# - The grid has 1001 points. The counting only works while the grid is finer than the narrowest piece: for k = 6 a piece is 1/64 = 0.0156 wide, about 15 grid steps, which is fine. For k = 12 you would need a finer grid, or the count would be too small.
# - `fold(x, k)` applies the same two-unit layer k times. It is a deep network of width 2, and every layer re-uses the same weights. Real networks learn different weights per layer, but the folding idea is the same.
#
# **Try also (optional)**
#
# - Plot `tent_relu(xs)` and `2 * relu_np(xs)` and `-4 * relu_np(xs - 0.5)` on one axis to see how the two ReLU units add up to a tent.
# - Change the 1001 in `np.linspace` to 101 and look at the counts for k = 5 and 6. What goes wrong when the grid is too coarse?

# %% [markdown]
# ### ✏️ Exercise 1.1: Fold it seven times
#
# Your task: fold the tent **seven** times with `fold(xs, 7)` and count the pieces with `count_pieces`. First predict the answer in your head, then store the count in `pieces_7`. Compare it with your `my_pieces` from *Imagine first* (you guessed for six folds): by how much was your guess off? Why it matters: this doubling per layer is the reason why depth can describe very complicated functions with very few units.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
pieces_7 = None   # the number of straight pieces after 7 folds

# %%
if pieces_7 is None:
    print("Fill in pieces_7 above, then run this cell again.")
else:
    check("fold 7 times gives 2**7 pieces", pieces_7 == 128, "Each fold doubles the count: 2 * 2 * 2 * 2 * 2 * 2 * 2.")
    if my_pieces is not None:
        check("your guess for 6 folds (Imagine first) was 64", my_pieces == 64, f"You guessed {my_pieces}; the real number for 6 folds is {pieces[6]}.")
RESULTS["part1_pieces_7"] = pieces_7

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pieces_7 = count_pieces(fold(xs, 7))   # 128
# ```
#
# Seven folds make 128 pieces from only 14 ReLU units. The doubling is exact because each tent maps every rising or falling piece onto a full up-and-down copy of itself.
#
# </details>

# %% [markdown]
# ### Step 1.2. Count pieces and compare with a single wide layer
#
# How many units would a **one-hidden-layer** network need for the same zig-zag? A one-layer ReLU network with m units is `Σ cᵢ·relu(wᵢx + bᵢ) + d`: each unit adds at most one bend, so the curve has at most **m + 1** straight pieces. To reach 2**k pieces it therefore needs at least 2**k − 1 units, while the deep tent network needs only 2k. The first block fills the table for k = 1 … 10 and draws the units against k on a logarithmic y-axis. The second block tests the bound m + 1 with 200 random one-layer networks, so that you see it hold instead of just believing it.

# %%
ks = np.arange(1, 11)
pieces_k = 2 ** ks            # straight pieces after k folds
deep_units = 2 * ks           # the deep tent network: 2 units per layer, k layers
shallow_units = pieces_k - 1  # a one-layer network needs at least (pieces - 1) units

print(f"{'k':>3} {'pieces':>7} {'deep units':>11} {'shallow units (at least)':>26}")
for k, p, d, s in zip(ks, pieces_k, deep_units, shallow_units):
    print(f"{k:>3} {p:>7} {d:>11} {s:>26}")

plt.figure(figsize=(6, 3.6))
plt.semilogy(ks, shallow_units, "o-", c="tab:red", label="one hidden layer (at least 2**k - 1)")
plt.semilogy(ks, deep_units, "s-", c="tab:blue", label="deep tent network (2k)")
plt.xlabel("number of folds k"); plt.ylabel("units needed (log scale)"); plt.grid(alpha=0.3); plt.legend()
plt.title("Same zig-zag, two ways to build it"); plt.tight_layout(); plt.show()

# %%
# Test the bound "at most m + 1 pieces" with random one-layer ReLU networks
rng = np.random.RandomState(1)
grid = np.linspace(-3, 3, 20001)
m = 10
most = 0
for _ in range(200):
    w, b, c = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m)
    y = (c * relu_np(np.outer(grid, w) + b)).sum(axis=1)       # a one-layer network with m units
    most = max(most, count_pieces(y))
print(f"largest number of rising/falling pieces found in 200 random networks with m = {m} units: {most}  (the bound says at most {m + 1})")
assert most <= m + 1
if my_shallow_units is not None:
    print(f"you guessed {my_shallow_units} units for one layer (6 folds); the bound says at least {2 ** 6 - 1}")

# %% [markdown]
# **What you should see:** You should see a table whose pieces column doubles each row (2, 4, 8 … 1024), whose `deep units` column grows by 2 each row (2, 4, 6 … 20) and whose last column grows as 1, 3, 7 … 1023. At k = 10 the deep network needs 20 units and the one-layer network at least 1023, about 51 times more. On the log plot the red line is a straight line rising steeply (exponential growth) and the blue line is almost flat by comparison. The random test finds at most 5 pieces for m = 10 units: random networks stay well below the bound of 11, which is a ceiling that only carefully chosen weights reach. Compare with your *Imagine first* guesses: six folds gave 64 pieces, 12 deep units, and at least 63 units for one layer.

# %% [markdown]
# **Tips and common mistakes**
#
# - This is a statement about what a network **can** represent with a given number of units, not about what gradient descent will find. Step 1.3 shows the difference.
# - The bound m + 1 counts rising and falling pieces. A real one-layer network can have a few bends fewer than m (two units may bend at the same place), never more.
# - `np.outer(grid, w) + b` builds a 20001 × 10 table of pre-activations; `(c * relu(...)).sum(axis=1)` adds the m units. If you see a shape error, print `np.outer(grid, w).shape`.
#
# **Try also (optional)**
#
# - Change `m` to 3 and check that you never see more than 4 pieces.
# - Add a third curve to the log plot for a network with **two** hidden layers of width `w`: it can reach about `(w/2 + 1)**2 …` pieces. You do not need the formula; just notice that depth multiplies where width adds.

# %% [markdown]
# ### ✏️ Exercise 1.2: A million pieces
#
# Your task: using the formulas above, work out how many folds k you need to reach at least **one million** straight pieces (the smallest k with 2**k ≥ 1 000 000), and how many ReLU units the deep tent network uses for it. Store them in `k_million` and `deep_units_million`. Then store in `shallow_units_million` the least number of units a one-hidden-layer network would need. Why it matters: a deep network gets exponentially many pieces from a linear number of units; this is the mathematical heart of the lesson section "Why depth".
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
k_million = None             # smallest k with 2**k >= 1_000_000
deep_units_million = None    # units of the deep tent network for that k
shallow_units_million = None # least units for ONE hidden layer to reach 1_000_000 pieces

# %%
if k_million is None or deep_units_million is None or shallow_units_million is None:
    print("Fill in the three numbers above, then run this cell again.")
else:
    check("k = 20 folds", k_million == 20, "2**19 = 524288 is not enough, 2**20 = 1048576 is.")
    check("40 deep units", deep_units_million == 40, "Two units per fold: 2 * 20.")
    check("at least 1_000_000 - 1 shallow units", shallow_units_million >= 999_999, "One layer with m units has at most m + 1 pieces, so m >= pieces - 1.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# k_million = int(np.ceil(np.log2(1_000_000)))   # 20
# deep_units_million = 2 * k_million               # 40
# shallow_units_million = 1_000_000 - 1            # 999,999
# ```
#
# Twenty folds (40 units) give 1,048,576 pieces, while one hidden layer needs almost a million units. That is the difference between a laptop and a data centre.
#
# </details>

# %% [markdown]
# ### Step 1.3. Learn a zig-zag with PyTorch: shallow against deep
#
# Now let a network **learn** the zig-zag instead of building it by hand. The target is the tent folded three times (8 pieces) on 200 random points in [0, 1] with a little noise. We train three kinds of network with Adam for 2000 epochs and compare them on a clean grid of 400 points: (a) a shallow network `1 → 33 → 1` with 100 parameters, (b) a shallow network `1 → 133 → 1` with 400 parameters, and (c) a deep narrow network `1 → 6 → 6 → 6 → 1` with 103 parameters, so that (a) and (c) have the same budget. Each is trained with three different seeds and we report the mean and the range of the test error. Then you see a second, surprising result: a deep network that we set by hand.

# %%
rng = np.random.RandomState(0)
x_train = np.sort(rng.uniform(0, 1, 200))
y_train = fold(x_train, 3) + rng.normal(0, 0.02, 200)       # 8 pieces plus a little noise
x_test = np.linspace(0, 1, 400)
y_test = fold(x_test, 3)                                     # the clean zig-zag

def to_t(a):
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
Xtr, Ytr, Xte, Yte = to_t(x_train), to_t(y_train), to_t(x_test), to_t(y_test)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def make_net(kind, seed):
    torch.manual_seed(seed)
    if kind == "shallow 100":
        return nn.Sequential(nn.Linear(1, 33), nn.ReLU(), nn.Linear(33, 1))
    if kind == "shallow 400":
        return nn.Sequential(nn.Linear(1, 133), nn.ReLU(), nn.Linear(133, 1))
    return nn.Sequential(nn.Linear(1, 6), nn.ReLU(), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 1))

def fit(model, epochs=2000, lr=0.01):
    """Train with Adam on the noisy points; return the loss curve and the error on the clean grid."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(Xtr) - Ytr) ** 2).mean()
        loss.backward(); opt.step()
        losses.append(float(loss))
    with torch.no_grad():
        test_mse = float(((model(Xte) - Yte) ** 2).mean())
    return losses, test_mse

print("parameters:", {k: count_params(make_net(k, 0)) for k in ("shallow 100", "shallow 400", "deep 103")})

# %%
kinds = ["shallow 100", "shallow 400", "deep 103"]
runs = {k: [] for k in kinds}
for kind in kinds:
    for seed in (0, 1, 2):
        model = make_net(kind, seed)
        losses, test_mse = fit(model)
        runs[kind].append((test_mse, losses, model))

print(f"{'network':<12} {'params':>6} {'mean test MSE':>14} {'best':>9} {'worst':>9}")
for kind in kinds:
    errs = [r[0] for r in runs[kind] ]
    print(f"{kind:<12} {count_params(runs[kind][0][2]):>6} {np.mean(errs):>14.4f} {min(errs):>9.4f} {max(errs):>9.4f}")

fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
axes[0].plot(x_test, y_test, "k", lw=2, label="target")
axes[0].scatter(x_train, y_train, s=6, c="gray", alpha=0.5)
for kind, colr in zip(kinds, ("tab:red", "tab:green", "tab:blue")):
    best = min(runs[kind], key=lambda r: r[0])
    with torch.no_grad():
        axes[0].plot(x_test, best[2](Xte).numpy().ravel(), c=colr, label=f"{kind} (best seed)")
    axes[1 + kinds.index(kind)].semilogy(best[1], c=colr); axes[1 + kinds.index(kind)].set_title(f"loss: {kind}")
axes[0].legend(fontsize=7); axes[0].set_title("fits (best seed of each)")
plt.tight_layout(); plt.show()
RESULTS["part1_zigzag_mean_test_mse"] = {k: round(float(np.mean([r[0] for r in runs[k] ])), 4) for k in kinds}

# %%
# A deep network SET BY HAND: 1 -> 2 -> 2 -> 2 -> 1, every layer is the same tent, 19 parameters in total
hand = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
with torch.no_grad():
    hand[0].weight.copy_(torch.tensor( [ [1.0], [1.0] ] ));  hand[0].bias.copy_(torch.tensor([0.0, -0.5]))
    for layer in (hand[2], hand[4]):
        layer.weight.copy_(torch.tensor( [ [2.0, -4.0], [2.0, -4.0] ] ));  layer.bias.copy_(torch.tensor([0.0, -0.5]))
    hand[6].weight.copy_(torch.tensor( [ [2.0, -4.0] ] ));  hand[6].bias.copy_(torch.tensor([0.0]))
with torch.no_grad():
    hand_mse = float(((hand(Xte) - Yte) ** 2).mean())
print("hand-built deep net:", count_params(hand), "parameters, test MSE =", f"{hand_mse:.1e}")

small = []
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 6), nn.ReLU(), nn.Linear(6, 1))      # 19 parameters, shallow
    small.append(fit(net)[1])
print("trained shallow net with the same 19 parameters (3 seeds): test MSE =", [round(e, 4) for e in small])
assert hand_mse < 1e-10

# %% [markdown]
# **What you should see:** The first block prints the parameter counts `{'shallow 100': 100, 'shallow 400': 400, 'deep 103': 103}`. The second prints a table with the mean, the best and the worst test error over three seeds. The measured means are {'shallow 100': 0.0589, 'shallow 400': 0.0018, 'deep 103': 0.0741}. Read this honestly. The wide shallow network with 400 parameters is clearly the best (mean 0.0018); the two networks with about 100 parameters are far behind (0.059 and 0.074), and in this experiment the deep narrow ReLU network (mean 0.074) is **not** better than the shallow one at the same budget (0.059). Gradient descent did not find the folding solution from a random start. The third block shows that the folding solution exists: the hand-built network has only 19 parameters and a test MSE of 1e-14 (zero up to rounding), while a trained shallow net with the same 19 parameters stays at about 0.078. So depth is a matter of **what can be represented per parameter**; whether training **finds** it is a separate question. This is one small example, with three seeds: do not turn it into a law.

# %% [markdown]
# **Tips and common mistakes**
#
# - Why the deep ReLU network struggles: with only 6 units per layer a few dead ReLUs (units that output 0 for every input) already destroy the folding, and every layer's gradient is a product of the layers above it. You meet exactly this in Parts 4 to 6.
# - Seeds matter. `make_net(kind, seed)` fixes the initial weights, so you can reproduce a run. A single lucky seed could make the deep network look good or bad; three seeds with the range are the minimum you should report.
# - The test error is measured on a clean grid, not on the noisy training points, so it tells you how close the network is to the true zig-zag rather than to the noise.
# - Do not compare the loss curves of different networks on their own: look at the test error as well.
#
# **Try also (optional)**
#
# - Change the number of epochs from 2000 to 6000 for the deep network. In our measurement (seeds 0 to 5, not part of the block) the deep ReLU net stayed between 0.06 and 0.09 for every seed: more training did not help.
# - Build the deep net with `nn.Tanh()` instead of `nn.ReLU()` and train it with the same seeds. With 3000 epochs and seeds 0 to 3 we measured test errors of 0.002, 0.032, 0.033 and 0.027: one seed in four found a good fit, the others got stuck at about 0.03. Depth plus a smooth activation behaves differently, but it is still luck of the start.

# %% [markdown]
# ### ✏️ Exercise 1.3: Count parameters
#
# Your task: write the parameter count of a network `1 → 8 → 8 → 1` **by hand** (weights plus biases of each layer) and store it in `params_1_8_8_1`. Then build the network with `nn.Sequential` and compare it with `count_params`. Why it matters: when you compare networks you must compare their budgets, and the count is just weights times units plus one bias per unit, layer after layer.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
params_1_8_8_1 = None   # by hand: (1*8 + 8) + (8*8 + 8) + (8*1 + 1)

# %%
if params_1_8_8_1 is None:
    print("Fill in params_1_8_8_1 above, then run this cell again.")
else:
    real = count_params(nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1)))
    check("matches count_params of the real network", params_1_8_8_1 == real, f"The real network has {real} parameters; recount each layer as inputs * outputs + outputs.")
    check("the hand-built tent network has 19 parameters", count_params(hand) == 19, "4 + 6 + 6 + 3.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# params_1_8_8_1 = (1 * 8 + 8) + (8 * 8 + 8) + (8 * 1 + 1)   # 16 + 72 + 9 = 97
# ```
#
# Each layer has inputs × outputs weights plus one bias per output unit: 16 + 72 + 9 = 97. The same formula gives 4 + 6 + 6 + 3 = 19 for the hand-built network.
#
# </details>

# %% [markdown]
# ### Step 1.4. Hierarchy: layers re-describe the data
#
# The second argument for depth is **hierarchy**: a layer does not only compute, it gives the next layer a new description of the data. The data set `make_circles` is a small disc inside a ring, which no straight line can separate. The block trains one-hidden-layer networks with 1, 2, 3 and 8 tanh units and two-layer networks with 2 + 2 and 3 + 3 units, each with five seeds, and prints the accuracies. Then it draws the decision regions of the 2-unit and the 2 + 2 network and plots what the **first hidden layer** of the 2 + 2 network does to the points: the two outputs of that layer are the new coordinates in which the second layer works.

# %%
Xc, yc = make_circles(200, seed=0)

def make_mlp(hidden, seed):
    """Fully connected tanh network 2 -> hidden[0] -> hidden[1] ... -> 1 (one output logit)."""
    torch.manual_seed(seed)
    layers, d = [], 2
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 1))

def fit_circles(hidden, seed, epochs=1500):
    model = make_mlp(hidden, seed)
    hist = train_model(model, Xc, yc, epochs=epochs, lr=0.03, opt="adam", seed=seed)
    return model, hist["acc"]

configs = [ [1], [2], [3], [8], [2, 2], [3, 3] ]
table = {}
for hidden in configs:
    table[tuple(hidden)] = [fit_circles(hidden, s)[1] for s in range(5)]

print(f"{'hidden layers':<14} {'mean acc':>9}   accuracy of the 5 seeds")
for hidden, accs in table.items():
    print(f"{str(list(hidden)):<14} {np.mean(accs):>9.3f}   {[round(a, 3) for a in accs]}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, hidden in zip(axes[:2], ( [2], [2, 2] )):
    model, acc = fit_circles(hidden, 0)
    predict = lambda P, m=model: torch.sigmoid(m(torch.tensor(P))).detach().numpy().ravel()
    plot_decision(predict, Xc, yc, ax=ax, title=f"hidden {hidden}: accuracy {acc:.3f}")

model22, _ = fit_circles( [2, 2], 0)
with torch.no_grad():
    h1 = model22[1](model22[0](torch.tensor(Xc))).numpy()          # output of the first hidden layer
axes[2].scatter(h1[yc == 0, 0], h1[yc == 0, 1], marker="s", c="tab:blue", edgecolor="k", s=26, label="class 0 (ring)")
axes[2].scatter(h1[yc == 1, 0], h1[yc == 1, 1], marker="o", c="tab:orange", edgecolor="k", s=26, label="class 1 (disc)")
axes[2].set_title("the points as the first hidden layer sees them"); axes[2].set_xlabel("unit 1"); axes[2].set_ylabel("unit 2"); axes[2].legend(fontsize=7)
plt.tight_layout(); plt.show()
RESULTS["part1_circles_mean_acc"] = {str(list(k)): round(float(np.mean(v)), 3) for k, v in table.items()}

# %% [markdown]
# **What you should see:** The table shows (mean accuracy over five seeds): {'[1]': 0.704, '[2]': 0.86, '[3]': 1.0, '[8]': 1.0, '[2, 2]': 0.931, '[3, 3]': 1.0}. A single tanh unit can only draw a line, so it stays at about 0.7. Two units can draw a band but not a closed region, and reach about 0.86 at best. With three units the network can draw a triangle around the disc and reaches [1.0, 1.0, 1.0, 1.0, 1.0] on the five seeds, and 8 units reach [1.0, 1.0, 1.0, 1.0, 1.0]. The two-layer 2 + 2 network, with the same two units in the first layer, gets between 0.87 and 0.98 (mean 0.931): better than the single layer of 2 units, but not perfect; with 3 + 3 units it reaches 1.0 on all five seeds. In the left figure the 2-unit network cuts a band through the data, in the middle figure the 2 + 2 network draws a rounded region, and in the right figure you see the first layer's re-coding: it turns the ring and the disc into two clouds that a second, simple layer can separate. Be careful with the claim: with 2 units in the first layer the information that survives is limited, so "2 + 2 succeeds" is true only roughly here, and "3 + 3 succeeds" is the clean statement.

# %% [markdown]
# **Tips and common mistakes**
#
# - `train_model` comes from the Setup. It uses binary cross-entropy with Adam here, `lr=0.03`, full batch, and returns the final accuracy in `hist["acc"]` (the fraction of correct points).
# - `plot_decision(predict, X, y)` needs a function that takes an (N, 2) array and returns probabilities, so we apply `torch.sigmoid` to the network's logit.
# - `model22[1](model22[0](x))` first applies the linear layer, then the tanh: `model22[0]` and `model22[1]` are the first two modules of the Sequential. The output of module 1 is the first hidden layer's activations.
#
# **Try also (optional)**
#
# - Replace `nn.Tanh()` by `nn.ReLU()` in `make_mlp` and rerun the table. ReLU networks with very few units lose units more easily (a unit that outputs 0 for every point cannot learn), so the small configurations become less reliable.
# - Plot the first hidden layer for the 3 + 3 network. Its first layer has three coordinates; plot the first two.

# %% [markdown]
# ### ✏️ Exercise 1.4: How many units does one layer need?
#
# Your task: find the smallest number of hidden units `H` for a **single** hidden layer (tanh) for which all three seeds 0, 1, 2 reach an accuracy of at least 0.95 on the circles. Use `fit_circles( [H], seed)` in a loop over H = 1 … 6 and seeds 0 … 2. Store the answer in `smallest_h` and write one sentence in a markdown cell: why can a network with 2 hidden units not enclose a disc? Why it matters: it makes a width-versus-depth trade-off concrete, and one layer of 3 units is the least that can draw a closed region.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
smallest_h = None   # the smallest H for which seeds 0, 1 and 2 all reach 0.95

# %%
if smallest_h is None:
    print("Fill in smallest_h above, then run this cell again.")
else:
    check("smallest_h is 3 or 4", smallest_h in (3, 4), "Loop over H = 1..6 and look for the first H where min(acc over the 3 seeds) >= 0.95.")
    check("1 and 2 units are not enough", all(min(fit_circles( [h], s)[1] for s in range(3)) < 0.95 for h in (1, 2)), "With one or two units the accuracy stays around 0.7 to 0.87.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# for H in range(1, 7):
#     accs = [fit_circles( [H], s)[1] for s in range(3)]
#     print(H, [round(a, 3) for a in accs])
#     if min(accs) >= 0.95:
#         smallest_h = H
#         break
# ```
#
# Two units can only draw a band between two parallel-ish lines (a strip), never a closed region; three units can draw a triangle around the disc. A deeper network reaches the same by re-coding the points first.
#
# </details>

# %% [markdown]
# ### Step 1.5. Imagine and write: what do the layers learn?
#
# Hierarchy is easy to say and hard to feel, so finish the part with writing. For a **photo** and for a **sentence**, list three levels of features a deep network might learn (low, middle, high level), and name one problem where depth would **not** help. The code only stores your text and prints it as a card, so that Part 7 can pick it up.

# %%
def show_levels(levels):
    """Print your answers as a small card."""
    for kind in ("photo", "sentence"):
        print(f"--- {kind} ---")
        for name, text in zip(("low", "middle", "high"), levels[kind]):
            print(f"  {name:<6}: {text or '(not written yet)'}")
    print("depth would NOT help for:", levels["no_depth"] or "(not written yet)")

# %% [markdown]
# **What you should see:** No numbers here; compare your card with ours. For a photo: edges and colour blobs, then textures and parts (wheel, eye, window), then whole objects (a car, a face). For a sentence: letters and word pieces, then words and short phrases (a noun with its adjective), then meaning (who did what to whom, the tone). Depth would **not** help for a tiny, nearly linear problem such as predicting a person's height from their shoe size: a straight line is already almost perfect, and extra layers only add parameters to fit noise.

# %% [markdown]
# **Tips and common mistakes**
#
# - Low level means close to the raw numbers (pixels, characters). High level means close to what a person would name.
# - The words "learn" and "level" are a picture, not a proof: networks do not always learn clean human-named levels. In vision, studies of trained networks do find such a progression, which is why the picture is useful.

# %% [markdown]
# ### ✏️ Exercise 1.5: Write the card
#
# Your task: fill in the three levels for a photo and for a sentence, and one problem where depth does not help. Then run the check: it only verifies that you wrote something for every field. Why it matters: being able to say in words what each layer adds is the quickest test of whether you understood hierarchy.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
levels = {
    "photo":    ["", "", ""],      # low, middle, high-level features of a photo
    "sentence": ["", "", ""],      # low, middle, high-level features of a sentence
    "no_depth": "",                # one problem where depth would not help
}
show_levels(levels)

# %%
filled = all(levels["photo"]) and all(levels["sentence"]) and bool(levels["no_depth"])
if not filled:
    print("Write something in every field above, then run this cell again.")
else:
    check("six levels and one counter-example written", True)
RESULTS["part1_levels"] = levels

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# levels = {
#     "photo":    ["edges and colour blobs", "textures and parts (wheel, eye, window)", "whole objects: a car, a face"],
#     "sentence": ["letters and word pieces", "words and short phrases", "meaning: who did what, and the tone"],
#     "no_depth": "predicting height from shoe size: an almost straight line already fits",
# }
# show_levels(levels)
# ```
#
# Depth pays off when the target is built from reusable parts, as in images and language. For a nearly linear relationship one or no hidden layer is enough.
#
# </details>

# %% [markdown]
# ## Part 2. The forward pass and the computation graph (20 min)
#
# *Lesson section: "Forward".*
#
# The forward pass is the first half of every training step: the signal travels from the input through the layers to the prediction, and a loss says how wrong the prediction is. In this part you do it three ways on the small network of the lesson, with plain Python lists, with NumPy matrices and as a drawn computation graph, and you learn which numbers the network has to remember for the backward pass. The last step puts a price on that memory.

# %% [markdown]
# ### ✏️ Imagine first
#
# The lesson's example network has 2 inputs, 2 hidden units and 1 output, all with sigmoid activations. The input is (1, 0.5) and the target is 1. Before you compute anything, guess the prediction `a` of the output unit: is it above or below 0.5, and by roughly how much? Then guess the loss `½(a − 1)²`. A sigmoid output is always between 0 and 1, so the loss is between 0 and 0.5.

# %%
# Imagine first: the [2, 2, 1] sigmoid network with input (1, 0.5) and target 1.
my_output = None   # your guess for the prediction a (a number between 0 and 1)
my_loss = None     # your guess for the loss 0.5 * (a - 1) ** 2

# %% [markdown]
# ### Step 2.1. The forward pass by hand, with plain Python lists
#
# The network of the lesson has these numbers: hidden weights `W1 = [ [0.5, −0.3], [0.8, 0.2] ]` (one row per hidden unit), hidden biases `b1 = [0.1, −0.1]`, output weights `W2 = [ [1.0, −1.5] ]` and output bias `b2 = [0.2]`. The input is `x = (1, 0.5)` and the target is `y = 1`. Every unit does two things: it first computes a weighted sum `z` of its inputs plus the bias, and then the activation `a = σ(z)`. The block does this with loops only, so that you see every multiplication, prints `z` and `a` of each of the three units, the prediction and the loss `½(a − y)²`, and asserts that the loss is the 0.15448 of the lesson.

# %%
def sigmoid_s(z):
    """The sigmoid of one number."""
    return 1 / (1 + math.exp(-z))

W1 = [ [0.5, -0.3], [0.8, 0.2] ]     # row j = the weights of hidden unit j
b1 = [0.1, -0.1]
W2 = [ [1.0, -1.5] ]                 # one output unit with two weights
b2 = [0.2]
x_ex, y_ex = [1.0, 0.5], 1.0         # the example input and its target

def forward_lists(x):
    """Forward pass with loops. Returns z and a of the hidden layer, then z and a of the output."""
    z1, a1 = [], []
    for j in range(2):                         # two hidden units
        z = b1[j]
        for i in range(2):                     # two inputs
            z += W1[j][i] * x[i]
        z1.append(z); a1.append(sigmoid_s(z))
    z2 = b2[0]
    for j in range(2):
        z2 += W2[0][j] * a1[j]
    return z1, a1, z2, sigmoid_s(z2)

z1, a1, z2, a2 = forward_lists(x_ex)
for j in range(2):
    print(f"hidden unit {j}: z = {z1[j]:.4f}   a = {a1[j]:.4f}")
print(f"output unit  : z = {z2:.4f}   a = {a2:.4f}   <- the prediction")
loss = 0.5 * (a2 - y_ex) ** 2
print(f"loss = 0.5 * (a - y)**2 = {loss:.5f}")
assert abs(loss - 0.15448) < 1e-5
if my_output is not None:
    print(f"you guessed a = {my_output} and loss = {my_loss}; the real values are {a2:.4f} and {loss:.5f}")

# %% [markdown]
# **What you should see:** You should see hidden unit 0 with `z = 0.45` and `a = 0.6106`, hidden unit 1 with `z = 0.8` and `a = 0.69`, an output with `z = -0.2243` and `a = 0.4442`, and the loss `0.15448`. The prediction 0.444 is below 0.5, although the target is 1: the network is wrong and the loss says by how much. Check your guesses: if you guessed an output between 0.4 and 0.5 you were close. The hidden unit 1 has z = 0.8 because 0.8·1 + 0.2·0.5 − 0.1 = 0.8. Keep these six numbers; Part 3 recomputes them with an automatic engine.

# %% [markdown]
# **Tips and common mistakes**
#
# - `z1 += ...` adds to a running total that starts at the bias. If you start from 0 and forget the bias, you get slightly different numbers.
# - `W1[j][i]` is the weight from input i to hidden unit j. Row j of the matrix describes unit j. That convention is why the matrix form in Step 2.2 is `W @ a` and not `a @ W`.
# - The last unit has a negative weight (−1.5) on the second hidden unit, which pulls the output down. This is the reason why the prediction is below 0.5.
#
# **Try also (optional)**
#
# - Change the input to (0, 0) and predict the output before you run. With zero input only the biases count: what is `z` of hidden unit 0?
# - Replace `sigmoid_s` by `math.tanh` in `forward_lists` and see how the prediction changes.

# %% [markdown]
# ### ✏️ Exercise 2.1: A different example
#
# Your task: run the same network on the input `x = (−1, 2)` with the target `y = 0`, using `forward_lists`. Store the prediction in `pred_new` and the loss `½(a − y)²` in `loss_new` (rounded to 4 decimals). Before you run: with this input, will the prediction be higher or lower than before? Why it matters: the forward pass is just the same loop with other numbers; it is cheap and exactly repeatable, which is what makes training possible.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
pred_new = None   # prediction a of the output unit for x = (-1, 2)
loss_new = None   # 0.5 * (a - 0) ** 2, 4 decimals

# %%
if pred_new is None or loss_new is None:
    print("Fill in pred_new and loss_new above, then run this cell again.")
else:
    _, _, _, a_ref = forward_lists( [-1.0, 2.0] )
    check("pred_new is the output for (-1, 2)", abs(pred_new - a_ref) < 1e-3, "Call forward_lists( [-1.0, 2.0] ) and take the last number it returns.")
    check("loss_new = 0.5 * a**2", abs(loss_new - 0.5 * a_ref ** 2) < 1e-3, "The target is 0, so the loss is 0.5 * (a - 0)**2.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# _, _, _, pred_new = forward_lists( [-1.0, 2.0] )
# pred_new = round(pred_new, 4)
# loss_new = round(0.5 * (pred_new - 0.0) ** 2, 4)
# ```
#
# The prediction is 0.476 and the loss 0.1131. A single forward pass never changes the weights: it only measures how wrong the network is.
#
# </details>

# %% [markdown]
# ### Step 2.2. The same forward pass with NumPy matrices
#
# A real network has thousands of units, and nobody writes loops over units. A whole layer is one matrix product: with `a` the vector of the previous layer, `z = W @ a + b` computes all the weighted sums of the layer at once, and the activation is applied to the whole vector. The block puts the weights into NumPy arrays, runs a loop **over layers** (not over units) and stores every `z` and every `a` in lists. It prints the shapes, asserts that the result equals the list version, and also runs all four XOR corners at once to show how a batch is processed.

# %%
def sigmoid_np(z):
    return 1 / (1 + np.exp(-z))

layers = [ (np.array(W1), np.array(b1)), (np.array(W2), np.array(b2)) ]     # (weights, bias) per layer

def forward_np(layers, x):
    """Forward pass layer by layer. Returns the list of z's and the list of a's (a[0] is the input)."""
    a = np.asarray(x, dtype=float)
    zs, acts = [], [a]
    for W, b in layers:
        z = W @ a + b                  # all weighted sums of the layer in one product
        a = sigmoid_np(z)
        zs.append(z); acts.append(a)
    return zs, acts

zs, acts = forward_np(layers, x_ex)
for l, (W, b) in enumerate(layers, 1):
    print(f"layer {l}: W {W.shape} @ a {acts[l - 1].shape} + b {b.shape} -> z {zs[l - 1].shape}  values {np.round(zs[l - 1], 4)}")
print("prediction:", np.round(acts[-1], 4), "| same as the list version:", np.allclose(acts[-1], a2))
assert np.allclose(zs[0], z1) and np.allclose(acts[1], a1) and np.allclose(acts[2], a2)

# %%
# A whole batch at once: each ROW of X is one example, so the product is A @ W.T + b
X_corners = np.array( [ [0, 0], [0, 1], [1, 0], [1, 1] ], dtype=float)
A = X_corners
for W, b in layers:
    A = sigmoid_np(A @ W.T + b)
print("batch input shape", X_corners.shape, "-> batch output shape", A.shape)
print("outputs for the four corners:", np.round(A.ravel(), 4))
single = [forward_np(layers, row)[1][-1][0] for row in X_corners]
assert np.allclose(A.ravel(), single)

# %% [markdown]
# **What you should see:** The first block prints two lines. Layer 1: `W (2, 2) @ a (2,) + b (2,) -> z (2,)` with values `[0.45 0.8 ]`, and layer 2: `W (1, 2) @ a (2,) + b (1,) -> z (1,)` with value `[-0.2243]`. The prediction is `[0.4442]`, identical to the loop version, and both asserts pass. The second block turns the single example into a batch of four: the input has shape (4, 2) and the output (4, 1). The four outputs are [0.5031 0.4657 0.4609 0.4275]: for each corner a prediction between 0 and 1 (the network has not been trained, so they are not meaningful yet). The two versions agree up to rounding errors of about 1e-16.

# %% [markdown]
# **Tips and common mistakes**
#
# - With a **batch**, each row is one example, so the product is `A @ W.T`: the same weights are applied to every row. If you wrote `A @ W` you would get a shape error, because (4, 2) @ (2, 2) works for the hidden layer by accident but (4, 2) @ (1, 2) does not for the output layer. Check the shapes on paper first.
# - `np.allclose(a, b)` compares numbers up to a tiny tolerance. Never compare floating-point numbers with `==`.
# - A bias `b` of shape (2,) is **broadcast**: it is added to every row of the batch. This is the NumPy feature that makes the batch version so short.
#
# **Try also (optional)**
#
# - Add a third layer of one unit to `layers` (a new W of shape (1, 1) and a bias of shape (1,)), and run `forward_np` again. What do you have to change in the loop? (Nothing.)
# - Print the stored `acts` list and count the numbers in it: 2 + 2 + 1 = 5. That is what must be remembered for the backward pass in Step 2.4.

# %% [markdown]
# ### ✏️ Exercise 2.2: A batch forward pass
#
# Your task: write the function `forward_batch(layers, X)` that takes a matrix `X` with one example per row and returns the matrix of outputs, shape `(n, 1)`. Use the pattern of the block above (`A = sigmoid_np(A @ W.T + b)` in a loop over layers). Why it matters: every library works with batches; one matrix product handles all examples at once and keeps the CPU or GPU busy.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
def forward_batch(layers, X):
    A = np.asarray(X, dtype=float)
    # YOUR CODE HERE: loop over the layers and update A
    return np.zeros((len(A), 1))      # placeholder: replace by the real output

out_batch = forward_batch(layers, X_corners)
print(out_batch.shape)

# %%
out_batch = forward_batch(layers, X_corners)
check("output has shape (4, 1)", out_batch.shape == (4, 1), "After the last layer A should have one column.")
check("same numbers as the loop over single examples", np.allclose(out_batch.ravel(), single), "Use A = sigmoid_np(A @ W.T + b) for every (W, b).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def forward_batch(layers, X):
#     A = np.asarray(X, dtype=float)
#     for W, b in layers:
#         A = sigmoid_np(A @ W.T + b)
#     return A
#
# out_batch = forward_batch(layers, X_corners)
# print(out_batch.shape)
# ```
#
# Looping over layers keeps the function short and works for any depth; the per-example work happens inside the matrix product.
#
# </details>

# %% [markdown]
# ### Step 2.3. Draw the computation graph of a tiny example
#
# Every forward pass is a chain of small operations. Drawing the chain as a **computation graph** shows what the backward pass will walk through in reverse. We use the smallest interesting case, one neuron with one input: `x → ×w → +b → σ → (a − y)²`. The values are x = 1.5, w = 0.8, b = −0.2 and target y = 1. The block computes every node, prints the values and draws the graph with Matplotlib: each box shows the operation and the value flowing out of it, and the arrows show the direction of the forward pass.

# %%
x, w, b, y = 1.5, 0.8, -0.2, 1.0
p = w * x                       # node 1: product            p = w*x
z = p + b                       # node 2: sum with the bias  z = p + b
a = 1 / (1 + math.exp(-z))      # node 3: sigmoid            a = sigma(z)
L = (a - y) ** 2                # node 4: squared error      L = (a - y)^2
print(f"x = {x}, w = {w}, b = {b}, y = {y}")
print(f"p = w*x = {p:.4f} | z = p + b = {z:.4f} | a = sigmoid(z) = {a:.4f} | L = (a - y)^2 = {L:.4f}")

# %%
fig, ax = plt.subplots(figsize=(11, 2.6))
nodes = [("x", x), ("w", w), ("b", b), ("p = w*x", p), ("z = p + b", z), ("a = sigmoid(z)", a), ("L = (a - y)^2", L)]
pos = [ (0, 1.0), (0, 0.0), (2, -0.3), (2, 0.5), (4, 0.5), (6, 0.5), (8, 0.5) ]
for (name, val), (px, py) in zip(nodes, pos):
    ax.text(px, py, f"{name}\n= {val:.4f}", ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f0fe", ec="k"))
for src, dst in [(0, 3), (1, 3), (3, 4), (2, 4), (4, 5), (5, 6)]:
    ax.annotate("", xy=pos[dst], xytext=pos[src], arrowprops=dict(arrowstyle="->", color="gray", shrinkA=26, shrinkB=26))
ax.text(8, -0.5, f"y = {y} (the target)", ha="center", fontsize=9)
ax.set_xlim(-1, 9.2); ax.set_ylim(-0.9, 1.5); ax.axis("off")
ax.set_title("Forward pass: the signal flows left to right")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see `p = w*x = 1.2 | z = p + b = 1.0 | a = sigmoid(z) = 0.7311 | L = (a - y)^2 = 0.0723`. The figure has seven boxes: the inputs x, w and b on the left, then p, z, a and the loss L from left to right, each with its value. Compare with the lesson figure of the computation graph: it is the same picture. Now think about the backward pass: to find how L changes when w changes, we walk the arrows backwards and multiply the local slopes. Which of the stored numbers will we need? That is Exercise 2.3.

# %% [markdown]
# **Tips and common mistakes**
#
# - The chain rule needs **the numbers from the forward pass**: if you throw them away you must recompute them. This is why `loss.backward()` in PyTorch only works after a forward pass that remembered its intermediate values.
# - A box with two incoming arrows (p and b going into z) is a sum: it only needs the two numbers to be added. A box with a single input (σ) needs its input or its output.
# - The `\\n` in the label is a line break inside the Matplotlib string. If you see the characters `\\n` in a box, check that you typed two backslashes inside the f-string in the notebook source, or remove them.
#
# **Try also (optional)**
#
# - Change the target to y = 0 and redraw: only L changes, the arrows stay.
# - Add a second hidden unit by hand on a sheet of paper and draw how the graph forks and joins again: that is every layer of a real network.

# %% [markdown]
# ### ✏️ Exercise 2.3: What must be stored?
#
# Your task: to compute the gradients `dL/dw` and `dL/db` in the backward pass, which of the forward-pass values `x, w, b, p, z, a, y` do you have to **keep**? Think of the local slopes: `dL/da = 2(a − y)` needs `a` and `y`; `da/dz = a(1 − a)` needs `a`; `dz/db = 1` and `dz/dp = 1` need nothing; `dp/dw = x` needs `x`. Store your answer as a set of names in `stored`. Why it matters: the set of stored values is exactly the memory a training step needs on top of the weights, and Step 2.4 puts a number on it.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
stored = set()     # for example {"x", "w"}: replace by the names you think must be kept
print("I keep:", sorted(stored))

# %%
need = {"x", "a", "y"}
if not stored:
    print("Write the names of the values to keep in stored (a set of strings), then run this cell again.")
else:
    check("contains x, a and y", need <= stored, "Look at the local slopes: x is needed for dp/dw, a for the sigmoid slope, y for dL/da.")
    check("does not contain w, b or p", not ({"w", "b", "p"} & stored), "The weights and biases are already in memory as parameters, and p does not appear in any slope.")
    check("z is allowed but not needed", stored <= need | {"z"}, "Only x, a, y (and optionally z) are needed.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# stored = {"x", "a", "y"}
# ```
#
# With the sigmoid slope written as `a(1 − a)` you only need x, a and y. A framework that keeps z as well (to recompute the slope from z) stores a little more. Either way, the cost grows with the number of nodes in the graph.
#
# </details>

# %% [markdown]
# ### Step 2.4. The memory of training: what must be stored
#
# Training stores the activation of every layer for the whole batch, because the backward pass needs them (Exercise 2.3 showed why). Inference does not: it can forget each layer's input as soon as the next layer is computed. The block counts, for a network `[784, 128, 64, 10]` (a small image classifier for 28 × 28 pictures), the parameters and the stored activations for batch sizes 1, 64 and 1024. The activations of one example are the input (784), the hidden layers (128 and 64) and the output (10): 986 numbers. We count in float32, 4 bytes per number, and quote megabytes as millions of bytes.

# %%
sizes = [784, 128, 64, 10]
n_params = sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))     # weights + biases of every layer
per_example = sum(sizes)                                              # activations kept for ONE example (input and all layers)
MB = 4 / 1e6                                                          # bytes per float32 -> megabytes

print(f"parameters: {n_params}  = {n_params * MB:.3f} MB")
print(f"stored activations per example: {per_example}")
print(f"{'batch':>6} {'activations':>12} {'MB':>8} {'x parameters':>14}")
for B in (1, 64, 1024):
    n_act = B * per_example
    print(f"{B:>6} {n_act:>12} {n_act * MB:>8.3f} {n_act / n_params:>14.2f}")

# Training also keeps one gradient per parameter, and Adam keeps two more numbers per parameter
train_params = 4 * n_params
print(f"parameters + gradients + Adam's two states = {train_params} numbers = {train_params * MB:.2f} MB")
assert n_params == 109386

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.4))
batches = [1, 16, 64, 256, 1024, 4096]
ax.loglog(batches, [B * per_example * MB for B in batches], "o-", label="stored activations (training)")
ax.axhline(n_params * MB, c="tab:red", ls="--", label="parameters")
ax.axhline(train_params * MB, c="tab:orange", ls=":", label="parameters + gradients + Adam")
ax.set_xlabel("batch size"); ax.set_ylabel("megabytes (float32)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
ax.set_title("Training memory of a [784, 128, 64, 10] network")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see `parameters: 109386  = 0.438 MB` (that is 784·128 + 128 = 100,480 plus 128·64 + 64 = 8,256 plus 64·10 + 10 = 650), `stored activations per example: 986`, and the table: batch 1 stores 986 numbers (0.004 MB, about 0.01 times the parameters), batch 64 stores 63,104 (0.252 MB, 0.58 times), batch 1024 stores 1,009,664 (4.039 MB, 9.23 times). Parameters, gradients and Adam's two states cost 437,544 numbers or 1.75 MB. So for a small batch the parameters dominate; for a big batch the stored activations dominate and training needs far more memory than just the weights. This is why a training run on a large model needs a big GPU although the finished model for inference fits in much less. The figure shows the straight line of the activations (twice the batch, twice the memory) crossing the dashed parameter line.

# %% [markdown]
# **Tips and common mistakes**
#
# - The activations grow **linearly with the batch size** and with the width and depth of the network. For images with many channels or for long texts they dominate everything else. Techniques such as gradient checkpointing trade memory for time by recomputing some activations.
# - Adam keeps two extra numbers (a running mean of the gradient and of its square) for every parameter, so it costs 2 × n_params more memory than plain SGD.
# - Only the batch size changes the activation memory; the parameter memory is fixed by the architecture.
#
# **Try also (optional)**
#
# - Change `sizes` to `[784, 512, 512, 10]` and watch the parameters (about 0.66 million) and the activation count per example (1,818) change by different factors.
# - Estimate the same numbers for a network with 100 hidden layers of 1,000 units (about 100 million parameters at batch 1: `100 * 1000 * 1000`) and compare them with your laptop's memory.

# %% [markdown]
# ### ✏️ Exercise 2.4: Where does the memory cross over?
#
# Your task: find the smallest batch size `B` for which the stored activations (986 numbers per example) are **more numerous than the parameters** (109,386). Compute it with a formula, not by trying, and store it in `crossover_batch`. Then write two sentences in a markdown cell: why does training need more memory than inference, and which one grows with the batch size? Why it matters: batch size is the knob you turn first when a training run runs out of memory.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
crossover_batch = None   # smallest integer B with B * per_example > n_params

# %%
if crossover_batch is None:
    print("Fill in crossover_batch above, then run this cell again.")
else:
    check("B * 986 is larger than the parameters", crossover_batch * per_example > n_params, "Divide the parameter count by 986 and round up.")
    check("B - 1 is not enough", (crossover_batch - 1) * per_example <= n_params, "You need the smallest such B.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# crossover_batch = n_params // per_example + 1   # 109386 / 986 = 110.9, so 111
# ```
#
# 109,386 / 986 = 110.9, so from a batch of 111 onwards the stored activations outnumber the parameters. Inference needs only the current layer's activations; training keeps all of them until the backward pass has used them.
#
# </details>

# %% [markdown]
# ## Part 3. The chain rule and a mini autograd engine (35 min)
#
# *Lesson section: "Chain rule and Backward".*
#
# Backpropagation is the chain rule from calculus, organised so that a computer can do it for any network. In this part you do it first on a single neuron by hand and check it with a finite difference, then you tabulate the local derivatives of the activations, and then you build the engine itself: a small class `Value` (in the spirit of Andrej Karpathy's micrograd) that remembers how every number was made and can run the chain rule backwards. You use it on the lesson's network, compare all nine gradients with PyTorch's autograd, and finally train a small network with your own engine.

# %% [markdown]
# ### ✏️ Imagine first
#
# Take one neuron with the loss L = (σ(w·x + b) − y)². Use w = 0.8, b = −0.2, x = 1.5 and target y = 1. The question is how the loss changes when you increase the weight w a little. Before you compute anything: will dL/dw be **positive or negative**, and roughly how large is it in absolute value (close to 0, around 0.1, around 1, or bigger)? Think about what the neuron does: is its output too low or too high compared with the target of 1, and what would a bigger w do to the output?

# %%
# Imagine first: the sign and the size of dL/dw for L = (sigmoid(w*x + b) - y)**2
my_sign = None   # write "positive" or "negative"
my_size = None   # your guess for the absolute value, for example 0.1

# %% [markdown]
# ### Step 3.1. One scalar example by hand, then a finite-difference check
#
# The chain from the weight to the loss has four links: `p = w·x`, `z = p + b`, `a = σ(z)`, `L = (a − y)²`. The chain rule says that the slope of the whole chain is the **product of the local slopes**: `dL/dw = (dL/da) · (da/dz) · (dz/dp) · (dp/dw)`, where `dL/da = 2(a − y)`, `da/dz = a(1 − a)`, `dz/dp = 1` and `dp/dw = x`. The block computes every factor, multiplies them, and then checks the result with a **finite difference**: nudge w by a tiny ε in both directions and measure how the loss changes, `(L(w + ε) − L(w − ε)) / (2ε)`. This check needs no calculus at all, which makes it the best friend of anyone writing gradient code.

# %%
x, w, b, y = 1.5, 0.8, -0.2, 1.0

def loss_scalar(w, b):
    """The whole chain in one function: L = (sigmoid(w*x + b) - y)**2."""
    z = w * x + b
    a = 1 / (1 + math.exp(-z))
    return (a - y) ** 2

# forward pass, keeping the numbers
z = w * x + b
a = 1 / (1 + math.exp(-z))

# the local slopes
dL_da = 2 * (a - y)         # slope of the squared error
da_dz = a * (1 - a)         # slope of the sigmoid
dz_dp = 1.0                 # a sum passes the slope through
dp_dw = x                   # slope of w*x with respect to w
dL_dw = dL_da * da_dz * dz_dp * dp_dw
dL_db = dL_da * da_dz * dz_dp * 1.0

eps = 1e-6
num_w = (loss_scalar(w + eps, b) - loss_scalar(w - eps, b)) / (2 * eps)
num_b = (loss_scalar(w, b + eps) - loss_scalar(w, b - eps)) / (2 * eps)

print(f"local slopes: dL/da = {dL_da:.4f}, da/dz = {da_dz:.4f}, dz/dp = {dz_dp}, dp/dw = {dp_dw}")
print(f"{'':<4} {'by the chain rule':>18} {'finite difference':>18} {'relative error':>15}")
print(f"{'w':<4} {dL_dw:>18.8f} {num_w:>18.8f} {abs(dL_dw - num_w) / abs(dL_dw):>15.2e}")
print(f"{'b':<4} {dL_db:>18.8f} {num_b:>18.8f} {abs(dL_db - num_b) / abs(dL_db):>15.2e}")
assert abs(dL_dw - num_w) / abs(dL_dw) < 1e-6 and abs(dL_db - num_b) / abs(dL_db) < 1e-6
if my_sign is not None:
    print("you guessed:", my_sign, "and about", my_size, "-> the real value is", round(dL_dw, 4))

# %% [markdown]
# **What you should see:** You should see the local slopes `dL/da = -0.5379` (negative, because the output 0.731 is below the target 1), `da/dz = 0.1966` (a sigmoid with z = 1.0 is on a steep part), `dz/dp = 1.0` and `dp/dw = 1.5`. The chain rule gives `dL/dw = -0.15863128` and `dL/db = -0.10575419`; the finite differences agree to about 8 digits, with a relative error around 1e-9 (the exact digits depend on rounding, anything below 1e-6 is perfect). The sign is **negative**: increasing w increases the output of the neuron, which is too low, so the loss goes **down**. That is why gradient descent, which steps against the gradient, will increase w. The size, about 0.16, is moderate. Compare with your *Imagine first* guess.

# %% [markdown]
# **Tips and common mistakes**
#
# - The finite difference uses the **central** form `(f(w+ε) − f(w−ε)) / (2ε)`, whose error shrinks like ε². The one-sided form `(f(w+ε) − f(w))/ε` is less accurate (error like ε). With ε = 1e-6 and 64-bit numbers the central form is accurate to about 9 digits; with ε much smaller you start to lose digits to rounding.
# - Finite differences are slow (two extra forward passes per parameter), so they are a **test**, never a training method. A network with a million parameters would need two million forward passes for one gradient; backpropagation needs one backward pass.
# - Notice that every factor was computed from a number of the forward pass: `a` and `x`. That is what Exercise 2.3 predicted.
#
# **Try also (optional)**
#
# - Change w to 2.5 and run again. The sigmoid is now closer to saturation: what happens to `da/dz` and to the gradient? (Try it: the gradient shrinks although the neuron is still wrong. This is a first taste of vanishing gradients.)
# - Replace the loss by `abs(a - y)` and think about what the slope `dL/da` would be (±1). Why do we prefer the squared error for gradient descent?

# %% [markdown]
# ### ✏️ Exercise 3.1: Write the chain rule as a function
#
# Your task: write the function `grad_w(w, b, x, y)` that returns the analytic `dL/dw` for any w, b, x, y (the product of the four local slopes). The check compares it with a finite difference at several points. Why it matters: this is a complete, tiny backward pass, and once it works you can see that every deeper network is the same thing with more links.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
def grad_w(w, b, x, y):
    # YOUR CODE HERE: forward pass to get a, then the product of the local slopes
    return 0.0       # placeholder: replace by the real gradient

print("grad_w(0.8, -0.2, 1.5, 1.0) =", grad_w(0.8, -0.2, 1.5, 1.0))

# %%
def numeric_grad_w(w, b, x, y, eps=1e-6):
    f = lambda w_: (1 / (1 + math.exp(-(w_ * x + b))) - y) ** 2
    return (f(w + eps) - f(w - eps)) / (2 * eps)

cases = [ (0.8, -0.2, 1.5, 1.0), (-1.0, 0.5, 2.0, 0.0), (2.5, 0.0, -1.0, 1.0), (0.1, 0.1, 0.5, 0.0) ]
ok = all(abs(grad_w(*c) - numeric_grad_w(*c)) < 1e-6 for c in cases)
check("grad_w agrees with the finite difference in 4 cases", ok, "Compute a = sigmoid(w*x + b) first, then 2*(a - y) * a*(1 - a) * x.")
check("grad_w(0.8, -0.2, 1.5, 1.0) is the value above", abs(grad_w(0.8, -0.2, 1.5, 1.0) - dL_dw) < 1e-9, "It should equal dL_dw computed in the block.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def grad_w(w, b, x, y):
#     a = 1 / (1 + math.exp(-(w * x + b)))
#     return 2 * (a - y) * a * (1 - a) * x
# ```
#
# The function is the block above in one line: `dL/da · da/dz · dz/dp · dp/dw` with `dz/dp = 1` left out.
#
# </details>

# %% [markdown]
# ### Step 3.2. The local derivatives of the activations
#
# The chain rule multiplies local slopes, and the slope of the activation function is one of them in every layer. The block defines the derivatives of the three activations, `σ′(z) = σ(z)(1 − σ(z))`, `tanh′(z) = 1 − tanh²(z)` and `ReLU′(z) = 1` for `z > 0` and `0` otherwise, checks each against a finite difference at a few points, draws σ with σ′ (marking the maximum) and the three derivatives side by side, and asserts that the largest slope of the sigmoid is 0.25. Remember this number: it is the reason why deep sigmoid networks have vanishing gradients (Part 5).

# %%
def d_sigmoid(z):
    s = sigmoid_np(z)
    return s * (1 - s)

def d_tanh(z):
    return 1 - np.tanh(z) ** 2

def d_relu(z):
    return (np.asarray(z) > 0).astype(float)

# check each derivative against a finite difference at a few points
for name, f, df in [("sigmoid", sigmoid_np, d_sigmoid), ("tanh", np.tanh, d_tanh), ("relu", relu_np, d_relu)]:
    pts = np.array([-2.0, -0.5, 0.7, 3.0])
    num = (f(pts + 1e-6) - f(pts - 1e-6)) / 2e-6
    print(f"{name:<8} formula {np.round(df(pts), 4)}  finite difference {np.round(num, 4)}")
    assert np.allclose(df(pts), num, atol=1e-5)

zz = np.linspace(-8, 8, 1601)
print("max of sigmoid':", round(float(d_sigmoid(zz).max()), 6), "at z =", float(zz[d_sigmoid(zz).argmax()]))
print("max of tanh'   :", round(float(d_tanh(zz).max()), 6), "| max of relu':", float(d_relu(zz).max()))
assert abs(d_sigmoid(zz).max() - 0.25) < 1e-9

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(zz, sigmoid_np(zz), label="sigmoid(z)")
axes[0].plot(zz, d_sigmoid(zz), label="sigmoid'(z)")
axes[0].scatter([0], [0.25], c="tab:red", zorder=3); axes[0].annotate("maximum 0.25", (0, 0.25), (1.5, 0.35), arrowprops=dict(arrowstyle="->"))
axes[0].set_title("the sigmoid and its slope"); axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].plot(zz, d_sigmoid(zz), label="sigmoid'")
axes[1].plot(zz, d_tanh(zz), label="tanh'")
axes[1].plot(zz, d_relu(zz), label="relu'")
axes[1].set_ylim(-0.1, 1.2); axes[1].set_title("the three slopes"); axes[1].grid(alpha=0.3); axes[1].legend()
for ax in axes: ax.set_xlabel("z")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see three lines in which the formula and the finite difference agree to four decimals, for example for the sigmoid `[0.105  0.235  0.2217 0.0452]` at z = −2, −0.5, 0.7, 3. Then `max of sigmoid': 0.25 at z = 0.0`, `max of tanh': 1.0` and `max of relu': 1.0`. In the left figure the slope of the sigmoid is a bell with its top at 0.25 and almost 0 beyond |z| = 5; in the right figure tanh′ has height 1 in the middle and also dies out, while ReLU′ is a step: exactly 1 for positive z, exactly 0 for negative. Think ahead: every sigmoid layer multiplies the backward signal by at most 0.25, every ReLU layer by 0 or 1.

# %% [markdown]
# **Tips and common mistakes**
#
# - `d_relu(0)` is a convention: ReLU has a corner at 0 and no slope there. Frameworks use 0, as we do.
# - A slope near 0 on both sides (the flat tails of sigmoid and tanh) is called **saturation**. A saturated unit passes almost no gradient back, whatever the loss says.
# - The finite-difference check uses `atol=1e-5` here because the step is small and we only print four decimals. For the ReLU, avoid points very close to 0 where the corner would fool the check.
#
# **Try also (optional)**
#
# - Add the derivative of the leaky ReLU (`1` for z > 0 and `0.01` otherwise) and plot it. Why might a small non-zero slope help (see Part 5, dead ReLUs)?
# - Evaluate `d_sigmoid` at z = 10 and at z = 20 and compare.

# %% [markdown]
# ### ✏️ Exercise 3.2: Where does the sigmoid go flat?
#
# Your task: find, with a grid search such as `np.linspace(0, 10, 1001)`, the smallest z ≥ 0 for which the sigmoid slope falls **below 0.01**, round it to one decimal and store it in `z_flat`. Why it matters: it tells you how far from zero a pre-activation can drift before the unit stops learning.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
z_flat = None   # the smallest z >= 0 with sigmoid'(z) < 0.01 (one decimal)

# %%
if z_flat is None:
    print("Fill in z_flat above, then run this cell again.")
else:
    check("z_flat is about 4.6", abs(z_flat - 4.6) <= 0.15, "Search a grid for the first z where d_sigmoid(z) < 0.01.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# grid = np.linspace(0, 10, 1001)
# z_flat = round(float(grid[np.argmax(d_sigmoid(grid) < 0.01)]), 1)   # 4.6
# ```
#
# At z = 4.6 the sigmoid slope is 0.0100 (exactly: σ′(4.595) = 0.01). A unit whose input sum is beyond ±5 is almost frozen.
#
# </details>

# %% [markdown]
# ### Step 3.3. Build a tiny autograd engine: the class Value
#
# Now the engine. A `Value` wraps one number and remembers (a) which `Value`s it was computed from (its **parents**) and (b) a small function `_backward` that knows the local slope of the operation that made it. Calling `backward()` on the final loss does three things: it orders all nodes so that every node comes after its parents (a **topological order**), sets the gradient of the loss to 1, and then walks the nodes from the loss back to the inputs calling each `_backward`, which **adds** (`+=`) the node's gradient times the local slope to its parents. The class below is complete except for one operation, `sigmoid`, which is yours (Exercise 3.3). The first block defines the class and tests it against PyTorch.

# %%
class Value:
    """A number that remembers how it was made, so it can compute gradients (like micrograd)."""
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0                   # d(final loss)/d(this value), filled in by backward()
        self._backward = lambda: None     # how to pass the gradient on to the parents
        self._parents = tuple(parents)
        self.op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad            # a sum copies the gradient to both inputs
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad     # a product swaps the inputs
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):          return self * -1
    def __sub__(self, other):   return self + (-other)
    def __radd__(self, other):  return self + other
    def __rmul__(self, other):  return self * other
    def __rsub__(self, other):  return (-self) + other

    def backward(self):
        order, seen = [], set()
        def visit(v):                      # topological order: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        for v in order:
            v.grad = 0.0                   # start from zero, so you may call backward() again
        self.grad = 1.0                    # d(loss)/d(loss) = 1
        for v in reversed(order):          # walk from the loss back to the inputs
            v._backward()

# Test 1 by hand: d = a*b + c
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
d = a * b + c
d.backward()
print("d =", d.data, "| da =", a.grad, "| db =", b.grad, "| dc =", c.grad)
assert (d.data, a.grad, b.grad, c.grad) == (4.0, -3.0, 2.0, 1.0)

# Test 2 against PyTorch: f = relu(tanh(a*b + c) * 2 - 1) ** 2 + a ** 3
def f_value(a, b, c): return ((a * b + c).tanh() * 2 - 1).relu() ** 2 + a ** 3
va, vb, vc = Value(0.7), Value(-0.4), Value(0.9)
fv = f_value(va, vb, vc); fv.backward()
ta, tb, tc = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.7, -0.4, 0.9)]
ft = (torch.relu(torch.tanh(ta * tb + tc) * 2 - 1)) ** 2 + ta ** 3
ft.backward()
print("Value :", round(fv.data, 6), [round(v.grad, 6) for v in (va, vb, vc)])
print("torch :", round(float(ft), 6), [round(float(t.grad), 6) for t in (ta, tb, tc)])
assert abs(fv.data - float(ft)) < 1e-12 and all(abs(v.grad - float(t.grad)) < 1e-12 for v, t in zip((va, vb, vc), (ta, tb, tc)))

# %% [markdown]
# **What you should see:** The block prints `d = 4.0 | da = -3.0 | db = 2.0 | dc = 1.0` (a product swaps the inputs: da = b = −3, db = a = 2; the sum passes the gradient 1 to c) and then two lines that must be identical: `Value : 0.353456 [1.356085, 0.19935, 0.284786]` and the same from `torch`. Your engine and PyTorch's engine agree to 12 digits on a function with a product, a sum, a tanh, a ReLU and two powers. Then do Exercise 3.3 below: its starter cell prints the output of the sigmoid, 0.5744, and the slope passed to the input; with the placeholder the slope is `0.0` and your job is to make it `0.2445`.

# %% [markdown]
# **Tips and common mistakes**
#
# - `self.grad += ...` with a **plus** is essential. A node that is used twice (for example `a * a`) receives two contributions that must be added. Writing `=` would silently lose one of them. Step 3.6 shows this.
# - `backward()` first sets all gradients in the graph to zero. In a training loop this is your "zero the gradients" step; in PyTorch you have to do it yourself with `optimizer.zero_grad()` because PyTorch accumulates across calls.
# - The topological order is what makes this work for any graph: a node's `_backward` is called only after all the nodes that use it have already passed their gradient to it.
#
# **Try also (optional)**
#
# - Write the `exp` method (value `e**x`, slope `e**x`) and test it against `torch.exp`.
# - Print the `op` of every node of `fv` (follow the `_parents`) to see the graph that your engine built.

# %% [markdown]
# ### ✏️ Exercise 3.3: Give Value a sigmoid
#
# Your task: complete `value_sigmoid` so that `Value.sigmoid()` passes the gradient back: for `out = σ(self)`, `self.grad += s·(1 − s) · out.grad`, where `s` is the sigmoid value (that is the local slope `a(1 − a)` from Step 3.1). Replace the `pass` in the starter cell. The check compares your slope with the formula and with a finite difference. If you do not finish, the next steps still run: they will switch to a reference implementation and tell you. Why it matters: writing one local rule yourself shows how little an autograd engine knows about calculus.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
def value_sigmoid(self):
    s = 1 / (1 + math.exp(-self.data))
    out = Value(s, (self,), "sigmoid")
    def _backward():
        pass       # YOUR CODE HERE: self.grad += (the slope of the sigmoid at this point) * out.grad
    out._backward = _backward
    return out

Value.sigmoid = value_sigmoid

v = Value(0.3); out = v.sigmoid(); out.backward()
print("output", round(out.data, 4), "| slope passed to the input:", round(v.grad, 4), "(it should be 0.2445)")

# %%
def sigmoid_slope_ok():
    good = True
    for z0 in (-2.0, 0.3, 4.0):
        v = Value(z0); o = v.sigmoid(); o.backward()
        s = 1 / (1 + math.exp(-z0))
        good = good and abs(v.grad - s * (1 - s)) < 1e-9
    return good

check("Value.sigmoid passes the slope s*(1-s)", sigmoid_slope_ok(), "In _backward: self.grad += s * (1 - s) * out.grad")
w_ = Value(0.8); o_ = (w_ * 1.5 + (-0.2)).sigmoid(); o_.backward()
check("the gradient of sigmoid(w*1.5 - 0.2) with respect to w is 1.5 * s * (1 - s)", abs(w_.grad - 1.5 * o_.data * (1 - o_.data)) < 1e-9, "Remember to multiply by out.grad, which already contains the 1.5 from the product.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def value_sigmoid(self):
#     s = 1 / (1 + math.exp(-self.data))
#     out = Value(s, (self,), "sigmoid")
#     def _backward():
#         self.grad += s * (1 - s) * out.grad
#     out._backward = _backward
#     return out
#
# Value.sigmoid = value_sigmoid
#
# v = Value(0.3); out = v.sigmoid(); out.backward()
# print("output", round(out.data, 4), "| slope passed to the input:", round(v.grad, 4))
# ```
#
# The local rule is the slope you tabulated in Step 3.2, multiplied by the gradient that arrives from above (`out.grad`). That multiplication is the chain rule.
#
# </details>

# %% [markdown]
# ### Step 3.4. Use Value on the lesson's network and compare with PyTorch
#
# First a safety net: if your sigmoid from Exercise 3.3 is not correct yet, the next cell swaps in the reference version and says so, so that nothing downstream breaks. Then the engine is applied to the scalar example of Step 3.1, which must give the same gradient. Then the full network of Part 2 (`[2, 2, 1]`, sigmoid everywhere, loss `½(a − y)²`) is built out of `Value`s: nine parameters (four hidden weights, two hidden biases, two output weights, one output bias). After `backward()` every node carries its gradient, so the block prints the **deltas** `δ = dL/dz` of the output and hidden units, the gradient of the weight `W1[0][1]` that the lesson highlights (−0.016314), and then compares all nine gradients with `torch.autograd` and with finite differences.

# %%
# (safety net) swap in a reference sigmoid when yours is not correct yet
def sigmoid_reference(self):
    s = 1 / (1 + math.exp(-self.data))
    out = Value(s, (self,), "sigmoid")
    def _backward():
        self.grad += s * (1 - s) * out.grad
    out._backward = _backward
    return out

def sigmoid_is_correct():
    v = Value(0.3); o = v.sigmoid(); o.backward()
    s = 1 / (1 + math.exp(-0.3))
    return abs(v.grad - s * (1 - s)) < 1e-9

if not sigmoid_is_correct():
    print("Your Value.sigmoid is not correct yet (Exercise 3.3): using the reference version for the next cells.")
    Value.sigmoid = sigmoid_reference
else:
    print("Your Value.sigmoid works.")

# %%
# the scalar example of Step 3.1 with the engine
wv, bv = Value(0.8), Value(-0.2)
av = (wv * 1.5 + bv).sigmoid()
Lv = (av - 1.0) ** 2
Lv.backward()
print(f"loss {Lv.data:.6f} | dL/dw = {wv.grad:.8f} | dL/db = {bv.grad:.8f}")
print(f"step 3.1 : loss {loss_scalar(0.8, -0.2):.6f} | dL/dw = {dL_dw:.8f} | dL/db = {dL_db:.8f}")
assert abs(wv.grad - dL_dw) < 1e-12 and abs(bv.grad - dL_db) < 1e-12

# %%
# the [2, 2, 1] network built from Values
def make_params():
    W1v = [ [Value(0.5), Value(-0.3)], [Value(0.8), Value(0.2)] ]      # hidden weights, row j = unit j
    b1v = [Value(0.1), Value(-0.1)]
    W2v = [Value(1.0), Value(-1.5)]                                      # output weights
    b2v = Value(0.2)
    return W1v, b1v, W2v, b2v

def forward_value(params, x):
    W1v, b1v, W2v, b2v = params
    z1 = [sum((W1v[j][i] * x[i] for i in range(2)), b1v[j]) for j in range(2)]
    a1 = [z.sigmoid() for z in z1]
    z2 = sum((W2v[j] * a1[j] for j in range(2)), b2v)
    return z1, a1, z2, z2.sigmoid()

def loss_value(params, x, y):
    return 0.5 * (forward_value(params, x)[3] - y) ** 2

params = make_params()
z1v, a1v, z2v, a2v = forward_value(params, [1.0, 0.5])
Lnet = 0.5 * (a2v - 1.0) ** 2
Lnet.backward()
print(f"prediction {a2v.data:.5f} | loss {Lnet.data:.5f}")
print(f"delta output (dL/dz2)  = {z2v.grad:.6f}")
print(f"delta hidden (dL/dz1)  = {[round(z.grad, 6) for z in z1v]}")
print(f"dL/dW1[0][1] = {params[0][0][1].grad:.6f}")
assert abs(Lnet.data - 0.15448) < 1e-5 and abs(params[0][0][1].grad - (-0.016314)) < 1e-6

# %%
# all nine gradients: Value engine, torch.autograd, finite differences
names = ["W1[0][0]", "W1[0][1]", "W1[1][0]", "W1[1][1]", "b1[0]", "b1[1]", "W2[0]", "W2[1]", "b2"]
flat = lambda P: [P[0][0][0], P[0][0][1], P[0][1][0], P[0][1][1], P[1][0], P[1][1], P[2][0], P[2][1], P[3] ]
mine = [p.grad for p in flat(params)]

t = [torch.tensor(p.data, dtype=torch.float64, requires_grad=True) for p in flat(make_params())]
T1 = torch.stack( [torch.stack([t[0], t[1] ]), torch.stack([t[2], t[3] ]) ] )
h = torch.sigmoid(T1 @ torch.tensor([1.0, 0.5], dtype=torch.float64) + torch.stack([t[4], t[5] ]))
o = torch.sigmoid(torch.stack([t[6], t[7] ]) @ h + t[8])
(0.5 * (o - 1.0) ** 2).backward()
theirs = [float(p.grad) for p in t]

numeric = []
for i in range(9):
    up, down = make_params(), make_params()
    flat(up)[i].data += 1e-6; flat(down)[i].data -= 1e-6
    numeric.append((loss_value(up, [1.0, 0.5], 1.0).data - loss_value(down, [1.0, 0.5], 1.0).data) / 2e-6)

print(f"{'parameter':<10} {'Value engine':>14} {'torch.autograd':>16} {'finite diff':>14}")
for n, m_, t_, f_ in zip(names, mine, theirs, numeric):
    print(f"{n:<10} {m_:>14.8f} {t_:>16.8f} {f_:>14.8f}")
assert np.allclose(mine, theirs, atol=1e-12) and np.allclose(mine, numeric, atol=1e-8)
print("all 9 gradients agree")

# %% [markdown]
# **What you should see:** The safety-net cell prints `Your Value.sigmoid works.` (or the notice that it uses the reference). The scalar example gives the same loss and gradients as Step 3.1 to 12 digits. For the network you should see `prediction 0.44415 | loss 0.15448`, `delta output (dL/dz2) = -0.137228`, `delta hidden (dL/dz1) = [-0.032627, 0.044032]` and `dL/dW1[0][1] = -0.016314`: the loss is the 0.15448 of the lesson and the gradient is the −0.016314 that appears in the lesson figure. The table lists the nine gradients: [-0.0326, -0.0163, 0.044, 0.022, -0.0326, 0.044, -0.0838, -0.0947, -0.1372]. The three columns agree to at least eight decimals, and the last assert prints `all 9 gradients agree`. Read the numbers: the output delta is the largest (the output unit feels the error directly), the hidden deltas are much smaller (each is the output delta times a weight times a sigmoid slope ≤ 0.25), and so the hidden-layer weights get small gradients. Your own 40 lines of Python just reproduced what `loss.backward()` does.

# %% [markdown]
# **Tips and common mistakes**
#
# - `sum(generator, start)` adds the products to a start value; because `+` is defined on `Value`, the start can be the bias, which gives exactly `z = b + Σ wᵢxᵢ`.
# - The finite-difference step changes `.data` of a parameter in place. Because `make_params()` builds fresh `Value`s for every perturbation, there is no leftover state from the previous test.
# - A mismatch of the form "1.5 times too small" usually means that you wrote `=` instead of `+=` somewhere, or that a node used twice lost a contribution.
#
# **Try also (optional)**
#
# - Change the target to 0 (use `0.5 * (a2v - 0.0) ** 2`) and see which gradients change sign.
# - Print `z.grad` for the hidden units and compare each with `out.grad * W2 * sigmoid'`. This is the "hidden delta" equation of the lesson, discovered by the engine.

# %% [markdown]
# ### ✏️ Exercise 3.4: Another example, same engine
#
# Your task: run the engine on the input `x = (−1, 2)` with target `y = 0` (as in Exercise 2.1). Build a fresh set of parameters with `make_params()`, compute the loss with `loss_value(params, x, y)`, call `backward()` and store the gradient of the second output weight `W2[1]` (that is `params[2][1].grad`) in `grad_w2_1`. The check compares it with a finite difference. Why it matters: the engine does not care about the numbers, the architecture or the number of parameters; this is the point of automatic differentiation.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
params2 = make_params()
loss2 = loss_value(params2, [-1.0, 2.0], 0.0)
# YOUR CODE HERE: call backward on loss2 and read the gradient of W2[1]
grad_w2_1 = None   # params2[2][1].grad after loss2.backward()

# %%
if grad_w2_1 is None:
    print("Fill in grad_w2_1 above, then run this cell again.")
else:
    up, down = make_params(), make_params()
    up[2][1].data += 1e-6; down[2][1].data -= 1e-6
    ref = (loss_value(up, [-1.0, 2.0], 0.0).data - loss_value(down, [-1.0, 2.0], 0.0).data) / 2e-6
    check("grad_w2_1 matches the finite difference", abs(grad_w2_1 - ref) < 1e-7, f"Expected about {ref:.6f}. Did you call loss2.backward() before reading the gradient?")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# params2 = make_params()
# loss2 = loss_value(params2, [-1.0, 2.0], 0.0)
# loss2.backward()
# grad_w2_1 = params2[2][1].grad
# ```
#
# `backward()` fills the `.grad` of every node in the graph, so after one call you can read all nine gradients, not only one.
#
# </details>

# %% [markdown]
# ### Step 3.5. Train a network with your own engine
#
# An engine that only computes gradients is half of the story; the other half is the step `parameter -= learning_rate × gradient`. The block builds a `2 → 3 → 1` network with tanh in the hidden layer and a sigmoid output out of `Value`s, with random starting weights drawn from `random.Random(seed)` so that the run is reproducible, and trains it on the four XOR points `(0,0) → 0`, `(0,1) → 1`, `(1,0) → 1`, `(1,1) → 0` with the squared-error loss summed over the four points. It prints the loss every 40 steps of gradient descent (200 steps in all) and then the four predictions. XOR is the classic task that one neuron cannot do (Week 2) and a hidden layer can.

# %%
X4 = [ [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0] ]
y4 = [0.0, 1.0, 1.0, 0.0]

def make_xor_net(n_hidden, seed):
    rnd = random.Random(seed)
    W1 = [ [Value(rnd.uniform(-1, 1)) for _ in range(2)] for _ in range(n_hidden)]
    b1 = [Value(0.0) for _ in range(n_hidden)]
    W2 = [Value(rnd.uniform(-1, 1)) for _ in range(n_hidden)]
    b2 = Value(0.0)
    return W1, b1, W2, b2

def forward_xor(P, x):
    W1, b1, W2, b2 = P
    h = [sum((W1[j][i] * x[i] for i in range(2)), b1[j]).tanh() for j in range(len(W1))]     # tanh hidden layer
    return sum((W2[j] * h[j] for j in range(len(h))), b2).sigmoid()                          # sigmoid output

def all_params(P):
    W1, b1, W2, b2 = P
    return [w for row in W1 for w in row] + b1 + W2 + [b2]

def train_xor(n_hidden=3, lr=0.5, steps=200, seed=SEED, every=40, verbose=True):
    P = make_xor_net(n_hidden, seed)
    losses = []
    for step_ in range(steps + 1):
        loss = sum((forward_xor(P, x) - t) ** 2 for x, t in zip(X4, y4))      # one graph over all four points
        loss.backward()
        losses.append(loss.data)
        if verbose and step_ % every == 0:
            print(f"step {step_:>3}   loss {loss.data:.4f}")
        for p in all_params(P):
            p.data -= lr * p.grad                                           # gradient descent
    preds = [forward_xor(P, x).data for x in X4]
    return losses, preds

losses_xor, preds_xor = train_xor()
print("predictions for (0,0) (0,1) (1,0) (1,1):", [round(p, 3) for p in preds_xor], "| targets:", y4)
assert losses_xor[-1] < losses_xor[0]

# %%
plt.figure(figsize=(5.5, 3.2))
plt.semilogy(losses_xor)
plt.xlabel("gradient-descent step"); plt.ylabel("loss (log scale)"); plt.grid(alpha=0.3); plt.title("XOR with the Value engine")
plt.tight_layout(); plt.show()
RESULTS["part3_xor_value_final_loss"] = round(losses_xor[-1], 4)

# %% [markdown]
# **What you should see:** You should see the loss at steps 0, 40, 80, 120, 160, 200 decrease like [0.9674, 0.3352, 0.0594, 0.0263, 0.0161, 0.0114]. After 200 steps the predictions are [0.028, 0.941, 0.946, 0.065] against the targets `[0.0, 1.0, 1.0, 0.0]`: the two ones are above 0.9 and the two zeros below 0.1, so the network has learned XOR with a final loss of 0.0114. The curve in the figure falls by a factor of about 85 over the 200 steps. How robust is this? We also tried the seeds 0 to 9 with three hidden units: all ten ended below a loss of 0.025. With only two hidden units (`train_xor(n_hidden=2)`) the same seed 42 got stuck at a loss of about 0.51 with one point wrong: a tiny network can fail even when a solution exists.

# %% [markdown]
# **Tips and common mistakes**
#
# - The loss is built as **one graph** over the four points (a sum of four squared errors). `backward()` is called once per step; because it zeroes all gradients first, no extra zeroing is needed.
# - `p.data -= lr * p.grad` is the whole optimiser. PyTorch's `SGD` does exactly this (plus options).
# - If the loss does not move, check the sign: the update **subtracts** the gradient. If you add it, the loss climbs.
#
# **Try also (optional)**
#
# - Run `train_xor(n_hidden=2)` and `train_xor(n_hidden=4)` and compare the final loss (we measured about 0.51 and 0.011 with the seed 42). Then try `n_hidden=2` with other seeds: does the result depend on the seed?
# - Replace the tanh hidden units by `relu()` and train again. With only three ReLU units XOR can fail: a unit that is off for all four points never learns (a dead unit).

# %% [markdown]
# ### ✏️ Exercise 3.5: Learning rate and the Value engine
#
# Your task: train the same network with a **much smaller learning rate**, `train_xor(lr=0.05, verbose=False)`, store the final loss in `loss_small_lr` (the value at the last step) and compare it with `losses_xor[-1]`. Which one learns faster in 200 steps, and why? Write the reason in one sentence. Why it matters: the learning rate decides how big a step you take along the gradient; the engine only supplies the direction.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
loss_small_lr = None   # final loss of train_xor(lr=0.05, verbose=False)

# %%
if loss_small_lr is None:
    print("Fill in loss_small_lr above, then run this cell again.")
else:
    check("the small learning rate is slower in 200 steps", loss_small_lr > losses_xor[-1], "With 0.05 the steps are ten times smaller, so 200 steps move much less.")
    check("but it still went down", loss_small_lr < losses_xor[0], "The loss at step 0 is about 1.1; the small lr should be below that.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# losses_small, _ = train_xor(lr=0.05, verbose=False)
# loss_small_lr = losses_small[-1]
# ```
#
# With a tenth of the step size the network travels about a tenth of the distance in 200 steps, so its loss is still near the plateau. Part 4 and Part 5 return to the learning rate.
#
# </details>

# %% [markdown]
# ### Step 3.6. Observe: a sum copies the gradient, a product swaps the inputs
#
# Two sentences summarise most of the engine, and you can now see them in your own numbers. First: the local slope of `a + b` with respect to either input is 1, so a sum **copies** the gradient from above to both inputs. Second: the local slope of `a × b` with respect to `a` is `b` (and vice versa), so a product **swaps** the inputs: each input receives the gradient from above times the *other* factor. The block shows both on single numbers, then a node that is used twice in the same expression, where the two contributions **add** up, which is why the engine uses `+=`.

# %%
a, b = Value(3.0), Value(4.0)
s = a + b; s.backward()
print("sum    : s =", s.data, "| ds/da =", a.grad, "| ds/db =", b.grad, "(the gradient 1 is copied)")
a, b = Value(3.0), Value(4.0)
p = a * b; p.backward()
print("product: p =", p.data, "| dp/da =", a.grad, "(= b) | dp/db =", b.grad, "(= a)   (the inputs are swapped)")

# a node used twice: f = a*b + a*c  ->  df/da = b + c (two contributions add up)
a, b, c = Value(2.0), Value(3.0), Value(5.0)
f = a * b + a * c
f.backward()
print(f"reuse  : f = {f.data} | df/da = {a.grad} (= b + c = {b.data + c.data}) | df/db = {b.grad} (= a) | df/dc = {c.grad} (= a)")
assert a.grad == b.data + c.data

# a gradient scaled by the upstream value: g = 10 * (a*b) -> everything is multiplied by 10
a, b = Value(3.0), Value(4.0)
g = 10 * (a * b); g.backward()
print("scaled : dg/da =", a.grad, "(= 10 * b)")

# %% [markdown]
# **What you should see:** You should see `sum    : s = 7.0 | ds/da = 1.0 | ds/db = 1.0`, `product: p = 12.0 | dp/da = 4.0 (= b) | dp/db = 3.0 (= a)`, `reuse  : f = 16.0 | df/da = 8.0 (= b + c = 8.0) | df/db = 2.0 (= a) | df/dc = 2.0 (= a)` and `scaled : dg/da = 40.0 (= 10 * b)`. In words: addition passes the incoming gradient through unchanged (that is why a bias gradient equals the delta of its unit), multiplication hands each input the incoming gradient times the **other** factor (that is why a weight's gradient is the delta times the input that went through it), and a value used in two places collects the sum of the two gradients (that is why a hidden unit that feeds two outputs adds the blame from both).

# %% [markdown]
# **Tips and common mistakes**
#
# - A hidden unit that feeds several units in the next layer collects one contribution from each: this is the sum over the next layer in the hidden-delta equation `δ = (Wᵀ δ_next) ⊙ φ′`.
# - The residual connection of Part 6 (`h + f(h)`) is a sum, so it copies the gradient back to `h` unchanged. Keep this in mind: that copy is what makes residual networks easy to train.

# %% [markdown]
# ### ✏️ Exercise 3.6: Predict, then compute
#
# Your task: for `f = a·b + a·c` with `a = 2, b = 3, c = 5` the block printed `df/da = 8`. Now take `g = (a + b) * (a + c)` with the same numbers. **First** derive `dg/da`, `dg/db` and `dg/dc` on paper using the two rules (a sum copies, a product swaps, a reused node adds), store the numbers in `answers`, and write one sentence for each rule in your own words in a markdown cell. Then compute them with the engine to check. Why it matters: reading a graph and predicting a gradient is the skill behind every "why is my gradient zero?" question.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
answers = {"dg_da": None, "dg_db": None, "dg_dc": None}    # your hand-derived values
sentences = {"sum copies": "", "product swaps": ""}          # one sentence each, in your own words

# %%
a, b, c = Value(2.0), Value(3.0), Value(5.0)
g = (a + b) * (a + c); g.backward()
true = {"dg_da": a.grad, "dg_db": b.grad, "dg_dc": c.grad}
if any(v is None for v in answers.values()):
    print("Fill in the three numbers in answers above, then run this cell again.")
else:
    for k in true:
        check(f"{k} = {true[k]}", abs(answers[k] - true[k]) < 1e-9, "Product rule: (a+b) * (a+c). The first factor's slope is 1, so dg/db = (a+c); a appears in both factors, so add the two contributions.")
    check("you wrote both sentences", all(sentences.values()), "Write one sentence for each rule.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answers = {"dg_da": 12.0, "dg_db": 7.0, "dg_dc": 5.0}
# sentences = {"sum copies": "a sum gives its incoming gradient unchanged to each input", "product swaps": "a product gives each input the incoming gradient times the other factor"}
# ```
#
# With a = 2, b = 3, c = 5: (a + b) = 5 and (a + c) = 7. The product rule gives dg/db = 7 and dg/dc = 5. The variable a feeds both factors, so dg/da = 7 + 5 = 12: two contributions add up.
#
# </details>

# %% [markdown]
# ## Part 4. Backpropagation on layers: the four equations in NumPy (30 min)
#
# *Lesson section: "Backward".* In Part 3 you built a tiny autograd engine that traced every number by itself. Here you open that engine and write the same idea for whole **layers** with matrices, the way every deep-learning library does it. Backpropagation is only four equations: the blame at the output, the blame passed back through the weights, and the two formulas that turn a layer's blame into the gradient of its weights and biases. You will implement them for a general network in NumPy, prove them correct with a finite-difference **gradient check**, compare them with PyTorch's autograd, train a network with your own code, and watch how the gradient shrinks towards the first layer. ==Backpropagation is the chain rule, applied layer by layer from the output backwards, reusing the numbers stored in the forward pass.==

# %% [markdown]
# ### ✏️ Imagine first
#
# Picture a relay of messengers carrying **blame** backwards through a network of three sigmoid layers. At the output the blame is a normal-sized number. Every time it passes back through a layer it is multiplied by the weights and by the slope of the sigmoid, which never exceeds 0.25. Which layer will end up with the **larger gradient**, the first (next to the input) or the last (next to the output), and roughly how many times larger? Write your guess for the layer in `guess_layer` ("first" or "last") and for the factor in `guess_ratio`. You test it in Step 4.5.

# %%
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

guess_layer = ""        # YOUR TURN: "first" or "last"
guess_ratio = None      # YOUR TURN: roughly how many times bigger? (e.g. 10, 100, 1000)
print("Your guess: the", guess_layer if guess_layer else "(nothing yet)", "layer is bigger, by a factor of about", guess_ratio)

# %% [markdown]
# ### 4.1 Forward with memory, backward with four equations
#
# The forward pass of Part 2 computed every `z = Wa + b` and `a = φ(z)`. Now we **store** them, because the backward pass needs them. The backward pass starts with the **blame at the output**, `δ = a − y` for a sigmoid output with cross-entropy loss, or `δ = (a − y)·a(1 − a)` for the squared error. Then, layer by layer from the end, it applies the four equations: **(1)** the output blame, **(2)** the hidden blame `δ = (Wᵀδ_next) ⊙ φ′(z)`, **(3)** the weight gradient `dW = δ aᵀ` (the blame of a unit times the activation that came into it) and **(4)** the bias gradient `db = δ`. We use a **batch** of examples at once, one example per row, so the gradients are averaged over the batch. Read `forward` and `backward` slowly: each line is one of the four equations. The last cell of the block builds a `[3, 5, 4, 1]` network, runs a batch of 7 examples and prints the shape of everything.

# %%
# Activation functions and their derivatives. The derivative is written in terms of the OUTPUT a,
# because backpropagation has a (the stored activation) at hand: sigmoid' = a(1-a), tanh' = 1-a^2.
ACTS = {
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
    "tanh":    (np.tanh,                            lambda a: 1.0 - a ** 2),
    "relu":    (lambda z: np.maximum(0.0, z),       lambda a: (a > 0).astype(float)),
}

def init_params(sizes, seed=0):
    """Weights ~ N(0, 1/n_in), biases 0. params[l] = (W, b) with W of shape (n_out, n_in)."""
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, 1 / np.sqrt(n_in), size=(n_out, n_in)), np.zeros(n_out))
            for n_in, n_out in zip(sizes[:-1], sizes[1:])]

def forward(params, X, act="sigmoid"):
    """X has one example per ROW. Returns zs (pre-activations) and As (activations, As[0] = X).
    Hidden layers use `act`; the output layer always uses a sigmoid."""
    As, zs = [X], []
    for l, (W, b) in enumerate(params):
        z = As[-1] @ W.T + b                      # (n, n_in) @ (n_in, n_out) -> (n, n_out)
        f = ACTS["sigmoid" if l == len(params) - 1 else act][0]
        zs.append(z); As.append(f(z))
    return zs, As

def loss_value(a, Y, kind="bce"):
    """Mean over the batch. 'bce' = binary cross-entropy, 'mse' = 1/2 (a - y)^2 summed over outputs."""
    if kind == "mse":
        return float(np.mean(np.sum(0.5 * (a - Y) ** 2, axis=1)))
    e = 1e-12
    return float(np.mean(-np.sum(Y * np.log(a + e) + (1 - Y) * np.log(1 - a + e), axis=1)))

def backward(params, zs, As, Y, kind="bce", act="sigmoid"):
    """The four equations. Returns grads[l] = (dW, db) for every layer, averaged over the batch."""
    n, L = len(Y), len(params)
    a = As[-1]
    delta = (a - Y) if kind == "bce" else (a - Y) * a * (1 - a)    # equation 1: delta at the output
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        dW = delta.T @ As[l] / n                                   # equation 3: dW = delta a_prev^T
        db = delta.mean(axis=0)                                    # equation 4: db = delta
        grads[l] = (dW, db)
        if l > 0:                                                  # equation 2: send the blame back
            delta = (delta @ params[l][0]) * ACTS[act][1](As[l])
    return grads

# a first look at the shapes: a [3, 5, 4, 1] network and a batch of 7 examples
params = init_params([3, 5, 4, 1], seed=1)
rng = np.random.default_rng(2)
Xb, Yb = rng.normal(size=(7, 3)), (rng.random((7, 1)) > 0.5).astype(float)
zs, As = forward(params, Xb)
grads = backward(params, zs, As, Yb)
for l, ((W, b), (dW, db)) in enumerate(zip(params, grads)):
    print(f"layer {l+1}: W {W.shape}  b {b.shape}  a {As[l+1].shape}   ->   dW {dW.shape}  db {db.shape}")
    assert dW.shape == W.shape and db.shape == b.shape, "a gradient must have the shape of its parameter"
print("loss on the batch:", round(loss_value(As[-1], Yb), 4))
RESULTS["p4_shapes_ok"] = True

# %% [markdown]
# **What you should see.** For `[3, 5, 4, 1]` and a batch of 7 you should see `W (5, 3)`, `W (4, 5)`, `W (1, 4)`, activations of shape (7, 5), (7, 4), (7, 1), and for each layer a `dW` with **exactly the shape of its `W`** (the `assert` checks it). The loss on this random batch is 0.7503. A gradient always has the shape of the thing it is the gradient of; if yours does not, there is a transpose missing.

# %% [markdown]
# **Try also (optional):**
#
# - Change the loss to `"mse"` in the call to `backward` and print the shape of `dW` again: do the shapes change?
# - Print `As[1]` and `zs[0]` and say which of the two is the input to the next layer.
# - Run the cell with a batch of 1 example (`Xb[:1]`) and with 100 examples (make them with `rng.normal`) and confirm that nothing in the code mentions the batch size.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.1 - shapes of the blame
#
# (a) What is the shape of the blame `delta` that arrives at the hidden layer with 5 units for a batch of 32 examples? (b) What is the shape of `dW` for a layer that reads 5 numbers and has 3 neurons? (c) Which of the four equations has the transpose `W.T` (or the matrix `W` seen from the other side) in it, and why is that natural? Why it matters: almost every bug in hand-written backpropagation is a shape bug, and knowing the shapes in advance lets you catch it in seconds.

# %%
shape_delta = None     # YOUR ANSWER: a tuple, e.g. (rows, columns)
shape_dW = None        # YOUR ANSWER: a tuple
equation_with_W = None # YOUR ANSWER: 1, 2, 3 or 4

# %%
# Check
if shape_delta is None or shape_dW is None or equation_with_W is None:
    print("Fill in your three answers above, then run this cell again.")
else:
    check("delta at a 5-unit layer for 32 examples is (32, 5)", tuple(shape_delta) == (32, 5), "one row per example, one column per unit")
    check("dW of a 3-neuron layer reading 5 numbers is (3, 5)", tuple(shape_dW) == (3, 5), "dW has the shape of W: (n_out, n_in)")
    check("the equation that carries the blame back is number 2", equation_with_W == 2, "the blame of the next layer is pushed back through its weights")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# shape_delta = (32, 5)
# shape_dW = (3, 5)
# equation_with_W = 2
# ```
# Every example has one blame number per unit, so `delta` is (examples, units); `dW` has the shape of `W` because there is one gradient per weight. Equation 2 is the only one that multiplies by a weight matrix, because the blame of a hidden unit is the sum of the blame of the units it feeds, weighted by the connections.
# </details>

# %% [markdown]
# ### 4.2 Check it on the lesson's example, then on everything with a gradient check
#
# Two checks, from small to large. First the `[2, 2, 1]` example from the lesson (all sigmoid, loss `½(a − y)²`, input (1, 0.5), target 1): the loss must be **0.15448** and the gradient of the weight `W1[0][1]` must be **−0.016314**, the same numbers as in the lesson's figures and in your Part 3 engine. Then a **gradient check**: `grad_check` nudges every single parameter up and down by `ε = 10⁻⁵`, recomputes the loss, and compares the finite-difference slope `(L(w+ε) − L(w−ε)) / 2ε` with the gradient from `backward`. It returns the worst relative error over all parameters. This is the standard way to test any hand-written backward pass, and it takes five minutes that can save five days.

# %%
# the lesson's [2, 2, 1] example network, all sigmoid, loss 1/2 (a - y)^2, input (1, 0.5), target 1
W1 = np.array([ [0.5, -0.3], [0.8, 0.2] ]); b1 = np.array([0.1, -0.1])
W2 = np.array([ [1.0, -1.5] ]);             b2 = np.array([0.2])
ex_params = [ (W1.copy(), b1.copy()), (W2.copy(), b2.copy()) ]
x_ex, y_ex = np.array([ [1.0, 0.5] ]), np.array([ [1.0] ])

zs, As = forward(ex_params, x_ex)
loss = loss_value(As[-1], y_ex, "mse")
grads = backward(ex_params, zs, As, y_ex, "mse")
print("hidden activations a1 =", As[1].round(4), "  prediction a2 =", As[2].round(4))
print("loss                  =", round(loss, 5))
print("dL/dW1 =\n", grads[0][0].round(6))
print("dL/db1 =", grads[0][1].round(6), "  dL/dW2 =", grads[1][0].round(6), "  dL/db2 =", grads[1][1].round(6))
print("dL/dW1[0][1] =", round(float(grads[0][0][0][1]), 6), "   (the lesson says -0.016314)")
assert abs(loss - 0.15448) < 1e-5 and abs(grads[0][0][0][1] + 0.016314) < 1e-6, "should match the lesson's numbers"
RESULTS["p4_example_loss"] = round(loss, 5)
RESULTS["p4_example_dW1_01"] = round(float(grads[0][0][0][1]), 6)

# %% [markdown]
# **What you should see.** You should see hidden activations `[0.6106 0.69]`, the prediction `0.4442`, the loss `0.15448` and `dL/dW1[0][1] = -0.016314`. The whole `dL/dW1` is `[ [-0.032627, -0.016314], [0.044032, 0.022016] ]`, the output-layer gradients are `dL/dW2 = [-0.083797, -0.094684]` and `dL/db2 = -0.137228`. The `assert` passes silently. All gradients are negative at the output because the prediction (0.44) is below the target (1): the weights must grow.

# %%
def grad_check(params, X, Y, kind="bce", act="sigmoid", eps=1e-5):
    """Compare the backward() gradient of EVERY parameter with a centred finite difference.
    Returns the largest relative error. Parameters are changed in place and put back."""
    zs, As = forward(params, X, act)
    grads = backward(params, zs, As, Y, kind, act)
    worst = 0.0
    for l, (W, b) in enumerate(params):
        for arr, g in ((W, grads[l][0]), (b, grads[l][1])):
            for idx in np.ndindex(*arr.shape):
                old = arr[idx]
                arr[idx] = old + eps; lp = loss_value(forward(params, X, act)[1][-1], Y, kind)
                arr[idx] = old - eps; lm = loss_value(forward(params, X, act)[1][-1], Y, kind)
                arr[idx] = old
                num = (lp - lm) / (2 * eps)
                rel = abs(num - g[idx]) / max(abs(num) + abs(g[idx]), 1e-8)
                worst = max(worst, rel)
    return worst

e1 = grad_check(ex_params, x_ex, y_ex, "mse")
print("worst relative error, [2,2,1] example, MSE        :", f"{e1:.2e}")
params = init_params([3, 5, 4, 1], seed=1)
e2 = grad_check(params, Xb, Yb, "bce")
e3 = grad_check(params, Xb, Yb, "bce", act="tanh")
print("worst relative error, [3,5,4,1], BCE, sigmoid    :", f"{e2:.2e}")
print("worst relative error, [3,5,4,1], BCE, tanh       :", f"{e3:.2e}")
check("all three gradient checks are below 1e-6", max(e1, e2, e3) < 1e-6, "if not, one of the four equations has a bug")
RESULTS["p4_gradcheck_worst"] = float(max(e1, e2, e3))

# %% [markdown]
# **What you should see.** The worst relative errors are about `3.4e-11` (the example), `1.3e-8` (the `[3,5,4,1]` network with sigmoid and cross-entropy) and `3.1e-9` (the same with `tanh` hidden units), far below the `1e-6` threshold: the ✓ line appears. If one of them is above 1e-6 (try breaking `backward` by deleting `* ACTS[act][1](As[l])`) the gradient check tells you immediately, and the error per layer tells you where to look.

# %% [markdown]
# **Try also (optional):**
#
# - Change `eps` to `1e-2` and then to `1e-9`. The error gets worse at both ends. Why? (Large `ε` has truncation error; tiny `ε` loses digits to rounding.)
# - Run `grad_check(..., kind="mse")` on the `[3,5,4,1]` network.
# - Print the largest single relative error per parameter matrix instead of one number.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.2 - the same example with cross-entropy
#
# Run the same example network with the **cross-entropy** loss instead of the mean squared error. (a) What is the loss now (store it in `loss_bce`)? (b) What is `dL/dW1[0][1]` (store it in `g01_bce`)? (c) Is the gradient larger or smaller than with MSE (-0.016314), and why? Hint: with BCE the output blame is simply `a - y`, while with MSE it carries an extra factor `a(1-a)` which is at most 0.25. Why it matters: this is the reason classification networks use cross-entropy, a stronger blame signal at the output.

# %%
loss_bce = None        # YOUR CODE HERE: use forward(...) and loss_value(..., "bce")
g01_bce = None         # YOUR CODE HERE: use backward(..., "bce") and take [0][0][0][1]
bce_larger = None      # YOUR ANSWER: True if |g01_bce| is larger than 0.016314

# %%
# Check
if loss_bce is None or g01_bce is None:
    print("Fill in `loss_bce` and `g01_bce` above, then run this cell again.")
else:
    zs_c, As_c = forward(ex_params, x_ex)
    want_loss = loss_value(As_c[-1], y_ex, "bce")
    want_g = backward(ex_params, zs_c, As_c, y_ex, "bce")[0][0][0][1]
    check("the BCE loss is right (about 0.8116)", abs(loss_bce - want_loss) < 1e-6, "loss_value(As[-1], y_ex, 'bce')")
    check("dL/dW1[0][1] is right", abs(g01_bce - want_g) < 1e-8, "backward(..., 'bce')[0][0][0][1]")
    check("you noticed that BCE gives a larger gradient", bce_larger == (abs(g01_bce) > 0.016314), "compare absolute values")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# zs_c, As_c = forward(ex_params, x_ex)
# loss_bce = loss_value(As_c[-1], y_ex, "bce")
# g01_bce = float(backward(ex_params, zs_c, As_c, y_ex, "bce")[0][0][0][1])
# bce_larger = abs(g01_bce) > 0.016314
# ```
# With cross-entropy the output blame is `a - y` (about -0.556) instead of `(a - y) a (1 - a)` (about -0.137), so every gradient behind it is several times larger and the network learns faster when it is confidently wrong.
# </details>

# %% [markdown]
# ### 4.3 Compare with PyTorch autograd
#
# Your NumPy gradients should agree with PyTorch's `loss.backward()` to many digits. `to_torch` builds the same network as an `nn.Sequential`, copies your weights in, runs the same forward pass, calls `backward()` and compares every `dW` and `db`. We work in double precision (`.double()`), because single precision would show a difference of about 1e-8 that has nothing to do with correctness. The second block repeats the comparison on the bigger network for both sigmoid and tanh hidden units. Notice what you are doing: you are checking an industrial autograd engine against forty lines you wrote yourself.

# %%
import torch.nn.functional as F

def to_torch(params):
    """Build the same network in PyTorch (sigmoid everywhere, double precision) and copy our weights in."""
    layers = []
    for W, b in params:
        lin = nn.Linear(W.shape[1], W.shape[0]).double()
        lin.weight.data = torch.tensor(W); lin.bias.data = torch.tensor(b)
        layers += [lin, nn.Sigmoid()]
    return nn.Sequential(*layers)

net = to_torch(ex_params)
out = net(torch.tensor(x_ex))
loss_t = 0.5 * ((out - torch.tensor(y_ex)) ** 2).sum(1).mean()
loss_t.backward()
zs, As = forward(ex_params, x_ex)
g_np = backward(ex_params, zs, As, y_ex, "mse")
lins = [m for m in net if isinstance(m, nn.Linear)]
print("loss  numpy / torch:", round(loss_value(As[-1], y_ex, "mse"), 6), "/", round(float(loss_t), 6))
for l, lin in enumerate(lins):
    dw_ok = np.allclose(g_np[l][0], lin.weight.grad.numpy(), atol=1e-10)
    db_ok = np.allclose(g_np[l][1], lin.bias.grad.numpy(), atol=1e-10)
    print(f"layer {l+1}: dW equal? {dw_ok}   db equal? {db_ok}")
    assert dw_ok and db_ok, "our gradients should equal autograd"
print("torch dL/dW1[0][1] =", round(float(lins[0].weight.grad[0, 1]), 6))

# %% [markdown]
# **What you should see.** The loss is `0.154483` in both, and both layers print `dW equal? True   db equal? True`; PyTorch's `dL/dW1[0][1]` is `-0.016314`, the number from the lesson. The second block prints differences of about `1e-17`: the two methods agree to machine precision.

# %%
# a bigger test: [3,5,4,1], BCE, batch of 7, both activation functions that PyTorch has
for act, mod in (("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh)):
    params = init_params([3, 5, 4, 1], seed=1)
    layers = []
    for l, (W, b) in enumerate(params):
        lin = nn.Linear(W.shape[1], W.shape[0]).double()
        lin.weight.data = torch.tensor(W); lin.bias.data = torch.tensor(b)
        layers += [lin] + ([mod()] if l < len(params) - 1 else [nn.Sigmoid()])
    net = nn.Sequential(*layers)
    F.binary_cross_entropy(net(torch.tensor(Xb)), torch.tensor(Yb)).backward()
    zs, As = forward(params, Xb, act)
    g = backward(params, zs, As, Yb, "bce", act)
    lins = [m for m in net if isinstance(m, nn.Linear)]
    worst = max(np.abs(g[l][0] - lins[l].weight.grad.numpy()).max() for l in range(3))
    print(f"{act:8s}: largest difference between our dW and autograd's dW over all layers = {worst:.2e}")
    assert worst < 1e-9
RESULTS["p4_matches_torch"] = True

# %% [markdown]
# **What you should see.** Both lines show a largest difference around 1e-17 (1.4e-17 for sigmoid, 5.6e-17 for tanh). That is the proof that `loss.backward()` and your four equations compute the same thing: the engine is just a clever, general implementation of them.

# %% [markdown]
# **Try also (optional):**
#
# - Replace the output activation of the PyTorch network by nothing and use `BCEWithLogitsLoss`; check that the gradients are still equal.
# - Remove `.double()` and see what the difference becomes (about 1e-8 to 1e-9).
# - Use `torch.autograd.grad` instead of `.backward()` to get the gradient of the loss with respect to the first weight matrix.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.3 - find the bug in a broken backward
#
# Below is a **buggy** backward function that forgets the factor `phi'` (the derivative of the activation) in equation 2. Run it on the `[3,5,4,1]` network, compare each layer's `dW` with the correct `backward`, and store in `wrong_layers` the list of layer numbers (0 = the layer next to the input) whose gradient is wrong. Then explain in `why_output_ok` (one sentence) why the last layer is still right. Why it matters: this is exactly how you will debug your own gradients, layer by layer, from the output backwards.

# %%
def backward_buggy(params, zs, As, Y, kind="bce", act="sigmoid"):
    n, L = len(Y), len(params)
    delta = (As[-1] - Y)
    grads = [None] * L
    for l in range(L - 1, -1, -1):
        grads[l] = (delta.T @ As[l] / n, delta.mean(axis=0))
        if l > 0:
            delta = delta @ params[l][0]            # BUG: the factor phi'(a) is missing
    return grads

params = init_params([3, 5, 4, 1], seed=1)
zs, As = forward(params, Xb)
good = backward(params, zs, As, Yb, "bce")
bad = backward_buggy(params, zs, As, Yb, "bce")
wrong_layers = None      # YOUR CODE HERE: e.g. [l for l in range(3) if not np.allclose(good[l][0], bad[l][0])]
why_output_ok = ""       # YOUR ANSWER: one sentence

# %%
# Check
if wrong_layers is None:
    print("Fill in `wrong_layers` above, then run this cell again.")
else:
    check("layers 0 and 1 are wrong", list(wrong_layers) == [0, 1], "compare good[l][0] and bad[l][0] with np.allclose for l = 0, 1, 2")
    check("your explanation has at least 10 words", n_words(why_output_ok) >= 10, "the output delta (a - y) does not use phi' from a hidden layer")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# wrong_layers = [l for l in range(3) if not np.allclose(good[l][0], bad[l][0])]
# why_output_ok = "The last layer's blame is just a - y, which is computed before any hidden derivative is needed, so the missing factor only damages the layers behind it."
# ```
# The error appears in layer 2 and everything behind it (layers 0 and 1), never in front of it, because the bug is in the step that passes the blame backwards. A gradient check that prints the error per layer therefore points at the first wrong layer from the output side.
# </details>

# %% [markdown]
# ### 4.4 Train a network with your own backpropagation
#
# A gradient is only useful if you step against it. `train_np` repeats, for 1000 epochs: forward pass, loss, `backward`, and the update `W ← W − η dW` on all the layers (full batch, so every epoch uses all 300 points of `make_moons`). The network is `[2, 8, 8, 1]` with `tanh` hidden units, and the learning rate is 0.3. Then we build **exactly the same network in PyTorch** with the same starting weights and the same optimizer and learning rate, and compare the two training runs: the loss curves, the decision regions, the accuracy. There is no randomness left, so they should be the same run.

# %%
Xm, ym = make_moons(300, noise=0.2, seed=0)
Ym = ym.reshape(-1, 1).astype(float)

def train_np(sizes, X, Y, act="tanh", lr=0.3, epochs=1000, seed=0):
    """Full-batch gradient descent with our own forward() and backward(). Returns (params, loss history)."""
    params = init_params(sizes, seed)
    hist = []
    for _ in range(epochs):
        zs, As = forward(params, X, act)
        hist.append(loss_value(As[-1], Y))
        grads = backward(params, zs, As, Y, "bce", act)
        for (W, b), (dW, db) in zip(params, grads):
            W -= lr * dW; b -= lr * db                  # in place
    return params, hist

def np_accuracy(params, X, Y, act):
    return float(np.mean((forward(params, X, act)[1][-1] > 0.5) == (Y > 0.5)))

t0 = time.time()
params_np, hist_np = train_np([2, 8, 8, 1], Xm.astype(float), Ym, act="tanh", lr=0.3, epochs=1000)
acc_np = np_accuracy(params_np, Xm.astype(float), Ym, "tanh")
print(f"NumPy network [2,8,8,1]: loss {hist_np[0]:.3f} -> {hist_np[-1]:.3f}, accuracy {acc_np:.3f}  ({time.time()-t0:.1f} s)")
assert hist_np[-1] < hist_np[0] * 0.5, "the loss should fall"
RESULTS["p4_numpy_acc"] = round(acc_np, 3)

# %% [markdown]
# **What you should see.** You should see the loss fall from `0.745` to `0.038` and a training accuracy of `0.987` (in about 0.2 seconds). Your NumPy code, with no libraries beyond NumPy, learned the moons.

# %%
# the same network, the same starting weights, trained by PyTorch autograd
init = init_params([2, 8, 8, 1], seed=0)
tnet = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)).double()
for lin, (W, b) in zip([m for m in tnet if isinstance(m, nn.Linear)], init):
    lin.weight.data = torch.tensor(W); lin.bias.data = torch.tensor(b)
opt = torch.optim.SGD(tnet.parameters(), lr=0.3)
Xt, Yt = torch.tensor(Xm.astype(float)), torch.tensor(Ym)
hist_t = []
for _ in range(1000):
    opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(tnet(Xt), Yt); loss.backward(); opt.step()
    hist_t.append(float(loss))
with torch.no_grad():
    acc_t = float(((tnet(Xt) > 0).double() == Yt).double().mean())
print(f"PyTorch network, same start: loss {hist_t[0]:.3f} -> {hist_t[-1]:.3f}, accuracy {acc_t:.3f}")
print("largest difference between the two loss curves:", f"{np.max(np.abs(np.array(hist_np) - np.array(hist_t))):.2e}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].plot(hist_np, label="our NumPy"); ax[0].plot(hist_t, "--", label="PyTorch"); ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
plot_decision(lambda P: forward(params_np, P.astype(float), "tanh")[1][-1][:, 0], Xm, ym, ax=ax[1], title=f"NumPy, accuracy {acc_np:.2f}")
plot_decision(lambda P: torch.sigmoid(tnet(torch.tensor(P.astype(float)))).detach().numpy()[:, 0], Xm, ym, ax=ax[2], title=f"PyTorch, accuracy {acc_t:.2f}")
plt.tight_layout(); plt.show()
RESULTS["p4_torch_acc"] = round(acc_t, 3)

# %% [markdown]
# **What you should see.** PyTorch, from the same starting weights, ends with the loss `0.038` and accuracy `0.987` too, and the largest difference between the two loss curves over 1000 epochs is about `2e-12`. The three pictures show the loss curves lying on top of each other, and two identical S-shaped decision boundaries. Your 30 lines are not an approximation of `loss.backward()` plus `SGD`: they **are** that, for this architecture.

# %% [markdown]
# **Try also (optional):**
#
# - Change the learning rate to 0.01 and to 2.0 and describe the loss curve in each case.
# - Use `[2, 4, 1]` and `[2, 16, 16, 1]` and compare the accuracies.
# - Add a line that records the accuracy every 100 epochs and plot it next to the loss.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.4 - swap the activation
#
# Change the hidden activation to `"relu"` and find a learning rate `lr_mine` (try 0.05, 0.1, 0.3) for which the network reaches at least 0.85 accuracy in 1000 epochs. Store the accuracy in `acc_mine` and say in `which_better` ("tanh" or "relu") which activation gave the lower final loss at the same learning rate 0.1. Why it matters: it shows that your NumPy engine is general: one line changes the activation because `ACTS` holds the derivative too.

# %%
act_mine = "tanh"       # YOUR CODE HERE: try "relu"
lr_mine = 0.3           # YOUR CODE HERE: try 0.05, 0.1, 0.3
p_mine, h_mine = train_np([2, 8, 8, 1], Xm.astype(float), Ym, act=act_mine, lr=lr_mine, epochs=1000)
acc_mine = np_accuracy(p_mine, Xm.astype(float), Ym, act_mine)
print(act_mine, lr_mine, "-> final loss", round(h_mine[-1], 4), "accuracy", round(acc_mine, 3))
which_better = None     # YOUR ANSWER: "tanh" or "relu" (compare the final losses at lr = 0.1)

# %%
# Check
check("you used relu", act_mine == "relu", "set act_mine = 'relu'")
check("accuracy at least 0.85", acc_mine >= 0.85, "try lr_mine = 0.1 or 0.3")
if which_better is None:
    print("Fill in `which_better` (compare the final losses of tanh and relu at lr 0.1), then run this cell again.")
else:
    l_tanh = train_np([2, 8, 8, 1], Xm.astype(float), Ym, act="tanh", lr=0.1, epochs=1000)[1][-1]
    l_relu = train_np([2, 8, 8, 1], Xm.astype(float), Ym, act="relu", lr=0.1, epochs=1000)[1][-1]
    print(f"final loss at lr 0.1: tanh {l_tanh:.4f}, relu {l_relu:.4f}")
    check("your comparison is right", which_better == ("tanh" if l_tanh < l_relu else "relu"), "run both and compare")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# act_mine = "relu"
# lr_mine = 0.1
# p_mine, h_mine = train_np([2, 8, 8, 1], Xm.astype(float), Ym, act=act_mine, lr=lr_mine, epochs=1000)
# acc_mine = np_accuracy(p_mine, Xm.astype(float), Ym, act_mine)
# which_better = "tanh"
# ```
# At learning rate 0.1 tanh ends with the lower loss (0.067 against 0.109). Which activation wins at one learning rate is a fact about this data and this size, not a rule; the lesson to keep is that you can swap the activation by changing a name, because the derivative travels with it.
# </details>

# %% [markdown]
# ### 4.5 Watch the gradient flow
#
# Back to the Imagine-first question. `layer_norms` returns the size (the Frobenius norm) of the gradient `dW` of every layer. We build a `[2, 8, 8, 8, 8, 1]` network (four hidden layers) with **sigmoid** hidden units, record the norms at **epoch 0, 100 and 500** during training, print them and draw them as grouped bars on a log scale. Pay attention to two things: the *shape* of the bars at epoch 0 (which layer is largest?) and how the picture changes once the network has trained for a while.

# %%
def layer_norms(params, X, Y, act="sigmoid", kind="bce"):
    """Frobenius norm of dW for every layer, input side first, at the current weights."""
    zs, As = forward(params, X, act)
    return [float(np.linalg.norm(dW)) for dW, db in backward(params, zs, As, Y, kind, act)]

sizes = [2, 8, 8, 8, 8, 1]            # four hidden layers
params = init_params(sizes, seed=0)
snap = {}
for epoch in range(501):
    if epoch in (0, 100, 500):
        snap[epoch] = layer_norms(params, Xm.astype(float), Ym, "sigmoid")
    zs, As = forward(params, Xm.astype(float), "sigmoid")
    for (W, b), (dW, db) in zip(params, backward(params, zs, As, Ym, "bce", "sigmoid")):
        W -= 0.5 * dW; b -= 0.5 * db
for e, v in snap.items():
    print(f"epoch {e:3d}: gradient norm per layer (input side first):", np.round(v, 5))
    print(f"           first / last = {v[0] / v[-1]:.4f}")
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.27
for k, (e, v) in enumerate(snap.items()):
    ax.bar(np.arange(5) + (k - 1) * w, v, w, label=f"epoch {e}")
ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xticklabels(["layer 1\n(input side)", "2", "3", "4", "layer 5\n(output)"])
ax.set_ylabel("gradient norm (log scale)"); ax.set_title("Sigmoid network, 4 hidden layers"); ax.legend()
plt.tight_layout(); plt.show()
RESULTS["p4_ratio_epoch0"] = round(snap[0][0] / snap[0][-1], 4)

# %% [markdown]
# **What you should see.** At epoch 0 the norms from the first to the last layer are `[0.00046, 0.00257, 0.01364, 0.0512, 0.21336]`: each layer is about **four to five times smaller** than the one after it, and the first is about **460 times smaller** than the last (first / last = `0.0022`). This is the vanishing gradient in its smallest form: four hidden layers, and the first layer's gradient is already only about 0.2 percent of the last layer's. The factor per layer is close to 0.25, the maximum slope of the sigmoid, times the weights, which are of order 1. After 100 epochs the picture is flat: `[0.00082, 0.001, 0.00109, 0.00105, 0.00096]`, ratio `0.85`, and after 500 epochs `[0.00265, ..., 0.00325]`, ratio `0.82`. Once the network has found a decent solution all layers carry gradients of the same order. The vanishing problem is at its worst at the **start** of training, and in deeper networks it prevents training from ever getting off the ground (Part 5).

# %% [markdown]
# **Try also (optional):**
#
# - Repeat with `"tanh"` instead of `"sigmoid"` in the training loop and compare the epoch-0 ratios.
# - Use only two hidden layers (`[2, 8, 8, 1]`) and see how the ratio changes with depth.
# - Plot the first-layer norm over all 500 epochs, not only three snapshots.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.5 - the first measurement of the vanishing gradient
#
# Measure the same ratio **first layer / last layer at epoch 0** for (a) the sigmoid network above (`ratio_sigmoid`) and (b) a network with `"tanh"` hidden layers (`ratio_tanh`), both with `sizes = [2, 8, 8, 8, 8, 1]` and the same seed. Which is bigger, and why does the maximum slope (0.25 for sigmoid, 1 for tanh) explain it? Why it matters: it is the first measurement of the vanishing gradient, which is the whole of Part 5.

# %%
ratio_sigmoid = None    # YOUR CODE HERE: layer_norms(init_params(sizes, 0), ..., "sigmoid"), then first / last
ratio_tanh = None       # YOUR CODE HERE: the same with "tanh"

# %%
# Check
if ratio_sigmoid is None or ratio_tanh is None:
    print("Fill in the two ratios above, then run this cell again.")
else:
    ns = layer_norms(init_params(sizes, 0), Xm.astype(float), Ym, "sigmoid")
    nt = layer_norms(init_params(sizes, 0), Xm.astype(float), Ym, "tanh")
    check("sigmoid ratio is right", abs(ratio_sigmoid - ns[0] / ns[-1]) < 1e-6, "layer_norms(...)[0] / layer_norms(...)[-1]")
    check("tanh ratio is right", abs(ratio_tanh - nt[0] / nt[-1]) < 1e-6, "same, with 'tanh'")
    check("the tanh ratio is bigger (a healthier first layer)", ratio_tanh > ratio_sigmoid, "tanh' goes up to 1, sigmoid' only to 0.25")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ratio_sigmoid = (lambda v: v[0] / v[-1])(layer_norms(init_params(sizes, 0), Xm.astype(float), Ym, "sigmoid"))
# ratio_tanh = (lambda v: v[0] / v[-1])(layer_norms(init_params(sizes, 0), Xm.astype(float), Ym, "tanh"))
# ```
# Each step back multiplies the blame by the local slope, at most 0.25 for sigmoid and up to 1 for tanh, so the sigmoid network's first layer receives a much weaker signal.
# </details>

# %% [markdown]
# ### 4.6 Explain it in your own words
#
# Close the part by putting backpropagation into plain language. Use the picture of **blame**: the output is blamed first, the blame goes back through the connections, each weight gets the share of the blame that passed through it, and the blame is scaled down at every step. If you can say this simply, you understand the four equations.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.6 - the blame explanation
#
# Explain backpropagation as **blame** in your own words, in five sentences: who gets blamed first, how the blame moves back, what a weight's share of the blame is, why the stored activations are needed, and why the blame can fade. Use the words *delta*, *weight* and *slope*. Why it matters: if you can say it simply, you understand the four equations.

# %%
blame_text = ""     # YOUR ANSWER: five sentences
print(blame_text if blame_text else "(write your five sentences in blame_text and run this cell again)")

# %%
# Check
if not blame_text:
    print("Write your explanation above, then run this cell again.")
else:
    check("at least 40 words", n_words(blame_text) >= 40, "five sentences are about 40 to 70 words")
    for kw in ("delta", "weight", "slope"):
        check(f"you used the word '{kw}'", kw in blame_text.lower(), f"include '{kw}'")
    RESULTS["p4_blame_text"] = blame_text

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# blame_text = ("The output unit is blamed first: its delta is how wrong the prediction was. "
#               "That blame travels backwards, each hidden unit receiving the deltas of the units it feeds, multiplied by the weights that connect them. "
#               "A weight's share of the blame is the delta of the unit it points to times the activation that went through it. "
#               "That is why the forward pass must store the activations. "
#               "At every step the blame is also multiplied by the slope of the activation, so with small slopes it fades on the way to the first layers.")
# ```
# Any version that says: blame starts at the output, goes back through the weights, is scaled by slopes, uses stored activations, can shrink - is correct.
# </details>

# %% [markdown]
# **Before you move on (Part 4).** Check that you can say each of these in your own words: the four equations are the output blame `δ`, the hidden blame `(Wᵀδ) ⊙ φ′`, `dW = δ aᵀ` and `db = δ`; a gradient has the shape of its parameter; a gradient check compares every analytic gradient with a finite difference; and the gradient of the first layers of a sigmoid network is much smaller than the last layer's at the start of training.

# %% [markdown]
# ### Imagine first, revisited
#
# Run the next cell to compare your Imagine-first prediction with the facts.

# %%
print("Your guess: the", guess_layer if guess_layer else "(none)", "layer is bigger, by a factor of about", guess_ratio)
ratio0 = RESULTS["p4_ratio_epoch0"]
print(f"Measured at epoch 0: first / last = {ratio0}  ->  the {'last' if ratio0 < 1 else 'first'} layer has the larger gradient, by a factor of {1 / ratio0 if ratio0 < 1 else ratio0:.0f}")
RESULTS["p4_guess"] = (guess_layer, guess_ratio)

# %% [markdown]
# In a markdown cell write whether you were right about the layer and about the order of magnitude. If you expected the first layer to be larger, which part of the picture (the weights, or the slopes) did you not count?

# %% [markdown]
# ## Part 5. Watching gradients vanish and explode (30 min)
#
# *Lesson sections: "Vanishing" and "Exploding".* In Part 4 you saw the first layer of a four-layer sigmoid network get a gradient 460 times smaller than the last. Here you make the networks deeper and **measure** what happens. A deep network multiplies one factor per layer on the way back; a factor below 1 makes the gradient **vanish**, a factor above 1 makes it **explode**. You will use `make_deep` (the helper from Setup) to build networks with 2 to 24 layers, `layer_grad_norms` to measure the gradient of every layer, and `train_model` to train them for real, and you will see a sigmoid network stay frozen at the loss `ln 2` while a ReLU network of the same depth learns. ==The gradient reaching the first layer of a deep network is roughly a product of one factor per layer, so a small deviation from 1 grows exponentially with depth.==

# %% [markdown]
# ### ✏️ Imagine first
#
# Take a network with **10 hidden layers** of 16 units, trained on the two moons, with PyTorch's default initialization. Predict the **ratio of the first layer's gradient norm to the last layer's** for (a) a sigmoid network and (b) a ReLU network: something near 1, a thousandth, a millionth, a billionth? Write your guesses in `guess_ratio_sigmoid` and `guess_ratio_relu`. You test them in Step 5.2.

# %%
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

guess_ratio_sigmoid = None     # YOUR TURN: first-layer gradient / last-layer gradient, e.g. 1, 0.1, 0.001, 1e-6
guess_ratio_relu = None        # YOUR TURN: the same ratio for a ReLU network of the same depth
print("Your guesses  sigmoid:", guess_ratio_sigmoid, "   relu:", guess_ratio_relu)

# %% [markdown]
# ### 5.1 The product of factors
#
# Start with the arithmetic behind everything in this part. If every layer multiplies the backward signal by 0.25 (the *best case* for sigmoid, since `σ′ ≤ 0.25`), then after `n` layers the signal is `0.25ⁿ`. If every layer multiplies by 1.5, it is `1.5ⁿ`. The block prints both for `n = 2, 5, 10, 20, 50` (the same numbers as the lesson's tables) and plots them on a **log scale**, where an exponential becomes a straight line.

# %%
ns = [2, 5, 10, 20, 50]
print(" n   |   0.25^n      |   1.5^n")
for n in ns:
    print(f"{n:3d}  |  {0.25 ** n:12.3e}  |  {1.5 ** n:12.3e}")
n_axis = np.arange(0, 51)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(n_axis, 0.25 ** n_axis, label="0.25^n (best case for sigmoid)")
ax.plot(n_axis, 1.5 ** n_axis, label="1.5^n (each layer amplifies by 1.5)")
ax.axhline(1, c="gray", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("number of layers n"); ax.set_ylabel("gradient factor (log scale)")
ax.set_title("The same factor, multiplied n times"); ax.legend(); plt.tight_layout(); plt.show()
assert abs(0.25 ** 10 - 9.5367e-07) < 1e-10
RESULTS["p5_factors"] = {"0.25^10": float(0.25 ** 10), "1.5^20": float(1.5 ** 20)}

# %% [markdown]
# **What you should see.** You should see: `0.25^10 = 9.537e-07`, `0.25^20 = 9.095e-13`, `0.25^50 = 7.889e-31`, and on the other side `1.5^10 = 57.67`, `1.5^20 = 3325`, `1.5^50 = 6.4e+08`. On the plot the 0.25 line falls and the 1.5 line rises, both straight, crossing 1 at n = 0. After only 10 layers the best-case sigmoid signal is below a millionth; after 20 layers an amplification of 1.5 per layer is already a factor of three thousand.

# %% [markdown]
# **Try also (optional):**
#
# - Plot the factors 0.9, 1.0 and 1.1 for n up to 100. Which of them is 'safe'?
# - Compute how many layers it takes for a factor 0.5 to go below 1e-6.
# - Compute `0.25 ** 10 * 1.0e6`: how large must the signal at the output be for the first layer to see 1.0?

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.1 - how many layers until it is gone?
#
# (a) How many layers does it take until `0.25 ** n` is below one millionth (1e-6)? Store it in `n_vanish`. (b) How many until `1.5 ** n` is above one million? Store it in `n_explode`. (c) A factor of exactly 1.0 stays 1.0 forever: in one sentence, why is that not a practical recipe? Why it matters: a deep network multiplies a factor per layer, so a tiny difference from 1 is amplified exponentially.

# %%
n_vanish = None      # YOUR CODE HERE: a loop or math.log
n_explode = None     # YOUR CODE HERE
why_not_one = ""     # YOUR ANSWER

# %%
# Check
if n_vanish is None or n_explode is None:
    print("Fill in `n_vanish` and `n_explode` above, then run this cell again.")
else:
    check("n_vanish is 10", n_vanish == 10, "the smallest n with 0.25**n < 1e-6")
    check("n_explode is 35", n_explode == 35, "the smallest n with 1.5**n > 1e6")
    check("your sentence has at least 10 words", n_words(why_not_one) >= 10, "think about real weights and real inputs: the factor is never exactly 1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# n_vanish = next(n for n in range(1, 100) if 0.25 ** n < 1e-6)
# n_explode = next(n for n in range(1, 100) if 1.5 ** n > 1e6)
# why_not_one = "Real factors depend on the weights and the data, so they are never exactly one, and even 0.99 or 1.01 grows or shrinks exponentially over many layers."
# ```
# `0.25 ** 10` is 9.5e-7 (so 10 layers) and `1.5 ** 35` is 1.6e6 (so 35 layers). The factor is not a design choice you can set to 1: it comes from the weights and the slopes, which change with every step of training.
# </details>

# %% [markdown]
# ### 5.2 Per-layer gradient norms: sigmoid, tanh and ReLU
#
# Now the real thing. `make_deep(depth=10, width=16, act=..., init="default")` builds a stack of 10 hidden layers with PyTorch's default weight initialization; `layer_grad_norms(model, X, y)` does one forward and one backward pass with the cross-entropy loss on 300 points of `make_moons` and returns the gradient norm of each of the 11 weight matrices, the one next to the input first. We do this for the three activation functions with the same seed and draw all the norms in one bar chart with a log y-axis. Then a tolerant check confirms that the sigmoid network's first layer receives less than 1/100 of the last layer's gradient.

# %%
Xd, yd = make_moons(300, noise=0.2, seed=0)
acts_list = ["sigmoid", "tanh", "relu"]
norms = {}
for act in acts_list:
    torch.manual_seed(SEED)
    model = make_deep(depth=10, width=16, act=act, init="default")
    norms[act] = layer_grad_norms(model, Xd, yd)
    print(f"{act:8s} first layer {norms[act][0]:.2e}   last layer {norms[act][-1]:.2e}   first / last = {norms[act][0] / norms[act][-1]:.2e}")

fig, ax = plt.subplots(figsize=(8, 3.8))
for k, act in enumerate(acts_list):
    ax.bar(np.arange(11) + (k - 1) * 0.27, norms[act], 0.27, label=act)
ax.set_yscale("log"); ax.set_xticks(range(11)); ax.set_xticklabels([str(i + 1) for i in range(10)] + ["out"])
ax.set_xlabel("layer (1 = next to the input)"); ax.set_ylabel("gradient norm (log scale)")
ax.set_title("Depth 10, width 16, PyTorch default initialisation"); ax.legend(); plt.tight_layout(); plt.show()
RESULTS["p5_ratio_sigmoid"] = norms["sigmoid"][0] / norms["sigmoid"][-1]
RESULTS["p5_ratio_tanh"] = norms["tanh"][0] / norms["tanh"][-1]
RESULTS["p5_ratio_relu"] = norms["relu"][0] / norms["relu"][-1]

# %% [markdown]
# **What you should see.** The three lines read (your last digits may differ a little on another machine):
#
# - sigmoid: first layer `8.9e-10`, last layer `0.148`, ratio **`6.0e-09`**, a few parts in a billion;
# - tanh: first layer `6.1e-04`, last layer `0.022`, ratio **`2.8e-02`**;
# - relu: first layer `3.0e-06`, last layer `0.0139`, ratio **`2.1e-04`**.
#
# On the chart the sigmoid bars fall steeply and steadily from right to left, by roughly a factor of 7 per layer; tanh falls gently; ReLU sits in between. Nothing is exactly 1 with the default initialization, and the gradient at the front of a 10-layer sigmoid network is a billionth of the one at the back. Such a network cannot learn at the start.

# %%
check("sigmoid: the first layer gets less than 1/100 of the last layer's gradient", RESULTS["p5_ratio_sigmoid"] < 0.01, "the vanishing gradient")
# how does your measured ratio compare with the product of the slopes?  0.25 per layer is the BEST case for sigmoid:
print("0.25^9 =", f"{0.25 ** 9:.1e}", "(the ceiling for 9 sigmoid layers); the weights make the real factor a bit different")

# %% [markdown]
# **What you should see.** You should see the ✓ line, then `0.25^9 = 3.8e-06`. The measured sigmoid ratio (6.0e-09) is even smaller than the ceiling `0.25⁹`, because the real factor per layer is the slope (at most 0.25, usually less) times the size of the weights (below 1 for the default initialization).

# %% [markdown]
# **Try also (optional):**
#
# - Repeat with `depth=5` and `depth=20` and print the three ratios.
# - Use a different seed (`torch.manual_seed(7)`) and check that the ratio changes by a small factor but not by orders of magnitude.
# - Print `norms["relu"]` and describe in a sentence how it behaves from layer to layer.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.2 - initialization changes everything
#
# Measure the ratio for `tanh` with Xavier initialization (`init="xavier"`) and for `relu` with He initialization (`init="he"`), both depth 10, width 16. Store them in `ratio_tanh_xavier` and `ratio_relu_he`. Which of the five networks you have now measured (sigmoid, tanh, relu with default initialization; tanh with Xavier; relu with He) has the ratio closest to 1? Why it matters: the *initialization* is a design choice that changes the ratio by orders of magnitude.

# %%
ratio_tanh_xavier = None    # YOUR CODE HERE
ratio_relu_he = None        # YOUR CODE HERE
closest_to_one = ""         # YOUR ANSWER: e.g. "relu default"

# %%
# Check
if ratio_tanh_xavier is None or ratio_relu_he is None:
    print("Fill in the two ratios above, then run this cell again.")
else:
    torch.manual_seed(SEED); n1 = layer_grad_norms(make_deep(10, 16, "tanh", "xavier"), Xd, yd)
    torch.manual_seed(SEED); n2 = layer_grad_norms(make_deep(10, 16, "relu", "he"), Xd, yd)
    check("tanh + Xavier ratio is right", abs(ratio_tanh_xavier - n1[0] / n1[-1]) < 1e-6 * max(1, abs(n1[0] / n1[-1])), "first / last of layer_grad_norms")
    check("relu + He ratio is right", abs(ratio_relu_he - n2[0] / n2[-1]) < 1e-6 * max(1, abs(n2[0] / n2[-1])), "first / last of layer_grad_norms")
    check("you named one of the networks", closest_to_one != "", "compare how far each ratio is from 1 (look at the log)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# torch.manual_seed(SEED); n1 = layer_grad_norms(make_deep(10, 16, "tanh", "xavier"), Xd, yd)
# torch.manual_seed(SEED); n2 = layer_grad_norms(make_deep(10, 16, "relu", "he"), Xd, yd)
# ratio_tanh_xavier = n1[0] / n1[-1]; ratio_relu_he = n2[0] / n2[-1]
# closest_to_one = "see the printed ratios"
# ```
# Compare the ratios on a log scale (how many powers of ten away from 1); the answer is whichever number is closest to 1.0 when you print it.
# </details>

# %% [markdown]
# ### 5.3 A depth sweep
#
# How fast does the problem grow with depth? The block builds networks with 2, 4, 8, 12, 16 and 20 hidden layers, once with sigmoid and PyTorch's default initialization, once with ReLU and **He** initialization (more on it in Part 6), and measures the **first layer's** gradient norm in each. The log-scale line plot shows the trend; the `assert` records that the sigmoid gradient at depth 20 is thousands of times smaller than at depth 2.

# %%
depths = [2, 4, 8, 12, 16, 20]
first = {"sigmoid (default init)": [], "relu (He init)": []}
for d in depths:
    torch.manual_seed(SEED); m1 = make_deep(d, 16, "sigmoid", "default")
    torch.manual_seed(SEED); m2 = make_deep(d, 16, "relu", "he")
    first["sigmoid (default init)"].append(layer_grad_norms(m1, Xd, yd)[0])
    first["relu (He init)"].append(layer_grad_norms(m2, Xd, yd)[0])
print("depth |  first-layer gradient norm: sigmoid     relu(He)")
for i, d in enumerate(depths):
    print(f"{d:5d} |                            {first['sigmoid (default init)'][i]:.2e}   {first['relu (He init)'][i]:.2e}")
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for name, v in first.items():
    ax.plot(depths, v, "o-", label=name)
ax.set_yscale("log"); ax.set_xlabel("depth (hidden layers)"); ax.set_ylabel("gradient norm of the FIRST layer")
ax.set_title("Depth sweep"); ax.legend(); plt.tight_layout(); plt.show()
sig_first = first["sigmoid (default init)"]
assert sig_first[-1] < sig_first[0] * 1e-3, "the sigmoid gradient should shrink with depth"
RESULTS["p5_sweep_sigmoid_20"] = sig_first[-1]

# %% [markdown]
# **What you should see.** The sigmoid column falls like this: `7.3e-03` (depth 2), `1.1e-04` (4), `5.6e-08` (8), `9.1e-12` (12), `7.2e-15` (16), `1.4e-18` (20). On the log plot that is a nearly straight line: **a constant factor lost per layer**, about 1/7 per layer in this experiment. The ReLU/He column stays between `0.05` and `0.35` at all depths: it neither vanishes nor explodes (it wobbles by a factor of a few, which is normal). Gradients below about 1e-7 are lost entirely in single precision next to the weights.

# %% [markdown]
# **Try also (optional):**
#
# - Add a third line, tanh with Xavier initialization (`init="xavier"`).
# - Add depth 30 and 40 to the sweep. What happens to the sigmoid line near the floating-point limit?
# - Plot the *ratio* first / last instead of the first-layer norm.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.3 - the exponential shape
#
# Find the smallest depth in the sweep for which the sigmoid network's first-layer gradient norm falls below `1e-6`, and store it in `depth_1e6` (use `None` if it never does). Then add the depth-24 point yourself: build `make_deep(24, 16, "sigmoid", "default")`, measure its first-layer gradient, and store it in `g24`. Why it matters: it shows the exponential shape of the problem: each extra layer costs a constant factor.

# %%
depth_1e6 = "unset"     # YOUR CODE HERE: loop over `depths` and `sig_first`; None if it never happens
g24 = None              # YOUR CODE HERE

# %%
# Check
if g24 is None or depth_1e6 == "unset":
    print("Fill in `depth_1e6` and `g24` above, then run this cell again.")
else:
    want = next((d for d, g in zip(depths, sig_first) if g < 1e-6), None)
    check("depth_1e6 is right", depth_1e6 == want, "the first depth whose value in sig_first is below 1e-6")
    torch.manual_seed(SEED); want24 = layer_grad_norms(make_deep(24, 16, "sigmoid", "default"), Xd, yd)[0]
    check("g24 is right", abs(g24 - want24) <= 1e-6 * abs(want24) + 1e-30, "layer_grad_norms(model, Xd, yd)[0]")
    check("depth 24 is smaller than depth 20", g24 < sig_first[-1], "more layers, smaller gradient")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# depth_1e6 = next((d for d, g in zip(depths, sig_first) if g < 1e-6), None)
# torch.manual_seed(SEED); g24 = layer_grad_norms(make_deep(24, 16, "sigmoid", "default"), Xd, yd)[0]
# ```
# The straight line you see on the log plot is the exponential decay; extending it by four layers lowers the gradient by the same factor many times over.
# </details>

# %% [markdown]
# ### 5.4 Exploding gradients
#
# The opposite failure. The lesson models a layer's gradient factor as a function of the **scale** of the weights: if the weights are drawn from a normal distribution with standard deviation `s/√fan_in`, a ReLU network amplifies for `s` above `√2 = 1.41` (He) and shrinks below it. The first block sweeps `s` for a depth-10 ReLU network and prints how the first layer's gradient compares with that of He scale. The second block takes the extreme case, weights with standard deviation 1.0 (`init="large"`), trains it with SGD for 100 epochs next to a He-initialized network, and records the **loss** and the **size of all the weights**.

# %%
def set_scale(model, s):
    """Re-draw every weight as N(0, s^2 / fan_in), the lesson's 'scale s'. Biases to 0. He initialization is s = sqrt(2) = 1.41."""
    for lin in linear_layers(model):
        fan_in = lin.weight.shape[1]
        nn.init.normal_(lin.weight, 0, s / math.sqrt(fan_in)); nn.init.zeros_(lin.bias)

print("scale s | first-layer norm | last-layer norm | first / last")
rows = {}
for s in (0.7, 1.0, 1.41, 2.0, 3.0):
    torch.manual_seed(SEED)
    m = make_deep(10, 16, "relu", "default"); set_scale(m, s)
    nm = layer_grad_norms(m, Xd, yd)
    rows[s] = nm
    print(f"  {s:4.2f}  |   {nm[0]:12.3e}   |  {nm[-1]:12.3e}   | {nm[0] / nm[-1]:10.3e}")
base = rows[1.41][0]
print("\nfirst-layer gradient relative to He (s = 1.41):", {s: round(r[0] / base, 3) for s, r in rows.items()})
assert rows[2.0][0] > 5 * rows[1.41][0] and rows[0.7][0] < rows[1.41][0] / 5, "off-scale weights should change the gradient a lot"
RESULTS["p5_scale2_over_he"] = rows[2.0][0] / base
RESULTS["p5_scale07_over_he"] = rows[0.7][0] / base

# %% [markdown]
# **What you should see.** You should see that the first-layer gradient, relative to He scale (`s = 1.41`), is `0.0013` at s = 0.7 (about 800 times smaller), `0.045` at s = 1.0, `1` at s = 1.41, **`13.8`** at s = 2 and `802` at s = 3. That is the lesson's story in numbers: a scale only 40 percent too high amplifies the gradient 14 times, and a scale 50 percent too low reduces it about 800-fold (your lesson figure used a slightly different network, so its factors are a little different; the shape is the same). In this network the ratio first / last stays near 0.1 to 0.5 because both ends change together; what explodes or vanishes is the whole gradient.

# %%
def train_watch(model, X, y, epochs=100, lr=0.1):
    """Plain SGD with BCE; records the loss and the size of all weights every epoch."""
    X = torch.as_tensor(X, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses, wnorm = [], []
    for _ in range(epochs):
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(X), y); loss.backward(); opt.step()
        losses.append(float(loss))
        wnorm.append(float(torch.sqrt(sum((p ** 2).sum() for p in model.parameters()))))
    return losses, wnorm

torch.manual_seed(SEED); big = make_deep(10, 16, "relu", "large")           # weights N(0, 1)
loss_big, w_big = train_watch(big, Xd, yd, epochs=100, lr=0.1)
torch.manual_seed(SEED); ok = make_deep(10, 16, "relu", "he")
loss_ok, w_ok = train_watch(ok, Xd, yd, epochs=100, lr=0.1)
for e in (0, 1, 2, 5, 10, 50, 99):
    print(f"epoch {e:3d}:  large-init loss {loss_big[e]:12.4g}   |w| {w_big[e]:12.4g}      He-init loss {loss_ok[e]:8.4f}   |w| {w_ok[e]:7.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(np.nan_to_num(loss_big, nan=1e30, posinf=1e30), label="init N(0,1)"); ax[0].plot(loss_ok, label="He init")
ax[0].set_yscale("log"); ax[0].set_title("Loss (log scale)"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(np.nan_to_num(w_big, nan=1e30, posinf=1e30), label="init N(0,1)"); ax[1].plot(w_ok, label="He init")
ax[1].set_yscale("log"); ax[1].set_title("Size of all weights (log scale)"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()
RESULTS["p5_large_final_loss"] = loss_big[-1]

# %% [markdown]
# **What you should see.** Large initialization: the very first loss is **1232** (the network is hugely over-confident), after one update the loss is `9.8e+18`, and from epoch 2 on it is `nan` ("not a number"): the weights have become infinite. The He-initialized network is calm: the loss goes `0.608 → 0.575 → 0.546 → 0.428 → 0.414 → 0.235 → 0.195`, and the weight norm stays near 18.5 to 18.7. On the plots the line of the exploding run shoots off the top of the chart (we replaced `nan` by 1e30 only to be able to draw it); the other one descends calmly.

# %% [markdown]
# **Try also (optional):**
#
# - Try the scale sweep for a sigmoid network: does a larger scale rescue it?
# - Run the exploding experiment with `lr=0.001` and watch what happens in the first five epochs.
# - Print `float(torch.sqrt(sum((p**2).sum() for p in big.parameters())))` before training. How big are the weights at the start?

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.4 - the instability and the learning rate
#
# Train the He-initialized ReLU network of depth 10 with a larger learning rate: store in `lr_blow` the smallest learning rate from `[0.1, 0.3, 1, 3, 10]` for which the loss becomes `nan` or larger than 10 at some epoch (100 epochs). Then store in `lr_stable` a learning rate for which the **large-init** network ends with a finite loss below 10 (its starting loss is 1232; hint: much smaller, try powers of ten from 1e-3 down). Why it matters: exploding gradients and a too-large learning rate are two sides of the same instability.

# %%
lr_blow = None       # YOUR CODE HERE: loop over the list, train a fresh He network each time
lr_stable = 0.1      # YOUR CODE HERE: try a much smaller number, e.g. 1e-4

# %%
# Check
if lr_blow is None:
    print("Fill in `lr_blow` above, then run this cell again.")
else:
    def run(init, lr):
        torch.manual_seed(SEED); m = make_deep(10, 16, "relu", init)
        return train_watch(m, Xd, yd, epochs=100, lr=lr)[0]
    blew = [lr for lr in (0.1, 0.3, 1, 3, 10) if not np.all(np.isfinite(run("he", lr))) or max(run("he", lr)) > 10]
    check("lr_blow is the smallest learning rate that blows up", lr_blow == (blew[0] if blew else None), f"the learning rates that blow up are {blew}")
    ls = run("large", lr_stable)
    check("the large-init network ends finite and below 10", np.isfinite(ls[-1]) and ls[-1] < 10, f"final loss {ls[-1]:.3g}: try a smaller lr_stable")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# lr_blow = None
# for lr in (0.1, 0.3, 1, 3, 10):
#     torch.manual_seed(SEED); l = train_watch(make_deep(10, 16, "relu", "he"), Xd, yd, epochs=100, lr=lr)[0]
#     if (not np.all(np.isfinite(l))) or max(l) > 10:
#         lr_blow = lr; break
# lr_stable = 1e-4
# ```
# Both numbers depend on the depth and the data; the pattern is general: the more the layers amplify, the smaller the safe step.
# </details>

# %% [markdown]
# ### 5.5 Train deep networks for real
#
# Gradients on paper are one thing; training is what matters. `run_deep` trains a depth-10, width-16 network on the moons with plain SGD (learning rate 0.1, 400 epochs, full batch) using the `train_model` helper from Setup, which also records the first-layer gradient norm at every epoch. We compare a **sigmoid** network (default initialization) and a **ReLU** network (He initialization). Both see exactly the same data and learning rate. A network that has learned *nothing* predicts probability 0.5 for everything, which has a cross-entropy loss of `ln 2 = 0.6931`, so look for that number.

# %%
def run_deep(act, init, depth=10, lr=0.1, epochs=400, seed=SEED, opt="sgd"):
    torch.manual_seed(seed)
    model = make_deep(depth, 16, act, init)
    h = train_model(model, Xd, yd, epochs=epochs, lr=lr, opt=opt, seed=seed)
    return h

h_sig = run_deep("sigmoid", "default")
h_rel = run_deep("relu", "he")
print(f"sigmoid depth 10: loss {h_sig['loss'][0]:.4f} -> {h_sig['loss'][-1]:.4f}   accuracy {h_sig['acc']:.3f}")
print(f"relu    depth 10: loss {h_rel['loss'][0]:.4f} -> {h_rel['loss'][-1]:.4f}   accuracy {h_rel['acc']:.3f}")
print("ln 2 =", round(math.log(2), 4), "  (the loss of a network that always says 'probably 50 %')")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(h_sig["loss"], label="sigmoid"); ax[0].plot(h_rel["loss"], label="relu + He"); ax[0].axhline(math.log(2), c="gray", ls=":")
ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(h_sig["grad_first"], label="sigmoid, first layer"); ax[1].plot(h_rel["grad_first"], label="relu, first layer")
ax[1].set_yscale("log"); ax[1].set_title("First-layer gradient norm"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()
RESULTS["p5_sigmoid_acc"] = h_sig["acc"]; RESULTS["p5_relu_acc"] = h_rel["acc"]
check("the sigmoid net is stuck near ln 2 (loss above 0.6)", h_sig["loss"][-1] > 0.6, "if not, it escaped: try lr 0.05")
check("the relu net learned (accuracy above 0.9)", h_rel["acc"] > 0.9, "check the learning rate")

# %% [markdown]
# **What you should see.** You should see `sigmoid depth 10: loss 0.7059 -> 0.6931   accuracy 0.500` and `relu depth 10: loss 0.6077 -> 0.1725   accuracy 0.943`. The sigmoid network sits on the line `ln 2 = 0.6931`: it has learned **nothing**, and its accuracy is that of a coin flip. The ReLU network of the same depth and width learns: loss 0.17, accuracy 0.94. The right plot shows why: the sigmoid net's first-layer gradient stays between 9e-10 and 1.1e-9 for all 400 epochs, so the first layers never move, while the ReLU network's first-layer gradient is a healthy number (between 0.015 and 3) throughout. Both ✓ lines appear.

# %% [markdown]
# **Try also (optional):**
#
# - Increase the epochs of the sigmoid network to 2000. Does it ever escape the plateau?
# - Try `lr=1.0` for the sigmoid network. What changes?
# - Give the sigmoid network only 4 hidden layers (`depth=4`) and see that it learns: depth is the problem, not sigmoid itself.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.5 - does a better optimizer repair it?
#
# Give the stuck sigmoid network a better optimizer: train `run_deep("sigmoid", "default", opt="adam", lr=0.01)` and store its accuracy in `acc_sig_adam`. Does Adam repair the vanishing gradient, or only work around it? Write your answer in `adam_note` (one or two sentences) after you compare `grad_first` for SGD and Adam. Why it matters: Adam rescales each weight's step by its own gradient history, so a tiny but consistent gradient still moves; the signal is not bigger, the step is.

# %%
acc_sig_adam = None     # YOUR CODE HERE
adam_note = ""          # YOUR ANSWER

# %%
# Check
if acc_sig_adam is None:
    print("Fill in `acc_sig_adam` above, then run this cell again.")
else:
    h_a = run_deep("sigmoid", "default", opt="adam", lr=0.01)
    print("Adam: first-layer gradient at epoch 0 =", f"{h_a['grad_first'][0]:.2e}", "  SGD:", f"{h_sig['grad_first'][0]:.2e}")
    check("your accuracy matches a fresh Adam run", abs(acc_sig_adam - h_a["acc"]) < 0.02, "opt='adam', lr=0.01, 400 epochs")
    check("your note has at least 12 words", n_words(adam_note) >= 12, "mention that the gradient is equally small at the start")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# acc_sig_adam = run_deep("sigmoid", "default", opt="adam", lr=0.01)["acc"]
# adam_note = "The first-layer gradient at the start is just as tiny with Adam, so the vanishing is not repaired; Adam only rescales the step per weight, so tiny gradients still move the weights."
# ```
# Adam works around the problem by normalising the step, which is why it helps, but the information in the gradient is still thin and noisy in a deep sigmoid network.
# </details>

# %% [markdown]
# ### 5.6 Dead ReLUs
#
# ReLU has its own failure mode. A unit whose input is negative for *every* example has slope 0 everywhere on the data, so no gradient reaches its weights and it can never recover: it is **dead**. The block builds a depth-6 He network, shifts all biases to a negative constant and uses `active_fraction` to count, for every layer, the share of units that are never positive on the data. Then the second block trains a network from a bad state (bias -0.5) twice from the same starting weights: with ReLU, and with **Leaky ReLU** (slope 0.1 for negative inputs), which never has zero slope.

# %%
import copy

def active_fraction(model, X):
    """For every hidden Linear layer: the fraction of units whose pre-activation is positive for at least one example.
    A unit that is never positive is 'dead' for a ReLU (its slope is 0 everywhere on the data)."""
    h, fr, lins = torch.as_tensor(X, dtype=torch.float32), [], linear_layers(model)
    with torch.no_grad():
        for m in model:
            h = m(h)
            if isinstance(m, nn.Linear) and m is not lins[-1]:
                fr.append(float((h > 0).any(dim=0).float().mean()))
    return fr

print("bias shift | dead units per layer (fraction never active), depth 6, width 16")
for shift in (0.0, -0.25, -0.5, -1.0, -2.0):
    torch.manual_seed(SEED); m = make_deep(6, 16, "relu", "he")
    for lin in linear_layers(m): nn.init.constant_(lin.bias, shift)
    dead = [1 - f for f in active_fraction(m, Xd)]
    print(f"  {shift:5.2f}   |", np.round(dead, 2), "  mean", round(float(np.mean(dead)), 2))

# %% [markdown]
# **What you should see.** The mean dead fraction over the six layers grows with the negative shift: `0.10` at 0 (some units are dead even for the default bias), `0.25` at -0.25, `0.62` at -0.5, `0.86` at -1 and `0.93` at -2; the last layers die first: at -1 every unit of layers 3 to 6 is dead.

# %%
def train_from_dead(leaky, shift=-0.5, epochs=300, lr=0.1):
    torch.manual_seed(SEED); m = make_deep(6, 16, "relu", "he")
    for lin in linear_layers(m): nn.init.constant_(lin.bias, shift)
    if leaky:                                                   # swap every ReLU for a LeakyReLU, same weights
        m = nn.Sequential(*[nn.LeakyReLU(0.1) if isinstance(l, nn.ReLU) else l for l in m])
    d0 = float(np.mean([1 - f for f in active_fraction(m, Xd)]))
    h = train_model(m, Xd, yd, epochs=epochs, lr=lr)
    d1 = float(np.mean([1 - f for f in active_fraction(m, Xd)]))
    return h, d0, d1

h_r, d0_r, d1_r = train_from_dead(False)
h_l, d0_l, d1_l = train_from_dead(True)
print(f"ReLU      : dead {d0_r:.2f} -> {d1_r:.2f}   loss {h_r['loss'][0]:.4f} -> {h_r['loss'][-1]:.4f}   accuracy {h_r['acc']:.3f}")
print(f"LeakyReLU : dead {d0_l:.2f} -> {d1_l:.2f}   loss {h_l['loss'][0]:.4f} -> {h_l['loss'][-1]:.4f}   accuracy {h_l['acc']:.3f}")
RESULTS["p5_dead_relu_acc"] = h_r["acc"]; RESULTS["p5_leaky_acc"] = h_l["acc"]

# %% [markdown]
# **What you should see.** Starting at bias -0.5 (62 percent dead): the ReLU network does not improve, `loss 0.7241 -> 0.6931`, accuracy `0.500`, and the dead fraction stays at `0.62`. The Leaky network, same weights, goes `loss 0.7262 -> 0.4486`, accuracy **`0.780`**, and its dead fraction falls from 0.55 to **0.38**: gradients leaked through the negative side and pulled units back. (Leaky ReLU is not a magic cure: with a stronger shift such as -1 the same experiment stays stuck at 0.693. Not starting from a bad state, which is the job of a good initialization, is the first remedy.)

# %% [markdown]
# **Try also (optional):**
#
# - Use `nn.ELU()` or `nn.GELU()` instead and compare.
# - Train from bias shift 0 with a very large learning rate (2.0) and measure the dead fraction afterwards: a big step can kill units too.
# - Change the leak to 0.01 (PyTorch's default) and 0.3.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.6 - how much damage
#
# Find, with the `active_fraction` function, the bias shift (from `[0, -0.25, -0.5, -1, -2]`) that is the **mildest** shift for which the mean dead fraction of the 6-layer He network exceeds 0.25 on the data. Store it in `shift_25`. Then say in `dead_note` why a dead unit cannot recover by itself under plain gradient descent. Why it matters: dead units are capacity you paid for but cannot use, and the cause is hidden: the loss just stops improving.

# %%
shift_25 = None      # YOUR CODE HERE: loop over the shifts like the first cell of this step
dead_note = ""       # YOUR ANSWER

# %%
# Check
if shift_25 is None:
    print("Fill in `shift_25` above, then run this cell again.")
else:
    def mean_dead(shift):
        torch.manual_seed(SEED); m = make_deep(6, 16, "relu", "he")
        for lin in linear_layers(m): nn.init.constant_(lin.bias, shift)
        return float(np.mean([1 - f for f in active_fraction(m, Xd)]))
    want = next((s for s in (0, -0.25, -0.5, -1, -2) if mean_dead(s) > 0.25), None)
    check("shift_25 is right", shift_25 == want, "the first shift in the list whose mean dead fraction is above 0.25")
    check("your note has at least 10 words", n_words(dead_note) >= 10, "think about the gradient of a ReLU for a negative input")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def _md(shift):
#     torch.manual_seed(SEED); m = make_deep(6, 16, "relu", "he")
#     for lin in linear_layers(m): nn.init.constant_(lin.bias, shift)
#     return float(np.mean([1 - f for f in active_fraction(m, Xd)]))
# shift_25 = next((s for s in (0, -0.25, -0.5, -1, -2) if _md(s) > 0.25), None)
# dead_note = "For a negative input the ReLU's slope is exactly zero, so no gradient reaches the unit's weights, they never change, and the unit stays negative for every example."
# ```
# Zero slope means zero gradient means no update; LeakyReLU keeps a small slope (0.1 here) so the unit can still be pulled back.
# </details>

# %% [markdown]
# **Before you move on (Part 5).** Check that you can say each of these in your own words: the backward signal is a product of one factor per layer, so a factor below 1 vanishes and above 1 explodes exponentially; sigmoid's slope is at most 0.25, so a deep sigmoid network is stuck near `ln 2`; ReLU does not vanish like that but can explode with a large initialization and can die; and a gradient of exactly zero means no learning, whatever the optimizer.

# %% [markdown]
# ### Imagine first, revisited
#
# Run the next cell to compare your Imagine-first prediction with the facts.

# %%
print("Your guesses  sigmoid:", guess_ratio_sigmoid, "  relu:", guess_ratio_relu)
print(f"Measured      sigmoid: {RESULTS['p5_ratio_sigmoid']:.1e}   relu: {RESULTS['p5_ratio_relu']:.1e}   (tanh: {RESULTS['p5_ratio_tanh']:.1e})")
RESULTS["p5_guesses"] = (guess_ratio_sigmoid, guess_ratio_relu)

# %% [markdown]
# In a markdown cell write how far off you were (in powers of ten) and for which of the two networks your intuition was worse. Why is the ReLU ratio not 1 either?

# %% [markdown]
# ## Part 6. Repairing deep networks (25 min)
#
# *Lesson section: "Remedies".* You can now measure a sick deep network. This part is the pharmacy. You build one benchmark where the baseline fails (20 sigmoid layers on the moons) and try the remedies from the lesson **one at a time**, always with the same data, learning rate and number of epochs: ReLU with He initialization, tanh with Xavier initialization, layer normalization, residual connections and the Adam optimizer. Then you meet **gradient clipping** for exploding gradients, compute initializations by hand and check that the weights really have the standard deviation they should, and write a debugging checklist. ==A remedy works by keeping the factor per layer close to 1, so the gradient reaches the first layer at a usable size.==

# %% [markdown]
# ### ✏️ Imagine first
#
# Here are four remedies for a 20-layer network that does not train: **ReLU with He initialization**, **tanh with Xavier initialization**, **LayerNorm**, and **residual connections**. Write them in `rank_guess` from the one you expect to help **most** to the one you expect to help **least**, using the exact names "relu+he", "tanh+xavier", "layernorm", "residual". You compare with the measured table in Step 6.2.

# %%
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

rank_guess = []      # YOUR TURN: order these from MOST to LEAST helpful: "relu+he", "tanh+xavier", "layernorm", "residual"
print("Your ranking:", rank_guess if rank_guess else "(nothing yet: write a list of the four names and run this cell again)")

# %% [markdown]
# ### 6.1 A benchmark where the baseline fails
#
# One fair test needs one fixed setup. The block creates 300 points of `make_moons` (noise 0.2), shuffles them with a fixed seed and splits them into **200 for training and 100 for validation**. The function `run(...)` builds a network with `make_deep`, trains it for 300 epochs of full-batch SGD (learning rate 0.05) with `train_model`, and returns training and validation accuracy, the final loss, and the **gradient ratio first / last layer at epoch 0**, the diagnosis number from Part 5. The baseline is the failure case: **20 sigmoid layers of width 16 with the default initialization.**

# %%
Xall, yall = make_moons(300, noise=0.2, seed=1)
perm = np.random.RandomState(0).permutation(300)
itr, iva = perm[:200], perm[200:]
Xtr, ytr, Xva, yva = Xall[itr], yall[itr], Xall[iva], yall[iva]
print("train", Xtr.shape, "validation", Xva.shape, " class balance in train:", round(float(ytr.mean()), 2))

def run(act="sigmoid", init="default", depth=20, norm=False, residual=False, opt="sgd", lr=0.05, epochs=300, seed=SEED):
    """Train one depth-`depth` network and return the numbers we care about."""
    torch.manual_seed(seed)
    m = make_deep(depth, 16, act, init, norm=norm, residual=residual)
    h = train_model(m, Xtr, ytr, epochs=epochs, lr=lr, opt=opt, Xv=Xva, yv=yva, seed=seed)
    return {"train": h["acc"], "val": h["acc"] if False else h["val_acc"], "loss": h["loss"][-1],
            "ratio0": h["grad_first"][0] / h["grad_last"][0], "hist": h}

base = run()
print(f"baseline (sigmoid, default init, depth 20): train {base['train']:.3f}  val {base['val']:.3f}  final loss {base['loss']:.4f}  first/last gradient at epoch 0 = {base['ratio0']:.2e}")
RESULTS["p6_baseline_val"] = base["val"]; RESULTS["p6_baseline_ratio"] = base["ratio0"]

# %% [markdown]
# **What you should see.** You should see `train (200, 2) validation (100, 2)  class balance in train: 0.51`, and then `baseline (sigmoid, default init, depth 20): train 0.510  val 0.480  final loss 0.6929  first/last gradient at epoch 0 = 3.41e-17`. The network predicts one class for everything (the accuracies are those of the class balance) and the loss is `ln 2`. The ratio is 3e-17: the first layer receives the gradient of the last divided by thirty quadrillion.

# %% [markdown]
# **Try also (optional):**
#
# - Change the learning rate of the baseline to 0.5 and 5.0. Does a bigger step help a gradient that is 1e-17 of the right size?
# - Change the seed to 1 and 2 and confirm that the baseline fails in the same way.
# - Print the validation accuracy of a network that always answers the majority class, to calibrate what '0.48' means.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.1 - how fast does it get worse
#
# Run the baseline at depth 10 and at depth 20 and store the two epoch-0 ratios in `ratio_d10` and `ratio_d20`. By how many powers of ten does the ratio change when the depth doubles? Store that rough number in `powers_of_ten` (a whole number). Why it matters: it tells you how fast the problem grows with depth.

# %%
ratio_d10 = None        # YOUR CODE HERE: run(depth=10)["ratio0"]
ratio_d20 = base["ratio0"]
powers_of_ten = None    # YOUR ANSWER: use math.log10(ratio_d10 / ratio_d20)

# %%
# Check
if ratio_d10 is None or powers_of_ten is None:
    print("Fill in `ratio_d10` and `powers_of_ten` above, then run this cell again.")
else:
    want = run(depth=10)["ratio0"]
    check("ratio_d10 is right", abs(ratio_d10 - want) < 1e-6 * abs(want) + 1e-30, "run(depth=10)['ratio0']")
    check("depth 20 is worse than depth 10", ratio_d20 < ratio_d10, "the ratio should fall with depth")
    check("powers_of_ten is within 1 of the truth", abs(powers_of_ten - math.log10(want / ratio_d20)) <= 1, "math.log10(ratio_d10 / ratio_d20), rounded")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ratio_d10 = run(depth=10)["ratio0"]
# powers_of_ten = round(math.log10(ratio_d10 / ratio_d20))
# ```
# Ten extra layers cost a roughly constant number of powers of ten, which is what an exponential looks like on a log scale.
# </details>

# %% [markdown]
# ### 6.2 One remedy at a time
#
# The seven rows are: the baseline; **ReLU + He**; **tanh + Xavier**; **sigmoid + LayerNorm** (a `nn.LayerNorm` after every linear layer); **sigmoid + residual** (every block adds its input to its output, `h + f(h)`); **ReLU + residual** (default initialization); and the baseline with the **Adam** optimizer (learning rate 0.01). Each row is trained with three different seeds (0, 1, 2) and we report the mean training and validation accuracy, the minimum and maximum validation accuracy, and the mean epoch-0 gradient ratio. It takes about 20 seconds. Three seeds is the smallest number that shows how much a result moves; it is still a small experiment, so read the *pattern*, not the third decimal.

# %%
configs = {
    "baseline: sigmoid":        dict(act="sigmoid", init="default"),
    "ReLU + He":                dict(act="relu", init="he"),
    "tanh + Xavier":            dict(act="tanh", init="xavier"),
    "sigmoid + LayerNorm":      dict(act="sigmoid", init="default", norm=True),
    "sigmoid + residual":       dict(act="sigmoid", init="default", residual=True),
    "ReLU + residual":          dict(act="relu", init="default", residual=True),
    "baseline + Adam":          dict(act="sigmoid", init="default", opt="adam", lr=0.01),
}
table = {}
t0 = time.time()
for name, cfg in configs.items():
    runs = [run(seed=s, **cfg) for s in (0, 1, 2)]
    table[name] = {k: float(np.mean([r[k] for r in runs])) for k in ("train", "val", "ratio0")}
    table[name]["val_range"] = (min(r["val"] for r in runs), max(r["val"] for r in runs))
print(f"(took {time.time() - t0:.0f} s)  mean of 3 seeds")
print(f"{'remedy':22s} | train acc | val acc (min..max)   | first/last gradient")
for name, r in table.items():
    print(f"{name:22s} |   {r['train']:.3f}   | {r['val']:.3f} ({r['val_range'][0]:.2f}..{r['val_range'][1]:.2f}) | {r['ratio0']:.2e}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
names = list(table)
ax[0].barh(names, [table[n]["val"] for n in names], color="tab:green"); ax[0].set_xlim(0.4, 1); ax[0].set_title("Validation accuracy"); ax[0].invert_yaxis()
ax[1].barh(names, [table[n]["ratio0"] for n in names], color="tab:purple"); ax[1].set_xscale("log"); ax[1].set_title("First / last gradient at epoch 0"); ax[1].invert_yaxis()
plt.tight_layout(); plt.show()
RESULTS["p6_table"] = {n: {"val": round(r["val"], 3), "ratio0": r["ratio0"]} for n, r in table.items()}

# %% [markdown]
# **What you should see.** Our run (mean of three seeds; yours should be close):
#
# | remedy | train acc | val acc (min..max) | first/last gradient |
# |---|---|---|---|
# | baseline: sigmoid | 0.510 | 0.480 (0.48..0.48) | 3.3e-17 |
# | ReLU + He | 0.950 | 0.933 (0.88..0.97) | 0.58 |
# | tanh + Xavier | 0.943 | 0.943 (0.89..0.98) | 3.4 |
# | sigmoid + LayerNorm | 0.510 | 0.480 (0.48..0.48) | 3.1e-09 |
# | sigmoid + residual | 0.625 | 0.613 (0.52..0.79) | 4.2e-03 |
# | ReLU + residual | 0.970 | 0.963 (0.95..0.98) | 0.11 |
# | baseline + Adam | 0.510 | 0.480 (0.48..0.48) | 3.3e-17 |
#
# Read it honestly. (1) The three remedies that bring the gradient ratio within a factor of ten of 1 (ReLU + He, tanh + Xavier, ReLU + residual) all reach 0.93 to 0.96 validation accuracy, with ReLU + residual the best in this experiment and the most consistent (0.95..0.98). (2) The remedies that leave the ratio tiny do not repair the network: LayerNorm (3e-9) lifts the ratio by eight powers of ten but not enough, because the 0.25 slope of 20 sigmoids is still there; Adam keeps the gradient 1e-17 and cannot rescue it. (3) Sigmoid + residual improves the ratio from 1e-17 to 4e-3 and the accuracy to 0.61 on average, but it is slow and seed-dependent (0.52..0.79 in 300 epochs). (4) The ranking of the three good rows is within the noise of three seeds. The robust conclusions are: **choose an activation and initialization that keep the factor near 1; a gradient ratio near 1 predicts a network that trains; normalization and an optimizer do not replace that.**

# %% [markdown]
# **Try also (optional):**
#
# - Add a row `act="relu", init="he", norm=True` (ReLU + He + LayerNorm). Does normalization help when the activation is ReLU?
# - Add a row for sigmoid + residual with `lr=0.3` and see how much slower learning was the problem.
# - Increase the epochs to 1000 for the baseline + Adam row. Does it ever escape?

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.2 - two remedies do not always add up
#
# Two remedies do not always add up. Train a **ReLU residual** network with **He** initialization (`act="relu", init="he", residual=True`) with seed 0 and store its validation accuracy in `val_he_res`. Compare it with the row "ReLU + residual" of the table (default initialization). Then, in `res_note`, explain why this combination goes wrong: in a residual block the output is `h + f(h)`, so what happens to the size of the signal after 19 blocks if `f(h)` is already as large as `h`? Finally write your own ranking of the seven rows by *validation accuracy* in `ranking` (best first). Why it matters: a remedy is a tool with a purpose, and stacking tools blindly can break things.

# %%
val_he_res = None    # YOUR CODE HERE: run(act="relu", init="he", residual=True, seed=0)["val"]
res_note = ""        # YOUR ANSWER: 2 sentences
ranking = []         # YOUR ANSWER: the seven names of `table`, best first

# %%
# Check
if val_he_res is None or not ranking:
    print("Fill in `val_he_res` and `ranking` above, then run this cell again.")
else:
    check("He + residual does clearly worse than ReLU + residual", val_he_res < table["ReLU + residual"]["val"] - 0.1, "run(act='relu', init='he', residual=True, seed=0)['val'] should be near 0.5")
    best = sorted(table, key=lambda n: -table[n]["val"])
    check("your ranking lists all seven names", sorted(ranking) == sorted(table), "use the names printed in the table")
    check("your top entry is among the top two in the table", ranking[0] in best[:2], f"the table's best two are {best[:2]}")
    check("your note has at least 15 words", n_words(res_note) >= 15, "say what the sum h + f(h) does to the size of the signal")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# val_he_res = run(act="relu", init="he", residual=True, seed=0)["val"]
# res_note = "Each block adds a branch that is as large as its input, so the signal roughly doubles in variance per block and after 19 blocks it is huge; the output saturates and the network cannot learn. Residual blocks need a small initial branch (or a normalised one), not He weights."
# ranking = sorted(table, key=lambda n: -table[n]["val"])
# ```
# Your numbers decide the ranking. The key point is that He initialization assumes a plain stack, where each layer sees only the previous one; a residual sum adds the branch to the input and doubles the variance each time.
# </details>

# %% [markdown]
# ### 6.3 Gradient clipping
#
# Clipping is the remedy for **exploding** gradients. If the gradient vector `g` is longer than a limit `c`, rescale it to length `c` and keep its direction: `g ← g · c/‖g‖` when `‖g‖ > c`. This is what `torch.nn.utils.clip_grad_norm_` does. First the lesson's example by hand: `g = (30, 40)` has length 50; with `c = 5` it becomes `(3, 4)`. Then a real run. A 10-layer ReLU network with He initialization is trained with a high learning rate (0.5), once without clipping and once with `clip=1.0`.

# %%
def clip_by_norm(g, c):
    """Scale the gradient vector g down so that its length is at most c (the idea of nn.utils.clip_grad_norm_)."""
    g = np.asarray(g, dtype=float)
    n = np.linalg.norm(g)
    return g * (c / n) if n > c else g

g = np.array([30.0, 40.0])
print("g =", g, " norm", np.linalg.norm(g), " clipped to c = 5 ->", clip_by_norm(g, 5.0))
assert np.allclose(clip_by_norm(g, 5.0), [3.0, 4.0]) and np.isclose(np.linalg.norm(clip_by_norm(g, 5.0)), 5.0)
assert np.allclose(clip_by_norm([0.3, 0.4], 5.0), [0.3, 0.4]), "a small gradient is left alone"

# %% [markdown]
# **What you should see.** You should see `g = [30. 40.]  norm 50.0  clipped to c = 5 -> [3. 4.]`, and no assertion error: the clipped vector has length exactly 5, and a small gradient `(0.3, 0.4)` is left alone.

# %%
def exploding_run(clip, lr=0.5, epochs=100):
    torch.manual_seed(SEED)
    m = make_deep(10, 16, "relu", "he")
    return train_model(m, Xtr, ytr, epochs=epochs, lr=lr, clip=clip, seed=SEED)

h_free = exploding_run(None)
h_clip = exploding_run(1.0)
for e in (0, 1, 2, 3, 5, 10, 50, 99):
    print(f"epoch {e:3d}:  no clipping loss {h_free['loss'][e]:12.4g}   clip 1.0 loss {h_clip['loss'][e]:8.4f}")
print(f"accuracy: no clipping {h_free['acc']:.3f}   clip 1.0 {h_clip['acc']:.3f}")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(np.nan_to_num(h_free["loss"], nan=1e6, posinf=1e6), label="no clipping"); ax.plot(h_clip["loss"], label="clip_grad_norm_ = 1.0")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)"); ax.set_title("Same network, same learning rate 0.5"); ax.legend()
plt.tight_layout(); plt.show()
RESULTS["p6_clip_final_loss"] = h_clip["loss"][-1]; RESULTS["p6_free_max_loss"] = float(np.nanmax(h_free["loss"]))

# %% [markdown]
# **What you should see.** Without clipping the loss bounces: `0.603, 0.589, 1.235, 0.639, 0.845, 0.685, ... 0.240` at epoch 50 and `0.693` at epoch 99. Accuracy at the end is **0.510**: the run was thrown into a region where the network predicts one class. With `clip_grad_norm_ = 1.0` the first steps are the same (the gradient was small enough), then the clipped run stays under control: `0.403` at epoch 10, `0.166` at epoch 50, **`0.098`** at epoch 99, accuracy **0.960**. Clipping did not change the learning rate or the direction: it only refused to take a step longer than the limit.

# %% [markdown]
# **Try also (optional):**
#
# - Try `clip=0.1` and `clip=10`. What happens at each end?
# - Plot the gradient norm per epoch (hint: record `float(sum(p.grad.norm() for p in model.parameters()))` in your own loop).
# - Combine clipping with the large-init network of Step 5.4 and a learning rate of 0.01.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.3 - clipping by hand
#
# Implement the check that clipping never changes the **direction**: for `g = (6, 8)` and `c = 2`, compute the clipped vector with `clip_by_norm` and store it in `g_clipped`; store the cosine between `g` and `g_clipped` in `cosine` (it should be 1). Then answer: if the gradient is `(0.3, 0.4)` and `c = 5`, what does clipping do (store `"nothing"` or `"scales down"` in `small_case`)? Why it matters: clipping is a safety belt for rare spikes, not a change of the method.

# %%
g_clipped = None     # YOUR CODE HERE
cosine = None        # YOUR CODE HERE: dot(g, g_clipped) / (norm(g) * norm(g_clipped))
small_case = ""      # YOUR ANSWER: "nothing" or "scales down"

# %%
# Check
if g_clipped is None or cosine is None:
    print("Fill in `g_clipped` and `cosine` above, then run this cell again.")
else:
    check("g_clipped is (1.2, 1.6)", np.allclose(g_clipped, [1.2, 1.6]), "6 * 2/10 and 8 * 2/10")
    check("the cosine is 1 (same direction)", abs(cosine - 1) < 1e-9, "dot product over the product of the lengths")
    check("a small gradient is left alone", small_case == "nothing", "clip_by_norm only acts when the norm is above c")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# g_clipped = clip_by_norm([6, 8], 2.0)
# gg = np.array([6.0, 8.0])
# cosine = float(gg @ g_clipped / (np.linalg.norm(gg) * np.linalg.norm(g_clipped)))
# small_case = "nothing"
# ```
# The norm of (6, 8) is 10, so the vector is multiplied by 2/10 = 0.2: same direction, length 2.
# </details>

# %% [markdown]
# ### 6.4 Initialization by hand
#
# The lesson gives two recipes for the standard deviation of the starting weights of a layer with `n_in` inputs: **Xavier** (Glorot) `√(1/n_in)`, designed for tanh and sigmoid, and **He** `√(2/n_in)`, designed for ReLU, which loses half the signal by cutting negative values. For `n_in = 256` that is `0.0625` and `0.0884`. The first block computes them, samples 65,536 weights from each distribution, and verifies the empirical standard deviation is within 5 percent. The second pushes 500 random inputs through 10 ReLU layers of width 256 with three choices of standard deviation (0.01, He, 0.2) and prints the standard deviation of the activations after each layer, so that you *see* the signal vanish and explode in the forward direction (the backward direction behaves the same way).

# %%
n_in = 256
xavier_std, he_std = math.sqrt(1 / n_in), math.sqrt(2 / n_in)
print(f"Xavier std = sqrt(1/{n_in}) = {xavier_std:.4f}     He std = sqrt(2/{n_in}) = {he_std:.4f}")
rng = np.random.default_rng(0)
Wx = rng.normal(0, xavier_std, size=(256, n_in)); Wh = rng.normal(0, he_std, size=(256, n_in))
print("empirical std of 65,536 sampled weights:  Xavier", round(Wx.std(), 4), "  He", round(Wh.std(), 4))
assert abs(Wx.std() / xavier_std - 1) < 0.05 and abs(Wh.std() / he_std - 1) < 0.05, "within 5 %"
RESULTS["p6_xavier_std"] = round(xavier_std, 4); RESULTS["p6_he_std"] = round(he_std, 4)

# %% [markdown]
# **What you should see.** You should see `Xavier std = sqrt(1/256) = 0.0625     He std = sqrt(2/256) = 0.0884` and `empirical std ...: Xavier 0.0625   He 0.0887`: the samples agree with the formulas well within 5 percent (their ratio to the formula is 1.00 and 1.003).

# %%
def activation_scale(std, layers=10, width=256, seed=0):
    """Push 500 random inputs through `layers` ReLU layers whose weights have the given std; return the std of the activations after each layer."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(500, width)); out = []
    for _ in range(layers):
        h = np.maximum(0.0, h @ rng.normal(0, std, size=(width, width)))
        out.append(float(h.std()))
    return out

curves = {"std 0.01 (too small)": activation_scale(0.01), f"He, std {he_std:.4f}": activation_scale(he_std), "std 0.2 (too big)": activation_scale(0.2)}
print("layer | " + " | ".join(f"{k:>22s}" for k in curves))
for i in range(10):
    print(f"{i + 1:5d} | " + " | ".join(f"{v[i]:22.3e}" for v in curves.values()))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for k, v in curves.items():
    ax.plot(range(1, 11), v, "o-", label=k)
ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_ylabel("std of the activations"); ax.set_title("Signal size through 10 ReLU layers"); ax.legend()
plt.tight_layout(); plt.show()
assert curves["std 0.01 (too small)"][-1] < 1e-6 and curves["std 0.2 (too big)"][-1] > 100, "off-scale weights should change the signal a lot"

# %% [markdown]
# **What you should see.** With std **0.01** the activation size is `0.093` after layer 1 and `3.4e-10` after layer 10: the signal has vanished. With **He** it stays between `0.83` and `1.03` through all ten layers: stable. With **0.2** it grows from `1.9` to **`3532`**: it has exploded. Only a factor of 2.3 in the weights' standard deviation separates the stable network from the exploding one (0.0884 versus 0.2), which is why the initialization has a formula instead of a guess. The `assert` checks both extremes.

# %% [markdown]
# **Try also (optional):**
#
# - Try std 0.0884 on a network of 50 layers and 3 times 0.0884. How fragile is the stable case?
# - Use `np.tanh` instead of the ReLU, with the Xavier std.
# - Change the width from 256 to 64 and recompute He: `√(2/64)`.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.4 - initialization numbers and tanh
#
# (a) Compute the He standard deviation for `n_in = 512` and the Xavier one, and store them in `he_512` and `xavier_512`. (b) Predict, then measure, what happens to the activation scale in a **tanh** network (copy `activation_scale`, replace the ReLU `np.maximum(0.0, ...)` by `np.tanh(...)`, and compare the standard deviation after layer 10 for the He and the Xavier standard deviation). Does the signal vanish or explode as it did for ReLU with std 0.01 and 0.2? Store your short answer in `tanh_note`. Why it matters: the right initialization depends on the activation function, which is why `make_deep` has both `"he"` and `"xavier"`.

# %%
he_512 = None        # YOUR CODE HERE
xavier_512 = None    # YOUR CODE HERE
tanh_note = ""       # YOUR ANSWER: what happens to the signal size with tanh, for He and for Xavier?

# %%
# Check
if he_512 is None or xavier_512 is None:
    print("Fill in `he_512` and `xavier_512` above, then run this cell again.")
else:
    check("he_512 is 0.0625", abs(he_512 - 0.0625) < 1e-4, "sqrt(2/512)")
    check("xavier_512 is 0.0442", abs(xavier_512 - 0.0442) < 1e-4, "sqrt(1/512)")
    check("your note has at least 10 words", n_words(tanh_note) >= 10, "describe what you measured")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# he_512, xavier_512 = math.sqrt(2 / 512), math.sqrt(1 / 512)
# def _tanh_scale(std, layers=10, width=256):
#     rng = np.random.default_rng(0); h = rng.normal(size=(500, width)); out = []
#     for _ in range(layers):
#         h = np.tanh(h @ rng.normal(0, std, size=(width, width))); out.append(float(h.std()))
#     return out
# print("tanh with He std:", np.round(_tanh_scale(he_std), 3), " with Xavier std:", np.round(_tanh_scale(xavier_std), 3))
# tanh_note = "With tanh neither choice explodes or vanishes: after 10 layers the activations have std about 0.56 with He and 0.23 with Xavier, a factor of about 2 apart, not the factor of 10 to the power 10 we saw for ReLU, because tanh squashes large values."
# ```
# He gives 0.0625 and Xavier 0.0442 for 512 inputs. For tanh the squashing keeps the signal in a narrow band for both choices (about 0.56 and 0.23 after 10 layers); ReLU has no such protection, which is why its initialization matters so much.
# </details>

# %% [markdown]
# ### 6.5 A debugging checklist
#
# The most useful thing you can take from this part is a page you will actually reread when a deep network does not learn. Write it as a dictionary with **five symptoms** and the **action** you would take for each. Each action should say what you would *measure or change* and why, in one sentence. The Check only counts entries and words, but a good checklist is concrete: it names the number to look at and the setting to change.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.5 - your checklist
#
# Write your debugging checklist for "my deep network does not learn": a dictionary with **five symptoms** as keys and the action you would take as values. Use at least these symptoms: loss is `nan`, loss stays at ln 2 = 0.693, the first-layer gradient is a millionth of the last, training accuracy is high but validation is low, the loss falls and then jumps up. Why it matters: it is the page you will actually keep.

# %%
checklist = {
    # "symptom": "what I would do",
}
for s, a in checklist.items():
    print(f"- {s}  ->  {a}")

# %%
# Check
if len(checklist) < 5:
    print(f"You have {len(checklist)} entries; add {5 - len(checklist)} more, then run this cell again.")
else:
    check("five or more symptoms", len(checklist) >= 5, "")
    check("every action has at least 6 words", all(n_words(a) >= 6 for a in checklist.values()), "say what you would change and why")
    RESULTS["p6_checklist"] = checklist

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# checklist = {
#     "loss is nan": "Lower the learning rate by ten, add gradient clipping at 1.0 and check the input data for nan or huge values.",
#     "loss stays at ln 2 = 0.693": "Suspect a dead or vanishing network: measure the per-layer gradient norms and switch to ReLU with He initialization.",
#     "first-layer gradient is a millionth of the last": "Reduce the depth, use ReLU and He initialization, or add residual connections or LayerNorm.",
#     "training accuracy high, validation low": "This is overfitting rather than a gradient problem: use more data, a smaller model, weight decay or early stopping.",
#     "loss falls then jumps up": "The step is too large at some point: lower the learning rate, use a decay schedule or clip the gradient norm.",
# }
# ```
# Your five entries should each map one visible symptom to one measurable cause and one action.
# </details>

# %% [markdown]
# **Before you move on (Part 6).** Check that you can say each of these in your own words: the remedies work by keeping the per-layer factor near 1 (ReLU/He and tanh/Xavier by initialization and slope, residual by the identity path, normalization by rescaling); the gradient ratio first / last is a diagnosis and the validation accuracy is the outcome; clipping rescales a too-long gradient without changing its direction; He is `√(2/n_in)` and Xavier is `√(1/n_in)`.

# %% [markdown]
# ### Imagine first, revisited
#
# Run the next cell to compare your Imagine-first prediction with the facts.

# %%
measured = sorted(table, key=lambda n: -table[n]["val"])
print("Your ranking :", rank_guess)
print("Measured, best first (validation accuracy):")
for n in measured:
    print(f"   {table[n]['val']:.3f}  {n}")
RESULTS["p6_rank_guess"] = rank_guess

# %% [markdown]
# In a markdown cell write which remedy surprised you most, and one thing the measurements cannot tell you (hint: three seeds, one data set, one learning rate).

# %% [markdown]
# ## Part 7. Design and reflect (25 min)
#
# *Lesson sections: all of Week 3.* Time to use what you know. You will describe a deep network for a problem you care about, build and train a small deep network on a harder synthetic problem (two intertwined spirals) while recording the gradient norm of **every layer** during training, deliberately **break** a network and diagnose it using only gradient numbers, and then answer four reflection questions. ==A deep network is designed by choosing, for every layer, what it should learn and how its signal and its gradient will stay a usable size.==

# %% [markdown]
# ### ✏️ Imagine first
#
# Think of a problem where a **deep** network would clearly beat a shallow one: reading handwriting, recognizing animals in photos, understanding a spoken command, translating a sentence. Write the problem in one sentence in `my_problem`. You turn it into a design in Step 7.1.

# %%
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

my_problem = ""       # YOUR TURN: one sentence - the problem you want a deep network to solve (not toy XOR)
print(my_problem if my_problem else "(write the problem in one sentence and run this cell again)")

# %% [markdown]
# ### 7.1 Imagination design: a spec card
#
# Fill in a **spec** for a deep network that solves your problem. For each field give a concrete answer, not a slogan: the input and output; how many layers and how wide; what you expect the early, middle and late layers to learn (for an image network: strokes and edges, then shapes and parts, then whole objects); the activation, the initialization, whether you use normalization or residual connections, the loss; **how you will monitor the gradient norms** while it trains; and the biggest risk. The block prints it as a card, and the next block checks that nothing is empty.

# %% [markdown]
# ### ✏️ Your turn: fill in your spec
#
# Edit the dictionary below, then run the cell and the one after it.

# %%
spec = {
    "problem": "",                       # e.g. "Recognise handwritten digits on a school form"
    "input": "",                         # what goes in (size, type)
    "output_and_loss": "",               # what comes out, and which loss
    "depth_and_width": "",               # how many layers, how wide
    "what_layers_learn": ["", "", ""],   # three levels: low, middle, high
    "activation": "",
    "initialization": "",
    "normalization_or_residual": "",
    "how_I_monitor_gradients": "",       # which numbers, how often
    "biggest_risk": "",
}
filled = {k: (v if isinstance(v, str) else " | ".join(v)) for k, v in spec.items()}
print("+" + "-" * 70 + "+")
for k, v in filled.items():
    print(f"| {k:26s}: {v if v else '(empty)'}")
print("+" + "-" * 70 + "+")

# %%
empty = [k for k, v in filled.items() if not v.replace("|", "").strip()]
if len(empty) == len(filled):
    print("Fill in the spec above, then run this cell again.")
else:
    check("every field is filled in", not empty, "empty fields: " + ", ".join(empty))
    check("the three feature levels are all present", all(s.strip() for s in spec["what_layers_learn"]), "write low, middle and high level features")
    RESULTS["p7_spec"] = filled

# %% [markdown]
# **What you should see.** When you have filled in every field you should see ✓ twice. An example of a good card for reading digits: depth 6, width 128, ReLU with He initialization, LayerNorm between layers, cross-entropy loss, `what_layers_learn = ["short strokes and edges", "loops, crossings and corners", "whole digits"]`, `how_I_monitor_gradients = "print the gradient norm of every layer every 50 epochs and stop if first/last is below 1e-4"`, and the biggest risk "overfitting a small training set". Notice how the monitoring field uses the tools of Part 5.

# %% [markdown]
# **Try also (optional):**
#
# - Write a second spec for the same problem with a much shallower network and list one thing that would get worse.
# - Add a field `parameters` with the count of parameters of your design (you can use the formula from Week 2).
# - Ask a friend to find the weakest field of your card.

# %% [markdown]
# ### 7.2 Build a deep network and watch its gradients
#
# The synthetic problem is **two intertwined spirals** (`make_spirals`, 300 points, noise 0.25), which a shallow network learns badly but a deep one handles. We split 220 points for training and 80 for validation. The design: `make_deep(depth=8, width=16, act="relu", init="he", norm=True)` (ReLU, He initialization, LayerNorm) trained with Adam for 1500 epochs. `train_record` stores the gradient norm of every layer every 50 epochs; we show it as a **heat map** (colour is the log10 of the norm, rows are layers, columns are time) next to the loss curve and the decision regions.

# %%
def make_spirals(n=300, noise=0.25, seed=0):
    """Two intertwined spirals (labels 0 and 1): a problem that needs several layers."""
    r = np.random.RandomState(seed)
    m = n // 2
    t = np.sqrt(r.uniform(0.04, 1, m)) * 2.4 * math.pi / 1.5
    a = np.stack([t * np.cos(t), t * np.sin(t)], 1) / 4.0
    X = np.concatenate([a, -a]) + r.normal(0, noise, size=(2 * m, 2))
    y = np.concatenate([np.zeros(m), np.ones(m)])
    return X.astype(np.float32), y.astype(np.float32)

Xs, ys = make_spirals(300, seed=0)
perm = np.random.RandomState(1).permutation(300)
jtr, jva = perm[:220], perm[220:]
Xs_tr, ys_tr, Xs_va, ys_va = Xs[jtr], ys[jtr], Xs[jva], ys[jva]

def train_record(model, X, y, Xv, yv, epochs=1500, lr=0.01, every=50):
    """Adam, full batch; records the per-layer gradient norms every `every` epochs."""
    Xt = torch.as_tensor(X); yt = torch.as_tensor(y).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rec, losses = [], []
    for e in range(epochs):
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(Xt), yt); loss.backward()
        if e % every == 0:
            rec.append([float(l.weight.grad.norm()) for l in linear_layers(model)])
        opt.step(); losses.append(float(loss))
    with torch.no_grad():
        acc = float(((model(Xt) > 0).float() == yt).float().mean())
        vacc = float(((model(torch.as_tensor(Xv)) > 0).float() == torch.as_tensor(yv).reshape(-1, 1)).float().mean())
    return np.array(rec), losses, acc, vacc

print("spiral data:", Xs.shape, "balance", ys.mean())

# %%
torch.manual_seed(SEED)
good = make_deep(depth=8, width=16, act="relu", init="he", norm=True)
epochs72 = 600 if QUICK else 1500
rec_good, loss_good, acc_good, vacc_good = train_record(good, Xs_tr, ys_tr, Xs_va, ys_va, epochs=epochs72)
print(f"depth 8, width 16, ReLU + He + LayerNorm, Adam: loss {loss_good[0]:.3f} -> {loss_good[-1]:.4f}   train {acc_good:.3f}   validation {vacc_good:.3f}")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
im = ax[0].imshow(np.log10(rec_good.T + 1e-12), aspect="auto", origin="lower", cmap="viridis")
ax[0].set_xlabel("record number (every 50 epochs)"); ax[0].set_ylabel("layer (0 = next to the input)"); ax[0].set_title("log10 gradient norm per layer")
plt.colorbar(im, ax=ax[0])
ax[1].plot(loss_good); ax[1].set_title("Training loss"); ax[1].set_xlabel("epoch"); ax[1].set_yscale("log")
plot_decision(lambda P: torch.sigmoid(good(torch.as_tensor(P))).detach().numpy()[:, 0], Xs, ys, ax=ax[2], title=f"validation accuracy {vacc_good:.2f}", lim=2.2)
plt.tight_layout(); plt.show()
RESULTS["p7_train_acc"] = round(acc_good, 3); RESULTS["p7_val_acc"] = round(vacc_good, 3)
check("the design learns the spirals (validation accuracy above 0.85)", vacc_good > 0.85, "try more epochs or a wider network")

# %% [markdown]
# **What you should see.** You should see `spiral data: (300, 2) balance 0.5` and then: `depth 8, width 16, ReLU + He + LayerNorm, Adam: loss 1.035 -> 0.0736   train 0.977   validation 0.925` (training takes about five seconds). In the heat map there are nine rows (eight hidden layers and the output). At the start all nine gradient norms lie between `0.5` and `1.9`, within a factor of four of each other; as the loss falls all the colours darken together, and no row separates from the others. No layer starves, because the design keeps the factor near 1. The loss falls steadily on a log scale and the decision picture shows a double spiral boundary. The validation accuracy (0.925 on 80 points: each point is worth 1.25 percentage points) is a little below the training accuracy (0.977), as expected with 220 training points.

# %% [markdown]
# **Try also (optional):**
#
# - Raise the noise to 0.4 and see what happens to the gap between training and validation accuracy.
# - Plot the first and last layer's gradient norm as two lines instead of the heat map.
# - Train without `norm=True` and compare the heat maps.

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.2 - change one thing
#
# Change one design choice in the cell above (width 32, no `norm`, or `act="tanh"` with `init="xavier"`), run it again, and store the new validation accuracy in `vacc_changed` and one sentence about what you saw in `change_note`. Why it matters: a design is only understood after you have moved one part of it.

# %%
torch.manual_seed(SEED)
changed = make_deep(depth=8, width=16, act="relu", init="he", norm=True)      # YOUR CODE HERE: change one thing
rec_ch, loss_ch, acc_ch, vacc_changed = train_record(changed, Xs_tr, ys_tr, Xs_va, ys_va, epochs=epochs72)
print("validation accuracy of your variant:", round(vacc_changed, 3))
change_note = ""     # YOUR ANSWER

# %%
# Check
if not change_note:
    print("Write your one-sentence `change_note` above, then run this cell again.")
else:
    check("your note has at least 10 words", n_words(change_note) >= 10, "say what you changed and what happened")
    check("your variant is a real number", 0 <= vacc_changed <= 1, "")
    RESULTS["p7_change_note"] = change_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# torch.manual_seed(SEED)
# changed = make_deep(depth=8, width=32, act="relu", init="he", norm=True)
# rec_ch, loss_ch, acc_ch, vacc_changed = train_record(changed, Xs_tr, ys_tr, Xs_va, ys_va, epochs=epochs72)
# change_note = "I made the network wider (32 units instead of 16) and the validation accuracy went from 0.925 to 0.850, so more width did not help here; with only 80 validation points one seed is a noisy comparison."
# ```
# Any single change with an honest comparison to the baseline numbers is a good answer.
# </details>

# %% [markdown]
# ### 7.3 Break it, diagnose it, fix it
#
# Every engineer needs to read a failing run. The first block builds the **broken** network: 16 sigmoid layers with the default initialization on the same spirals. The cell prints the gradient norm of every layer at epoch 0, and you diagnose *from these numbers only*. The second block repairs the network with **one change**, residual connections (`residual=True`), keeping the data, loss, optimizer and the other settings.

# %%
# BREAK: a depth-16 sigmoid network with the default initialisation, on the same spirals
torch.manual_seed(SEED)
broken = make_deep(depth=16, width=16, act="sigmoid", init="default")
rec_bad, loss_bad, acc_bad, vacc_bad = train_record(broken, Xs_tr, ys_tr, Xs_va, ys_va, epochs=epochs72)
n0 = rec_bad[0]
print(f"BROKEN: loss {loss_bad[0]:.4f} -> {loss_bad[-1]:.4f}   train {acc_bad:.3f}   validation {vacc_bad:.3f}")
print("gradient norm per layer at epoch 0, input side first:")
print(np.array2string(n0, precision=2, formatter={"float_kind": lambda v: f"{v:.1e}"}))
print(f"first / last = {n0[0] / n0[-1]:.2e}")

# %% [markdown]
# **What you should see.** The broken run: `loss 0.7108 -> 0.6928   train 0.514   validation 0.463`: chance level, loss at `ln 2`. The numbers say why: the gradient norms rise with every layer from `6.5e-15` at the input side (layer 1) through `1.1e-06` (layer 11) to `0.2` at the output, and `first / last = 3.27e-14`. That is a textbook **vanishing gradient**: the first layers receive a gradient about thirty trillion times smaller than the last, so they never learn, and the whole network behaves like a single-layer one stuck at the base rate.

# %%
# DIAGNOSE with the numbers only, then REPAIR with ONE change: residual connections (the loss and the data stay the same)
diagnosis = "first / last gradient is tiny, so the early layers barely learn: a vanishing gradient"      # your sentence from the numbers
torch.manual_seed(SEED)
fixed = make_deep(depth=16, width=16, act="sigmoid", init="default", residual=True)
rec_fix, loss_fix, acc_fix, vacc_fix = train_record(fixed, Xs_tr, ys_tr, Xs_va, ys_va, epochs=epochs72)
n1 = rec_fix[0]
print(f"FIXED : loss {loss_fix[0]:.4f} -> {loss_fix[-1]:.4f}   train {acc_fix:.3f}   validation {vacc_fix:.3f}   first / last at epoch 0 = {n1[0] / n1[-1]:.2e}")
print(f"before / after validation accuracy: {vacc_bad:.3f} -> {vacc_fix:.3f}")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(loss_bad, label="broken"); ax.plot(loss_fix, label="residual fix"); ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title("Break and fix"); ax.legend()
plt.tight_layout(); plt.show()
RESULTS["p7_broken_val"] = round(vacc_bad, 3); RESULTS["p7_fixed_val"] = round(vacc_fix, 3)
RESULTS["p7_diagnosis"] = diagnosis
check("the repair raised the validation accuracy", vacc_fix > vacc_bad + 0.05, "if not, try ReLU with He initialization as the one change")

# %% [markdown]
# **What you should see.** After the repair: `loss 2.6114 -> 0.1319   train 0.941   validation 0.938   first / last at epoch 0 = 2.60e-03`, and `before / after validation accuracy: 0.463 -> 0.938`. One change moved the ratio from 3e-14 to 3e-3 (eleven powers of ten) and the validation accuracy from chance to 0.94. The loss at the start is high (2.6), and Adam brings it down. The ✓ line appears.

# %% [markdown]
# **Try also (optional):**
#
# - Try ReLU with He initialization as the one change instead and compare.
# - Break it a different way: ReLU with `init="large"` at depth 16. What do the gradient norms look like?
# - Repair the broken network with Adam and `lr=0.001` only, without changing the architecture. Does the optimizer alone fix it?

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.3 - your own break and fix
#
# Do a second break-and-fix of your own: choose another way to break the network (the starter uses a ReLU network with a far too small initialization), show the failure with numbers, repair it with **one** change, and write `break_fix_note` (three sentences: what broke, what the numbers said, what fixed it). Why it matters: you now know how to read a failing run.

# %%
torch.manual_seed(SEED)
broken2 = make_deep(depth=16, width=16, act="relu", init="small")             # YOUR CODE HERE: your own way to break it (init="small" gives std 0.05)
rec_b2, loss_b2, acc_b2, vacc_b2 = train_record(broken2, Xs_tr, ys_tr, Xs_va, ys_va, epochs=300)
print("broken variant: validation accuracy", round(vacc_b2, 3), "  loss at the end", round(loss_b2[-1], 4))
print("first-layer and last-layer gradient norms at epoch 0:", f"{rec_b2[0][0]:.1e}", f"{rec_b2[0][-1]:.1e}", "  (look at the SIZES, not only the ratio)")
break_fix_note = ""   # YOUR ANSWER: three sentences

# %%
# Check
if not break_fix_note:
    print("Write `break_fix_note` above, then run this cell again.")
else:
    check("your note has at least 25 words", n_words(break_fix_note) >= 25, "what broke, what the numbers said, what fixed it")
    RESULTS["p7_break_fix_note"] = break_fix_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# torch.manual_seed(SEED)
# fixed2 = make_deep(depth=16, width=16, act="relu", init="he")
# rec_f2, loss_f2, acc_f2, vacc_f2 = train_record(fixed2, Xs_tr, ys_tr, Xs_va, ys_va, epochs=300)
# print("fixed variant: validation accuracy", round(vacc_f2, 3))
# break_fix_note = ("Initialising a 16-layer ReLU network with standard deviation 0.05 shrank the signal at every layer, so the loss stayed at 0.693 and the validation accuracy at chance. "
#                   "The numbers showed gradient norms tiny in the first AND the last layer, so a ratio near 1 did not mean healthy: the whole signal had vanished. "
#                   "Switching the initialization to He restored the scale of the signal and the network learned the spirals.")
# ```
# Your own failure and repair are the answer; what counts is that the numbers justify the diagnosis.
# </details>

# %% [markdown]
# ### 7.4 The RESULTS summary
#
# Every important number you stored in `RESULTS` during the lab is printed here. This is your checklist for the deliverables: if a number you expect is missing, run the matching step again.

# %%
print(f"{len(RESULTS)} results saved this lab:\n")
for k, v in RESULTS.items():
    s = str(v)
    print(f"  {k:28s} {s if len(s) < 80 else s[:77] + '...'}")

# %% [markdown]
# **What you should see.** You should see one line for every result stored in Parts 4 to 7 (with the earlier parts of the lab, many more). Long values are shortened with '...'. Among them: `p4_example_dW1_01 -0.016314`, `p4_numpy_acc 0.987`, `p5_ratio_sigmoid` about `6e-09`, `p5_sigmoid_acc 0.5`, `p5_relu_acc 0.943`, `p6_he_std 0.0884`, `p7_broken_val 0.463` and `p7_fixed_val 0.938`.

# %% [markdown]
# ### 7.5 Four reflection questions
#
# Write the four answers in your own words, at least 25 words each. Quote at least two numbers from the lab (for example the 460-fold gap in Step 4.5, the ratios in Step 5.2, the table of Step 6.2). Questions: (1) Why does depth help? (2) Why does the first layer get the weakest signal in a sigmoid network? (3) What does `loss.backward()` actually do? (4) Which remedy would you reach for first, and why?

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.5 - reflection
#
# Answer the four reflection questions in your own words (at least 25 words each), citing a number from this lab in at least two of them: (1) Why does depth help? (2) Why does the first layer get the weakest signal in a sigmoid network? (3) What does `loss.backward()` actually do? (4) Which remedy would you reach for first, and why? Why it matters: the answers are the best test of whether the lab changed how you think about deep networks.

# %%
reflection = {
    "1_why_depth_helps": "",
    "2_weakest_first_layer": "",
    "3_what_backward_does": "",
    "4_first_remedy": "",
}

# %%
# Check
short = [k for k, v in reflection.items() if n_words(v) < 25]
if len(short) == len(reflection):
    print("Write your four answers above (at least 25 words each), then run this cell again.")
else:
    check("all four answers have at least 25 words", not short, "still short: " + ", ".join(short))
RESULTS["p7_reflection"] = reflection

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reflection = {
#     "1_why_depth_helps": "Each layer re-describes the data in terms of the layer before, so depth builds features of features. The folding example gave 2^k pieces from only 2k units, which a single wide layer could not match.",
#     "2_weakest_first_layer": "Every step backward multiplies the blame by a slope of at most 0.25 for sigmoid, so after many layers it is tiny: in my depth-10 run the first layer received a very small fraction of the last layer's gradient.",
#     "3_what_backward_does": "loss.backward() walks the stored computation graph from the loss to every parameter, applying the chain rule at each node, and leaves the result in each parameter's .grad, which the optimizer then uses for its step.",
#     "4_first_remedy": "I would first switch to ReLU with He initialization, because it fixed the gradient ratio in my table with a single change and costs nothing, and only then add normalization or residual connections if the depth is large.",
# }
# ```
# Your answers should cite your own numbers; there is no single right answer.
# </details>

# %% [markdown]
# **Before you finish.** Check that you can say each of these in your own words: depth composes features so that a few units multiply their effect; a deep sigmoid network gives the first layer a gradient that is a product of small slopes; `loss.backward()` applies the chain rule through the stored graph and writes each parameter's gradient into `.grad`; and you diagnose a network by measuring per-layer gradient norms before touching anything else.

# %% [markdown]
# ### Imagine first, revisited
#
# Run the next cell to compare your Imagine-first prediction with the facts.

# %%
print("Your problem:", my_problem if my_problem else "(not written)")
print("Your design:", RESULTS.get("p7_spec", {}).get("depth_and_width", "(spec not filled in)"))
RESULTS["p7_problem"] = my_problem

# %% [markdown]
# In a markdown cell write one sentence: did the spiral experiment of Step 7.2 change what you wrote about depth and width in your design?

# %% [markdown]
# ## Wrap-up
#
# What you learned this week:
#
# - **Backpropagation is four equations.** The blame at the output, the blame passed back through the weights and scaled by the slope, `dW = delta aᵀ` and `db = delta`. You wrote them in NumPy, checked them against finite differences and PyTorch's autograd, and trained a network with them. `loss.backward()` is the same computation, done for any network.
# - **Depth multiplies one factor per layer.** With sigmoid hidden units the factor is at most 0.25, so the gradient of the first layers vanishes (first / last = 6e-09 at depth 10, 3e-17 at depth 20); with weights that are too large it explodes, and a dead ReLU passes no gradient at all. You measured all three.
# - **Remedies keep the factor near 1.** ReLU with He initialization, tanh with Xavier, residual connections, and clipping for exploding gradients each turned a network that learns nothing into one that learns, and the per-layer gradient norm is the diagnosis to reach for first.
#
# **Looking ahead.** In Week 4 you meet convolutional neural networks: the same layers, gradients and remedies, with weights shared across an image so that depth can be put to work on pictures. The gradient-norm habit from this lab will tell you whether a deeper network is training or merely deep.
#
# **Before you finish.** Files in `/content` vanish when the Colab session ends. Download anything you want to keep, and use **File → Save a copy in Drive** (or download the notebook with its outputs) so your answers survive. Submit it the way your instructor asks. The lesson for Week 3 is your reference for the ideas behind each step.
#
# **Reflect:** double-click this cell and write two sentences: which result of this lab surprised you most, and which remedy will you check first the next time a deep network does not learn?
