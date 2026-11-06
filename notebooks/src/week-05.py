# %% [markdown]
# # Week 5 — Prelim Exam, Part B: practical exam
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept. (On the school's DGX Spark you can instead upload this file to JupyterLab.)
#
# This is Part B of the Prelim exam: **60 of the 100 points, 120 minutes**. Part A (the written exam, 40 points) is taken in the course app. The notebook has five tasks:
#
# | Task | Topic | Points | Minutes |
# |---|---|---|---|
# | 1 | Generative AI and language models | 10 | 20 |
# | 2 | Supervised learning, neurons and generalisation | 12 | 25 |
# | 3 | Backpropagation and deep networks | 16 | 30 |
# | 4 | Convolutional networks | 16 | 30 |
# | 5 | Synthesis: short written answers | 6 | 15 |
#
# Your instructor announces the date, the room and how the exam counts towards the course.
#
# ### Rules
#
# - Open book for the course only: you may use the lessons and your own lab notes.
# - No AI assistants, no sharing of code or answers, no internet searches for answers.
# - Run the cells from top to bottom. Replace every `# YOUR CODE HERE` and every `None` with your own work. The starter cells run without errors while they are unfinished.
# - Everything runs on a CPU, nothing is downloaded, and every cell should finish in under 90 seconds.
# - Keep an eye on the clock. If a task is taking too long, write what you have, and move on.
#
# ### How the self-checks work
#
# A **Self-check** cell prints ✓ or ✗ for each numerical or shape answer. It never prints the right value or a hint, and a ✓ is not a promise of full marks: your instructor marks your code, your numbers and your written explanations. Written answers go in the `ANSWERS` dictionary (one string per question) or in a markdown cell that begins "Your answer:".
#
# ### What to submit
#
# Submit this notebook **with all outputs visible** (run everything from the top just before you submit), together with the output of the last cell, which prints your `ANSWERS`. Submit it the way your instructor asks.
#
# <details><summary>How you are marked (the marking rules)</summary>
#
# - Each sub-question B*n*.*m* shows its points. A numerical answer earns its points when the value is right (small rounding differences are fine). If the value is wrong but your code or working shows the correct method, you can still earn part of the points.
# - Code questions are marked on what the code does, not on how it looks. A hard-coded number with no working earns at most half of the points.
# - Written answers are marked on whether they are correct, use the right terms from the lessons and refer to your own numbers. Short answers are fine: say what is true and why.
# - A ✓ in a self-check does not replace the marking, and a ✗ does not remove method points.
# - No points are removed for a wrong answer; leaving a question empty earns 0 for it.
# </details>

# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed and defines the helper functions you used in the labs of Weeks 2 to 4: the datasets `make_xor`, `make_moons`, `make_circles` and `make_shapes`; `plot_decision`; `make_deep`, `linear_layers`, `layer_grad_norms` and `train_model` (Week 3); `conv2d_np`, `count_params` and `train_classifier` (Week 4). In this exam `check(name, ok)` only tells you whether a result is right (✓ or ✗); it gives **no hints**. Everything runs on a CPU and nothing is downloaded.

# %%
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
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone
print("Running on:", device)

ANSWERS = {}   # keep your final answers here; the last cell prints them so you can copy them into your submission

def check(name, ok, hint=""):
    """Exam version: prints a tick or a cross and never a hint."""
    print(("✓ " if ok else "✗ ") + name)

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

def show_images(X, y=None, n=10, cols=5, titles=None):
    """Show the first n pictures of X (shape (N, 1, H, W) or (N, H, W))."""
    X = np.asarray(X); n = min(n, len(X)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.7 * rows))
    for a in np.ravel(axes): a.axis("off")
    for i, a in enumerate(np.ravel(axes)[:n]):
        a.imshow(X[i].reshape(X[i].shape[-2:]), cmap="gray_r", vmin=0, vmax=1)
        if titles is not None: a.set_title(str(titles[i]), fontsize=8)
        elif y is not None: a.set_title(SHAPE_NAMES[int(y[i])], fontsize=8)
    plt.tight_layout(); plt.show()

def conv2d_np(x, k, stride=1, pad=0):
    """Convolution as deep learning means it (cross-correlation: the kernel is not flipped) on one 2-D array, in plain NumPy loops."""
    x = np.pad(np.asarray(x, dtype=float), pad); k = np.asarray(k, dtype=float)
    kh, kw = k.shape
    oh, ow = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(x[i * stride:i * stride + kh, j * stride:j * stride + kw] * k)
    return out

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
# ### Exam helpers
#
# One more small cell: it defines `same(...)`, which the self-checks use to compare your number with the expected one without ever showing it, and `n_words(...)`, which counts the words of a written answer.

# %%
import hashlib
def _fp(v, nd):
    return hashlib.sha256(f"w5|{nd}|{round(float(v), nd):.{nd}f}".encode()).hexdigest()[:12]
def same(x, fingerprint, nd=3):
    """True if x, rounded to nd decimals, is the number the exam expects. The notebook stores only a fingerprint, so it cannot tell you the number."""
    try:
        x = float(x)
    except Exception:
        return False
    if not math.isfinite(x):
        return False
    step = 10.0 ** -nd
    return any(_fp(x + d * 0.4 * step, nd) == fingerprint for d in (-1, 0, 1))
def same_word(s, fingerprint):
    return isinstance(s, str) and hashlib.sha256(("w5|" + s.strip().lower()).encode()).hexdigest()[:12] == fingerprint
def n_words(x):
    """How many words are in x (0 if x is not text)."""
    return len(x.split()) if isinstance(x, str) else 0
assert same(0.5004, _fp(0.5, 3)) and not same(0.52, _fp(0.5, 3))   # the helper itself works
print("Exam helpers ready.")

# %% [markdown]
# ## Task 1. Generative AI and language models (10 points, 20 minutes)
#
# A small language model is choosing the next word of "The weather in Manila today is ...". Its raw scores (logits) for four candidate words are below. All numbers in this task are invented for the exam.
#
# | word | hot | humid | rainy | cold |
# |---|---|---|---|---|
# | logit | 3.1 | 2.2 | 0.4 | -0.7 |
#
# **B1.1 Softmax with temperature (2 points).** Write `softmax_t(logits, T)`, which divides the logits by T and turns them into probabilities. Store the probability of "hot" at T = 1 in `p_hot_T1` (1 point) and at T = 0.5 in `p_hot_T05` (1 point). Then write in `ANSWERS["B1.1"]` one sentence on what the lower temperature did to the distribution.

# %%
words = ["hot", "humid", "rainy", "cold"]
logits = np.array([3.1, 2.2, 0.4, -0.7])

def softmax_t(logits, T=1.0):
    """Probabilities from logits at temperature T."""
    return None   # YOUR CODE HERE

probs = softmax_t(logits, 1.0)
p_hot_T1 = None    # YOUR CODE HERE
p_hot_T05 = None   # YOUR CODE HERE
ANSWERS["B1.1"] = ""   # YOUR ANSWER: one sentence

# %%
# Self-check B1.1
check("B1.1 p_hot_T1 (1 point)", same(p_hot_T1, "342f4d1f1509", 3))
check("B1.1 p_hot_T05 (1 point)", same(p_hot_T05, "882f6bd4fad4", 3))
check("B1.1 written sentence present", n_words(ANSWERS.get("B1.1")) >= 8)

# %% [markdown]
# **B1.2 Top-k and top-p (2 points).** Write `filter_top(probs, top_k=None, top_p=None)`: keep only the `top_k` most likely words, or the smallest group of most likely words whose probabilities add up to at least `top_p`, set the others to 0 and renormalise so the result adds up to 1. Use the probabilities at T = 1 (`probs`). Store in `n_kept` how many words are kept with `top_p = 0.95` (1 point), and in `p_hot_topk2` the new probability of "hot" with `top_k = 2` (1 point).

# %%
def filter_top(probs, top_k=None, top_p=None):
    """Return the renormalised probabilities after top-k and/or top-p filtering."""
    return None   # YOUR CODE HERE

n_kept = None        # YOUR CODE HERE
p_hot_topk2 = None   # YOUR CODE HERE

# %%
# Self-check B1.2
check("B1.2 n_kept (1 point)", same(n_kept, "13f2edb36cb3", 0))
check("B1.2 p_hot_topk2 (1 point)", same(p_hot_topk2, "8cae7f798a6b", 3))

# %% [markdown]
# **B1.3 The chain rule and perplexity (2 points).** The model reads the sentence "We eat rice every day" and gives each token the probability below, given the tokens before it. Compute `log_prob`, the natural logarithm of the probability of the whole sentence (0.5 point), `perplexity`, exp(−mean ln p) (1 point), and `most_surprising`, the word the model found least likely (0.5 point).

# %%
sentence = ["We", "eat", "rice", "every", "day"]
p_tokens = [0.12, 0.30, 0.25, 0.08, 0.50]

log_prob = None          # YOUR CODE HERE
perplexity = None        # YOUR CODE HERE
most_surprising = None   # YOUR CODE HERE (one of the words)

# %%
# Self-check B1.3
check("B1.3 log_prob (0.5 point)", same(log_prob, "d669aba41f50", 2))
check("B1.3 perplexity (1 point)", same(perplexity, "51f673fcbf77", 2))
check("B1.3 most_surprising (0.5 point)", same_word(most_surprising, "b2c9df41982c"))

# %% [markdown]
# **B1.4 A bigram model from a tiny corpus (2 points).** The corpus below is the whole training text of a bigram model. A full stop is a token, and a full stop is added at the start so that the first word of the first sentence also has a previous token. Write `bigram_counts(tokens)`, which returns a dictionary from a previous token to a dictionary of next-token counts, and `p_next(counts, prev, nxt)`, the estimated probability of `nxt` after `prev` (count divided by the total count after `prev`). Then store in `p_rice_after_makes` the probability of "rice" after "makes" (1 point), and in `ppl_test` the perplexity of the test sentence "the cook makes eggs ." under the model, counting the five transitions that start from the leading full stop (1 point).

# %%
corpus = ("the canteen opens at seven . the cook makes rice and the cook makes rice again . "
          "the students eat rice and the students eat eggs . the cook makes eggs . the canteen closes at four .")
tokens = ["."] + corpus.split()

def bigram_counts(tokens):
    """Return {previous token: {next token: count}}."""
    counts = {}
    # YOUR CODE HERE
    return counts

def p_next(counts, prev, nxt):
    """Estimated probability of nxt after prev."""
    return None   # YOUR CODE HERE

counts = bigram_counts(tokens)
test = [".", "the", "cook", "makes", "eggs", "."]
p_rice_after_makes = p_next(counts, "makes", "rice")
ppl_test = None   # YOUR CODE HERE

# %%
# Self-check B1.4
check("B1.4 p_rice_after_makes (1 point)", same(p_rice_after_makes, "d02c7d594ab3", 3))
check("B1.4 ppl_test (1 point)", same(ppl_test, "5cccbc60e7d0", 3))

# %% [markdown]
# **B1.5 Fluent is not the same as true (2 points).** A student asks a chatbot (temperature 0.9) for three sources for a school report. It answers with three book titles, with authors and years, in perfect English. The student cannot find any of the books. In `ANSWERS["B1.5"]` write 3 to 5 sentences: (a) explain, using what you know about how a language model writes, why this can happen (1 point); (b) give two things the student should do before using the answer (1 point).

# %%
ANSWERS["B1.5"] = ""   # YOUR ANSWER: 3 to 5 sentences

# %%
# Self-check B1.5
check("B1.5 written answer present (30 or more words)", n_words(ANSWERS.get("B1.5")) >= 30)

# %% [markdown]
# ## Task 2. Supervised learning, neurons and generalisation (12 points, 25 minutes)
#
# **B2.1 One neuron (2 points).** A neuron has the inputs x = (0.6, −1.5, 2.0), the weights w = (0.7, 0.2, −0.5) and the bias b = 0.3. Compute the weighted sum `z` (1 point) and the output `a_sigmoid` when the activation is the sigmoid (1 point).

# %%
x = np.array([0.6, -1.5, 2.0])
w = np.array([0.7, 0.2, -0.5])
b = 0.3

z = None          # YOUR CODE HERE
a_sigmoid = None  # YOUR CODE HERE

# %%
# Self-check B2.1
check("B2.1 z (1 point)", same(z, "19f09a7d215b", 3))
check("B2.1 a_sigmoid (1 point)", same(a_sigmoid, "2a50cf1f3cc8", 3))

# %% [markdown]
# **B2.2 Counting parameters (2 points).** (a) How many parameters (weights and biases) does a fully connected network with layer sizes 6, 10, 8, 3 have? Store it in `n_params_a` (1 point). (b) A network has layer sizes 4, 5, 2 and **no activation function** between the layers. Store its number of parameters in `n_params_b`, and the number of parameters of the *single* layer that computes exactly the same function in `n_params_single` (1 point for both). In `ANSWERS["B2.2"]` add one sentence on what this says about stacking layers without an activation function. You may build the networks in PyTorch and use `count_params`.

# %%
n_params_a = None       # YOUR CODE HERE
n_params_b = None       # YOUR CODE HERE
n_params_single = None  # YOUR CODE HERE
ANSWERS["B2.2"] = ""    # YOUR ANSWER: one sentence

# %%
# Self-check B2.2
check("B2.2 n_params_a (1 point)", same(n_params_a, "ec9a96338ed2", 0))
check("B2.2 n_params_b and n_params_single (1 point)", same(n_params_b, "b3ea16fb474a", 0) and same(n_params_single, "fdf5f6592cb8", 0))
check("B2.2 written sentence present", n_words(ANSWERS.get("B2.2")) >= 8)

# %% [markdown]
# **B2.3 Two losses (2 points).** (a) A model predicts ŷ = (2.6, 1.9, 4.5, 1.2) for the true values y = (3.0, 1.5, 4.0, 2.0). Store the mean squared error in `mse` (1 point). (b) A 3-class model produces the logits (1.2, −0.3, 0.5) and the true class is class 2 (counting from 0). Turn the logits into probabilities with softmax and store the cross-entropy loss −ln p of the true class in `ce` (1 point).

# %%
y_true = np.array([3.0, 1.5, 4.0, 2.0])
y_pred = np.array([2.6, 1.9, 4.5, 1.2])
class_logits = np.array([1.2, -0.3, 0.5])
true_class = 2

mse = None   # YOUR CODE HERE
ce = None    # YOUR CODE HERE

# %%
# Self-check B2.3
check("B2.3 mse (1 point)", same(mse, "dac3ad0c8cc7", 4))
check("B2.3 ce (1 point)", same(ce, "895efe1cbb03", 3))

# %% [markdown]
# **B2.4 Gradient descent by hand-in-code (2 points).** A model predicts ŷ = w·x with one weight. The data are x = (1, 2, 3) and y = (1.8, 4.5, 5.7), the loss is the mean squared error, and the start is w = 0. For this loss the gradient is dL/dw = (2/n) Σ (w·xᵢ − yᵢ) xᵢ. Take **4** gradient-descent steps with learning rate 0.05. Store the final weight in `w_final` (1 point) and the loss at that weight in `loss_final` (1 point).

# %%
xs = np.array([1.0, 2.0, 3.0])
ys = np.array([1.8, 4.5, 5.7])
w_now, lr = 0.0, 0.05

# YOUR CODE HERE: 4 steps of w_now = w_now - lr * gradient
w_final = None      # YOUR CODE HERE
loss_final = None   # YOUR CODE HERE

# %%
# Self-check B2.4
check("B2.4 w_final (1 point)", same(w_final, "e7303534517e", 3))
check("B2.4 loss_final (1 point)", same(loss_final, "0c259fe78cfa", 3))

# %% [markdown]
# **B2.5 Why a hidden layer? An experiment (2 points).** The XOR data below are split into a training set and a validation set. Train (a) a single linear neuron `linear` and (b) the small tanh network `hidden` with `train_model` (300 epochs, Adam, learning rate 0.05, giving it `Xv=Xva, yv=yva`). Store the **validation** accuracies, which `train_model` returns under `"val_acc"`, in `acc_linear` and `acc_hidden` (1 point). Then plot the two decision regions with `plot_decision` if you wish, and in `ANSWERS["B2.5"]` explain in 2 to 3 sentences what the two numbers show and why (1 point).

# %%
X, y = make_xor(n=300, noise=0.12, seed=7)
Xtr, ytr, Xva, yva = X[:200], y[:200], X[200:], y[200:]
torch.manual_seed(SEED)
linear = nn.Sequential(nn.Linear(2, 1))
hidden = make_deep(depth=2, width=8, act="tanh", init="xavier")

# YOUR CODE HERE: train both models with train_model(...)
acc_linear = None   # validation accuracy of the linear neuron
acc_hidden = None   # validation accuracy of the hidden network
ANSWERS["B2.5"] = ""   # YOUR ANSWER: 2 to 3 sentences

# %%
# Self-check B2.5
ok = acc_linear is not None and acc_hidden is not None
check("B2.5 both accuracies are valid numbers between 0 and 1", ok and 0 <= acc_linear <= 1 and 0 <= acc_hidden <= 1)
check("B2.5 accuracy of the linear neuron", ok and 0.1 < acc_linear < 0.75)
check("B2.5 accuracy of the hidden network", ok and acc_hidden > 0.8)
check("B2.5 written answer present (25 or more words)", n_words(ANSWERS.get("B2.5")) >= 25)

# %% [markdown]
# **B2.6 Overfitting (2 points).** The moons data below have only 40 training points and a lot of noise. Train `big` (3 hidden layers of 64 ReLU units) and `small` (one hidden layer of 3 tanh units) with `train_model` (800 epochs, Adam, learning rate 0.01, `Xv=Xte, yv=yte`). For each model store the gap `train accuracy − test accuracy` (train accuracy is `"acc"`, test accuracy is `"val_acc"`) in `gap_big` and `gap_small` (1 point). In `ANSWERS["B2.6"]` write 3 to 4 sentences: which model shows the signs of overfitting, how you can tell from the numbers, and one thing you could change to reduce it (1 point).

# %%
Xtr, ytr = make_moons(n=40, noise=0.40, seed=3)
Xte, yte = make_moons(n=400, noise=0.40, seed=4)
torch.manual_seed(SEED)
big = make_deep(depth=3, width=64, act="relu", init="he")
small = make_deep(depth=1, width=3, act="tanh", init="xavier")

# YOUR CODE HERE: train both models with train_model(...)
gap_big = None     # train accuracy minus test accuracy of `big`
gap_small = None   # train accuracy minus test accuracy of `small`
ANSWERS["B2.6"] = ""   # YOUR ANSWER: 3 to 4 sentences

# %%
# Self-check B2.6
ok = gap_big is not None and gap_small is not None
check("B2.6 both gaps are valid numbers between -1 and 1", ok and abs(gap_big) <= 1 and abs(gap_small) <= 1)
check("B2.6 gap of `big`", ok and 0.05 < gap_big < 0.8)
check("B2.6 gap of `small`", ok and -0.1 < gap_small < 0.8)
check("B2.6 written answer present (35 or more words)", n_words(ANSWERS.get("B2.6")) >= 35)

# %%
import hashlib

# Checking helpers for Tasks 3 to 5 (do not edit). They compare your numbers with stored fingerprints, so a check never shows the expected value.
def _hv(v, nd):
    return hashlib.sha256(f"B|{round(float(v), nd) + 0.0:.{nd}f}".encode()).hexdigest()[:10]

def b_matches(v, digests, nd=4, shape=None):
    """True if v agrees with the reference to about nd decimals (the reference itself is not stored in this notebook)."""
    if v is None:
        return False
    try:
        arr = np.asarray(v, dtype=float)
        if shape is not None and arr.shape != tuple(shape):
            return False
        vals = np.ravel(arr)
        ks = (-1, 0, 1) if nd > 0 else (0,)
        return len(vals) == len(digests) and all(any(_hv(round(t, nd) + k * 10.0 ** -nd, nd) == dg for k in ks) for t, dg in zip(vals, digests))
    except Exception:
        return False

def b_fingerprint(a):
    a = np.asarray(a, dtype=float)
    return float(np.sum(a * np.cos(np.arange(a.size)).reshape(a.shape)))

def b_word(s, digest):
    return isinstance(s, str) and hashlib.sha256(("B|" + s.strip().lower()).encode()).hexdigest()[:10] == digest

def b_written(s, n=40):
    """True when a written answer has at least n characters (the instructor reads it; this only tells you that something is there)."""
    return isinstance(s, str) and len(s.strip()) >= n

# %% [markdown]
# ## Task 3 — Backpropagation and deep networks (16 points, 30 minutes)
#
# Week 3 covered the forward and backward pass, the chain rule, and why gradients vanish or explode. In B3.1 to B3.3 you follow one small network through a forward pass, a backward pass, and two independent checks. In B3.4 and B3.5 you read the gradients of deep networks and repair one.
#
# **The network.** Two inputs, two hidden units with a sigmoid, one output unit with a sigmoid, and the loss L = ½ (a⁽²⁾ − y)², where a⁽²⁾ is the output. The numbers are in the next cell. Write the arrays with NumPy; use the names given.

# %%
x = np.array([0.8, -1.5]); y = 0.0                       # input and target
W1 = np.array([[0.4, -0.7], [0.9, 0.3]]); b1 = np.array([0.1, -0.2])   # hidden layer (one row per hidden unit)
W2 = np.array([[0.6, -0.5]]); b2 = np.array([0.15])                    # output layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# %% [markdown]
# *Exercise B3.1 (3 points): the forward pass.* Compute the hidden pre-activations `z1` and activations `a1`, the output pre-activation `z2` and output `a2`, and the loss `loss`. Use the formulas of Week 3 and the arrays above.

# %%
z1 = None      # YOUR CODE HERE
a1 = None      # YOUR CODE HERE
z2 = None      # YOUR CODE HERE
a2 = None      # YOUR CODE HERE
loss = None    # YOUR CODE HERE

# %%
check("B3.1 hidden layer: z1 and a1", b_matches(z1, ['f2fe4d4509', 'c424d86ff5'], shape=(2,)) and b_matches(a1, ['5e6c73265c', '6733ac0198'], shape=(2,)))
check("B3.1 output: z2 and a2", b_matches(z2, ['7be25d5ac6']) and b_matches(a2, ['4bf41226b7']))
check("B3.1 loss", b_matches(loss, ['155e464fb3']))

# %% [markdown]
# *Exercise B3.2 (4 points): the backward pass.* Using the chain rule and the two backpropagation equations of Week 3, compute by hand-in-code the blame at the output `delta2`, the gradients `dW2` and `db2` of the output layer, the blame at the hidden layer `delta1`, and the gradients `dW1` and `db1` of the hidden layer. Keep the shapes of the arrays they belong to. Do not use autograd here.

# %%
delta2 = None   # YOUR CODE HERE
dW2 = None      # YOUR CODE HERE
db2 = None      # YOUR CODE HERE
delta1 = None   # YOUR CODE HERE
dW1 = None      # YOUR CODE HERE
db1 = None      # YOUR CODE HERE

# %%
check("B3.2 output layer: delta2, dW2, db2", b_matches(delta2, ['d960b11091'], 5, (1,)) and b_matches(dW2, ['d22a7ac659', '9118978038'], 5, (1, 2)) and b_matches(db2, ['d960b11091'], 5, (1,)))
check("B3.2 hidden layer: delta1 and db1", b_matches(delta1, ['d3d65ac505', '42b4497596'], 5, (2,)) and b_matches(db1, ['d3d65ac505', '42b4497596'], 5, (2,)))
check("B3.2 hidden layer: dW1", b_matches(dW1, ['0617ab7679', 'd5dc9a64d9', 'fa0707d07e', '3893312077'], 5, (2, 2)))

# %% [markdown]
# *Exercise B3.3 (2 points): verify your gradients two ways.* (a) Build the same network in PyTorch with `torch.float64` tensors that require gradients, call `loss.backward()`, and store the four gradients as NumPy arrays in the dictionary `auto_grads` (keys `"W1"`, `"b1"`, `"W2"`, `"b2"`). Store in `max_diff` the largest absolute difference between your hand-in-code gradients (`dW1`, `db1`, `dW2`, `db2`) and these. (b) The function `loss_at` below returns the loss for any hidden weight matrix. Use it with `eps = 1e-5` to estimate the derivative of the loss with respect to `W1[0, 1]` by a central difference, and store it in `fd`.

# %%
def loss_at(W1_):
    """Loss of the network above when the hidden weights are W1_ (everything else unchanged)."""
    a1_ = sigmoid(W1_ @ x + b1)
    return float(np.sum(0.5 * (sigmoid(W2 @ a1_ + b2) - y) ** 2))

eps = 1e-5
auto_grads = None   # YOUR CODE HERE: {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
max_diff = None     # YOUR CODE HERE
fd = None           # YOUR CODE HERE

# %%
check("B3.3 (a) your gradients agree with autograd to 1e-6", max_diff is not None and dW1 is not None and max_diff < 1e-6)
check("B3.3 (b) the finite difference agrees with your dW1[0, 1]", fd is not None and dW1 is not None and abs(fd - dW1[0, 1]) < 1e-6)

# %% [markdown]
# *Exercise B3.4 (3 points): read the gradients of three deep networks.* The function `build` below returns three networks, each with 8 hidden layers of 16 units, built the same way except for what the Week 3 lesson calls the activation and the initialisation. They are trained on the same data. For each network compute the gradient norm of every layer (use `layer_grad_norms` and the data `Xm, ym`) and store the lists in the dictionary `norms`. Then decide whether each network is `"vanishing"`, `"exploding"` or `"healthy"` and put the words in `ANSWERS["B3.4"]` (2 points). In `ANSWERS["B3.4_why"]` write two or three sentences that quote the numbers you used for the decision (1 point).

# %%
Xm, ym = make_moons(200, noise=0.15, seed=3)
Xv, yv = make_moons(200, noise=0.15, seed=4)       # validation data, used in B3.5

def build(name):
    """Network N1, N2 or N3 (always the same weights, because the seed is set inside)."""
    torch.manual_seed(7)
    if name == "N1": return make_deep(8, 16, "relu", "large")
    if name == "N2": return make_deep(8, 16, "sigmoid", "default")
    if name == "N3": return make_deep(8, 16, "relu", "he")

norms = {}
for name in ("N1", "N2", "N3"):
    pass   # YOUR CODE HERE: norms[name] = layer_grad_norms(...), and print it
ANSWERS["B3.4"] = {"N1": "", "N2": "", "N3": ""}   # YOUR ANSWER: "vanishing", "exploding" or "healthy" for each network
ANSWERS["B3.4_why"] = ""                           # YOUR ANSWER: two or three sentences with the numbers

# %%
a34 = ANSWERS.get("B3.4", {})
check("B3.4 network N1", b_word(a34.get("N1"), "3917e9e312"))
check("B3.4 network N2", b_word(a34.get("N2"), "2976fbf6b5"))
check("B3.4 network N3", b_word(a34.get("N3"), "4038ac4e59"))
check("B3.4 evidence written", b_written(ANSWERS.get("B3.4_why"), 60))

# %% [markdown]
# *Exercise B3.5 (4 points): repair the vanishing network.* One of the three networks cannot learn. (a) For the original unhealthy network store in `ratio_before` the ratio (gradient norm of the first layer divided by the norm of the last layer, before any training), and in `acc_before` its validation accuracy after training it with `train_model(model, Xm, ym, epochs=300, lr=0.05, Xv=Xv, yv=yv)` (the result dictionary has the key `"val_acc"`). (b) Build a repaired network in the variable `fixed` with one change from Week 3 (set `torch.manual_seed(7)` first, and keep 8 hidden layers of 16 units), and store in `ratio_after` and `acc_after` the same two numbers for it. A good repair raises the ratio by a factor of at least 1000 and reaches a validation accuracy of at least 0.9 (2 points for the repaired network and its numbers, 1 point for the before numbers). (c) In `ANSWERS["B3.5"]` say in two or three sentences which change you made and why it works, quoting your before and after numbers (1 point).

# %%
ratio_before = None   # YOUR CODE HERE
acc_before = None     # YOUR CODE HERE
fixed = None          # YOUR CODE HERE: the repaired network (torch.manual_seed(7) first)
ratio_after = None    # YOUR CODE HERE
acc_after = None      # YOUR CODE HERE
ANSWERS["B3.5"] = ""  # YOUR ANSWER: which change, why it works, with your numbers

# %%
check("B3.5 (a) the 'before' numbers", ratio_before is not None and acc_before is not None and b_matches(np.log10(ratio_before), ['a2fc78e558'], 1) and b_matches(acc_before, ['70848d9697'], 2))
check("B3.5 (b) the repaired network is a model and its ratio improved 1000 times", isinstance(fixed, nn.Module) and ratio_after is not None and ratio_before is not None and ratio_after >= 1000 * ratio_before)
check("B3.5 (b) the repaired network reaches validation accuracy 0.9", acc_after is not None and acc_after >= 0.9)
check("B3.5 (c) explanation written", b_written(ANSWERS.get("B3.5"), 60))

# %% [markdown]
# ## Task 4 — Convolutional networks (16 points, 30 minutes)
#
# Week 4 covered convolution, output sizes, parameters, the backward pass of a convolution, and why a CNN suits pictures. In B4.1 and B4.2 you compute by hand-in-code; in B4.3 and B4.4 you build, train and interpret a small CNN against a dense network.

# %% [markdown]
# *Exercise B4.1 (4 points): a convolution and its sizes.* (a) Convolve the picture `x1` with the kernel `K1` using stride 2 and padding 1 (deep learning convolution, no flipping) and store the result in `out1` (1 point). Store in `n_out` the side length of the output that the output-size formula gives, computed from the formula without looking at the shape of `out1` (1 point). (b) A convolution layer has 3 input channels and 12 filters of size 5 × 5 (with biases). Store its number of parameters in `conv_params` (1 point). Give it a 3 × 40 × 40 picture with stride 2 and padding 2: store the output side length in `out_side`, and in `dense_params` the number of parameters of a fully connected layer (with biases) that maps the flattened 3 × 40 × 40 input to the flattened output of that convolution, 12 × `out_side` × `out_side` (1 point).

# %%
x1 = np.array([[1, 0, 3, 2, 4, 1, 0],
               [2, 5, 1, 0, 3, 3, 1],
               [0, 4, 2, 1, 5, 0, 2],
               [3, 1, 0, 4, 2, 2, 5],
               [1, 2, 5, 3, 0, 4, 1],
               [4, 0, 1, 2, 3, 5, 0],
               [2, 3, 4, 0, 1, 1, 3]], dtype=float)
K1 = np.array([[2, -1, 0], [1, 0, -1], [0, 1, -2]], dtype=float)
out1 = None          # YOUR CODE HERE
n_out = None         # YOUR CODE HERE (from the formula)
conv_params = None   # YOUR CODE HERE
out_side = None      # YOUR CODE HERE
dense_params = None  # YOUR CODE HERE

# %%
check("B4.1 (a) the convolution output", out1 is not None and np.shape(out1) == (4, 4) and b_matches(b_fingerprint(out1), ['45887cb2c1'], 3))
check("B4.1 (a) the output side from the formula", b_matches(n_out, ['e54b140538'], 0))
check("B4.1 (b) parameters of the convolution layer", b_matches(conv_params, ['90ad586faf'], 0))
check("B4.1 (b) output side and dense parameters", b_matches(out_side, ['8c32de89fa'], 0) and b_matches(dense_params, ['9593f06880'], 0))

# %% [markdown]
# *Exercise B4.2 (5 points): the backward pass of a convolution.* The loss is L = the sum of `G2` times `conv2d_np(x2, K2)` (stride 1, no padding), so `G2` is the upstream gradient dL/d(output). Write `conv_backward_kernel(x, G, kh, kw)`, which returns dL/dK (2 points), and `conv_backward_input(K, G, in_shape)`, which returns dL/dx (2 points), with explicit loops over the output positions and the kernel cells. Then verify them (1 point): store in `err_auto` the largest absolute difference between your two gradients and PyTorch's (use `F.conv2d` on `float64` tensors with `requires_grad`, remember that PyTorch also wants shapes (1, 1, H, W)), and in `fd_K` a central-difference estimate (`eps = 1e-6`) of dL/dK[1, 2] computed from `conv2d_np`.

# %%
x2 = np.array([[2, 0, 1, 3, 1, 0],
               [1, 4, 2, 0, 2, 1],
               [0, 3, 1, 2, 5, 3],
               [2, 1, 0, 1, 3, 0],
               [4, 2, 3, 0, 1, 2],
               [1, 0, 2, 4, 0, 1]], dtype=float)
K2 = np.array([[1, 2, 0], [-1, 1, 3], [0, -2, 1]], dtype=float)
G2 = np.array([[1, 0, -2, 1],
               [2, 1, 0, -1],
               [0, -1, 3, 1],
               [1, 2, -1, 0]], dtype=float)

def conv_backward_kernel(x, G, kh, kw):
    """dL/dK for stride 1 and no padding. Returns an array of shape (kh, kw)."""
    dK = np.zeros((kh, kw))
    # YOUR CODE HERE
    return dK

def conv_backward_input(K, G, in_shape):
    """dL/dx for stride 1 and no padding. Returns an array of shape in_shape."""
    dx = np.zeros(in_shape)
    # YOUR CODE HERE
    return dx

dK2 = conv_backward_kernel(x2, G2, 3, 3)
dx2 = conv_backward_input(K2, G2, x2.shape)

# %%
err_auto = None   # YOUR CODE HERE
fd_K = None       # YOUR CODE HERE

# %%
check("B4.2 dL/dK", np.shape(dK2) == (3, 3) and b_matches(b_fingerprint(dK2), ['0e1763d908'], 3))
check("B4.2 dL/dx", np.shape(dx2) == (6, 6) and b_matches(b_fingerprint(dx2), ['d41ea2c541'], 3))
check("B4.2 verification: your gradients agree with PyTorch to 1e-6 and with the finite difference", err_auto is not None and fd_K is not None and err_auto < 1e-6 and abs(fd_K - dK2[1, 2]) < 1e-5)

# %% [markdown]
# *Exercise B4.3 (5 points): a CNN against a dense network, under shift.* The data below are shapes of size 20 × 20: `Xtr, ytr` (500 pictures, centred) for training, `X0, y0` (centred) and `X4, y4` (every shape moved up to 4 pixels away from the centre) for testing. (a) Build a CNN `cnn` (`nn.Sequential`, at least two convolution layers, at most 3000 parameters, 5 outputs) and a dense network `mlp` (`Flatten`, a hidden layer with ReLU, an output layer) whose parameter count is within 15 % of the CNN's (2 points). Call `torch.manual_seed(0)` before building each. (b) Train both with `train_classifier(model, Xtr, ytr, X0, y0, epochs=25, lr=0.01)` and fill the dictionary `RES` with the test accuracies `"cnn_centre"`, `"cnn_shift"`, `"mlp_centre"`, `"mlp_shift"` measured with `accuracy` on `X0` and `X4` (2 points). (c) Store the drops `drop_cnn` and `drop_mlp` (centre accuracy minus shifted accuracy) (1 point).

# %%
Xtr, ytr = make_shapes(500, 20, noise=0.2, shift=0, seed=11)
X0, y0 = make_shapes(300, 20, noise=0.2, shift=0, seed=12)
X4, y4 = make_shapes(300, 20, noise=0.2, shift=4, seed=13)

def accuracy(model, X, y):
    """Fraction of pictures the model classifies correctly."""
    model.eval()
    with torch.no_grad():
        return float((model(torch.as_tensor(X)).argmax(1) == torch.as_tensor(y)).float().mean())

torch.manual_seed(0)
cnn = None   # YOUR CODE HERE
torch.manual_seed(0)
mlp = None   # YOUR CODE HERE

# %%
RES = {}
# YOUR CODE HERE: train both networks and fill RES
drop_cnn = None   # YOUR CODE HERE
drop_mlp = None   # YOUR CODE HERE

# %%
def _built_ok():
    try:
        convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
        with torch.no_grad():
            ok_shape = tuple(cnn(torch.zeros(2, 1, 20, 20)).shape) == (2, 5) and tuple(mlp(torch.zeros(2, 1, 20, 20)).shape) == (2, 5)
        return len(convs) >= 2 and count_params(cnn) <= 3000 and ok_shape and abs(count_params(mlp) - count_params(cnn)) <= 0.15 * count_params(cnn)
    except Exception:
        return False
check("B4.3 (a) both networks have the right shape and size", _built_ok())
def _trained_ok():
    try:
        return all(k in RES for k in ("cnn_centre", "cnn_shift", "mlp_centre", "mlp_shift")) and \
            all(abs(RES[k] - accuracy(m, X, y)) < 1e-9 for k, m, X, y in [("cnn_centre", cnn, X0, y0), ("cnn_shift", cnn, X4, y4), ("mlp_centre", mlp, X0, y0), ("mlp_shift", mlp, X4, y4)])
    except Exception:
        return False
check("B4.3 (b) RES matches the trained networks", _trained_ok())
check("B4.3 (b) both networks learn the centred shapes (at least 0.9)", _trained_ok() and RES["cnn_centre"] >= 0.9 and RES["mlp_centre"] >= 0.9)
check("B4.3 (c) the drops", _trained_ok() and drop_cnn is not None and drop_mlp is not None and abs(drop_cnn - (RES["cnn_centre"] - RES["cnn_shift"])) < 1e-9 and abs(drop_mlp - (RES["mlp_centre"] - RES["mlp_shift"])) < 1e-9)

# %% [markdown]
# *Exercise B4.4 (2 points): interpret.* In `ANSWERS["B4.4"]` write three or four sentences. Say which network is more robust to the shift and by how many percentage points the two drops differ (use your numbers), explain the difference with the ideas of Week 4 (what the weights of each kind of layer are attached to), and name one change that would make the weaker network cope better.

# %%
ANSWERS["B4.4"] = ""   # YOUR ANSWER

# %%
check("B4.4 answer written", b_written(ANSWERS.get("B4.4"), 80))

# %% [markdown]
# ## Task 5 — Synthesis (6 points, 15 minutes)
#
# Three short scenarios that join several weeks. Answer each in `ANSWERS` in two to five sentences (2 points each). Use numbers where the question asks for them; you may use code cells for the arithmetic.

# %% [markdown]
# *Exercise B5.1 (2 points; Weeks 2 and 3).* A classmate trains a network with 10 hidden layers, all sigmoid, default initialisation, on 5,000 rows with a binary-cross-entropy loss. The training loss starts at 0.693 and is still 0.693 after 200 epochs; the validation loss is the same. She plans to collect more data and to lower the learning rate. (a) What does the value 0.693 tell you about the predictions? (b) Explain why her two plans will not help, quoting the Week 3 number for how much a sigmoid layer can pass back and what it gives over 10 layers. (c) Name two changes that would help.

# %% [markdown]
# *Exercise B5.2 (2 points; Weeks 2 and 4).* A team builds a leaf-disease app. It has 300 training photos and 100 test photos, 28 × 28 pixels in grey, three classes. A dense network 784 → 128 → 3 reaches 100 % on the training photos and 62 % on the test photos, and users complain that it fails when the leaf is not in the middle. (a) Count the parameters of that dense network, and of one convolution layer with 8 filters of size 5 × 5 on one channel. (b) Name the problem in Week 2 words, and say why a convolutional network would suit this task better, using Week 4 ideas. (c) Give one fix from the data side and one from the model side.

# %% [markdown]
# *Exercise B5.3 (2 points; Weeks 1 and 2).* A language model has to choose the next word. For three candidate words its scores (logits) are 3.0, 1.0 and 0.0. (a) Compute the probability of the top word at temperature 1 and at temperature 0.5. (b) During training the correct next word was the second candidate. Compute the cross-entropy loss −ln p at temperature 1, and say in which direction training moves the score of that word. (c) A school wants a chatbot for exam-revision answers that are accurate and repeatable. Which temperature setting would you choose, and what would go wrong with a very high temperature?

# %%
ANSWERS["B5.1"] = ""   # YOUR ANSWER
ANSWERS["B5.2"] = ""   # YOUR ANSWER
ANSWERS["B5.3"] = ""   # YOUR ANSWER

# %%
for _k in ("B5.1", "B5.2", "B5.3"):
    check(_k + " answer written", b_written(ANSWERS.get(_k), 80))

# %%
print("Your ANSWERS (copy them into your submission if your instructor asks):")
for k in sorted(ANSWERS):
    print(f"\n{k}: {ANSWERS[k]}")

# %% [markdown]
# ## Finishing the exam
#
# 1. Check that every sub-question has an answer: the ✓ and ✗ lines above only tell you whether a number or a shape is right; your instructor marks the notebook with the marking scheme (correct result, method and interpretation), so make your method visible and write your reasoning in the `ANSWERS` entries.
# 2. Run **Runtime → Restart session and run all** (Colab) or **Kernel → Restart Kernel and Run All Cells** (JupyterLab) once, to make sure the notebook runs from top to bottom without errors.
# 3. In Colab, use **File → Save a copy in Drive** now if you have not done so, and download the notebook (**File → Download → .ipynb**); files in the session disappear when it ends. In JupyterLab, save the notebook (Ctrl+S).
# 4. Submit the notebook the way your instructor asks.
#
# Good luck, and well done for getting this far in the course.
