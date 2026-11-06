# %% [markdown]
# # Week 11 — Multimodal AI: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is fine for everything; no GPU is needed. Everything is drawn with NumPy and PIL, so nothing is downloaded except, in Part 6, the real CLIP model (`openai/clip-vit-base-patch32`, about 600 MB, once). **Time:** about 3 hours for this notebook. Every cell finishes in seconds to about two minutes.
#
# This lab is called **Imagine a model that sees and reads**. Week 11 asks how a Transformer, which reads sequences of tokens, can look at a picture, and how a picture and a sentence can end up in the same space. You answer by building the story from the bottom, with pictures that you draw yourself: a toy problem in which each modality alone is ambiguous, the cutting of a picture into patches, a proof that a bag of patches has no layout, a tiny Vision Transformer of your own, a miniature CLIP trained from scratch, and finally the real CLIP.
#
# ### What you will do
#
# 1. **Part 1. What is multimodal? Modalities, fusion and complementarity** (25 min): the shape and size of each modality, a data set of coloured shapes with captions, colour and shape features, early and late fusion, and three real applications.
# 2. **Part 2. Patchify: from pixels to tokens** (20 min): cut pictures into patches by hand, with `unfold` and with a strided `Conv2d`, count the tokens and parameters of a real ViT, and see why patches make attention affordable.
# 3. **Part 3. Why position embeddings?** (25 min): prove that self-attention cannot tell the order of its tokens, see it on shuffled pictures, add learned and sinusoidal 2D positions, and resize them for a bigger picture.
# 4. **Part 4. Build a tiny Vision Transformer** (30 min): write `PatchEmbed` and `TinyViT`, count the parameters, train it on line shapes and look at the attention of the CLS token.
# 5. **Part 5. ViT experiments: positions, pooling, patch size** (25 min): a left-or-right task that a ViT without positions cannot solve, CLS against mean pooling, patch size, and how much data a ViT needs.
# 6. **Part 6. Contrastive learning and CLIP** (30 min): the contrastive loss by hand, the temperature, a miniature CLIP trained from scratch with zero-shot on unseen combinations, and the real CLIP.
# 7. **Part 7. Design a multimodal system, and reflect** (15 min): a spec card, a token budget, a small evaluation per group and four reflection questions.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells (the exercises) are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried. Your key numbers are collected in a dictionary called `RESULTS` that Part 7 prints.

# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed so your results match the ones in the lesson, and defines helpers: `check(...)` to test your answers; three small picture data sets drawn with NumPy so that nothing has to be downloaded: `make_shapes(...)` (five line shapes, from Week 4), `make_colored_shapes(...)` (a red, green or blue square, circle or cross, with a short text caption such as "red circle") and `make_left_right(...)` (a plus sign and a distracting square at random places; the label says whether the plus sign is in the left half of the picture: the answer depends on **where** the plus is, which is what position embeddings are for); `show_images(...)` to look at them; `count_params(...)` and `train_classifier(...)` to train any PyTorch classifier and keep its learning curves; and, for the real CLIP model of Part 6, `draw_shape_pil(...)` (a clean 224 × 224 picture of a coloured shape) and `load_clip()` (a few hundred MB, downloaded once; on the school's Spark it comes from the offline cache). Everything else runs on a CPU in a few minutes.

# %%
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
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone
print("Running on:", device)

RESULTS = {}   # a place to keep your key numbers and answers; the last part prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

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
        m = _mask(SHAPES3[shp[i]], size, cy, cx, 4)
        for ch in range(3): X[i, ch][m] = _RGB[COLOURS[col[i]]][ch]
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

# %% [markdown]
# ## Part 1. What is multimodal? Modalities, fusion and complementarity (25 min)
#
# *Lesson section: "Multimodal".* A **modality** is a kind of data: pictures, sound, text, video. A *multimodal* model takes in, or gives out, more than one of them. The reason is simple: each modality alone is often ambiguous, and two together can remove the doubt. In this part you do not touch a big model yet. You build the smallest honest example of the idea with pictures you draw yourself: a red or green or blue square, circle or cross. The colour is one piece of information and the shape is another, and a caption such as "red circle" needs both. You will measure how well "colour only", "shape only" and "both together" do, compare two ways of putting them together (early and late fusion), and then think about three real applications and how they can go wrong.

# %% [markdown]
# ### ✏️ Imagine first
#
# The data set of this part has nine kinds of picture (three colours times three shapes) and a classifier must name the nine-way caption, for example "green cross". Suppose one model sees only the colour of the picture, a second model sees only the shape, and a third model sees both. Before you run any code, write three predictions: what test accuracy (a number between 0 and 1) will each of the three reach? Remember that guessing among nine classes gives about 0.11. You will compare your answers with Step 1.4.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: the nine-way caption task (colour x shape)
my_colour_only = None   # your guess for the test accuracy of a model that sees only the colour (a number between 0 and 1)
my_shape_only = None    # ... of a model that sees only the shape
my_both = None          # ... of a model that sees both

# %% [markdown]
# ### Step 1.1. Modalities are tensors
#
# Every modality reaches a computer as numbers, and what changes from one to the other is the *shape* of the array and the *amount* of numbers. A colour picture is a 3-D array (channels, height, width); a sound is a long 1-D array of samples; a sentence is a short list of integer token ids that a lookup table turns into vectors. The block below builds one dummy example of each (filled with zeros: only the shapes matter) and prints a small table of shape and size. The fourth row is a real 16 by 16 greyscale picture from `make_shapes`, the data of Week 4.

# %%
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

# %% [markdown]
# **What you should see.** You should see a table with four rows. The 224 by 224 RGB picture has shape (3, 224, 224) and **150,528** numbers; one second of audio at 16 kHz has shape (16000,) and **16,000** numbers; the sentence is only **12** integer ids; and the 16 by 16 greyscale picture of Week 4 has **256** numbers. The silent assertions at the end passed. Notice the range: a single photograph is about ten times larger than a second of sound and thousands of times larger than a sentence, and yet a sentence can carry more meaning. The three modalities have different shapes and very different sizes, and a multimodal model has to bring them to one common format (vectors of the same length, a sequence of tokens).

# %% [markdown]
# **Tips and common mistakes**
#
# - `arr.size` is the number of entries and `arr.shape` the tuple of sizes; `np.zeros` is only a stand-in, because in this table only the shapes and counts matter.
# - A *token id* is not yet a meaningful number: it is an index into an embedding table (Week 7). The 12 ids become a 12 by d array only after that lookup.
# - Numbers are not information: a 150,528-number picture of a blank wall carries far less than a 12-token sentence that tells you something new. Later parts show how models shorten the long modalities into tokens.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.1. Count the numbers
#
# Your task: compute how many numbers are in (a) 30 seconds of audio at 16 kHz and (b) one 640 by 480 RGB photograph, and store them in `n_audio30` and `n_photo`. Why it matters: the size of the raw input is the first thing a designer of a multimodal system must know: it decides what must be shortened before a Transformer can read it.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
n_audio30 = None   # number of numbers in 30 seconds of audio at 16 kHz
n_photo = None     # number of numbers in a 640 x 480 RGB photograph

# %%
if n_audio30 is None or n_photo is None:
    print("Fill in n_audio30 and n_photo above, then run this cell again.")
else:
    check("n_audio30 is right", n_audio30 == 30 * 16000, "Samples per second times seconds.")
    check("n_photo is right", n_photo == 3 * 480 * 640, "Three channels times height times width.")
    RESULTS["p1_counts"] = (int(n_audio30), int(n_photo))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# n_audio30 = 30 * 16000       # 480,000
# n_photo = 3 * 480 * 640      # 921,600
# ```
#
# `n_audio30` is 480,000 and `n_photo` is 921,600. Thirty seconds of audio is already 480,000 numbers, and a single phone photograph is almost a million: this is why audio and image models do not read the raw samples one by one but first shorten them into patches or frames.
#
# </details>

# %% [markdown]
# ### Step 1.2. Look at the data
#
# From now on we use `make_colored_shapes`. It draws small 16 by 16 colour pictures with NumPy (nothing is downloaded): a red, green or blue square, circle or cross near the centre of a white background, plus a little noise, and it returns a caption such as "red circle" for each. Always look at your data before you model it: the block shows ten pictures with their captions and counts how many pictures each of the nine possible captions has.

# %%
Xc, col, shp, caps = make_colored_shapes(300, seed=SEED)
print("pictures:", Xc.shape, Xc.dtype, " pixel range", round(float(Xc.min()), 2), "to", round(float(Xc.max()), 2))
show_images(Xc, caps, n=10, cols=5)
counts = Counter(caps)
for name, k in sorted(counts.items()):
    print(f"{name:14s} {k}")
print("different captions:", len(counts), " total:", sum(counts.values()))

# %% [markdown]
# **What you should see.** You should see a figure with ten small pictures in two rows of five (a red, green or blue square, circle or cross, each with its caption above), and text like this: the array has shape (300, 3, 16, 16), type float32, and pixel values between 0.0 and 1.0. The nine captions occur between 30 and 37 times each: `blue circle` 30, `blue cross` 33, `blue square` 37, `green circle` 35, `green cross` 32, `green square` 33, `red circle` 34, `red cross` 34 and `red square` 32; 9 different captions and 300 pictures in total. So the classes are well balanced: the most frequent caption, `blue square`, has 37 pictures (12.3 percent) and a model that always answers with it would be right about one time in nine, not better. Every picture is only 16 by 16 pixels, so a square or a circle is a coarse blob; that is enough for this lab.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.2. The most frequent caption
#
# Your task: find the caption that occurs most often in `caps` and how many times, and store them in `top_caption` (a string) and `top_count` (an integer). Why it matters: before you train anything you should know whether the classes are balanced: a model that always answers with the most frequent class sets the baseline that every real model must beat.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
top_caption = None   # the most frequent caption (a string such as "red circle")
top_count = None     # how many pictures have it

# %%
if top_caption is None or top_count is None:
    print("Fill in top_caption and top_count above, then run this cell again.")
else:
    t_name, t_count = Counter(caps).most_common(1)[0]
    check("top_caption is right", top_caption == t_name, "Counter(caps).most_common(1) gives a list with one (caption, count) pair.")
    check("top_count is right", top_count == t_count, "The second entry of that pair.")
    check("it beats a uniform share", top_count >= 300 / 9, "The largest class has at least 300 / 9 pictures.")
    RESULTS["p1_top_caption"] = (top_caption, int(top_count))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# top_caption, top_count = Counter(caps).most_common(1)[0]
# ```
#
# `top_caption` is `blue square` and `top_count` is 37. The nine classes have between 30 and 37 pictures each: balanced.
#
# </details>

# %% [markdown]
# ### Step 1.3. Two hand-made features: colour and shape
#
# We now describe each picture by a few numbers, the way people did before deep learning. The **colour feature** is the mean red, green and blue value of the pixels that are *not* near-white (the ink pixels). The **shape feature** is made of the number of ink pixels, the width-to-height ratio of the bounding box of the ink, and the *fill* (ink pixels divided by the area of the bounding box). The first one looks only at colours and the others look only at the geometry: two "modalities" of the same picture. The functions below compute them and the block prints three examples.

# %%
def ink_mask(X, thr=0.6):
    """True where a pixel is not near-white: its smallest channel is below thr. Shape (n, H, W)."""
    return X.min(axis=1) < thr

def colour_features(X):
    """Mean R, G, B of the ink pixels of each picture: shape (n, 3)."""
    m = ink_mask(X)
    return np.array([[X[i, ch][m[i]].mean() for ch in range(3)] for i in range(len(X))], np.float32)

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

# %% [markdown]
# **What you should see.** You should see three lines, one for each of the first three pictures. The first is a `blue circle`: colour about (0.15, 0.25, 0.98), that is, no red, a little green and a lot of blue; shape (69 pixels, ratio 1.0, fill 0.85). The next two are `blue square` pictures with the same colour and shape (81 pixels, ratio 1.0, fill 1.0). The last line prints all the width-to-height ratios of the 300 pictures: there is only **one** value, `[1.]`. All three shapes fit a 9 by 9 box, so the ratio tells nothing, and a feature that never changes can teach nothing: that is why we also computed the pixel count and the fill. The colour features separate red, green and blue clearly (the tallest value is the colour of the ink); the shape features separate the three shapes (about 81, 69 and 45 ink pixels for a square, a circle and a cross), but neither says anything about the other task.

# %% [markdown]
# **Tips and common mistakes**
#
# - The threshold 0.6 works because the background is near 1.0 in every channel and each shape has at least one channel below 0.3. The noise (standard deviation 0.05) never crosses it.
# - A feature that is constant (the ratio here) carries no information. Real feature engineering means finding out which numbers do vary with the thing you want to know.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.3. Which shape has the most ink?
#
# Your task: compute the mean number of ink pixels (first column of `shape_f`) for each of the three shapes and store them in the dictionary `mean_pixels`, with the keys `"square"`, `"circle"` and `"cross"`, rounded to one decimal. Which shape has the most ink and which the least? Why it matters: this is how a single geometric number separates a task: here the pixel count alone will tell the three shapes apart, but it says nothing about the colour.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
mean_pixels = {"square": None, "circle": None, "cross": None}   # mean number of ink pixels per shape, rounded to 1 decimal

# %%
if any(v is None for v in mean_pixels.values()):
    print("Fill in mean_pixels above, then run this cell again.")
else:
    truth = {s: round(float(shape_f[shp == k, 0].mean()), 1) for k, s in enumerate(SHAPES3)}
    check("the three means are right", all(abs(mean_pixels[s] - truth[s]) < 0.11 for s in SHAPES3), "Select the rows with shp equal to the shape index, then average column 0.")
    check("the order is right (square > circle > cross)", mean_pixels["square"] > mean_pixels["circle"] > mean_pixels["cross"], "A cross is two thin bars, a square is filled.")
    RESULTS["p1_mean_pixels"] = dict(mean_pixels)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# mean_pixels = {s: round(float(shape_f[shp == k, 0].mean()), 1) for k, s in enumerate(SHAPES3)}
# ```
#
# `mean_pixels` is `{'square': 81.0, 'circle': 69.0, 'cross': 45.0}`. The square has the most ink (a filled 9 by 9 block), the cross the least. The pixel count alone is enough to separate the three shapes, but gives no information about colour.
#
# </details>

# %% [markdown]
# ### Step 1.4. Two ambiguous tasks, one fused answer (early fusion)
#
# The goal is the nine-way caption (colour times shape), with the label `y9 = colour * 3 + shape`. We train three small classifiers, each a single `nn.Linear` layer (logistic regression) trained with the `train_classifier` helper: (a) on the colour features only, (b) on the shape features only, (c) on both feature sets glued together side by side. Gluing the inputs *before* the model is called **early fusion**. We generate 900 new pictures, use 600 for training and 300 for testing, and standardise every feature (subtract the training mean, divide by the training standard deviation) so that the pixel counts, which are large, do not drown the small numbers.

# %%
Xa, ca, sa, _ = make_colored_shapes(900, seed=SEED + 1)
Fc_all, Fs_all, y9 = colour_features(Xa), shape_features(Xa), ca * 3 + sa
tr, te = np.arange(900) < 600, np.arange(900) >= 600

def prep(Fm):
    """Standardise with the training mean and standard deviation (a constant column becomes zeros)."""
    mu, sd = Fm[tr].mean(0), Fm[tr].std(0)
    return ((Fm - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)

feats = {"colour only": prep(Fc_all), "shape only": prep(Fs_all)}
feats["both (early fusion)"] = np.concatenate([feats["colour only"], feats["shape only"]], axis=1)

acc9 = {}
for name, Fm in feats.items():
    torch.manual_seed(SEED)
    h = train_classifier(nn.Linear(Fm.shape[1], 9), Fm[tr], y9[tr], Fm[te], y9[te], epochs=60, lr=0.05, seed=SEED)
    acc9[name] = h["test_acc"][-1]
    print(f"{name:22s} features: {Fm.shape[1]}   train {h['train_acc'][-1]:.3f}   test {acc9[name]:.3f}")
print("chance for nine classes:", round(1 / 9, 3), "  my guesses:", my_colour_only, my_shape_only, my_both)

# %% [markdown]
# **What you should see.** You should see three lines (the exact decimals are stable here because the pictures are drawn with a fixed seed):
#
# - colour only (3 features): train 0.363, test **0.343**;
# - shape only (3 features): train 0.358, test **0.343**;
# - both, early fusion (6 features): train 1.000, test **1.000**;
#
# and the line "chance for nine classes: 0.111" followed by your own guesses. Each single modality reaches only about one third: the colour tells you which of the three colours it is, but then three shapes are still possible, so the best you can do is to pick one of them and be right one time in three. The two modalities together are **perfect** on these clean pictures. This is an easy toy problem, so do not expect 1.000 on real data; the point is the pattern: 0.34 and 0.34 become 1.00. Compare with your guesses of the Imagine-first cell: here the two single modalities are equally ambiguous, and neither can be better than one third.

# %% [markdown]
# **Why this happens.** The colour features know *which colour* but nothing about the shape, so among the nine captions they can only narrow the answer down to three candidates; the best such classifier is right one time in three. The shape features are the mirror image. Each modality alone is *ambiguous* for this task, and the two are *complementary*: what one lacks, the other has. Together they identify the caption. This is the same reason why a photo and its caption, or a voice and a face, can be worth more together than the sum of their parts.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.4. Complementarity in one number
#
# Your task: compute the share of the nine-way test accuracy that the fused model gains over the better single-modality model: `gain = acc9["both (early fusion)"] - max(acc9["colour only"], acc9["shape only"])`, rounded to 3 decimals, and store whether the fused model is more than twice as accurate as the better single one in `more_than_double` (True or False). Why it matters: you will do this comparison for every multimodal system you ever evaluate: the question is not "is the model accurate" but "does the second modality add something the first did not have".
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
gain = None              # accuracy of the fused model minus the accuracy of the better single-modality model (3 decimals)
more_than_double = None  # True or False

# %%
if gain is None or more_than_double is None:
    print("Fill in gain and more_than_double above, then run this cell again.")
else:
    best_single = max(acc9["colour only"], acc9["shape only"]); fused = acc9["both (early fusion)"]
    check("gain is right", abs(gain - round(fused - best_single, 3)) < 0.0015, "fused accuracy minus the larger of the two single accuracies.")
    check("more_than_double is right", bool(more_than_double) == (fused > 2 * best_single), "Compare fused with 2 times the better single accuracy.")
    RESULTS["p1_gain"] = (float(gain), bool(more_than_double))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_single = max(acc9["colour only"], acc9["shape only"])
# gain = round(acc9["both (early fusion)"] - best_single, 3)
# more_than_double = acc9["both (early fusion)"] > 2 * best_single
# ```
#
# `gain` is 0.657 (1.000 minus 0.343) and `more_than_double` is `True`. The fused model is about three times as accurate as either single modality, the clearest sign of complementarity.
#
# </details>

# %% [markdown]
# ### Step 1.5. Late fusion: combine the answers, not the inputs
#
# There is another way to combine two modalities: build one expert for each and merge their *answers*. This is **late fusion**. Here we train a colour classifier (three classes, colour features only) and a shape classifier (three classes, shape features only), turn each output into probabilities with a softmax and multiply them: the probability of the caption "colour c and shape s" is `p_colour[c] * p_shape[s]`, an "AND" of two independent opinions. The answer is the caption with the largest product. The block also shows what happens when one modality is missing (we replace the shape expert's opinion by a uniform one).

# %%
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

# %% [markdown]
# **What you should see.** You should see four lines: the colour expert alone is right on **1.000** of the test pictures when it only has to name the colour (three classes); the shape expert alone, **1.000** for the shape (three classes); late fusion, the product of the two, gets the nine-way caption right on **1.000** of the test pictures, the same as early fusion; and when the shape expert is switched off (replaced by a uniform opinion) the nine-way accuracy falls to **0.333**: without the shape the model can only get the colour right and has to pick one of the three shapes. So on this clean data early and late fusion are equally good, and late fusion has the extra property that it survives a missing modality (it falls back to one third instead of failing).

# %% [markdown]
# **Early against late fusion.** Early fusion lets one model see all the inputs at once, so it can learn *interactions* between modalities (a red circle might look different from a red square in a way that no single feature shows), but it needs data in which all modalities are present together and it breaks if one is missing. Late fusion keeps the experts separate: you can train each on its own data, swap one out, and still answer when one modality is missing, but the experts cannot learn from the interaction. Real systems sit between the two: CLIP (Part 6) fuses at the very end, in a shared space, and a vision-language model fuses early, by turning picture patches into tokens next to the words.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.5. Which modality is missing?
#
# Your task: in the block above we switched the *shape* expert off. Do the same for the *colour* expert: replace `p_col` by a uniform opinion (`1/3` for every colour), compute the nine-way test accuracy and store it in `colour_missing_acc` (3 decimals). Why it matters: a robust system should degrade gracefully when one sensor fails; knowing how much it degrades is part of the evaluation.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
colour_missing_acc = None   # nine-way test accuracy of late fusion when the colour expert says "I do not know" (3 decimals)

# %%
if colour_missing_acc is None:
    print("Fill in colour_missing_acc above, then run this cell again.")
else:
    _pred = (np.full((900, 3, 1), 1 / 3) * p_shp[:, None, :]).reshape(900, 9).argmax(1)
    truth = round(float((_pred[te] == y9[te]).mean()), 3)
    check("colour_missing_acc is right", abs(colour_missing_acc - truth) < 0.0015, "Multiply a uniform colour opinion with p_shp, then take the argmax over the nine captions.")
    check("it is clearly worse than late fusion with both", colour_missing_acc < late_acc - 0.3, "Without colour you can only get the shape right.")
    RESULTS["p1_colour_missing"] = float(colour_missing_acc)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# uniform = np.full((900, 3, 1), 1 / 3)
# pred = (uniform * p_shp[:, None, :]).reshape(900, 9).argmax(1)
# colour_missing_acc = round(float((pred[te] == y9[te]).mean()), 3)
# ```
#
# `colour_missing_acc` is 0.357. With no colour information the late-fusion model gets the shape right but has to pick one of three colours, so it is right about one time in three (a little more than 0.333 because the pictures are not exactly balanced).
#
# </details>

# %% [markdown]
# ### Step 1.6. Three applications: inputs, outputs, failures, oversight
#
# This last step is a writing exercise. A multimodal system is useful only if you know what goes in, what comes out, how it is likely to fail, and which human check keeps those failures from doing harm. The dictionary below has one entry per application. For **image captioning**, **visual question answering** (VQA: a picture and a question go in, an answer comes out) and **speech recognition** (sound in, text out) fill four fields: `inputs` (the modalities that go in), `output` (the modality that comes out), `failure` (a likely way to go wrong, in a short sentence) and `oversight` (one human-oversight step). A tip: think about a blind user who relies on the caption, a medical question about an X-ray, and a classroom where many students have regional accents.

# %%
def show_apps(apps):
    """Print the application dictionary as a table."""
    print(f"{'application':26s} {'inputs':24s} {'output':8s} failure | oversight")
    for name, d in apps.items():
        print(f"{name:26s} {str(d['inputs']):24s} {str(d['output']):8s} {d['failure']} | {d['oversight']}")

demo = {"machine translation": {"inputs": "text (French)", "output": "text",
                                "failure": "It translates an idiom word by word.",
                                "oversight": "A bilingual person checks official letters."}}
show_apps(demo)

# %% [markdown]
# **What you should see.** You should see one row: the demonstration entry "machine translation" with inputs "text (French)", output "text", and a failure and oversight sentence. It shows you the level of detail we expect: a short phrase for the modalities and a full sentence for the failure and the human check. In the exercise below you will fill in the three applications of the lesson and print the table of your own.

# %% [markdown]
# ### ✏️ Your turn: Exercise 1.6. Fill in the application table
#
# Your task: replace every `None` in the dictionary `apps` by a short text of your own (for `inputs` and `output` a few words, for `failure` and `oversight` one sentence), then print the table again. Why it matters: this is the habit you will need in Part 7: for every system, name the inputs and outputs, then name how it fails and who checks it.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
apps = {
    "image captioning":   {"inputs": None, "output": None, "failure": None, "oversight": None},
    "visual question answering": {"inputs": None, "output": None, "failure": None, "oversight": None},
    "speech recognition": {"inputs": None, "output": None, "failure": None, "oversight": None},
}
show_apps(apps)

# %%
filled = all(isinstance(v, str) and len(v.strip()) >= 4 for d in apps.values() for v in d.values())
if not filled:
    print("Fill in every field of apps above with a text, then run this cell again.")
else:
    check("every failure and oversight entry is a real sentence", all(len(d["failure"]) >= 15 and len(d["oversight"]) >= 15 for d in apps.values()), "Write at least a short sentence for failure and oversight.")
    check("no two failures are identical", len({d["failure"] for d in apps.values()}) == 3, "Each application has its own typical failure.")
    RESULTS["p1_apps"] = apps

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# apps = {
#     "image captioning": {"inputs": "picture", "output": "text",
#         "failure": "It names objects that are not there, or misses a detail that matters.",
#         "oversight": "A person reads captions of important pictures before they are published."},
#     "visual question answering": {"inputs": "picture + question (text)", "output": "text",
#         "failure": "It answers confidently from the language alone, without really looking.",
#         "oversight": "Medical or legal answers are always checked by a qualified person."},
#     "speech recognition": {"inputs": "audio", "output": "text",
#         "failure": "It makes more errors for some accents, noisy rooms and names.",
#         "oversight": "Transcripts of meetings are proofread, and the error rate is measured per accent."},
# }
# show_apps(apps)
# ```
#
# Your sentences will differ; what matters is that each one is specific. For example: image captioning takes a picture and gives text, it may name objects that are not there, and a person reads the captions of important pictures before they are published; visual question answering takes a picture and a question and gives text, it may answer from the words alone without really looking, and medical or legal answers are checked by a qualified person; speech recognition takes audio and gives text, it may make more errors for some accents and noisy rooms, and transcripts are proofread while the error rate is measured per accent. A good failure names a concrete wrong behaviour (a hallucinated object, an answer from the text alone, an accent that is understood worse) and a good oversight step names *who* checks *what* and *when*.
#
# </details>

# %% [markdown]
# ## Part 2. Patchify: from pixels to tokens (20 min)
#
# *Lesson section: "Vision Transformer".* A Transformer reads a *sequence* of tokens, so to let it look at a picture we must first turn the picture into a sequence. The idea of the Vision Transformer (ViT) is to cut the picture into a grid of small squares, called **patches**, flatten every square into a vector, and treat the vectors as the tokens (the "words") of the picture. In this part you do the cutting yourself, first by hand with NumPy, then with the two standard PyTorch tools (`unfold` and a strided `Conv2d`). Then you count: how many tokens, how many numbers per token, how many parameters in the first layer, and why this cutting is what makes a Transformer affordable on pictures at all. These are the ideas of the lesson's figure on the patch grid.

# %% [markdown]
# ### ✏️ Imagine first
#
# A colour picture of 224 by 224 pixels is cut into square patches. Before you run any code, write two predictions: how many patches (tokens) do you get with patches of 16 by 16 pixels, and how many with patches of 8 by 8 pixels? Halving the patch side is a big change: think about what happens to the number of patches. You will compare your answers with Step 2.3.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: patches of a 224 x 224 picture
my_tokens_p16 = None   # number of patches with 16 x 16 patches (an integer)
my_tokens_p8 = None    # number of patches with 8 x 8 patches (an integer)

# %% [markdown]
# ### Step 2.1. Patchify by hand with NumPy
#
# A picture is an array of shape (C, H, W): C channels, H rows, W columns. To cut it into square patches of side `p` we only have to *reshape* the array: split the rows into blocks of `p` and the columns into blocks of `p`, move the "block" axes to the front, and flatten each block. The function `patchify` below returns an array of shape (N, C*p*p), one row per patch, read from left to right and from top to bottom. `unpatchify` does the reverse. First look at a toy: a 4 by 4 picture with the numbers 0 to 15, cut into four patches of 2 by 2.

# %%
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

# %% [markdown]
# **What you should see.** You should see the toy picture (the numbers 0 to 15 in a 4 by 4 grid) and below it four rows, one per patch: `[0 1 4 5]` is the top-left patch (pixels 0, 1 of the first row and 4, 5 of the second row), `[2 3 6 7]` the top-right, `[8 9 12 13]` the bottom-left and `[10 11 14 15]` the bottom-right. Then three lines for the real picture: with p = 2 you get **64** patches of 4 numbers, with p = 4 **16** patches of 16 numbers, with p = 8 only **4** patches of 64 numbers, and the round trip through `unpatchify` is exact every time (the `assert` would stop the cell otherwise). The number of numbers never changes (64 times 4 = 16 times 16 = 4 times 64 = 256); patchifying only rearranges them into tokens.

# %% [markdown]
# **Tips and common mistakes**
#
# - `reshape` never changes the order of the numbers in memory, it only changes how we look at them. The `transpose` is what moves the numbers into patch order.
# - If H or W is not a multiple of `p`, the reshape raises an error. Real ViTs resize or crop the picture first (224 is divisible by 16, 14, 8 and 32).
# - The order inside a patch (channel, row, column) matches PyTorch's `unfold` and the weights of `Conv2d`, which is what Step 2.2 relies on.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.1. Which patch contains a pixel?
#
# Your task: in a 16 by 16 picture cut into patches of 4 by 4, the pixel in row 5 and column 9 (counting from 0) lies in one patch. Store in `patch_id` the index of that patch (counting patches in reading order from 0) and in `inside` the index of the pixel inside the flattened patch (row by row, from 0 to 15). Do it with arithmetic first, then confirm it with `patchify`. Why it matters: an image model must be able to go back and forth between "pixel coordinates" and "token number"; later, attention maps are drawn by exactly this conversion.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
patch_id = None   # index of the patch that contains the pixel (row 5, column 9)
inside = None     # index of that pixel inside the flattened 4 x 4 patch

# %%
if patch_id is None or inside is None:
    print("Fill in patch_id and inside above, then run this cell again.")
else:
    ramp = np.arange(256).reshape(1, 16, 16)               # every pixel holds its own index 16 * row + column
    rows = patchify(ramp, 4)
    r_, c_ = np.nonzero(rows == 16 * 5 + 9)
    check("patch_id is right", patch_id == int(r_[0]), "Patch row is row // 4, patch column is column // 4; patch_id = patch row * 4 + patch column.")
    check("inside is right", inside == int(c_[0]), "Row % 4 and column % 4 give the place inside the patch: inside = (row % 4) * 4 + (column % 4).")
    RESULTS["p2_pixel"] = (int(patch_id), int(inside))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# patch_id = (5 // 4) * 4 + (9 // 4)       # patch row 1, patch column 2 -> 6
# inside = (5 % 4) * 4 + (9 % 4)           # row 1, column 1 inside the patch -> 5
# ```
#
# `patch_id` is 6 and `inside` is 5. The pixel is in patch row 1 and patch column 2 (the 4 by 4 patches), so patch 1 * 4 + 2 = 6; inside the patch it is at row 1, column 1, so 1 * 4 + 1 = 5.
#
# </details>

# %% [markdown]
# ### Step 2.2. With PyTorch: unfold and a strided Conv2d
#
# Real ViT code does not write loops or reshapes; it uses two PyTorch tools. `F.unfold(x, kernel_size=p, stride=p)` cuts a batch of pictures into patches, exactly like `patchify`. And `nn.Conv2d(C, d, kernel_size=p, stride=p)` slides a window of side `p` with step `p`, so the windows do not overlap: it multiplies every patch by the same matrix and adds a bias, which is the same as "patchify, then a linear layer with d outputs". This is the **standard trick for the patch embedding** of a ViT. The block proves it: it copies the convolution's weights into a plain `nn.Linear` and checks that both give the same numbers.

# %%
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

# %% [markdown]
# **What you should see.** You should see: `unfold gives (2, 16, 192)` and `True` (PyTorch's `unfold` and your `patchify` agree: 16 patches per picture, 192 = 3 * 8 * 8 numbers each); then the convolution and the linear layer both give a tensor of shape (2, 16, 16) and are `equal: True` (to within 1e-5); finally the conv has **3088** parameters, which is 192 * 16 + 16: one weight per number of the patch and per output, plus one bias per output. So a strided convolution *is* "cut into patches and apply one shared linear layer to every patch". It is used in practice because it is faster than materialising the patches.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.2. Parameters of the real patch embedding
#
# Your task: ViT-Base cuts a 224 by 224 colour picture with patches of 16 by 16 and embeds each patch in d = 768 dimensions. Compute by formula the number of parameters of `nn.Conv2d(3, 768, kernel_size=16, stride=16)` (weights plus biases) and store it in `conv_params`; then confirm it by building the layer and counting with `count_params`. Why it matters: it is the first layer of the network and you will meet this number again in the lesson's table of ViT-Base parameters (590,592).
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
conv_params = None   # parameters of nn.Conv2d(3, 768, kernel_size=16, stride=16)

# %%
if conv_params is None:
    print("Fill in conv_params above, then run this cell again.")
else:
    real = count_params(nn.Conv2d(3, 768, kernel_size=16, stride=16))
    check("conv_params matches count_params", conv_params == real, "Weights: 768 filters of 3 * 16 * 16 numbers; biases: 768.")
    check("it is the lesson's 590,592", conv_params == 590592, "768 * 768 + 768.")
    RESULTS["p2_conv_params"] = int(conv_params)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# conv_params = 3 * 16 * 16 * 768 + 768        # 768 numbers per patch times 768 outputs, plus 768 biases = 590,592
# print(conv_params, count_params(nn.Conv2d(3, 768, kernel_size=16, stride=16)))
# ```
#
# `conv_params` is 590,592 (768 * 768 + 768). A patch has 3 * 16 * 16 = 768 numbers and there are 768 outputs, so the weights are 768 * 768 = 589,824, plus 768 biases.
#
# </details>

# %% [markdown]
# ### Step 2.3. The counting formulas
#
# Now we count. For a picture of H by W pixels with C channels, patches of side `p` and embedding size `d`: the number of patches is (H/p) times (W/p); each patch is a vector of p times p times C numbers; the sequence the Transformer reads has one more token, the special **CLS** token, so its length is N + 1; and the patch embedding has (p*p*C)*d weights plus d biases. The function `vit_counts` returns all four numbers, and the block prints a table for five settings, including the lesson's ViT-Base (224, patch 16) and a tiny setting that we use from Part 4 on (16 by 16 pictures, patch 4, one channel, d = 32).

# %%
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

# %% [markdown]
# **What you should see.** You should see a table with five rows (the last line reminds you of your Imagine-first guesses, and prints `None` if you did not fill them in):
#
# - 224, p = 16: **196** patches, 768 numbers per patch, sequence **197**, 590,592 embedding parameters (this is the ViT-Base of the lesson);
# - 224, p = 32: **49** patches of 3,072 numbers, sequence 50, 2,360,064 parameters;
# - 224, p = 8: **784** patches of 192 numbers, sequence 785, 148,224 parameters;
# - 384, p = 16: 576 patches, sequence 577, 590,592 parameters;
# - 16 by 16 grey, p = 4, d = 32: 16 patches of 16 numbers, sequence 17, 544 parameters (the model of Part 4).
#
# The answers to the Imagine-first cell are 196 and 784: halving the patch side gives four times as many patches (14 by 14 against 28 by 28), but the vector of each patch gets four times shorter. Notice that the embedding has the **same** 590,592 parameters for 224 and 384 pixels: the parameter count depends on the patch, not on the size of the picture; only the number of tokens grows.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.3. ViT-Large with 14 by 14 patches
#
# Your task: ViT-Large/14 reads 224 by 224 pictures with patches of 14 by 14 and an embedding size of d = 1024. Use `vit_counts` to compute its numbers and store the result in `large14`. Which of the four numbers surprises you? Why it matters: the notation "ViT-L/14" in papers means exactly this: the model size L and the patch side 14. You should be able to turn it into token counts in your head.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
large14 = None   # the dictionary that vit_counts returns for a 224 x 224 colour picture, p = 14, d = 1024

# %%
if large14 is None:
    print("Fill in large14 above, then run this cell again.")
else:
    check("patches are right", large14["patches"] == 256, "224 / 14 = 16 patches per side, 16 * 16 in total.")
    check("patch_len is right", large14["patch_len"] == 14 * 14 * 3, "p * p * 3 numbers per patch.")
    check("seq_len and embed_params are right", large14["seq_len"] == 257 and large14["embed_params"] == 588 * 1024 + 1024, "Do not forget the CLS token and the 1024 biases.")
    RESULTS["p2_large14"] = dict(large14)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# large14 = vit_counts(224, 224, 14, 3, 1024)
# print(large14)
# ```
#
# `large14` is `{'patches': 256, 'patch_len': 588, 'seq_len': 257, 'embed_params': 603136}`. The surprise is usually that ViT-L/14 has more tokens (256) than ViT-B/16 (196), because the patches are smaller, so a "bigger" model is also more expensive in attention.
#
# </details>

# %% [markdown]
# ### Step 2.4. The cost of tokens: why patches?
#
# Why not use every pixel as a token? Because of **self-attention**: each head computes one score for every pair of tokens, so the number of scores is the *square* of the sequence length. The block counts the scores per head for a 224 by 224 picture twice: with 16 by 16 patches (197 tokens including CLS) and with one token per pixel (50,176 tokens), and then plots the score count against the patch size on a logarithmic axis.

# %%
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

# %% [markdown]
# **What you should see.** You should see: with patches of 16 there are **38,809** attention scores per head (197 times 197); with one token per pixel there are **2,517,630,976** (50,176 times 50,176); the ratio is **64,872 times** more scores. The figure shows the score count against the patch side on log-log axes: a falling line, steep at the left. Every doubling of the patch side divides the number of tokens by four and the number of scores by about sixteen. At p = 1 (one token per pixel) the picture needs about 2.5 billion scores per head and per layer, far too many; at p = 16 only about forty thousand. The patch is not a detail of the architecture: it is what makes a Transformer on pictures affordable.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.4. Doubling the picture side
#
# Your task: a 448 by 448 picture with 16 by 16 patches has how many attention scores per head (with the CLS token)? Store it in `scores_448`. By what factor is that larger than for the 224 by 224 picture (`scores_patch`)? Store the factor, rounded to one decimal, in `growth`. Why it matters: doubling the side of the picture quadruples the number of tokens and multiplies the attention cost by about sixteen: this is why high-resolution input is expensive and why models tile or downscale it.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
scores_448 = None   # attention scores per head for a 448 x 448 picture, patches of 16, with CLS
growth = None       # scores_448 / scores_patch, rounded to 1 decimal

# %%
if scores_448 is None or growth is None:
    print("Fill in scores_448 and growth above, then run this cell again.")
else:
    check("scores_448 is right", scores_448 == (28 * 28 + 1) ** 2, "(448 / 16) = 28 patches per side; the sequence is 28 * 28 + 1 tokens; square it.")
    check("growth is right", abs(growth - round((28 * 28 + 1) ** 2 / scores_patch, 1)) < 0.06, "scores_448 / scores_patch.")
    RESULTS["p2_scores_448"] = (int(scores_448), float(growth))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# scores_448 = vit_counts(448, 448, 16, 3, 768)["seq_len"] ** 2       # 785 ** 2
# growth = round(scores_448 / scores_patch, 1)
# ```
#
# `scores_448` is 616,225 (785 times 785) and `growth` is 15.9. The picture has 4 times as many patches, and the number of attention scores grows by almost 16, the square of 4.
#
# </details>

# %% [markdown]
# ### Step 2.5. Patch size on real pictures: empty patches
#
# Small patches give a finer description but cost many tokens, and in our line pictures most tokens are blank. We make 200 *clean* pictures (`noise=0`, so that blank patches are exactly zero), cut them with p = 2, 4 and 8, and measure the mean ink (sum of the pixel values) per patch and the share of patches that are completely empty. The figure shows one picture with its 4 by 4 grid and the empty patches shaded.

# %%
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
ax[0].set_title(f"{SHAPE_NAMES[yclean[0]]}, empty patches shaded", fontsize=9)
ax[1].bar([str(p) for p in empty_share], [100 * v for v in empty_share.values()]); ax[1].set_xlabel("patch side p"); ax[1].set_ylabel("% of patches that are empty")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** You should see a table with the three patch sizes:
#
# - p = 2: 64 tokens of 4 numbers, mean ink per patch 0.237, **87.5 percent** of the patches empty;
# - p = 4: 16 tokens of 16 numbers, mean ink 0.950, **72.5 percent** empty;
# - p = 8: 4 tokens of 64 numbers, mean ink 3.800, **35.0 percent** empty.
#
# The figure has the first picture with its 4 by 4 grid (the empty patches are shaded blue) and a bar chart of the empty share. In these line pictures most of the area is blank, so with small patches most tokens carry nothing, and each of them still takes part in the attention. With big patches there are few empty tokens, but each token mixes a lot of detail into a single vector. That is the trade-off of the patch size: resolution of detail against the number of tokens (compute). Part 5 measures the effect on the accuracy.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.5. The limit p = 1
#
# Your task: with p = 1 every pixel is a patch. Compute the share of exactly-zero pixels in `Xclean` and store it in `empty_p1` (a number between 0 and 1, 3 decimals). Is it larger or smaller than the share for p = 2? Why it matters: a token that carries no ink still costs attention and memory; ViTs for large pictures with big empty areas (documents, sky, X-rays) waste a lot of tokens that way.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
empty_p1 = None   # share of pixels equal to 0 in Xclean (3 decimals)

# %%
if empty_p1 is None:
    print("Fill in empty_p1 above, then run this cell again.")
else:
    check("empty_p1 is right", abs(empty_p1 - round(float((Xclean == 0).mean()), 3)) < 0.0015, "The mean of the boolean array (Xclean == 0).")
    check("it is larger than the p = 2 share", empty_p1 > empty_share[2], "The smaller the patch, the more patches are empty.")
    RESULTS["p2_empty"] = {1: float(empty_p1), 2: empty_share[2], 4: empty_share[4], 8: empty_share[8]}

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# empty_p1 = round(float((Xclean == 0).mean()), 3)
# ```
#
# `empty_p1` is 0.941: 94.1 percent of the pixels are exactly zero, more than the 87.5 percent of empty 2 by 2 patches. The smaller the patch, the more of them are empty, until at the limit nearly all tokens are blank.
#
# </details>

# %% [markdown]
# ### Step 2.6. Imagine and write: a patch is like a word
#
# The lesson says a ViT patch plays the role of a *word*. Write two short sentences: one that says **what is the same** (what the Transformer does with a patch and with a word) and one that says **where the analogy fails** (think about what a word is made of and what a patch is made of, about the dictionary of words against the endless variety of patches, and about what a neighbouring patch means). There is no code to run: store your two sentences in the dictionary `analogy`.

# %% [markdown]
# ### ✏️ Your turn: Exercise 2.6. The word-patch analogy
#
# Your task: fill in `analogy["alike"]` and `analogy["differs"]` with one sentence each. Why it matters: writing the analogy and its limits is the best way to remember what a patch embedding is: a learned vector, with no fixed dictionary behind it.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
analogy = {"alike": None, "differs": None}   # two sentences of your own

# %%
if any(v is None for v in analogy.values()):
    print("Fill in both sentences of analogy above, then run this cell again.")
else:
    check("both are real sentences", all(isinstance(v, str) and len(v.split()) >= 8 for v in analogy.values()), "Write at least eight words for each.")
    check("they are different", analogy["alike"] != analogy["differs"], "One says what is the same, the other where it fails.")
    RESULTS["p2_analogy"] = dict(analogy)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# analogy = {
#     "alike": "Both are turned into a vector by a learned layer, and the Transformer then treats all vectors in the same way, with attention between them.",
#     "differs": "A word comes from a finite dictionary and carries meaning by itself, while a patch is a piece of continuous pixels that can be cut anywhere and means little alone.",
# }
# ```
#
# For example: "Both are turned into a vector by a learned layer, and the Transformer then treats all the vectors in the same way, with attention between them" and "A word comes from a finite dictionary and carries meaning by itself, while a patch is a piece of continuous pixels that can be cut anywhere and means little alone." The key points: a patch and a word are both embedded into a vector by a learned layer and then handled by attention in the same way; but words come from a finite dictionary with meaning of their own, whereas patches are cut from a continuous image, can be cut anywhere, and mean little alone.
#
# </details>

# %% [markdown]
# ## Part 3. Why position embeddings? (25 min)
#
# *Lesson section: "Positions".* Once a picture is a set of patch tokens, a worry appears: self-attention treats its tokens as a *set*, not as a *sequence*. If you swap two tokens, the attention layer does not notice where they came from. For text, Week 8 solved this by adding a position vector to each word. For a picture the problem is worse, because the answer to many questions depends on **where** things are: is the cat left of the dog, is the pointer above or below the dial? In this part you prove the problem with NumPy, see it on a real picture, then fix it with two kinds of position embeddings (learned and sinusoidal 2D), look at what they encode, and learn how to reuse them for a bigger picture. These are the ideas of the lesson's figure on shuffled patches.

# %% [markdown]
# ### ✏️ Imagine first
#
# A self-attention layer receives six patch tokens and returns six output vectors, one per token. Now we shuffle the six tokens and run the same layer again. Before you run any code, write your prediction: will the new outputs be (a) the same six vectors but in the shuffled order, (b) six completely different vectors, or (c) the same six vectors in the original order? Write "reordered", "different" or "same order". You will compare with Step 3.1.
#
# Write your guesses in the starter cell below (replace each `None`), then run it. Do not look ahead: you compare with the real numbers later in this part.

# %%
# Imagine first: shuffle the tokens, what does self-attention return?
my_prediction = None   # "reordered", "different" or "same order"

# %% [markdown]
# ### Step 3.1. Permutation equivariance, by hand
#
# Here is a tiny self-attention layer in NumPy, exactly the one of Week 8 with a single head and with the queries, keys and values all equal to `X W` (to keep the matrices small). The six tokens have four numbers each, and both matrices are written out so that you can follow them with a pencil. We apply the layer to `X` and to a shuffled copy `X[perm]` and compare. The word for what you will find is **permutation equivariance**: if the input rows are permuted, the output rows are permuted in the same way.

# %%
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

# %% [markdown]
# **What you should see.** You should see two printed matrices of 6 rows and 4 columns that are **row-for-row the same numbers** (for example the first row is `0.326 0.273 0.151 0.052`), then `True` for "rows of the original output, reordered, equal the new output" and `False` for "same output row by row". So the answer to the Imagine-first cell is **"reordered"**: the layer returns the same six vectors, in the order in which you gave the tokens. The layer cannot tell which token came first; if the order carries meaning (a sentence, the layout of a picture), the layer is blind to it.

# %% [markdown]
# **Why it happens.** The attention score between two tokens depends only on the two tokens themselves, never on their places. So when the rows of `X` are permuted, the matrix of scores is permuted in rows and columns, the softmax acts row by row, and the weighted sum gives the same vectors in the new order. Nothing in the layer can tell the first token from the last. A ViT that reads patches with no position information therefore sees a *bag* of patches.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.1. Try your own shuffle
#
# Your task: choose any permutation of the numbers 0 to 5 as `my_perm` (a list with each number once) and store in `equivariant` whether `out[my_perm]` equals the output of the layer on `X[my_perm]` (`True` or `False`; use `np.allclose`). Why it matters: the claim must hold for every permutation, not just the one in the text; testing a claim on several cases is how you gain trust in it.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
my_perm = [5, 4, 3, 2, 1, 0]   # change it if you like: a list with the numbers 0..5 once each
equivariant = None             # True or False

# %%
if equivariant is None:
    print("Fill in equivariant above, then run this cell again.")
else:
    check("my_perm is a real permutation", sorted(my_perm) == list(range(6)), "Each of 0..5 exactly once.")
    check("equivariant is True", bool(equivariant) is True, "Compute np.allclose(out[my_perm], self_attention(X[my_perm], W)[0]).")
    RESULTS["p3_equivariant"] = bool(equivariant)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# equivariant = bool(np.allclose(out[my_perm], self_attention(X[my_perm], W)[0]))
# ```
#
# `equivariant` is `True` for every permutation. The layer only reorders its outputs in the same way as its inputs.
#
# </details>

# %% [markdown]
# ### Step 3.2. The picture version: a bag of patches
#
# Now the same thing on pictures. The task of Part 5, `make_left_right`, is "is the plus sign in the **left half** of the picture?": each picture has one plus sign (the target, arms of 2 pixels) and one filled 3 by 3 square (a distractor). We cut a picture into 16 patches of 4 by 4, shuffle the patches (`shuffled = unpatchify(patchify(pic)[order])`) and look at the result. To know where the plus sign goes, the helper `find_plus` first finds it in the original (ink at the centre and 2 pixels away on the four sides, but not at the four diagonal points, unlike the square), and we shuffle a mask of the plus with the same order of patches; the new label is 1 if the centre of the moved plus pixels is in the left half. The block picks a picture where the shuffle moves the plus sign to the other half, shows both pictures, and checks that the two **bags of patches are identical**: if we sort the patch vectors, the sorted lists are the same, although the labels differ.

# %%
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

# %% [markdown]
# **What you should see.** You should see:
#
# - `find_plus agrees with the true label on 200 of 200 pictures`;
# - picture 0: the plus sign is at row 12, column 2 (left half, label 1); after the shuffle of the 16 patches the plus pixels have moved to the right half, so the label is **0**;
# - `sorted bags of patch vectors identical: True`;
#
# and a figure with the original and the shuffled picture side by side: two pictures that look different, with the labels 1 and 0. The pieces are the same, the answer is not. A network that is given only the bag of patches, without any information about where each patch came from, would have to give the same answer for both pictures, so it is wrong for at least one of them. This is why a ViT without positions can only guess on the left-or-right task.

# %%
def flip_study(X, y, p=4, seed=1):
    """Shuffle the patches of every picture once. Return the share of pictures whose label (where the plus sign went) changed."""
    rng = np.random.RandomState(seed); n_patches = (16 // p) ** 2; flipped = 0
    for img, lab in zip(X, y):
        flipped += int(moved_label(img, p, rng.permutation(n_patches)) != lab)
    return flipped / len(X)

print(f"p = 4: the label changed in {flip_study(XLR, yLR, p=4):.1%} of the 200 shuffled pictures")

# %% [markdown]
# **What you should see.** You should see: `p = 4: the label changed in 50.5% of the 200 shuffled pictures`. A random shuffle sends the plus sign to either half with about the same chance, so for a position-blind model about half of the shuffled copies of a picture would need the opposite answer.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.2. Bigger patches, smaller patches
#
# Your task: repeat the flip study with patches of 8 by 8 (`p=8`, only four patches) and of 2 by 2 (`p=2`, 64 patches). Store the shares of changed labels in `flip8` and `flip2` (3 decimals). Why it matters: whatever the patch size, a model that only sees the bag of patches cannot tell the original from the shuffled picture, yet in a large share of the cases the right answer is different. The best such a model can do is guess.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
flip8 = None   # share of pictures whose label changes when the 8 x 8 patches are shuffled (3 decimals)
flip2 = None   # the same for 2 x 2 patches

# %%
if flip8 is None or flip2 is None:
    print("Fill in flip8 and flip2 above, then run this cell again.")
else:
    check("flip8 is right", abs(flip8 - round(flip_study(XLR, yLR, p=8), 3)) < 0.0015, "Call flip_study(XLR, yLR, p=8).")
    check("flip2 is right", abs(flip2 - round(flip_study(XLR, yLR, p=2), 3)) < 0.0015, "Call flip_study(XLR, yLR, p=2).")
    check("a random shuffle changes the answer in roughly half of the cases", 0.3 < flip8 < 0.7 and 0.3 < flip2 < 0.7, "A shuffle sends the plus to either half with about equal chance.")
    RESULTS["p3_flip"] = (float(flip8), float(flip2))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# flip8 = round(flip_study(XLR, yLR, p=8), 3)
# flip2 = round(flip_study(XLR, yLR, p=2), 3)
# ```
#
# `flip8` is 0.48 and `flip2` is 0.46. With four large patches or with 64 small ones, about half of the labels change (48 and 46 percent): the same message as before.
#
# </details>

# %% [markdown]
# ### Step 3.3. Add positions: learned and sinusoidal 2D
#
# The fix is to add to every patch embedding a vector that tells *where* the patch was. There are two common ways. A **learned** table: a trainable array of shape (N, d), one row per position, that the network adjusts like any other weight (BERT and the original ViT do this). A **sinusoidal 2D** embedding: no training, built from the formula of Week 8, `sin(pos / 10000^(2i/d))` and `cos(...)`, in which half of the d numbers encode the **row** of the patch and the other half the **column**. The function `sincos_2d(grid, d)` below returns an array of shape (grid*grid, d). We print the vectors for a 4 by 4 grid with d = 8 and check that all 16 are different. (Week 8 interleaves sine and cosine; here the sines of one axis come first and then its cosines: the information is the same.)

# %%
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

# %% [markdown]
# **What you should see.** You should see: the learned table has shape (16, 8) and 128 trainable numbers, the sinusoidal table has shape (16, 8) and is printed as a table of 16 rows: the first four numbers of a row encode the row of the patch (two sines then two cosines) and the last four the column. The first row (the patch at the top-left) is `0 0 1 1 0 0 1 1`: sines of 0 and cosines of 0. The second row (row 0, column 1) is `0 0 1 1 0.84 0.01 0.54 1`: the same row code and a different column code. The last line says **16 of 16** position vectors are different, so every patch has its own label.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.3. The position of the first patch
#
# Your task: build `sincos_2d(14, 768)`, the table for ViT-Base without the CLS token, and store its shape in `pe_shape` (a tuple) and the sum of the numbers of the vector of the first patch (row 0, column 0) in `sum00`. Predict the sum first: what are the sine and the cosine of 0? Why it matters: the first patch is at the origin, so all its sines are 0 and all its cosines are 1: a sanity check that you understand the formula.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
pe_shape = None   # shape of sincos_2d(14, 768)
sum00 = None      # sum of the 768 numbers of the position vector of the first patch

# %%
if pe_shape is None or sum00 is None:
    print("Fill in pe_shape and sum00 above, then run this cell again.")
else:
    _pe = sincos_2d(14, 768)
    check("pe_shape is right", tuple(pe_shape) == _pe.shape == (196, 768), "A 14 x 14 grid has 196 positions.")
    check("sum00 is right", abs(sum00 - _pe[0].sum()) < 1e-6, "Half of the numbers are sin(0) = 0 and half are cos(0) = 1.")
    RESULTS["p3_pe"] = (tuple(pe_shape), float(sum00))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pe = sincos_2d(14, 768)
# pe_shape = pe.shape
# sum00 = float(pe[0].sum())
# ```
#
# `pe_shape` is (196, 768) and `sum00` is 384.0: half of the 768 numbers are cos(0) = 1 and the other half sin(0) = 0.
#
# </details>

# %% [markdown]
# ### Step 3.4. Positions break the symmetry
#
# Repeat Step 3.1, but add a position vector to each token *before* the attention layer. The position vectors belong to the **slots**: slot 0 always gets the vector for position 0, whatever token is put there. So when we shuffle the tokens, the positions do not move with them. If the layer is now to give the same rows in a different order, positions would have to follow the tokens; they do not, and the outputs are no longer permutations of each other. We use `sincos_1d` for six positions with d = 4.

# %%
Ppos = sincos_1d(np.arange(6), 4)                      # six position vectors of 4 numbers each
out_p, _ = self_attention(X + Ppos, W)
out_p_shuf, _ = self_attention(X[perm] + Ppos, W)       # tokens shuffled, positions stay in their slots
print("without positions: largest difference between out[perm] and the shuffled output:", np.abs(out[perm] - out_shuf).max())
print("with positions:    largest difference between out[perm] and the shuffled output:", round(float(np.abs(out_p[perm] - out_p_shuf).max()), 3))
print("equal with positions:", np.allclose(out_p[perm], out_p_shuf))
assert not np.allclose(out_p[perm], out_p_shuf)

# %% [markdown]
# **What you should see.** You should see: without positions the largest difference between `out[perm]` and the shuffled output is about 1e-16 (rounding error: it is zero); with positions it is **0.194**, and `equal with positions: False`. The positions stay in their slots when the tokens move, so after the shuffle each token meets a different position vector and the layer gives different results. The layer is now **not** permutation equivariant, and it can use the place of a token.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.4. Measure the broken symmetry
#
# Your task: store in `max_diff_pos` the largest absolute difference between `out_p[perm]` and `out_p_shuf`, rounded to 3 decimals, and in `breaks` whether it is bigger than 0.01 (`True` or `False`). Why it matters: equivariance is an exact property, so even a small difference means it is gone; but how large the difference is tells you how strongly the positions change the result.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
max_diff_pos = None   # largest absolute difference (3 decimals)
breaks = None         # True if it is larger than 0.01

# %%
if max_diff_pos is None or breaks is None:
    print("Fill in max_diff_pos and breaks above, then run this cell again.")
else:
    check("max_diff_pos is right", abs(max_diff_pos - round(float(np.abs(out_p[perm] - out_p_shuf).max()), 3)) < 0.0015, "np.abs(out_p[perm] - out_p_shuf).max()")
    check("breaks is right", bool(breaks) is True, "The positions make the difference clearly larger than 0.01.")
    RESULTS["p3_break"] = float(max_diff_pos)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# max_diff_pos = round(float(np.abs(out_p[perm] - out_p_shuf).max()), 3)
# breaks = max_diff_pos > 0.01
# ```
#
# `max_diff_pos` is 0.194 and `breaks` is `True`. The outputs are clearly not permutations of each other any more.
#
# </details>

# %% [markdown]
# ### Step 3.5. Similarity of positions
#
# What do the 2D sinusoidal vectors say about the geometry of the grid? We take an 8 by 8 grid with d = 64, normalise every position vector to length 1, and compute the cosine similarity between position (0, 0) (the top-left corner) and every other position. The result is an 8 by 8 map that we draw as a heat map with the numbers written in.

# %%
Pe = sincos_2d(8, 64)
Pn = Pe / np.linalg.norm(Pe, axis=1, keepdims=True)
sim = (Pn[0] @ Pn.T).reshape(8, 8)                       # cosine similarity of every position with (0, 0)
plt.figure(figsize=(5.2, 4.4)); plt.imshow(sim, cmap="viridis")
for r in range(8):
    for c in range(8): plt.text(c, r, f"{sim[r, c]:.2f}", ha="center", va="center", color="w", fontsize=7)
plt.colorbar(label="cosine similarity with position (0, 0)"); plt.title("2D sinusoidal positions, 8 x 8 grid, d = 64"); plt.tight_layout(); plt.show()
dist = np.add.outer(np.arange(8), np.arange(8))           # Manhattan distance from (0, 0) = row + column
for k in (0, 1, 2, 4, 8, 14): print(f"Manhattan distance {k:2d}: mean similarity {sim[dist == k].mean():.3f}")

# %% [markdown]
# **What you should see.** You should see a heat map of 8 by 8 numbers that is brightest (1.00) at the top-left corner and fades smoothly towards the opposite corner. The table lists the mean similarity by Manhattan distance (rows plus columns) from the corner: 1.000 at distance 0, **0.979** at distance 1, 0.938 at 2, 0.862 at 4, 0.786 at 8 and **0.714** at 14 (the far corner). So neighbouring positions have similar vectors and far positions are less alike: a smooth geometry that costs no training. The map is not a perfect distance: for example position (0, 7) is more similar (0.857) than the far corner although both are 7 steps away in one direction; high-frequency components make some far positions resemble the corner again.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.5. Near and far
#
# Your task: store in `sim_near` the cosine similarity between position (0, 0) and its neighbour (0, 1), and in `sim_far` the similarity with the opposite corner (7, 7), both rounded to 3 decimals. Is the neighbour more similar? Why it matters: if close positions have similar vectors, the network gets a hint of "closeness" for free; but the hint is not perfect, as the map shows.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
sim_near = None   # similarity of position (0, 0) with (0, 1)
sim_far = None    # similarity of position (0, 0) with (7, 7)

# %%
if sim_near is None or sim_far is None:
    print("Fill in sim_near and sim_far above, then run this cell again.")
else:
    check("sim_near is right", abs(sim_near - round(float(sim[0, 1]), 3)) < 0.0015, "The entry in row 0, column 1 of sim.")
    check("sim_far is right", abs(sim_far - round(float(sim[7, 7]), 3)) < 0.0015, "The entry in row 7, column 7 of sim.")
    check("the neighbour is more similar than the far corner", sim_near > sim_far, "Check which value is larger.")
    RESULTS["p3_sim"] = (float(sim_near), float(sim_far))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# sim_near = round(float(sim[0, 1]), 3)
# sim_far = round(float(sim[7, 7]), 3)
# ```
#
# `sim_near` is 0.979 and `sim_far` is 0.714. The neighbour is clearly more similar than the opposite corner.
#
# </details>

# %% [markdown]
# ### Step 3.6. Another picture size: interpolate the table
#
# A learned position table has a fixed number of rows. Suppose we trained a ViT on a 4 by 4 grid of patches and now want to use it on a picture with a 6 by 6 grid (a bigger picture, or smaller patches). The sum of the 36 patch embeddings and the 16 positions fails with a shape error. The common remedy is to treat the table as a small picture with d channels and **resize it with bilinear interpolation** (`F.interpolate`), then fine-tune. The block shows the error, the repair, and the numbers.

# %%
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

# %% [markdown]
# **What you should see.** You should see: the first sum fails with a shape error (`The size of tensor a (36) must match the size of tensor b (16)...`); the resized table has shape (1, 36, 32) and the sum works, giving a result of shape (1, 36, 32). The old table has 16 * 32 = **512** numbers, the new one 36 * 32 = **1152**. Interpolation does not add trainable parameters by itself: it computes the 1,152 numbers from the 512; they are a good starting point that the network then adjusts when it is trained further (fine-tuned) at the new resolution.

# %% [markdown]
# ### ✏️ Your turn: Exercise 3.6. ViT-Base at 384 pixels
#
# Your task: the position table of ViT-Base for 224-pixel pictures has a 14 by 14 grid (ignoring the CLS token). For a 384-pixel picture with the same patches of 16 the grid is 24 by 24. Use `resize_pos` on a table of zeros to get the new table; store its shape in `new_shape` and its number of entries in `new_numbers`. Why it matters: this is what libraries do when you load a ViT trained at 224 and feed it a larger image; knowing that the positions are resized, not re-learned, explains why accuracy at a new size usually needs a short fine-tuning.
#
# Replace the `None` values (or write the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
new_shape = None     # shape of the resized table
new_numbers = None   # how many numbers it holds

# %%
if new_shape is None or new_numbers is None:
    print("Fill in new_shape and new_numbers above, then run this cell again.")
else:
    check("new_shape is right", tuple(new_shape) == (1, 576, 768), "24 * 24 = 576 positions of 768 numbers.")
    check("new_numbers is right", new_numbers == 576 * 768, "576 * 768 = 442,368 (the old table had 196 * 768 = 150,528).")
    RESULTS["p3_resize"] = (tuple(new_shape), int(new_numbers))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# t = resize_pos(torch.zeros(1, 196, 768), 14, 24)
# new_shape, new_numbers = tuple(t.shape), t.numel()
# ```
#
# `new_shape` is (1, 576, 768) and `new_numbers` is 442,368 (the old table had 196 * 768 = 150,528). ViT-Base at 384 pixels has 576 patches, almost three times as many tokens as at 224.
#
# </details>

# %% [markdown]
# ## Part 4. Build a tiny Vision Transformer (30 min)
#
# *Lesson section: "Vision Transformer".* In Part 2 you cut pictures into patches and counted tokens, and in Part 3 you saw why a Transformer needs position embeddings. Now you put the pieces together and build a complete Vision Transformer of your own, small enough to train in a few seconds on a CPU. It has five parts: a patch embedding that turns each patch into a vector, a learned CLS token that collects the answer, a table of learned position embeddings, a stack of Transformer encoder layers, and a linear head. ==You will write the patch embedding and the forward pass yourself, count every parameter by formula and with the computer, run two sanity checks that every new model should pass, train the model on the five line shapes of Week 4, compare it honestly with a small CNN, and finally look inside to see where the CLS token looks.== A tiny ViT trained from scratch on a few hundred pictures is a weak learner by design (it has no built-in idea that nearby pixels belong together), so this part reports what you measure, not what a big pretrained ViT would do. All numbers are for `SEED = 42`.
#
# **✏️ Imagine first.** Picture the model as a filing office. A picture arrives and is cut into 16 small squares (patches). Each square is copied onto a card with 32 numbers (the patch embedding). One extra blank card (the CLS token) is added to the pile and every card gets a sticker with its place number (the position embedding). Then two rounds of clerks (the encoder layers) pass information between the cards, and in the end only the extra card is read to decide which of 5 shapes the picture shows. Every number printed on a card, a sticker or a clerk's rulebook is a parameter. Before you run any code, guess how many numbers the whole office holds: a few thousand, a few tens of thousands, or hundreds of thousands? Write your guess in the cell below. You compare it with Step 4.3.

# %%
# Imagine first: how many numbers does a tiny ViT hold? (d = 32, 2 layers, 4 heads, patch 4, 16 x 16 pictures, 5 classes)
my_guess_params = None   # write a number, for example 5000, 30000 or 300000

# %% [markdown]
# ### Step 4.1. From patches to a sequence of tokens: PatchEmbed
#
# A ViT starts by cutting the picture into patches and mapping every patch with the same linear layer to a vector of d numbers. Writing that as loops is slow, so everybody uses a trick you met in Part 2: a convolution whose kernel is as large as the patch and whose stride equals the patch size looks at each patch exactly once, and its weights are the weights of the linear layer. The output of the convolution is a grid of d-vectors, and two reshaping operations turn the grid into a sequence. The cell below follows the shape of a batch of 8 pictures of 16 x 16 pixels through the three operations.

# %%
x = torch.randn(8, 1, 16, 16)                                              # a batch of 8 greyscale pictures
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=4)  # one 4 x 4 window per patch, no overlap
y = conv(x)
print("after the Conv2d       :", tuple(y.shape), " (batch, d, rows, columns of the patch grid)")
y = y.flatten(2)
print("after flatten(2)       :", tuple(y.shape), " (batch, d, N)")
y = y.transpose(1, 2)
print("after transpose(1, 2)  :", tuple(y.shape), " (batch, N, d): a sequence of 16 tokens")
print("parameters of the Conv2d:", count_params(conv), "= 1*4*4*32 weights + 32 biases")

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.1
#
# Write the class `MyPatchEmbed` yourself: a `Conv2d` with `kernel_size=patch` and `stride=patch`, and a forward method that applies it, flattens the two grid axes and moves the d axis to the end. Why it matters: this class is the entry door of every Vision Transformer. If you can write it, you know exactly what a patch token is.

# %%
class MyPatchEmbed(nn.Module):
    def __init__(self, img=16, patch=4, in_ch=1, d=32):
        super().__init__()
        self.n_patches = (img // patch) ** 2
        self.proj = None      # YOUR CODE: a Conv2d from in_ch channels to d channels with kernel_size=patch and stride=patch
    def forward(self, x):
        return None           # YOUR CODE: apply self.proj, flatten the two grid axes (flatten(2)), put d last (transpose(1, 2))

# %%
pe = MyPatchEmbed()
if pe.proj is None:
    print("Fill in MyPatchEmbed above, then run this cell again.")
else:
    out = pe(torch.randn(8, 1, 16, 16))
    check("output shape is (8, 16, 32)", out is not None and tuple(out.shape) == (8, 16, 32), "flatten(2) then transpose(1, 2) gives (batch, N, d)")
    check("544 parameters", count_params(pe) == 544, "Conv2d(1, 32, kernel_size=4, stride=4) has 1*4*4*32 weights and 32 biases")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# class MyPatchEmbed(nn.Module):
#     def __init__(self, img=16, patch=4, in_ch=1, d=32):
#         super().__init__()
#         self.n_patches = (img // patch) ** 2
#         self.proj = nn.Conv2d(in_ch, d, kernel_size=patch, stride=patch)
#     def forward(self, x):
#         return self.proj(x).flatten(2).transpose(1, 2)
# ```
#
# `Conv2d(1, 32, 4, 4)` gives a tensor of shape (batch, 32, 4, 4); `flatten(2)` merges the last two axes into 16; `transpose(1, 2)` swaps the d axis and the token axis. The check also tests the 544 parameters. Below the exercise we define `PatchEmbed`, which is the same class, and use it from now on.
#
# </details>

# %%
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

# %% [markdown]
# **What you should see.**
#
# You see the shapes (8, 32, 4, 4) after the convolution (a 4 x 4 grid of patches, 32 numbers per patch), (8, 32, 16) after `flatten(2)` (the 16 grid cells in a row) and (8, 16, 32) after `transpose(1, 2)`: a batch of 8 sequences of 16 tokens with 32 numbers each. The convolution has 544 parameters: 1 x 4 x 4 x 32 = 512 weights plus 32 biases. **This is the whole patch embedding.** Everything after it treats the 16 vectors like the 16 word vectors of a sentence.
#
# Tips and common mistakes:
#
# - `stride=patch` is what makes the patches do not overlap. With a stride of 1 you would get one vector per pixel position, and a sequence far too long.
# - The batch axis never changes. When a shape surprises you, print it after every operation, as the cell does.
#
# Try also (optional):
#
# - Change the patch size to 8 and the number of output channels to 64. Predict the shape after `transpose(1, 2)` before you run the cell. We get (8, 4, 64).

# %% [markdown]
# ### Step 4.2. The TinyViT: patch tokens, a CLS token, positions, a Transformer encoder, a head
#
# Now the whole model. The cell below builds each piece by hand and prints the shape after every operation, so you see what the forward pass has to do. The CLS token is one learned vector that is copied in front of every picture's sequence; after the encoder, only its output goes to the classification head, so the encoder is forced to collect everything the head needs in that one vector. The position table has 17 rows, one for each patch and one for the CLS token. For the encoder we use PyTorch's `nn.TransformerEncoder` (the same building block as your Week 8 block, with `batch_first=True` and `norm_first=True`, a feed-forward width of 4d, and a final LayerNorm). Then you write the forward pass of the model in an exercise, and below it we define the full `TinyViT` (with the options for the experiments of Part 5).

# %%
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

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.2
#
# Fill in the `forward` method of `MyTinyViT` in five lines: (1) the patch tokens, (2) the CLS token in front, (3) the positions added, (4) the encoder, (5) the head applied to the CLS output (position 0). Why it matters: it is the whole ViT in five lines. Once you can write them, the recipe of the lesson is no longer a diagram but something you can run.

# %%
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

# %%
torch.manual_seed(0); mine = MyTinyViT()
out = mine(torch.randn(8, 1, 16, 16))
if out is None:
    print("Fill in the forward method above, then run this cell again.")
else:
    check("class scores have shape (8, 5)", tuple(out.shape) == (8, 5), "use head(h[:, 0]) on the encoder output h of shape (8, 17, 32)")
    check("the output depends on the picture", float((mine(torch.randn(8, 1, 16, 16)) - out).abs().max()) > 1e-6, "did you use x?")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def forward(self, x):
#     t = self.patch_embed(x)                                   # (batch, 16, d)
#     t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)   # (batch, 17, d)
#     t = t + self.pos                                          # add the position table
#     h = self.encoder(t)                                       # (batch, 17, d)
#     return self.head(h[:, 0])                                 # the CLS output decides
# ```
#
# Paste the method into the class `MyTinyViT` (replace its `forward`).
#
# The check tests the shape of the scores (8, 5) and that they change with the picture. The comparison in the next cell is stricter: it loads our weights into your class and compares the scores. In the notebook the solution below shows only the forward method; paste it into the class.
#
# </details>

# %%
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

# %% [markdown]
# **What you should see.**
#
# The six lines print (8, 16, 32) for the patch tokens, (8, 17, 32) after the CLS token is put in front and after the positions are added, (8, 17, 32) after the encoder (a Transformer layer never changes the shape), (8, 32) for the CLS output and (8, 5) for the class scores. After your exercise, the cell under it builds `TinyViT`, prints that a batch of 8 pictures gives scores of shape (8, 5), that every picture has 17 tokens of 32 numbers, and checks that your `MyTinyViT` gives exactly the same scores as ours when it receives our weights (a green tick). **If the tick is there, your forward pass is the same computation as ours.**
#
# Tips and common mistakes:
#
# - The CLS token starts at zero and the position table at small random numbers (0.02 times a normal random number); both are trained like all other weights. The head also starts with small weights, so that the first guess is almost uniform (you check this in Step 4.4).
# - `self.cls.expand(len(t), -1, -1)` copies the one CLS vector for every picture in the batch without using extra memory. Without `expand` the shapes do not match.
# - `TinyViT` has the options `use_pos=False` (no position information), `pool="mean"` (average of the patch outputs instead of the CLS output) and `pos_kind="sincos"` (fixed positions). They are for Part 5; with the defaults it is exactly your model.
#
# Try also (optional):
#
# - Print `vit.encoder` to see the two layers with their attention, feed-forward and normalisation modules.
# - Replace `h[:, 0]` by `h[:, 5]` in your forward pass. The model still runs, but the head now reads a patch token. What would you expect the training to do?

# %% [markdown]
# ### Step 4.3. Count the parameters, by formula and with count_params
#
# Every parameter of the model belongs to one of six groups: the patch embedding (the convolution), the CLS vector, the position table, the encoder layers, the final LayerNorm and the head. The cell below writes the count as a formula (the Week 8 counting, with the exact bias terms of PyTorch's layer) and compares it with `count_params`. It also defines `vit_counts(H, W, p, C, d)`, the token arithmetic of Part 2, and then applies the same formula to ViT-Base (d = 768, 12 layers, 224 x 224 pictures, 16 x 16 patches) without building it: a model with 85 million parameters needs no training for us to count it.

# %%
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

# %%
# The same formula for the real ViT-Base (d = 768, 12 layers, 224 x 224 pictures, 16 x 16 patches): we only count, we do not build it
c = vit_counts(224, 224, 16, 3, 768)
base, base_layer = vit_param_formula(768, 12, c["patches"], c["patch_dim"])
print("tokens: %d patches of %d numbers, sequence %d; patch embedding %d parameters" % (c["patches"], c["patch_dim"], c["seq_len"], c["embed_params"]))
print("per encoder layer:", base_layer, "  12 layers:", base["layers"])
print("positions:", base["positions"], " CLS:", base["cls"], " total without a head: %d (about %.1f M)" % (sum(base.values()), sum(base.values()) / 1e6))
assert (c["patches"], c["patch_dim"], c["seq_len"], c["embed_params"]) == (196, 768, 197, 590592)
assert base_layer == 7087872 and base["layers"] == 85054464 and base["positions"] == 151296
RESULTS["p4_vit_base"] = sum(base.values())

# %% [markdown]
# **What you should see.**
#
# For the tiny model the formula gives patch embedding 544, CLS 32, positions 544 (17 x 32), the two layers 25,408 (12,704 each), the final LayerNorm 64 and the head 165 (32 x 5 + 5): **26,757 in total, exactly what `count_params` finds**, and the assert passes. A layer has 12 d squared + 13 d numbers: 4 d squared + 4 d for attention (the query, key, value and output matrices with their biases), 8 d squared + 5 d for the feed-forward part (d to 4 d and back, with biases) and 4 d for the two LayerNorms (2 d each). The 13 d are the bias terms of the Week 8 counting, which uses only the squares. Your Imagine-first guess is compared with 26,757 in the first block.
#
# For ViT-Base the second cell gives 196 patches of 768 numbers, a sequence of 197 and a patch embedding of 590,592 parameters (768 x 768 + 768), exactly the lesson's numbers. One encoder layer has 7,087,872 parameters, twelve have 85,054,464, the position table has 197 x 768 = 151,296 numbers, the CLS token 768 and the final LayerNorm 1,536. The total without a head is **85,798,656, about 85.8 million**. The 85,054,464 of the lesson counts only the twelve blocks; adding the embedding, the CLS token, the positions and the last norm gives the 85.8 million of the published model.
#
# Tips and common mistakes:
#
# - Do not forget the final LayerNorm and the biases: leaving them out is the usual reason why your count is a few hundred short.
# - The parameters of the patch embedding (590,592) are tiny next to the blocks (85 million): almost all the numbers of a ViT sit in the encoder layers, as in a language model.
#
# Try also (optional):
#
# - Change the head to 1000 classes for ViT-Base with `n_classes=1000`: the head adds 769,000 parameters. How much does it change the total in percent? (Less than 1 percent.)

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.3
#
# Count the parameters of a ViT with d = 48, 2 layers, 4 heads, patch 4, 16 x 16 pictures and 2 classes (the model of Part 5) and store the number in `params_by_hand`. Why it matters: counting by hand is the way to know what a model costs before you train it, and it is the check that you understood the architecture.

# %%
params_by_hand = None   # YOUR CODE: the total for d = 48, 2 layers, 4 heads, patch 4, 16 x 16 pictures, 2 classes, using vit_param_formula or the pieces

# %%
if params_by_hand is None:
    print("Fill in params_by_hand above, then run this cell again.")
else:
    torch.manual_seed(0); m48 = TinyViT(d=48, heads=4, layers=2, n_classes=2)
    check("matches count_params", params_by_hand == count_params(m48), "patch embed 16*48+48, CLS 48, positions 17*48, 2 layers of 12d^2+13d, final norm 2d, head 48*2+2")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# parts48, _ = vit_param_formula(48, 2, 16, 16, n_classes=2)
# params_by_hand = sum(parts48.values())
# print(params_by_hand)
# ```
#
# The patch embedding has 16 x 48 + 48 = 816, the CLS 48, the positions 17 x 48 = 816, each layer 12 x 48 x 48 + 13 x 48 = 28,272 (two layers 56,544), the final norm 96 and the head 48 x 2 + 2 = 98: together 58,418. Notice that more than 96 percent of the numbers are in the two layers.
#
# </details>

# %% [markdown]
# ### Step 4.4. Sanity checks before training: the starting loss and an overfitting test
#
# Before you train a new model for a long time, check two things that cost seconds. First, a model that knows nothing about 5 equally likely classes should start with a loss of ln(5) = 1.609: if you see 5 or 0.1 at the start, something is wrong with the shapes, the labels or the initialisation. Second, a model must be able to overfit a tiny data set: if it cannot memorise 64 pictures, it will never learn 400. We make 400 training and 200 test pictures with `make_shapes` (centred, five shapes, seeds 1 and 2) and run both checks.

# %%
Xs_tr, ys_tr = make_shapes(400, seed=1)       # five centred line shapes, 16 x 16, 400 for training
Xs_te, ys_te = make_shapes(200, seed=2)       # 200 the model never sees
EP = 3 if QUICK else 30
torch.manual_seed(SEED); m = TinyViT()
with torch.no_grad():
    loss0 = float(F.cross_entropy(m(torch.as_tensor(Xs_tr)), torch.as_tensor(ys_tr)))
print("starting loss: %.3f   ln(5) = %.3f" % (loss0, math.log(5)))
assert abs(loss0 - math.log(5)) < 0.15

# %%
torch.manual_seed(SEED); m = TinyViT()
h64 = train_classifier(m, Xs_tr[:64], ys_tr[:64], Xs_tr[:64], ys_tr[:64], epochs=10 if QUICK else 60, lr=3e-3, batch=16)
print("overfit 64 pictures: train accuracy by epoch 10, 30, 60:", [round(h64["train_acc"][i], 2) for i in (9, 29, 59) if i < len(h64["train_acc"])])
print("final loss %.3f, train accuracy %.2f" % (h64["loss"][-1], h64["train_acc"][-1]))
RESULTS["p4_overfit64"] = round(h64["train_acc"][-1], 2)
assert QUICK or h64["train_acc"][-1] > 0.95

# %% [markdown]
# **What you should see.**
#
# The starting loss is 1.618 against ln(5) = 1.609: the small head makes the first guess almost uniform. The second cell trains on 64 pictures with batches of 16 and a learning rate of 0.003; the training accuracy is already 1.00 after 10 epochs and the loss is 0.004 after 60. **Both checks pass**, so the architecture can learn and the training loop works. Without the small head initialisation (PyTorch's default) we measured a starting loss of 1.84, which is not wrong but is a worse start.
#
# Tips and common mistakes:
#
# - Always run these two checks first when you build a model from scratch. A model that fails the overfitting test has a bug, not a data problem.
# - The overfitting test says nothing about how well the model will do on new pictures; that is the job of the test set in Step 4.5.
#
# Try also (optional):
#
# - Break the model on purpose: remove the `self.pos` addition from `embed` in a copy and watch the overfitting test. It still passes, because a bag of patches is enough to tell five centred line shapes apart. This is the point of Part 5, where the task needs the positions.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.4
#
# Store the starting loss of a model that knows nothing about 5 classes in `expected_start` (natural logarithm). Why it matters: you will use this number every time you train a classifier in the course.

# %%
expected_start = None   # YOUR CODE: the loss of a model that knows nothing about 5 equally likely classes (natural log)

# %%
if expected_start is None:
    print("Fill in expected_start above, then run this cell again.")
else:
    check("ln(5)", abs(expected_start - math.log(5)) < 1e-6, "cross-entropy of a uniform guess over 5 classes is -ln(1/5) = ln(5)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# expected_start = math.log(5)
# print(round(expected_start, 4))
# ```
#
# A uniform guess gives each class the probability 1/5, so the loss is minus the log of 1/5, which is ln(5) = 1.6094.
#
# </details>

# %% [markdown]
# ### Step 4.5. Train on make_shapes and compare with a small CNN
#
# Now the real training: 30 epochs of Adam at a learning rate of 0.003 on the 400 pictures. For a fair comparison we build a small CNN in the style of Week 4 (two convolutions with 32 and 64 channels, each followed by ReLU and max pooling, then a linear layer) with a similar number of parameters, and train it with the same recipe. We print a table and draw the learning curves.

# %%
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

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for h, name in [(h_vit, "TinyViT"), (h_cnn, "SmallCNN")]:
    ax[0].plot(h["loss"], label=name); ax[1].plot(h["test_acc"], label=name)
ax[0].set_title("training loss"); ax[1].set_title("test accuracy"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.**
#
# The TinyViT has 26,757 parameters, the CNN 23,941. Both reach a **training accuracy of 1.000 and a test accuracy of 1.000** after 30 epochs; the ViT takes about 2.4 seconds and the CNN about 2.7 (times will differ on your machine). So the honest result is a tie: the five centred line shapes are an easy task, and a small ViT can solve it. The curves show that the ViT's test accuracy passes 0.95 already in the second epoch (Exercise 4.5). **Do not read this as 'ViTs are as good as CNNs'**: Step 5.5 shows what happens when the task is harder or the data smaller.
#
# Tips and common mistakes:
#
# - Equal parameter counts do not mean an equal amount of computation, but both models are tiny, so the comparison is fair enough for a lesson.
# - If your numbers are a little different (for example 0.995), your library version is different. The result that matters is that both models solve the task.
#
# Try also (optional):
#
# - Train both models with `shift=2` in `make_shapes` (the shapes move up to 2 pixels from the centre) and compare. This is the hint for Step 5.5.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.5
#
# Find the first epoch (counting from 1) at which the TinyViT's test accuracy is 0.95 or more and store it in `best_epoch_vit` (write -1 if it never happens). Why it matters: it tells you how fast the model learns, not just where it ends.

# %%
best_epoch_vit = None   # YOUR CODE: the first epoch (counting from 1) at which the TinyViT's test accuracy reaches 0.95 or more (use h_vit["test_acc"]); -1 if it never does

# %%
if best_epoch_vit is None:
    print("Fill in best_epoch_vit above, then run this cell again.")
else:
    ref = next((i + 1 for i, a in enumerate(h_vit["test_acc"]) if a >= 0.95), -1)
    check("first epoch with test accuracy of at least 0.95", best_epoch_vit == ref, "loop over h_vit['test_acc'] with enumerate; remember that epochs start at 1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_epoch_vit = next((i + 1 for i, a in enumerate(h_vit["test_acc"]) if a >= 0.95), -1)
# print(best_epoch_vit)
# ```
#
# With `enumerate` over `h_vit["test_acc"]` and `next(...)` you find the first index and add 1. We get epoch 2. The CNN is similar; a ViT does not need many epochs on this easy task.
#
# </details>

# %% [markdown]
# ### Step 4.6. Look inside: where does the CLS token look?
#
# A trained ViT can show what its CLS token looks at. In the last encoder layer the CLS token computes attention weights over all 17 tokens (itself and 16 patches); the weights for the 16 patches, averaged over the 4 heads, tell how much each patch contributes to the answer. We took the layer's attention module and ran it with `need_weights=True` (the method `cls_attention` of `TinyViT`), and draw the 16 weights as a 4 x 4 heat map under a picture of each shape.

# %%
vit45.eval()
idx = [int(np.where(ys_te == k)[0][0]) for k in range(5)]            # one test picture of each shape
att = vit45.cls_attention(torch.as_tensor(Xs_te[idx]))               # (5, 16): CLS-to-patch attention, last layer
fig, ax = plt.subplots(2, 5, figsize=(10, 4.2))
for j, i in enumerate(idx):
    ax[0, j].imshow(Xs_te[i, 0], cmap="gray_r"); ax[0, j].set_title(SHAPE_NAMES[ys_te[i]], fontsize=8); ax[0, j].axis("off")
    ax[1, j].imshow(att[j].reshape(4, 4).numpy(), cmap="viridis"); ax[1, j].set_title("CLS attention", fontsize=8); ax[1, j].axis("off")
plt.tight_layout(); plt.show()
print("each row of attention sums to", [round(float(a), 2) for a in att.sum(1)], "(the rest goes to the CLS token itself)")
print("attention to the 4 central patches (index 5, 6, 9, 10): %.2f of the attention on patches" % float((att[:, [5, 6, 9, 10]].sum(1) / att.sum(1)).mean()))
RESULTS["p4_centre_attention"] = round(float((att[:, [5, 6, 9, 10]].sum(1) / att.sum(1)).mean()), 2)
assert att.shape == (5, 16)

# %% [markdown]
# **What you should see.**
#
# You see five pictures (one of each shape) with a small heat map under each. The attention weights of the 16 patches add up to between 0.82 and 0.98 for the five pictures (the rest is the CLS token attending to itself). The four central patches (numbers 5, 6, 9 and 10) receive on average **0.33 of the attention on patches**; if the attention were spread evenly they would receive 4/16 = 0.25. So the model looks a little more at the centre, where the shapes are, but the maps are not sharp: with 400 pictures and 30 epochs the model has found a working rule, not a neat picture of the object. For the first picture (a horizontal bar) the strongest patch is at row 1, column 0 (Exercise 4.6).
#
# Tips and common mistakes:
#
# - Attention maps are a hint about what the model uses, not a proof. A patch with low attention can still matter through the values of the other tokens and the earlier layers.
# - Real ViTs trained on millions of pictures produce much sharper maps on objects; those are the pictures in the lesson.
#
# Try also (optional):
#
# - Draw the map for the first layer instead of the last (take `self.encoder.layers[0]`). Do the maps look different?
# - Average the attention over 50 pictures of each shape and compare the five average maps.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.6
#
# Find the patch the CLS token attends to most in the first of the five pictures and store its (row, column) in `top_patch`. Why it matters: it connects the numbers of the attention matrix to a place in the picture.

# %%
top_patch = None   # YOUR CODE: for the first picture (j = 0) the (row, column) in the 4 x 4 grid of the patch the CLS token attends to most (use att[0].argmax() and divmod(..., 4))

# %%
if top_patch is None:
    print("Fill in top_patch above, then run this cell again.")
else:
    ref = divmod(int(att[0].argmax()), 4)
    check("the strongest patch", tuple(top_patch) == ref, "index = row * 4 + column, so divmod(index, 4) gives (row, column)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# top_patch = divmod(int(att[0].argmax()), 4)
# print(top_patch)
# ```
#
# The flat index of the patch is `att[0].argmax()`; `divmod(index, 4)` converts it to (row, column) of the 4 x 4 grid. We get (1, 0).
#
# </details>

# %% [markdown]
# ### Part 4 checkpoint: Build a tiny Vision Transformer
#
# You should now have:
#
# - the patch embedding and the forward pass of a ViT that you wrote and checked yourself (4.1 and 4.2)
# - the exact parameter count of the tiny ViT (26,757), the formula behind it and the count for ViT-Base (85,798,656 without a head) (4.3)
# - the sanity checks (starting loss near ln 5, overfitting 64 pictures), the comparison with a CNN and the CLS attention map (4.4 to 4.6)

# %% [markdown]
# ## Part 5. ViT experiments: positions, pooling, patch size (25 min)
#
# *Lesson sections: "Positions" and "Vision Transformer".* Part 3 proved on paper that a Transformer without position information cannot tell where anything is. Now you test it on a real task, and then you vary three design choices of the ViT: pooling, patch size and the amount of training data. The task is `make_left_right`: every picture has a plus sign (the target) and a filled square (a distractor), and the label says whether the plus sign is in the left half. The answer depends only on **where** the plus is. ==You will train a small CNN, a ViT with position embeddings and a ViT without them, show with shuffled patches why the last one is stuck at chance, and then compare CLS and mean pooling, three patch sizes and four data sizes, ending with fixed sine-cosine positions.== All numbers are for `SEED = 42`; a test set of 300 pictures has a standard error of about 0.03 near an accuracy of 0.5, so differences of two or three points are noise.
#
# **✏️ Imagine first.** Picture the ViT as a jury that receives the 16 patches of the picture on separate cards, shuffled in a bag, with no numbers on the cards. The jury can describe every card (this one has a vertical line, that one is empty) but nobody knows which card came from the left. The question is: is the plus sign in the left half? Before you run any code, guess the jury's accuracy on 300 test pictures. Is it near 0.5 (pure guessing), near 0.75, or near 1.0? Write your guess in the cell below. You compare it with Step 5.1.

# %%
# Imagine first: the accuracy of a ViT WITHOUT position embeddings on "is the plus sign in the left half?" (chance is 0.5)
my_guess_nopos = None   # write a number between 0.5 and 1.0

# %% [markdown]
# ### Step 5.1. The left/right task: a CNN, a ViT with positions, a ViT without
#
# We draw 600 training and 300 test pictures (seeds 1 and 2), look at ten of them, and train three models with the same recipe (40 epochs, Adam, learning rate 0.001): a small CNN with 5,826 parameters, a TinyViT with d = 48, 2 layers and 4 heads with learned positions, and the same TinyViT with `use_pos=False`. Chance is 0.5, because the two classes are almost balanced (48 percent 'left' in the training set).

# %%
Xlr_tr, ylr_tr = make_left_right(600, seed=1)     # 600 training pictures
Xlr_te, ylr_te = make_left_right(300, seed=2)     # 300 test pictures
show_images(Xlr_tr, titles=["left" if v else "right" for v in ylr_tr], n=10)
print("share of 'left' in the training set: %.2f   chance accuracy: 0.50" % ylr_tr.mean())

# %%
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

# %%
fig, ax = plt.subplots(figsize=(5, 3.2))
for name, h in lr_hist.items(): ax.plot(h["test_acc"], label=name)
ax.axhline(0.5, color="k", ls=":", label="chance"); ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("your Imagine-first guess:", my_guess_nopos)

# %% [markdown]
# **What you should see.**
#
# The three lines show: the **small CNN reaches 1.000 on both the training and the test set**; the **ViT with positions reaches 0.993 on the test set** (training 1.000); the **ViT without positions reaches only 0.537 on the test set**, although its training accuracy is 0.722: it memorised part of the noise of the 600 training pictures but found no rule that carries over, and a test accuracy of 0.537 on 300 pictures is within two standard errors of 0.5. Your Imagine-first guess: the answer is near 0.5. The learning curves show the same: the ViT with positions climbs to 0.99 while the one without wanders around 0.5.
#
# The reason is not that the no-position ViT is a bad model: on `make_shapes` (Step 4.5) the same architecture is perfect. It is that **this task cannot be solved from the bag of patches**: the plus sign and the square look the same in the left and in the right half. Step 5.2 proves it.
#
# Tips and common mistakes:
#
# - A test accuracy of 0.50 or 0.54 'without positions' is the expected result; if you see 0.8 something is wrong (the positions were not removed).
# - The ViT is about ten times larger than the CNN in parameters (58,418 against 5,826) and solves the task no better. The CNN has the idea of 'where' built in, because its filters slide over the picture and the flatten layer remembers the place; the ViT has to learn it from the position table.
#
# Try also (optional):
#
# - Train the no-position ViT for 100 epochs. Does the test accuracy move away from 0.5?
# - Use `d=32` for both ViTs. Does the ViT with positions still reach 0.99?

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.1
#
# Store the difference in test accuracy (last epoch) between the ViT with positions and the ViT without positions in `gap_positions`. Why it matters: it is the number that sums up the experiment, the value of the positions on this task.

# %%
gap_positions = None   # YOUR CODE: test accuracy of "ViT with positions" minus test accuracy of "ViT without positions" (last epoch, from lr_hist)

# %%
if gap_positions is None:
    print("Fill in gap_positions above, then run this cell again.")
else:
    ref = lr_hist["ViT with positions"]["test_acc"][-1] - lr_hist["ViT without positions"]["test_acc"][-1]
    check("the gap the positions make", abs(gap_positions - ref) < 1e-9, "take [-1] of the test_acc lists of the two entries of lr_hist and subtract")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# gap_positions = lr_hist["ViT with positions"]["test_acc"][-1] - lr_hist["ViT without positions"]["test_acc"][-1]
# print(round(gap_positions, 3))
# ```
#
# 0.993 minus 0.537 is 0.457 (the exact value depends on the unrounded accuracies). About 46 points of accuracy come from the position embeddings alone: same architecture, same data, same recipe.
#
# </details>

# %% [markdown]
# ### Step 5.2. Why the ViT without positions fails: shuffle the patches
#
# Here is the proof from Part 3, on the trained models. We define `shuffle_patches_img`, which cuts every test picture into its 16 patches and puts them back in a random order (a different order for every picture), and we feed the shuffled pictures to the ViT without positions and the ViT with positions. If the first model really sees a bag, its answers cannot change. For the shuffled pictures the label can flip (the plus sign may land in the other half), so we also report the accuracy against the original labels.

# %%
def shuffle_patches_img(X, patch=4, seed=0):
    """Cut every picture of X (n, C, H, W) into patches and put the patches back in a random order (a different order per picture)."""
    r = np.random.RandomState(seed); n, c, H, W = X.shape; g = H // patch
    P = X.reshape(n, c, g, patch, g, patch).transpose(0, 2, 4, 1, 3, 5).reshape(n, g * g, c, patch, patch)
    P = np.stack([P[i, r.permutation(g * g)] for i in range(n)])
    return np.ascontiguousarray(P.reshape(n, g, g, c, patch, patch).transpose(0, 3, 1, 4, 2, 5).reshape(n, c, H, W))

Xlr_sh = shuffle_patches_img(Xlr_te, 4, seed=0)
show_images(np.concatenate([Xlr_te[:5], Xlr_sh[:5]]), titles=["original"] * 5 + ["shuffled"] * 5, n=10)

# %%
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

# %% [markdown]
# **What you should see.**
#
# The table says: the **ViT without positions gives the same answer on 100 percent of the pictures** (agreement 1.000) and its accuracy is 0.537 before and after: shuffling changes nothing, because for this model the shuffled picture is the same bag. The **ViT with positions agrees with itself only on 51.3 percent** of the pictures, and its accuracy falls from 0.993 to 0.507, which is chance: shuffling destroys the information it used. The assert in the cell is the Part 3 theorem: a Transformer encoder with a CLS token and no positions is blind to the order of its tokens. The exercise adds the CNN: it agrees with itself on 51.3 percent of the shuffled pictures (the same share as the ViT with positions), because it also uses the layout.
#
# Tips and common mistakes:
#
# - The agreement of the ViT with positions is close to 0.5 and not near 0 because a random shuffle puts the plus sign in the correct half about half of the time.
# - If the agreement of the no-position model is 0.99 and not 1.000, the difference is a rounding effect of float numbers in a few pictures near a decision border; the exact value is not important.
#
# Try also (optional):
#
# - Shuffle only within the left half or only within the right half of the picture and re-test the ViT with positions. How much accuracy does it keep?

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.2
#
# Compute the share of test pictures for which the small CNN gives the same answer on the original and on the shuffled pictures and store it in `agree_cnn`. Why it matters: it tells you that a CNN is not blind to layout, even though it has no position table.

# %%
agree_cnn = None   # YOUR CODE: the share of test pictures for which the small CNN gives the same answer on Xlr_te and on the patch-shuffled Xlr_sh (use predict)

# %%
if agree_cnn is None:
    print("Fill in agree_cnn above, then run this cell again.")
else:
    ref = float((predict(lr_models["small CNN"], Xlr_te) == predict(lr_models["small CNN"], Xlr_sh)).mean())
    check("agreement of the CNN", abs(agree_cnn - ref) < 1e-9, "compare predict(model, Xlr_te) with predict(model, Xlr_sh) and take the mean")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# cnn = lr_models["small CNN"]
# agree_cnn = float((predict(cnn, Xlr_te) == predict(cnn, Xlr_sh)).mean())
# print(round(agree_cnn, 3))
# ```
#
# Use `predict(cnn, Xlr_te)` and `predict(cnn, Xlr_sh)`, compare them and take the mean. We get 0.513: the CNN's answers change on almost half of the pictures, as they should when the plus sign moves.
#
# </details>

# %% [markdown]
# ### Step 5.3. CLS token or mean pooling?
#
# The classification head reads either the CLS token (what you built) or the average of the 16 patch outputs ('mean pooling', which some ViTs prefer). We train a mean-pooling model on both tasks with the same recipe and put the four results in a table. The helper `fit_vit(task, ...)` trains a TinyViT with the recipes of Step 4.5 (`shapes`) and Step 5.1 (`lr`) and returns the model, the history and the seconds; the CLS entries reuse the models already trained.

# %%
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

# %% [markdown]
# **What you should see.**
#
# The table: on `make_shapes` both poolings reach **1.000**; on left/right the CLS model reaches **0.993** and the mean-pooling model **0.993**. So there is no difference between the two on these tasks: both are tied (the exercise answer is `tie`). That is the honest finding. Large ViTs trained on millions of pictures show small differences between the two, and the choice there is made by experiments on the big data set, not by a rule.
#
# Tips and common mistakes:
#
# - With positions added, mean pooling still sees where each patch is, because the position information is inside each patch token.
# - Do not conclude 'pooling never matters': on harder tasks and larger models the gap can be a point or two in either direction.
#
# Try also (optional):
#
# - Repeat the comparison with `d=32` on the left/right task. Is the tie still there?

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.3
#
# Decide which pooling is better on the left/right task according to `pool_tab` and store `"cls"`, `"mean"` or `"tie"` in `better_pool_lr`. Why it matters: it forces you to read the table instead of assuming that one of them must win.

# %%
better_pool_lr = None   # YOUR CODE: "cls" or "mean", whichever has the higher left/right accuracy in pool_tab (write "tie" if they are equal)

# %%
if better_pool_lr is None:
    print("Fill in better_pool_lr above, then run this cell again.")
else:
    a, b = pool_tab[("cls", "lr")], pool_tab[("mean", "lr")]
    ref = "tie" if abs(a - b) < 1e-9 else ("cls" if a > b else "mean")
    check("the better pooling on left/right", better_pool_lr == ref, "compare pool_tab[('cls', 'lr')] with pool_tab[('mean', 'lr')]")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# a, b = pool_tab[("cls", "lr")], pool_tab[("mean", "lr")]
# better_pool_lr = "tie" if a == b else ("cls" if a > b else "mean")
# print(better_pool_lr)
# ```
#
# Both entries are 0.993, so the answer is `tie`. With a different seed one of them may win by 0.003: that is a difference of one picture in 300.
#
# </details>

# %% [markdown]
# ### Step 5.4. Patch size: 2, 4 or 8?
#
# Patch size is the main design choice of a ViT: small patches give long sequences and fine detail, large patches give short sequences and less detail. For the 16 x 16 pictures we compare patch 2 (64 tokens), patch 4 (16 tokens) and patch 8 (4 tokens), on both tasks. The table shows the number of tokens, the parameters (the patch embedding and the position table change with p), the seconds per epoch on `make_shapes`, and the two test accuracies; the plot shows accuracy against tokens.

# %%
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

# %%
fig, ax = plt.subplots(figsize=(5, 3.2))
toks = [r_["tokens"] for r_ in patch_rows]
ax.plot(toks, [r_["shapes"] for r_ in patch_rows], "o-", label="make_shapes"); ax.plot(toks, [r_["lr"] for r_ in patch_rows], "s-", label="left/right")
ax.set_xscale("log", base=2); ax.set_xlabel("tokens per picture (patch 8, 4, 2 from left to right)"); ax.set_ylabel("test accuracy"); ax.legend()
for r_ in patch_rows: ax.annotate("p=%d" % r_["p"], (r_["tokens"], r_["lr"]), fontsize=8, xytext=(3, -10), textcoords="offset points")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.**
#
# The table: patch 2 has **64 tokens**, 27,909 parameters, about 0.20 seconds per epoch, accuracy 1.000 on `make_shapes` and **1.000** on left/right. Patch 4 has 16 tokens, 26,757 parameters, about 0.08 seconds per epoch, 1.000 and **0.993**. Patch 8 has only **4 tokens**, 27,909 parameters, about 0.06 seconds per epoch, 1.000 and **0.860**. The 5-shape task does not care (every patch size solves it), but the left/right task drops to 0.86 with 4 tokens: with patches of 8 x 8 pixels a plus sign of 5 x 5 pixels can lie across two patches, a patch holds the plus sign and the square together, and the model's only way to find the plus is through a few big, mixed patches. The number of attention scores per head grows with the square of the sequence: 65 squared against 17 squared is 14.6 times more for patch 2 than for patch 4 (Exercise 5.4), while the time per epoch grew about 2.5 times, because the small model spends part of its time outside attention.
#
# The trade-off in one sentence: **smaller patches keep more detail but cost much more; larger patches are cheap but can blur what you need.** For real 224 x 224 pictures the lesson's numbers are 49 tokens for patch 32, 196 for patch 16 and 784 for patch 8.
#
# Tips and common mistakes:
#
# - The parameters hardly change with the patch size because the encoder layers (the bulk of the model) do not depend on it; only the embedding (p x p x d) and the position table ((N + 1) x d) change.
# - The seconds per epoch depend on your machine. Compare the ratios, not the absolute values.
#
# Try also (optional):
#
# - Try patch 16 (one token) on the left/right task. What accuracy would you expect? (A single patch is the whole picture: the model is a plain MLP in disguise.)

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.4
#
# Compute how many times more attention scores per head the patch-2 model computes than the patch-4 model (sequence length: tokens plus 1 for the CLS token; scores: the length squared) and store it in `cost_ratio`. Why it matters: it is the reason why people do not use 1 x 1 patches on real pictures.

# %%
cost_ratio = None   # YOUR CODE: how many times more attention scores per head does the patch-2 model compute than the patch-4 model? (sequence length = tokens + 1 for the CLS token; scores = length squared)

# %%
if cost_ratio is None:
    print("Fill in cost_ratio above, then run this cell again.")
else:
    check("ratio of attention scores", abs(cost_ratio - (65 ** 2) / (17 ** 2)) < 1e-6, "(64 + 1) squared divided by (16 + 1) squared")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# cost_ratio = (64 + 1) ** 2 / (16 + 1) ** 2
# print(round(cost_ratio, 2))
# ```
#
# (64 + 1) squared is 4,225 and (16 + 1) squared is 289; the ratio is 14.62.
#
# </details>

# %% [markdown]
# ### Step 5.5. Data hunger: how many pictures does each model need?
#
# Lesson: a ViT has less built-in knowledge about pictures than a CNN, so it needs more data. Test it. We make a harder version of `make_shapes` (the shape may sit up to 2 pixels from the centre, noise 0.3), keep 300 test pictures, and train the TinyViT and the CNN on the first 50, 100, 200 and 400 training pictures. To be fair each model takes about the same number of gradient steps, so the small training sets get more epochs.

# %%
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

# %%
fig, ax = plt.subplots(figsize=(5, 3.2))
for name, v in curve.items(): ax.plot(sizes_n, v, "o-", label=name)
ax.set_xscale("log"); ax.set_xlabel("training pictures"); ax.set_ylabel("test accuracy"); ax.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.**
#
# The table: with 50 pictures the **ViT reaches 0.430 and the CNN 0.887**; with 100 pictures 0.623 against 0.957; with 200 0.690 against 0.980; with 400 pictures **0.880 against 0.993**. The CNN is better at every size, and the gap is 0.457 at 50 pictures and 0.113 at 400: the CNN needs a few dozen pictures to be almost perfect, the ViT still climbs at 400. This is the 'lack of inductive bias' of the lesson: a convolution already assumes that the same pattern may appear anywhere in the picture (translation) and that nearby pixels belong together, so the CNN does not need to learn this from data. A ViT must find it by itself, and the shifted shapes make this expensive.
#
# On the easy, centred data of Step 4.5 you saw no difference. **The comparison depends on the task and the data size.** Large pretrained ViTs beat CNNs because they are trained on hundreds of millions of pictures; a ViT trained from scratch on a few hundred pictures should not be expected to.
#
# Tips and common mistakes:
#
# - Each number is a single run. With a different seed the ViT may move by a few points, but the ordering of the two curves is stable.
# - More epochs for the small sets help the ViT only a little; the problem is the data, not the optimiser.
#
# Try also (optional):
#
# - Add 800 and 1,600 training pictures (make a larger set with `make_shapes`). Does the ViT catch up?

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.5
#
# Store the difference between the CNN's and the ViT's accuracy at 50 training pictures in `gap_at_50`. Why it matters: it quantifies the data hunger in one number.

# %%
gap_at_50 = None   # YOUR CODE: SmallCNN accuracy minus TinyViT accuracy with 50 training pictures (index 0 of the lists in curve)

# %%
if gap_at_50 is None:
    print("Fill in gap_at_50 above, then run this cell again.")
else:
    check("the gap at 50 pictures", abs(gap_at_50 - (curve["SmallCNN"][0] - curve["TinyViT"][0])) < 1e-9, "curve['SmallCNN'][0] - curve['TinyViT'][0]")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# gap_at_50 = curve["SmallCNN"][0] - curve["TinyViT"][0]
# print(round(gap_at_50, 3))
# ```
#
# `curve["SmallCNN"][0]` is 0.887 and `curve["TinyViT"][0]` is 0.430: the gap is 0.457.
#
# </details>

# %% [markdown]
# ### Step 5.6. Where else does 'where' matter? Fixed sine-cosine positions instead of a learned table
#
# So far the position table was learned. A popular alternative is a fixed table made from the sine-cosine formula of Week 8, extended to two dimensions: half of the d numbers encode the row, half the column. It has no parameters, and it can be computed for any grid size. The cell defines `sincos_2d(grid, d)`, checks the table for d = 8 on a 4 x 4 grid, and trains a ViT with `pos_kind="sincos"` on the left/right task. Then you write down three answers about where else 'where' matters.

# %%
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

# %% [markdown]
# **What you should see.**
#
# The table has shape (16, 8) and all 16 position vectors are different. The ViT with fixed sine-cosine positions reaches **1.000** on the left/right task, against 0.993 with learned positions and 0.537 with no positions. So a position table that is not learned at all is enough here: **what the model needs is any code that makes the places different, not necessarily a trained one.** The difference of 0.007 between 1.000 and 0.993 is two pictures in 300, noise.
#
# Your three written answers go below the cell: for text a position is a single number along a line (a 1D code); for a map or satellite image the position has two coordinates and a real distance, so a 2D code or the geographic coordinates are natural; for an X-ray the same patch of grey can mean a healthy lung in one place and a problem at another, and scans of a different size need a code that can be computed for any grid (sine-cosine) or interpolated (Part 3.6).
#
# Tips and common mistakes:
#
# - The sine-cosine table is added to the tokens exactly like the learned one; it is stored as a buffer so that the optimiser leaves it alone.
# - Relative position codes (what matters is the offset between two patches, not the absolute place) are a third family used in many modern models.
#
# Try also (optional):
#
# - Count the trainable parameters of the sine-cosine model and of the learned one. Which is smaller and by how much? (The sine-cosine model is smaller by the 17 x 48 = 816 numbers of the table.)

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.6
#
# Write three answers (at least 12 words each) in `answers_56`: what position means in text, in a map or satellite image, and in an X-ray, and what you would use instead of a 1D learned table. Why it matters: choosing a position code is a design decision about the data, not a detail.

# %%
answers_56 = {
    "text":  "",   # YOUR ANSWER: in a line of text, which direction matters and what position would you give each word or patch?
    "maps":  "",   # YOUR ANSWER: for a map or satellite image, what is different about 'where' (think of coordinates and of a distance)?
    "xray":  "",   # YOUR ANSWER: for an X-ray, why can 'the same patch in another place' mean something else, and what would you use instead of a 1D learned table?
}

# %%
empty = [k for k, v in answers_56.items() if len(v.split()) < 12]
if empty:
    print("Write at least 12 words for:", empty)
else:
    check("three answers written", True)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answers_56 = {
#     "text": "In text the position is one number along a line, so a 1D table or a 1D sine-cosine code works; each word has a left neighbour and a right neighbour.",
#     "maps": "A map has two coordinates and real distances in metres, so I would give each patch its latitude and longitude, or a 2D sine-cosine code, and let nearby patches be similar.",
#     "xray": "A bright patch in the lung means something different from the same patch at the edge, so the position matters; I would use a 2D code (or a relative position) so that a larger scan still works.",
# }
# print({k: len(v.split()) for k, v in answers_56.items()})
# ```
#
# A good answer names the number of dimensions (1 for text, 2 for maps and scans), says whether distances matter, and says what happens if the test picture has a different size. The check only counts words; compare with the sample answers.
#
# </details>

# %% [markdown]
# ### Part 5 checkpoint: ViT experiments: positions, pooling, patch size
#
# You should now have:
#
# - the left/right table: CNN 1.000, ViT with positions 0.993, ViT without positions 0.537 (5.1)
# - the shuffle test showing why: agreement 1.000 for the model without positions, 0.513 for the one with positions (5.2)
# - the pooling, patch-size and data-size experiments, and fixed sine-cosine positions (5.3 to 5.6)

# %% [markdown]
# ## Part 6. Contrastive learning and CLIP (30 min)
#
# *Lesson section: "CLIP".* CLIP learns from pairs of a picture and its caption. It has two encoders, one for pictures and one for text, that map their inputs into the same space, and a loss that pulls the vectors of matching pairs together and pushes the other pairs in the batch apart. After training, a class name written as a sentence can be compared with a picture by the cosine of their vectors: zero-shot classification without a classifier trained on those classes. ==You will compute the contrastive loss by hand, see what the temperature does to it, build and train a miniature CLIP on coloured shapes with two combinations held out, test it zero-shot on combinations it never saw, and then use the real CLIP (openai/clip-vit-base-patch32) on pictures drawn with PIL, including its failures.== The toy model trains in a few seconds; the real CLIP is loaded once (about 600 MB; on the Spark it comes from the offline cache, so ask your instructor if the download fails). All numbers are for `SEED = 42`; the real CLIP outputs were measured with the library versions of the author and can differ in the last digits on your machine.
#
# **✏️ Imagine first.** A child has seen red squares, red circles, green squares, green circles, blue squares, blue crosses and a few other combinations, but never a green cross and never a blue circle. Then you ask: here is a picture of a green cross, which of these nine captions fits? A child who knows what 'green' and 'cross' mean separately would get it right. Does a model trained only on pairs of pictures and two-word captions do the same? Before you run any code, guess its accuracy on the two held-out combinations among the 9 captions (chance is 0.11). Write your guess in the cell below. You compare it with Step 6.4.

# %%
# Imagine first: a model trained only on image-caption pairs of 7 colour-shape combinations; can it name the 2 combinations it never saw?
my_guess_unseen = None   # write a number between 0 and 1 (chance among the 9 captions is 0.11)

# %% [markdown]
# ### Step 6.1. The contrastive loss by hand
#
# Take three pictures and their three captions, each as a vector of 2 numbers (real CLIP vectors have 512). Image i belongs to text i. The contrastive loss works in four moves: (1) normalise the vectors so that a dot product is a cosine, (2) compute the 3 x 3 matrix of cosines between every picture and every text, (3) multiply by a scale (the temperature trick: scale 10 here) to get the logits, and (4) read each row as a softmax over the texts for one picture and each column as a softmax over the pictures for one text, and take minus the log of the probability of the right partner. The loss is the mean over the rows and over the columns. The third picture is a hard example: its own text is farther away than text 1, so the loss has something to punish.

# %%
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

# %%
lt, labels = torch.tensor(logits), torch.arange(3)
ref = (F.cross_entropy(lt, labels) + F.cross_entropy(lt.T, labels)) / 2
print("F.cross_entropy version: %.4f" % float(ref))
assert abs(loss - float(ref)) < 1e-9
RESULTS["p6_loss_by_hand"] = round(float(loss), 4)

# %% [markdown]
# **What you should see.**
#
# The cosine matrix has 0.996, 0.997 and 0.716 on the diagonal. Picture 2 is almost as close to text 3 (1.000) as to its own text 2 (0.997): the two texts are almost the same vector. Picture 3 is closer to text 1 (0.922) than to its own text 3 (0.716). The image-to-text probabilities are 0.992, 0.493 and 0.106 for the right partners (picture 3 gives 0.832 to the wrong text); the text-to-image probabilities are 0.675, 0.963 and 0.055. The loss is **0.9869 in the image-to-text direction, 1.1093 in the text-to-image direction and 1.0481 on average**, and the assert passes: your NumPy loss equals `F.cross_entropy` applied to the logits and to their transpose. A model that knows nothing would score ln(3) = 1.099; this tiny example is only a little better because two of its three pairs are confusing.
#
# Tips and common mistakes:
#
# - Subtracting the largest value of each row before `np.exp` does not change the softmax but avoids overflow.
# - The two directions are not the same loss: a row asks 'which text fits this picture', a column asks 'which picture fits this text'. CLIP averages them so that both encoders learn.
#
# Try also (optional):
#
# - Make picture 3 closer to its text (change `[0.9, 0.5]` to `[0.4, 1.0]`) and watch the loss go down.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.1
#
# Write `clip_loss_np(logits)`: the symmetric contrastive loss of any square matrix of logits, as the average of the two directions. Why it matters: this function is the training objective of CLIP in a few lines, and you reuse the idea in the training loop of Step 6.3.

# %%
def clip_loss_np(logits):
    """The symmetric contrastive loss of a square matrix of logits (row i and column i belong together)."""
    return None   # YOUR CODE: softmax over each row, softmax over each column (the rows of logits.T), minus the log of the diagonal, average of the two directions

# %%
out = clip_loss_np(np.array([ [2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0] ]))
if out is None:
    print("Fill in clip_loss_np above, then run this cell again.")
else:
    t = torch.tensor([ [2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 3.0] ], dtype=torch.float64)
    ref = float((F.cross_entropy(t, torch.arange(3)) + F.cross_entropy(t.T, torch.arange(3))) / 2)
    check("agrees with F.cross_entropy", abs(out - ref) < 1e-9, "average the two directions; the second direction uses logits.T")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def clip_loss_np(logits):
#     def one_way(z):
#         z = z - z.max(axis=1, keepdims=True)
#         p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
#         return -np.log(np.diag(p)).mean()
#     return (one_way(logits) + one_way(logits.T)) / 2
# print(clip_loss_np(logits))
# ```
#
# Each direction is the mean of minus the log of the diagonal of a row softmax; the second direction is the first one applied to `logits.T`. The check compares your function with PyTorch on a new matrix.
#
# </details>

# %% [markdown]
# ### Step 6.2. Temperature: the scale decides how sharp the softmax is
#
# CLIP multiplies the cosines by a scale (the inverse of a temperature) before the softmax. The scale decides how sharp the distribution is. We use the matrix of Step 6.1 with a wrong pair in it, and compute the mean probability of the right partner and the loss for scales 1, 5, 10, 30 and 100.

# %%
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

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].semilogx([r_[0] for r_ in rows], [r_[1] for r_ in rows], "o-"); ax[0].set_title("mean probability of the right partner"); ax[0].set_xlabel("scale")
ax[1].semilogx([r_[0] for r_ in rows], [r_[2] for r_ in rows], "o-r"); ax[1].set_title("contrastive loss"); ax[1].set_xlabel("scale")
plt.tight_layout(); plt.show()
RESULTS["p6_scale_losses"] = {r_[0]: round(r_[2], 3) for r_ in rows}

# %% [markdown]
# **What you should see.**
#
# The table: at scale 1 the mean probability of the right partner is 0.391 and the loss 0.952; at 5 it is 0.531 and 0.803 (the lowest loss); at 10 it is 0.530 and 1.048; at 30 it is 0.494 and **2.589**; at 100 it is 0.478 and **8.302**. A uniform guess would score 1.099. The curve is a U: with a **scale that is too low** (1) every probability is soft and the loss stays near the uniform guess; with a **scale that is too high** the right pairs get near 1 but the one wrong pair (picture 3) is punished hard: the probability of its right partner is almost 0, and minus its log is large (8.3 on average at scale 100). Real CLIP learns its scale during training and ends at 100 (Step 6.5 prints it); our toy model starts at 10 and learns 12.7 (Step 6.3).
#
# This is a **brittleness** argument for a high scale: it only pays when the pairs are mostly right. In a batch with mislabelled or ambiguous pairs (a caption that fits two pictures), a high scale amplifies their loss.
#
# Tips and common mistakes:
#
# - The best scale here (5) depends on the example; it is not a general rule. The learnable `log_scale` finds a good value for the data automatically.
# - Why the log of the scale is learned, not the scale: the log is free to be any real number, the scale stays positive.
#
# Try also (optional):
#
# - Fix the third picture (see Step 6.1's Try also) and run the table again: does the best scale change?

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.2
#
# Store the scale with the lowest loss in `best_scale`. Why it matters: it makes you read the whole table instead of assuming that bigger is better.

# %%
best_scale = None   # YOUR CODE: the scale from `scales` with the lowest loss in `rows`

# %%
if best_scale is None:
    print("Fill in best_scale above, then run this cell again.")
else:
    ref = min(rows, key=lambda r_: r_[2])[0]
    check("the scale with the lowest loss", best_scale == ref, "look at the third number of each tuple in rows")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_scale = min(rows, key=lambda r_: r_[2])[0]
# print(best_scale)
# ```
#
# The tuple with the smallest third value is (5, 0.531, 0.803). We get 5.
#
# </details>

# %% [markdown]
# ### Step 6.3. A miniature CLIP from scratch, with two combinations held out
#
# Now we build and train a miniature CLIP. The image encoder is a small CNN that maps a 16 x 16 colour picture to 16 numbers. The text encoder embeds the two words of the caption (a colour word and a shape word, 6 words in total), averages the two embeddings and applies a linear layer to get 16 numbers. Both outputs are normalised to length 1, and a learnable `log_scale` (starting at ln 10) sets the temperature. We train with the symmetric contrastive loss on batches of 32 pairs for 500 steps (Adam, learning rate 0.003). The crucial design choice: **two combinations are removed from the training pictures**, 'green cross' and 'blue circle'. The model sees every colour and every shape, but never these pairs. One more detail: with only seven different captions, a batch of 32 contains the same caption several times, and the plain loss would punish the model for ranking a picture of a red circle next to a different red circle's caption. The helper `clip_loss(logits, ids)` hides these same-caption pairs in the softmax.

# %%
Xc, col, shp, caps = make_colored_shapes(1200, seed=1)         # 1,200 pictures with captions such as "red circle"
Xc_te, col_te, shp_te, caps_te = make_colored_shapes(450, seed=2)
print(Counter(caps).most_common(3), "...", len(set(caps)), "different captions")
show_images(Xc, titles=caps, n=10)
HELD = [("green", "cross"), ("blue", "circle")]                  # the two combinations the model must never see in training
held_ids = {COLOURS.index(c) * 3 + SHAPES3.index(s) for c, s in HELD}   # caption index = colour * 3 + shape
combo = col * 3 + shp
print("held-out caption indices:", sorted(held_ids))

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.3
#
# Build the boolean mask `keep` that is True for the pictures whose combination is not held out (use `np.isin(combo, list(held_ids))` and `~`). Why it matters: a held-out test is only fair if the held-out pictures were really never used for training; this one line decides it.

# %%
keep = None   # YOUR CODE: a boolean array over the 1,200 pictures that is True for pictures whose combination is NOT held out (use np.isin(combo, list(held_ids)) and ~)

# %%
if keep is None:
    print("Fill in keep above, then run this cell again.")
else:
    check("number of training pictures", int(np.sum(keep)) == int(np.sum(~np.isin(combo, list(held_ids)))), "keep = ~np.isin(combo, list(held_ids))")
    check("no held-out picture is kept", not np.isin(combo[keep], list(held_ids)).any(), "the held-out combinations must be removed")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# keep = ~np.isin(combo, list(held_ids))
# print(int(keep.sum()), "of", len(keep), "pictures are kept")
# ```
#
# `np.isin(combo, list(held_ids))` is True for pictures of the held-out combinations; the `~` flips it. The check counts the training pictures (926) and tests that none of the held-out ones are in it. The next cell uses its own `train_mask`, so it works even if yours is unfinished.
#
# </details>

# %%
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

# %%
plt.figure(figsize=(5, 3)); plt.plot(losses63); plt.xlabel("step"); plt.ylabel("contrastive loss"); plt.title("Toy CLIP training"); plt.tight_layout(); plt.show()
assert losses63[-1] < losses63[0], "the loss should go down"

# %% [markdown]
# **What you should see.**
#
# The data has 1,200 pictures and 9 caption classes with about 130 pictures each (the three most common are red cross 148, blue square 143, blue circle 140). The held-out caption indices are 5 (green cross) and 7 (blue circle), which leaves **926 training pictures** and 274 held out. The training prints a loss of **3.924 at step 0** (the random model, with 32 pairs of which many look alike) falling to **0.012 at step 100** and **0.000 at step 499**, while the learned scale rises from 10.0 to 12.7. The loss curve falls fast in the first 100 steps and then flattens: for seven captions on clean pictures the model has nothing left to learn. The assert checks that the loss went down.
#
# Tips and common mistakes:
#
# - A loss of exactly 0.000 means that the right partner gets a probability of nearly 1 for every pair of the batch; on the clean, nearly noise-free pictures that is plausible, and it does not mean the model is good on pictures it never saw (Step 6.4).
# - `clip_loss(..., ids=None)` is the plain loss. Real CLIP trains on millions of different captions, so repeated captions in a batch are rare, and it does not need the mask.
#
# Try also (optional):
#
# - Train for 100 steps only. How much of the final accuracy of Step 6.4 do you already have?
# - Remove the mask (`ids=None`) and compare the final loss. It cannot fall as low, because identical captions are counted as errors.

# %% [markdown]
# ### Step 6.4. Zero-shot with the toy CLIP: seen and unseen combinations
#
# Zero-shot classification with the toy model: we encode all nine captions (the two unseen ones too), encode the 450 test pictures, and give each picture the caption with the highest cosine. We report the accuracy separately for pictures of the seven seen combinations and the two unseen ones, and what the model says for each unseen caption.

# %%
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

# %%
for t in sorted(held_ids):
    m = te_true == t
    print("true caption %-12s (%3d pictures) -> predicted:" % (names9[t], m.sum()), {names9[k]: int(v) for k, v in Counter(te_pred[m]).most_common()})

# %% [markdown]
# **What you should see.**
#
# On the 351 test pictures of **seen** combinations the accuracy is **1.000**; on the 99 pictures of the **unseen** combinations it is **0.394**, against a chance of 0.111. **The colour is right on all 99 pictures (1.000), the shape on only 39 percent.** The breakdown: the 51 pictures of green crosses are named correctly 39 times and called 'green circle' 12 times; the 48 pictures of blue circles are **all** named 'blue square' (48 of 48). So the model does generalise partly: it never confuses the colour, and for green crosses it does well. It fails completely on blue circles. Your Imagine-first guess: not everything or nothing, but a partial yes.
#
# Why it can generalise: the text encoder builds the caption vector from the colour word and the shape word, which it has seen in other pairs ('green' with squares and circles, 'cross' in red and blue), so it can produce a vector for 'green cross' that it never trained on. Why it can fail: the image encoder has never seen a blue circle, and a small CNN trained on 926 pictures maps it to the nearest picture it knows, a blue square (a circle of radius 4 and a square differ in only a few corner pixels at 16 x 16). The language side was ready, the image side was not.
#
# Tips and common mistakes:
#
# - The result depends on which combinations are held out, on the seed and on the CNN. If you hold out other combinations the unseen accuracy changes; try it.
# - 'Compositional generalisation' (understanding a new combination of known parts) is one of the reasons the CLIP idea is attractive, and one of the places where it is known to be unreliable.
#
# Try also (optional):
#
# - Hold out 'red square' and 'green circle' instead and re-run Steps 6.3 and 6.4. Is the unseen accuracy higher or lower than 0.394?
# - Print the 9 x 9 cosine matrix of the captions to see which captions the text encoder places next to each other.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.4
#
# Compute the accuracy on the unseen pictures yourself and store it in `acc_unseen_by_hand`: the share of unseen-combination test pictures for which `te_pred` equals `te_true` (use the boolean array `unseen`). Why it matters: it is the number the whole experiment is about.

# %%
acc_unseen_by_hand = None   # YOUR CODE: the share of unseen-combination test pictures for which te_pred equals te_true (use the boolean array `unseen`)

# %%
if acc_unseen_by_hand is None:
    print("Fill in acc_unseen_by_hand above, then run this cell again.")
else:
    check("accuracy on unseen combinations", abs(acc_unseen_by_hand - acc_unseen) < 1e-9, "(te_pred == te_true)[unseen].mean()")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# acc_unseen_by_hand = float((te_pred == te_true)[unseen].mean())
# print(round(acc_unseen_by_hand, 3))
# ```
#
# the mean of the correct predictions (`te_pred` equal to `te_true`) over the pictures selected by `unseen` gives 0.394: 39 of 99 pictures.
#
# </details>

# %% [markdown]
# ### Step 6.5. The real CLIP: nine pictures, nine prompts, three templates
#
# Now the real CLIP. `load_clip()` returns the model and its processor; the first call downloads about 600 MB (on the Spark it reads the offline cache; if loading fails, the cell prints a clear message and the later cells skip). We draw nine clean 224 x 224 pictures with `draw_shape_pil` (red, green and blue; circle, square and triangle), encode them and the nine prompts of each of three templates, compute the 9 x 9 cosine matrix, and read the top-1 accuracy per template. We also print the zero-shot probabilities for one picture, using the model's own `logit_scale`.

# %%
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

# %%
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

# %%
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

# %% [markdown]
# **What you should see.**
#
# CLIP-B/32 loads with about **151 million parameters**. The accuracies are **0.889 for 'a photo of a {} {}' (8 of 9 right), 0.889 for 'a {} {}', and 1.000 for 'a picture of a {} {} on a white background'**. The wording of the prompt changes the result: the third template describes what the pictures really look like (a plain background), and CLIP gets all nine. The one error of the first two templates is the red square, which is read as a red triangle. The model's `logit_scale.exp()` is **100.0**, the maximum allowed in training, as in Step 6.2. The probabilities for the red circle are 0.781 for 'red circle', 0.124 for 'red triangle' and 0.057 for 'blue circle': the right answer but far from certain. The heat map shows a bright diagonal (right pairs have a mean cosine of 0.335, wrong pairs 0.260): in CLIP, cosines between a picture and a text are small numbers (a cosine of 0.33 is a good match) and what counts is the order, which is why the scale of 100 is needed.
#
# Tips and common mistakes:
#
# - If loading fails, read the printed message: on Colab it is usually the internet connection, on the Spark the offline cache. The other cells of this part still work; only the CLIP cells are skipped.
# - The exact probabilities can differ in the third digit between library versions; the ranking of the nine labels should not.
#
# Try also (optional):
#
# - Draw the shapes in other colours that CLIP knows ('orange', 'purple', 'yellow') and see how many of the nine it gets.
# - Try the template 'this is a {} {}' and 'a {} {} shape'.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.5
#
# Write `top1(img_emb, txt_emb)`, which returns for every picture the index of the best-matching text (both inputs are normalised, rows are items). Why it matters: it is the whole zero-shot classifier of CLIP in one line.

# %%
def top1(img_emb, txt_emb):
    """Index of the best-matching text for every picture (both are L2-normalised; rows are items)."""
    return None   # YOUR CODE: cosine matrix = img_emb @ txt_emb.T, then argmax over the texts

# %%
if clip_model is None:
    print("CLIP is not loaded: skipped.")
elif top1(zi, zt) is None:
    print("Fill in top1 above, then run this cell again.")
else:
    check("same as the cosine argmax", bool((top1(zi, zt) == (zi @ zt.T).argmax(1)).all()), "(img_emb @ txt_emb.T).argmax(1)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def top1(img_emb, txt_emb):
#     return (img_emb @ txt_emb.T).argmax(1)
# print(top1(zi, zt) if clip_model is not None else "CLIP not loaded")
# ```
#
# With normalised vectors the cosine matrix is `img_emb @ txt_emb.T`, and the prediction is the arg max over the texts. The check compares with the same expression on the nine pictures; the nine predictions are 0, 2, 2, 3, 4, 5, 6, 7, 8 for the first template: picture 1 (red square) is read as number 2 (red triangle).
#
# </details>

# %% [markdown]
# ### Step 6.6. Prompt wording and the limits of CLIP
#
# A good test of a model is where it fails. Three probes of the real CLIP. First, prompts that include something silly ('a photo of a dog') for a red circle, and a set of prompts in which no prompt is right; CLIP must still pick one. Second, counting: pictures with 2 to 6 black circles and prompts 'a picture of {two, three, four, five, six} black circles'. Third, text in the image: the words STOP and GO drawn with PIL and the prompts 'the word STOP' and 'the word GO'.

# %%
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

# %%
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

# %%
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

# %% [markdown]
# **What you should see.**
#
# (1) For a red circle, with a dog, a cat, 'red circle' and 'blue square' as options, 'a photo of a red circle' gets **0.994**; the dog gets 0.001. When **no** prompt is right (dog, cat, car), CLIP does not say 'none': it gives cat 0.502, dog 0.272 and car 0.226. A zero-shot classifier **always answers with one of the labels you give it**; the probabilities sum to 1 whatever the picture shows. (2) **Counting fails**: CLIP says 'three' for all five pictures (2, 3, 4, 5, 6 circles) and is right once (for the picture with 3 circles) in 5. Counting needs more than a similarity between a picture and a sentence. (3) **Words in pictures work**: the picture of STOP gets 0.996 for 'the word STOP', the picture of GO gets 0.989 for 'the word GO'. CLIP learned to read short words because captions on the internet often repeat the text that is visible in the picture. This is also why it can be fooled by a label stuck on an object (a typographic attack).
#
# Tips and common mistakes:
#
# - The rank order of the labels is what the model offers; do not read 0.502 as 'CLIP is 50 percent sure it is a cat'. The probabilities depend on the labels you chose.
# - For a real system add a 'none of the above' label and a threshold on the cosine, and let a person check what falls below it.
#
# Try also (optional):
#
# - Test 'a photo of a red square' against the red circle: is CLIP separating squares from circles at all?
# - Draw the word STOP in red on a green background and test again.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.6
#
# Write 3 to 5 sentences (at least 25 words) in `answer_66`: why can the captions CLIP learned from make it biased or wrong, and what would you check before using it in a school project? Why it matters: the previous probes show failures you can see; biases in the data are failures you have to look for.

# %%
answer_66 = ""   # YOUR ANSWER (3 to 5 sentences): why can the web captions CLIP learned from make it biased or wrong, and name one thing a person should check before using it in a school project

# %%
if len(answer_66.split()) < 25:
    print("Write at least 25 words in answer_66, then run this cell again.")
else:
    check("answer written", True)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_66 = ("CLIP learned from captions that people wrote on the internet, so it repeats what the web shows often: some countries, jobs, skin colours "
#              "and genders are described more, or more stereotypically, than others, and a caption rarely says the obvious. Before using it I would test it on "
#              "pictures from my own community and look at the accuracy for each group, not only the average, and keep a person in the loop.")
# print(len(answer_66.split()), "words")
# ```
#
# A good answer says that the captions come from the internet, which shows some groups, places and jobs far more often or more stereotypically than others, and that a caption rarely states the obvious; it proposes a test on pictures from your own community with accuracy per group (Step 7.3), not only the average, and a human in the loop.
#
# </details>

# %% [markdown]
# ### Part 6 checkpoint: Contrastive learning and CLIP
#
# You should now have:
#
# - the contrastive loss by hand (1.0481 for the example), equal to PyTorch's, and the effect of the scale (6.1 and 6.2)
# - a toy CLIP trained from scratch: 1.000 on seen and 0.394 on unseen combinations, with the reason (6.3 and 6.4)
# - the real CLIP on nine drawn shapes (0.889 and 1.000 depending on the prompt), its probabilities and its failures on counting and unusual prompts (6.5 and 6.6)

# %% [markdown]
# ## Part 7. Design a multimodal system, and reflect (15 min)
#
# *Lesson sections: all of Week 11.* You have patchified pictures, built a ViT, tested positions and trained a small CLIP. In this last part you stop building and start designing: you choose a real problem from your school or community, fill in a design card for a multimodal system, count its token budget with your own `vit_counts`, test how a model can be fair or unfair to groups, print your results, and answer four reflection questions. ==The goal is to show that you can connect the architecture to a real decision: what goes in, how it is fused, how it is checked, and who could be harmed.==

# %% [markdown]
# ### Step 7.1. Imagination design: a multimodal system for your school or community
#
# Choose a real problem. Examples: alt text for the pictures of the school newsletter; a flood-gauge photo plus a text report from a barangay volunteer; a voice-and-text helpdesk for enrolment questions; recognising jeepney routes from a photo of the signboard. Fill in the design card: what comes in, what comes out, which fusion pattern you would use (a CLIP-style shared space, a projector into a language model, cross-attention, or late fusion, with one sentence of why), where the data comes from and who agreed, how you evaluate each modality plus a human check, what the system does when it is unsure, and one risk. The printed card is part of your submission.

# %%
import textwrap
def show_card(design):
    print("=" * 70)
    for k, v in design.items():
        print("%-18s %s" % (k.replace("_", " ").upper() + ":", textwrap.fill(v or "(empty)", 50, subsequent_indent=" " * 19)))
    print("=" * 70)

# %% [markdown]
# **What you should see.**
#
# The cell prints your card with the eight fields in capitals. The check wants at least four words per field and a fusion pattern that names one of the four options. The sample answer (flood-gauge photos with text reports) uses a projector into a language model; yours should say why your choice fits your problem: a shared space such as CLIP is good for search and classification by label, a projector is good when the answer is free text, cross-attention is good when two inputs must be aligned in detail, and late fusion is the simplest and the easiest to check.
#
# Tips and common mistakes:
#
# - Be specific: 'accuracy' is not an evaluation, 'mean error in centimetres on 100 photos from last season plus a person who checks every reading above the warning level' is.
# - Consent and privacy are part of the design: photos of children or houses need a plan before the first picture is taken.
#
# Try also (optional):
#
# - Write a second card for the same problem with a different fusion pattern and compare the risks.

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.1
#
# Fill in the dictionary `design` for your own problem. Why it matters: a model without a design around it is not a system; the card is where the risks become visible before they happen.

# %%
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

# %%
show_card(design)
empty = [k for k, v in design.items() if len(v.split()) < 4]
if empty:
    print("Write at least 4 words for:", empty)
else:
    check("the card is complete", True)
    check("the fusion pattern is one of the four", any(w in design["fusion"].lower() for w in ["clip", "projector", "cross-attention", "late fusion", "shared space"]), "name one of: CLIP-style shared space, projector into a language model, cross-attention, late fusion")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# design = {
#     "problem": "Flood-gauge photos and short text reports from barangay volunteers arrive by chat, and the office reads them by hand.",
#     "modalities_in": "A photo of the gauge board and a text message in English or Filipino.",
#     "modalities_out": "A water level in centimetres with a confidence, and a one-line text summary for the office.",
#     "fusion": "Projector into a language model: a ViT turns the photo into tokens, a projector maps them into the model's embedding space, and the text report is read together with them.",
#     "data_and_consent": "Photos taken by volunteers who agreed in writing; no faces or house numbers kept; 300 labelled photos from last season.",
#     "evaluation": "Mean error in centimetres on photos, accuracy of the text summary against a person's note, and a human check of every reading above the warning level.",
#     "failure_handling": "If the confidence is below 0.7 or the photo is dark, the system asks the volunteer for another photo and alerts a person.",
#     "risk": "A wrong low reading in a storm could delay an evacuation, so a person must confirm every high-risk decision.",
# }
# show_card(design)
# ```
#
# The check prints your card and tests it for completeness. The cell below the exercise in the notebook contains a worked example (`Show a possible answer`).
#
# </details>

# %% [markdown]
# ### Step 7.2. A token budget for your design
#
# A design needs a budget. For an image of your chosen side and patch size, the number of visual tokens is (side / patch) squared; if the image tokens go into a language model together with a prompt of 64 tokens, the sequence has visual plus 64 tokens, and each attention head computes that number squared in scores. We use your `vit_counts` from Step 4.3 with the defaults of a typical visual encoder (336 pixels, patch 14), and compare with a patch half as large.

# %%
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

# %% [markdown]
# **What you should see.**
#
# For 336 x 336 pixels with patch 14: **576 visual tokens** (24 x 24) of 588 numbers each (14 x 14 x 3), a sequence of **640** with the prompt and 409,600 attention scores per head. With patch 7: **2,304 tokens** of 147 numbers, a sequence of **2,368** and **5,607,424** scores: halving the patch size multiplies the scores by **13.7** (not 16, because the 64 prompt tokens do not grow). This is why systems that read documents or screenshots, which need small patches to read small text, are expensive, and why they often split the picture into tiles or compress the visual tokens before they reach the language model.
#
# Tips and common mistakes:
#
# - Change `my_image_px` and `my_patch` to the values of your design; the assert reminds you that the side must be a multiple of the patch size.
# - The numbers per patch (588 or 147) do not enter the attention cost; they only set the size of the patch embedding.
#
# Try also (optional):
#
# - Double the image side instead of halving the patch. How many tokens do you get? (The same as halving the patch: 2,304.)

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.2
#
# Store the sequence length of your design (visual tokens plus the prompt tokens) in `seq_len_mine`. Why it matters: it is the number that sets the memory and the time of your system.

# %%
seq_len_mine = None   # YOUR CODE: the sequence length of your design: visual tokens for my_image_px and my_patch plus prompt_tokens

# %%
if seq_len_mine is None:
    print("Fill in seq_len_mine above, then run this cell again.")
else:
    check("sequence length", seq_len_mine == (my_image_px // my_patch) ** 2 + prompt_tokens, "(side // patch) squared tokens, plus the prompt tokens")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# seq_len_mine = (my_image_px // my_patch) ** 2 + prompt_tokens
# print(seq_len_mine)
# ```
#
# `(my_image_px // my_patch) ** 2 + prompt_tokens`: 24 x 24 + 64 = 640 with the defaults.
#
# </details>

# %% [markdown]
# ### Step 7.3. A mini evaluation: accuracy per group and the worst group
#
# A single accuracy hides who the system fails. You write `group_accuracy(pred, true, groups)`, the accuracy for each group, and apply it to the toy CLIP of Part 6: per colour of the picture, per shape, and for seen against unseen combinations. The worst group is the one a fairness check looks at first. In the exercise you write the function; the cell below it uses a reference version, so the table is printed even if yours is unfinished.

# %%
te_cols = [COLOURS[c] for c in col_te]; te_shps = [SHAPES3[s] for s in shp_te]

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.3
#
# Write `group_accuracy(pred, true, groups)`: for every distinct group name the share of its pictures with `pred` equal to `true`. Why it matters: it is the first tool of any fairness check.

# %%
def group_accuracy(pred, true, groups):
    """pred and true: arrays of caption indices; groups: a list with one group name per picture. Returns {group name: accuracy}."""
    return {}   # YOUR CODE: for every distinct group name, the share of its pictures with pred equal to true

# %%
out = group_accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), ["a", "a", "b", "b"])
if not out:
    print("Fill in group_accuracy above, then run this cell again.")
else:
    check("small test", out == {"a": 1.0, "b": 0.5}, "for each group take the mean of (pred == true) over the pictures of that group")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def group_accuracy(pred, true, groups):
#     out = {}
#     for g in sorted(set(groups)):
#         m = np.array([x == g for x in groups])
#         out[g] = float((pred[m] == true[m]).mean())
#     return out
# print(group_accuracy(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]), ["a", "a", "b", "b"]))
# ```
#
# Loop over `sorted(set(groups))`, make a boolean mask of the pictures of the group, and take the share of those pictures where `pred` and `true` agree. The check uses a four-picture example: group a is 1.0, group b is 0.5.
#
# </details>

# %%
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

# %% [markdown]
# **What you should see.**
#
# The toy CLIP has an accuracy of **1.000 on red pictures, 0.920 on green and 0.680 on blue**; per shape it is 1.000 for squares, 0.920 for crosses and 0.680 for circles; and 1.000 on seen against 0.394 on unseen combinations. The **worst colour group is blue (0.680)**, and the gap to the best (red) is 0.320. The average over all test pictures, about 0.87, would have hidden this. The reason is the one you know from Step 6.4: the blue circle was held out, and the blue pictures contain the unseen combination blue circle, while green contains green cross. In a real system the groups are people, languages, accents, skin tones or neighbourhoods, and the same reasoning applies: report the accuracy for every group that matters, not only the mean. A fairness check for your design would list the groups (for the flood-gauge system: day and night, rain and dry weather, each barangay), measure the error per group on enough examples, set a maximum gap you accept, and say who decides what to do if a group is above it.
#
# Tips and common mistakes:
#
# - A group with few pictures has a noisy accuracy; always print how many pictures each group has before you compare.
# - Groups that are not in your data are invisible to this check: the biggest risks are often the people who never took part in the data collection.
#
# Try also (optional):
#
# - Print the accuracy for each of the nine combinations. Which two are the worst, and why?

# %% [markdown]
# ### Step 7.4. Your RESULTS summary
#
# Print everything you stored in `RESULTS` during the lab. It is your summary for the submission: if an entry is missing, the step that stores it has not been run.

# %%
for k, v in RESULTS.items(): print("%-26s %s" % (k, v))
print(len(RESULTS), "stored results")

# %% [markdown]
# **What you should see.**
#
# You see about 20 entries of this part (plus the entries of Parts 1 to 3 if you ran them): the patch embedding shape (8, 16, 32), the 26,757 parameters of the tiny ViT and 85,798,656 for ViT-Base, the overfitting accuracy 1.0, ViT against CNN (1.0, 1.0), the central attention 0.33, the left/right table (CNN 1.0, ViT 0.993, no positions 0.537), the shuffle agreements 1.0 and 0.513, the pooling table, the patch-size table, the data-hunger lists, the sine-cosine accuracy 1.0, the loss by hand 1.0481, the scale losses, the toy CLIP's training loss (3.924 to 0.0) and zero-shot accuracies (1.0 and 0.394), CLIP's template accuracies, the counting score 1, the token budget (576, 640, 2304, 2368) and the worst group (blue, 0.68).
#
# Tips and common mistakes:
#
# - Copy the printed block into your submission together with your written answers.

# %% [markdown]
# ### Step 7.5. Four reflection questions
#
# Four questions, 25 words or more each. Cite a number from your own output where you can.

# %% [markdown]
# **What you should see.**
#
# The questions are: (1) what is a multimodal model and what are three real applications; (2) how does a ViT turn an image into tokens; (3) why are position embeddings needed; (4) how does CLIP make zero-shot classification possible, and what could go wrong. Good answers use the lab's numbers: 196 tokens of 768 numbers for a 224 x 224 picture with 16 x 16 patches; 0.993 against 0.537 for the ViT with and without positions; 0.394 on unseen combinations for the toy CLIP; and the failures of the real CLIP (it always answers, it cannot count, it repeats the biases of its captions).
#
# Tips and common mistakes:
#
# - The check counts only words. Quality is judged by your instructor with the rubric.

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.5
#
# Write your four answers in the dictionary `reflection`. Why it matters: explaining it in your own words is the best test that you understood it.

# %%
reflection = {
    "q1_multimodal": "",   # What is a multimodal model? Name three real applications.
    "q2_vit_tokens": "",   # How does a ViT turn an image into tokens? Use the numbers of Part 2 or Part 4.
    "q3_positions":  "",   # Why are position embeddings needed? Cite your left/right result of Part 5.
    "q4_clip":       "",   # How does CLIP make zero-shot classification possible, and what could go wrong?
}

# %%
short = [k for k, v in reflection.items() if len(v.split()) < 25]
if short:
    print("Write at least 25 words for:", short)
else:
    check("four reflection answers written", True)
    RESULTS["p7_reflection_words"] = {k: len(v.split()) for k, v in reflection.items()}

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reflection = {
#     "q1_multimodal": "A multimodal model handles more than one kind of data, such as pictures, text and sound, in one system. Real examples are image captioning, visual question answering and speech recognition with a text correction step.",
#     "q2_vit_tokens": "A ViT cuts the picture into square patches, flattens each patch into a vector and maps it with one linear layer to a token; a 224 by 224 colour picture with 16 by 16 patches gives 196 tokens of 768 numbers, plus a CLS token.",
#     "q3_positions": "Self-attention alone treats the tokens as a bag, so shuffling the patches changes nothing; in my experiment the ViT without positions stayed near chance on the left/right task while the one with positions learned it.",
#     "q4_clip": "CLIP learns a shared space where a picture and its caption are close, so a class name written as a prompt can be compared with the picture by cosine; it can go wrong with unusual wording, counting, and the biases of internet captions.",
# }
# print({k: len(v.split()) for k, v in reflection.items()})
# ```
#
# A good answer is specific, short and uses numbers; see the sample answers.
#
# </details>

# %% [markdown]
# ### Step 7.6. Go further
#
# Two tutorials of the University of Amsterdam Deep Learning course continue where this lab stops. Tutorial 15, Vision Transformers (`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html`), builds a ViT for CIFAR10 from scratch, trains it, compares it with a CNN and explains why the ViT needs far more data, which is what you saw in Step 5.5 on a small scale. Tutorial 17, SimCLR (`https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html`), explains contrastive learning on pictures with the same kind of loss as Step 6.1, but with two augmented views of the same picture as the positive pair instead of a picture and a caption. Credit: the tutorials are by Phillip Lippe and are free to read; they are not part of the marked work and no code from them is copied here.

# %% [markdown]
# ### Part 7 checkpoint: Design a multimodal system, and reflect
#
# You should now have:
#
# - a design card for a multimodal system with its fusion pattern, evaluation, failure handling and risk (7.1)
# - a token budget for your design and the cost of halving the patch (7.2)
# - accuracy per group and the worst group, your printed RESULTS and four reflection answers (7.3 to 7.5)

# %% [markdown]
# ## Wrap-up
#
# **What you learned in Parts 4 to 7**
#
# - A Vision Transformer is a patch embedding (a Conv2d with stride equal to the patch size), a CLS token, position embeddings, a Transformer encoder and a head: 26,757 parameters for your tiny one, 85.8 million for ViT-Base. Without positions it is blind to layout (0.537 on the left/right task against 0.993 with positions), and a ViT trained from scratch needs more data than a CNN (0.43 against 0.89 with 50 pictures).
# - CLIP trains two encoders with a symmetric contrastive loss so that a picture and its caption land close together; the scale sets the sharpness (a U-shaped loss in your table), a toy version got the colour of unseen combinations right but their shape only 39 percent of the time, and the real CLIP reads colours, shapes and short words but cannot count, always answers with one of your labels and repeats the biases of its captions.
# - A multimodal system is a design: inputs, fusion pattern, token budget (576 visual tokens for a 336 pixel image with patch 14), evaluation per group (the worst colour group had 0.68 accuracy while the mean looked good) and failure handling with a person in the loop.
#
# **Reflect.** Double-click this cell and write two sentences: which result in this lab was most different from what you guessed in an Imagine-first cell, and what would you measure next?
#
# **Next week (Week 12)** is image generation and GANs: after models that read pictures, models that draw them.
#
# **Before you close this tab.** Files on the Colab runtime disappear when the session ends. Use **File → Save a copy in Drive** (and download the notebook with its outputs if you want a copy on your computer). Submit your work the way your instructor asks. If you want to go further, the two University of Amsterdam tutorials of Step 7.6 continue where this lab stops.
