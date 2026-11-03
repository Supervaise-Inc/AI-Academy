# %% [markdown]
# # Week 11 — Multimodal AI: lab notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive**, so your changes are kept in your own copy.
#
# **Runtime:** CPU is enough. (A GPU is not needed: everything here is small.) **Time:** about 3 hours. **Download:** one model, CLIP ViT-B/32, about 0.6 GB, in Step 6.
#
# **What you will do**
# 1. Cut a small image into patches with `reshape`, and check the result against PyTorch's `unfold`.
# 2. Count tokens and parameters for real image sizes.
# 3. See why a Vision Transformer needs position embeddings.
# 4. Train a tiny ViT on handwritten digits.
# 5. Draw simple pictures with PIL and classify them with CLIP, with no training (zero-shot).
# 6. Compare prompt wordings.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.

# %%
import os, math, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
QUICK = bool(os.environ.get("NB_TEST"))

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1 — Patchify an image
#
# In the lesson section "Vision Transformers", a ViT cuts an image into patches and treats each patch as a token. We start with a small 8 × 8 handwritten digit from scikit-learn (no download). The trick is `reshape` to split the rows and columns into a grid of tiles, then `transpose` so each tile's pixels sit together.

# %%
digits = load_digits()
img = digits.images[0]                      # shape (8, 8), values 0 to 16
p = 2                                       # patch side
n = 8 // p                                  # patches per row
patches = img.reshape(n, p, n, p).transpose(0, 2, 1, 3).reshape(-1, p * p)
print(patches.shape)                        # (16, 4)
print(patches[1], img[0:2, 2:4].reshape(-1))   # patch 1 = rows 0-1, columns 2-3: the same four numbers
assert np.array_equal(patches[1], img[0:2, 2:4].reshape(-1))

plt.figure(figsize=(2.2, 2.2)); plt.imshow(img, cmap="gray_r"); plt.title("digit " + str(digits.target[0])); plt.axis("off"); plt.show()

# %% [markdown]
# ### ✏️ Your turn: other patch sizes
#
# Repeat for `p = 4` and `p = 8`. Set `table` to a dictionary `{p: (number_of_patches, numbers_per_patch)}` for p = 2, 4 and 8. Then notice what patches × numbers per patch is every time.

# %%
table = None   # YOUR CODE HERE, e.g. {2: (16, 4), 4: (?, ?), 8: (?, ?)}

# %%
# Check
if table is None:
    print("Fill in your answer above, then run this cell again.")
else:
    for q in (2, 4, 8):
        m = 8 // q
        pt = img.reshape(m, q, m, q).transpose(0, 2, 1, 3).reshape(-1, q * q)
        check(f"p={q}: {pt.shape}", tuple(table.get(q, ())) == pt.shape, "patches = (8/p)^2 and numbers per patch = p*p")
    check("patches x numbers = 64 each time", all(a * b == 64 for a, b in table.values()), "cutting into patches only reorganises the 64 numbers")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# table = {2: (16, 4), 4: (4, 16), 8: (1, 64)}
# ```
#
# Smaller patches mean more tokens with fewer numbers each. The product is always 64, because patchifying loses no information.
# </details>

# %% [markdown]
# ## Step 2 — The same thing with `unfold`
#
# PyTorch can do the cutting in one call: `F.unfold` slides a window and, with `stride` equal to the window size, gives non-overlapping patches. This is the same operation that UvA's Tutorial 15 does in its `img_to_patch` helper (see "Go further" at the end).

# %%
x = torch.tensor(img, dtype=torch.float32)[None, None]       # (batch=1, channels=1, 8, 8)
u = F.unfold(x, kernel_size=p, stride=p)                     # (1, p*p, number_of_patches)
print(u.shape)                                               # torch.Size([1, 4, 16])
u_patches = u[0].T                                           # (16, 4): one row per patch
print("same as reshape:", torch.equal(u_patches, torch.tensor(patches, dtype=torch.float32)))
assert torch.equal(u_patches, torch.tensor(patches, dtype=torch.float32))

# %% [markdown]
# ## Step 3 — Count tokens and parameters for a real size
#
# From the lesson: a 224 × 224 colour image with 16 × 16 patches gives 14² = 196 patches of 16 × 16 × 3 = 768 numbers. With a hidden size D = 768, the linear projection has 768 × D + D weights, and adding the CLS token gives 197 tokens. We write a small function so you can try other sizes.

# %%
def vit_counts(side, patch, channels=3, D=768):
    n_patches = (side // patch) ** 2
    per_patch = patch * patch * channels
    tokens = n_patches + 1                      # + CLS token
    return {
        "patches": n_patches,
        "numbers per patch": per_patch,
        "projection params": per_patch * D + D,
        "tokens": tokens,
        "position params": tokens * D,
        "attention scores per head": tokens * tokens,
    }

base = vit_counts(224, 16)
print(base)
assert base["patches"] == 196 and base["numbers per patch"] == 768
assert base["projection params"] == 590592 and base["position params"] == 151296

# %% [markdown]
# ### ✏️ Your turn: a bigger image
#
# For a 448 × 448 image with 16 × 16 patches: how many tokens (with CLS) are there, and how many attention scores per head? Then say how many times more scores there are than for 224 × 224, rounded to a whole number.

# %%
tokens_448 = None   # YOUR CODE HERE
scores_448 = None   # YOUR CODE HERE
ratio = None        # YOUR CODE HERE

# %%
# Check
if tokens_448 is None or scores_448 is None or ratio is None:
    print("Fill in your answer above, then run this cell again.")
else:
    t = vit_counts(448, 16)
    check(f"tokens = {t['tokens']}", tokens_448 == t["tokens"], "(448/16)^2 patches plus the CLS token")
    check(f"scores per head = {t['attention scores per head']}", scores_448 == t["attention scores per head"], "tokens squared")
    check(f"ratio = {round(t['attention scores per head'] / base['attention scores per head'])}", ratio == round(t["attention scores per head"] / base["attention scores per head"]), "divide by 197 x 197 = 38809")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# t = vit_counts(448, 16)
# tokens_448 = t["tokens"]                          # 785
# scores_448 = t["attention scores per head"]       # 616225
# ratio = round(scores_448 / 38809)                 # 16
# ```
#
# Doubling the side gives four times the patches, and attention compares every token with every other token, so the cost grows about 4 × 4 = 16 times. That is why high-resolution images are expensive for ViTs.
# </details>

# %% [markdown]
# ## Step 4 — Why position embeddings?
#
# Self-attention treats its input as a set: if you shuffle the tokens, you get the same outputs, only shuffled. So a ViT adds a learned **position embedding** to each patch token. Let us see both facts on a tiny Transformer layer.

# %%
torch.manual_seed(SEED)
D, n_tok = 16, 6
layer = nn.TransformerEncoderLayer(D, 2, 32, dropout=0.0, batch_first=True).eval()
tokens = torch.randn(1, n_tok, D)                 # six "patch" vectors
perm = torch.randperm(n_tok)

with torch.no_grad():
    out = layer(tokens)
    out_shuffled = layer(tokens[:, perm])
print("without positions, shuffling the input just shuffles the output:",
      torch.allclose(out[:, perm], out_shuffled, atol=1e-5))
assert torch.allclose(out[:, perm], out_shuffled, atol=1e-5)

pos = torch.randn(1, n_tok, D)                    # stand-in for a learned position embedding
with torch.no_grad():
    a = layer(tokens + pos)                       # positions attached to the original order
    b = layer(tokens[:, perm] + pos)              # same patches, new order, same position vectors
print("with positions, the shuffled patches give different results:", not torch.allclose(a[:, perm], b, atol=1e-5))

# %% [markdown]
# ### ✏️ Your turn: how many position parameters?
#
# A ViT with 32 × 32 patches on a 224 × 224 image and D = 768 has how many learned position-embedding parameters (including the CLS token's position)? Use `vit_counts`.

# %%
answer = None   # YOUR CODE HERE

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true = vit_counts(224, 32)["position params"]
    check(f"position params = {true}", answer == true, "(224/32)^2 patches + 1 CLS token, times 768")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = vit_counts(224, 32)["position params"]   # 50 tokens x 768 = 38400
# ```
#
# (224 / 32)² = 49 patches, plus the CLS token, gives 50 tokens. Each has a 768-number position vector.
# </details>

# %% [markdown]
# ## Step 5 — A tiny ViT on digits
#
# Now we put the pieces together: 2 × 2 patches of an 8 × 8 digit give 16 patches plus a CLS token, a linear projection, learned position embeddings, two Transformer layers and a small head. The digits are tiny, so this is a toy, and it trains in seconds on a CPU.

# %%
class TinyViT(nn.Module):
    def __init__(self, p=2, D=32, n_classes=10):
        super().__init__()
        self.p = p
        self.proj = nn.Linear(p * p, D)                                   # patch -> vector
        self.cls = nn.Parameter(torch.zeros(1, 1, D))                     # learned CLS token
        self.pos = nn.Parameter(torch.randn(1, (8 // p) ** 2 + 1, D) * 0.02)   # learned positions
        layer = nn.TransformerEncoderLayer(D, 4, 64, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(D, n_classes)

    def forward(self, imgs):                                              # imgs: (B, 8, 8)
        B, p = imgs.shape[0], self.p
        x = imgs.reshape(B, 8 // p, p, 8 // p, p).permute(0, 1, 3, 2, 4).reshape(B, -1, p * p)
        t = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], dim=1) + self.pos
        return self.head(self.enc(t)[:, 0])                               # read out the CLS token

X = torch.tensor(digits.images, dtype=torch.float32) / 16.0
y = torch.tensor(digits.target)
Xtr, ytr, Xte, yte = X[:1400].to(device), y[:1400].to(device), X[1400:].to(device), y[1400:].to(device)
model = TinyViT().to(device)
print(model(Xtr[:8]).shape)                                  # torch.Size([8, 10])
print("encoder parameters:", sum(q.numel() for q in model.enc.parameters()))   # 17088
assert sum(q.numel() for q in model.enc.parameters()) == 17088

# %%
def train_vit(p, epochs):
    torch.manual_seed(SEED)
    m = TinyViT(p=p).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for epoch in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(m(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(Xte).argmax(dim=1) == yte).float().mean().item()
    return loss.item(), acc

epochs = 15 if QUICK else 25
results = {}
for patch in (2, 4):
    results[patch] = train_vit(patch, epochs)
    print(f"patch {patch}x{patch}: last loss {results[patch][0]:.3f}, test accuracy {results[patch][1]:.3f}")
# Sanity check: clearly better than random guessing (10%) after training
assert results[2][1] > 0.5, "the ViT should beat random guessing"

# %% [markdown]
# ### ✏️ Your turn: which patch size worked best?
#
# Look at the accuracies above. Set `best_patch` to the patch size (2 or 4) with the higher test accuracy in **your** run, and write in a comment why you think so. (Accuracy varies with the random seed, so do not compare exact numbers with a classmate.)

# %%
best_patch = None   # YOUR ANSWER HERE: 2 or 4

# %%
# Check
if best_patch is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true = max(results, key=lambda k: results[k][1])
    check(f"best patch size in this run = {true}", best_patch == true, "compare the two test accuracies printed above")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_patch = max(results, key=lambda k: results[k][1])
# ```
#
# A possible reason: with 4 × 4 patches there are only 4 tokens, so attention has little to mix, and each patch has 16 numbers to encode. With 2 × 2 patches the model gets 16 tokens and finer detail. Whatever you saw, state it as your observation, since the digits are tiny and a ViT has little to learn from.
# </details>

# %% [markdown]
# ## Step 6 — CLIP zero-shot classification
#
# CLIP (lesson section "CLIP") has an image encoder and a text encoder trained so that a picture and its caption land close together. To classify, we write each label as a sentence, encode it, and pick the label closest to the image. We draw three simple pictures with PIL, so nothing is downloaded except the model.
#
# **Download:** `openai/clip-vit-base-patch32`, about 0.6 GB, one time per session.

# %%
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

def make(shape, color):
    im = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(im)
    box = (52, 52, 172, 172)
    if shape == "circle":
        draw.ellipse(box, fill=color)
    else:
        draw.rectangle(box, fill=color)
    return im

images = {"red circle": make("circle", "red"), "blue square": make("square", "blue"), "green circle": make("circle", "green")}
fig, ax = plt.subplots(1, 3, figsize=(6, 2.2))
for a_, (k, im) in zip(ax, images.items()):
    a_.imshow(im); a_.set_title(k, fontsize=9); a_.axis("off")
plt.show()

# %%
name = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(name).eval()
proc = CLIPProcessor.from_pretrained(name)
labels = ["a red circle", "a blue square", "a green circle", "a yellow triangle"]

def zero_shot(img, labels, template="a photo of {}"):
    inp = proc(text=[template.format(l) for l in labels], images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = clip(**inp).logits_per_image          # (1, number of labels)
    return logits.softmax(dim=1)[0]

for k, im in images.items():
    probs = zero_shot(im, labels)
    assert abs(float(probs.sum()) - 1.0) < 1e-4        # a softmax always sums to 1
    print(k, {l: round(float(v), 3) for l, v in zip(labels, probs)})

# %% [markdown]
# ### ✏️ Your turn: does the winner match?
#
# For each picture, set `predictions[name]` to the label CLIP gave the highest probability. Then, in a comment, say one way zero-shot classification could fail.

# %%
predictions = None   # YOUR CODE HERE: a dictionary like {"red circle": "a red circle", ...}

# %%
# Check
if predictions is None:
    print("Fill in your answer above, then run this cell again.")
else:
    for k, im in images.items():
        pr = zero_shot(im, labels)
        top = labels[int(pr.argmax())]
        check(f"{k}: top label is '{top}'", predictions.get(k) == top, "read the label with the largest probability in the printed row")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# predictions = {k: labels[int(zero_shot(im, labels).argmax())] for k, im in images.items()}
# ```
#
# For simple shapes and colours the matching label usually wins, but that is not guaranteed, so report what you see. One way it can fail: the correct label is missing from your list, since the probabilities only compare the labels you supply.
# </details>

# %% [markdown]
# ## Step 7 — Prompt wording
#
# The text you give CLIP matters. We try three templates on the red circle and look at the probability of the correct label.

# %%
templates = ["{}", "a photo of {}", "a drawing of {}"]
correct = "a red circle"
rows = {}
for t in templates:
    pr = zero_shot(images["red circle"], labels, template=t)
    rows[t] = float(pr[labels.index(correct)])
    print(f"{t!r:20} P(correct) = {rows[t]:.3f}")

# %% [markdown]
# ### ✏️ Your turn: which wording is best?
#
# Set `best_template` to the template with the highest probability for the correct label.

# %%
best_template = None   # YOUR CODE HERE

# %%
# Check
if best_template is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true = max(rows, key=rows.get)
    check(f"best template is {true!r}", best_template == true, "compare the three P(correct) values printed above")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_template = max(rows, key=rows.get)
# ```
#
# The template can change the probabilities noticeably, even when the winner stays the same. Try adding two labels of your own, or an image drawn with PIL (`ImageDraw.polygon` makes a triangle), and see what changes. Free memory when you are done: `del clip, proc`.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - An image becomes a sequence of tokens by cutting it into patches (`reshape` or `unfold`); smaller patches mean more tokens and attention cost that grows with the square of the token count.
# - Self-attention ignores order, so a ViT adds position embeddings; a ViT then reads the CLS token.
# - CLIP puts images and text in one space, so you can classify with text labels it was never trained on, and the wording of the labels matters.
#
# **Reflect:** double-click to write two sentences: one risk of using a model like CLIP on photos of people, and one application you would build.
#
# Files in `/content` vanish when the session ends, so download anything you want to keep. Submit this notebook the way your instructor asks.
#
# **Go further** (both download data from the internet, so run them in their own Colab or on a laptop):
# - UvA Deep Learning Tutorial 15, [Vision Transformers](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html), by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence). It builds `img_to_patch`, a CLS token, learned position embeddings and an MLP head on CIFAR10.
# - UvA Deep Learning Tutorial 17, [SimCLR](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html), by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence). Contrastive learning on STL10, the idea behind CLIP-style training.
