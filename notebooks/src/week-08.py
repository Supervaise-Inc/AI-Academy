# %% [markdown]
# # Week 8 — The Transformer & Autoregressive Training: lab notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive** so that your own work is saved in your own copy.
#
# **Runtime:** a CPU is enough (nothing to change). **Estimated time:** about 3 hours; the training cells take well under a minute each on a CPU.
#
# This is Laboratory 3, Part 2. You build the pieces of a tiny Transformer yourself and train it to predict the next character. No pretrained model and no download is needed.
#
# **What you will do**
# - Step 1: sinusoidal positional encoding in NumPy.
# - Step 2: causal multi-head attention in PyTorch, with shapes and a parameter count.
# - Step 3: a Transformer block and a tiny language model.
# - Step 4: train it on a repeated string and watch the loss fall.
# - Step 5: generate text, then remove the causal mask and see what goes wrong.
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you whether you are right. Open the **Show a possible answer** block only after you have tried.

# %% [markdown]
# ## Setup
#
# Colab already has PyTorch, NumPy and Matplotlib. The cell below imports them, fixes the seed and defines the `check` helper.

# %%
import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
QUICK = bool(os.environ.get("NB_TEST"))  # shortens long loops when the notebook is tested


def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))


# %% [markdown]
# ## Step 1: sinusoidal positional encoding
#
# Attention has no sense of order by itself: it treats the tokens as a set. The lesson section *Positional encoding* fixed this by adding a position-dependent vector to each token embedding. Position `pos` and dimension pair `i` get `sin(pos / 10000^(i/d_model))` in the even columns and `cos(...)` in the odd columns.

# %%
def positional_encoding(n_pos, d_model):
    pos = np.arange(n_pos)[:, None]                 # positions 0..n_pos-1 as a column
    i = np.arange(0, d_model, 2)[None, :]           # even dimension indices as a row
    angle = pos / (10000 ** (i / d_model))
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle)                     # even columns: sine
    pe[:, 1::2] = np.cos(angle)                     # odd columns: cosine
    return pe


pe = positional_encoding(24, 16)
print(pe.shape, pe[0, :4], pe[1, :4].round(4))
assert pe.shape == (24, 16)
assert np.allclose(pe[0, :4], [0, 1, 0, 1])
assert np.allclose(pe[1, :4], [0.8415, 0.5403, 0.3110, 0.9504], atol=1e-4)

plt.figure(figsize=(6, 3))
plt.imshow(pe, cmap="RdBu", aspect="auto")
plt.xlabel("dimension")
plt.ylabel("position")
plt.colorbar()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ✏️ Your turn: positional encoding by hand
#
# For `d_model = 4`, work out the row for position 1 by hand. Remember the angles are `1 / 10000^(0/4) = 1` and `1 / 10000^(2/4) = 0.01`. Store the four numbers (rounded to 4 decimals) in `answer`. Then look at the picture above: how do the columns change from left to right?

# %%
answer = None  # YOUR ANSWER HERE, for example [a, b, c, d]

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true_row = positional_encoding(3, 4)[1]
    check("PE(position 1, d_model=4)", len(answer) == 4 and bool(np.allclose(answer, true_row, atol=1e-3)),
          "Order is sin(1), cos(1), sin(0.01), cos(0.01).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = positional_encoding(3, 4)[1].round(4).tolist()   # [0.8415, 0.5403, 0.01, 1.0]
# ```
#
# By hand: `sin(1) = 0.8415`, `cos(1) = 0.5403`, `sin(0.01) = 0.0100`, `cos(0.01) = 1.0000`. In the picture, the left columns change quickly from one position to the next (fast waves), while the columns on the right change very slowly (slow waves). Together they give every position a distinct pattern.
#
# </details>

# %% [markdown]
# ## Step 2: causal multi-head attention
#
# Multi-head attention splits the model width `d_model` into `h` heads of size `d_head = d_model / h`, runs scaled dot-product attention in each head, then joins the heads and mixes them with a final linear layer. For next-token prediction we add a **causal mask**: position `t` may only look at positions `≤ t`. We set the forbidden scores to −∞ *before* the softmax so their weights become exactly 0.

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # one layer makes q, k and v together
        self.out = nn.Linear(d_model, d_model)       # mixes the heads

    def forward(self, x, causal=True):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, D) -> (B, heads, T, d_head)
        q, k, v = [t.reshape(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)   # (B, heads, T, T)
        if causal:
            keep = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~keep, float("-inf"))   # mask BEFORE softmax
        w = scores.softmax(dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, D)             # join the heads
        return self.out(y), w


mha = MultiHeadAttention(d_model=32, n_heads=4)
y, w = mha(torch.randn(2, 5, 32))
n_params = sum(p.numel() for p in mha.parameters())
print(y.shape, w.shape, w.sum(-1)[0, 0].detach().numpy(), w[0, 0].triu(1).abs().sum().item())
print("parameters:", n_params)
assert y.shape == (2, 5, 32) and w.shape == (2, 4, 5, 5)
assert torch.allclose(w.sum(-1), torch.ones(2, 4, 5))    # each row of weights sums to 1
assert w.triu(1).abs().sum().item() == 0                 # nothing above the diagonal: no peeking ahead
assert n_params == 4224

# %% [markdown]
# ### ✏️ Your turn: count the parameters
#
# The layers are `qkv` (d_model → 3·d_model) and `out` (d_model → d_model), both with biases. By formula that is `4 · d_model² + 4 · d_model`. Compute the count for `d_model = 512` and store it in `answer`.

# %%
answer = None  # YOUR CODE HERE, for example: answer = 4 * 512**2 + 4 * 512

# %%
# Check (we compare with a real layer, so the check cannot be wrong)
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    real = sum(p.numel() for p in MultiHeadAttention(512, 8).parameters())
    check("parameter count for d_model=512", answer == real, f"Use 4 * d_model**2 + 4 * d_model (the real count is a whole number).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = 4 * 512**2 + 4 * 512   # 1,050,624
# ```
#
# `qkv` has 512 × 1536 weights plus 1536 biases and `out` has 512 × 512 weights plus 512 biases; adding them gives 4·512² + 4·512. The number of heads does not change the count.
#
# </details>

# %% [markdown]
# ## Step 3: a Transformer block and a tiny language model
#
# A block is: attention, add the input back (residual), LayerNorm, then a feed-forward network, add the input back, LayerNorm. This is the post-norm layout of the original Transformer, as in the lesson section *The Transformer block*. The language model embeds characters, adds the positional encoding, applies the block and predicts the next character at every position.

# %%
class Block(nn.Module):
    def __init__(self, d, h, causal=True):
        super().__init__()
        self.causal = causal
        self.attn = MultiHeadAttention(d, h)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, causal=self.causal)[0])   # residual 1 around attention
        return self.ln2(x + self.ff(x))                          # residual 2 around the feed-forward net


class TinyLM(nn.Module):
    def __init__(self, vocab, d=64, h=4, n_ctx=16, causal=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        pe = torch.tensor(positional_encoding(n_ctx, d), dtype=torch.float32)
        self.register_buffer("pe", pe)                           # fixed, not trained
        self.block = Block(d, h, causal=causal)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx) + self.pe[: idx.size(1)]
        return self.head(self.block(x))


demo = TinyLM(vocab=11)
print(demo(torch.randint(0, 11, (2, 16))).shape)   # (batch, positions, vocab)

# %% [markdown]
# ### ✏️ Your turn: where are the residual connections?
#
# In `Block.forward`, there are two residual connections. Which two lines of code contain them? Store the words in `answer` as a short string, for example by naming the variable that is added back each time.

# %%
answer = None  # YOUR ANSWER HERE, as a string

# %%
# Check (loose: looks for the idea)
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    t = str(answer).lower()
    check("mentions adding x back (x + ...)", "x" in t and ("+" in t or "add" in t or "residual" in t),
          "The input x is added to the output of attention, and again to the output of the feed-forward net.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = "x + self.attn(...)  and  x + self.ff(x)"
# ```
#
# The residual connections are the two `x + ...` sums inside the LayerNorm calls: one around attention, one around the feed-forward network. They let the input and the gradient flow straight through the block.
#
# </details>

# %% [markdown]
# ## Step 4: train on a repeated string
#
# We train on the text "the cat sat on the mat. " repeated 40 times. The string has 11 different characters. The target for each position is the *next* character, so `y` is `x` shifted by one. Because the string repeats exactly, a good model can reach a very low loss.

# %%
text = "the cat sat on the mat. " * 40
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
T = 16
print("vocab size:", len(chars), "  text length:", len(text))


def get_batch(bs=32):
    ix = torch.randint(0, len(data) - T - 1, (bs,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])    # shifted by one
    return x.to(device), y.to(device)


def train(causal, steps, seed=0):
    torch.manual_seed(seed)
    model = TinyLM(len(chars), n_ctx=T, causal=causal).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(steps + 1):
        x, y = get_batch()
        loss = nn.functional.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step:4d}  loss {loss.item():.3f}")
    return model, losses


STEPS = 150 if QUICK else 400
print("training WITH the causal mask")
model, losses_masked = train(causal=True, steps=STEPS)

assert len(chars) == 11
assert losses_masked[-1] < losses_masked[0], "the loss should go down"
print("✓ the loss went down:", round(losses_masked[0], 3), "->", round(losses_masked[-1], 3))

# %% [markdown]
# ### ✏️ Your turn: what loss should a random guesser have?
#
# A model that guesses uniformly among 11 characters has loss `ln 11`. Compute it and store it (3 decimals) in `answer`. Compare it with the first loss printed above. Also think: why is `y` the input shifted by one?

# %%
answer = None  # YOUR CODE HERE, for example: answer = round(math.log(11), 3)

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("ln 11", abs(answer - math.log(11)) < 0.005, "ln means the natural logarithm (math.log).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = round(math.log(11), 3)   # 2.398
# ```
#
# The first loss is usually somewhat close to 2.4 (it can be higher or lower because the weights are random). `y` is `x` shifted by one because every position learns to predict the character that comes *after* it, and all positions are trained at the same time.
#
# </details>

# %% [markdown]
# ## Step 5: generate, then break it
#
# Generation feeds the text so far into the model and appends the most likely next character, again and again. We cut the context to the last `T` characters because the positional table only has `T` rows.

# %%
def generate(model, prompt, n=30):
    model.eval()
    ids = torch.tensor([[stoi[c] for c in prompt]], device=device)
    with torch.no_grad():
        for _ in range(n):
            nxt = model(ids[:, -T:])[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    return "".join(chars[i] for i in ids[0].tolist())


print(repr(generate(model, "the cat s")))

# %% [markdown]
# Now the experiment. We train the **same model with `causal=False`**: every position may look at every other position, including the one it is supposed to predict. We plot both loss curves.

# %%
print("training WITHOUT the causal mask")
model_nomask, losses_nomask = train(causal=False, steps=STEPS)

plt.figure(figsize=(6, 3))
plt.plot(losses_masked, label="with causal mask")
plt.plot(losses_nomask, label="without mask")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.show()

print("final loss with mask   :", round(float(np.mean(losses_masked[-20:])), 3))
print("final loss without mask:", round(float(np.mean(losses_nomask[-20:])), 3))
print("generated with mask   :", repr(generate(model, "the cat s")))
print("generated without mask:", repr(generate(model_nomask, "the cat s")))

# %% [markdown]
# ### ✏️ Your turn: explain what the unmasked model learned
#
# Compare the two loss curves and the two generated texts. In your own words (2 to 3 sentences), what did the unmasked model learn to do instead of predicting? Why does its generated text fail even though its training loss looks so good? Write your answer in `answer`.

# %%
answer = ""  # YOUR ANSWER HERE, in a few sentences

# %%
# Check (loose: looks for the key idea)
if not answer.strip():
    print("Fill in your answer above, then run this cell again.")
else:
    t = answer.lower()
    check("mentions copying / seeing the answer / next token",
          any(w in t for w in ["copy", "cheat", "answer", "next token", "next character", "future", "look ahead", "peek"]),
          "The model can read the target from the next position instead of predicting it.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# Without the mask, position `t` can attend to position `t+1`, which holds exactly the character it is supposed to predict. The model learns to copy it from there, so the training loss falls quickly without any real prediction. At generation time the future characters do not exist yet, so the copying trick has nothing to copy and the output is poor. The mask is what forces the model to learn the pattern from the past only.
#
# </details>

# %% [markdown]
# ### Sanity check on the mask
#
# One last test with a quick proof: the output at position 3 of the **masked** model must not change when we change a *later* character, but the unmasked model's output does change.

# %%
x1 = data[:T].unsqueeze(0).to(device)
x2 = x1.clone()
x2[0, 10] = (x2[0, 10] + 1) % len(chars)      # change a later character
model.eval(); model_nomask.eval()
with torch.no_grad():
    same_masked = torch.allclose(model(x1)[0, 3], model(x2)[0, 3], atol=1e-5)
    same_nomask = torch.allclose(model_nomask(x1)[0, 3], model_nomask(x2)[0, 3], atol=1e-5)
assert same_masked, "the masked model must not see the future"
assert not same_nomask, "the unmasked model reads later positions"
print("✓ masked output at position 3 ignores later characters; unmasked output changes")

# %% [markdown]
# ## Wrap-up
#
# - Positional encoding gives attention the order information it lacks, and multi-head attention with a causal mask is the core of an autoregressive Transformer block.
# - Training predicts the next token at every position at once, with targets equal to the input shifted by one. A uniform guess has loss ln(vocab size).
# - Removing the causal mask lets the model read the answer, so the loss looks great but generation fails.
#
# **Reflection.** Double-click here to write short answers: (a) why does the feed-forward network see each token separately? (b) What would happen to word order if you removed the positional encoding? (c) Why is the mask needed in training but automatic in generation?
#
# **Stretch goals.** Stack three blocks; sample with temperature instead of `argmax`; change `n_heads` from 4 to 1 and compare the final loss.
#
# Files in `/content` disappear when the Colab session ends, so download anything you want to keep (File → Download). Submit your work the way your instructor asks; dates and weights are announced by your instructor.
#
# **Go further.** Tutorial 6, *Transformers and Multi-Head Attention*, by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence): [Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html). It builds `MultiheadAttention`, `EncoderBlock` and `PositionalEncoding` classes and trains a Transformer on the reverse-sequence task, where the heads learn to attend to the flipped position. It uses PyTorch Lightning and downloads data, so it takes longer to set up than this notebook.
