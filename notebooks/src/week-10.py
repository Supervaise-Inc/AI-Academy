# %% [markdown]
# # Week 10 — Midterm Exam review: practice notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive**, so your changes are kept in your own copy.
#
# **Runtime:** CPU is enough (no GPU needed). **Time:** about 2 hours, at your own pace.
#
# There is no new lab this week. This notebook is **retrieval practice**: you compute the small calculations that can appear when you review Weeks 6–9, then check your answer against the true value. Your instructor announces the date, room and format of the Midterm.
#
# **What you will do**
# - Week 6: one BPE merge, a cosine similarity and a bigram probability.
# - Weeks 7–8: scaled dot-product attention, a sinusoidal positional-encoding value, a causal mask and the parameters of a Transformer block.
# - Week 9: the Chinchilla compute-optimal size for a budget.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.

# %%
import os, math, random
from collections import Counter

import numpy as np
import torch

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
QUICK = bool(os.environ.get("NB_TEST"))

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

def todo(value):
    """Return True when the student has not filled in the answer yet."""
    if value is None:
        print("Fill in your answer above, then run this cell again.")
        return True
    return False

# %% [markdown]
# ## Part 1 — Tokens and embeddings (Week 6)
#
# **Reminder.** BPE builds a vocabulary of sub-word pieces by repeatedly merging the most frequent pair of neighbouring symbols, with each pair counted once per occurrence and weighted by the word count. Cosine similarity compares the direction of two vectors: cos = (a·b) / (|a| |b|). A bigram model estimates P(next | previous) by counting: P(w2 | w1) = count(w1 w2) / count(w1).
#
# ### Exercise 1: one BPE merge
#
# The corpus has these word counts: `hug` 10, `pug` 5, `pun` 12, `bun` 4, `hugs` 5. Below, a helper counts every neighbouring pair. Use it to find the pair that BPE merges first.

# %%
corpus = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}

def pair_counts(corpus):
    counts = Counter()
    for word, n in corpus.items():
        for a, b in zip(word, word[1:]):   # every neighbouring pair of characters
            counts[a + b] += n             # weighted by how often the word occurs
    return counts

counts = pair_counts(corpus)
print(counts.most_common(5))

# %% [markdown]
# ### ✏️ Your turn: the first merge
#
# Set `answer` to the pair (as a string such as `"xy"`) that BPE merges first, and `n_answer` to how many times it occurs.

# %%
answer = None      # YOUR CODE HERE, e.g. "ab"
n_answer = None    # YOUR CODE HERE

# %%
# Check
if not todo(answer) and not todo(n_answer):
    best, best_n = counts.most_common(1)[0]
    check("the first merge is " + best, answer == best, "look for the largest count in the list above")
    check("its count is " + str(best_n), n_answer == best_n, "add the counts of every word that contains the pair")
assert counts["ug"] == 20 and counts["pu"] == 17 and counts["un"] == 16 and counts["hu"] == 15

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = "ug"
# n_answer = 20      # hug (10) + pug (5) + hugs (5)
# ```
#
# `ug` is inside three words, so its count is 10 + 5 + 5 = 20. The next best pair is `pu` with 5 + 12 = 17. After this merge `ug` becomes one symbol and the counting starts again.
# </details>

# %% [markdown]
# ### Exercise 2: cosine similarity
#
# Compute the cosine similarity of `a = (3, 4)` and `b = (4, 3)`. Round to two decimals.

# %%
a = np.array([3.0, 4.0])
b = np.array([4.0, 3.0])
answer = None   # YOUR CODE HERE: a number rounded to 2 decimals

# %%
# Check
if not todo(answer):
    true = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    check(f"cosine similarity = {true:.2f}", abs(answer - true) < 0.005, "cos = (a·b) / (|a| |b|); |a| is the square root of the sum of squares")
assert abs(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) - 0.96) < 1e-9

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 2)   # 0.96
# ```
#
# a·b = 12 + 12 = 24 and |a| = |b| = 5, so cos = 24 / 25 = 0.96: almost the same direction.
# </details>

# %% [markdown]
# ### Exercise 3: a bigram probability
#
# Here is a tiny training text. A bigram model counts how often each word follows another. Compute `P(cat | the)`, the probability that the next word is `cat` given that the current word is `the`.

# %%
text = "the cat sat . the cat ran . the dog sat ."
words = text.split()
bigrams = Counter(zip(words, words[1:]))
unigrams = Counter(words[:-1])          # how often each word appears as a "previous" word
print(bigrams[("the", "cat")], bigrams[("the", "dog")], unigrams["the"])
answer = None   # YOUR CODE HERE: P(cat | the) as a fraction or a decimal

# %%
# Check
if not todo(answer):
    true = bigrams[("the", "cat")] / unigrams["the"]
    check(f"P(cat | the) = {true:.3f}", abs(answer - true) < 0.005, "divide count('the cat') by count('the')")
assert unigrams["the"] == 3 and bigrams[("the", "cat")] == 2

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = bigrams[("the", "cat")] / unigrams["the"]   # 2 / 3 = 0.667
# ```
#
# `the` appears 3 times and is followed by `cat` twice and by `dog` once, so P(cat | the) = 2/3. One limit of a bigram model: it sees only one previous word, so it cannot use longer context.
# </details>

# %% [markdown]
# ## Part 2 — Attention and the Transformer (Weeks 7–8)
#
# **Reminder.** Scaled dot-product attention works in four moves: score (q·k), scale (divide by √d_k), softmax, mix (weighted sum of the values). Attention alone ignores order, so a positional encoding is added: PE(pos, 2i) = sin(pos / 10000^(2i/d)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d)). A causal mask hides later tokens.
#
# ### Exercise 4: attention weights
#
# The lesson example used `q = (1, 1)`. This time the query is `q = (2, 0)`, with the same three keys and `d_k = 2`. Compute the scores and the attention weights.

# %%
q = np.array([2.0, 0.0])
K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
d_k = 2

def softmax(x):
    e = np.exp(x - x.max())     # subtracting the max keeps the numbers safe and gives the same result
    return e / e.sum()

# Lesson example first, so you see the four moves: it should print 0.25 0.25 0.5
q0 = np.array([1.0, 1.0])
print("lesson weights:", softmax(K @ q0 / math.sqrt(d_k)).round(2))

scores = None    # YOUR CODE HERE: raw scores q·k for the three keys (array of 3 numbers)
weights = None   # YOUR CODE HERE: the three weights after scaling and softmax

# %%
# Check
if not todo(scores) and not todo(weights):
    true_scores = K @ q
    true_w = softmax(true_scores / math.sqrt(d_k))
    check("scores " + str(true_scores), np.allclose(scores, true_scores), "score = dot product of q with each key")
    check("weights " + str(true_w.round(3)), np.allclose(weights, true_w, atol=0.005), "divide the scores by sqrt(2) before the softmax; the weights sum to 1")
assert np.allclose(softmax(K @ q0 / math.sqrt(d_k)), [0.248, 0.248, 0.503], atol=1e-3)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# scores = K @ q                          # [2, 0, 2]
# weights = softmax(scores / math.sqrt(d_k))
# ```
#
# The scores are 2, 0 and 2. After dividing by √2 they are 1.41, 0 and 1.41. Softmax gives about 0.45, 0.11 and 0.45. Keys 1 and 3 tie because they match the query equally well.
# </details>

# %% [markdown]
# ### Exercise 5: a positional-encoding value
#
# Take `d_model = 16`. Compute PE(pos = 3, dimension 4). Dimension 4 is an even index, so it is a sine with i = 2, and the divisor is 10000^(2i/d).

# %%
d_model = 16
pos = 3
answer = None   # YOUR CODE HERE: a number rounded to 3 decimals

# %%
# Check
def pe(pos, dim, d_model):
    i = dim // 2
    angle = pos / 10000 ** (2 * i / d_model)
    return math.sin(angle) if dim % 2 == 0 else math.cos(angle)

if not todo(answer):
    true = pe(pos, 4, d_model)
    check(f"PE(3, 4) = {true:.3f}", abs(answer - true) < 0.001, "angle = 3 / 10000^(4/16) = 3 / 10; then take the sine")
assert abs(pe(2, 0, 16) - 0.9093) < 1e-3 and abs(pe(2, 1, 16) - (-0.4161)) < 1e-3   # lesson: 0.91 and -0.42

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = round(math.sin(3 / 10000 ** (4 / 16)), 3)   # 0.296
# ```
#
# 10000^(4/16) = 10000^0.25 = 10, so the angle is 0.3 radians and sin(0.3) = 0.296. Low dimensions use fast waves and high dimensions use slow waves, so each position gets a unique pattern.
# </details>

# %% [markdown]
# ### Exercise 6: a causal mask
#
# Build the causal mask for 6 tokens (1 = visible, 0 = hidden) and count the visible entries. Hint: `np.tril` keeps the lower triangle of a matrix.

# %%
n_tokens = 6
mask = None      # YOUR CODE HERE: a 6 x 6 array of 0s and 1s
visible = None   # YOUR CODE HERE: how many entries are 1

# %%
# Check
if not todo(mask) and not todo(visible):
    true_mask = np.tril(np.ones((n_tokens, n_tokens), dtype=int))
    check("mask is lower-triangular", np.array_equal(np.array(mask), true_mask), "row i can see columns 0..i, including itself")
    check(f"visible entries = {true_mask.sum()}", visible == true_mask.sum(), "1 + 2 + ... + 6")
assert np.tril(np.ones((5, 5))).sum() == 15   # the lesson's 5-token case

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# mask = np.tril(np.ones((n_tokens, n_tokens), dtype=int))
# visible = int(mask.sum())      # 21
# ```
#
# Row i sees i + 1 entries, so 1 + 2 + 3 + 4 + 5 + 6 = 21 out of 36. The diagonal is visible because a token may attend to itself.
# </details>

# %% [markdown]
# ### Exercise 7: parameters of a Transformer block
#
# A block has `d_model = 256`, 4 attention heads and a feed-forward network that goes 256 → 1024 → 256. First count the weights of the four attention projections (W_Q, W_K, W_V, W_O), ignoring biases. Then count **all** the parameters of one PyTorch block, including biases and the two layer norms.

# %%
d, d_ff = 256, 1024
attn_weights = None   # YOUR CODE HERE: weights in W_Q, W_K, W_V, W_O (no biases)
block_total = None    # YOUR CODE HERE: every parameter of one nn.TransformerEncoderLayer

# %%
# Check
block = torch.nn.TransformerEncoderLayer(d, 4, d_ff, batch_first=True)
real_total = sum(p.numel() for p in block.parameters())
if not todo(attn_weights):
    check("attention weights = 4 x d^2 = 262144", attn_weights == 4 * d * d, "four matrices of d x d")
if not todo(block_total):
    check(f"block total = {real_total}", block_total == real_total,
          "attention 4d^2 + 4d biases; feed-forward 2*d*d_ff + d_ff + d; two layer norms of 2d each")
assert real_total == 4 * d * d + 4 * d + 2 * d * d_ff + d_ff + d + 4 * d

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# attn_weights = 4 * d * d                                  # 262,144
# block_total = (4*d*d + 4*d) + (2*d*d_ff + d_ff + d) + 2*(2*d)
# ```
#
# Attention: 4 × 256² = 262,144 weights plus 4 × 256 biases. Feed-forward: 2 × 256 × 1024 = 524,288 weights plus 1024 + 256 biases. Two layer norms add 4 × 256. The total is 789,760 parameters here. The number of heads does not change the count, because each head uses a slice of the same matrices.
# </details>

# %% [markdown]
# ## Part 3 — Families, scaling and prompting (Week 9)
#
# **Reminder.** Two approximations: compute C ≈ 6·N·D, and the compute-optimal amount of data D ≈ 20·N (Hoffmann et al., 2022). Then C = 120·N², so N = √(C / 120).
#
# ### Exercise 8: Chinchilla arithmetic
#
# A budget is C = 1.2 × 10²² FLOPs. Find the compute-optimal number of parameters N and training tokens D.

# %%
C = 1.2e22
N = None   # YOUR CODE HERE: parameters
D = None   # YOUR CODE HERE: tokens

# %%
# Check
if not todo(N) and not todo(D):
    true_N = math.sqrt(C / 120)
    true_D = 20 * true_N
    check(f"N = {true_N:.3g}", math.isclose(N, true_N, rel_tol=0.01), "N = sqrt(C / 120)")
    check(f"D = {true_D:.3g}", math.isclose(D, true_D, rel_tol=0.01), "D = 20 x N")
    check("6 x N x D gives back C", math.isclose(6 * N * D, C, rel_tol=0.02), "recompute 6*N*D")
assert math.isclose(math.sqrt(4.8e22 / 120), 2e10, rel_tol=1e-9)   # the lesson example: 20 billion parameters

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# N = math.sqrt(C / 120)    # 1e10, so 10 billion parameters
# D = 20 * N                # 2e11, so 200 billion tokens
# ```
#
# C / 120 = 10²⁰, whose square root is 10¹⁰. Check: 6 × 10¹⁰ × 2 × 10¹¹ = 1.2 × 10²². Doubling the model size needs four times the compute, because N depends on the square root of C.
# </details>

# %% [markdown]
# ### Last self-check
#
# Run this cell as a final sanity check, then go back to any exercise that still says "Fill in your answer".

# %%
# Quick asserts so you notice a broken environment.
assert pair_counts(corpus).most_common(1)[0] == ("ug", 20)
assert math.isclose(math.sqrt(C / 120), 1e10, rel_tol=1e-9)
print("Environment OK. Now go back to any exercise that still says 'Fill in your answer'.")

# %% [markdown]
# ## Wrap-up
#
# - You recomputed, in code, a BPE merge, a cosine similarity and a bigram probability (Week 6).
# - You computed attention weights, a positional-encoding value, a causal mask and a Transformer block's parameter count (Weeks 7–8), and the Chinchilla arithmetic (Week 9).
# - Remaining weak spots tell you where to restart your review: redo that exercise with new numbers.
#
# **Reflect:** double-click to write two sentences: which calculation was hardest, and what will you practise again?
#
# Files in `/content` vanish when the session ends, so download anything you want to keep (File → Download). Submit this notebook the way your instructor asks.
