# %% [markdown]
# # Week 7 — Attention Mechanisms: lab notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive** so that your own work is saved in your own copy.
#
# **Runtime:** a CPU is enough (nothing to change). **Estimated time:** about 3 hours, but the code itself runs in seconds.
#
# This is Laboratory 3, Part 1: *Scaled dot-product attention*. You compute attention by hand, check it in PyTorch, then run two small experiments.
#
# **What you will do**
# - Part A: explain in words why soft attention can be trained and hard attention cannot.
# - Part B: compute scores, weights and output for one query, three keys and three values, by hand and then in PyTorch.
# - Part C: test what the √d_k scaling does as the key size grows, and compare `argmax` (hard) with `softmax` (soft) attention.
# - Reflect on what surprised you.
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you whether you are right. Open the **Show a possible answer** block only after you have tried.

# %% [markdown]
# ## Setup
#
# Colab already has everything we need (PyTorch, NumPy, Matplotlib), so nothing is installed. The cell below imports the libraries, fixes the random seed and defines a small `check` helper that prints ✓ or ✗.

# %%
import os
import numpy as np
import torch
import torch.nn.functional as F
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
# ## Part A: soft versus hard attention (concepts)
#
# In the lesson section *Soft weights and parallel computation* we saw two ways to use scores. **Hard attention** picks the single key with the highest score and copies its value. **Soft attention** turns all scores into weights that sum to 1 (with softmax) and returns a weighted average of all the values.
#
# ### ✏️ Your turn: write the explanation
#
# In your own words (3 to 5 sentences): why can soft attention be trained by gradient descent, while hard attention cannot? Think about what happens to the output when one score changes by a tiny amount.

# %%
part_a = ""  # YOUR ANSWER HERE: write your explanation between the quotes

# %%
# Check: it only looks at whether you wrote something and mentioned the key ideas.
if not part_a.strip():
    print("Fill in your answer above, then run this cell again.")
else:
    text = part_a.lower()
    check("You wrote at least a few words", len(text.split()) >= 15, "Write 3 to 5 full sentences.")
    check("You mention gradients, smoothness or derivatives",
          any(w in text for w in ["gradient", "smooth", "derivative", "differentiab"]),
          "Say what happens to the gradient or whether the output changes smoothly.")
    check("You compare soft with hard", "soft" in text and "hard" in text, "Mention both soft and hard attention.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# Soft attention is a smooth weighted average: if a score changes a little, the weights and the output change a little, so there is a gradient to follow. Hard attention picks the maximum. The output jumps when a different key becomes the maximum and does not change at all otherwise, so the gradient is zero almost everywhere and the model gets no signal to learn from.
#
# </details>

# %% [markdown]
# ## Part B: attention by hand, then in PyTorch
#
# We use the example from the lesson (Figure 3): one query `q = (1, 0)`, three keys and three values, with `d_k = 2`. The steps are:
#
# 1. scores: `q · kᵢ / √d_k`
# 2. weights: softmax of the scores (they sum to 1)
# 3. output: the weighted sum of the values
#
# ### ✏️ Your turn: do it by hand first
#
# On paper, compute the three dot products `q · kᵢ`, divide each by √2, then the softmax, then the output. Type your three scores here (rounded to 4 decimals). The next cell checks them against PyTorch.

# %%
my_scores = None  # YOUR ANSWER HERE, for example [a, b, c]

# %%
# Check: compare your hand-computed scores with the real computation.
q = torch.tensor([[1., 0.]])                        # 1 query
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])    # 3 keys
V = torch.tensor([[1., 0.], [0., 1.], [2., 2.]])    # 3 values
true_scores = (q @ K.T / (2 ** 0.5))[0]
if my_scores is None:
    print("Fill in your answer above, then run this cell again.")
else:
    ok = len(my_scores) == 3 and bool(np.allclose(my_scores, true_scores.numpy(), atol=1e-3))
    check("scores match", ok, "Dot products are 1, 0 and 1; then divide each by √2 ≈ 1.4142.")

# %% [markdown]
# Now the same computation in PyTorch, step by step. Compare each printed value with your paper work.

# %%
scores = q @ K.T / (2 ** 0.5)       # step 1: similarity of the query to each key, scaled
w = F.softmax(scores, dim=-1)       # step 2: turn scores into weights that sum to 1
out = w @ V                         # step 3: weighted average of the values
print("scores :", scores.numpy().round(4))
print("weights:", w.numpy().round(4))
print("output :", out.numpy().round(4))

# Sanity checks with the values computed for this example
assert np.allclose(scores.numpy(), [[0.7071, 0.0, 0.7071]], atol=1e-4)
assert np.allclose(w.numpy(), [[0.4011, 0.1978, 0.4011]], atol=1e-4)
assert np.allclose(out.numpy(), [[1.2033, 1.0]], atol=1e-4)
assert abs(w.sum().item() - 1.0) < 1e-6
print("✓ matches scores [0.7071, 0, 0.7071], weights [0.4011, 0.1978, 0.4011], output [1.2033, 1.0]")

# %% [markdown]
# Keys 1 and 3 have the same score, so they get the same weight (0.4011 each). Key 2 is less similar to the query, so it gets less (0.1978).
#
# Now we wrap the three steps in a reusable function. It also works for batches and for many queries at once, which we need in Part C.

# %%
def scaled_dot_product_attention(query, key, value, scale=True):
    """Return (output, weights). Set scale=False to skip the 1/sqrt(d_k) factor."""
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1))
    if scale:
        scores = scores / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights


out2, w2 = scaled_dot_product_attention(q, K, V)
print(out2.numpy().round(4), w2.numpy().round(4))
assert torch.allclose(out2, out)

# %% [markdown]
# ### ✏️ Your turn: remove the scaling
#
# Call the function with `scale=False` on the same `q`, `K`, `V`. Store the weights (a list of 3 numbers) in `answer`. Do the weights move closer together or further apart?

# %%
answer = None  # YOUR CODE HERE, for example: answer = scaled_dot_product_attention(q, K, V, scale=False)[1][0].tolist()

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("unscaled weights are right", bool(np.allclose(answer, [0.4223, 0.1554, 0.4223], atol=1e-3)),
          "Use scale=False and take the weights (second return value).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# _, w_unscaled = scaled_dot_product_attention(q, K, V, scale=False)
# answer = w_unscaled[0].tolist()   # [0.4223, 0.1554, 0.4223]
# ```
#
# Without scaling the scores are 1, 0, 1 instead of 0.7071, 0, 0.7071, so the weights move further apart: the big scores get more of the weight (0.4223 instead of 0.4011).
#
# </details>

# %% [markdown]
# ## Part C: experiments
#
# ### C1. Why divide by √d_k?
#
# A dot product of two random vectors adds up `d_k` terms, so its typical size grows with `d_k`. Large scores make softmax very "peaky": one key takes almost all the weight and the gradients become tiny. Dividing by √d_k keeps the scores at a similar size.
#
# We test it. For each `d_k` we draw 2000 random queries and 3 random keys per query from a standard normal distribution, then measure the **mean largest softmax weight**. With 3 keys, a perfectly even split would give 0.333 and a one-hot split 1.0.

# %%
torch.manual_seed(SEED)
n_trials = 500 if QUICK else 2000
dims = [2, 8, 32, 64, 256, 1024]
unscaled, scaled = [], []
for d in dims:
    Q = torch.randn(n_trials, 1, d)     # n_trials queries
    Kr = torch.randn(n_trials, 3, d)    # 3 keys each
    _, wu = scaled_dot_product_attention(Q, Kr, Kr, scale=False)
    _, ws = scaled_dot_product_attention(Q, Kr, Kr, scale=True)
    unscaled.append(wu.max(dim=-1).values.mean().item())
    scaled.append(ws.max(dim=-1).values.mean().item())
    print(f"d_k={d:5d}   without scaling: {unscaled[-1]:.3f}   with scaling: {scaled[-1]:.3f}")

plt.figure(figsize=(5, 3))
plt.plot(dims, unscaled, "o-", label="without √d_k")
plt.plot(dims, scaled, "s-", label="with √d_k")
plt.xscale("log", base=2)
plt.xlabel("d_k")
plt.ylabel("mean largest weight")
plt.legend()
plt.tight_layout()
plt.show()

# Sanity checks (these hold with large margins for standard-normal data and 3 keys)
assert unscaled[-1] > 0.9, "unscaled attention should become very peaky for large d_k"
assert all(0.45 < s < 0.7 for s in scaled), "scaled attention should stay moderate"

# %% [markdown]
# Without scaling, the largest weight climbs from about 0.6 towards 1 as `d_k` grows (the softmax saturates). With scaling it stays roughly between 0.55 and 0.6 for every `d_k`. This is why the Transformer uses `softmax(QKᵀ/√d_k)V`.
#
# ### ✏️ Your turn: report the effect at d_k = 64
#
# Put the mean largest weight at `d_k = 64` without scaling in `answer_unscaled`, and with scaling in `answer_scaled` (read them from the table above, or compute them from the lists `unscaled`, `scaled`, `dims`).

# %%
answer_unscaled = None  # YOUR CODE HERE
answer_scaled = None    # YOUR CODE HERE

# %%
# Check
if answer_unscaled is None or answer_scaled is None:
    print("Fill in your answer above, then run this cell again.")
else:
    i = dims.index(64)
    check("unscaled value at d_k=64", abs(answer_unscaled - unscaled[i]) < 0.02, "Read the row d_k=   64.")
    check("scaled value at d_k=64", abs(answer_scaled - scaled[i]) < 0.02, "Read the row d_k=   64.")
    check("unscaled is clearly more peaky", answer_unscaled > answer_scaled + 0.2, "Unscaled should be much larger.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# i = dims.index(64)
# answer_unscaled = unscaled[i]   # about 0.93
# answer_scaled = scaled[i]       # about 0.60
# ```
#
# At `d_k = 64` the unscaled softmax already gives one key about 93% of the weight on average, so it behaves almost like hard attention (and its gradients are tiny). The scaled version still blends the keys.
#
# </details>

# %% [markdown]
# ### C2. Hard (`argmax`) versus soft (`softmax`) on the tied example
#
# In our example keys 1 and 3 have exactly the same score. What does hard attention do with a tie? What does soft attention do? We also nudge the score of key 3 by a tiny amount to see how each output reacts.

# %%
def hard_attention(query, key, value):
    scores = query @ key.T / (key.shape[-1] ** 0.5)
    idx = scores.argmax(dim=-1)          # pick one key only
    return value[idx], idx


h_out, h_idx = hard_attention(q, K, V)
print("hard: picks key", int(h_idx) + 1, "-> output", h_out.numpy().round(4))
print("soft: weights", w.numpy().round(4), "-> output", out.numpy().round(4))

# Nudge key 3 slightly: make it a tiny bit more similar to the query
K_nudged = K.clone()
K_nudged[2, 0] = 1.001
h2, i2 = hard_attention(q, K_nudged, V)
s2, _ = scaled_dot_product_attention(q, K_nudged, V)
print("after a 0.001 nudge -> hard picks key", int(i2) + 1, "output", h2.numpy().round(4))
print("after a 0.001 nudge -> soft output", s2.numpy().round(4))

assert int(h_idx) == 0            # argmax returns the first maximal index in PyTorch
assert int(i2) == 2               # a tiny change flips the hard choice
assert float((s2 - out).abs().max()) < 0.01   # the soft output barely moves

# %% [markdown]
# With a tie, `argmax` quietly picks key 1 (the first one), which is arbitrary. A 0.001 nudge makes it jump to key 3 and the output jumps from `[1, 0]` to `[2, 2]`. The soft output moves by a tiny amount, because it splits the weight equally between tied keys and changes smoothly. That smoothness is what gradient descent needs.
#
# ### ✏️ Your turn: what does hard attention return?
#
# Without running anything new, which value vector does `hard_attention(q, K, V)` return for the original tied example? Store the vector as a list in `answer`.

# %%
answer = None  # YOUR ANSWER HERE, for example [x, y]

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("hard attention returns value 1", bool(np.allclose(answer, [1.0, 0.0])),
          "argmax picks the first of the tied maximum scores, which is key 1, so it returns v1.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = V[0].tolist()   # [1.0, 0.0]
# ```
#
# The scores are 0.7071, 0, 0.7071. `argmax` returns the first index with the maximum, key 1, so the output is `v1 = [1, 0]`. Soft attention gives `[1.2033, 1.0]` instead, which uses all three values.
#
# </details>

# %% [markdown]
# ## Wrap-up
#
# - Attention scores each key against the query, turns the scores into weights with softmax, and returns a weighted sum of the values: `softmax(QKᵀ/√d_k)V`.
# - Dividing by √d_k stops the softmax from saturating when `d_k` is large. Soft attention is smooth, so it can be trained with gradients; hard attention is not.
# - You can verify a formula by computing a tiny case by hand and then confirming it in code.
#
# **Reflection.** Double-click here to write two sentences: (a) what surprised you, and (b) one thing attention can do that a plain RNN cannot.
#
# Files in `/content` disappear when the Colab session ends, so download anything you want to keep (File → Download). Submit your work the way your instructor asks; dates and weights are announced by your instructor.
#
# **Go further.** Tutorial 6, *Transformers and Multi-Head Attention*, by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence): [Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html). Its first sections cover attention as a weighted average and scaled dot-product attention. Read those sections now; the rest is next week's topic.
