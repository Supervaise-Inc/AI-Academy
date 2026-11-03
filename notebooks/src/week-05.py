# %% [markdown]
# # Week 5 — Prelim Exam Review: practice notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your answers are saved in your own copy.
#
# **Runtime:** CPU is enough (the default). **Time:** about 2 hours.
#
# There is no lab this week. This is a **practice notebook** for the Prelim review of Weeks 1–4: short reminders, then small exercises where you compute the answer in code. Each one has a Check against the true value and a worked answer. This notebook says nothing about the date, room, format or weight of the exam; your instructor announces those.
#
# **What you will do**
# - Weeks 1–2: generative or discriminative, a neuron's output, MSE, cross-entropy, one gradient-descent step.
# - Week 3: a backprop gradient with the chain rule, vanishing-gradient arithmetic.
# - Week 4: convolution values, output sizes and parameter counts.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open **Show a possible answer** only after you tried. Try each one on paper first, then use code as a calculator and to check yourself.

# %% [markdown]
# ## Setup

# %%
import os, random, math
import numpy as np
import torch, torch.nn.functional as F

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
QUICK = bool(os.environ.get("NB_TEST"))   # used to shorten long loops when testing

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

def check_close(name, answer, true, hint="", tol=1e-3):
    # Used by every numeric exercise: handles an unfinished answer without crashing
    if answer is None:
        print("Fill in your answer above, then run this cell again.")
    else:
        check(name, abs(float(answer) - float(true)) <= tol, hint)

# A sanity check that the helper itself works (a fixed case)
assert abs(0.25 ** 2 - 0.0625) < 1e-12

# %% [markdown]
# # Part 1: Weeks 1–2, generative AI, learning and neurons
#
# A **generative model** creates new content; a **discriminative model** labels or scores data that already exists. Learning means lowering a **loss** with **gradient descent**: new weight = old weight − learning rate × gradient. A neuron computes z = w·x + b and then applies an activation.

# %% [markdown]
# ### ✏️ Exercise 1.1: generative or discriminative?
#
# Fill in the dictionary: for each system write `"generative"` or `"discriminative"`.

# %%
systems = {
    "Writes a new paragraph from a prompt": None,        # YOUR CODE HERE
    "Labels an email as spam or not spam": None,
    "Predicts a house price from its size": None,
    "Draws a new image from a text description": None,
    "Decides whether a face matches the phone owner": None,
}

# %%
truth = {
    "Writes a new paragraph from a prompt": "generative",
    "Labels an email as spam or not spam": "discriminative",
    "Predicts a house price from its size": "discriminative",
    "Draws a new image from a text description": "generative",
    "Decides whether a face matches the phone owner": "discriminative",
}
if any(v is None for v in systems.values()):
    print("Fill in your answer above, then run this cell again.")
else:
    for k, v in truth.items():
        check(k, systems[k] == v, "Does it create new content, or return a label or number for existing data?")
print("Total correct:", sum(systems[k] == v for k, v in truth.items()), "of", len(truth))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# systems = {k: ("generative" if k.startswith(("Writes", "Draws")) else "discriminative") for k in systems}
# ```
#
# Writing a paragraph and drawing an image create new content. Spam labelling, price prediction and face matching each return a label or a number for data that already exists.
# </details>

# %% [markdown]
# ### ✏️ Exercise 1.2: a neuron's output
#
# A neuron has weights w = (1, −2), bias b = 1 and input x = (2, 1). Compute z = w·x + b (before any activation), then the sigmoid of z, σ(z) = 1 / (1 + e^(−z)).

# %%
z = None         # YOUR CODE HERE
sigma_z = None   # YOUR CODE HERE

# %%
w, x, b = np.array([1, -2]), np.array([2, 1]), 1
true_z = float(w @ x + b)
true_sigma = 1 / (1 + math.exp(-true_z))
check_close("z", z, true_z, "Multiply matching numbers, add them up, then add the bias. Watch the minus sign.")
check_close("sigmoid(z)", sigma_z, true_sigma, "Use 1 / (1 + exp(-z)).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# z = 1*2 + (-2)*1 + 1                  # = 1
# sigma_z = 1 / (1 + math.exp(-z))      # about 0.7311
# ```
#
# z = 2 − 2 + 1 = 1, and σ(1) ≈ 0.7311.
# </details>

# %% [markdown]
# ### ✏️ Exercise 1.3: mean squared error
#
# A model predicts `[2, 4, 7, 5]` and the true values are `[3, 3, 4, 2]`. Compute the MSE: the mean of the squared errors.

# %%
pred = [2, 4, 7, 5]
true_vals = [3, 3, 4, 2]
mse = None   # YOUR CODE HERE

# %%
true_mse = float(np.mean((np.array(pred) - np.array(true_vals)) ** 2))
check_close("MSE", mse, true_mse, "Square each error, add them up, then divide by how many values there are.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# mse = sum((p - t) ** 2 for p, t in zip(pred, true_vals)) / len(pred)   # 5.0
# ```
#
# The errors are −1, 1, 3, 3; their squares are 1, 1, 9, 9, which sum to 20; divide by 4 to get 5.
# </details>

# %% [markdown]
# ### ✏️ Exercise 1.4: cross-entropy for one example
#
# For a classifier, the cross-entropy loss on one example is −ln(p), where p is the probability the model gave to the **correct** class. The model outputs the probabilities `[0.7, 0.2, 0.1]` for classes 0, 1, 2, and the correct class is 0. Compute the loss. Then compute it again if the correct class were 2.

# %%
probs = [0.7, 0.2, 0.1]
ce_class0 = None   # YOUR CODE HERE
ce_class2 = None   # YOUR CODE HERE

# %%
check_close("cross-entropy, correct class 0", ce_class0, -math.log(0.7), "Take the natural log (math.log) of 0.7 and flip the sign.")
check_close("cross-entropy, correct class 2", ce_class2, -math.log(0.1), "Use the probability of class 2, which is 0.1.")
# Cross-check against PyTorch's own function (a fixed case, so a broken environment is noticed)
assert abs(F.cross_entropy(torch.log(torch.tensor([probs])), torch.tensor([0])).item() - (-math.log(0.7))) < 1e-5

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ce_class0 = -math.log(0.7)   # about 0.3567
# ce_class2 = -math.log(0.1)   # about 2.3026
# ```
#
# The loss is small when the model gives the correct class a high probability and large when it gives it a low one.
# </details>

# %% [markdown]
# ### ✏️ Exercise 1.5: one gradient-descent step
#
# For L(w) = (w − 3)², the gradient is 2(w − 3). Start at w = 0 with learning rate 0.1. Compute the gradient and the new w after one step. Then run 3 steps in a loop and give the final w.

# %%
grad0 = None     # YOUR CODE HERE: gradient at w = 0
w_after1 = None  # YOUR CODE HERE: w after one step
w_after3 = None  # YOUR CODE HERE: w after three steps

# %%
w_ = 0.0; hist = []
for _ in range(3):
    w_ = w_ - 0.1 * 2 * (w_ - 3); hist.append(w_)
check_close("gradient at 0", grad0, -6, "2 * (0 - 3).")
check_close("w after 1 step", w_after1, hist[0], "new w = old w - learning rate * gradient.")
check_close("w after 3 steps", w_after3, hist[2], "Repeat the same update three times, using the new w each time.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# w = 0.0
# for step in range(3):
#     w = w - 0.1 * 2 * (w - 3)
# # after step 1: 0.6, after step 3: 1.464
# grad0 = -6; w_after1 = 0.6; w_after3 = w
# ```
#
# w goes 0 → 0.6 → 1.08 → 1.464. The gradient is negative, so w moves up, towards the minimum at 3.
# </details>

# %% [markdown]
# # Part 2: Week 3, backpropagation and vanishing gradients
#
# A backward pass multiplies local derivatives (the chain rule). The sigmoid's derivative is at most 0.25, so each sigmoid layer passes back at most a quarter of the gradient.

# %% [markdown]
# ### ✏️ Exercise 2.1: a backprop gradient by hand
#
# One neuron: z = w·x + b, y = σ(z), loss L = (y − t)². Let x = 1, w = 0.5, b = 0 and target t = 1. Find dL/dw = dL/dy × dy/dz × dz/dw. Compute the three factors and their product. (Recall dL/dy = 2(y − t), dy/dz = y(1 − y), dz/dw = x.)

# %%
dL_dy = None   # YOUR CODE HERE
dy_dz = None   # YOUR CODE HERE
dz_dw = None   # YOUR CODE HERE
dL_dw = None   # YOUR CODE HERE: the product of the three

# %%
x_, w0, b0, t_ = 1.0, 0.5, 0.0, 1.0
y_ = 1 / (1 + math.exp(-(w0 * x_ + b0)))
true_factors = (2 * (y_ - t_), y_ * (1 - y_), x_)
true_grad = true_factors[0] * true_factors[1] * true_factors[2]
check_close("dL/dy", dL_dy, true_factors[0], "First find y = sigmoid(0.5), then 2 * (y - t).", tol=2e-3)
check_close("dy/dz", dy_dz, true_factors[1], "y * (1 - y).", tol=2e-3)
check_close("dz/dw", dz_dw, true_factors[2], "z = w*x + b, so the slope with respect to w is x.")
check_close("dL/dw", dL_dw, true_grad, "Multiply the three numbers.", tol=2e-3)

# %% [markdown]
# Let PyTorch's autograd confirm the true value. This is a check on the number above, not a new idea.

# %%
wt = torch.tensor(0.5, requires_grad=True)
loss = (torch.sigmoid(wt * 1.0 + 0.0) - 1.0) ** 2
loss.backward()
print("autograd gradient:", round(wt.grad.item(), 4), "  by hand:", round(true_grad, 4))
assert abs(wt.grad.item() - true_grad) < 1e-5

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# y = 1 / (1 + math.exp(-0.5))        # 0.6225
# dL_dy = 2 * (y - 1)                 # -0.7551
# dy_dz = y * (1 - y)                 # 0.2350
# dz_dw = 1.0
# dL_dw = dL_dy * dy_dz * dz_dw       # -0.1774
# ```
#
# The gradient is negative, so gradient descent raises w. These are the same numbers as the lesson's forward-and-backward figure.
# </details>

# %% [markdown]
# ### ✏️ Exercise 2.2: a simpler chain rule
#
# With y = w·x, x = 2, w = 3, target t = 4 and loss L = (y − t)², compute dL/dw.

# %%
dL_dw_simple = None   # YOUR CODE HERE

# %%
y2 = 3 * 2
true_simple = 2 * (y2 - 4) * 2
check_close("dL/dw", dL_dw_simple, true_simple, "dL/dy = 2(y - t), dy/dw = x. Multiply them.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# y = 3 * 2                       # 6
# dL_dw_simple = 2 * (y - 4) * 2  # dL/dy = 4, dy/dw = 2, product 8
# ```
# </details>

# %% [markdown]
# ### ✏️ Exercise 2.3: vanishing-gradient arithmetic
#
# Each sigmoid layer multiplies the gradient by at most 0.25. What fraction of the gradient is left after 5 layers? After 10 layers? Then find the smallest number of layers after which less than 1 in a million (1e-6) is left.

# %%
left5 = None       # YOUR CODE HERE
left10 = None      # YOUR CODE HERE
layers_1e6 = None  # YOUR CODE HERE: smallest n with 0.25**n < 1e-6

# %%
n_true = next(n for n in range(1, 100) if 0.25 ** n < 1e-6)
check_close("after 5 layers", left5, 0.25 ** 5, "Multiply 0.25 by itself 5 times (0.25 ** 5).", tol=1e-9)
check_close("after 10 layers", left10, 0.25 ** 10, "0.25 ** 10.", tol=1e-12)
check_close("smallest n below 1e-6", layers_1e6, n_true, "Loop n = 1, 2, 3 ... until 0.25 ** n is below 1e-6.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# left5 = 0.25 ** 5                                  # 0.0009765625, about 0.001
# left10 = 0.25 ** 10                                # about 9.5e-07, roughly one millionth
# layers_1e6 = next(n for n in range(1, 100) if 0.25 ** n < 1e-6)   # 10
# ```
#
# After 10 layers the gradient is about 0.00000095 of its original size, so the first layers barely learn. ReLU, careful initialisation, residual connections and normalisation all help.
# </details>

# %% [markdown]
# # Part 3: Week 4, convolutional networks
#
# A convolution slides a kernel over the image: multiply matching numbers and add. Output size is ⌊(n + 2p − k) / s⌋ + 1. A conv layer has k × k × C_in × C_out + C_out parameters.

# %% [markdown]
# ### ✏️ Exercise 3.1: one convolution value
#
# A 3 × 3 kernel whose every row is `(1, 0, −1)` sits over a patch whose every row is `(5, 5, 1)`. What number does the convolution write? (The kernel is not flipped.)

# %%
conv_value = None   # YOUR CODE HERE

# %%
patch = torch.tensor([[5.0, 5.0, 1.0]] * 3).reshape(1, 1, 3, 3)
ker = torch.tensor([[1.0, 0.0, -1.0]] * 3).reshape(1, 1, 3, 3)
true_conv = F.conv2d(patch, ker).item()
check_close("convolution value", conv_value, true_conv, "Left column sums to 15, the middle column is multiplied by 0, the right column is -(1+1+1).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# conv_value = (5+5+5)*1 + (5+5+5)*0 + (1+1+1)*(-1)   # 15 - 3 = 12
# ```
#
# A large value means a strong vertical-edge response at this patch.
# </details>

# %% [markdown]
# ### ✏️ Exercise 3.2: the 4 × 4 convolution from the lesson
#
# The input is the 4 × 4 grid below, the kernel is `[[1, 0], [0, -1]]`, stride 1, padding 0. Compute the whole output grid with a small loop (or by hand first) and store it in `out_grid` as a NumPy array.

# %%
img = np.array([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], dtype=float)
kern = np.array([[1, 0], [0, -1]], dtype=float)
out_grid = None   # YOUR CODE HERE: a 3 x 3 array

# %%
true_grid = F.conv2d(torch.tensor(img).reshape(1, 1, 4, 4).float(), torch.tensor(kern).reshape(1, 1, 2, 2).float()).squeeze().numpy()
if out_grid is None:
    print("Fill in your answer above, then run this cell again.")
else:
    og = np.array(out_grid, dtype=float)
    check("shape is 3 x 3", og.shape == (3, 3), "Output size is (4 - 2) / 1 + 1 = 3.")
    check("all nine values", og.shape == (3, 3) and np.allclose(og, true_grid), "Each cell is 1*top-left + 0*top-right + 0*bottom-left + (-1)*bottom-right of its patch.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# out_grid = np.array([[img[i:i+2, j:j+2].ravel() @ kern.ravel() for j in range(3)] for i in range(3)])
# ```
#
# Each output is (top-left of the patch) − (bottom-right of the patch). For the top-left patch that is 1 − 1 = 0.
# </details>

# %% [markdown]
# ### ✏️ Exercise 3.3: output sizes
#
# Write a function `out_size(n, k, p, s)` using the formula, then use it for a 28 × 28 input with kernel 3, padding 1, stride 2, and for a 32 × 32 input with kernel 5, padding 0, stride 1.

# %%
def out_size(n, k, p, s):
    return None   # YOUR CODE HERE

size_a = out_size(28, 3, 1, 2)
size_b = out_size(32, 5, 0, 1)
print(size_a, size_b)

# %%
if size_a is None or size_b is None:
    print("Fill in out_size above, then run this cell again.")
else:
    real_a = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)(torch.zeros(1, 1, 28, 28)).shape[-1]
    real_b = torch.nn.Conv2d(1, 1, 5, stride=1, padding=0)(torch.zeros(1, 1, 32, 32)).shape[-1]
    check("28, k3, p1, s2", size_a == real_a, "floor((28 + 2 - 3) / 2) + 1. Do not forget the + 1, or the floor.")
    check("32, k5, p0, s1", size_b == real_b, "floor((32 - 5) / 1) + 1.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def out_size(n, k, p, s):
#     return (n + 2 * p - k) // s + 1
# ```
#
# (28 + 2 − 3) // 2 + 1 = 13 + 1 = 14, and (32 − 5) // 1 + 1 = 28. We compared with real PyTorch layers in the Check.
# </details>

# %% [markdown]
# ### ✏️ Exercise 3.4: parameter counts
#
# (a) A conv layer has 16 filters of size 5 × 5 on a 3-channel input, one bias per filter. How many parameters? (b) A fully connected layer maps a 28 × 28 image to 100 units. How many parameters? (c) How many times more is (b) than a conv layer with eight 3 × 3 filters on 1 channel? Give a number rounded to one decimal.

# %%
params_conv = None   # YOUR CODE HERE (a)
params_fc = None     # YOUR CODE HERE (b)
ratio = None         # YOUR CODE HERE (c)

# %%
true_conv_p = sum(p.numel() for p in torch.nn.Conv2d(3, 16, 5).parameters())
true_fc_p = sum(p.numel() for p in torch.nn.Linear(784, 100).parameters())
true_small = sum(p.numel() for p in torch.nn.Conv2d(1, 8, 3).parameters())
check_close("(a) conv parameters", params_conv, true_conv_p, "5*5*3 weights per filter, times 16 filters, plus 16 biases.")
check_close("(b) FC parameters", params_fc, true_fc_p, "784 * 100 weights plus 100 biases.")
check_close("(c) ratio", ratio, true_fc_p / true_small, "Divide the FC count by 80, then round.", tol=0.06)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# params_conv = 16 * (5 * 5 * 3) + 16   # 1216
# params_fc = 784 * 100 + 100           # 78500
# ratio = round(78500 / (3*3*1*8 + 8), 1)   # 78500 / 80 = 981.2
# ```
#
# The convolution's parameter count does not depend on the image size; the fully connected count grows with it.
# </details>

# %% [markdown]
# ### ✏️ Exercise 3.5: max pooling
#
# Apply 2 × 2 max pooling with stride 2 to the 4 × 4 map below and store the 2 × 2 result in `pooled`.

# %%
fmap = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 3, 2], [2, 6, 1, 1]], dtype=float)
pooled = None   # YOUR CODE HERE: a 2 x 2 array

# %%
true_pool = F.max_pool2d(torch.tensor(fmap).reshape(1, 1, 4, 4).float(), 2).squeeze().numpy()
if pooled is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("pooled map", np.array(pooled).shape == (2, 2) and np.allclose(pooled, true_pool), "Split into four 2 x 2 blocks and keep the biggest number in each.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pooled = np.array([[fmap[:2, :2].max(), fmap[:2, 2:].max()], [fmap[2:, :2].max(), fmap[2:, 2:].max()]])
# # [[4, 5], [6, 3]]
# ```
#
# Pooling has no weights; it halves the width and height.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - You can compute a neuron output, MSE, cross-entropy and a gradient-descent step, and tell generative from discriminative.
# - You can multiply local derivatives along a path (the chain rule) and explain with numbers why deep sigmoid networks lose the gradient.
# - You can compute convolution values, output sizes and parameter counts, and say why weight sharing saves parameters.
#
# **Reflection:** double-click to write two sentences: which exercise was slowest for you, and what is your plan for it before the exam?
#
# Files in `/content` vanish when your Colab session ends, so download anything you want to keep (File → Download). Submit it the way your instructor asks. Your instructor announces the exam details; this notebook does not.
