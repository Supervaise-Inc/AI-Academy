# %% [markdown]
# # Week 3 — Backpropagation and Training Deep Networks: lab notebook
#
# Built for **Google Colab**. First click **File → Save a copy in Drive**, so your work is saved in your own copy.
#
# **Runtime:** a CPU is enough. You do not need a GPU. **Time:** about 3 hours.
#
# **What you will do**
# - Check a hand-computed backward pass with PyTorch autograd.
# - Check it a second time with finite differences, which need no framework.
# - Build a 10-layer network and measure the gradient size in every layer.
# - Compare sigmoid, ReLU and Kaiming initialisation, and see gradients vanish and explode.
# - See how depth changes the vanishing gradient, then clip an exploding gradient.
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.

# %% [markdown]
# ## Setup
#
# This cell imports what we need, fixes the random seed so your numbers repeat, and defines a small `check` helper that prints ✓ or ✗. Nothing is installed and nothing is downloaded.

# %%
import os, math
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "(this notebook runs on the CPU either way)")
QUICK = bool(os.environ.get("NB_TEST"))   # shortens long loops when the notebook is tested

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1. Backprop on paper
#
# From the lesson section on the chain rule: take one sigmoid neuron with a squared-error loss. The forward pass is z = w·x + b, y = σ(z), L = (y − t)². The backward pass multiplies local derivatives: dL/dy = 2(y − t), dy/dz = y(1 − y), so dL/dz = dL/dy · dy/dz, and then dL/dw = dL/dz · x and dL/db = dL/dz.
#
# With x = 1.5, w = 0.8, b = −0.4, t = 1, do this on paper to four decimals. Then do the same for **x = −1 and t = 0** (w and b unchanged). You will check both next.

# %% [markdown]
# ## Step 2. Autograd check
#
# PyTorch can compute the same gradients for us. `L.backward()` walks the computation graph backwards and fills `w.grad` and `b.grad`. We wrap it in a function that returns z, y, L, dL/dw and dL/db.

# %%
def grads(x, t, w0=0.8, b0=-0.4):
    w = torch.tensor(w0, requires_grad=True); b = torch.tensor(b0, requires_grad=True)
    z = w * x + b; y = torch.sigmoid(z); L = (y - t) ** 2
    L.backward()
    return [round(v, 4) for v in (z.item(), y.item(), L.item(), w.grad.item(), b.grad.item())]

first = grads(1.5, 1.0)
print("x=1.5, t=1 :", first)    # z, y, L, dL/dw, dL/db
print("x=-1,  t=0 :", grads(-1.0, 0.0))
assert first == [0.8, 0.69, 0.0961, -0.199, -0.1326], "autograd values changed: check your PyTorch install"

# %% [markdown]
# ### ✏️ Your turn: x = −1, t = 0
#
# Type your hand-computed dL/dw for x = −1 and t = 0 (four decimals). Then, for the first case (x = 1.5, t = 1), apply **one gradient-descent step** with learning rate 0.5 (w ← w − 0.5·dL/dw, same for b) and compute the **new loss** by hand. Put both in the variables below.

# %%
answer_dw = None    # YOUR ANSWER: dL/dw for x = -1, t = 0
answer_loss = None  # YOUR ANSWER: the loss for x = 1.5, t = 1 after one step with lr 0.5

# %%
# Check
true_dw = grads(-1.0, 0.0)[3]
g = grads(1.5, 1.0)
w_new, b_new = 0.8 - 0.5 * g[3], -0.4 - 0.5 * g[4]
true_loss = (1 / (1 + math.exp(-(w_new * 1.5 + b_new))) - 1.0) ** 2
if answer_dw is None or answer_loss is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("dL/dw for x = -1, t = 0", abs(answer_dw - true_dw) < 1e-3, "dL/dz = 2(y - t) * y(1 - y), then multiply by x")
    check("loss after one step", abs(answer_loss - true_loss) < 1e-3, "update w and b first, then run the forward pass again")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_dw = -0.0824          # z = -1.2, y = 0.2315, dL/dz = 2*0.2315*0.2315*0.7685 = 0.0824, times x = -1
# w_new, b_new = 0.8 + 0.5 * 0.199, -0.4 + 0.5 * 0.1326     # 0.8995 and -0.3337
# answer_loss = (1 / (1 + math.exp(-(w_new * 1.5 + b_new))) - 1) ** 2     # 0.0707
# ```
# One step downhill lowers the loss from 0.0961 to about 0.0707, which is what gradient descent should do.
# </details>

# %% [markdown]
# ## Step 3. Finite differences
#
# A second check that needs no framework: nudge one parameter up and down by a tiny ε and see how the loss changes. The slope is (L(w + ε) − L(w − ε)) / (2ε). This is the "numerical gradient" idea from the lesson.

# %%
def loss(w, b, x=1.5, t=1.0):
    y = 1 / (1 + math.exp(-(w * x + b)))
    return (y - t) ** 2

eps = 1e-6
dw = (loss(0.8 + eps, -0.4) - loss(0.8 - eps, -0.4)) / (2 * eps)
db = (loss(0.8, -0.4 + eps) - loss(0.8, -0.4 - eps)) / (2 * eps)
print(dw, db)
assert abs(dw - first[3]) < 1e-3 and abs(db - first[4]) < 1e-3, "finite differences should match autograd"

# %% [markdown]
# ### ✏️ Your turn: why is this too slow?
#
# For one gradient of a network with P parameters, this method needs two forward passes per parameter (one for +ε and one for −ε). **How many forward passes** does it need for a network with one million parameters? Put the number in `answer`. Backprop needs about one forward and one backward pass instead.

# %%
answer = None   # YOUR ANSWER: forward passes for one gradient with 1_000_000 parameters

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("forward passes", answer == 2_000_000, "two forward passes (+eps and -eps) for each of the 1,000,000 parameters")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = 2 * 1_000_000
# ```
# Two million forward passes for a single gradient, against about one forward and one backward pass for backprop. That is why we use backprop to train, and finite differences only to check a small example.
# </details>

# %% [markdown]
# ## Step 4. Build a 10-layer network and measure gradients
#
# From the lesson section on vanishing and exploding gradients. `make_mlp` builds a stack of `depth` linear layers with an activation after each one, plus a final linear layer. `grad_norms` runs one forward and backward pass on random data and returns the size (the norm) of the weight gradient in every linear layer. Layer 1 is the closest to the input.

# %%
def make_mlp(act, init="default", depth=10, width=32):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), act()]
    net = nn.Sequential(*layers, nn.Linear(width, 1))
    for m in net:
        if isinstance(m, nn.Linear) and init != "default":
            if isinstance(init, str):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # init == "kaiming"
            else:
                nn.init.normal_(m.weight, std=init)                      # otherwise init is a number: the std
            nn.init.zeros_(m.bias)
    return net

def grad_norms(net, seed=0):
    torch.manual_seed(seed)
    x, t = torch.randn(64, 32), torch.randn(64, 1)
    net.zero_grad()
    ((net(x) - t) ** 2).mean().backward()
    return [m.weight.grad.norm().item() for m in net if isinstance(m, nn.Linear)]

torch.manual_seed(0)
g_demo = grad_norms(make_mlp(nn.ReLU, "kaiming"))
print(len(g_demo), "numbers:", [round(v, 3) for v in g_demo])
assert len(g_demo) == 11

# %% [markdown]
# ### ✏️ Your turn: count the numbers
#
# `grad_norms` returned a list. How many numbers are in it for the 10-layer network, and why? Store the count in `answer`, and think about the reason: the network has 10 hidden linear layers and one more.

# %%
answer = None   # YOUR ANSWER: the length of the list

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("number of gradient norms", answer == len(g_demo), "10 hidden linear layers plus 1 output layer")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = len(g_demo)    # 11
# ```
# `make_mlp` returns the network, `grad_norms` returns one weight-gradient norm per linear layer. There are 11 because the network has 10 hidden `Linear` layers and one final `Linear` layer that outputs the single number.
# </details>

# %% [markdown]
# ## Step 5. Compare activations and initialisations
#
# We build five networks that differ only in the activation or the starting weights, and plot their gradient norms on a log axis (gradient sizes span many orders of magnitude). Kaiming initialisation sets the weight spread so that signals keep about the same size from layer to layer when you use ReLU.

# %%
torch.manual_seed(0)
cases = {"sigmoid, default": make_mlp(nn.Sigmoid), "ReLU, default": make_mlp(nn.ReLU),
         "ReLU, Kaiming": make_mlp(nn.ReLU, "kaiming"),
         "ReLU, std 0.01": make_mlp(nn.ReLU, 0.01), "ReLU, std 0.5": make_mlp(nn.ReLU, 0.5)}
norms = {}
plt.figure(figsize=(6, 4))
for name, net in cases.items():
    g = grad_norms(net)
    norms[name] = g
    plt.semilogy(range(1, 12), g, marker="o", label=name)
    print("%-17s first layer %.1e, last layer %.1e" % (name, g[0], g[-1]))
plt.xlabel("layer (1 = closest to the input)"); plt.ylabel("gradient norm"); plt.legend(fontsize=8); plt.show()

# %%
# Sanity check: the pattern the lesson describes
sig, kai = norms["sigmoid, default"], norms["ReLU, Kaiming"]
assert sig[0] < 1e-4 * sig[-1], "sigmoid: the first layer should get a far smaller gradient than the last"
assert max(kai) / min(kai) < 100, "Kaiming: all layers should be within about two orders of magnitude"
print("sigmoid first/last = %.1e, Kaiming max/min = %.1f" % (sig[0] / sig[-1], max(kai) / min(kai)))

# %% [markdown]
# ### ✏️ Your turn: vanishing, healthy or exploding?
#
# Look at the plot and the printed numbers. Fill in the dictionary: for each case write `"vanishing"`, `"healthy"` or `"exploding"`. The case with Kaiming initialisation is your reference for "healthy". Then explain each choice in a sentence, using the numbers.

# %%
answer = {
    "sigmoid, default": None,   # YOUR ANSWER
    "ReLU, default": None,
    "ReLU, Kaiming": None,
    "ReLU, std 0.01": None,
    "ReLU, std 0.5": None,
}

# %%
# Check: compare each case's first-layer gradient with the Kaiming reference
ref = norms["ReLU, Kaiming"][0]
def label(name):
    f = norms[name][0] / ref
    return "vanishing" if f < 0.01 else ("exploding" if f > 100 else "healthy")
truth = {k: label(k) for k in cases}
if any(v is None for v in answer.values()):
    print("Fill in your answers above, then run this cell again.")
else:
    for k in cases:
        check(k, answer[k] == truth[k], "compare its first-layer norm to the Kaiming case: 100x smaller is vanishing, 100x larger is exploding")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = {"sigmoid, default": "vanishing", "ReLU, default": "vanishing",
#           "ReLU, Kaiming": "healthy", "ReLU, std 0.01": "vanishing", "ReLU, std 0.5": "exploding"}
# ```
# The sigmoid slope is at most 0.25, so each layer shrinks the gradient. A standard deviation of 0.01 shrinks the signal at every layer. A standard deviation of 0.5 grows it at every layer. Kaiming keeps the norms within about one order of magnitude of each other. With PyTorch's default initialisation a ReLU network is smaller throughout, because the default weights are a little too small for ReLU.
# </details>

# %% [markdown]
# ## Step 6. How does depth matter?
#
# The more layers a gradient passes back through, the more times it is multiplied by a small number. We measure the ratio *first-layer norm ÷ last-layer norm* for sigmoid networks of different depths.

# %%
ratios = {}
for depth in [2, 5, 10, 20]:
    g = grad_norms(make_mlp(nn.Sigmoid, depth=depth))
    ratios[depth] = g[0] / g[-1]
    print(depth, "first/last = %.1e" % ratios[depth])
assert ratios[20] < ratios[10] < ratios[5] < ratios[2], "the ratio should fall as depth grows"

# %% [markdown]
# ### ✏️ Your turn: the trend
#
# By roughly what factor does the ratio shrink for **each extra layer** between depth 5 and depth 10? Compute `(ratios[10] / ratios[5]) ** (1 / 5)` in your own words: the ratio changed by `ratios[10] / ratios[5]` over 5 layers. Store the per-layer factor in `answer`.

# %%
answer = None   # YOUR CODE HERE: the per-layer factor between depth 5 and depth 10

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    true_f = (ratios[10] / ratios[5]) ** (1 / 5)
    check("per-layer factor", abs(answer - true_f) / true_f < 0.05, "take the 5th root of ratios[10] / ratios[5]")
    print("It shrinks by about a factor of %.2f per extra layer." % true_f)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = (ratios[10] / ratios[5]) ** (1 / 5)
# ```
# The gradient is multiplied by a factor below 1 at every layer, so the ratio falls exponentially with depth: each extra layer makes the early layers' gradient smaller by roughly the same factor.
# </details>

# %% [markdown]
# ## Step 7. Gradient clipping
#
# For the exploding case there is a direct fix: if the total gradient norm is larger than a limit, scale all gradients down so that the norm equals the limit. `clip_grad_norm_` does this and returns the norm it measured **before** clipping.

# %%
torch.manual_seed(0)
net = make_mlp(nn.ReLU, 0.5)
x, t = torch.randn(64, 32), torch.randn(64, 1)
((net(x) - t) ** 2).mean().backward()
before = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
after = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters()))
print("norm before clipping: %.3e, after: %.4f" % (before.item(), after.item()))
assert before.item() > 1 and abs(after.item() - 1.0) < 1e-4

# %% [markdown]
# ### ✏️ Your turn: read `before` and `after`
#
# Store in `answer` the **factor by which clipping scaled every gradient** (a number between 0 and 1). Then think about why you would clip instead of just using a smaller learning rate: clipping only acts when the gradient is too large, and keeps its direction.

# %%
answer = None   # YOUR CODE HERE: the scale factor, from `before` and `after`

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("scale factor", abs(answer - after.item() / before.item()) < 1e-9, "after / before")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = after.item() / before.item()
# ```
# `before` is the real gradient size (far above 1), `after` is exactly 1.0, the limit we set. Every gradient was multiplied by the same factor, so the direction is unchanged and only the step length is cut. A smaller learning rate would also shrink the harmless, small gradients.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - You checked a hand-computed backward pass two ways: autograd and finite differences.
# - You measured gradient norms layer by layer and saw them vanish (sigmoid, tiny weights), stay healthy (Kaiming) and explode (large weights).
# - You saw that depth makes vanishing worse, and that gradient clipping caps an exploding gradient.
#
# **Reflection.** Double-click here to write 4–5 sentences you could say to a classmate: why is a very deep sigmoid network hard to train, and what are two fixes?
#
# Files in `/content` vanish when the session ends, so download anything you want to keep (File → Download). Submit your notebook the way your instructor asks.
#
# **Go further.** [Tutorial 3, Activation functions](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html) plots the activations with their gradients and looks at dead ReLU neurons. [Tutorial 4, Optimization and initialization](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html) explains Xavier and Kaiming initialisation and builds SGD, momentum and Adam from scratch. Both download FashionMNIST. Tutorials 3 and 4 by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence).
