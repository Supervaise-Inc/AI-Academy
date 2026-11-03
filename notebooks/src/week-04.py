# %% [markdown]
# # Week 4 — Convolutional Neural Networks: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** CPU is enough (the default). **Time:** about 3 hours if you do every exercise; the code itself runs in a few minutes.
#
# **What you will do**
# - Compute a convolution by hand and check it with PyTorch.
# - Test the output-size formula against real layers.
# - Build a small CNN and a fully connected network of similar size, and count their parameters.
# - Train both on 8×8 handwritten digits and compare them.
# - Look at what the first layer learned and test both networks on shifted images.
# - Follow the shape of a tensor through the CNN.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you to fill in. A **Check** cell tells you whether you are right. Open **Show a possible answer** only after you have tried.

# %% [markdown]
# ## Setup
#
# We use `torch`, `numpy`, `matplotlib` and `scikit-learn`, which Colab already has. The dataset is `sklearn.datasets.load_digits`: 1,797 greyscale digits, each 8 × 8 pixels with values 0 to 16. It ships inside scikit-learn, so nothing is downloaded.

# %%
import os, random
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
QUICK = bool(os.environ.get("NB_TEST"))   # shortens long loops when the notebook is tested

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1: convolution by hand and with PyTorch
#
# In the lesson section *The convolution operation*, a convolution places a kernel on a patch of the image, multiplies matching numbers, adds them up, and slides one step. PyTorch's `F.conv2d` does not flip the kernel, so your hand calculation must not flip it either.
#
# The image has three bright columns (9) and two dark columns (0). The kernel responds to a bright-to-dark vertical edge.

# %%
A = torch.tensor([[9, 9, 9, 0, 0]] * 5, dtype=torch.float32).reshape(1, 1, 5, 5)
K = torch.tensor([[1, 0, -1]] * 3, dtype=torch.float32).reshape(1, 1, 3, 3)
out = F.conv2d(A, K).squeeze()
print(out)                      # every row should read [0, 27, 27]
assert out.tolist() == [[0.0, 27.0, 27.0]] * 3, "unexpected convolution result"

# %% [markdown]
# Now the same operation written as plain loops, including stride and padding. We test it on a 6 × 6 image that goes dark to bright, with stride 2 and padding 1.

# %%
def conv2d_manual(img, ker, stride=1, pad=0):
    img = np.pad(img, pad)                       # border of zeros
    k = ker.shape[0]
    n = (img.shape[0] - k) // stride + 1         # output size
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            patch = img[i*stride:i*stride+k, j*stride:j*stride+k]
            out[i, j] = (patch * ker).sum()      # multiply matching numbers, add up
    return out

a2 = np.array([[0, 0, 0, 9, 9, 9]] * 6, dtype=float)
k2 = np.array([[1, 0, -1]] * 3, dtype=float)
mine = conv2d_manual(a2, k2, stride=2, pad=1)
torch_out = F.conv2d(torch.tensor(a2).reshape(1, 1, 6, 6).float(),
                     torch.tensor(k2).reshape(1, 1, 3, 3).float(), stride=2, padding=1).squeeze().numpy()
print(mine)
print(torch_out)
assert np.allclose(mine, torch_out), "manual and PyTorch results differ"

# %% [markdown]
# ### ✏️ Your turn: flip the image
#
# Change the image so every row reads `0, 0, 9, 9, 9` (dark on the left, bright on the right). Work out the **first output row** by hand (three numbers, stride 1, no padding), then type it below. Why do you think the sign changes?

# %%
answer = None   # YOUR CODE HERE, for example answer = [0, 0, 0]

# %%
B = torch.tensor([[0, 0, 9, 9, 9]] * 5, dtype=torch.float32).reshape(1, 1, 5, 5)
true_row = F.conv2d(B, K).squeeze()[0].tolist()
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("first output row", [float(v) for v in answer] == true_row,
          "Compute left column minus right column for each patch. Remember the kernel is not flipped.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = [-27, -27, 0]
# ```
#
# The first patch covers columns 1–3 (0, 0, 9): left column sums to 0, right column to 27, so 0 − 27 = −27. The kernel looks for bright-to-dark; this image goes dark-to-bright, so the response flips sign.
# </details>

# %% [markdown]
# ## Step 2: the output-size formula
#
# From the lesson section *Stride, padding and output size*: for input size n, kernel k, padding p and stride s, the output size is ⌊(n + 2p − k) / s⌋ + 1. We check it against real `nn.Conv2d` layers.

# %%
def out_size(n, k, p=0, s=1):
    return (n + 2 * p - k) // s + 1              # // rounds down

cases = [(28, 3, 0, 1), (28, 3, 1, 1), (28, 5, 2, 2), (7, 3, 0, 2), (32, 3, 1, 2), (224, 7, 3, 2)]
for n, k, p, s in cases:
    real = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, n, n)).shape[-1]
    print(f"n={n:3d} k={k} p={p} s={s}  formula: {out_size(n, k, p, s):3d}  PyTorch: {real:3d}")
    assert out_size(n, k, p, s) == real

# %% [markdown]
# ### ✏️ Your turn: predict two sizes
#
# By hand, find the output size for **(n = 50, k = 5, p = 0, s = 1)** and for **(n = 10, k = 3, p = 0, s = 2)**. Why is rounding down needed in the second case?

# %%
answer_a = None   # YOUR CODE HERE: size for n=50, k=5, p=0, s=1
answer_b = None   # YOUR CODE HERE: size for n=10, k=3, p=0, s=2

# %%
if answer_a is None or answer_b is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("n=50, k=5, p=0, s=1", answer_a == out_size(50, 5, 0, 1), "Compute (50 + 0 - 5) / 1 + 1.")
    check("n=10, k=3, p=0, s=2", answer_b == out_size(10, 3, 0, 2), "Compute (10 + 0 - 3) / 2, round down, then add 1.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_a = 46   # (50 - 5) / 1 + 1
# answer_b = 4    # floor(7 / 2) + 1 = 3 + 1
# ```
#
# In the second case (10 − 3) / 2 = 3.5, and the kernel can only be placed at whole positions (0, 2, 4, 6), so we round down to 3 and add 1. The last column is never reached.
# </details>

# %% [markdown]
# ## Step 3: load the digits, build both networks, count parameters
#
# We build a small CNN and a fully connected (FC) network of about the same size, so the comparison is fair. A conv layer expects `(batch, channels, height, width)`, so we add the channel dimension with `.unsqueeze(1)`, and we divide the pixels by 16 to keep inputs between 0 and 1.

# %%
from sklearn.datasets import load_digits
d = load_digits()
X = torch.tensor(d.images / 16.0, dtype=torch.float32).unsqueeze(1)   # (1797, 1, 8, 8)
y = torch.tensor(d.target)
g = torch.Generator().manual_seed(SEED)
idx = torch.randperm(len(X), generator=g)
tr, va = idx[:1400], idx[1400:]

def make_cnn():
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64, 10))

def make_fc():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 25), nn.ReLU(), nn.Linear(25, 10))

count = lambda m: sum(p.numel() for p in m.parameters())
print(X.shape, "CNN params:", count(make_cnn()), "FC params:", count(make_fc()))
assert (count(make_cnn()), count(make_fc())) == (1898, 1885)

# %% [markdown]
# ### ✏️ Your turn: where do the 1,898 come from?
#
# A conv layer has k × k × C_in × C_out + C_out parameters; a linear layer has inputs × outputs + outputs. Compute the parameters of the CNN's **second** conv layer (8 input channels, 16 filters, 3 × 3) and the final `Linear(64, 10)` layer.

# %%
conv2_params = None    # YOUR CODE HERE
linear_params = None   # YOUR CODE HERE

# %%
if conv2_params is None or linear_params is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("conv 2", conv2_params == 3*3*8*16 + 16, "k*k*C_in*C_out + C_out = 3*3*8*16 + 16.")
    check("linear", linear_params == 64*10 + 10, "inputs*outputs + outputs.")
    check("all three layers add up to 1898", 80 + conv2_params + linear_params == 1898 if conv2_params and linear_params else False,
          "Conv 1 has 80, conv 2 and the linear layer should make 1898 together.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# conv2_params = 3*3*8*16 + 16   # 1168
# linear_params = 64*10 + 10     # 650
# ```
#
# Conv 1: 3×3×1×8 + 8 = 80. Conv 2: 1,168. Linear: 650. Total 80 + 1,168 + 650 = 1,898. We chose 25 hidden units for the FC network because 64×25 + 25 + 25×10 + 10 = 1,885, almost the same size as the CNN.
# </details>

# %% [markdown]
# ## Step 4: train and compare
#
# Both networks get the same data, the same optimiser (Adam, a variant of gradient descent that adapts the step size of each parameter) and the same number of epochs. The loop is the same five steps as in Week 2: forward, loss, zero the gradients, backward, update. Labels must be integers for `CrossEntropyLoss`; `d.target` already is.

# %%
EPOCHS = 5 if QUICK else 30

def fit(model, epochs=EPOCHS, lr=0.01, seed=SEED):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    first = last = None
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            loss = loss_fn(model(X[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if first is None: first = loss.item()
            last = loss.item()
    print(f"loss at first step {first:.3f}, at last step {last:.3f}")
    assert last < first, "the loss did not go down"
    return model

def acc(model, Xs, ys):
    with torch.no_grad():
        return model(Xs).argmax(1).eq(ys).float().mean().item()

torch.manual_seed(SEED); cnn = fit(make_cnn())
torch.manual_seed(SEED); fc = fit(make_fc())
print(f"validation accuracy  CNN: {acc(cnn, X[va], y[va]):.3f}   FC: {acc(fc, X[va], y[va]):.3f}")

# %% [markdown]
# ### ✏️ Your turn: read the result
#
# Which network won on the validation set, and by how many percentage points? Set `winner` to `"cnn"` or `"fc"` (or `"tie"` if they are equal) and `gap_points` to the difference in percentage points, rounded to one decimal.

# %%
winner = None       # YOUR CODE HERE: "cnn", "fc" or "tie"
gap_points = None   # YOUR CODE HERE: e.g. 1.5

# %%
a_cnn, a_fc = acc(cnn, X[va], y[va]), acc(fc, X[va], y[va])
true_winner = "tie" if a_cnn == a_fc else ("cnn" if a_cnn > a_fc else "fc")
true_gap = round(abs(a_cnn - a_fc) * 100, 1)
if winner is None or gap_points is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("winner", winner == true_winner, "Compare the two validation accuracies printed above.")
    check("gap", abs(gap_points - true_gap) <= 0.1, "Subtract the smaller accuracy from the larger and multiply by 100.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# winner = "cnn" if acc(cnn, X[va], y[va]) > acc(fc, X[va], y[va]) else "fc"
# gap_points = round(abs(acc(cnn, X[va], y[va]) - acc(fc, X[va], y[va])) * 100, 1)
# ```
#
# Digits is an easy dataset and these images are tiny, so both networks usually score high and the gap is small. One run with one seed is not strong evidence: the winner can change with the seed.
# </details>

# %% [markdown]
# ## Step 5: what did the first layer learn?
#
# The first conv layer has eight 3 × 3 kernels. Because they are just 9 numbers each, we can draw them as tiny pictures. In the lesson we wrote an edge kernel by hand; a trained network finds its own.

# %%
fig, axes = plt.subplots(1, 8, figsize=(10, 1.6))
for a, w in zip(axes, cnn[0].weight.detach()):
    a.imshow(w[0].numpy(), cmap="gray"); a.axis("off")
plt.suptitle("The eight 3×3 kernels of the first layer")
plt.show()

# %% [markdown]
# ### ✏️ Your turn: describe the kernels
#
# Double-click this cell and write two sentences: do any of the eight kernels look like the vertical-edge kernel from Step 1 (a column of positive values next to a column of negative values)? Are some of them mostly smooth blobs? Be honest about what you see; with a small network and a small dataset the kernels are often not clean edge detectors.

# %% [markdown]
# ## Step 6: shifted images
#
# Max pooling makes a network a little tolerant to small shifts, and weight sharing lets a kernel find a pattern anywhere. A fully connected layer has a separate weight for every pixel position. We shift every validation image one pixel to the right (wrapping around) and test both networks. Run it with a few seeds, because one run on a small dataset is noisy.

# %%
Xs = torch.roll(X[va], shifts=1, dims=3)         # shift right by 1 pixel, wrapping around
seeds = [0, 1] if QUICK else [0, 1, 2]
rows = []
for sd in seeds:
    c = fit(make_cnn(), seed=sd); f = fit(make_fc(), seed=sd)
    rows.append((sd, acc(c, X[va], y[va]), acc(c, Xs, y[va]), acc(f, X[va], y[va]), acc(f, Xs, y[va])))
print("seed | CNN normal -> shifted | FC normal -> shifted")
for sd, cn, cs, fn, fs in rows:
    print(f"{sd:4d} | {cn:.3f} -> {cs:.3f} (drop {cn-cs:+.3f}) | {fn:.3f} -> {fs:.3f} (drop {fn-fs:+.3f})")

# %% [markdown]
# ### ✏️ Your turn: explain what you saw
#
# Before looking at the numbers again, which network would you expect to lose more accuracy, and why (think about weight sharing)? Set `prediction` to `"cnn"` or `"fc"` (the network you expect to lose more). The Check cell reports what your runs showed; with tiny images and few runs the result can be noisy, so it does not mark you wrong either way.

# %%
prediction = None   # YOUR CODE HERE: "cnn" or "fc"

# %%
mean_drop_cnn = np.mean([r[1] - r[2] for r in rows])
mean_drop_fc = np.mean([r[3] - r[4] for r in rows])
if prediction is None:
    print("Fill in your answer above, then run this cell again.")
else:
    observed = "cnn" if mean_drop_cnn > mean_drop_fc else "fc"
    print(f"Mean accuracy drop: CNN {mean_drop_cnn:.3f}, FC {mean_drop_fc:.3f}. The network that lost more in your runs: {observed}.")
    print("Your prediction matched." if prediction == observed else "Your prediction did not match these runs. Write down why that might be.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# The usual expectation is that the CNN loses less, because the same kernels are applied at every position and pooling tolerates small shifts, while the FC layer ties each weight to one pixel position. But our images are only 8 × 8, one pixel is a big shift at that size, and we ran a few seeds, so report your own numbers and do not claim more than they show.
# </details>

# %% [markdown]
# ## Step 7: shapes through the CNN
#
# Shape bookkeeping is a classic place for mistakes. We send one blank 8 × 8 image through the network, layer by layer, and print the shape after each one. This is the digits version of the lesson's *Shapes through a small CNN* figure.

# %%
x = torch.zeros(1, 1, 8, 8)
shapes = []
for layer in make_cnn():
    x = layer(x)
    shapes.append((type(layer).__name__, tuple(x.shape)))
    print(f"{type(layer).__name__:10s}", tuple(x.shape))
assert shapes[-2][1] == (1, 64) and shapes[-1][1] == (1, 10)

# %% [markdown]
# ### ✏️ Your turn: change the pooling
#
# Suppose we remove the **second** `MaxPool2d(2)` from the CNN. The last conv layer then outputs 16 channels at the size after the first pool. How many numbers does `Flatten` produce, and so what must the final `Linear` layer's input size be? Fill in `new_flat`.

# %%
new_flat = None   # YOUR CODE HERE

# %%
test_net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
true_flat = test_net(torch.zeros(1, 1, 8, 8)).shape[1]
if new_flat is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("flatten size", new_flat == true_flat, "After the first pool the map is 4 x 4 with 16 channels.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# new_flat = 4 * 4 * 16   # 256
# ```
#
# The map stays 4 × 4 with 16 channels, so Flatten gives 256 numbers and the final layer must be `nn.Linear(256, 10)`. If you leave `Linear(64, 10)` unchanged, PyTorch raises a shape error.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - A convolution is multiply-and-add under a sliding kernel; the output size is ⌊(n + 2p − k) / s⌋ + 1 and a layer has k × k × C_in × C_out + C_out parameters.
# - Weight sharing gives a CNN far fewer parameters than a fully connected layer on the same image, and it never depends on image size.
# - Following the tensor shape layer by layer is how you avoid shape errors when you change a network.
#
# **Reflection:** double-click to write two sentences: (a) one advantage of convolution over a fully connected layer for images, and (b) why you should not read too much into a single run on a small dataset like this one.
#
# Files in `/content` vanish when your Colab session ends, so download anything you want to keep (File → Download). Submit your notebook the way your instructor asks.
#
# **Go further:** [UvA Deep Learning Tutorial 5: Inception, ResNet, DenseNet](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html) by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence). It downloads CIFAR10 and is better with a GPU.
