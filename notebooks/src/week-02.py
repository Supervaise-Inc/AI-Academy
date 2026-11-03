# %% [markdown]
# # Week 2 — Machine Learning and Neural Networks: lab notebook
#
# Built for **Google Colab**. First click **File → Save a copy in Drive**, so your work is saved in your own copy.
#
# **Runtime:** a CPU is enough. You do not need a GPU. **Time:** about 3 hours.
#
# **What you will do**
# - Compute one neuron by hand and check it in code.
# - Make a noisy XOR dataset and split it into training and validation sets.
# - Build and train a small PyTorch network with the five-step training loop.
# - Plot the decision boundary.
# - Change the hidden size and the learning rate, and compare the results.
# - Overfit on purpose, then write your reflection.
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.

# %% [markdown]
# ## Setup
#
# This cell imports what we need, fixes the random seed so your numbers repeat, and defines a small `check` helper that prints ✓ or ✗. Colab already has PyTorch, NumPy, scikit-learn and matplotlib, so nothing needs to be installed and nothing is downloaded: we generate the data ourselves.

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
# ## Step 1. One neuron by hand and in code
#
# From the lesson section on the artificial neuron: a neuron multiplies each input by a weight, adds a bias to get a score *z*, and passes *z* through an activation (here the sigmoid). The loss we use for numbers is the mean squared error. Run the cell and see that the printed values match the comments.

# %%
x = torch.tensor([2.0, 1.0]); w = torch.tensor([0.5, -1.0]); b = torch.tensor(0.25)
z = (w * x).sum() + b                       # z = 0.5*2 + (-1)*1 + 0.25
print(round(z.item(), 4), round(torch.sigmoid(z).item(), 4))    # 0.25 and 0.5622
y_true = torch.tensor([3.0, 5.0, 2.0]); p = torch.tensor([2.5, 5.5, 4.0])
mse = ((y_true - p) ** 2).mean()            # (0.25 + 0.25 + 4) / 3
print(mse.item())                           # 1.5
assert abs(mse.item() - 1.5) < 1e-6 and abs(torch.sigmoid(z).item() - 0.5622) < 1e-4

# %% [markdown]
# ### ✏️ Your turn: change the neuron
#
# Change the bias to −0.25 and the first weight to 1.5 (keep the same inputs `[2.0, 1.0]` and second weight −1.0). **First work out the sigmoid output on paper**, then type it below (four decimals).

# %%
answer = None   # YOUR ANSWER: sigmoid(z) for the new weights and bias, e.g. 0.1234

# %%
# Check
z_new = 1.5 * 2.0 + (-1.0) * 1.0 + (-0.25)
true_out = 1 / (1 + math.exp(-z_new))
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("sigmoid output", abs(answer - true_out) < 1e-3, "z = w1*x1 + w2*x2 + b, then apply 1 / (1 + e^(-z))")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# z = 1.5 * 2.0 + (-1.0) * 1.0 + (-0.25)     # 1.75
# print(1 / (1 + math.exp(-z)))              # 0.8520
# ```
# The score is 1.75, and the sigmoid of 1.75 is about 0.8520. A larger first weight and a smaller bias pushed the score up.
# </details>

# %% [markdown]
# ## Step 2. Make and split the data
#
# We draw 400 points in the square from −1 to 1. The label is 1 when *x₁* and *x₂* have opposite signs (that is the XOR pattern from the lesson). Then we add noise to the positions, so some points end up on the wrong side of their label. Finally we keep 300 points for training and 100 for validation.

# %%
torch.manual_seed(SEED)
n = 400
X = torch.rand(n, 2) * 2 - 1
y = ((X[:, 0] * X[:, 1]) < 0).float()        # label 1 when the signs differ
X = X + 0.1 * torch.randn(n, 2)              # noisy positions
idx = torch.randperm(n)
tr, va = idx[:300], idx[300:]
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
plt.figure(figsize=(4, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12)
plt.title("Noisy XOR: red = 1, blue = 0"); plt.show()
print(Xtr.shape, Xva.shape, "share of label 1 in training:", round(ytr.mean().item(), 3))

# %% [markdown]
# ### ✏️ Your turn: why a straight line fails
#
# The four quadrants form an XOR pattern. A model with no hidden layer can only draw one straight line. Fit scikit-learn's `LogisticRegression` (a model with no hidden layer) on the training data, and store its **validation accuracy** in `answer`.

# %%
from sklearn.linear_model import LogisticRegression
answer = None   # YOUR CODE HERE: fit LogisticRegression on Xtr, ytr (as numpy) and score it on Xva, yva

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("accuracy is a fraction between 0 and 1", 0 <= answer <= 1, "use .score(...)")
    check("a straight line is near chance (about 0.5)", 0.35 <= answer <= 0.7, "fit on the XOR data Xtr.numpy(), ytr.numpy()")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# lin = LogisticRegression().fit(Xtr.numpy(), ytr.numpy())
# answer = lin.score(Xva.numpy(), yva.numpy())
# ```
# The opposite corners share a label, so any single line puts about half of each class on the wrong side. Accuracy stays close to 50%, the same as guessing.
# </details>

# %% [markdown]
# ## Step 3. The model and the training loop
#
# From the lesson section on training: a network with 2 inputs, 4 tanh hidden neurons and 1 output has 17 parameters (2×4 + 4 for the first layer, 4×1 + 1 for the second). The loop has five stages: predict, compute the loss, clear old gradients, backward, update. We use `BCEWithLogitsLoss`, which takes the raw score (the "logit") and applies the sigmoid inside.

# %%
def train(H=4, lr=0.1, epochs=2000, seed=0, X=Xtr, y=ytr):
    if QUICK: epochs = min(epochs, 2000)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, H), nn.Tanh(), nn.Linear(H, 1))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(epochs):
        logits = model(X).squeeze(1)            # stage 1: predict
        loss = loss_fn(logits, y)               # stage 2: loss
        opt.zero_grad(); loss.backward()        # stage 3: clear old gradients, stage 4: backward
        opt.step()                              # stage 5: update
        losses.append(loss.item())
    return model, losses

def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).squeeze(1) > 0).float().eq(y).float().mean().item()

# %%
model, losses = train()
n_params = sum(p.numel() for p in model.parameters())
print("parameters:", n_params)
print("first loss %.3f, last loss %.3f" % (losses[0], losses[-1]))
print("train acc %.3f, validation acc %.3f" % (accuracy(model, Xtr, ytr), accuracy(model, Xva, yva)))
plt.figure(figsize=(4, 3)); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.show()
assert n_params == 17 and losses[-1] < losses[0], "the loss should go down"

# %% [markdown]
# ### ✏️ Your turn: read the loop
#
# Which of the five stages is the one where the gradients are computed? Put its number (1–5) in `answer`. Then think about this in your own words: why does `BCEWithLogitsLoss` take raw scores rather than probabilities? (Hint: it combines the sigmoid and the loss in one step, which is more numerically stable.)

# %%
answer = None   # YOUR ANSWER: a number from 1 to 5

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("stage where gradients are computed", answer == 4, "look for the line that calls loss.backward()")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = 4     # loss.backward() fills in the gradients
# ```
# Stage 4 is the backward pass. `BCEWithLogitsLoss` takes raw scores because it applies the sigmoid and the log together inside one function, which avoids rounding problems when probabilities get very close to 0 or 1.
# </details>

# %% [markdown]
# ## Step 4. Plot the decision boundary
#
# We evaluate the network on a grid of points and colour each point by its predicted probability. The line where the colour changes is the **decision boundary**. We draw it on the validation points, which the network never trained on.

# %%
def plot_boundary(model, X, y, title=""):
    g = torch.linspace(-1.5, 1.5, 200)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        prob = torch.sigmoid(model(grid)).reshape(200, 200)
    plt.figure(figsize=(4, 4))
    plt.contourf(gx.numpy(), gy.numpy(), prob.numpy(), levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12)
    plt.title(title); plt.show()

plot_boundary(model, Xva, yva, "Validation points and the learned boundary")

# %% [markdown]
# ### ✏️ Your turn: quadrants
#
# How many of the four quadrants does the network get right? Test the centre of each quadrant, at (±0.5, ±0.5): the true label is 1 when the signs differ. Store the number of centres that the network classifies correctly in `answer`.

# %%
answer = None   # YOUR CODE HERE: predict the 4 points below with `model` and count the matches
centres = torch.tensor([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
truth = torch.tensor([0.0, 1.0, 0.0, 1.0])

# %%
# Check
with torch.no_grad():
    correct = ((model(centres).squeeze(1) > 0).float() == truth).sum().item()
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("number of quadrants right", answer == correct, "predict with model(centres), compare to truth")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pred = (model(centres).squeeze(1) > 0).float()
# answer = int((pred == truth).sum())
# ```
# A trained network usually gets all four quadrants right. Its mistakes cluster near the two axes, where the noise pushed points across the line between classes, so the true label and the position disagree.
# </details>

# %% [markdown]
# ## Step 5. Experiments: hidden size and learning rate
#
# Now change one setting at a time. First the number of hidden neurons H, then the learning rate. This is the lesson's point that settings like these are chosen by you and checked on validation data.

# %%
results = {}
for H in [1, 2, 4, 16]:
    m, l = train(H=H)
    results[H] = (l[-1], accuracy(m, Xtr, ytr), accuracy(m, Xva, yva))
    print("H =", H, "loss", round(l[-1], 3), "train", round(results[H][1], 3), "val", round(results[H][2], 3))

# %%
lr_results = {}
for lr in [0.01, 0.1, 1.0]:
    m, l = train(lr=lr)
    lr_results[lr] = (l[-1], accuracy(m, Xva, yva))
    print("lr =", lr, "loss", round(l[-1], 3), "val", round(lr_results[lr][1], 3))

# %% [markdown]
# ### ✏️ Your turn: read the table
#
# Which is the **smallest hidden size** in `[1, 2, 4, 16]` whose validation accuracy is at least 0.85? Put it in `answer`. Then write one sentence in your own words: what happens with lr = 0.01 after 2000 epochs, and why? (Look at its loss in the output above.)

# %%
answer = None   # YOUR CODE HERE: the smallest H with validation accuracy >= 0.85 (use the `results` dict)

# %%
# Check
good = [H for H in sorted(results) if results[H][2] >= 0.85]
true_H = good[0] if good else None
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("smallest working hidden size", answer == true_H, "loop over sorted(results) and stop at the first H with results[H][2] >= 0.85")
    check("lr = 0.01 learned less than lr = 0.1", lr_results[0.01][0] > lr_results[0.1][0], "compare the final losses")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = next(H for H in sorted(results) if results[H][2] >= 0.85)
# ```
# One hidden neuron can only build one line, so it cannot solve XOR. With lr = 0.01 the steps are too small: after 2000 epochs the loss is still high, so the network has not learned much yet.
# </details>

# %% [markdown]
# ## Step 6. Overfitting on purpose
#
# We train a wide network (64 hidden neurons) on only 20 training points and compare the accuracy on those points with the accuracy on the validation set. A big gap means **overfitting**: the model memorised its training points instead of learning the pattern, so it **generalises** worse to new data.

# %%
m_over, l_over = train(H=64, epochs=3000, X=Xtr[:20], y=ytr[:20])
train_acc = accuracy(m_over, Xtr[:20], ytr[:20]); val_acc = accuracy(m_over, Xva, yva)
print("train", round(train_acc, 3), "val", round(val_acc, 3))
plot_boundary(m_over, Xva, yva, "64 hidden neurons, 20 training points")

# %% [markdown]
# ### ✏️ Your turn: measure the gap
#
# Store the **gap** (training accuracy minus validation accuracy) of the overfitted network in `answer`. Then write two or three sentences in a text cell using the words *overfitting* and *generalisation*. Does the boundary look sensible?

# %%
answer = None   # YOUR CODE HERE: train_acc - val_acc

# %%
# Check
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("gap computed correctly", abs(answer - (train_acc - val_acc)) < 0.01, "answer = train_acc - val_acc")
    check("training accuracy is higher than validation accuracy", train_acc > val_acc, "rerun the overfitting cell")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = train_acc - val_acc
# ```
# The network fits its 20 training points almost perfectly, but the validation accuracy is lower: that gap is overfitting. The boundary bends around individual training points instead of following the four quadrants, so it generalises less well. The problem is simple, so the gap is smaller than it would be on a harder one; the pattern is what matters.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - You trained a two-layer network with the five-step loop and saw it solve a problem that no straight line can.
# - You saw that hidden size and learning rate change the result, and that you judge a model on validation data, not training data.
# - You made a model overfit and measured the gap between training and validation accuracy.
#
# **Reflection.** Double-click here to write two sentences: (a) which one setting would you check first if a loss curve did not fall, and (b) why do we measure accuracy on the validation set instead of the training set?
#
# Files in `/content` vanish when the session ends, so download anything you want to keep (File → Download). Submit your notebook the way your instructor asks.
#
# **Go further.** [Tutorial 2, Introduction to PyTorch](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html) covers tensors, the computation graph, a small linear-tanh-linear network on noisy XOR, the `DataLoader` and more. Tutorial 2 by Phillip Lippe, University of Amsterdam (UvA Deep Learning Tutorials, MIT licence).
