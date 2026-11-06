# %% [markdown]
# # Week 2 — Supervised Learning and Neural Networks: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is fine for everything; no GPU is needed and nothing is downloaded (all the small data sets are generated inside the notebook). **Time:** about 3 hours.
#
# This lab is called **Imagine a neural network, then build one**. First you picture a network: as a machine of dials, as a team of voters, as a landscape of mistakes. Then you build one by hand, in NumPy and in PyTorch, watch it learn, watch it fail (too small, too big, wrong learning rate) and finally design a small classifier of your own.
#
# ### What you will do
#
# 1. **Part 1. Supervised learning by hand: be the model** (25 min): invent a rule for ripe mangoes, turn it into a model with three dials and train it by random search.
# 2. **Part 2. Loss: measure the mistake** (20 min): write mean squared error and cross-entropy, and walk through a loss landscape.
# 3. **Part 3. One neuron, built by hand** (25 min): code a neuron and its activations, set the weights of AND and OR, and see why XOR is impossible for one neuron.
# 4. **Part 4. Layers and stacking: a network from scratch** (30 min): a layer as a matrix, a hand-built XOR network, counting parameters.
# 5. **Part 5. Learning: gradient descent you can watch** (30 min): learning rates, gradients checked against PyTorch, backpropagation in NumPy.
# 6. **Part 6. The same thing in PyTorch, then break it** (25 min): the standard training loop, capacity, overfitting and how to fix it.
# 7. **Part 7. Design your own classifier and reflect** (25 min): a spec, a network, a stress test and your reflection.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried. Your key numbers are collected in a dictionary called `RESULTS` that Part 7 prints.

# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed so your results match the ones in the lesson, and creates a few helpers: `check(...)` to test your answers, `make_xor / make_moons / make_circles` to create small labelled datasets, `plot_decision(...)` to draw what a model has learned, and `plot_network_diagram(...)` to draw a network. Everything in this notebook runs on a CPU in under a few minutes; no downloads are needed.

# %%
import os, gc, math, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
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

# %% [markdown]
# ## Part 1. Supervised learning by hand: be the model (25 min)
#
# *Lesson section: "Supervised learning: data, labels, model".*
#
# Supervised learning means learning from examples that come with the correct answer. In this part you are the model: you look at a small table of mangoes, each with a known label (ripe or not ripe), invent a rule, and measure how often the rule is right. Then you turn the rule into a model with three dials (parameters) and find good dial settings by plain random search, the dumbest possible way to train. Along the way you meet the words from the lesson section "Supervised learning": example, feature, label, model, parameter, prediction, training set and test set.

# %% [markdown]
# ### ✏️ Imagine first
#
# You will invent a rule that tells ripe mangoes from unripe ones using only two numbers, the weight and the colour score. Before you look at the data, predict how often your best rule will be right on mangoes it has never seen: write a number between 0 and 1 (0.5 is a coin flip, 1.0 is perfect). You will compare it with the real test accuracy at the end of the part.

# %%
# Imagine first: how often (0 to 1) will your best rule be right on mangoes it has never seen?
my_prediction = None   # replace None with a number between 0 and 1, for example my_prediction = 0.8

# %% [markdown]
# ### Step 1.1. Read a labelled table and split it
#
# Below is a tiny data set written by hand: 24 mangoes from a school garden. Every row is one **example**. It has two **features** (numbers the model may look at: the weight in grams and a colour score from 1 for green to 10 for golden) and one **label** (the correct answer: 1 = ripe, 0 = not ripe yet). The block prints the table and splits it in two by slicing: the first 18 rows are the **training set** (the model may learn from them) and the last 6 rows are the **test set** (kept hidden until the end, to see whether the rule works on new mangoes). The rows are already in mixed order, so both sets contain ripe and unripe fruit.

# %%
# Each row: weight in grams, colour score (1 = green ... 10 = golden), label (1 = ripe, 0 = not ripe yet)
mangoes = [
    (210, 2.0, 0), (320, 8.5, 1), (250, 3.5, 0), (290, 7.0, 1), (180, 1.5, 0), (340, 9.0, 1),
    (300, 4.0, 0), (260, 7.5, 1), (220, 5.0, 0), (310, 6.5, 1), (200, 3.0, 0), (350, 8.0, 1),
    (270, 2.5, 0), (240, 6.0, 1), (230, 4.5, 0), (330, 7.2, 1), (190, 6.5, 0), (280, 5.5, 1),
    (160, 2.2, 0), (300, 8.8, 1), (255, 5.8, 0), (225, 5.2, 1), (310, 3.8, 0), (270, 6.8, 1),
]

print(f"{'#':>2} {'weight_g':>9} {'colour':>7} {'ripe':>5}")
for i, (weight, colour, ripe) in enumerate(mangoes):
    print(f"{i:>2} {weight:>9} {colour:>7.1f} {ripe:>5}")

train, test = mangoes[:18], mangoes[18:]           # the split, by slicing: 18 for training, 6 for testing
print("\nexamples:", len(mangoes), "| train:", len(train), "| test:", len(test))
print("ripe in train:", sum(r for _, _, r in train), "of", len(train), "| ripe in test:", sum(r for _, _, r in test), "of", len(test))

# %% [markdown]
# **What you should see:** You should see 24 numbered rows, then `examples: 24 | train: 18 | test: 6`, then `ripe in train: 9 of 18 | ripe in test: 3 of 6`. Half of each set is ripe, so a model that always answers "ripe" would be right exactly half of the time: 0.5 is the accuracy you must beat.

# %% [markdown]
# ### ✏️ Exercise 1.1: Read a labelled table and split it
#
# Your task: write down, in a markdown cell, the **features** (two names), the **label** (its name and what 1 and 0 mean), one whole **example**, and the number of examples in each set. Why it matters: every supervised-learning project starts with exactly this vocabulary, whether the data are 24 mangoes or 24 million photos. Then fill in the three numbers below.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
n_features = None   # how many features does each example have?
n_train = None      # how many examples are in the training set?
n_test = None       # how many examples are in the test set?

# %%
if n_features is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("2 features per example", n_features == 2, "Count the numbers the model may look at: weight and colour.")
    check("18 training examples", n_train == 18, "We sliced mangoes[:18].")
    check("6 test examples", n_test == 6, "We sliced mangoes[18:].")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# n_features = 2   # weight_g and colour
# n_train = len(train)   # 18
# n_test = len(test)     # 6
# ```
#
# The label is not a feature: it is the answer the model is trying to predict, so it is never an input.
#
# </details>

# %% [markdown]
# ### Step 1.2. Be the model: write your own rule
#
# Now do what a model does. A **rule** is a function that takes the features of one mango and returns a prediction, 1 or 0. The block below holds a first rule that looks only at the weight ("heavy means ripe"), plus two helpers that score any rule: **accuracy** (the fraction of examples the rule gets right, found with a loop) and **MSE**, the mean squared error between your 0/1 predictions and the labels (`mean((prediction − label)²)`). For 0/1 answers a wrong guess costs exactly 1, so MSE here equals the fraction of mistakes.

# %%
def weight_rule(weight, colour):
    """First attempt: heavy mangoes are ripe."""
    return 1 if weight > 270 else 0

def accuracy(rule, rows):
    """Fraction of rows where the rule's answer equals the label."""
    right = 0
    for weight, colour, ripe in rows:
        if rule(weight, colour) == ripe:
            right += 1
    return right / len(rows)

def rule_mse(rule, rows):
    """Mean squared error of 0/1 predictions against the labels."""
    return sum((rule(w, c) - r) ** 2 for w, c, r in rows) / len(rows)

print("weight rule  train accuracy:", round(accuracy(weight_rule, train), 3), "| test accuracy:", round(accuracy(weight_rule, test), 3))
print("weight rule  train MSE     :", round(rule_mse(weight_rule, train), 3))

# %% [markdown]
# **What you should see:** You should see a train accuracy of 0.833 (15 of 18 right), a test accuracy of only 0.5 (3 of 6) and a train MSE of 0.167. The weight alone is not enough: some heavy mangoes are still green, and on the six new mangoes the rule is no better than a coin flip. Keep that gap between train and test in mind.

# %% [markdown]
# ### ✏️ Exercise 1.2: Be the model: write your own rule
#
# Your task: write your own function `my_rule(x1, x2)` that beats the weight rule. Use the weight (`x1`), the colour (`x2`), or both, as a threshold or a line. Then run the check, which scores your rule on the training set. Write down in a markdown cell **what you changed** and why you thought it would help (for example "colour looks more important than weight, so I used colour"). Do not look at the test mangoes while you design the rule; that is the whole point of a test set.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
def my_rule(x1, x2):
    """x1 = weight in grams, x2 = colour score. Return 1 (ripe) or 0 (not ripe yet). Change the rule!"""
    return 1 if x2 > 5.5 else 0

my_train_acc = accuracy(my_rule, train)
print("my_rule  train accuracy:", round(my_train_acc, 3), "| train MSE:", round(rule_mse(my_rule, train), 3))

# %%
check("my_rule returns only 0 or 1", all(my_rule(w, c) in (0, 1) for w, c, _ in mangoes), "Return 1 or 0 only.")
check("my_rule beats the weight rule on the training set", my_train_acc > accuracy(weight_rule, train),
      "Try the colour score: ripe mangoes tend to have a high colour.")
check("my_rule gets at least 0.85 on the training set", my_train_acc >= 0.85, "Combine weight and colour, or move the threshold.")
RESULTS["part1_my_rule_train_acc"] = round(my_train_acc, 3)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def my_rule(x1, x2):
#     return 1 if x2 > 5.5 else 0     # colour matters more than weight
#
# # a rule that also uses the weight:  return 1 if x2 + (x1 - 250) / 100 > 5.5 else 0
# ```
#
# Colour separates the classes much better than weight, but a few mangoes (a green-looking ripe one, a coloured unripe one) are always hard.
#
# </details>

# %% [markdown]
# ### Step 1.3. A model with parameters
#
# A hand-written rule has numbers hidden inside it (270 g, 5.5). A **model** makes those numbers explicit so that a computer can search for them. Ours is `predict = 1 if w1·x1 + w2·x2 + b > 0`, with three **parameters**: two weights and a bias. First the features are rescaled so that both are of similar size (`(weight − 250) / 100` and `colour − 5`), otherwise the weight in grams would swamp the colour. The block tries five parameter settings by hand and prints the training accuracy of each. This is the picture of "training": a search for good values of (w1, w2, b).

# %%
def scale(rows):
    """Turn rows into a feature matrix X (rescaled) and a label vector y."""
    X = np.array([[(w - 250) / 100, c - 5.0] for w, c, _ in rows])
    y = np.array([r for _, _, r in rows])
    return X, y

X_train, y_train = scale(train)
X_test, y_test = scale(test)

def predict(X, w1, w2, b):
    """1 where w1*x1 + w2*x2 + b > 0, else 0."""
    return ((w1 * X[:, 0] + w2 * X[:, 1] + b) > 0).astype(int)

def acc(X, y, w1, w2, b):
    return float((predict(X, w1, w2, b) == y).mean())

settings = [(0, 0, 0), (1, 0, 0), (0, 1, 1.5), (0, 1, 0), (0.5, 1, -0.3)]
print(f"{'w1':>5} {'w2':>5} {'b':>5}   train accuracy")
for w1, w2, b in settings:
    print(f"{w1:>5} {w2:>5} {b:>5}   {acc(X_train, y_train, w1, w2, b):.3f}")

# %% [markdown]
# **What you should see:** You should see train accuracies of 0.500 for (0, 0, 0) (the sum is 0 for every mango, so everything is predicted 0, which is wrong for the 9 ripe mangoes), 0.833 for the weight-only setting (1, 0, 0), 0.778 for (0, 1, 1.5) (colour only, but the bias 1.5 pushes the threshold too low, so many unripe mangoes are called ripe), 0.944 for the colour-only setting (0, 1, 0) and 0.944 for (0.5, 1, −0.3). The bias matters: the same weights with a bad bias lose accuracy. Three numbers and a threshold already make a model.

# %% [markdown]
# ### ✏️ Exercise 1.3: A model with parameters
#
# Your task: find your own setting of (w1, w2, b) with a training accuracy of at least 0.85 by changing the dials by hand, and write down in a markdown cell what each dial did when you turned it (the weight `w1` makes the weight feature count more, the bias `b` moves the threshold). Why it matters: **training a neural network is this same search, only with thousands of dials and a smarter way to turn them**.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
w1, w2, b = 0.5, 1.0, -0.3     # change these three numbers and run again
my_acc = acc(X_train, y_train, w1, w2, b)
print("my setting", (w1, w2, b), "-> train accuracy", round(my_acc, 3))

# %%
check("Training accuracy is at least 0.85", my_acc >= 0.85, "Raise w2 (colour) or move b a little to the left.")
RESULTS["part1_my_params"] = (w1, w2, b)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# w1, w2, b = 0.5, 1.0, -0.3     # 16 of 18 right
# ```
#
# Many settings give the same accuracy: accuracy only changes when a point crosses the line, so a big region of dial settings is equally good.
#
# </details>

# %% [markdown]
# ### Step 1.4. Random search: the dumbest way to train
#
# If you cannot decide the dials by hand, let the computer try many. **Random search** draws 2000 random triples (w1, w2, b), scores each one on the training set, and keeps the best. It is very crude (it does not learn from earlier tries), but it shows the idea of training without any maths. The key honest step: the winner is chosen using the **training** set only, and only then do we look at the test set once. The second block draws the data and the line `w1·x1 + w2·x2 + b = 0` that the winner found.

# %%
def random_search(X, y, n_tries=2000, seed=SEED):
    """Try n_tries random (w1, w2, b) with values in -3..3; return the best one and its accuracy on (X, y)."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-3, 3, size=(n_tries, 3))                       # one row per try: w1, w2, b
    guesses = (X @ W[:, :2].T + W[:, 2]) > 0                         # shape (n_examples, n_tries)
    accs = (guesses == y[:, None]).mean(axis=0)                     # accuracy of every try
    best = int(np.argmax(accs))
    return W[best], float(accs[best])

(best_w1, best_w2, best_b), best_train = random_search(X_train, y_train, 2000)
best_test = acc(X_test, y_test, best_w1, best_w2, best_b)
print(f"best of 2000 tries: w1={best_w1:.2f}, w2={best_w2:.2f}, b={best_b:.2f}")
print(f"train accuracy {best_train:.3f} | test accuracy {best_test:.3f}")
RESULTS["part1_random_search"] = {"train": round(best_train, 3), "test": round(best_test, 3)}

# %% [markdown]
# **What you should see:** You should see `best of 2000 tries: w1=2.23, w2=0.81, b=-0.43` and `train accuracy 1.000 | test accuracy 0.667`. The search found a line that gets all 18 training mangoes right, but it is right on only 4 of the 6 test mangoes. The winner trusts the weight (w1 = 2.23) much more than the colour (w2 = 0.81): it has fitted the quirks of the 18 training mangoes. That gap is the first sight of **overfitting**, which returns in Part 6. With only 6 test mangoes the number is noisy; do not read too much into 0.667 versus 0.833.

# %% [markdown]
# ### ✏️ Exercise 1.4: Random search: the dumbest way to train
#
# Your task: run the search with fewer and with more tries, and **repeat each 20 times with different seeds** (the average is more trustworthy than one run). Store the average best-train accuracy for 10 tries in `avg10` and for 2000 tries in `avg2000`. Then write two sentences: does more search always help, and what is the risk of choosing the winner on the training set only? Why it matters: this is the smallest experiment that shows the shape of every later training curve: more search or more training gives better scores on the data you search on.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
avg10 = None      # average best-train accuracy of random_search with 10 tries (20 seeds)
avg2000 = None    # the same with 2000 tries

# %%
if avg10 is None or avg2000 is None:
    print("Fill in avg10 and avg2000 above, then run this cell again.")
else:
    check("avg2000 is between 0.5 and 1", 0.5 < avg2000 <= 1.0, "It is an accuracy, so a fraction.")
    check("More tries find a better setting on average", avg2000 > avg10, "Compute both averages with the same loop; 2000 tries should win.")
    assert avg2000 >= avg10, "random search with more tries should not be worse on average"

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# avg10 = float(np.mean([random_search(X_train, y_train, 10, seed=s)[1] for s in range(20)]))
# avg2000 = float(np.mean([random_search(X_train, y_train, 2000, seed=s)[1] for s in range(20)]))
# print(round(avg10, 3), round(avg2000, 3))
# ```
#
# With 10 tries you often miss the good region; with 2000 you almost always land in it. But "best on the training set" can be a little optimistic, which is why the test set exists.
#
# </details>

# %% [markdown]
# ### Step 1.5. Draw the line, and look at what the model gets wrong
#
# Draw the 18 training mangoes (circles: ripe, squares: not ripe), the 6 test mangoes (with a black edge) and the line the random search found. Then list the mangoes the model gets wrong. Looking at the mistakes is how a data scientist learns what is hard about a problem.

# %%
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for X, y, edge, name in [(X_train, y_train, "none", "train"), (X_test, y_test, "k", "test")]:
    ax.scatter(X[y == 0, 0], X[y == 0, 1], marker="s", c="tab:blue", edgecolor=edge, s=60, label=f"{name}: not ripe")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], marker="o", c="tab:orange", edgecolor=edge, s=60, label=f"{name}: ripe")
xs = np.linspace(-1.5, 1.5, 50)
ax.plot(xs, -(best_w1 * xs + best_b) / best_w2, "k-", lw=1.5, label="learned line")   # where w1*x1 + w2*x2 + b = 0
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-5, 5)
ax.set_xlabel("weight (rescaled)"); ax.set_ylabel("colour (rescaled)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

print("Mistakes of the learned line (weight_g, colour, true label):")
for (w, c, r), pred in zip(mangoes, predict(np.vstack([X_train, X_test]), best_w1, best_w2, best_b)):
    if pred != r:
        print("  ", (w, c, r), "predicted", pred)

# %% [markdown]
# **What you should see:** You should see the plot, with the line tilted so that heavy mangoes lie on the ripe side, and two mistakes, both of them test mangoes: `(255, 5.8, 0) predicted 1` (a medium-weight mango with a fairly high colour score that is still not ripe) and `(225, 5.2, 1) predicted 0` (a light mango that is ripe even though its colour is only medium). Both are borderline cases that a person might also label differently, which leads to the next exercise.

# %% [markdown]
# ### ✏️ Exercise 1.5: Draw the line, and look at what the model gets wrong
#
# Your task: in a markdown cell write four or five sentences: (a) what makes a **label** good or bad (who decided that a mango is "ripe", and by what test: the farmer's thumb, a lab sugar meter, a taste test?); (b) one way labels could be wrong or **biased** in the real world (for example a spam filter trained on labels from a single person, or a hospital model whose labels come only from patients who could afford a test); (c) what the model would learn from wrong labels. Save your answer in `label_note` and check that it has at least 25 words. Why it matters: a supervised model can only be as good as its labels.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
label_note = """
(write your sentences here)
"""

# %%
n_words = len(label_note.split())
check("label_note has at least 25 words", n_words >= 25, f"You wrote {n_words} words. Explain (a), (b) and (c).")
RESULTS["part1_label_note"] = label_note.strip()

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# label_note = """A good label comes from a clear, repeatable test, such as a sugar meter, while "ripe" by thumb pressure depends on the person.
# Labels can be biased when only one kind of example gets labelled, for example a spam filter trained on one person's inbox.
# The model copies whatever the labels say, including their mistakes, because it has nothing else to learn from."""
# ```
#
# There is no single correct text; the answer must mention how labels are decided and what happens if they are wrong.
#
# </details>

# %% [markdown]
# ### ✏️ Imagine first, revisited
#
# Run the next cell to compare your prediction with the facts.

# %%
# Imagine first, revisited
RESULTS["part1_test_acc"] = round(best_test, 3)
if my_prediction is None:
    print("You did not write a prediction at the top of Part 1; the real test accuracy was", round(best_test, 3))
else:
    print(f"You predicted {my_prediction}. The random-search model scored {best_test:.3f} on the test mangoes.")
assert 0.0 <= best_test <= 1.0

# %% [markdown]
# **What you should see:** You should see your prediction next to `0.667`, the test accuracy of the random-search model (4 of the 6 test mangoes right). If you guessed 0.9 or more, you met your first lesson of the course: scores on the training set flatter a model, and a small test set is a noisy judge. Write one sentence about it.

# %% [markdown]
# ## Part 2. Loss: measure the mistake (20 min)
#
# *Lesson section: "Loss: how wrong is the model?".*
#
# To improve a model you first need a number that says how bad its mistakes are. That number is the **loss**: small is good, zero is perfect. In this part you write the two losses of the lesson yourself, mean squared error for numbers and cross-entropy (`−ln p`) for categories, check them against PyTorch, and then look at the loss as a landscape: a curve with a lowest point that training will try to reach. Finally you see how the choice of loss changes which model counts as best.

# %% [markdown]
# ### ✏️ Imagine first
#
# Two models predict a number for 10 examples. Model A is wrong by 0.5 on every one of the 10 examples. Model B is exactly right on 9 examples and wrong by 3 on the last one. Which model has the larger mean squared error? Write `"A"` or `"B"`, and keep your reason in one sentence. You will check it in step 2.1.

# %%
# Imagine first: which model has the larger mean squared error, "A" (0.5 off on all 10) or "B" (3 off on just one)?
my_pick = None   # replace None with "A" or "B"

# %% [markdown]
# ### Step 2.1. Write mse and bce yourself
#
# **Mean squared error** (MSE) is the average of the squared differences between the true values and the predictions: `mean((y − ŷ)²)`. **Binary cross-entropy** (BCE) is for yes/no labels: the model gives the probability `p` that the label is 1, and the loss for one example is `−ln` of the probability the model gave to the **true** class (`−ln p` when the label is 1, `−ln(1 − p)` when it is 0), averaged over the examples. The block implements both in NumPy, tries them on a small worked example, and compares them with PyTorch's `F.mse_loss` and `F.binary_cross_entropy`. The `assert` lines stop with an error if your numbers and PyTorch's disagree.

# %%
import torch.nn.functional as F

def mse(y_true, y_pred):
    """Mean squared error: the average of (y - y_hat)^2."""
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def bce(y_true, p, eps=1e-12):
    """Binary cross-entropy: the average of -ln(probability given to the TRUE class)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)      # keep p away from exactly 0 or 1
    p_true = np.where(y_true == 1, p, 1 - p)                   # the probability the model gave to the right answer
    return float(np.mean(-np.log(p_true)))

# Worked example 1: predicting numbers (say, rice meals sold)
y = [3, 5, 2, 7]
y_hat = [2.5, 5.5, 2, 9]
print("MSE:", mse(y, y_hat), "(squared errors 0.25, 0.25, 0, 4 -> mean 1.125)")

# Worked example 2: predicting yes/no (is the mango ripe?)
labels = [1, 0, 1, 1]
probs = [0.9, 0.2, 0.6, 0.99]
print("per-example -ln p:", [round(float(-np.log(p if l == 1 else 1 - p)), 4) for l, p in zip(labels, probs)])
print("BCE:", round(bce(labels, probs), 4))

# Compare with PyTorch: the two must agree
t_mse = F.mse_loss(torch.tensor(y_hat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).item()
t_bce = F.binary_cross_entropy(torch.tensor(probs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)).item()
print("PyTorch MSE:", round(t_mse, 4), "| PyTorch BCE:", round(t_bce, 4))
assert abs(mse(y, y_hat) - t_mse) < 1e-5 and abs(bce(labels, probs) - t_bce) < 1e-5

# Imagine first, revisited: A is 0.5 off on all 10, B is 3 off on one of 10
mse_A = mse(np.zeros(10), np.full(10, 0.5))
mse_B = mse(np.zeros(10), np.array([0] * 9 + [3]))
print("Model A MSE:", mse_A, "| Model B MSE:", mse_B)
RESULTS["part2_mse_A_B"] = (mse_A, mse_B)
print("Your pick was", my_pick, "- the larger MSE belongs to", "A" if mse_A > mse_B else "B")

# %% [markdown]
# **What you should see:** You should see `MSE: 1.125`, per-example `-ln p` values of `[0.1054, 0.2231, 0.5108, 0.0101]` and `BCE: 0.2123`, and PyTorch printing `1.125` and `0.2123` too, so the assert passes. Then `Model A MSE: 0.25 | Model B MSE: 0.9`: B has the larger MSE, because squaring makes one big miss (3² = 9) cost far more than ten small ones (ten times 0.5² = 0.25 in total, divided by 10). If you picked A you were in good company: ten mistakes feels worse than one, but the loss does not see it that way.

# %% [markdown]
# ### ✏️ Exercise 2.1: Write mse and bce yourself
#
# Your task: compute the BCE of two models on four mangoes with the true labels `[1, 1, 0, 0]`. The **hedging** model always says `p = 0.5`. The **bold** model says `p = 0.9` for every mango. Store the two losses in `loss_hedge` and `loss_bold`. Then write one sentence on why the bold model, which is right about two mangoes and wrong about two, has the larger loss. Why it matters: this is what "confidently wrong is punished hard" looks like in numbers, and it is why classifiers are trained with cross-entropy.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
true_labels = [1, 1, 0, 0]
loss_hedge = None   # BCE when the model says p = 0.5 for all four
loss_bold = None    # BCE when the model says p = 0.9 for all four

# %%
if loss_hedge is None or loss_bold is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("Hedging model loss is about 0.693 (that is -ln 0.5)", abs(loss_hedge - 0.6931) < 0.01, "Use bce(true_labels, [0.5] * 4).")
    check("Bold model loss is about 1.204", abs(loss_bold - 1.2040) < 0.01, "Two mangoes cost -ln 0.9 = 0.105, two cost -ln 0.1 = 2.303.")
    assert loss_bold > loss_hedge
    RESULTS["part2_bce_hedge_bold"] = (round(loss_hedge, 3), round(loss_bold, 3))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# true_labels = [1, 1, 0, 0]
# loss_hedge = bce(true_labels, [0.5] * 4)    # 0.693
# loss_bold = bce(true_labels, [0.9] * 4)     # (0.105 + 0.105 + 2.303 + 2.303) / 4 = 1.204
# ```
#
# Being confidently wrong twice costs more than hedging four times, even though the bold model is right half of the time.
#
# </details>

# %% [markdown]
# ### Step 2.2. The −ln p table and curve
#
# Why does cross-entropy punish confident mistakes so hard? Look at the function `−ln p`, where `p` is the probability the model gave to the **true** class. The block prints its value for five probabilities and plots the curve from 0.01 to 1. A model that is right and sure (p near 1) pays almost nothing; a model that gave the truth a tiny probability pays a lot.

# %%
print(f"{'p (given to the truth)':>24}   -ln p")
for p in (0.99, 0.9, 0.5, 0.1, 0.01):
    print(f"{p:>24}   {-math.log(p):.3f}")

ps = np.linspace(0.01, 1.0, 200)
plt.figure(figsize=(6, 3.5))
plt.plot(ps, -np.log(ps), c="tab:red")
for p in (0.9, 0.5, 0.1, 0.01):
    plt.scatter([p], [-math.log(p)], c="k", zorder=3)
    plt.annotate(f"p={p}: {-math.log(p):.2f}", (p, -math.log(p)), textcoords="offset points", xytext=(8, 4), fontsize=8)
plt.xlabel("p, the probability the model gave to the true class"); plt.ylabel("loss = -ln p")
plt.title("The cross-entropy loss of one example"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see the table `0.99 -> 0.010`, `0.9 -> 0.105`, `0.5 -> 0.693`, `0.1 -> 2.303`, `0.01 -> 4.605`, and a curve that is nearly flat near p = 1 and shoots up as p moves toward 0. Cutting p from 0.5 to 0.1 more than triples the loss (0.693 to 2.303), and from 0.1 to 0.01 it doubles again. As p → 0 the loss grows without limit.

# %% [markdown]
# ### ✏️ Exercise 2.2: The −ln p table and curve
#
# Your task: in a markdown cell write two or three sentences: what happens to `−ln p` as p goes to 0, and why does this punish a model that is **confident and wrong** much more than one that is unsure? (Link it to Week 1: GPT-2 was trained with exactly this loss on the next word.) Then compute the probability that would give a loss of exactly 5 and store it in `p_for_5`.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
p_for_5 = None   # the probability p for which -ln p = 5 (hint: undo the logarithm with math.exp)

# %%
if p_for_5 is None:
    print("Fill in p_for_5 above, then run this cell again.")
else:
    check("p_for_5 is about 0.0067", abs(p_for_5 - math.exp(-5)) < 1e-3, "-ln p = 5 means p = e^(-5).")
    check("-ln(p_for_5) is 5", abs(-math.log(p_for_5) - 5) < 0.2, "Take -math.log(p_for_5) to test your answer.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# p_for_5 = math.exp(-5)    # 0.0067: the model gave the truth less than 1 chance in 100
# ```
#
# A loss of 5 means the model believed the truth had under 1 chance in 100.
#
# </details>

# %% [markdown]
# ### Step 2.3. The loss landscape of a one-weight model
#
# Take the simplest model there is: `ŷ = w·x`, one dial `w`. The five points below follow the line `y = 2x` with a little noise (so the true slope is about 2). For 81 values of `w` from −2 to 6 the block computes the MSE and plots the result against `w`. The curve is a parabola, a bowl: this is the picture of "downhill" in the lesson. The best `w` is the bottom of the bowl, found here with `argmin`. Training a network is walking down such a bowl, only with thousands of dials at once.

# %%
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])             # about 2 * x, with a little noise

ws = np.linspace(-2, 6, 81)                          # 81 values of the dial, spacing 0.1
losses = np.array([mse(y, w * x) for w in ws])       # the loss for each value

i_best = int(np.argmin(losses))
print(f"lowest loss on the grid: w = {ws[i_best]:.1f}, MSE = {losses[i_best]:.3f}")
print(f"loss at w = -2: {losses[0]:.1f} | at w = 6: {losses[-1]:.1f}")

plt.figure(figsize=(6, 3.5))
plt.plot(ws, losses, c="tab:blue")
plt.scatter([ws[i_best]], [losses[i_best]], c="tab:red", zorder=3, label=f"minimum at w = {ws[i_best]:.1f}")
plt.xlabel("w (the dial)"); plt.ylabel("MSE"); plt.title("The loss landscape of y = w * x")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
RESULTS["part2_best_w"] = round(float(ws[i_best]), 2)

# %% [markdown]
# **What you should see:** You should see `lowest loss on the grid: w = 2.0, MSE = 0.022`, the loss at both ends of the grid huge and almost equal (`176.3` at w = −2 and `175.7` at w = 6, because both are 4 away from the minimum), and a smooth, symmetric bowl with its bottom at w = 2. Anywhere you stand on the bowl, the direction of the slope tells you which way is downhill.

# %% [markdown]
# ### ✏️ Exercise 2.3: The loss landscape of a one-weight model
#
# Your task: compute the MSE for a model that always predicts 0 (`w = 0`) and for the model `w = 4`, store them in `loss_w0` and `loss_w4`, and say in one sentence why the two numbers are almost the same (both dials are 2 away from the best value, on opposite sides of the bowl). Why it matters: the loss at a particular value of `w` is a height on the landscape; the gradient (Part 5) is the slope there.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
loss_w0 = None   # MSE of the model y_hat = 0 * x
loss_w4 = None   # MSE of the model y_hat = 4 * x

# %%
if loss_w0 is None or loss_w4 is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("loss at w = 0", abs(loss_w0 - mse(y, 0 * x)) < 1e-6, "Use mse(y, 0 * x).")
    check("loss at w = 4", abs(loss_w4 - mse(y, 4 * x)) < 1e-6, "Use mse(y, 4 * x).")
    check("w = 0 and w = 4 have almost the same loss", abs(loss_w0 - loss_w4) < 1.0, "Both are 2 away from the minimum at w = 2.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# loss_w0 = mse(y, 0 * x)    # 44.18
# loss_w4 = mse(y, 4 * x)    # 43.86
# ```
#
# A bowl of this kind is a parabola, and a parabola is symmetric around its lowest point: equal distances to the left and to the right give nearly equal heights.
#
# </details>

# %% [markdown]
# ### Step 2.4. Two losses, two behaviours: an outlier
#
# An **outlier** is an example that does not fit the pattern (a typing mistake, a broken sensor). Add one to the data (x = 6 with y = 30, where the pattern says 12) and ask which `w` is best for two different losses: MSE (squares the error) and **mean absolute error** (MAE, `mean(|y − ŷ|)`). A finer grid of 801 values of `w` is used so that the answers are accurate to 0.01.

# %%
def mae(y_true, y_pred):
    """Mean absolute error: the average of |y - y_hat|."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

def best_w(loss, x, y, grid=np.linspace(-2, 6, 801)):
    """The value of w on the grid with the lowest loss."""
    return float(grid[int(np.argmin([loss(y, w * x) for w in grid]))])

x_out = np.append(x, 6.0)
y_out = np.append(y, 30.0)                           # the outlier: the pattern says about 12

w_mse_clean, w_mae_clean = best_w(mse, x, y), best_w(mae, x, y)
w_mse_out, w_mae_out = best_w(mse, x_out, y_out), best_w(mae, x_out, y_out)
print(f"{'':<12}{'MSE':>8}{'MAE':>8}")
print(f"{'clean data':<12}{w_mse_clean:>8.2f}{w_mae_clean:>8.2f}")
print(f"{'with outlier':<12}{w_mse_out:>8.2f}{w_mae_out:>8.2f}")
RESULTS["part2_outlier_w"] = {"mse": round(w_mse_out, 2), "mae": round(w_mae_out, 2)}

# %% [markdown]
# **What you should see:** You should see, for the best `w`: clean data 2.00 (MSE) and 2.02 (MAE); with the outlier 3.19 (MSE) but still 2.02 (MAE). One bad point drags the MSE answer from 2.0 to 3.2, a push of about 60 %, while the MAE answer does not move at all (MAE behaves like a median, MSE like a mean). Squaring makes a big error count far more than a small one, so the MSE loss "listens" to the outlier. That is sometimes what you want (large mistakes really are costly) and sometimes not (the point is a typing error).

# %% [markdown]
# ### ✏️ Exercise 2.4: Two losses, two behaviours: an outlier
#
# Your task: choose your own outlier height `outlier_y` for x = 6 (anything far from 12), and compute how far each best `w` moves from its clean value: `shift_mse` and `shift_mae`. Then write one sentence in a markdown cell on why squaring matters, and one on a situation where you would prefer MAE.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
outlier_y = 30         # the y value of the extra point at x = 6 (the pattern says about 12)
shift_mse = None       # |best w for MSE with the outlier  -  best w for MSE on the clean data|
shift_mae = None       # the same for MAE

# %%
if shift_mse is None or shift_mae is None:
    print("Fill in shift_mse and shift_mae above, then run this cell again.")
else:
    xo, yo = np.append(x, 6.0), np.append(y, float(outlier_y))
    want_mse = abs(best_w(mse, xo, yo) - w_mse_clean)
    want_mae = abs(best_w(mae, xo, yo) - w_mae_clean)
    check("Your outlier is far from the pattern", abs(outlier_y - 12) > 5, "Pick a y value far from 12, for example 30 or -10.")
    check("shift_mse is right", abs(shift_mse - want_mse) < 0.02, "Use best_w(mse, ...) with and without the extra point.")
    check("shift_mae is right", abs(shift_mae - want_mae) < 0.02, "Use best_w(mae, ...) with and without the extra point.")
    assert want_mse > want_mae, "MSE should be moved more by an outlier than MAE"

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# xo, yo = np.append(x, 6.0), np.append(y, float(outlier_y))
# shift_mse = abs(best_w(mse, xo, yo) - w_mse_clean)     # 1.19 for outlier_y = 30
# shift_mae = abs(best_w(mae, xo, yo) - w_mae_clean)     # 0.00 for outlier_y = 30
# ```
#
# Squaring turns one big error into a huge penalty, so the best w moves to reduce that single error; the absolute error grows only in proportion, so the other points still win the vote.
#
# </details>

# %% [markdown]
# ### ✏️ Imagine first, revisited
#
# Run the next cell to compare your prediction with the facts.

# %%
print("Part 2 summary:", {k: v for k, v in RESULTS.items() if k.startswith("part2")})

# %% [markdown]
# **What you should see:** You should see one line with your Part 2 numbers: the two MSEs of the imagine question, the BCE of the hedging and bold models, the best `w` and the outlier results.

# %% [markdown]
# ## Part 3. One neuron, built by hand (25 min)
#
# *Lesson section: "One artificial neuron".*
#
# An artificial neuron is the smallest working piece of a neural network: it multiplies each input by a weight, adds a bias, and passes the sum through an activation function. In this part you code that recipe in a few lines, move its dials and watch the straight line it draws, set the weights for the logic gates AND and OR by hand, and then discover what one neuron cannot do: XOR. At the end you train a single neuron with PyTorch, so the same pieces appear once by hand and once by library.

# %% [markdown]
# ### ✏️ Imagine first
#
# Picture one neuron looking at two numbers (x1, x2) and answering yes or no. In words, what shape of boundary can it draw between the "yes" region and the "no" region of the plane: a straight line, a curve, a circle? Then predict whether one neuron can learn each of three logic gates (the inputs are 0 or 1): **AND** (yes only if both inputs are 1), **OR** (yes if at least one input is 1) and **XOR** (yes if exactly one input is 1). For each gate write `"yes"` or `"no"`.

# %%
# Imagine first: can ONE neuron learn each gate? Write "yes" or "no" for each.
my_prediction = {"AND": None, "OR": None, "XOR": None}
boundary_shape = ""   # in a few words: what shape of boundary can one neuron draw?

# %% [markdown]
# ### Step 3.1. Code the neuron and its four activations
#
# A neuron computes `z = w1·x1 + w2·x2 + b` and then `a = φ(z)`, where φ is the **activation function**. Four common activations are coded below: **step** (1 if z > 0, else 0), **sigmoid** `1 / (1 + e^(−z))` (a smooth step between 0 and 1), **tanh** (a smooth step between −1 and 1) and **ReLU** `max(0, z)`. The block reproduces the worked example of the lesson section "One artificial neuron": inputs `x = (2, 1)`, weights `w = (0.5, −1)`, bias `b = 0.25`. Then it plots the four activations on −5..5 and marks the value of `z` for this example.

# %%
def step(z):    return np.where(np.asarray(z, dtype=float) > 0, 1.0, 0.0)
def sigmoid(z): return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))
def tanh(z):    return np.tanh(np.asarray(z, dtype=float))
def relu(z):    return np.maximum(0.0, np.asarray(z, dtype=float))

ACTIVATIONS = {"step": step, "sigmoid": sigmoid, "tanh": tanh, "relu": relu}

def neuron(x, w, b, act="sigmoid"):
    """One neuron: z = w . x + b, then a = act(z). Returns (z, a)."""
    z = float(np.dot(w, x) + b)
    return z, float(ACTIVATIONS[act](z))

x, w, b = (2, 1), (0.5, -1), 0.25
for name in ACTIVATIONS:
    z, a = neuron(x, w, b, name)
    print(f"{name:<8} z = {z:.2f}   a = {a:.3f}")

zs = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharex=True)
for ax, (name, f) in zip(axes, ACTIVATIONS.items()):
    ax.plot(zs, f(zs), c="tab:blue")
    z0 = neuron(x, w, b, name)[0]
    ax.scatter([z0], [f(z0)], c="tab:red", zorder=3)                 # the worked example
    ax.set_title(name); ax.grid(alpha=0.3); ax.set_xlabel("z")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see `z = 0.25` for all four lines (2·0.5 + 1·(−1) + 0.25 = 0.25) and the activations `step 1.000`, `sigmoid 0.562`, `tanh 0.245`, `relu 0.250`. The sigmoid value 0.562 is the one in the lesson. In the plot the red dot sits just right of zero: for step it is already at 1, for sigmoid just above the middle (0.5), for ReLU it equals z.

# %% [markdown]
# ### ✏️ Exercise 3.1: Code the neuron and its four activations
#
# Your task: compute a second neuron by hand first, then with the code: inputs `x = (1, 3)`, weights `w = (0.4, −0.2)`, bias `b = 0.1`. Write the arithmetic of `z` in a markdown cell, then store `z_mine`, the ReLU output `a_relu` and the sigmoid output `a_sigmoid` (rounded to 3 decimals). Why it matters: every network in this course, up to the biggest models, is built from this one calculation repeated many times.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
z_mine = None         # z = w1*x1 + w2*x2 + b for x = (1, 3), w = (0.4, -0.2), b = 0.1
a_relu = None         # relu(z)
a_sigmoid = None      # sigmoid(z), 3 decimals

# %%
if z_mine is None or a_relu is None or a_sigmoid is None:
    print("Fill in your answers above, then run this cell again.")
else:
    z_true, a_true = neuron((1, 3), (0.4, -0.2), 0.1, "sigmoid")
    check("z is -0.1", abs(z_mine - z_true) < 1e-6, "0.4*1 + (-0.2)*3 + 0.1 = 0.4 - 0.6 + 0.1")
    check("ReLU of a negative z is 0", abs(a_relu - 0.0) < 1e-6, "relu(z) = max(0, z)")
    check("sigmoid(-0.1) is about 0.475", abs(a_sigmoid - a_true) < 1e-3, "1 / (1 + e^0.1)")
    assert abs(neuron((2, 1), (0.5, -1), 0.25, "sigmoid")[1] - 0.562) < 1e-3     # the lesson's example

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# z_mine, a_sigmoid = neuron((1, 3), (0.4, -0.2), 0.1, "sigmoid")
# a_relu = neuron((1, 3), (0.4, -0.2), 0.1, "relu")[1]
# a_sigmoid = round(a_sigmoid, 3)      # z = -0.1, relu = 0.0, sigmoid = 0.475
# ```
#
# A negative z is switched off by ReLU (output 0) but still passes a small positive value through the sigmoid, because the sigmoid never reaches exactly 0.
#
# </details>

# %% [markdown]
# ### Step 3.2. Weights and bias as dials
#
# A neuron says "yes" where `w1·x1 + w2·x2 + b > 0`. The edge between yes and no is the line `w1·x1 + w2·x2 + b = 0`. The block draws that line for different dial settings. On the left the weights stay at (1, 1) and the **bias** moves through −2..2; on the right the bias stays 0 and the **weights** change. It also prints the distance from the origin to the line, `|b| / ‖w‖`.

# %%
xs = np.linspace(-3, 3, 100)

def line_y(w, b, xs):
    """Solve w1*x + w2*y + b = 0 for y."""
    return -(w[0] * xs + b) / w[1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for b in (-2, -1, 0, 1, 2):
    axes[0].plot(xs, line_y((1, 1), b, xs), label=f"b = {b}")
axes[0].set_title("weights (1, 1): the bias slides the line")
for w in [(1, 1), (1, 0.3), (0.3, 1), (1, -1)]:
    axes[1].plot(xs, line_y(w, 0, xs), label=f"w = {w}")
axes[1].set_title("bias 0: the weights tilt the line")
for ax in axes:
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
    ax.legend(fontsize=7); ax.set_aspect("equal")
plt.tight_layout(); plt.show()

for b in (-2, 0, 2):
    print(f"w = (1, 1), b = {b:>2}: distance from the origin to the line = {abs(b) / np.linalg.norm((1, 1)):.2f}")
print("w = (2, 2), b = 2: distance =", round(abs(2) / np.linalg.norm((2, 2)), 2), "(doubling all dials keeps the line closer)")

# %% [markdown]
# **What you should see:** You should see five parallel lines on the left: they all have slope −1, and the bias only moves the line (b = 0 passes through the origin, b = 2 lies below it). On the right the line turns when the weights change: (1, 0.3) is steep, (0.3, 1) is flat, (1, −1) tilts the other way. The printed distances are `1.41`, `0.00` and `1.41`, and `0.71` for (2, 2) with b = 2: doubling both weights but keeping the bias halves the distance. In short: **the weights set the direction of the line, the bias sets its position**.

# %% [markdown]
# ### ✏️ Exercise 3.2: Weights and bias as dials
#
# Your task: choose weights and a bias so that the neuron answers 1 **only for points above the line y = x** (where x2 > x1). Think first: what must `w1` and `w2` look like if the line `w1·x1 + w2·x2 + b = 0` must be `x2 = x1`, and which side is "yes"? Store them in `w_above` (a pair) and `b_above`. The check tests six points.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
w_above = None    # a pair (w1, w2)
b_above = None    # the bias

# %%
if w_above is None or b_above is None:
    print("Fill in w_above and b_above above, then run this cell again.")
else:
    above = [(0, 1), (-1, 2), (1, 3)]                 # x2 > x1: the neuron must say 1
    below = [(1, 0), (2, 1), (0, -1)]                 # x2 < x1: the neuron must say 0
    says = lambda p: neuron(p, w_above, b_above, "step")[1]
    check("fires for the points above y = x", all(says(p) == 1 for p in above), "z must be positive when x2 > x1. Try w = (-1, 1).")
    check("stays silent for the points below y = x", all(says(p) == 0 for p in below), "z must be 0 or negative when x2 < x1.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# w_above = (-1, 1)     # z = -x1 + x2, which is positive exactly when x2 > x1
# b_above = 0
# ```
#
# The line goes through the origin, so b = 0; the weights point at the "yes" side. (−2, 2) and bias 0 works too: only the direction of w matters here.
#
# </details>

# %% [markdown]
# ### Step 3.3. One neuron learns AND and OR
#
# Logic gates take two inputs that are 0 or 1. **AND** answers 1 only when both inputs are 1; **OR** answers 1 when at least one input is 1. Both can be written as one neuron with a step activation. Choose the weights (1, 1) for both gates; AND needs a threshold between 1 and 2 (bias −1.5), OR a threshold between 0 and 1 (bias −0.5). The block prints the truth tables and checks them.

# %%
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

def gate_outputs(w, b):
    """The step-neuron's answer for the four input pairs."""
    return [int(neuron(p, w, b, "step")[1]) for p in inputs]

and_w, and_b = (1, 1), -1.5
or_w, or_b = (1, 1), -0.5

print(f"{'x1':>3} {'x2':>3} | {'AND':>4} {'OR':>4}")
for p, a_out, o_out in zip(inputs, gate_outputs(and_w, and_b), gate_outputs(or_w, or_b)):
    print(f"{p[0]:>3} {p[1]:>3} | {a_out:>4} {o_out:>4}")

check("AND truth table", gate_outputs(and_w, and_b) == [0, 0, 0, 1], "AND is 1 only for (1, 1).")
check("OR truth table", gate_outputs(or_w, or_b) == [0, 1, 1, 1], "OR is 0 only for (0, 0).")
assert gate_outputs(and_w, and_b) == [0, 0, 0, 1] and gate_outputs(or_w, or_b) == [0, 1, 1, 1]
RESULTS["part3_and_or_weights"] = {"AND": (and_w, and_b), "OR": (or_w, or_b)}

# %% [markdown]
# **What you should see:** You should see the table with AND `0 0 0 1`, OR `0 1 1 1`, and two ticks. For AND the sum x1 + x2 is 0, 1 or 2 and the bias −1.5 makes z positive only for 2; for OR the bias −0.5 makes z positive as soon as the sum reaches 1. Same weights, different bias: **the bias moves the threshold**.

# %% [markdown]
# ### ✏️ Exercise 3.3: One neuron learns AND and OR
#
# Your task: find the weights and bias of a **NAND** gate (the opposite of AND: 0 only for input (1, 1)) with a single step neuron. Hint: flip the signs of the AND neuron. Store them in `nand_w` (a pair) and `nand_b`, and in a markdown cell explain how you flipped the signs.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
nand_w = None     # a pair (w1, w2)
nand_b = None     # the bias

# %%
if nand_w is None or nand_b is None:
    print("Fill in nand_w and nand_b above, then run this cell again.")
else:
    out = gate_outputs(nand_w, nand_b)
    check("NAND truth table is 1 1 1 0", out == [1, 1, 1, 0], f"Your neuron gives {out}. Try negative weights and a positive bias.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# nand_w = (-1, -1)
# nand_b = 1.5       # z = 1.5 - x1 - x2: positive for sums 0 and 1, negative for sum 2
# ```
#
# Negating all the weights and the bias flips every answer, and NAND is the exact opposite of AND, so (−1, −1, +1.5) works.
#
# </details>

# %% [markdown]
# ### Step 3.4. One neuron cannot learn XOR
#
# **XOR** answers 1 when exactly one input is 1. We cannot set its weights by hand, so let the computer try. The block gives a single step neuron 50,000 random sets of dials (w1, w2, b), each from −5 to 5, scores each set on the four-row truth table, and reports the best accuracy for AND, OR and XOR. The second plot draws the four XOR points: no straight line can put the two orange circles on one side and the two blue squares on the other.

# %%
def search(X, y, n_tries=50000, seed=SEED, scale=5.0):
    """Try n_tries random (w1, w2, b); return all accuracies and the winning dials."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(-scale, scale, size=(n_tries, 3))
    guesses = (X @ W[:, :2].T + W[:, 2]) > 0
    accs = (guesses == y[:, None]).mean(axis=0)
    return accs, W[int(np.argmax(accs))]

gates_X = np.array(inputs, dtype=float)
gate_y = {"AND": np.array([0, 0, 0, 1]), "OR": np.array([0, 1, 1, 1]), "XOR": np.array([0, 1, 1, 0])}

best_acc = {}
all_accs = {}
for name, y_gate in gate_y.items():
    all_accs[name], dials = search(gates_X, y_gate)
    best_acc[name] = float(all_accs[name].max())
    print(f"{name:<4} best accuracy over 50,000 random neurons: {best_acc[name]:.2f}")
RESULTS["part3_xor_best_acc"] = best_acc["XOR"]

# the four XOR points
yx = gate_y["XOR"]
plt.figure(figsize=(4, 4))
plt.scatter(gates_X[yx == 0, 0], gates_X[yx == 0, 1], marker="s", s=200, c="tab:blue", edgecolor="k", label="0")
plt.scatter(gates_X[yx == 1, 0], gates_X[yx == 1, 1], marker="o", s=200, c="tab:orange", edgecolor="k", label="1")
plt.plot([-0.3, 1.3], [0.8, -0.8], "k--", lw=1, label="x1 + x2 = 0.5: 3 of 4 right")      # the best a line can do
plt.xlim(-0.3, 1.3); plt.ylim(-0.3, 1.3); plt.gca().set_aspect("equal")
plt.title("XOR: no straight line works"); plt.legend(fontsize=7); plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see `AND best accuracy ...: 1.00`, `OR ...: 1.00` and `XOR ...: 0.75`. Out of 50,000 random neurons, XOR never reaches more than 3 of 4 rows right (0.75). It is not bad luck, it is geometry: the two 1s sit on one diagonal and the two 0s on the other, and a straight line cannot separate diagonals. This is the limit of a single neuron, and the reason for hidden layers in Part 4.

# %% [markdown]
# ### ✏️ Exercise 3.4: One neuron cannot learn XOR
#
# Your task: count how many of the 50,000 random neurons got a **perfect** score on XOR and on OR, store the counts in `n_perfect_xor` and `n_perfect_or`, and compare with the predictions you made at the top of Part 3. Write two sentences: which gates could one neuron learn, and why can the same neuron solve OR but not XOR? Why it matters: the number 0 for XOR is not a failure of the search, it is a proof by experiment that the model is too simple.
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
n_perfect_xor = None   # how many of the 50,000 tries scored 1.0 on XOR?
n_perfect_or = None    # ... on OR?

# %%
if n_perfect_xor is None or n_perfect_or is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("No random neuron solves XOR", n_perfect_xor == 0, "Count the entries of all_accs['XOR'] that equal 1.0.")
    check("Many random neurons solve OR", n_perfect_or > 1000, "Count the entries of all_accs['OR'] that equal 1.0.")
    assert best_acc["XOR"] <= 0.75 + 1e-9

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# n_perfect_xor = int((all_accs["XOR"] == 1.0).sum())    # 0
# n_perfect_or = int((all_accs["OR"] == 1.0).sum())      # 2121 with SEED = 42
# ```
#
# OR is separable by a line, XOR is not: that is exactly the difference between "yes" and "no" in your Imagine-first predictions.
#
# </details>

# %% [markdown]
# ### Step 3.5. Train a single neuron with PyTorch
#
# Now let a neuron find its own weights by training, using the standard PyTorch loop (it is explained line by line in Part 6, so only skim it for now). The neuron is `nn.Linear(2, 1)` followed by a sigmoid, the loss is binary cross-entropy (`nn.BCELoss`) and the optimiser is SGD, which nudges the dials downhill. The data are the two interleaved half-moons from `make_moons`. A single neuron draws one straight line, so it can only do so well on curved data. A second set of moons with a different seed is the test set.

# %%
def train_neuron(X, y, epochs=300, lr=0.5, seed=SEED):
    """Train one sigmoid neuron on (X, y). Returns the model, the loss at every epoch and the train accuracy."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())          # z = w.x + b, then sigmoid
    Xt, yt = torch.tensor(X), torch.tensor(y).unsqueeze(1)
    loss_fn = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()                                            # forget the last step's gradients
        loss = loss_fn(model(Xt), yt)                              # measure the mistake
        loss.backward()                                            # work out which way is downhill
        opt.step()                                                 # take a small step that way
        losses.append(loss.item())
    with torch.no_grad():
        acc = ((model(Xt) > 0.5).float() == yt).float().mean().item()
    return model, losses, acc

def as_predictor(model):
    """Wrap a model so that plot_decision can call it with a NumPy array."""
    return lambda P: model(torch.tensor(P)).detach().numpy().ravel()

X_moons, y_moons = make_moons(200, noise=0.15, seed=0)
X_moons_test, y_moons_test = make_moons(200, noise=0.15, seed=1)
neuron_model, neuron_losses, acc_train = train_neuron(X_moons, y_moons)
acc_test = float(((as_predictor(neuron_model)(X_moons_test) > 0.5) == y_moons_test).mean())
print(f"loss: {neuron_losses[0]:.3f} -> {neuron_losses[-1]:.3f} | train accuracy {acc_train:.3f} | test accuracy {acc_test:.3f}")
assert neuron_losses[-1] < neuron_losses[0], "training should lower the loss"
RESULTS["part3_moons_neuron"] = {"train": round(acc_train, 3), "test": round(acc_test, 3)}

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(neuron_losses); axes[0].set_title("loss during training"); axes[0].set_xlabel("epoch")
plot_decision(as_predictor(neuron_model), X_moons, y_moons, ax=axes[1], title=f"one neuron on moons (train acc {acc_train:.2f})")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see:** You should see `loss: 0.702 -> 0.267 | train accuracy 0.875 | test accuracy 0.825`. The loss starts near 0.70 (that is `-ln 0.5`: an untrained neuron says "50 %" about everything) and falls steadily. The plot shows a nearly horizontal straight boundary (the trained weights are about (1.28, −4.28) with a bias of 0.13, so the vertical position matters most): it separates the upper moon from the lower one reasonably well, but the two tips of the moons reach across the line, so 12.5 % of the training points and 17.5 % of the test points are on the wrong side. A single neuron cannot bend its line, and that is the limit you hit here.

# %% [markdown]
# ### ✏️ Exercise 3.5: Train a single neuron with PyTorch
#
# Your task: a straight line is perfect when the classes really are separated by a line. Make a **linearly separable** data set yourself with NumPy (two blobs of 100 points: one around (−1, −1), one around (+1, +1), spread 0.5), train the same neuron on it with `train_neuron`, and store the training accuracy in `acc_blobs`. Compare it with the moons accuracy and write two sentences: what does this tell you about when one neuron is enough?
#
# Replace the `None` values (or edit the code) in the starter cell, run it, then run the **Check** cell below it.

# %%
acc_blobs = None   # train accuracy of one neuron on two well-separated blobs

# %%
if acc_blobs is None:
    print("Fill in acc_blobs above, then run this cell again.")
else:
    check("One neuron separates the blobs almost perfectly", acc_blobs >= 0.95, "Build the blobs with rng.normal around (-1, -1) and (1, 1), then call train_neuron.")
    check("... and does better than on the moons", acc_blobs > acc_train, "A line fits blobs better than moons.")
    assert acc_blobs <= 1.0

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# rng = np.random.RandomState(0)
# X_blobs = np.vstack([rng.normal(-1, 0.5, size=(100, 2)), rng.normal(1, 0.5, size=(100, 2))]).astype(np.float32)
# y_blobs = np.array([0] * 100 + [1] * 100, dtype=np.float32)
# blob_model, blob_losses, acc_blobs = train_neuron(X_blobs, y_blobs)
# print("blobs train accuracy:", round(acc_blobs, 3))
# ```
#
# When a straight line can separate the classes, one neuron is enough. When they curve around each other, you need more neurons and a hidden layer, which is Part 4.
#
# </details>

# %% [markdown]
# ### ✏️ Imagine first, revisited
#
# Run the next cell to compare your prediction with the facts.

# %%
# Imagine first, revisited: the truth about one neuron
truth = {"AND": "yes", "OR": "yes", "XOR": "no"}
hits = sum(my_prediction.get(g) == t for g, t in truth.items())
RESULTS["part3_gate_predictions_right"] = hits
print(f"You predicted {hits} of the 3 gates correctly. The facts: {truth}")
print("Your boundary shape:", repr(boundary_shape) if boundary_shape else "(you did not write one; the right answer is: a straight line)")

# %% [markdown]
# **What you should see:** You should see how many of your three gate predictions were right (the facts are AND yes, OR yes, XOR no). Compare your sentence about the boundary shape with what you have seen in 3.2 to 3.5: one neuron always draws a **straight line** (in 2D; a flat plane in 3D), never a curve.

# %% [markdown]
# ## Part 4. Layers and stacking: a network from scratch (30 min)
#
# In Part 3 you built **one** neuron and saw that it can only draw a straight line. Now you stack neurons into **layers** and layers into a **network**. You will use the lesson sections **Layers** and **Hidden layers**.
#
# Five steps: (4.1) see a layer as one matrix multiplication, (4.2) build by hand a tiny network that solves XOR, (4.3) prove numerically that layers without an activation collapse into one, (4.4) count a network's parameters and memory, and (4.5) draw networks and design one for handwritten digits.

# %% [markdown]
# ### ✏️ Imagine first
#
# Picture a team of **two specialist neurons** that both look at the same two inputs, followed by **one decider neuron** that only listens to the two specialists. Nobody on the team can draw anything but a straight line. Can the *team* solve XOR (output 1 when exactly one input is 1)?
#
# Write one or two sentences in `team_idea`: what could each specialist be good at, and what would the decider do with their answers? Then answer `team_solves_xor` with `True` or `False`. You will test your idea in step 4.2.

# %%
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

def sigmoid(z):
    """The sigmoid activation, for numbers or NumPy arrays."""
    return 1.0 / (1.0 + np.exp(-z))

team_idea = ""            # YOUR TURN: e.g. "One specialist notices 'at least one input is on', the other 'both are on' ..."
team_solves_xor = None    # YOUR TURN: True or False
print("Your idea:", team_idea if team_idea else "(nothing yet: write your sentence above and run this cell again)")
print("Your prediction:", team_solves_xor)

# %% [markdown]
# ### 4.1 A layer is a matrix
#
# A **layer** is many neurons side by side, all reading the *same* inputs. Neuron number *i* has its own row of weights and its own bias. Put all the rows on top of each other and you get a **weight matrix** `W` with one row per neuron and one column per input. The whole layer is then one line of maths:
#
# $$a = \varphi(Wx + b)$$
#
# Below we build a layer of **3 neurons** that reads **2 inputs**. The matrix `W` has shape (3, 2), the bias `b` has 3 numbers, and `x` has 2 numbers. We compute the answer twice: once with a loop over the neurons (the way you built a neuron in Part 3) and once with a single matrix product. They must agree.

# %%
x = np.array([2.0, 1.0])                       # 2 inputs
W = np.array([[ 1.0, -1.0],                    # neuron 0: weights for (x1, x2)
              [ 0.5,  2.0],                    # neuron 1
              [-1.5,  0.5]])                   # neuron 2   -> shape (3, 2)
b = np.array([0.0, -1.0, 0.5])                 # one bias per neuron

# way 1: neuron by neuron
a_loop = np.array([sigmoid(np.dot(W[i], x) + b[i]) for i in range(3)])
# way 2: the whole layer at once
z = W @ x + b
a_matrix = sigmoid(z)

print("shapes: W", W.shape, " x", x.shape, " b", b.shape, " z", z.shape)
print("z (before the activation):", z)
print("a by loop  :", a_loop.round(3))
print("a by matrix:", a_matrix.round(3))
assert np.allclose(a_loop, a_matrix), "the loop and the matrix product should agree"
RESULTS["p4_layer_output"] = a_matrix.round(3).tolist()

# %% [markdown]
# **What you should see.** `z = [ 1.  2. -2.]` and `a = [0.731 0.881 0.119]`. The shapes are (3, 2) times (2,) gives (3,): *out × in* times *in* gives *out*. This is the rule you will use every time you meet a layer.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.1 - tune one neuron of the layer
#
# Neuron 0 currently gives `a[0] = 0.731` for `x = (2, 1)`. Choose a new weight row `row0` (two numbers) and bias `b0` so that neuron 0 gives **more than 0.9** for this same input. Hint: what is the `z` you need? Remember `sigmoid(2.2) ≈ 0.90`.
#
# Then compute the full layer output `a_new` with your new row in place of `W[0]`, and store the **shape of the matrix** if the layer had 5 neurons and 4 inputs in `shape_5x4` (as a tuple).

# %%
row0 = [1.0, -1.0]     # YOUR CODE HERE: two weights for neuron 0
b0 = 0.0               # YOUR CODE HERE: its bias
shape_5x4 = None       # YOUR ANSWER: a tuple like (rows, columns)

W_new = W.copy(); W_new[0] = row0
b_new = b.copy(); b_new[0] = b0
a_new = sigmoid(W_new @ x + b_new)
print("new layer output:", a_new.round(3))

# %%
# Check
check("neuron 0 now fires above 0.9", a_new[0] > 0.9, "you need w1*2 + w2*1 + b0 > 2.2, e.g. row0 = [1.5, 0.5], b0 = 0")
check("the other two neurons are unchanged", np.allclose(a_new[1:], a_matrix[1:]))
if shape_5x4 is None:
    print("Fill in `shape_5x4` above, then run this cell again.")
else:
    check("shape of a 5-neuron, 4-input weight matrix is (5, 4)", tuple(shape_5x4) == (5, 4), "rows = neurons (out), columns = inputs (in)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# row0 = [1.5, 0.5]    # z = 1.5*2 + 0.5*1 + 0 = 3.5, sigmoid(3.5) = 0.971
# b0 = 0.0
# shape_5x4 = (5, 4)   # out x in
# ```
# Many answers work: all you need is `2*w1 + w2 + b0 > 2.2`. The matrix has one row per neuron (5) and one column per input (4).
# </details>

# %% [markdown]
# ### 4.2 A hand-built network that solves XOR
#
# Now the team from the Imagine-first box. The network is `[2, 2, 1]`: 2 inputs, 2 hidden neurons, 1 output neuron, all with sigmoid activations. We choose the weights by hand (these are the same weights as the XOR network in the lesson):
#
# - **hidden neuron 1 acts like OR**: it fires when `x1 + x2` is at least about 1 (weights 20, 20, bias -10).
# - **hidden neuron 2 acts like AND**: it fires only when both inputs are 1 (weights 20, 20, bias -30).
# - **the output neuron says "OR but not AND"**: weight +20 for the OR neuron, -20 for the AND neuron, bias -10.
#
# Big weights like 20 make the sigmoid almost a step: values very close to 0 or 1.

# %%
W1 = np.array([[20.0, 20.0],    # hidden 1 = OR
               [20.0, 20.0]])   # hidden 2 = AND
b1 = np.array([-10.0, -30.0])
W2 = np.array([[20.0, -20.0]])  # output = hidden1 AND NOT hidden2   (shape 1 x 2)
b2 = np.array([-10.0])

def forward(x):
    """Return (hidden activations, output) for one input x of 2 numbers."""
    h = sigmoid(W1 @ x + b1)
    out = sigmoid(W2 @ h + b2)
    return h, out[0]

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
H_points = []
print("x1 x2 | hidden1 (OR)  hidden2 (AND) | output")
for x1, x2 in inputs:
    h, out = forward(np.array([x1, x2], dtype=float))
    H_points.append(h)
    print(f" {x1}  {x2} |   {h[0]:.4f}        {h[1]:.4f}      | {out:.4f}")
H_points = np.array(H_points)

xor_table = [round(forward(np.array(p, dtype=float))[1]) for p in inputs]
assert xor_table == [0, 1, 1, 0], "the hand-built network should compute XOR"
RESULTS["p4_xor_truth_table"] = xor_table

# %% [markdown]
# **What you should see.** The output column reads 0.0000 (rounded), 1.0000, 1.0000, 0.0000 - that is XOR, which no single neuron can do. Look at the hidden columns: for the four inputs the hidden pair is about (0, 0), (1, 0), (1, 0) and (1, 1). The hidden layer has **re-coded** the inputs. In the new coordinates the two "1" inputs sit on the same spot, and one straight line separates them from the two "0" inputs. The picture below shows this.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
cols = ["tab:blue", "tab:orange", "tab:orange", "tab:blue"]           # XOR labels 0,1,1,0
for (x1, x2), c in zip(inputs, cols):
    axes[0].scatter([x1], [x2], c=c, s=120, edgecolor="k"); axes[0].annotate(f"({x1},{x2})", (x1 + 0.04, x2 + 0.04))
axes[0].set_title("Input space: no straight line works"); axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
axes[0].set_xlim(-0.3, 1.4); axes[0].set_ylim(-0.3, 1.4)
for (x1, x2), c, hp in zip(inputs, cols, H_points):
    axes[1].scatter([hp[0]], [hp[1]], c=c, s=120, edgecolor="k", alpha=0.8); axes[1].annotate(f"({x1},{x2})", (hp[0] + 0.03, hp[1] + 0.03))
xs = np.linspace(-0.2, 1.3, 50)
axes[1].plot(xs, xs - 0.5, "k--", lw=1)                                # the output neuron's line: h1 - h2 = 0.5
axes[1].set_title("Hidden space: one straight line works"); axes[1].set_xlabel("hidden 1 (OR)"); axes[1].set_ylabel("hidden 2 (AND)")
axes[1].set_xlim(-0.2, 1.3); axes[1].set_ylim(-0.3, 1.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.2 - make XNOR, and check your Imagine-first idea
#
# 1. **XNOR** is the opposite of XOR: output 1 when the two inputs are *equal*. Keep the same hidden layer and change **only the output neuron** (`W2_x`, `b2_x`) so that the network computes XNOR. Hint: what happens if you swap the signs, and what must the bias become so that (0, 0) gives 1?
# 2. Look at the hidden values for the input (1, 1). Which hidden neuron is "on" there and what does it do to the output?
# 3. Compare with your Imagine-first idea. Were you right that a team of two specialists plus a decider can solve XOR? Write one sentence in `xor_reflection`.

# %%
W2_x = np.array([[20.0, -20.0]])     # YOUR CODE HERE: change the output weights
b2_x = np.array([-10.0])             # YOUR CODE HERE: change the output bias
xor_reflection = None                # YOUR ANSWER: one sentence

def forward_x(x):
    h = sigmoid(W1 @ x + b1)
    return sigmoid(W2_x @ h + b2_x)[0]
xnor_table = [round(forward_x(np.array(p, dtype=float))) for p in inputs]
print("your table for (0,0) (0,1) (1,0) (1,1):", xnor_table, "  goal: [1, 0, 0, 1]")

# %%
# Check
check("the network now computes XNOR", xnor_table == [1, 0, 0, 1], "flip the signs of the two output weights and make the bias positive, so that (0,0) gives z > 0")
check("Imagine-first answer given", team_solves_xor is not None, "go back to the Imagine-first cell and set team_solves_xor")
if xor_reflection is None:
    print("Fill in `xor_reflection` above, then run this cell again.")
else:
    check("your sentence has at least 12 words", n_words(xor_reflection) >= 12, "say what each hidden neuron does and what the output does with them")
RESULTS["p4_xor_reflection"] = xor_reflection

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# W2_x = np.array([[-20.0, 20.0]])    # flip the signs
# b2_x = np.array([10.0])             # (0,0): hidden = (0,0), z = +10 -> output 1
# xor_reflection = ("Yes: hidden neuron 1 learns 'at least one is on' and hidden neuron 2 learns 'both are on', "
#                   "and the output neuron says 'one but not both', so a team of straight-line neurons can solve XOR.")
# ```
# For (1, 1) the hidden pair is (1, 1), so `z = -20 + 20 + 10 = 10` and XNOR gives 1. The second hidden neuron (AND) cancels the first one exactly there.
# </details>

# %% [markdown]
# ### 4.3 Without an activation, depth gives nothing
#
# Why does the lesson insist on a *non-linear* activation between layers? Because two linear layers in a row are the same as **one** linear layer: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. Let us check this with random numbers: a layer from 3 inputs to 4 neurons, then a layer from 4 to 2, **with no activation**.

# %%
rng = np.random.default_rng(0)
W1r, b1r = rng.normal(size=(4, 3)), rng.normal(size=4)
W2r, b2r = rng.normal(size=(2, 4)), rng.normal(size=2)
xr = rng.normal(size=3)

two_layers = W2r @ (W1r @ xr + b1r) + b2r           # layer 1 then layer 2, no activation
W_one, b_one = W2r @ W1r, W2r @ b1r + b2r           # ONE layer with these merged numbers
one_layer = W_one @ xr + b_one
print("merged matrix shape:", W_one.shape)
print("two layers:", two_layers.round(4), " one layer:", one_layer.round(4))
assert np.allclose(two_layers, one_layer), "two linear layers must equal one"

# with a tanh in the middle the merge no longer works
with_tanh = W2r @ np.tanh(W1r @ xr + b1r) + b2r
print("with tanh :", with_tanh.round(4), "  same as the merged layer?", np.allclose(with_tanh, one_layer))
RESULTS["p4_collapse_merged_shape"] = W_one.shape

# %% [markdown]
# **What you should see.** The two outputs are identical, and the merged matrix has shape (2, 3). With a `tanh` in the middle the answers differ: the bend of the activation is what lets depth add power.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.3 - how big is the collapsed network?
#
# Imagine a network `784 → 128 → 10` **with no activation**. (a) What is the shape of the single matrix `W_merged` that does the same job? Store it as a tuple in `merged_shape`. (b) How many parameters does this one merged layer have (weights plus biases)? Store in `merged_params`. (c) The real network has 101,770 parameters. In one sentence in `collapse_note`, say what the extra parameters would buy you **without** an activation.

# %%
merged_shape = None    # YOUR ANSWER: (rows, columns)
merged_params = None   # YOUR ANSWER: an integer
collapse_note = None   # YOUR ANSWER: one sentence

# %%
# Check
if merged_shape is None or merged_params is None:
    print("Fill in `merged_shape` and `merged_params` above, then run this cell again.")
else:
    check("merged shape is (10, 784)", tuple(merged_shape) == (10, 784), "rows = outputs (10), columns = inputs (784)")
    check("merged parameter count is 7850", merged_params == 7850, "10 * 784 weights plus 10 biases")
if collapse_note is None:
    print("Fill in `collapse_note` above, then run this cell again.")
else:
    check("your note has at least 12 words", n_words(collapse_note) >= 12)
RESULTS["p4_collapse_note"] = collapse_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# merged_shape = (10, 784)
# merged_params = 10 * 784 + 10          # 7850
# collapse_note = "Nothing: without an activation the 101,770 parameters can still only do what 7,850 do, a single straight-line (linear) map."
# ```
# Depth only helps when each layer ends with a non-linear activation such as tanh or ReLU.
# </details>

# %% [markdown]
# ### 4.4 Count the parameters
#
# A layer with `n_in` inputs and `n_out` neurons has `n_in × n_out` weights and `n_out` biases. A network is a list of layer sizes, so the total is a sum. We write a function, then check it against PyTorch, which counts the numbers inside real `nn.Sequential` models. Memory: each parameter is a 32-bit float, **4 bytes**.

# %%
def count_params(layers):
    """Parameters of a fully connected network with the given layer sizes, e.g. [2, 4, 1]."""
    return sum(layers[i] * layers[i + 1] + layers[i + 1] for i in range(len(layers) - 1))

def torch_model(layers):
    """The matching PyTorch model: Linear + ReLU between layers, Linear at the end."""
    mods = []
    for i in range(len(layers) - 1):
        mods.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2: mods.append(nn.ReLU())
    return nn.Sequential(*mods)

print(f"{'layers':<22}{'by formula':>12}{'by PyTorch':>12}{'memory (MB)':>14}")
for layers in ([2, 4, 1], [784, 128, 10], [784, 512, 512, 10]):
    mine = count_params(layers)
    theirs = sum(p.numel() for p in torch_model(layers).parameters())
    assert mine == theirs, f"{layers}: formula says {mine}, PyTorch says {theirs}"
    print(f"{str(layers):<22}{mine:>12,}{theirs:>12,}{mine * 4 / 1e6:>14.4f}")
RESULTS["p4_params_784_128_10"] = count_params([784, 128, 10])

# %% [markdown]
# **What you should see.** `[2, 4, 1]` has 17 parameters, `[784, 128, 10]` has 101,770 and `[784, 512, 512, 10]` has 669,706 - about 2.7 MB. Formula and PyTorch agree every time (the `assert` would stop the cell otherwise). Note where the parameters live: in `[784, 128, 10]` the first layer alone holds 100,480 of the 101,770.

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.4 - a bigger network
#
# 1. Use `count_params` for `[784, 1024, 1024, 10]` and store it in `params_big`. How many megabytes is that at 4 bytes per parameter (`mb_big`)?
# 2. Which single layer of that network holds the most parameters? Store its index (0 = first layer) in `biggest_layer`.
# 3. Compared with `[784, 128, 10]`, how many times more parameters is it? Store a rounded integer in `times_more`.

# %%
params_big = None      # YOUR CODE HERE: use count_params
mb_big = None          # YOUR CODE HERE
biggest_layer = None   # YOUR ANSWER: 0, 1 or 2
times_more = None      # YOUR ANSWER: an integer

# %%
# Check
true_big = count_params([784, 1024, 1024, 10])
if params_big is None:
    print("Fill in the answers above, then run this cell again.")
else:
    check("parameter count of [784, 1024, 1024, 10]", params_big == true_big, "784*1024+1024, then 1024*1024+1024, then 1024*10+10")
    check("memory in MB (4 bytes each)", mb_big is not None and abs(mb_big - true_big * 4 / 1e6) < 0.01, "multiply by 4 and divide by a million")
    check("biggest layer", biggest_layer == 1, "compare 784*1024, 1024*1024 and 1024*10")
    check("times more than [784,128,10]", times_more == round(true_big / 101770), "divide the two counts")
RESULTS["p4_params_big"] = params_big

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# params_big = count_params([784, 1024, 1024, 10])     # 1,863,690
# mb_big = params_big * 4 / 1e6                         # about 7.45 MB
# biggest_layer = 1                                     # the 1024 x 1024 middle layer (1,049,600)
# times_more = round(params_big / 101770)               # 18
# ```
# Layer sizes multiply, so widening the hidden layers makes the count grow fast.
# </details>

# %% [markdown]
# ### 4.5 Draw it, then design one
#
# `plot_network_diagram` draws a circle for every unit and a line for every weight. It also returns the parameter count. Look at the three drawings, and notice how quickly the lines multiply.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, layers in zip(axes, ([2, 3, 1], [2, 4, 4, 1], [4, 8, 3])):
    n = plot_network_diagram(layers, ax=ax, title=f"{layers}\n{count_params(layers)} parameters")
    assert n == count_params(layers)
    ax.text(0, -(max(layers) / 2 + 0.6) * 1.2, "input", ha="center"); ax.text(len(layers) - 1, -(max(layers) / 2 + 0.6) * 1.2, "output", ha="center")
    if len(layers) > 2: ax.text((len(layers) - 1) / 2, -(max(layers) / 2 + 0.6) * 1.2, "hidden", ha="center")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.5 - design a digit reader on paper
#
# A digit picture is 28 × 28 = 784 numbers, and the answer is one of 10 digits. Design a network: choose the size of one or two hidden layers in `my_digit_net` (it must start with 784 and end with 10), compute its parameters and memory, and explain in `design_note` what you expect each layer to do (for example: "the first hidden layer might detect small strokes"). Why does the output layer have exactly 10 neurons? Why would a bigger network not automatically be better?

# %%
my_digit_net = [784, 64, 10]     # YOUR CODE HERE: change the hidden sizes, add a layer if you like
design_note = None               # YOUR ANSWER: 2 or 3 sentences

my_params = count_params(my_digit_net)
print(f"{my_digit_net}: {my_params:,} parameters = {my_params * 4 / 1e6:.2f} MB")
fig, ax = plt.subplots(figsize=(5, 3))
shown = [min(n, 8) for n in my_digit_net]            # draw at most 8 circles per layer so it stays readable
plot_network_diagram(shown, ax=ax, title=f"{my_digit_net} (drawn with at most 8 units per layer)")
plt.show()

# %%
# Check
check("starts with 784 inputs", my_digit_net[0] == 784, "one input per pixel")
check("ends with 10 outputs", my_digit_net[-1] == 10, "one output per digit 0 to 9")
check("has at least one hidden layer", len(my_digit_net) >= 3, "a network with no hidden layer is a single layer of neurons")
if design_note is None:
    print("Fill in `design_note` above, then run this cell again.")
else:
    check("your note has at least 25 words", n_words(design_note) >= 25, "say what the layers do, why 10 outputs, and the cost of size")
RESULTS["p4_my_digit_net"] = my_digit_net
RESULTS["p4_my_digit_params"] = my_params
RESULTS["p4_design_note"] = design_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_digit_net = [784, 128, 64, 10]          # 784*128+128 + 128*64+64 + 64*10+10 = 109,386 parameters
# design_note = ("The first hidden layer can learn small strokes and edges from the 784 pixels, the second combines them into "
#                "shapes like loops and bars. The output has 10 neurons, one score for each digit, and softmax turns them into "
#                "probabilities. A bigger network costs more memory and time and can overfit, so size should match the problem.")
# ```
# The sizes here are a choice, not a rule; what matters is that you can count them and give a reason.
# </details>

# %% [markdown]
# ## Part 5. Learning: gradient descent you can watch (30 min)
#
# A network starts with random weights and is useless. **Training** finds good weights by repeating three steps: *guess, measure the loss, nudge every weight a little downhill.* You will use the lesson section **Training**.
#
# Five steps: (5.1) one weight and one step by hand, then twenty steps; (5.2) the learning-rate experiment; (5.3) two weights and a bias, with the gradient checked against PyTorch's `autograd`; (5.4) a whole `[2, 4, 1]` network trained in pure NumPy while you watch the boundary form; (5.5) write what you saw.

# %% [markdown]
# ### ✏️ Imagine first
#
# Training walks downhill on the loss landscape, and the **learning rate** `η` is the size of each step. Suppose you make `η` much too large. Sketch in words what the loss curve (loss on the vertical axis, step number on the horizontal axis) would look like. Choose one in `lr_prediction`: `"smooth fall"`, `"crawls"`, `"bounces then settles"`, or `"explodes"` - and write your reason in `lr_reason`. You will test this in step 5.2.

# %%
lr_prediction = "smooth fall"   # YOUR TURN: change to "crawls", "bounces then settles" or "explodes" if you disagree
lr_reason = ""                  # YOUR TURN: one sentence, e.g. "A huge step jumps over the valley and lands higher up than before."
print("Prediction for a huge learning rate:", lr_prediction, "|", lr_reason if lr_reason else "(add a reason and run again)")

# %% [markdown]
# ### 5.1 One weight, one step by hand
#
# The smallest model: `ŷ = w · x`. One training example `x = 2`, true answer `y = 4`, so the perfect weight is `w = 2`. We start at `w = 0.5`. The loss is the squared error `L = (w·x − y)²`. Its slope (gradient) with respect to `w` is `dL/dw = 2·(ŷ − y)·x`. The update rule is
#
# $$w \leftarrow w - \eta \cdot \frac{dL}{dw}$$
#
# With learning rate `η = 0.1`: prediction `ŷ = 1`, loss `9`, gradient `2·(1 − 4)·2 = −12`, so the new weight is `0.5 − 0.1·(−12) = 1.7`. The slope was negative (the loss falls as `w` grows), so the update *increased* `w`. Run it, then walk twenty steps.

# %%
x_t, y_t = 2.0, 4.0

def loss_1w(w):
    return (w * x_t - y_t) ** 2

def grad_1w(w):
    return 2 * (w * x_t - y_t) * x_t

w, eta = 0.5, 0.1
print(f"start: w={w}  prediction={w * x_t}  loss={loss_1w(w)}  gradient={grad_1w(w)}")
w = w - eta * grad_1w(w)
print(f"after ONE step: w={w:.4f}  loss={loss_1w(w):.4f}")
assert abs(w - 1.7) < 1e-9, "the lesson's worked step gives 1.7"

# now twenty steps, remembering where we walked
w, path = 0.5, [0.5]
for step in range(20):
    w = w - eta * grad_1w(w)
    path.append(w)
print("w after steps 0, 1, 2, 3, 5, 10, 20:", [round(path[i], 4) for i in (0, 1, 2, 3, 5, 10, 20)])
RESULTS["p5_w_after_20"] = round(path[-1], 6)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot([loss_1w(p) for p in path], "o-"); axes[0].set_yscale("log")
axes[0].set_xlabel("step"); axes[0].set_ylabel("loss (log scale)"); axes[0].set_title("loss per step")
grid = np.linspace(-0.5, 4.5, 200)
axes[1].plot(grid, loss_1w(grid), "k-")
axes[1].plot(path[:8], [loss_1w(p) for p in path[:8]], "o-", c="tab:red")
axes[1].set_xlabel("w"); axes[1].set_ylabel("loss"); axes[1].set_title("the first 7 steps walk down the parabola")
plt.tight_layout(); plt.show()

# %% [markdown]
# **What you should see.** One step gives `w = 1.7` and the loss falls from 9 to `0.36`. The error `w − 2` is multiplied by `0.2` at every step (because `1 − η·8 = 0.2`), so after 20 steps `w` is `2.0000` to four decimals and the loss has fallen from 9 to practically zero (about `1e-27`). Early steps are big (the slope is steep), later steps tiny: gradient descent slows down by itself as it nears the bottom.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.1 - do the steps yourself
#
# 1. Do one step by hand for `x = 3`, `y = 6`, `w = 1.0`, `η = 0.05`. Compute the prediction, the gradient `2·(ŷ − y)·x` and the new weight. Store the new weight in `w_next`.
# 2. Count how many steps `n_steps` it takes (with `η = 0.1`, starting from 0.5, in the problem above) before the loss drops below `0.001`. Use a loop.

# %%
w_next = None     # YOUR ANSWER: the new weight after one step
n_steps = None    # YOUR CODE HERE: loop with a counter until loss_1w(w) < 0.001

# %%
# Check
# x=3, y=6, w=1: yhat=3, gradient=2*(3-6)*3=-18, w_next=1-0.05*(-18)=1.9
if w_next is None:
    print("Fill in `w_next` above, then run this cell again.")
else:
    check("w_next is 1.9", abs(w_next - 1.9) < 1e-6, "yhat = 3, gradient = 2*(3-6)*3 = -18, w - 0.05*(-18)")
if n_steps is None:
    print("Fill in `n_steps` above, then run this cell again.")
else:
    check("n_steps is 3", n_steps == 3, "count the steps until the loss is below 0.001 (the loss is 9 * 0.04^k)")
RESULTS["p5_n_steps"] = n_steps

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# w_next = 1.0 - 0.05 * (2 * (3.0 * 1.0 - 6.0) * 3.0)     # 1.9
# w, n_steps = 0.5, 0
# while loss_1w(w) >= 0.001:
#     w = w - 0.1 * grad_1w(w); n_steps += 1
# ```
# The loss is multiplied by `0.2² = 0.04` each step: 9, 0.36, 0.0144, 0.000576, so it first falls below 0.001 after 3 steps.
# </details>

# %% [markdown]
# ### 5.2 The learning-rate experiment
#
# Same problem, same start (`w = 0.5`), six different learning rates. For this problem the error `w − 2` is multiplied at every step by `1 − 8η` (the slope of the gradient is `2x² = 8`). That one number explains everything:
#
# - if `|1 − 8η|` is **below 1**, the error shrinks: `w` converges (smoothly when `1 − 8η > 0`, bouncing from side to side when it is negative);
# - if it is **exactly 1** in size (`η = 0.25`), the error never shrinks: `w` jumps between two values forever;
# - if it is **above 1** (`η > 0.25`), every step overshoots by more than before: the loss **explodes**.
#
# The stability limit is therefore `η = 2 / (2x²) = 1/x² = 0.25` for `x = 2`.

# %%
def run_gd(eta, steps=20, w0=0.5):
    """Return the loss at every step of gradient descent on the one-weight problem."""
    w, losses = w0, [loss_1w(w0)]
    for _ in range(steps):
        w = w - eta * grad_1w(w)
        losses.append(loss_1w(w))
    return losses

etas = [0.001, 0.05, 0.2, 0.25, 0.3]
notes = {0.001: "crawls", 0.05: "converges", 0.2: "bounces, then settles", 0.25: "bounces forever", 0.3: "explodes"}
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for eta in etas:
    L = run_gd(eta)
    ax.plot(L, "o-", ms=3, label=f"η = {eta}  ({notes[eta]})")
    print(f"η = {eta:<6} factor 1-8η = {1 - 8 * eta:+.2f}   loss after 20 steps = {L[-1]:.3g}")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss (log scale)"); ax.set_title("Four behaviours of gradient descent")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

limit = 2 / (2 * x_t ** 2)
print("stability limit 2 / (2 x^2) =", limit)
assert abs(limit - 0.25) < 1e-12
assert run_gd(0.3)[-1] > run_gd(0.3)[0], "η = 0.3 should blow up"
assert run_gd(0.05)[-1] < 1e-6, "η = 0.05 should converge"
RESULTS["p5_stability_limit"] = limit

# %% [markdown]
# **What you should see.** `η = 0.001` is still at loss ≈ 6.5 after 20 steps (it crawls: the error shrinks by only 0.8 % per step). `η = 0.05` (factor +0.6) and `η = 0.2` (factor −0.6, so it hops from side to side of the minimum) both end near `1e-8`. `η = 0.25` stays at loss 9 forever (the error flips sign but keeps its size), and `η = 0.3` has reached about `6e+6` by step 20 and keeps growing. Your Imagine-first prediction: a too-large step does not fall smoothly; it jumps over the valley and can land higher than it started.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.2 - the limit for a different input
#
# Now suppose the training example is `x = 3, y = 6`. Using the idea above (stability limit `= 2 / (2x²)`), what is the largest learning rate that still converges? Store it in `limit_x3`. Then **test** it: run gradient descent for that problem with a learning rate just below your limit and just above it (use `run_gd` as a model, with a new loss and gradient), and set `below_ok` and `above_blows_up` to `True` or `False`. In `lr_lesson`, write one sentence: why must the learning rate depend on the size of the inputs?

# %%
limit_x3 = None            # YOUR ANSWER: a number
below_ok = None            # YOUR ANSWER: True or False
above_blows_up = None      # YOUR ANSWER: True or False
lr_lesson = None           # YOUR ANSWER: one sentence

def run_gd_xy(eta, x, y, steps=30, w0=0.5):
    """Same experiment for any single example (x, y)."""
    w = w0
    for _ in range(steps):
        w = w - eta * 2 * (w * x - y) * x
    return (w * x - y) ** 2          # the final loss

# %%
# Check
if limit_x3 is None:
    print("Fill in `limit_x3` above, then run this cell again.")
else:
    check("limit for x=3 is 1/9 = 0.111", abs(limit_x3 - 1 / 9) < 0.005, "2 / (2 * 3**2)")
    check("just below the limit it converges", run_gd_xy(limit_x3 * 0.95, 3, 6) < 1e-3)
    check("just above the limit it blows up", run_gd_xy(limit_x3 * 1.05, 3, 6) > 1e3)
    check("below_ok and above_blows_up are both True", below_ok is True and above_blows_up is True, "test with run_gd_xy(limit*0.95, 3, 6) and run_gd_xy(limit*1.05, 3, 6)")
if lr_lesson is not None:
    check("your sentence has at least 12 words", n_words(lr_lesson) >= 12)
RESULTS["p5_limit_x3"] = limit_x3
RESULTS["p5_lr_prediction"] = lr_prediction
RESULTS["p5_lr_reason"] = lr_reason

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# limit_x3 = 2 / (2 * 3 ** 2)          # 0.111...
# below_ok = run_gd_xy(limit_x3 * 0.95, 3, 6) < 1e-3     # True
# above_blows_up = run_gd_xy(limit_x3 * 1.05, 3, 6) > 1e3  # True
# lr_lesson = "A bigger input makes the loss curve steeper, so the same step size overshoots more and the safe learning rate gets smaller."
# ```
# This is why people scale their inputs (for example to the range -1..1) before training: then one learning rate suits all the weights.
# </details>

# %% [markdown]
# ### 5.3 Two weights and a bias, with gradients checked against `autograd`
#
# Real neurons have several weights. For one **sigmoid neuron** with inputs `x1, x2`, weights `w1, w2`, bias `b`, prediction `p = σ(w1·x1 + w2·x2 + b)` and the cross-entropy loss `L = −[y·ln p + (1−y)·ln(1−p)]`, calculus gives a lovely simple result:
#
# ```
# dL/dw1 = (p − y) · x1        dL/dw2 = (p − y) · x2        dL/db = (p − y)
# ```
#
# (averaged over the examples). "Error times input": the wrong-ness of the prediction, multiplied by how much that input contributed. We write this by hand and check it against PyTorch, which computes gradients automatically (`loss.backward()`).

# %%
# six labelled points: the label is 1 when x1 + x2 is large
Xs = np.array([[0.0, 0.5], [1.0, 0.0], [0.5, 1.0], [2.0, 1.5], [1.5, 2.0], [2.5, 2.0]])
ys = np.array([0, 0, 0, 1, 1, 1], dtype=float)

def loss_and_grads(w, b, X, y):
    """Mean cross-entropy and the gradients, written by hand."""
    p = sigmoid(X @ w + b)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    err = p - y                          # (p - y) for every example
    return loss, X.T @ err / len(y), err.mean()

w0, b0 = np.array([0.3, -0.2]), 0.1
loss0, gw, gb = loss_and_grads(w0, b0, Xs, ys)

# the same numbers in PyTorch
wt = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
bt = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
pt = torch.sigmoid(torch.tensor(Xs) @ wt + bt)
lt = nn.functional.binary_cross_entropy(pt, torch.tensor(ys))
lt.backward()

print("loss  by hand:", round(loss0, 6), "  PyTorch:", round(lt.item(), 6))
print("dL/dw by hand:", gw.round(6), "  PyTorch:", wt.grad.numpy().round(6))
print("dL/db by hand:", round(gb, 6), "  PyTorch:", round(bt.grad.item(), 6))
assert np.allclose(gw, wt.grad.numpy()) and abs(gb - bt.grad.item()) < 1e-9, "hand gradients must match autograd"
RESULTS["p5_grad_check"] = "hand-written gradients match autograd"

# %% [markdown]
# **What you should see.** The loss is `0.6586`, the gradient `dL/dw` is about `[−0.2692, −0.2572]` and `dL/db` about `0.0597`; hand and PyTorch agree to six decimals. The `assert` passes silently. This is the idea of *backpropagation* in miniature: the maths has a pattern, so a computer can produce every gradient.

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.3 - train it and pick a learning rate
#
# The loop below trains the neuron for 200 steps with each learning rate in `lrs`. Run it, look at the final losses, and choose the learning rate you trust most in `best_lr` (it must be one of the values in `lrs`). Also say in `best_why` why you did not pick the smallest or the largest one.

# %%
def train_neuron(lr, steps=200):
    w, b = np.array([0.3, -0.2]), 0.1
    for _ in range(steps):
        _, gw_, gb_ = loss_and_grads(w, b, Xs, ys)
        w, b = w - lr * gw_, b - lr * gb_
    return w, b, loss_and_grads(w, b, Xs, ys)[0]

lrs = [0.01, 0.1, 1.0, 10.0]
final = {}
for lr in lrs:
    w_, b_, L_ = train_neuron(lr)
    final[lr] = L_
    print(f"lr = {lr:<6} final loss = {L_:.4f}   w = {w_.round(2)}  b = {b_:.2f}")

best_lr = None     # YOUR ANSWER: one of the values in lrs
best_why = None    # YOUR ANSWER: one sentence

# %%
# Check
if best_lr is None:
    print("Fill in `best_lr` above, then run this cell again.")
else:
    check("best_lr is one of the tried values", best_lr in lrs, "choose from [0.01, 0.1, 1.0, 10.0]")
    if best_lr in lrs:
        check("it reaches the lowest loss of the four (within 5 %)", final[best_lr] <= min(final.values()) * 1.05 + 1e-9, "look at the 'final loss' column")
if best_why is not None:
    check("your explanation has at least 12 words", n_words(best_why) >= 12)
RESULTS["p5_best_lr_neuron"] = best_lr
RESULTS["p5_neuron_final_losses"] = {str(k): round(float(v), 4) for k, v in final.items()}

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_lr = min(final, key=final.get)     # 10.0 on our run: final loss 0.0026
# best_why = "0.01 and 0.1 are still far from the minimum after 200 steps; on this small, easy data a big rate such as 10 reaches the lowest loss, but on harder data it could overshoot."
# ```
# Whichever rate has the lowest printed loss is a correct answer. On our run the final losses are 0.535, 0.222, 0.032 and 0.0026 for 0.01, 0.1, 1 and 10. This data set is tiny and its inputs are small, so even 10 is stable here; with bigger inputs it would blow up (step 5.2).
# </details>

# %% [markdown]
# ### 5.4 A whole network trained in pure NumPy
#
# Now the real thing: the `[2, 4, 1]` network (2 inputs, 4 `tanh` hidden units, 1 `sigmoid` output) learns the XOR data set. Everything is written with NumPy, including the **backward pass** that produces the gradient of every weight. The function `backward` below is short and heavily commented: read it, run it, and you will see it is the same "error times input" idea as in 5.3, passed backwards one layer at a time.
#
# ==You do not need to memorize backpropagation. PyTorch does it for you in Part 6 with one call, `loss.backward()`.==

# %%
def init_net(hidden, seed=0):
    """Small random weights. Rows of W1 = inputs, columns = hidden units."""
    r = np.random.RandomState(seed)
    return {"W1": r.normal(0, 1, (2, hidden)), "b1": np.zeros(hidden),
            "W2": r.normal(0, 1, (hidden, 1)), "b2": np.zeros(1)}

def forward_np(p, X):
    """Forward pass: returns the hidden activations A1 and the output probabilities P."""
    A1 = np.tanh(X @ p["W1"] + p["b1"])                 # hidden layer: tanh(XW + b)
    P = sigmoid(A1 @ p["W2"] + p["b2"]).ravel()         # output layer: sigmoid
    return A1, P

def bce_np(P, y):
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(P) + (1 - y) * np.log(1 - P)))

def backward(p, X, y):
    """Gradient of the mean cross-entropy with respect to every parameter."""
    A1, P = forward_np(p, X)
    d2 = ((P - y) / len(y))[:, None]                    # output error: (p - y), averaged over the examples
    g = {"W2": A1.T @ d2, "b2": d2.sum(0)}              # "error times input": the input of the output layer is A1
    d1 = (d2 @ p["W2"].T) * (1 - A1 ** 2)               # send the error back through W2, times tanh'(z) = 1 - tanh(z)^2
    g["W1"] = X.T @ d1                                  # again "error times input": the input is X
    g["b1"] = d1.sum(0)
    return g

def train_np(X, y, hidden=4, lr=0.5, epochs=1500, seed=0, snap_at=(0, 50, 300, 1500)):
    """Full-batch gradient descent. Returns the final parameters, the loss per epoch and snapshots of the parameters."""
    p = init_net(hidden, seed)
    losses, snaps = [], {}
    for epoch in range(epochs + 1):
        losses.append(bce_np(forward_np(p, X)[1], y))
        if epoch in snap_at:
            snaps[epoch] = {k: v.copy() for k, v in p.items()}
        g = backward(p, X, y)
        for k in p:
            p[k] -= lr * g[k]                           # the update rule: w <- w - lr * gradient
    return p, losses, snaps

# first, prove that backward() is right: compare with PyTorch autograd on the same numbers
Xg, yg = make_xor(20, 0.12, seed=5)
pg = init_net(4, seed=1)
tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in pg.items()}
h = torch.tanh(torch.tensor(Xg, dtype=torch.float64) @ tp["W1"] + tp["b1"])
pr = torch.sigmoid(h @ tp["W2"] + tp["b2"]).squeeze(1)
nn.functional.binary_cross_entropy(pr, torch.tensor(yg, dtype=torch.float64)).backward()
mine = backward(pg, Xg, yg)
for k in pg:
    assert np.allclose(mine[k], tp[k].grad.numpy(), atol=1e-8), f"gradient of {k} is wrong"
print("hand-written backward() matches PyTorch autograd for W1, b1, W2, b2")

# %% [markdown]
# Now train on 200 XOR points and take snapshots at epochs 0, 50, 300 and 1500.

# %%
Xx, yx = make_xor(200, 0.12, seed=0)
params, losses, snaps = train_np(Xx, yx, hidden=4, lr=0.5)
acc_np = float(((forward_np(params, Xx)[1] > 0.5) == yx).mean())
print(f"loss: epoch 0 = {losses[0]:.3f}, epoch 50 = {losses[50]:.3f}, epoch 300 = {losses[300]:.3f}, epoch 1500 = {losses[1500]:.3f}")
print(f"final training accuracy: {acc_np:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, e in zip(axes, (0, 50, 300, 1500)):
    snap = snaps[e]
    a = float(((forward_np(snap, Xx)[1] > 0.5) == yx).mean())
    plot_decision(lambda Z, s=snap: forward_np(s, Z)[1], Xx, yx, ax=ax, title=f"epoch {e}: loss {losses[e]:.2f}, acc {a:.2f}")
plt.tight_layout(); plt.show()

plt.figure(figsize=(5.5, 3.4)); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("training loss"); plt.show()
assert losses[-1] < losses[0] * 0.5, "the loss should at least halve"
assert acc_np > 0.88, "the network should solve XOR (the data are noisy, so not 100 %)"
RESULTS["p5_numpy_acc"] = round(acc_np, 3)
RESULTS["p5_numpy_final_loss"] = round(losses[-1], 3)

# %% [markdown]
# **What you should see.** The loss falls from about `0.82` (epoch 0) to `0.57` (epoch 50), `0.19` (epoch 300) and `0.14` (epoch 1500). Training accuracy goes from about `0.55` to `0.935`. It never reaches 1.0 because `make_xor` adds noise, so a few points sit on the wrong side of the corner they belong to (the labels were set *before* the noise moved the points).

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.4 - change `lr` and `hidden`
#
# Train the network with several settings and fill the table. In `settings` give 4 or more pairs `(hidden, lr)`; the loop prints the final loss and accuracy for each. Try at least `hidden = 1`, `hidden = 2` and one `lr` that is too small (`0.05`). Then store in `smallest_hidden` the smallest number of hidden units that reaches accuracy of at least 0.88, and in `slow_lr_note` what you saw with the small learning rate.

# %%
settings = [(1, 0.5), (2, 0.5), (4, 0.5), (4, 0.05), (8, 0.5)]    # YOUR TURN: add your own pairs
table = {}
for hidden, lr in settings:
    pm, ls, _ = train_np(Xx, yx, hidden=hidden, lr=lr, epochs=1500, snap_at=())
    a = float(((forward_np(pm, Xx)[1] > 0.5) == yx).mean())
    table[(hidden, lr)] = (ls[-1], a)
    print(f"hidden = {hidden:<2} lr = {lr:<5} final loss = {ls[-1]:.3f}   accuracy = {a:.3f}")

smallest_hidden = None   # YOUR ANSWER: an integer
slow_lr_note = None      # YOUR ANSWER: one sentence

# %%
# Check
good = [h for (h, lr), (l, a) in table.items() if a >= 0.88]
if smallest_hidden is None:
    print("Fill in `smallest_hidden` above, then run this cell again.")
else:
    check("smallest_hidden matches your table", smallest_hidden == min(good) if good else False, "the smallest 'hidden' whose accuracy is at least 0.88; one unit cannot solve XOR")
check("you tried at least 4 settings", len(settings) >= 4)
if slow_lr_note is not None:
    check("your note has at least 10 words", n_words(slow_lr_note) >= 10)
RESULTS["p5_smallest_hidden"] = smallest_hidden
RESULTS["p5_slow_lr_note"] = slow_lr_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# smallest_hidden = 4      # with these settings the table shows 1 unit 0.68, 2 units 0.815, 4 units 0.935
# slow_lr_note = "With lr = 0.05 the loss was still 0.27 after 1500 epochs (against 0.14 for lr = 0.5): the steps were so small that it had not finished learning."
# ```
# Compare your own table. One hidden unit stays far from solving XOR (a single straight cut again); a few more units give a much better, bent boundary. Your exact numbers can differ slightly if you change the data or the seed.
# </details>

# %% [markdown]
# ### 5.5 Observe and write
#
# Look again at the four decision-region pictures from step 5.4 (epochs 0, 50, 300, 1500). Describe in your own words what the boundary looked like **early** and **late**, what the colours do, and where the network still makes mistakes. Use the words *boundary*, *loss* and *epoch*.

# %%
boundary_note = None   # YOUR ANSWER: 3 or 4 sentences

# %%
# Check
if boundary_note is None:
    print("Fill in `boundary_note` above, then run this cell again.")
else:
    check("your description has at least 30 words", n_words(boundary_note) >= 30, "describe early, late and the remaining mistakes")
    for word in ("boundary", "loss", "epoch"):
        check(f"you used the word '{word}'", word in boundary_note.lower())
RESULTS["p5_boundary_note"] = boundary_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# boundary_note = ("At epoch 0 the boundary is almost a random line that cuts the picture in two and the loss is high (about 0.8). "
#                  "By epoch 50 the boundary has started to bend, and at epoch 300 the loss is much lower and the four corners are "
#                  "mostly in the right colours. At epoch 1500 the boundary is a smooth X-shaped cross, and the remaining mistakes are "
#                  "the noisy points that lie right next to it.")
# ```
# Early: a nearly straight, random cut. Late: a bent boundary that follows the four corners; the leftover errors are noisy points near the boundary.
# </details>

# %% [markdown]
# ## Part 6. The same thing in PyTorch, then break it (25 min)
#
# In Part 5 you wrote the backward pass yourself. PyTorch does it for you, so you can spend your time on the interesting questions: *how big should the network be?* and *what happens when it is too big or trained too long?* You will use the lesson sections **Training** and **Generalization**.
#
# Five steps: (6.1) the standard PyTorch training loop on the circles data, (6.2) a capacity sweep, (6.3) watch overfitting happen, (6.4) fix it three ways, (6.5) a fair test with train/validation/test splits.

# %% [markdown]
# ### ✏️ Imagine first
#
# The `make_circles` data set is a small disc of class 1 inside a ring of class 0. A hidden layer with `tanh` units can cut the plane with one bent line per unit. **How many hidden units does it need to separate disc from ring well?** Choose a number in `h_guess` (1, 2, 4, 8 or 32) and write one sentence in `h_reason` (think: how many straight cuts does it take to fence in a round shape?). You will test it in step 6.2.

# %%
h_guess = 2          # YOUR TURN: 1, 2, 4, 8 or 32
h_reason = ""        # YOUR TURN: one sentence
print("My guess:", h_guess, "hidden units.", h_reason if h_reason else "(add a reason and run again)")

# %% [markdown]
# ### 6.1 The standard training loop
#
# We build `nn.Sequential(nn.Linear(2, H), nn.Tanh(), nn.Linear(H, 1))`. The last layer has **no sigmoid**: the loss `BCEWithLogitsLoss` applies the sigmoid inside, which is more stable numerically. The optimiser **Adam** is a popular, forgiving version of gradient descent. Every training loop in the world has the same five lines, annotated below.

# %%
def make_mlp(H, depth=1):
    """2 inputs -> `depth` hidden layers of H tanh units -> 1 output (a logit, not yet a probability)."""
    layers, n_in = [], 2
    for _ in range(depth):
        layers += [nn.Linear(n_in, H), nn.Tanh()]
        n_in = H
    return nn.Sequential(*layers, nn.Linear(n_in, 1))

def accuracy(model, X, y):
    with torch.no_grad():
        return float(((model(torch.tensor(X)).squeeze(1) > 0).float().numpy() == y).mean())

def prob_fn(model):
    """Wrap a model so that plot_decision can call it: (N, 2) array in, probabilities out."""
    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(Z))).squeeze(1).numpy()
    return predict

Xc, yc = make_circles(200, seed=0)           # training data
Xc_test, yc_test = make_circles(400, seed=1) # a separate test set from a different seed
Xt_, yt_ = torch.tensor(Xc), torch.tensor(yc)

torch.manual_seed(SEED)
model = make_mlp(8)
loss_fn = nn.BCEWithLogitsLoss()                         # sigmoid + cross-entropy in one
opt = torch.optim.Adam(model.parameters(), lr=0.05)      # the optimiser: knows the parameters and the learning rate
epochs = 150 if QUICK else 400
history = []
for epoch in range(epochs):
    opt.zero_grad()                                      # 1. forget the gradients from the last step
    logits = model(Xt_).squeeze(1)                       # 2. forward pass: predictions for all 200 points
    loss = loss_fn(logits, yt_)                          # 3. measure the loss
    loss.backward()                                      # 4. backpropagation: fill in every parameter's gradient
    opt.step()                                           # 5. nudge every parameter downhill
    history.append(loss.item())

train_acc, test_acc = accuracy(model, Xc, yc), accuracy(model, Xc_test, yc_test)
print(f"loss: first {history[0]:.3f} -> last {history[-1]:.4f}   train acc {train_acc:.3f}   test acc {test_acc:.3f}")
assert history[-1] < history[0] * 0.5, "the loss should go down a lot"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].plot(history); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss"); axes[0].set_title("loss curve (H = 8)")
plot_decision(prob_fn(model), Xc, yc, ax=axes[1], title=f"circles: train {train_acc:.2f}, test {test_acc:.2f}")
plt.tight_layout(); plt.show()
RESULTS["p6_circles_h8_test_acc"] = round(test_acc, 3)

# %% [markdown]
# **What you should see.** The loss falls from about `0.72` (a coin toss is `ln 2 = 0.693`) to about `0.004`, and the decision region is a roundish blob around the disc. Training accuracy is 1.0 and test accuracy about `0.99` (`0.988` in our run). Because the optimiser and the initial weights are random-ish, your last digits can differ a little.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.1 - read the loop
#
# 1. In `loop_notes`, write one short phrase for each of the five lines in the loop (`zero_grad`, forward, loss, `backward`, `step`): what does it do and what would go wrong if it were missing?
# 2. Try to predict: if you **removed** `opt.zero_grad()`, would training still work? Answer in `zero_grad_effect` (`"works the same"`, `"gradients pile up and training goes wrong"`, or `"crashes"`).
# 3. The first loss is close to `0.693`. Explain why in `why_693`.

# %%
loop_notes = None         # YOUR ANSWER: a few phrases (a string or a list of strings)
zero_grad_effect = None   # YOUR ANSWER: one of the three options
why_693 = None            # YOUR ANSWER: one sentence

# %%
# Check
if zero_grad_effect is None:
    print("Fill in `zero_grad_effect` above, then run this cell again.")
else:
    check("gradients add up when you do not reset them", zero_grad_effect == "gradients pile up and training goes wrong", "PyTorch ADDS new gradients to the old ones unless you call zero_grad()")
if why_693 is not None:
    check("your answer mentions the natural logarithm of 2 or a coin toss / 0.5 probability", any(w in why_693.lower() for w in ("ln 2", "ln(2)", "0.5", "coin", "half", "guess")), "think of a model that says 0.5 for everything")
if loop_notes is not None:
    check("your loop notes have at least 20 words", n_words(loop_notes if isinstance(loop_notes, str) else " ".join(loop_notes)) >= 20)
RESULTS["p6_loop_notes"] = loop_notes

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# loop_notes = ["zero_grad: clears old gradients, else they add up",
#               "forward: computes the predictions from the current weights",
#               "loss: scores the mistakes with one number",
#               "backward: computes the gradient of the loss for every weight",
#               "step: moves every weight a little downhill"]
# zero_grad_effect = "gradients pile up and training goes wrong"
# why_693 = "At the start the model says about 0.5 for everything, and -ln(0.5) = ln 2 = 0.693, the loss of a coin toss."
# ```
# </details>

# %% [markdown]
# ### 6.2 Capacity sweep: how many hidden units are enough?
#
# We write `train_model` once (the same five-line loop in a function) and train networks with `H = 1, 2, 4, 8, 32` hidden units on both the circles and the moons data sets. For each we record train and test accuracy. The test set is generated with a **different seed**, so the network has never seen those points.

# %%
def train_model(model, Xtr, ytr, Xev, yev, epochs=400, lr=0.05, wd=0.0):
    """Train with Adam. Returns the per-epoch training loss and evaluation loss (on Xev, yev), plus the best state seen."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t, Xev_t, yev_t = (torch.tensor(a) for a in (Xtr, ytr, Xev, yev))
    tr, ev, best = [], [], (1e9, 0, None)
    for epoch in range(epochs):
        opt.zero_grad()
        l = lf(model(Xtr_t).squeeze(1), ytr_t); l.backward(); opt.step()
        tr.append(l.item())
        with torch.no_grad():
            ev.append(lf(model(Xev_t).squeeze(1), yev_t).item())
        if ev[-1] < best[0]:
            best = (ev[-1], epoch, {k: v.clone() for k, v in model.state_dict().items()})
    return tr, ev, best

sweep_epochs = 150 if QUICK else 400
Hs = [1, 2, 4, 8, 32]
sweep = {}
fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
for row, (name, maker) in enumerate([("circles", make_circles), ("moons", make_moons)]):
    Xtr, ytr = maker(200, seed=0)
    Xte, yte = maker(400, seed=1)
    for col, H in enumerate(Hs):
        torch.manual_seed(SEED)
        m = make_mlp(H)
        train_model(m, Xtr, ytr, Xte, yte, epochs=sweep_epochs)
        sweep[(name, H)] = (accuracy(m, Xtr, ytr), accuracy(m, Xte, yte))
        plot_decision(prob_fn(m), Xtr, ytr, ax=axes[row, col], title=f"{name}, H={H}\ntrain {sweep[(name, H)][0]:.2f}  test {sweep[(name, H)][1]:.2f}")
plt.tight_layout(); plt.show()

print(f"{'data':<9}{'H':>4}{'train acc':>11}{'test acc':>10}")
for (name, H), (a_tr, a_te) in sweep.items():
    print(f"{name:<9}{H:>4}{a_tr:>11.3f}{a_te:>10.3f}")
assert sweep[("circles", 32)][1] > sweep[("circles", 1)][1] + 0.1, "more hidden units should clearly help on circles"
RESULTS["p6_sweep_circles_test"] = {H: round(sweep[("circles", H)][1], 3) for H in Hs}
RESULTS["p6_sweep_moons_test"] = {H: round(sweep[("moons", H)][1], 3) for H in Hs}

# %% [markdown]
# **What you should see.** On circles, test accuracy climbs from `0.67` (H = 1) to `0.82` (H = 2) to `0.98` (H = 4), `0.99` (H = 8) and `0.99` (H = 32). On moons, H = 1 and 2 reach about `0.88` and H ≥ 4 `0.99`. So **4 hidden units** is the smallest number that works well on both data sets in our runs; 8 and 32 add very little. One unit draws one cut, two units draw a wedge, and four units can fence in a round shape. Your exact decimals can vary a little with the PyTorch version.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.2 - the smallest network that works
#
# 1. Use the printed table. For each data set, what is the **smallest H** whose test accuracy is at least 0.95? Store them in `smallest_circles` and `smallest_moons`.
# 2. Was your Imagine-first guess (`h_guess`) too small, right or too big? Store `"too small"`, `"right"` or `"too big"` in `guess_verdict`.
# 3. In `capacity_note`, explain in one or two sentences why a network with H = 1 cannot solve circles however long you train it.

# %%
smallest_circles = None   # YOUR ANSWER: 1, 2, 4, 8 or 32
smallest_moons = None     # YOUR ANSWER
guess_verdict = None      # YOUR ANSWER
capacity_note = None      # YOUR ANSWER

# %%
# Check
def smallest_ok(name):
    ok = [H for H in Hs if sweep[(name, H)][1] >= 0.95]
    return min(ok) if ok else None
if smallest_circles is None or smallest_moons is None:
    print("Fill in the answers above, then run this cell again.")
else:
    check("smallest H on circles agrees with the table", smallest_circles == smallest_ok("circles"), "look for the first row with test acc >= 0.95")
    check("smallest H on moons agrees with the table", smallest_moons == smallest_ok("moons"), "look for the first row with test acc >= 0.95")
if guess_verdict is not None:
    expected = "right" if h_guess == smallest_ok("circles") else ("too small" if h_guess < (smallest_ok("circles") or 99) else "too big")
    check("your verdict matches your guess", guess_verdict == expected, f"your guess was {h_guess}, the smallest that works is {smallest_ok('circles')}")
if capacity_note is not None:
    check("your explanation has at least 15 words", n_words(capacity_note) >= 15)
RESULTS["p6_smallest_circles"] = smallest_circles
RESULTS["p6_smallest_moons"] = smallest_moons
RESULTS["p6_capacity_note"] = capacity_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# smallest_circles = 4
# smallest_moons = 4
# guess_verdict = "too small"      # if you guessed 1 or 2
# capacity_note = ("One tanh hidden unit gives one smooth cut of the plane, and a straight cut can never fence in a disc "
#                  "inside a ring, so with 1 unit it must give up part of the data however long it trains.")
# ```
# Capacity is the kind of shapes a network can draw. Too little capacity = underfitting, and no amount of training cures that.
# </details>

# %% [markdown]
# ### 6.3 Overfitting you can see
#
# Now the opposite problem. We give a **big** network (3 hidden layers of 256 units: about 132,000 parameters) a **tiny, noisy** training set: only 30 points of the moons data with a lot of noise (0.35). We also make a validation set (300 points) and a test set (1000 points) from other seeds. After each epoch we record the loss on the training set and on the validation set.
#
# Remember: the network can only learn from the training points. If it learns the *noise* in those 30 points, it will do worse on new points.

# %%
Xtr3, ytr3 = make_moons(30, noise=0.35, seed=3)
Xva3, yva3 = make_moons(300, noise=0.35, seed=4)
Xte3, yte3 = make_moons(1000, noise=0.35, seed=5)

torch.manual_seed(SEED)
big = make_mlp(256, depth=3)
print("parameters in the big network:", f"{sum(p.numel() for p in big.parameters()):,}")
tr3, va3, best3 = train_model(big, Xtr3, ytr3, Xva3, yva3, epochs=600, lr=0.005)
best_epoch = int(np.argmin(va3))
print(f"lowest validation loss {va3[best_epoch]:.3f} at epoch {best_epoch}; at epoch 599 the validation loss is {va3[-1]:.3f}")
print(f"final accuracies: train {accuracy(big, Xtr3, ytr3):.3f}  validation {accuracy(big, Xva3, yva3):.3f}  test {accuracy(big, Xte3, yte3):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(tr3, label="training loss"); axes[0].plot(va3, label="validation loss")
axes[0].axvline(best_epoch, c="k", ls=":", label=f"best epoch = {best_epoch}")
axes[0].set_ylim(0, 2.5); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].set_title("the two curves part ways")
plot_decision(prob_fn(big), Xtr3, ytr3, ax=axes[1], lim=2.5, title="after 600 epochs: a wiggly boundary")
plt.tight_layout(); plt.show()
assert va3[-1] > va3[best_epoch] * 1.5, "validation loss should end clearly above its minimum (overfitting)"
assert tr3[-1] < 0.05, "the big network should memorise the 30 training points"
RESULTS["p6_overfit_best_epoch"] = best_epoch
RESULTS["p6_overfit_test_acc_end"] = round(accuracy(big, Xte3, yte3), 3)

# %% [markdown]
# **What you should see.** The training loss sinks towards zero (it ends below 0.001) and training accuracy is `1.00`. The validation loss reaches its lowest point (about `0.39`) very early, around epoch 23, and then climbs steadily to about `1.6` at epoch 599. The boundary has become a wiggly shape that wraps around single training points. Validation accuracy is `0.77` and test accuracy only `0.74`. That gap between 1.00 on training data and 0.74 on new data is **overfitting**. The dotted line marks the *early-stopping point*: the epoch where the validation loss was lowest.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.3 - read the curves
#
# 1. Store the epoch of the lowest validation loss in `stop_epoch` (use `np.argmin`) and the validation loss at that epoch, rounded to 3 decimals, in `stop_loss`.
# 2. By how many times is the final validation loss bigger than the lowest? Store it rounded to 1 decimal in `blowup`.
# 3. In `overfit_note`, explain in two sentences what the network "learned" after the best epoch, using the words *noise* and *training*.

# %%
stop_epoch = None    # YOUR CODE HERE
stop_loss = None     # YOUR CODE HERE
blowup = None        # YOUR CODE HERE
overfit_note = None  # YOUR ANSWER

# %%
# Check
if stop_epoch is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("stop_epoch is the argmin of the validation loss", stop_epoch == int(np.argmin(va3)), "np.argmin(va3)")
    check("stop_loss matches", stop_loss is not None and abs(stop_loss - min(va3)) < 0.002, "min(va3), rounded to 3 decimals")
    check("blowup is the final loss divided by the lowest loss", blowup is not None and abs(blowup - va3[-1] / min(va3)) < 0.15, "va3[-1] / min(va3)")
if overfit_note is not None:
    check("your note uses 'noise' and 'training'", "noise" in overfit_note.lower() and "training" in overfit_note.lower())
RESULTS["p6_overfit_note"] = overfit_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# stop_epoch = int(np.argmin(va3))
# stop_loss = round(min(va3), 3)
# blowup = round(va3[-1] / min(va3), 1)
# overfit_note = ("After the best epoch the network keeps lowering the training loss by memorising the noise in the 30 training points. "
#                 "That helps the training data but hurts new data, which is why the validation loss goes up.")
# ```
# </details>

# %% [markdown]
# ### 6.4 Fix it: three remedies
#
# We try three fixes on the same problem and compare the **test accuracy**, averaged over 5 different random starting weights so one lucky start does not mislead us:
#
# 1. **More data:** 150 training points instead of 30 (same big network).
# 2. **Smaller network:** only 4 hidden units in one layer (about 17 parameters), plus a little weight decay.
# 3. **Early stopping:** keep the big network, but use the weights from the epoch with the lowest *validation* loss.
#
# The baseline is the big network trained for 600 epochs on 30 points. Note: we choose the stopping epoch with the **validation** set, never the test set.

# %%
def one_run(seed, n_train=30, H=256, depth=3, wd=0.0, epochs=600, lr=0.005, early=False):
    """Train one network (random start `seed`) and return its test accuracy. The 30-point training set is the one from 6.3."""
    Xtr, ytr = make_moons(n_train, noise=0.35, seed=3)       # n_train = 30 gives exactly the set of step 6.3
    torch.manual_seed(seed)
    m = make_mlp(H, depth)
    _, _, best = train_model(m, Xtr, ytr, Xva3, yva3, epochs=epochs, lr=lr, wd=wd)
    if early:
        m.load_state_dict(best[2])            # go back to the weights of the best validation epoch
    return accuracy(m, Xte3, yte3)

n_seeds = 2 if QUICK else 5
recipes = {
    "baseline: 30 points, big network": dict(),
    "1. more data: 150 points":         dict(n_train=150),
    "2. smaller net: H=4 + weight decay": dict(H=4, depth=1, wd=0.01, lr=0.02),
    "3. early stopping (validation)":   dict(early=True),
}
remedy = {}
for name, kw in recipes.items():
    accs = [one_run(s, **kw) for s in range(n_seeds)]
    remedy[name] = float(np.mean(accs))
    print(f"{name:<38} mean test accuracy = {remedy[name]:.3f}   (runs: {', '.join(f'{a:.2f}' for a in accs)})")
RESULTS["p6_remedies"] = {k: round(v, 3) for k, v in remedy.items()}
base = remedy["baseline: 30 points, big network"]
assert max(remedy.values()) > base + 0.02, "at least one remedy should beat the baseline"

# %% [markdown]
# **What you should see.** The baseline averages about `0.75`. Each remedy helps: more data about `0.855`, the small network with weight decay about `0.82` and early stopping about `0.78`. Early stopping helps least here because by the best epoch the network has seen the 30 points only briefly, and 30 noisy points simply do not contain enough information. The best possible accuracy on this noisy data set is only about `0.87` to `0.89` (the two moons overlap). The numbers are averages over 5 random starts on the same 30 points; another training set would give different values.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.4 - choose and justify
#
# Look at the table. Store the name of the remedy with the highest mean test accuracy in `best_remedy` (copy the exact name from the table). Then in `remedy_note` write 2 sentences: which remedy would you choose if you could *not* collect more data, and why? What does each of the other two remedies cost?

# %%
best_remedy = None    # YOUR ANSWER: copy a key from `remedy`
remedy_note = None    # YOUR ANSWER: 2 sentences

# %%
# Check
if best_remedy is None:
    print("Fill in `best_remedy` above, then run this cell again.")
else:
    check("best_remedy is one of the four names", best_remedy in remedy, "copy a name exactly as printed")
    if best_remedy in remedy:
        check("it has the highest mean test accuracy", remedy[best_remedy] >= max(remedy.values()) - 1e-9, "compare the four means")
if remedy_note is not None:
    check("your note has at least 25 words", n_words(remedy_note) >= 25)
RESULTS["p6_best_remedy"] = best_remedy
RESULTS["p6_remedy_note"] = remedy_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_remedy = max(remedy, key=remedy.get)
# remedy_note = ("Without more data I would use early stopping or a smaller network, because both are free: they only change how we train or how big the model is. "
#                "More data is usually the best cure but costs time and money to collect and label, early stopping needs a validation set, "
#                "and a smaller network can underfit if it is made too small.")
# ```
# The winner can change between runs because the differences are only a few percent; saying why is more important than the name.
# </details>

# %% [markdown]
# ### 6.5 A fair test
#
# Why do we need a **validation** set *and* a **test** set? The validation set is used while we build (to choose the number of epochs, hidden units, learning rate). The more we choose by looking at it, the more it quietly becomes part of training. The **test set** is locked away and used once, at the very end, for an honest score. Here is a 60 / 20 / 20 split using shuffled indices, and a leakage check that no example is in two sets.

# %%
Xall, yall = make_moons(500, noise=0.2, seed=11)
n = len(Xall)
perm = np.random.RandomState(SEED).permutation(n)             # shuffled indices 0..n-1
n_train, n_val = int(0.6 * n), int(0.2 * n)
idx_train, idx_val, idx_test = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
print("sizes:", len(idx_train), len(idx_val), len(idx_test))

def leakage(a, b):
    """The indices that appear in both sets (should be empty)."""
    return set(a.tolist()) & set(b.tolist())
assert not leakage(idx_train, idx_val) and not leakage(idx_train, idx_test) and not leakage(idx_val, idx_test), "overlap found!"
assert len(idx_train) + len(idx_val) + len(idx_test) == n
print("no overlap: every example is in exactly one set")

# a leak on purpose
bad_val = np.concatenate([idx_val, idx_train[:5]])
print("5 train examples smuggled into validation -> overlap size:", len(leakage(idx_train, bad_val)))

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.5 - split your own data
#
# 1. Split a data set of **n = 1000** examples 70 / 15 / 15 with shuffled indices. Store the three sizes in `sizes_1000` as a list `[train, val, test]`.
# 2. Write a function `has_leak(a, b)` that returns `True` if two index arrays share any example. Check it on your split and on a deliberately leaky one.
# 3. In `why_three`, explain in two sentences why we do not simply use the test set to pick the best number of epochs.

# %%
sizes_1000 = None   # YOUR ANSWER: [700, 150, 150]
why_three = None    # YOUR ANSWER

def has_leak(a, b):
    return False    # YOUR CODE HERE: return True if a and b share an index

# %%
# Check
p1000 = np.random.RandomState(0).permutation(1000)
a_, b_, c_ = p1000[:700], p1000[700:850], p1000[850:]
if sizes_1000 is None:
    print("Fill in `sizes_1000` above, then run this cell again.")
else:
    check("sizes are 700, 150, 150", list(sizes_1000) == [700, 150, 150], "0.70 * 1000, 0.15 * 1000, and the rest")
check("has_leak says False for a clean split", has_leak(a_, b_) is False and has_leak(a_, c_) is False and has_leak(b_, c_) is False, "compare the two index sets")
check("has_leak says True for a leaky split", has_leak(a_, np.concatenate([b_, a_[:3]])) is True, "use set(a) & set(b) and test whether it is empty")
if why_three is not None:
    check("your explanation has at least 20 words", n_words(why_three) >= 20)
RESULTS["p6_why_three_sets"] = why_three

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# sizes_1000 = [700, 150, 150]
# def has_leak(a, b):
#     return len(set(a.tolist()) & set(b.tolist())) > 0
# why_three = ("If we pick the number of epochs by looking at the test set, the test score is no longer independent: we tuned the model to it. "
#              "We keep a separate validation set for choices and touch the test set once, so the final score is an honest estimate for new data.")
# ```
# </details>

# %% [markdown]
# ## Part 7. Design your own classifier and reflect (25 min)
#
# Now you are the designer. You choose a real problem from your school or community, write the **spec** of a classifier for it, build a small synthetic version, train it with PyTorch, stress-test it with bad labels, collect your results, and reflect. You will use ideas from the whole lesson: supervised learning, loss, neurons, layers, training and generalization.
#
# Five steps: (7.1) the imagination design, (7.2) build and train, (7.3) the stress test, (7.4) your summary, (7.5) four reflection questions.

# %% [markdown]
# ### 7.1 Imagination design: the spec of your classifier
#
# Pick a **yes/no problem** you know. Some ideas: *spam or not spam* text messages; *ripe or unripe* mangoes; *flooded or clear* road; *jeepney crowded or not*; *will the canteen sell out of rice meals today*. Fill in the spec below. It must name **at least two numeric features** (numbers a computer could measure), one label, where the data would come from, how many examples you would need, the network shape (you can use `count_params`), the loss, how you would test it, and one fairness or data risk. A complete example is already filled in; replace every entry with your own idea.

# %%
spec = {
    "problem": "Is this road flooded or clear? (yes = flooded)",                                      # YOUR TURN: replace
    "features": ["rainfall in the last 3 hours (mm)", "water level of the nearest creek (cm)"],       # at least 2 numbers
    "label": "1 = flooded, 0 = clear (decided by a barangay volunteer who checks the road)",
    "data_source": "Volunteers record the two numbers and the road state every 30 minutes during rain for one rainy season.",
    "n_examples": 600,
    "network_shape": [2, 8, 1],
    "loss": "binary cross-entropy (BCE), because the answer is yes/no",
    "how_to_test": "Split by date, not at random: train on June to August, validate on September, test on October, so the test days are really new.",
    "risk": "Only roads near the creek were measured, so the model may fail on roads in other places, and a sensor failure during a storm would produce wrong labels.",
}
print(f"Network {spec['network_shape']} has {count_params(spec['network_shape'])} parameters.")
for k, v in spec.items():
    print(f"  {k}: {v}")

# %%
# Check
required = ["problem", "features", "label", "data_source", "n_examples", "network_shape", "loss", "how_to_test", "risk"]
check("all nine entries are there", all(k in spec and spec[k] not in (None, "", []) for k in required), "fill in: " + ", ".join(k for k in required if not spec.get(k)))
check("at least two features", isinstance(spec["features"], (list, tuple)) and len(spec["features"]) >= 2, "a classifier needs at least two numbers per example")
check("the input layer matches the number of features", spec["network_shape"][0] == len(spec["features"]), "network_shape[0] must equal len(features)")
check("the network ends in one output (a yes/no problem)", spec["network_shape"][-1] == 1, "use 1 output for yes/no")
check("you wrote a risk of at least 12 words", n_words(spec["risk"]) >= 12, "describe who could be treated unfairly or what data could be wrong")
check("you planned at least 100 examples", isinstance(spec["n_examples"], (int, float)) and spec["n_examples"] >= 100)
RESULTS["p7_spec"] = spec

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# The example spec above is one good answer. A second one:
#
# ```python
# spec = {
#     "problem": "Is a jeepney crowded when it arrives at my stop?",
#     "features": ["number of passengers who got on at the last 3 stops", "minutes since the previous jeepney"],
#     "label": "1 = crowded (no seat left), 0 = not crowded; noted by me when I board",
#     "data_source": "I note the two numbers and the label myself on my way to school for four weeks.",
#     "n_examples": 120,
#     "network_shape": [2, 6, 1],
#     "loss": "binary cross-entropy",
#     "how_to_test": "Keep the last week as the test set and never tune on it.",
#     "risk": "My route and my time of day are not typical, so the model could be wrong for other students; I should say so.",
# }
# ```
# </details>

# %% [markdown]
# ### 7.2 Build and train a synthetic version
#
# We do not have real flood or jeepney data, so we **invent** a data set with NumPy: two Gaussian blobs, one for each class. The numbers `mean0`, `mean1` and `spread` below describe your problem: where the typical "clear road" and the typical "flooded road" sit in feature space, and how much they overlap. Make the blobs overlap a little: real data are never perfectly separable. We make 200 points, split them 120 / 40 / 40 (train / validation / test) with shuffled indices, train a small PyTorch network, and plot the decision regions. (Features are scaled to be around -2..2.)

# %%
mean0 = np.array([-0.6, -0.5])    # YOUR TURN: typical feature values for class 0 (e.g. 'clear road')
mean1 = np.array([0.8, 0.7])      # YOUR TURN: typical feature values for class 1 (e.g. 'flooded')
spread = 0.7                      # YOUR TURN: bigger = more overlap = harder problem

def make_my_data(n, seed):
    """n points, half from each blob, as float32 arrays (X: n x 2, y: 0/1)."""
    r = np.random.RandomState(seed)
    m = n // 2
    X = np.concatenate([r.normal(mean0, spread, (m, 2)), r.normal(mean1, spread, (n - m, 2))]).astype(np.float32)
    y = np.concatenate([np.zeros(m), np.ones(n - m)]).astype(np.float32)
    return X, y

X7, y7 = make_my_data(200, seed=7)
perm7 = np.random.RandomState(SEED).permutation(200)
tr_i, va_i, te_i = perm7[:120], perm7[120:160], perm7[160:]
Xtr7, ytr7, Xva7, yva7, Xte7, yte7 = X7[tr_i], y7[tr_i], X7[va_i], y7[va_i], X7[te_i], y7[te_i]
assert not (set(tr_i) & set(va_i)) and not (set(tr_i) & set(te_i)) and not (set(va_i) & set(te_i))

torch.manual_seed(SEED)
net7 = make_mlp(8)
tr7, va7, _ = train_model(net7, Xtr7, ytr7, Xva7, yva7, epochs=200 if QUICK else 300, lr=0.02)
accs7 = {"train": accuracy(net7, Xtr7, ytr7), "validation": accuracy(net7, Xva7, yva7), "test": accuracy(net7, Xte7, yte7)}
print({k: round(v, 3) for k, v in accs7.items()})

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
axes[0].plot(tr7, label="train loss"); axes[0].plot(va7, label="validation loss"); axes[0].legend(); axes[0].set_xlabel("epoch")
plot_decision(prob_fn(net7), Xte7, yte7, ax=axes[1], lim=2.5, title=f"test points: accuracy {accs7['test']:.2f}")
plt.tight_layout(); plt.show()
RESULTS["p7_accuracies"] = {k: round(v, 3) for k, v in accs7.items()}

# %% [markdown]
# **What you should see (with the default blobs).** Training accuracy about `0.92`, validation `0.83` and test `0.95` in our run. These three numbers disagree by more than you might expect because the validation and test sets have only 40 points each, and one wrong point moves the score by 2.5 %. The boundary is a smooth diagonal line between the two clouds with a few points on the wrong side: that overlap is the part no model can fix.

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.2 - improve or explain
#
# 1. Change `spread` to `0.4` and then to `1.2` and re-run the cells of 7.2. How does accuracy change, and why? Put the three (spread, test accuracy) pairs in `spread_table` as a dictionary. (Hint: loop `for s in (0.4, 0.7, 1.2)` over a copy of the code.)
# 2. In `accuracy_note`, explain in two sentences why the test accuracy is lower than the training accuracy, and why a single 40-point test set is a weak measure.

# %%
def accuracy_for_spread(s, seed=0):
    """Train on a data set with spread s and return the test accuracy on 400 fresh points."""
    global spread
    old, spread = spread, s
    Xa, ya = make_my_data(200, seed=7); Xb, yb = make_my_data(400, seed=8)
    spread = old
    torch.manual_seed(seed)
    m = make_mlp(8)
    train_model(m, Xa[:120], ya[:120], Xa[120:160], ya[120:160], epochs=200 if QUICK else 300, lr=0.02)
    return accuracy(m, Xb, yb)

spread_table = {s: round(accuracy_for_spread(s), 3) for s in (0.4, 0.7, 1.2)}     # YOUR TURN: add other values
print("test accuracy by spread:", spread_table)
accuracy_note = None   # YOUR ANSWER: two sentences

# %%
# Check
check("spread_table has at least three entries", len(spread_table) >= 3)
check("more overlap gives lower accuracy", spread_table[0.4] > spread_table[1.2] + 0.05, "wider blobs overlap more, so even the best boundary makes more mistakes")
if accuracy_note is None:
    print("Fill in `accuracy_note` above, then run this cell again.")
else:
    check("your explanation has at least 25 words", n_words(accuracy_note) >= 25)
RESULTS["p7_spread_table"] = spread_table
RESULTS["p7_accuracy_note"] = accuracy_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# accuracy_note = ("The network was fitted to the training points, so it has seen them and can bend around some of their noise, while test points are new. "
#                  "A test set of 40 points is weak because one wrong prediction changes the score by 2.5 percentage points, so the number is mostly luck.")
# ```
# Our values (test accuracy on 400 fresh points): spread 0.4 gives 0.985, 0.7 gives 0.875 and 1.2 gives 0.745.
# </details>

# %% [markdown]
# ### 7.3 Stress test: label noise
#
# Real labels are made by people (or by sensors) and are sometimes wrong. We **flip 10 % of the training labels** (12 of the 120) on purpose, train a larger network that is able to memorise them, and measure the test accuracy on a bigger clean test set of 1000 points. To be fair we repeat with 5 different seeds and average.

# %%
Xbig, ybig = make_my_data(1000, seed=99)          # clean test set for the stress test

def noisy_vs_clean(seed, flip=0.10):
    r = np.random.RandomState(seed)
    Xa, ya = make_my_data(120, seed=seed + 20)
    ynoisy = ya.copy()
    k = int(round(flip * len(ya)))
    flipped = r.choice(len(ya), k, replace=False)
    ynoisy[flipped] = 1 - ynoisy[flipped]
    out = []
    for labels in (ya, ynoisy):
        torch.manual_seed(seed)
        m = make_mlp(32, depth=2)
        train_model(m, Xa, labels, Xbig, ybig, epochs=300, lr=0.01)
        out.append(accuracy(m, Xbig, ybig))
    return out

runs = [noisy_vs_clean(s) for s in range(3 if QUICK else 5)]
clean_acc = float(np.mean([r[0] for r in runs])); noisy_acc = float(np.mean([r[1] for r in runs]))
print("per seed (clean, noisy):", [(round(a, 3), round(b, 3)) for a, b in runs])
print(f"mean test accuracy with clean labels: {clean_acc:.3f}   with 10 % flipped labels: {noisy_acc:.3f}   drop: {clean_acc - noisy_acc:.3f}")
assert clean_acc > 0.8, "the clean model should work"
RESULTS["p7_clean_acc"] = round(clean_acc, 3)
RESULTS["p7_noisy_acc"] = round(noisy_acc, 3)

# %% [markdown]
# **What you should see.** Mean test accuracy with clean labels `0.877` and with 10 % flipped labels `0.810`: a drop of about `0.067` (every one of the 5 seeds lost between 0.06 and 0.08). The network is large enough to memorise the wrong labels, so it learns some of them. Read your own numbers rather than ours.

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.3 - how much noise is too much?
#
# 1. Store the average accuracy drop (clean minus noisy) from the cell above, rounded to 3 decimals, in `drop_10`.
# 2. Try `flip=0.30` for `noisy_vs_clean` (3 seeds are enough) and store the new average drop in `drop_30`. Is it bigger or smaller than `drop_10`?
# 3. In `clean_labels_note`, write one sentence on why clean labels matter even more than a fancy network.

# %%
drop_10 = None             # YOUR CODE HERE: round(clean_acc - noisy_acc, 3)
drop_30 = None             # YOUR CODE HERE: loop over seeds with flip=0.30
clean_labels_note = None   # YOUR ANSWER

# %%
# Check
if drop_10 is None:
    print("Fill in the answers above, then run this cell again.")
else:
    check("drop_10 matches the cell above", abs(drop_10 - (clean_acc - noisy_acc)) < 0.002, "clean_acc - noisy_acc")
    if drop_30 is not None:
        check("30 % noise hurts at least as much as 10 %", drop_30 >= drop_10 - 0.02, "more flipped labels should not hurt less (small differences are noise)")
if clean_labels_note is not None:
    check("your sentence has at least 12 words", n_words(clean_labels_note) >= 12)
RESULTS["p7_drop_10"] = drop_10
RESULTS["p7_drop_30"] = drop_30
RESULTS["p7_clean_labels_note"] = clean_labels_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# drop_10 = round(clean_acc - noisy_acc, 3)
# runs30 = [noisy_vs_clean(s, flip=0.30) for s in range(3)]
# drop_30 = round(float(np.mean([a - b for a, b in runs30])), 3)
# clean_labels_note = "The model learns exactly what the labels say, so wrong labels teach it wrong rules, and no architecture can fix a teacher who is wrong."
# ```
# Expect `drop_30` to be much larger than `drop_10` (about 0.24 on our run).
# </details>

# %% [markdown]
# ### 7.4 Collect your results
#
# Everything you stored in `RESULTS` during the lab (all parts) is printed together here. This is the summary you submit with your answers. If a part of the notebook was skipped, its entries will be missing, so go back and run it.

# %%
def show_value(v, limit=140):
    s = str(v)
    return s if len(s) <= limit else s[:limit] + " ..."

print("=" * 60)
print("MY WEEK 2 RESULTS")
print("=" * 60)
for key in sorted(RESULTS):
    print(f"{key:<30} {show_value(RESULTS[key])}")
print("=" * 60)
print(f"{len(RESULTS)} items collected.")
mine = [k for k in RESULTS if k[:2] in ("p4", "p5", "p6", "p7")]
assert len(mine) >= 20, "Parts 4 to 7 should have stored at least 20 items; run all their cells"

# %% [markdown]
# ### 7.5 Reflection
#
# Answer the four questions below in 2 to 3 sentences each. Be specific: refer to something you actually saw in this notebook (a number, a picture, a table).
#
# 1. What surprised you most this week, and why?
# 2. What does a **hidden layer** actually do? Use the XOR network of step 4.2 as your example.
# 3. When would you trust a model's 95 % accuracy, and when would you not? (Think about the test set, class balance and who is affected.)
# 4. How do **supervised learning** (this week) and the **generative** idea of Week 1 differ, and what do they share?

# %%
reflection = {
    "1_surprised": None,
    "2_hidden_layer": None,
    "3_trust_95": None,
    "4_supervised_vs_generative": None,
}

# %%
# Check
short = [k for k, v in reflection.items() if n_words(v) < 20]
if len(short) == len(reflection):
    print("Write your four answers above (at least 20 words each), then run this cell again.")
else:
    check("all four answers have at least 20 words", not short, "still short: " + ", ".join(short))
RESULTS["p7_reflection"] = reflection
print("\nYour reflection:")
for k, v in reflection.items():
    print(f"  {k}: {v if v else '(not filled in)'}")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reflection = {
#     "1_surprised": "I was surprised that two tiny neurons plus one decider solved XOR, while a single neuron stayed stuck at about 75 percent. Hand-writing the weights made it feel less like magic.",
#     "2_hidden_layer": "A hidden layer re-codes the inputs. In the XOR network, the OR and AND neurons turned the four corners into new coordinates where one straight line separates the classes, which the output neuron then draws.",
#     "3_trust_95": "I would trust 95 percent only if it was measured on a test set the model never saw, with classes that are not very uneven. I would not trust it if 95 percent of examples are one class, or if the test data came from the same noisy source as training.",
#     "4_supervised_vs_generative": "Supervised learning learns from examples that each have a correct label and is scored by a loss against that label, while a generative model learns the distribution of the data itself. Both start from data, both use a loss and gradient descent, and a language model is trained with cross-entropy just like our classifier.",
# }
# ```
# Your answers should come from what you saw. There is no single right answer.
# </details>

# %% [markdown]
# ## Wrap-up
#
# What you learned this week:
#
# - **A layer is a matrix, a network is a stack of layers.** `a = φ(Wx + b)`, parameter counts add up quickly, and without a non-linear activation a deep network collapses into one layer. A hidden layer re-codes the inputs so that the output layer needs only a straight line (XOR).
# - **Training is gradient descent.** Guess, measure the loss, nudge every weight downhill. The learning rate must be small enough to be stable (limit `1/x²` for our one-weight problem), and PyTorch's `loss.backward()` does the bookkeeping you wrote by hand in Part 5.
# - **What matters is how well a model does on new data.** Too small a network underfits, a huge one trained too long overfits, and honest testing needs separate train, validation and test sets and clean labels.
#
# **Looking ahead.** In Week 3 you meet the tools that make all this practical: the PyTorch ecosystem and the data pipeline that feeds a network. The training loop you wrote in step 6.1 is the one you will keep using, and the lesson for Week 2 is your reference for the ideas behind it.
#
# **Before you finish.** Files in `/content` vanish when the Colab session ends. Download anything you want to keep, and use **File → Save a copy in Drive** (or download the notebook with its outputs) so your answers survive. Submit it the way your instructor asks.
#
# **Reflect:** double-click this cell and write two sentences: which part of this lab would you like to explore further, and what would you try?
