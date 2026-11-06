# %% [markdown]
# # Week 10 — Midterm Exam, Part B: practical exam
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept. (On the school's DGX Spark you can instead upload this file to JupyterLab.)
#
# This is Part B of the Midterm exam: **60 of the 100 points, 120 minutes**. Part A (the written exam, 40 points) is taken in the course app. The notebook has five tasks:
#
# | Task | Topic | Points | Minutes |
# |---|---|---|---|
# | 1 | Tokens and language models (Week 6) | 12 | 20 |
# | 2 | Sequences and attention (Week 7) | 14 | 25 |
# | 3 | The Transformer and autoregressive training (Week 8) | 16 | 30 |
# | 4 | LLM families, scaling laws and prompting (Week 9) | 12 | 30 |
# | 5 | Synthesis: short written answers | 6 | 15 |
#
# Your instructor announces the date, the room and how the exam counts towards the course.
#
# ### Rules
#
# - Open book for the course only: you may use the lessons and your own lab notes.
# - No AI assistants, no sharing of code or answers, no internet searches for answers.
# - Run the cells from top to bottom. Replace every `# YOUR CODE HERE` and every `None` with your own work. The starter cells run without errors while they are unfinished.
# - Everything runs on a CPU, nothing is downloaded, and every cell should finish in under 90 seconds. Random seeds are fixed at 42.
# - Keep an eye on the clock. If a task is taking too long, write what you have, and move on.
#
# ### How the self-checks work
#
# A **Self-check** cell prints ✓ or ✗ for each numerical answer. It never prints the right value or a hint, and a ✓ is not a promise of full marks: your instructor marks your code, your numbers and your written explanations. Written answers go in the `ANSWERS` dictionary (one string per question) or in a markdown cell that begins "Your answer:".
#
# ### What to submit
#
# Submit this notebook **with all outputs visible** (run everything from the top just before you submit), together with the output of the last cell, which prints your `ANSWERS`. Submit it the way your instructor asks.
#
# <details><summary>How you are marked (the marking rules)</summary>
#
# - Each sub-question B*n*.*m* shows its points. A numerical answer earns its points when the value is right (small rounding differences are fine). If the value is wrong but your code or working shows the correct method, you can still earn part of the points.
# - Code questions are marked on what the code does, not on how it looks. A hard-coded number with no working earns at most half of the points.
# - Written answers are marked on whether they are correct, use the right terms from the lessons and refer to your own numbers. Short answers are fine: say what is true and why.
# - A ✓ in a self-check does not replace the marking, and a ✗ does not remove method points.
# - No points are removed for a wrong answer; leaving a question empty earns 0 for it.
# </details>

# %% [markdown]
# ## Setup
#
# Run this cell once. It loads the libraries, fixes the random seed and defines the helper functions you used in the labs of Weeks 6 to 9: `words_of`, `make_toy_corpus`, `cosine`, `nearest` and the texts `SCHOOL_TEXT`, `SCHOOL_TEST` (Week 6); `make_memory_task`, `one_hot_batch`, `train_seq_model`, `softmax_np`, `show_attention` (Week 7); `TOY_TEXT`, `PATTERN_TEXT`, `show_matrix`, `sample_lm` (Week 8); `make_grammar_text`, `CharData`, `TinyLM`, `param_count`, `train_lm`, `eval_lm` (Week 9). In this exam `check(name, ok)` only tells you whether a result is right (✓ or ✗); it gives **no hints**. Everything runs on a CPU and **nothing is downloaded**.

# %%
import os, gc, math, random, re, time, json, warnings
from collections import Counter, defaultdict
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

ANSWERS = {}   # keep your final answers here; the last cell prints them so you can copy them into your submission

def check(name, ok, hint=""):
    """Exam version: prints a tick or a cross and never a hint."""
    print(("✓ " if ok else "✗ ") + name)

SCHOOL_TEXT = """
The robotics club meets every Friday in the science room. The club has twelve students and two teachers. The students build small robots and the robots follow a black line on the floor. The teachers help the students write code and the students test the code on the robots. Every robot has a motor and a sensor and a small battery. The sensor reads the black line and the motor turns the wheels. The club shares one laptop and the students take turns on the laptop. When a robot fails the students look at the code and the teachers look at the battery. The students learn from every failure and the club gets better every week. In March the club will join a contest in the city. The contest has a track and a timer and a judge. The best robot follows the line without a mistake. The students practise on the track after class and the teachers bring snacks. The robots are not perfect but the students are proud of the robots. After the contest the club will show the robots to the whole school. The younger students will ask questions and the club will answer every question.
"""
SCHOOL_TEST = """
The club meets in the science room on Friday. The students test the robots on the track and the teachers look at the code. The robot follows the line and the sensor reads the floor. The students are proud of the club.
"""

def words_of(text):
    """Lower-case words of a text, with each full stop kept as its own token '.'."""
    return re.findall(r"[a-z']+|\.", text.lower())

_ANIMALS = ["cat", "dog", "mouse", "bird", "rabbit", "fox"]
_DRINKS = ["tea", "coffee", "milk", "juice", "water", "soda"]
_ROYALS = ["king", "queen", "prince", "princess", "duke", "duchess"]
_PLACES = ["kitchen", "garden", "market", "school", "castle", "kingdom"]

def make_toy_corpus(n=3000, seed=0):
    """A list of n short sentences (lists of words) built from templates about animals, drinks, royalty and places. Words of the same group
    appear in the same kinds of sentences, so their embeddings should end up close together."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        k = r.randrange(5)
        if k == 0:   s = ["the", r.choice(_ANIMALS), r.choice(["chased", "saw", "followed"]), "the", r.choice(_ANIMALS)]
        elif k == 1: s = ["the", r.choice(_ANIMALS), r.choice(["slept", "ran", "played"]), "in", "the", r.choice(["garden", "kitchen", "market"])]
        elif k == 2: s = [r.choice(["she", "he"]), r.choice(["drank", "bought", "liked"]), r.choice(_DRINKS), "in", "the", r.choice(["kitchen", "market", "school"])]
        elif k == 3: s = ["the", r.choice(_ROYALS), r.choice(["ruled", "visited", "protected"]), "the", r.choice(["kingdom", "castle"])]
        else:        s = ["the", r.choice(_ROYALS), "lived", "in", "the", "castle", "and", "drank", r.choice(_DRINKS)]
        out.append(s)
    return out

def cosine(a, b):
    """Cosine similarity of two vectors."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def nearest(E, vocab, word, k=5):
    """The k words whose vectors (rows of E, one per word in `vocab`) are most cosine-similar to `word`."""
    i = vocab.index(word)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sims = En @ En[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    return [(vocab[j], round(float(sims[j]), 3)) for j in order]

N_SYMBOLS = 8   # the memory task uses the symbols 0..7

def make_memory_task(n, length, seed=0, n_symbols=N_SYMBOLS):
    """n sequences of `length` random symbols in 0..n_symbols-1. The label of a sequence is its FIRST symbol, so to answer at the end
    the model must remember symbol 1 across `length - 1` more symbols. Returns X (n, length) int64 and y (n,) int64."""
    r = np.random.RandomState(seed)
    X = r.randint(0, n_symbols, size=(n, length)).astype(np.int64)
    return X, X[:, 0].copy()

def one_hot_batch(X, n_symbols=N_SYMBOLS):
    """(n, length) integer symbols -> (n, length, n_symbols) float32 one-hot vectors."""
    return F.one_hot(torch.as_tensor(X), n_symbols).float()

def train_seq_model(model, Xtr, ytr, Xte, yte, epochs=200, lr=0.01, batch=64, seed=0, clip=1.0, verbose=False):
    """Train a model that maps (batch, length, n_symbols) to class scores (batch, n_symbols) with cross-entropy and Adam.
    Returns a dict with per-epoch lists loss, train_acc, test_acc. Gradients are clipped to norm `clip`."""
    g = torch.Generator().manual_seed(seed)
    Xtr, ytr, Xte, yte = one_hot_batch(Xtr), torch.as_tensor(ytr), one_hot_batch(Xte), torch.as_tensor(yte)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0
        for b in range(0, len(Xtr), batch):
            idx = perm[b:b + batch]
            opt.zero_grad(); loss = F.cross_entropy(model(Xtr[idx]), ytr[idx]); loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            hist["loss"].append(tot / len(Xtr))
            hist["train_acc"].append(float((model(Xtr).argmax(1) == ytr).float().mean()))
            hist["test_acc"].append(float((model(Xte).argmax(1) == yte).float().mean()))
        if verbose and (ep % 20 == 0 or ep == epochs - 1): print(f"epoch {ep:3d}  loss {hist['loss'][-1]:.3f}  train {hist['train_acc'][-1]:.2f}  test {hist['test_acc'][-1]:.2f}")
    return hist

def softmax_np(z, axis=-1):
    """Softmax of a NumPy array along an axis (numerically stable)."""
    z = np.asarray(z, dtype=float); e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def show_attention(W, row_labels=None, col_labels=None, title="", ax=None):
    """Draw an attention-weight matrix W (rows = queries, columns = keys) as a heat map with the numbers in the cells."""
    W = np.asarray(W); ax = ax or plt.gca()
    im = ax.imshow(W, cmap="Greens", vmin=0, vmax=max(1e-9, W.max()))
    ax.set_xticks(range(W.shape[1])); ax.set_yticks(range(W.shape[0]))
    ax.set_xticklabels(col_labels or range(W.shape[1]), rotation=45, ha="right"); ax.set_yticklabels(row_labels or range(W.shape[0]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]): ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    return im

TOY_TEXT = (
    "the robotics club meets every friday in the science room. the students build small robots and the robots follow a black line. "
    "the teachers help the students write code and the students test the code on the robots. every robot has a motor and a sensor and a battery. "
    "the sensor reads the line and the motor turns the wheels. when a robot fails the students look at the code and the teachers look at the battery. "
    "the club learns from every failure and gets better every week. "
)
PATTERN_TEXT = "the cat sat on the mat. the dog sat on the log. " * 20

@torch.no_grad()
def sample_lm(model, data, prompt, n_new=100, block_size=32, temperature=1.0, top_k=None, seed=0):
    """Generate n_new characters after `prompt`, one at a time, feeding the model its own output (autoregressive generation)."""
    g = torch.Generator().manual_seed(seed); model.eval()
    ids = torch.tensor([data.encode(prompt)], dtype=torch.long)
    for _ in range(n_new):
        logits = model(ids[:, -block_size:])[:, -1, :] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
    model.train()
    return data.decode(ids[0])

def show_matrix(M, title="", ax=None, row_labels=None, col_labels=None, fmt="{:.2f}", cmap="RdBu_r"):
    """Draw a matrix as a heat map with the numbers written in the cells."""
    M = np.asarray(M); ax = ax or plt.gca()
    lim = max(abs(M).max(), 1e-9)
    ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    if row_labels is not None: ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_labels)
    if col_labels is not None: ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if M.size <= 150:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title(title)

_ADJ = ["red", "tall", "quiet", "old"]; _NOUN = ["robot", "student", "teacher", "river", "market", "kite"]
_VERB = ["sees", "builds", "finds", "follows"]; _PREP = ["near", "behind", "beside"]
GRAMMAR_BITS = math.log2(len(_ADJ) * len(_NOUN) * len(_VERB) * len(_PREP) * len(_NOUN))   # information per sentence, in bits

def make_grammar_text(n_sentences, seed=0):
    """n_sentences like 'the red robot sees the kite near the river.' joined by spaces, plus the exact average entropy in nats per character.
    Returns (text, nats_per_char). No model can do better than this loss on text from the grammar."""
    r = random.Random(seed); out = []
    for _ in range(n_sentences):
        out.append(f"the {r.choice(_ADJ)} {r.choice(_NOUN)} {r.choice(_VERB)} the {r.choice(_NOUN)} {r.choice(_PREP)} the {r.choice(_NOUN)}.")
    text = " ".join(out) + " "
    avg_len = len(text) / n_sentences
    return text, GRAMMAR_BITS * math.log(2) / avg_len

class CharData:
    """A character-level tokenizer and data set (see Week 8). `text` becomes a tensor of integer ids."""
    def __init__(self, text, chars=None):
        self.chars = chars or sorted(set(text)); self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}; self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, ids): return "".join(self.itos[int(i)] for i in ids)
    def get_batch(self, batch_size, block_size, gen=None):
        ix = torch.randint(0, len(self.data) - block_size - 1, (batch_size,), generator=gen)
        return (torch.stack([self.data[i:i + block_size] for i in ix]), torch.stack([self.data[i + 1:i + block_size + 1] for i in ix]))

class TinyLM(nn.Module):
    """A small decoder-only (GPT-style) character model: token + learned position embeddings, `layers` pre-norm blocks of causal multi-head attention
    and a feed-forward network (d_ff = 4·d), a final layer norm and an output layer tied to the embedding table."""
    def __init__(self, vocab, d=32, heads=2, layers=2, block=32):
        super().__init__()
        self.block, self.heads = block, heads
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d), ln2=nn.LayerNorm(d),
                                                         ff=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            q, k, v = b["qkv"](b["ln1"](x)).split(x.shape[-1], dim=-1)
            q, k, v = [t.view(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v)]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
            x = x + b["proj"](a)
            x = x + b["ff"](b["ln2"](x))
        return self.lnf(x) @ self.tok.weight.T

def param_count(model):
    """Number of trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_lm(model, data, steps=300, batch_size=32, block_size=32, lr=3e-3, seed=0, verbose=False):
    """Train a next-token model (model(x) -> logits (batch, block, vocab)) with cross-entropy and AdamW. Returns the list of step losses."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x, y = data.get_batch(batch_size, block_size, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss))
        if verbose and (step % 100 == 0 or step == steps - 1): print(f"step {step:4d}  loss {losses[-1]:.3f}")
    return losses

@torch.no_grad()
def eval_lm(model, data, block_size=32, n_batches=20, seed=123):
    """Average next-token cross-entropy (nats per character) of `model` on random windows of `data`."""
    gen = torch.Generator().manual_seed(seed); model.eval(); tot = 0.0
    for _ in range(n_batches):
        x, y = data.get_batch(32, block_size, gen)
        tot += float(F.cross_entropy(model(x).reshape(-1, model.tok.num_embeddings), y.reshape(-1)))
    model.train()
    return tot / n_batches

# %% [markdown]
# ### Exam helpers
#
# One more small cell: it defines `same(...)` and `same_array(...)`, which the self-checks use to compare your number with the expected one without ever showing it (the notebook stores only fingerprints), `same_word(...)` for a word, and `n_words(...)`, which counts the words of a written answer.

# %%
import hashlib
H1 = "w10-b1"   # a label that is mixed into every fingerprint
def _fp(v, nd):
    return hashlib.sha256(f"{H1}|{nd}|{round(float(v), nd) + 0.0:.{nd}f}".encode()).hexdigest()[:12]
def same(x, fingerprint, nd=3):
    """True if x, rounded to nd decimals, is the number the exam expects. The notebook stores only a fingerprint, so it cannot tell you the number."""
    try:
        x = float(x)
    except Exception:
        return False
    if not math.isfinite(x):
        return False
    step = 10.0 ** -nd
    return any(_fp(x + d * 0.4 * step, nd) == fingerprint for d in (-1, 0, 1))
def same_array(a, fingerprint, nd=3, shape=None):
    """Like same(), for a whole list or array: it compares one number that mixes all the entries (and, if given, the shape)."""
    try:
        a = np.asarray(a, dtype=float)
        if shape is not None and a.shape != tuple(shape):
            return False
        return same(float(np.sum(a * np.cos(np.arange(a.size)).reshape(a.shape))), fingerprint, nd)
    except Exception:
        return False
def same_word(s, fingerprint):
    return isinstance(s, str) and hashlib.sha256((H1 + "|" + s.strip().lower()).encode()).hexdigest()[:12] == fingerprint
def n_words(x):
    """How many words are in x (0 if x is not text)."""
    return len(x.split()) if isinstance(x, str) else 0
assert same(0.5004, _fp(0.5, 3)) and not same(0.52, _fp(0.5, 3))   # the helper itself works
print("Exam helpers ready.")

# %% [markdown]
# ## Task 1. Tokens and language models (12 points, 20 minutes)
#
# All the numbers in this task are invented for the exam. A school's ferry club keeps a log, and a bigram language model is trained on the 30-word text below (the full stop counts as a token, as in the labs). A full stop is added at the very start, so the first word also has a previous token.
#
# *Exercise B1.1:* **Counting bigrams (2 points).** Write `bigram_counts(tokens)`, which returns a dictionary from a previous token to a `Counter` of the tokens that followed it. Then store in `V` the number of different tokens, including the full stop (0.5 points), in `p_fish_after_the` the unsmoothed probability of "fish" after "the" (1 point), and in `most_common_after_the` the word that most often follows "the" (0.5 points).

# %%
TEXT1 = ("The ferry leaves the port at six . The ferry carries the fish to the market . "
         "The vendor sells the fish at the market . The buyer buys the fish and rice .")
tokens = ["."] + words_of(TEXT1)

def bigram_counts(tokens):
    """Return {previous token: Counter of next tokens}."""
    counts = defaultdict(Counter)
    # YOUR CODE HERE
    return counts

counts = bigram_counts(tokens)
V = None                      # YOUR CODE HERE
p_fish_after_the = None       # YOUR CODE HERE
most_common_after_the = None  # YOUR CODE HERE (a word)

# %%
# Self-check B1.1
check("B1.1 V (0.5 points)", same(V, "af3c229994d6", 0))
check("B1.1 p_fish_after_the (1 point)", same(p_fish_after_the, "faadf7517131", 3))
check("B1.1 most_common_after_the (0.5 points)", same_word(most_common_after_the, "ad48cabf6617"))

# %% [markdown]
# *Exercise B1.2:* **Add-one smoothing and perplexity (3 points).** The test sentence below has a pair that the training text never contained. (a) Store in `p_test_raw` the probability of the whole test sentence under the unsmoothed model, counting the six transitions that start from the leading full stop (0.5 points). (b) Store in `p_buys_after_vendor` the add-one smoothed probability of "buys" after "vendor", using your `V` (1 point). (c) Store in `ppl_smooth` the perplexity of the test sentence under the add-one model (1.5 points). In `ANSWERS["B1.2"]` add one sentence on why the perplexity of the unsmoothed model is not a useful number here.

# %%
test = [".", "the", "vendor", "buys", "the", "fish", "."]

p_test_raw = None             # YOUR CODE HERE
p_buys_after_vendor = None    # YOUR CODE HERE
ppl_smooth = None             # YOUR CODE HERE
ANSWERS["B1.2"] = ""          # YOUR ANSWER: one sentence

# %%
# Self-check B1.2
check("B1.2 p_test_raw (0.5 points)", same(p_test_raw, "62006ee3446a", 4))
check("B1.2 p_buys_after_vendor (1 point)", same(p_buys_after_vendor, "0b987dc60959", 4))
check("B1.2 ppl_smooth (1.5 points)", same(ppl_smooth, "d5cea8339e95", 2))
check("B1.2 written sentence present", n_words(ANSWERS.get("B1.2")) >= 8)

# %% [markdown]
# *Exercise B1.3:* **A BPE merge trace (3 points).** Below is a small word list with counts. Every word is written as its letters plus the end marker `</w>`. Complete `learn_bpe(corpus, n_merges)` so that it repeats: count every adjacent pair of symbols (weighted by the count of its word), take the most frequent pair, record it, and merge it in all the words. **If two pairs tie, take the one that was seen first** when you scan the words in the order of the list, left to right. Learn **6** merges. Store in `merge_string` the merges in order, written as `a+b` and separated by single spaces (for example `t+h h+e`) (2 points), and in `total_tokens` the number of symbols in the whole corpus after the 6 merges, counting each word as many times as its count (1 point).

# %%
CORPUS = [("rain", 6), ("train", 4), ("main", 3), ("mango", 5), ("range", 2)]

def merge_symbols(seq, a, b):
    """Replace every adjacent pair (a, b) in the tuple seq by the single symbol a + b."""
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(seq[i]); i += 1
    return tuple(out)

def learn_bpe(corpus, n_merges):
    """Return (merges, words): the list of merges [(a, b), ...] in order, and the final symbols of each word."""
    words = {w: tuple(w) + ("</w>",) for w, _ in corpus}
    merges = []
    for _ in range(n_merges):
        pass   # YOUR CODE HERE: count the pairs, pick the best one, record it, merge it in every word
    return merges, words

merges, final_words = learn_bpe(CORPUS, 6)
merge_string = None   # YOUR CODE HERE, for example "t+h h+e"
total_tokens = None   # YOUR CODE HERE
print(merges)

# %%
# Self-check B1.3
check("B1.3 merge_string (2 points)", same_word(merge_string, "8921287779bc"))
check("B1.3 total_tokens (1 point)", same(total_tokens, "99994d827cc1", 0))

# %% [markdown]
# *Exercise B1.4:* **Encode new words (1.5 points).** Write `encode_bpe(word, merges)`, which splits a word you have never seen into pieces by applying the merges in the order they were learned. Store in `pieces_drain` the pieces of "drain" joined by single spaces (0.5 points), and in `tokens_per_word` the mean number of pieces per word for the three words in `NEW_WORDS` (1 point).

# %%
NEW_WORDS = ["drain", "mangoes", "strain"]

def encode_bpe(word, merges):
    """Split a word into BPE pieces (the end marker </w> is part of the last piece)."""
    seq = tuple(word) + ("</w>",)
    # YOUR CODE HERE
    return list(seq)

pieces_drain = None      # YOUR CODE HERE (a string)
tokens_per_word = None   # YOUR CODE HERE

# %%
# Self-check B1.4
check("B1.4 pieces_drain (0.5 points)", same_word(pieces_drain, "4266c3093d11"))
check("B1.4 tokens_per_word (1 point)", same(tokens_per_word, "efb98ecbed9d", 3))

# %% [markdown]
# *Exercise B1.5:* **An embedding table (1.5 points).** Each word has a 3-number embedding, taken from the dictionary below. Store in `cos_mango_banana` the cosine similarity of "mango" and "banana" (0.5 points); in `closest_to_jeepney` the other word whose vector has the highest cosine similarity with "jeepney" (0.5 points); and in `emb_params` the number of parameters of an embedding table for a vocabulary of 50,000 tokens with 768 numbers per token (0.5 points). You may use `cosine`.

# %%
EMB = {"mango": [0.8, 0.1, 0.6], "banana": [0.7, 0.2, 0.5], "jeepney": [-0.2, 0.9, 0.1],
       "bus": [-0.1, 0.8, 0.3], "rice": [0.5, -0.4, 0.7]}

cos_mango_banana = None     # YOUR CODE HERE
closest_to_jeepney = None   # YOUR CODE HERE (a word)
emb_params = None           # YOUR CODE HERE

# %%
# Self-check B1.5
check("B1.5 cos_mango_banana (0.5 points)", same(cos_mango_banana, "a36e1ef5b51e", 3))
check("B1.5 closest_to_jeepney (0.5 points)", same_word(closest_to_jeepney, "eeff12645195"))
check("B1.5 emb_params (0.5 points)", same(emb_params, "5496ce2f315a", 0))

# %% [markdown]
# *Exercise B1.6:* **Which tokenizer? (1 point).** A team builds a model for a community language with many word forms (a root with several prefixes and endings). Tokenizer A has a word-level vocabulary of 30,000 words: a typical 1,000-word text becomes 1,000 tokens, of which about 60 are the unknown token. Tokenizer B is a BPE tokenizer with a vocabulary of 8,000 pieces: the same text becomes about 1,500 tokens and has no unknown token. In `ANSWERS["B1.6"]` write 3 to 4 sentences: which tokenizer would you choose and why, and what each choice costs, with one number for the embedding table at 512 numbers per token and one for the sequence length.

# %%
ANSWERS["B1.6"] = ""   # YOUR ANSWER: 3 to 4 sentences

# %%
check("B1.6 answer written", n_words(ANSWERS.get("B1.6")) >= 25)

# %% [markdown]
# ## Task 2. Sequences and attention (14 points, 25 minutes)
#
# *Exercise B2.1:* **Vanishing gradient through time (3 points).** In a one-number RNN, h_t = tanh(w·h_(t-1) + u·x_t), the error signal is multiplied at every step back by the factor f = w·(1 − h²). Suppose w = 0.9 and that the hidden value stays near h = 0.6. Store the factor in `f` (0.5 points), the product of the factors over 12 steps in `g12` (0.5 points), and the smallest number of steps `L_min` for which the product falls below 0.0001 (1 point). In `ANSWERS["B2.1"]` write 2 to 3 sentences: what happens to the product over many steps if |f| is below 1, what if |f| is above 1 (try w = 1.8 with the same h), and one remedy for the second case (1 point).

# %%
w, h_const = 0.9, 0.6
f = None       # YOUR CODE HERE
g12 = None     # YOUR CODE HERE
L_min = None   # YOUR CODE HERE (a whole number)
ANSWERS["B2.1"] = ""   # YOUR ANSWER: 2 to 3 sentences

# %%
# Self-check B2.1
check("B2.1 f (0.5 points)", same(f, "e0abf3197611", 4))
check("B2.1 g12 (0.5 points)", same(g12, "970b6e88d2c9", 5))
check("B2.1 L_min (1 point)", same(L_min, "af3c229994d6", 0))
check("B2.1 written answer present", n_words(ANSWERS.get("B2.1")) >= 15)

# %% [markdown]
# *Exercise B2.2:* **The LSTM cell highway (2 points).** Along the cell state of an LSTM the signal is multiplied at each step back by the forget gate. The forget gates of six steps are in `forget`. Store their product in `cell_grad` (1 point). Then find the constant forget gate value `f_const` for which the product over 40 steps is exactly 0.5 (1 point).

# %%
forget = np.array([0.97, 0.92, 0.99, 0.85, 0.95, 0.98])
cell_grad = None   # YOUR CODE HERE
f_const = None     # YOUR CODE HERE

# %%
# Self-check B2.2
check("B2.2 cell_grad (1 point)", same(cell_grad, "efcaa5e33d8b", 4))
check("B2.2 f_const (1 point)", same(f_const, "398210f776b0", 4))

# %% [markdown]
# *Exercise B2.3:* **Attention by hand (4 points).** One query attends over three keys with d_k = 4. Compute (a) the raw scores q·k for the three keys, in `scores` (1 point); (b) the attention weights when the scores are divided by √d_k and passed through softmax, in `w_scaled` (1 point); (c) the output (the weighted sum of the rows of `Vals`) for those weights, in `out_scaled` (1 point); (d) the largest attention weight when the scores are **not** divided by √d_k, in `w_unscaled_max` (0.5 points). In `ANSWERS["B2.3"]` explain in one or two sentences why the division matters when d_k is large (0.5 points). You may use `softmax_np`.

# %%
d_k = 4
q = np.array([1.0, 0.5, -1.0, 2.0])
K = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 2.0, -1.0, 0.5], [2.0, 1.0, 0.0, -1.0]])
Vals = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
scores = None           # YOUR CODE HERE (3 numbers)
w_scaled = None         # YOUR CODE HERE (3 numbers)
out_scaled = None       # YOUR CODE HERE (2 numbers)
w_unscaled_max = None   # YOUR CODE HERE
ANSWERS["B2.3"] = ""    # YOUR ANSWER: one or two sentences

# %%
# Self-check B2.3
check("B2.3 scores (1 point)", same_array(scores, "5b1535ff5228", 3, shape=(3,)))
check("B2.3 w_scaled (1 point)", same_array(w_scaled, "868373bc44c2", 3, shape=(3,)))
check("B2.3 out_scaled (1 point)", same_array(out_scaled, "214ac2dcc061", 3, shape=(2,)))
check("B2.3 w_unscaled_max (0.5 points)", same(w_unscaled_max, "8e8fe684755d", 3))
check("B2.3 written answer present", n_words(ANSWERS.get("B2.3")) >= 10)

# %% [markdown]
# *Exercise B2.4:* **Soft against hard attention, with autograd (2 points).** Three scores are in `s` and the three values they pick from are in `vals`. (a) Soft attention: compute `softmax(s)` times `vals`, summed, call `backward()` and store the gradient of the output with respect to the **first** score in `grad_soft_s0` (1 point). (b) Hard attention: take the value at the position of the highest score (`argmax`), call `backward()` on it, and store in `hard_grad_exists` whether `s_hard.grad` is not None (True or False) (0.5 points). In `ANSWERS["B2.4"]` add one sentence on what (b) means for training (0.5 points).

# %%
s_vals = [1.0, 2.0, 0.5]
vals_list = [4.0, -1.0, 2.0]
s = torch.tensor(s_vals, requires_grad=True)
vals = torch.tensor(vals_list, requires_grad=True)
s_hard = torch.tensor(s_vals, requires_grad=True)
vals_hard = torch.tensor(vals_list, requires_grad=True)

grad_soft_s0 = None      # YOUR CODE HERE
hard_grad_exists = None  # YOUR CODE HERE (True or False)
ANSWERS["B2.4"] = ""     # YOUR ANSWER: one sentence

# %%
# Self-check B2.4
check("B2.4 grad_soft_s0 (1 point)", same(grad_soft_s0, "2bce71827559", 4))
check("B2.4 hard_grad_exists (0.5 points)", same(float(hard_grad_exists) if hard_grad_exists is not None else None, "9e192d465e0a", 0))
check("B2.4 written answer present", n_words(ANSWERS.get("B2.4")) >= 8)

# %% [markdown]
# *Exercise B2.5:* **A memory experiment (3 points).** The memory task asks a model to output the first of a sequence of random symbols, so it must remember it for `length - 1` steps. `run_memory(kind, length)` below trains a model (`"rnn"` or `"lstm"`, 32 units) for 50 epochs and returns its test accuracy; chance is 0.125. (a) Fill the dictionary `acc` with the four accuracies `acc["rnn8"]`, `acc["lstm8"]`, `acc["rnn60"]` and `acc["lstm60"]` (the number is the sequence length) (1 point; this takes about 30 seconds). (b) Write `first_last_grad(kind)`, the mean gradient norm of the last output with respect to the input at the first step and at the last step in a **fresh, untrained** network (the starter builds the network and the input). Store `decades_rnn` and `decades_lstm`, each the value of −log10(first / last), that is, by how many powers of ten the signal from the first step is weaker than that of the last step (0.5 points). (c) In `ANSWERS["B2.5"]` write 4 to 6 sentences that interpret your own four accuracies and your two numbers: what do they show about remembering over a long distance, about the RNN and about the LSTM, and what could you change to help (1.5 points).

# %%
class SeqClassifier(nn.Module):
    """One-hot symbols -> nn.RNN or nn.LSTM -> the LAST output -> a linear layer with 8 class scores."""
    def __init__(self, kind="rnn", hidden=32):
        super().__init__()
        self.rnn = (nn.RNN if kind == "rnn" else nn.LSTM)(N_SYMBOLS, hidden, batch_first=True)
        self.out = nn.Linear(hidden, N_SYMBOLS)
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.out(o[:, -1])

def run_memory(kind, length, epochs=50, lr=0.003):
    """Train on 800 sequences, return the test accuracy on 300 other ones."""
    torch.manual_seed(0)
    Xtr, ytr = make_memory_task(800, length, seed=1)
    Xte, yte = make_memory_task(300, length, seed=2)
    hist = train_seq_model(SeqClassifier(kind), Xtr, ytr, Xte, yte, epochs=epochs, lr=lr, seed=0)
    return hist["test_acc"][-1]

acc = {}   # YOUR CODE HERE: acc["rnn8"] = run_memory("rnn", 8) and so on for the four runs
print(acc)

# %%
def first_last_grad(kind, T=40, n=20):
    """Mean over n random sequences of the gradient norm of the last output (summed) with respect to the input at step 1 and at step T."""
    torch.manual_seed(0)
    cell = (nn.RNN if kind == "rnn" else nn.LSTM)(8, 32, batch_first=True)
    x = torch.randn(n, T, 8, requires_grad=True)
    # YOUR CODE HERE
    return None, None   # (gradient norm at step 1, gradient norm at step T)

decades_rnn = None    # YOUR CODE HERE
decades_lstm = None   # YOUR CODE HERE
ANSWERS["B2.5"] = ""  # YOUR ANSWER: 4 to 6 sentences

# %%
# Self-check B2.5
ok4 = isinstance(acc, dict) and all(k in acc and acc[k] is not None for k in ("rnn8", "lstm8", "rnn60", "lstm60"))
check("B2.5 four accuracies present (0.5 points)", ok4)
check("B2.5 the RNN result at length 8 (0.25 points)", ok4 and same(float(acc["rnn8"] > 0.9), "449d211481e2", 0))
check("B2.5 the RNN result at length 60 (0.25 points)", ok4 and same(float(acc["rnn60"] < 0.3), "449d211481e2", 0))
check("B2.5 decades_rnn (0.25 points)", same(decades_rnn, "869a236b9c1e", 0))
check("B2.5 decades_lstm (0.25 points)", same(decades_lstm, "1c71b1eca352", 0))
check("B2.5 interpretation present", n_words(ANSWERS.get("B2.5")) >= 40)

# %% [markdown]
# ## Task 3 — The Transformer and autoregressive training (16 points, 30 minutes)
#
# Week 8 put the Transformer together: position signals, multi-head attention, layer normalisation, the causal mask, shifted targets and the count of parameters. In B3.1 to B3.5 you compute with those pieces on small numbers, in B3.6 you train two small GPT-style models and compare them, and in B3.7 you count the parameters of a whole configuration in two ways. All numbers in this task are invented for the exam.
#
# First run the small helper cell below once. It defines `same2(...)`, `match2(...)`, `word2(...)` and `written2(...)`, which the self-checks of Tasks 3 to 5 use to compare your result with the expected one without ever showing it.

# %%
import hashlib
def _fp2(v, nd):
    return hashlib.sha256(f"w10b|{nd}|{round(float(v), nd):.{nd}f}".encode()).hexdigest()[:12]
def same2(x, fingerprint, nd=3):
    """True if the number x, rounded to nd decimals, is the number the exam expects (the notebook stores only a fingerprint)."""
    try:
        x = float(x)
    except Exception:
        return False
    if not math.isfinite(x):
        return False
    step = 10.0 ** -nd
    return any(_fp2(x + d * 0.4 * step, nd) == fingerprint for d in (-1, 0, 1))
def match2(x, fingerprints, nd=3, shape=None):
    """True if the array x has the given shape and every entry matches its fingerprint (entries read in row order)."""
    try:
        a = np.asarray(x, dtype=float)
        if shape is not None and a.shape != tuple(shape):
            return False
        return a.size == len(fingerprints) and all(same2(v, f, nd) for v, f in zip(a.ravel(), fingerprints))
    except Exception:
        return False
def word2(s, fingerprint):
    return isinstance(s, str) and hashlib.sha256(("w10b|" + s.strip().lower()).encode()).hexdigest()[:12] == fingerprint
def written2(s, n=60):
    """True when a written answer has at least n characters (your instructor reads and marks the text; this only tells you that something is there)."""
    return isinstance(s, str) and len(s.strip()) >= n
assert same2(0.5004, _fp2(0.5, 3)) and not same2(0.52, _fp2(0.5, 3))   # the helpers themselves work
print("Helpers for Tasks 3 to 5 ready.")

# %% [markdown]
# *Exercise B3.1 (3 points): sinusoidal positions.* (a) Complete `sinusoid(pos, d)`, which returns the sinusoidal positional encoding of position `pos` as a NumPy vector of `d` numbers: even entries are sines and odd entries are cosines, the base is 10000, the angles are in radians, and `d` is even (1 point). (b) For d = 8 store in `dot_near` the dot product of the encodings of positions 20 and 21, and in `dot_far` the dot product of the encodings of positions 20 and 60 (1 point). (c) For d = 8, store in `wavelength` the number of positions after which the sine of pair i = 3 repeats (1 point).

# %%
def sinusoid(pos, d):
    """Sinusoidal positional encoding of position `pos`: a NumPy vector of d numbers."""
    return np.zeros(d)   # YOUR CODE HERE

dot_near = None    # YOUR CODE HERE
dot_far = None     # YOUR CODE HERE
wavelength = None  # YOUR CODE HERE

# %%
check("B3.1 (a) the function sinusoid", match2(sinusoid(13, 6), ['456091c83f7e', '81fcd719a101', '123fcfe32787', '06f77cac4c1a', 'ea83a39fc040', '7766013a6a62'], 3, (6,)) and match2(sinusoid(2, 4), ['b6b1f88add5b', 'cba28df6f054', '8152685bc722', '7766013a6a62'], 3, (4,)))
check("B3.1 (b) dot_near and dot_far", same2(dot_near, '1e31212ec07d', 3) and same2(dot_far, '65469d9a1ad4', 3))
check("B3.1 (c) wavelength", same2(wavelength, 'c56300246321', 0))

# %% [markdown]
# *Exercise B3.2 (2 points): multi-head shapes and parameters.* An attention layer has `d_model = 96` and `n_heads = 6`. (a) Pass the zero batch `xb` (4 sequences of 10 tokens) through an `nn.Linear(96, 96)` used as the query projection, split the result into heads the way the lesson does, and store in `q_shape` the shape tuple (batch, heads, tokens, width of one head) and in `d_head` the width of one head (1 point). (b) Store in `attn_formula` the parameter count of the whole attention layer (query, key, value and output projections, with biases) from the formula of the lesson, in `attn_torch` the number `param_count` gives for `nn.MultiheadAttention(96, 6)`, and in `attn_12` the formula count for the same `d_model` but 12 heads (1 point).

# %%
d_model, n_heads = 96, 6
xb = torch.zeros(4, 10, d_model)

q_shape = None        # YOUR CODE HERE
d_head = None         # YOUR CODE HERE
attn_formula = None   # YOUR CODE HERE
attn_torch = None     # YOUR CODE HERE
attn_12 = None        # YOUR CODE HERE

# %%
check("B3.2 (a) q_shape and d_head", match2(q_shape, ['375ce2f433bd', '5ca3832c5851', '4739b867a78f', '58c4b27462ab'], 0, (4,)) and same2(d_head, '58c4b27462ab', 0))
check("B3.2 (b) attn_formula, attn_torch and attn_12", same2(attn_formula, '790c153a0e69', 0) and same2(attn_torch, '790c153a0e69', 0) and same2(attn_12, '790c153a0e69', 0))

# %% [markdown]
# *Exercise B3.3 (2 points): layer normalisation.* Normalise the vector `v` of one token with the formula of Week 8: the mean and the variance are taken over the five numbers of the token (divide by 5, not by 4), `eps = 1e-5`. Store the normalised vector (before the learned scale and shift) in `v_hat` (1 point), and the output after the scale `gamma` and the shift `beta` in `ln_out` (1 point). Both are NumPy vectors of length 5.

# %%
v = np.array([4.0, 10.0, -2.0, 7.0, 1.0])
gamma = np.array([1.5, 1.0, 1.0, 0.5, 2.0]); beta = np.array([0.0, 0.0, 0.5, -1.0, 0.25]); eps = 1e-5

v_hat = None    # YOUR CODE HERE
ln_out = None   # YOUR CODE HERE

# %%
check("B3.3 normalised vector v_hat", match2(v_hat, ['bc07a0c75b31', 'b1d2aebe00b3', '73b267bb74f7', 'f6f89d120bcd', 'b7e48dc880c3'], 3, (5,)))
check("B3.3 output ln_out", match2(ln_out, ['bc07a0c75b31', 'b1d2aebe00b3', '8cc576bef585', '35d2a0cffb24', 'eef212f9350b'], 3, (5,)))

# %% [markdown]
# *Exercise B3.4 (2 points): one row of masked attention.* In a causal self-attention layer with `d_k = 16`, the query of the token at position 3 (counting from 0) has the raw dot products `raw` with the keys of the five tokens at positions 0 to 4. Apply the scaling, the causal mask and the softmax, and store the five attention weights in `w_row` (1 point). Then store in `out_row` the output for this token, the weighted sum of the rows of the value matrix `V_mat` (1 point).

# %%
d_k = 16
raw = np.array([2.0, -1.2, 3.6, 0.8, 5.2])   # query at position 3 against the keys at positions 0, 1, 2, 3, 4
V_mat = np.array([ [1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [-1.0, 1.0], [2.0, -2.0] ])   # one value row per position

w_row = None     # YOUR CODE HERE
out_row = None   # YOUR CODE HERE

# %%
check("B3.4 attention weights w_row", match2(w_row, ['927808b641f3', '5b67c4d45aaf', '5af4e8818055', '70833b66514a', 'bc07a0c75b31'], 3, (5,)))
check("B3.4 output out_row", match2(out_row, ['5cbc9278d4f4', '9402636990f0'], 3, (2,)))

# %% [markdown]
# *Exercise B3.5 (2 points): the shifted-target loss.* A model reads the first four tokens of the sequence `tokens` and, at each of the four positions, gives the probabilities in the rows of `P` (columns are the five token ids) for the next token. (a) Store in `targets` the list of the four correct next tokens (the sequence shifted by one), in `loss_shift` the average cross-entropy in nats (natural logarithm), and in `ppl` the perplexity (1 point). (b) A classmate wrote the targets as `tokens[:-1]` by mistake. Store in `loss_bug` the loss her code computes from the same `P`, and write one or two sentences in `ANSWERS["B3.5"]` on what such a bug makes the model learn (1 point).

# %%
tokens = [2, 0, 3, 1, 4]
P = np.array([ [0.30, 0.15, 0.25, 0.20, 0.10],
              [0.10, 0.20, 0.15, 0.45, 0.10],
              [0.20, 0.10, 0.20, 0.30, 0.20],
              [0.05, 0.30, 0.10, 0.30, 0.25] ])

targets = None     # YOUR CODE HERE
loss_shift = None  # YOUR CODE HERE
ppl = None         # YOUR CODE HERE
loss_bug = None    # YOUR CODE HERE
ANSWERS["B3.5"] = ""   # YOUR ANSWER

# %%
check("B3.5 (a) targets", match2(targets, ['8a8578704120', '0b1e0476f044', 'dc19710cf1c0', '375ce2f433bd'], 0, (4,)))
check("B3.5 (a) loss_shift and ppl", same2(loss_shift, '04e4e9d5a606', 3) and same2(ppl, '116fca716861', 2))
check("B3.5 (b) loss_bug", same2(loss_bug, '618ff11df0fe', 3))
check("B3.5 (b) explanation written", written2(ANSWERS.get("B3.5"), 50))

# %% [markdown]
# *Exercise B3.6 (3 points): train two settings and compare.* The cell below builds a small data set from the toy grammar of Week 9: `floor` is the lowest loss any model can reach on it (nats per character), `train_data` is 85% of the text and `val_data` the rest. Build two `TinyLM` models with `vocab = train_data.vocab_size`, `block = 32` and these settings: model A with `d = 16`, `heads = 2`, `layers = 1`, and model B with `d = 48`, `heads = 4`, `layers = 2`. Call `torch.manual_seed(42)` before building each one and train both with `train_lm(model, train_data, steps=250, block_size=32)`. Store the models in `model_a` and `model_b`, and the validation losses `eval_lm(model, val_data)` in `loss_a` and `loss_b` (1 point). Store in `gap_a` and `gap_b` how far each validation loss is above `floor`, and in `n_ratio` the number of parameters of B divided by that of A (1 point). In `ANSWERS["B3.6"]` write three or four sentences that compare the two models using your numbers, say what the floor means, and say what you would change first to improve the weaker one (1 point).

# %%
text, floor = make_grammar_text(120, seed=7)
cut = int(0.85 * len(text))
train_data = CharData(text[:cut]); val_data = CharData(text[cut:], chars=train_data.chars)
print(f"{train_data.vocab_size} characters in the vocabulary, {len(train_data.data)} training characters, floor = {floor:.3f} nats per character")

model_a = model_b = None          # YOUR CODE HERE: build and train the two models
loss_a = loss_b = None            # YOUR CODE HERE
gap_a = gap_b = n_ratio = None    # YOUR CODE HERE
ANSWERS["B3.6"] = ""              # YOUR ANSWER

# %%
def _b36_ok():
    try:
        return isinstance(model_a, TinyLM) and isinstance(model_b, TinyLM) and model_a.tok.embedding_dim == 16 and len(model_a.blocks) == 1 and model_b.tok.embedding_dim == 48 and len(model_b.blocks) == 2
    except Exception:
        return False
def _b36_trained():
    try:
        return abs(eval_lm(model_a, val_data) - loss_a) < 1e-6 and abs(eval_lm(model_b, val_data) - loss_b) < 1e-6 and max(loss_a, loss_b) < math.log(train_data.vocab_size) - 0.5
    except Exception:
        return False
check("B3.6 (a) the two models have the right settings", _b36_ok())
check("B3.6 (a) the models are trained and loss_a, loss_b are their validation losses", _b36_ok() and _b36_trained())
check("B3.6 (b) gap_a, gap_b", _b36_trained() and gap_a is not None and abs(gap_a - (loss_a - floor)) < 1e-6 and abs(gap_b - (loss_b - floor)) < 1e-6)
check("B3.6 (b) n_ratio", _b36_ok() and same2(n_ratio, '02aa629d0242', 2))
check("B3.6 (c) interpretation written", written2(ANSWERS.get("B3.6"), 120))

# %% [markdown]
# *Exercise B3.7 (2 points): count a whole configuration.* Consider `TinyLM(V, d=40, heads=4, layers=3, block=24)` with `V = train_data.vocab_size`: token embeddings, a learned position table, pre-norm blocks with attention and a feed-forward network with `d_ff = 4d` (all with biases and two layer norms per block), a final layer norm, and an output layer tied to the embedding table. (a) Store in `n_formula` the parameter count you get from the formulas of Week 8 (counted by hand-in-code, without building the model), and in `n_model` the number `param_count` gives for the model you build (1 point). (b) The rule of thumb says that the blocks have about 12 L d² parameters. Store in `rule_err_pct` by how many percent that rule is below the exact parameter count of the blocks (a positive number, 2 decimals) (1 point).

# %%
V, d, heads, L, block = train_data.vocab_size, 40, 4, 3, 24

n_formula = None     # YOUR CODE HERE
n_model = None       # YOUR CODE HERE
rule_err_pct = None  # YOUR CODE HERE

# %%
check("B3.7 (a) formula and model agree", n_formula is not None and n_model is not None and n_formula == n_model)
check("B3.7 (a) the count itself", same2(n_formula, '2715f26c0e61', 0) and same2(n_model, '2715f26c0e61', 0))
check("B3.7 (b) rule_err_pct", same2(rule_err_pct, '314dec417d49', 2))

# %% [markdown]
# ## Task 4 — LLM families, scaling laws and prompting (12 points, 30 minutes)
#
# Week 9 compared BERT, GPT and T5, showed how a mixture of experts keeps a big model cheap, turned the scaling laws into arithmetic, and ended with prompts that must be tested. In B4.1 to B4.5 you do the arithmetic of the three ideas in code, in B4.6 you write a small validator for a model's reply, and in B4.7 you decide whether a score difference is real. All numbers in this task are invented for the exam.

# %% [markdown]
# *Exercise B4.1 (1 point): training signals.* A sequence has `n = 400` tokens. A BERT-style model hides 15% of them and is trained only on the hidden positions; a GPT-style model is trained to predict the next token at every position that has a next token. Store in `n_mlm` the number of training targets of the BERT-style model, in `n_causal` the number of the GPT-style model, and in `signal_ratio` how many times more targets the GPT-style model gets (2 decimals).

# %%
n = 400
n_mlm = None         # YOUR CODE HERE
n_causal = None      # YOUR CODE HERE
signal_ratio = None  # YOUR CODE HERE

# %%
check("B4.1 n_mlm and n_causal", same2(n_mlm, '8651cf7bcfcf', 0) and same2(n_causal, '25faaff45f26', 0))
check("B4.1 signal_ratio", same2(signal_ratio, 'b7844f5d7908', 2))

# %% [markdown]
# *Exercise B4.2 (2 points): routing and parameters in a mixture of experts.* A layer has 6 experts and a router that sends each token to its top `k = 2` experts. For one token the router's scores are in `router_scores`. (a) Store in `top_idx` the indices of the two chosen experts (in increasing order) and in `top_w` their mixing weights, in the same order: the softmax over the two chosen scores only (1 point). (b) Every expert is a feed-forward network `d` to `d_ff` to `d` with biases (2·d·d_ff + d_ff + d parameters), here with `d = 256` and `d_ff = 1024`. Ignoring the router and everything outside the experts, store in `total_params` the parameters of the layer and in `active_params` the parameters one token uses (1 point).

# %%
router_scores = np.array([0.5, 2.3, -0.4, 1.7, 0.9, 1.1])
k, d_e, dff_e = 2, 256, 1024

top_idx = None        # YOUR CODE HERE
top_w = None          # YOUR CODE HERE
total_params = None   # YOUR CODE HERE
active_params = None  # YOUR CODE HERE

# %%
check("B4.2 (a) top_idx", match2(top_idx, ['dc19710cf1c0', '0b1e0476f044'], 0, (2,)))
check("B4.2 (a) top_w", match2(top_w, ['36cade7927cc', 'ebd9eb0e1212'], 3, (2,)))
check("B4.2 (b) total_params and active_params", same2(total_params, 'b71acdc3a2fb', 0) and same2(active_params, '275c7df3b1f2', 0))

# %% [markdown]
# *Exercise B4.3 (2 points): training compute and the Chinchilla rule.* (a) Use C = 6·N·D together with the rule D = 20·N to find the compute-optimal model for the budget `C_budget`: store N in billions of parameters in `N_opt_b` and D in billions of tokens in `D_opt_b` (1 point). (b) For the two training runs in `runs` (N parameters, D tokens), store in `flops_r1` the training compute of R1 in units of 10^21 FLOPs, and in `tpp_r1` and `tpp_r2` the tokens per parameter of R1 and R2 (1 point).

# %%
C_budget = 2.4e22
runs = {"R1": (12e9, 96e9), "R2": (3e9, 150e9)}   # (parameters N, training tokens D)

N_opt_b = None   # YOUR CODE HERE
D_opt_b = None   # YOUR CODE HERE
flops_r1 = None  # YOUR CODE HERE
tpp_r1 = None    # YOUR CODE HERE
tpp_r2 = None    # YOUR CODE HERE

# %%
check("B4.3 (a) N_opt_b and D_opt_b", same2(N_opt_b, '0a3771d70211', 3) and same2(D_opt_b, '663ab7ad5fa5', 2))
check("B4.3 (b) flops_r1", same2(flops_r1, '473def94cff1', 3))
check("B4.3 (b) tpp_r1 and tpp_r2", same2(tpp_r1, '2cf1b0cc6663', 2) and same2(tpp_r2, '1a3644484b57', 2))

# %% [markdown]
# *Exercise B4.4 (2 points): the fitted loss formula.* Week 9 used the fitted law L(N, D) = E + A / N^alpha + B / D^beta with E = 1.8172, A = 482.01, B = 2085.43, alpha = 0.3478 and beta = 0.3658 (N in parameters, D in tokens). (a) Complete `chin_loss(N, D)` (1 point). (b) Two designs spend the same compute: design A has `N = 14e9` and `D = 280e9`; design B is four times bigger (`N = 56e9`) and uses all of the same compute, `C = 6·14e9·280e9`, for its tokens. Store the fitted losses in `L_a` and `L_b`, and in `gap` the loss of B minus the loss of A (1 point).

# %%
E_, A_, B_, ALPHA, BETA = 1.8172, 482.01, 2085.43, 0.3478, 0.3658

def chin_loss(N, D):
    """The fitted loss of a model with N parameters trained on D tokens."""
    return None   # YOUR CODE HERE

L_a = None   # YOUR CODE HERE
L_b = None   # YOUR CODE HERE
gap = None   # YOUR CODE HERE

# %%
def _b44_fn():
    try:
        return same2(chin_loss(7e9, 1.1e11), '611e317706a6', 3) and same2(chin_loss(2e10, 4e11), '14c8140ce338', 3)
    except Exception:
        return False
check("B4.4 (a) the function chin_loss", _b44_fn())
check("B4.4 (b) L_a and L_b", same2(L_a, '5e31ef178204', 3) and same2(L_b, '7b812c3ebe64', 3))
check("B4.4 (b) gap", same2(gap, 'bd1183962867', 3))

# %% [markdown]
# *Exercise B4.5 (1 point): exact match and per-token accuracy.* A model gets each of the `L = 12` tokens of an answer right independently with probability `p = 0.93`. Store in `p_exact` the probability that the whole answer is right (3 decimals), and in `p_needed` the per-token probability that would give an exact-match probability of 0.5 (4 decimals).

# %%
p, L_ans = 0.93, 12
p_exact = None    # YOUR CODE HERE
p_needed = None   # YOUR CODE HERE

# %%
check("B4.5 p_exact", same2(p_exact, '94816bfae634', 3))
check("B4.5 p_needed", same2(p_needed, '69c1121cd1af', 4))

# %% [markdown]
# *Exercise B4.6 (2 points): validate a reply.* A program expects the model to reply with exactly one JSON object that has a text field `"name"` and a whole-number field `"score"`. Complete `check_reply(reply)` so that it returns `True` only if the whole reply (after `strip()`) is valid JSON, is an object, has a string `"name"` and an integer `"score"` (a decimal such as 3.5 is not an integer; other keys are allowed), and `False` for everything else without raising an error. Run it on the four replies in `TEST_REPLIES` and keep the list of four results in `results` (1 point for the four results, 1 point for a function that also gives the right answer on other replies).

# %%
TEST_REPLIES = [
    '{"name": "Ana", "score": 8}',
    '{"name": "Ben", "score": "9"}',
    'Sure! Here is the result: {"name": "Cy", "score": 7}',
    '{"score": 4, "name": "Eli", "note": "late"}',
]

def check_reply(reply):
    """True only if `reply` is one JSON object with a string "name" and an integer "score"."""
    return False   # YOUR CODE HERE

results = [check_reply(r) for r in TEST_REPLIES]
print(results)

# %%
_more = ['{"name": "Lea", "score": 3.5}', '[1, 2, 3]', 'not json at all', '{"name": "Max", "score": 10}', '{"name": 7, "score": 2}', '', '{"name": "Zed"}']
def _word(rs): return "".join("T" if r is True else "F" for r in rs)
check("B4.6 the four results", isinstance(results, list) and word2(_word(results), '12a8f4373bae'))
try:
    _more_res = [check_reply(r) for r in _more]
except Exception:
    _more_res = []
check("B4.6 the function on other replies", word2(_word(_more_res), '01c336e0685e'))

# %% [markdown]
# *Exercise B4.7 (2 points): is the difference real?* Two prompts were tried on the same `n_q = 90` test questions: prompt A got 63 right and prompt B got 54 right. (a) Store in `se_a` and `se_b` the standard error of each accuracy, sqrt(p(1 - p) / n) (4 decimals), and in `z` the difference of the two accuracies divided by sqrt(se_a² + se_b²) (2 decimals) (1 point). (b) In `ANSWERS["B4.7"]` write two or three sentences: is prompt A convincingly better, quoting `z`, and what would you do next (1 point).

# %%
n_q, correct_a, correct_b = 90, 63, 54

se_a = None   # YOUR CODE HERE
se_b = None   # YOUR CODE HERE
z = None      # YOUR CODE HERE
ANSWERS["B4.7"] = ""   # YOUR ANSWER

# %%
check("B4.7 (a) se_a and se_b", same2(se_a, '83b1996272a2', 4) and same2(se_b, '6ca7d3c9b28d', 4))
check("B4.7 (a) z", same2(z, '21ef9f46b544', 2))
check("B4.7 (b) interpretation written", written2(ANSWERS.get("B4.7"), 80))

# %% [markdown]
# ## Task 5 — Synthesis (6 points, 15 minutes)
#
# Three short scenarios that join several weeks. Answer each in `ANSWERS` in two to five sentences (2 points each). Use numbers where the question asks for them; you may use code cells for the arithmetic.

# %% [markdown]
# *Exercise B5.1 (2 points; Weeks 6 and 7).* A school wants a chatbot for Filipino and English text. A 400-character message is 400 tokens for a character-level model and about 100 tokens (4 characters per token) for a sub-word model with a vocabulary of 32,000 tokens. The character vocabulary has 100 symbols. The embedding width is 512. (a) How many parameters does the embedding table have in each design? (b) How many attention scores does one head compute per layer for the message in each design, and how many times more is the character-level one? (c) Name one cost of the large sub-word vocabulary, besides the table size, and one cost of the character-level design besides the scores.

# %% [markdown]
# *Exercise B5.2 (2 points; Weeks 7 and 8).* A team replaces an LSTM by a Transformer to read documents of 1,000 tokens. Information at token 1 must reach token 1,000. (a) In the LSTM, suppose each step passes on at most 0.9 of the gradient: how much reaches the first token over the 999 steps, and why does attention avoid the problem? (b) The Transformer's attention does not know the order of the tokens by itself: what do you add, and give the formula or the idea in one sentence. (c) With 8 heads, how many attention scores does the model compute in one layer for one document, and what does the KV cache save when the model later generates text?

# %% [markdown]
# *Exercise B5.3 (2 points; Weeks 8 and 9).* A start-up has a training budget of C = 1.0e23 FLOPs and plans a GPT-style assistant. (a) Using the Chinchilla rule, what model size N and how many tokens D would you choose (in billions)? (b) For one training sequence of 2,048 tokens, how many next-token targets does the causal model learn from, and about how many would a BERT-style model with 15% masking learn from? (c) An adviser suggests a mixture of experts with 8 experts and top-2 routing in the feed-forward layers, "a 100B model for the cost of a 25B one". Explain what is right and what must still be stored or paid for.

# %% [markdown]
# Write your three answers here (the check only tells you that something is there; your instructor reads and marks the text):

# %%
ANSWERS["B5.1"] = ""   # YOUR ANSWER
ANSWERS["B5.2"] = ""   # YOUR ANSWER
ANSWERS["B5.3"] = ""   # YOUR ANSWER

# %%
for _k in ("B5.1", "B5.2", "B5.3"):
    check(_k + " answer written", written2(ANSWERS.get(_k), 80))

# %%
print("Your ANSWERS (copy them into your submission if your instructor asks):")
for k in sorted(ANSWERS):
    print(f"\n{k}: {ANSWERS[k]}")

# %% [markdown]
# ## Finishing the exam
#
# 1. Check that every sub-question has an answer: the ✓ and ✗ lines above only tell you whether a number or a shape is right; your instructor marks the notebook with the marking scheme (correct result, method and interpretation), so make your method visible and write your reasoning in the `ANSWERS` entries.
# 2. Run **Runtime → Restart session and run all** (Colab) or **Kernel → Restart Kernel and Run All Cells** (JupyterLab) once, to make sure the notebook runs from top to bottom without errors.
# 3. In Colab, use **File → Save a copy in Drive** now if you have not done so, and download the notebook (**File → Download → .ipynb**); files in the session disappear when it ends. In JupyterLab, save the notebook (Ctrl+S).
# 4. Submit the notebook the way your instructor asks.
#
# Good luck, and well done for getting this far in the course.
