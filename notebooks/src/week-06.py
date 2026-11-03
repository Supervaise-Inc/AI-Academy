# %% [markdown]
# # Week 6 — NLP: Tokens & Word Embeddings: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** CPU is enough (the default). **Time:** about 3 hours if you do every exercise; the code runs in a few minutes. Two small downloads happen in Steps 3 and 4: the GPT-2 and BERT tokenizers (a few MB) and the MiniLM sentence-embedding model (about 90 MB).
#
# **What you will do**
# - Build a bigram language model from counts and generate text.
# - Implement byte-pair encoding (BPE) from scratch and compare it with the lesson's BPE figure.
# - Tokenise sentences with the real GPT-2 and BERT tokenizers and report what you see.
# - Compute sentence embeddings and cosine similarities.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open **Show a possible answer** only after you tried.

# %% [markdown]
# ## Setup
#
# Colab already has `torch` and `transformers`. We install `sentence-transformers` for Step 4.

# %%
#!pip install -q sentence-transformers

# %%
import os, random
from collections import defaultdict, Counter
import numpy as np
import torch

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
QUICK = bool(os.environ.get("NB_TEST"))   # shortens long loops when the notebook is tested

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1: a bigram model
#
# From the lesson section *N-gram models*: count which word follows which, and turn the counts into probabilities. We use the same 22-word text as the lesson figure, so you can check every count by hand.

# %%
text = "the cat sat on the mat . the cat saw the dog . the dog sat on the rug . the dog chased the cat ."
words = text.replace(".", " ").split()
counts = defaultdict(Counter)
for a, b in zip(words, words[1:]):
    counts[a][b] += 1                     # how often b follows a
print(counts["the"])

def prob(prev, nxt):
    return counts[prev][nxt] / sum(counts[prev].values())

print(prob("the", "cat"), prob("cat", "sat"), prob("the", "cat") * prob("cat", "sat"))
assert counts["the"] == Counter({"cat": 3, "dog": 3, "mat": 1, "rug": 1})
assert prob("the", "cat") * prob("cat", "sat") == 0.1875

# %% [markdown]
# Now generate text: start from a word and repeatedly pick the next word at random, weighted by the counts.

# %%
def generate(start, n=8, seed=0):
    rnd = random.Random(seed)
    out = [start]
    for _ in range(n):
        nxt = counts.get(out[-1])
        if not nxt:                        # no known follower: stop
            break
        out.append(rnd.choices(list(nxt), weights=list(nxt.values()))[0])
    return " ".join(out)

for s in range(3):
    print(generate("the", seed=s))

# %% [markdown]
# ### ✏️ Your turn: probabilities from counts
#
# Compute (a) P(mat | the), (b) P(sat | dog), and (c) the probability of the sequence "the dog sat" given the first word "the", that is P(dog | the) × P(sat | dog). Use the `prob` function.

# %%
p_mat = None      # YOUR CODE HERE
p_sat_dog = None  # YOUR CODE HERE
p_seq = None      # YOUR CODE HERE

# %%
if None in (p_mat, p_sat_dog, p_seq):
    print("Fill in your answers above, then run this cell again.")
else:
    check("P(mat | the)", abs(p_mat - 1/8) < 1e-9, "'the' has 8 followers in total; mat is one of them.")
    check("P(sat | dog)", abs(p_sat_dog - prob("dog", "sat")) < 1e-9, "Count how often 'dog' is followed by 'sat', divide by all followers of 'dog'.")
    check("P(dog sat | the)", abs(p_seq - prob("the", "dog") * prob("dog", "sat")) < 1e-9, "Multiply the two probabilities; do not add them.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# p_mat = prob("the", "mat")                       # 0.125
# p_sat_dog = prob("dog", "sat")
# p_seq = prob("the", "dog") * prob("dog", "sat")
# ```
#
# Probabilities of consecutive steps are multiplied. A pair that never appeared, such as "the" followed by "sat", gets probability zero: that is the n-gram limit.
# </details>

# %% [markdown]
# ### ✏️ Your turn: try a trigram
#
# A trigram model uses the last **two** words as the key. Build `tri_counts` (key = a tuple of two words, value = Counter of next words) and print `tri_counts[("on", "the")]`. What does a trigram fix, and what does it cost? Write your thoughts in a new text cell.

# %%
tri_counts = defaultdict(Counter)
# YOUR CODE HERE: fill tri_counts from `words`

# %%
if not tri_counts:
    print("Fill in your answer above, then run this cell again.")
else:
    check("('on', 'the') is followed by mat and rug once each",
          tri_counts[("on", "the")] == Counter({"mat": 1, "rug": 1}), "Loop over zip(words, words[1:], words[2:]).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# for a, b, c in zip(words, words[1:], words[2:]):
#     tri_counts[(a, b)][c] += 1
# ```
#
# A trigram sees more context, so its text is more fluent. The cost: far more possible keys, so most two-word contexts are never seen, and unseen sequences still get probability zero.
# </details>

# %% [markdown]
# ## Step 2: BPE from scratch
#
# From the lesson section *Tokens and byte-pair encoding*: start with characters (plus an end-of-word symbol `</w>`), count every neighbouring pair **weighted by word frequency**, merge the most frequent pair into one symbol, and repeat. The corpus is the lesson's: low ×5, lower ×2, newest ×6, widest ×3.

# %%
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
start_vocab = {tuple(w) + ("</w>",): f for w, f in corpus.items()}

def pair_counts(vocab):
    pc = Counter()
    for sym, f in vocab.items():
        for a, b in zip(sym, sym[1:]):
            pc[(a, b)] += f               # a word seen f times contributes f to each of its pairs
    return pc

def merge(vocab, pair):
    new = {}
    for sym, f in vocab.items():
        out, i = [], 0
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == pair[0] and sym[i + 1] == pair[1]:
                out.append(sym[i] + sym[i + 1]); i += 2
            else:
                out.append(sym[i]); i += 1
        new[tuple(out)] = f
    return new

def learn_bpe(vocab, n_merges, verbose=True):
    merges = []
    for step in range(n_merges):
        pc = pair_counts(vocab)
        if not pc:
            break
        best, n = pc.most_common(1)[0]
        merges.append(best)
        vocab = merge(vocab, best)
        if verbose:
            print(step + 1, best, n)
    return merges, vocab

merges, final_vocab = learn_bpe(start_vocab, 9)

# %% [markdown]
# Compare with the lesson's BPE figure: its first three merges are (e, s) 9, (es, t) 9 and (est, </w>) 9. Our first three should match exactly. The assert below turns the comparison into a check.

# %%
assert merges[:3] == [("e", "s"), ("es", "t"), ("est", "</w>")], "the first merges differ from the lesson figure"
assert pair_counts(start_vocab)[("e", "s")] == 9
print("First three merges match the lesson. Final symbols of each word:")
for sym in final_vocab:
    print(" ", sym)

# %% [markdown]
# ### ✏️ Your turn: BPE on your own six words
#
# Use the six words in the starter (each appears once). Before running anything, think about which pair is most frequent first. Then run `learn_bpe` for 5 merges and store the list of merges in `my_merges`.

# %%
my_corpus = {"walk": 1, "walked": 1, "walking": 1, "talk": 1, "talked": 1, "talking": 1}
my_merges = None   # YOUR CODE HERE: use learn_bpe

# %%
if my_merges is None:
    print("Fill in your answer above, then run this cell again.")
else:
    expected, _ = learn_bpe({tuple(w) + ("</w>",): f for w, f in my_corpus.items()}, 5, verbose=False)
    check("5 merges learned", len(my_merges) == 5, "Call learn_bpe(vocab, 5) with vocab built like start_vocab.")
    check("merges equal the true list", list(my_merges) == expected, "Build the starting vocab with tuple(word) + ('</w>',) and keep the default tie-breaking.")
    print("Your merges:", my_merges)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# vocab = {tuple(w) + ("</w>",): f for w, f in my_corpus.items()}
# my_merges, _ = learn_bpe(vocab, 5)
# ```
#
# In our run the merges are (a, l), (al, k), (w, alk), (t, alk) and (e, d): the shared piece "alk" is built first because all six words contain it. Ties are broken by whichever pair was counted first.
# </details>

# %% [markdown]
# Now encode words the corpus never contained, by applying the learned merges in order.

# %%
def encode(word, merges):
    v = {tuple(word) + ("</w>",): 1}
    for pair in merges:
        v = merge(v, pair)
    return list(next(iter(v)))

for w in ["lowest", "newer", "wider"]:
    print(w, "->", encode(w, merges))
assert encode("lowest", merges) == ["low", "est</w>"]

# %% [markdown]
# ### ✏️ Your turn: why does "newer" split like that?
#
# Run `encode("newer", merges)` yourself and put the list in `pieces_newer`. Then explain in a text cell why it ends in single characters, even though "new" was learned.

# %%
pieces_newer = None   # YOUR CODE HERE

# %%
if pieces_newer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("pieces for 'newer'", pieces_newer == encode("newer", merges), "Call encode('newer', merges) and copy the list it returns.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# pieces_newer = encode("newer", merges)
# print(pieces_newer)
# ```
#
# Only the nine learned merges can apply. "new" was merged, but none of them joins (e, r) or (r, </w>), so the last characters and the end-of-word symbol stay separate. A sub-word tokenizer can read a new word by falling back to smaller known pieces.
# </details>

# %% [markdown]
# ## Step 3: real tokenizers
#
# Our BPE has a vocabulary of a few dozen symbols. Real tokenizers learn tens of thousands of merges. GPT-2 uses byte-level BPE; BERT uses a similar method called WordPiece. We load both tokenizers (a few MB each, public and ungated) and look at the pieces they produce. We only report what we see: we do not guess the counts in advance.

# %%
from transformers import AutoTokenizer

sentences = ["unbelievably good", "Mabilis ang tren.", "Tokenization matters."]
results = {}
for name in ["gpt2", "bert-base-uncased"]:
    tok = AutoTokenizer.from_pretrained(name)
    for s in sentences:
        pieces = tok.tokenize(s)
        results[(name, s)] = pieces
        print(f"{name:18s} {len(pieces):2d} pieces  {s!r} -> {pieces}")

# %% [markdown]
# Look at the printed pieces. GPT-2 marks a word that follows a space in a special way; BERT marks a word continuation in another way. Which one is it, for each? And divide the number of pieces by the number of words in each sentence to compare how many tokens per word each sentence needs.

# %% [markdown]
# ### ✏️ Your turn: read the output
#
# Using the `results` dictionary, set `more_for_unbelievably` to the name (`"gpt2"` or `"bert-base-uncased"`) whose tokenizer splits `"unbelievably good"` into **more** pieces (or `"same"` if equal), and `tokens_per_word` to the highest pieces-per-word value over the three sentences with the GPT-2 tokenizer, rounded to two decimals.

# %%
more_for_unbelievably = None   # YOUR CODE HERE
tokens_per_word = None         # YOUR CODE HERE

# %%
g = len(results[("gpt2", "unbelievably good")]); bt = len(results[("bert-base-uncased", "unbelievably good")])
true_more = "same" if g == bt else ("gpt2" if g > bt else "bert-base-uncased")
true_tpw = max(round(len(results[("gpt2", s)]) / len(s.split()), 2) for s in sentences)
if more_for_unbelievably is None or tokens_per_word is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("tokenizer with more pieces", more_for_unbelievably == true_more, "Compare len(results[(name, 'unbelievably good')]) for both names.")
    check("highest tokens per word (GPT-2)", abs(tokens_per_word - true_tpw) <= 0.01, "For each sentence divide the number of pieces by len(s.split()), then take the max.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# g = len(results[("gpt2", "unbelievably good")]); b = len(results[("bert-base-uncased", "unbelievably good")])
# more_for_unbelievably = "same" if g == b else ("gpt2" if g > b else "bert-base-uncased")
# tokens_per_word = max(round(len(results[("gpt2", s)]) / len(s.split()), 2) for s in sentences)
# ```
#
# Your numbers come from the run above. In the lesson's words, a language that was rare in the tokenizer's training text is often split into more pieces, which makes it slower and more expensive to process. Check whether the sentence in Tagalog behaves that way in your output, and write down only what you saw.
# </details>

# %% [markdown]
# ## Step 4: sentence embeddings and cosine similarity
#
# A token is just an ID; an embedding is a vector that carries meaning. First the cosine formula by hand with the lesson's example: for (3, 4) and (4, 3) the dot product is 24 and both lengths are 5, so the cosine is 24 / 25 = 0.96.

# %%
def cosine(u, v):
    u, v = np.array(u, dtype=float), np.array(v, dtype=float)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

print(cosine([3, 4], [4, 3]))
assert abs(cosine([3, 4], [4, 3]) - 0.96) < 1e-9

# %% [markdown]
# Now real embeddings. `sentence-transformers/all-MiniLM-L6-v2` is a small public model (about 90 MB) that turns a sentence into a 384-number vector. Setting `normalize_embeddings=True` makes every vector length 1, so a dot product equals the cosine.

# %%
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
sents = ["A cat sat on the mat.", "A kitten rested on the rug.", "The stock market fell sharply today."]
E = model.encode(sents, normalize_embeddings=True)
S = E @ E.T
print(E.shape)
print(S.round(2))
assert E.shape == (3, 384) and np.allclose(np.diag(S), 1.0, atol=1e-4)

# %% [markdown]
# ### ✏️ Your turn: which pair is most similar?
#
# Look at the matrix. Set `best_pair` to the tuple of the two sentence **indices** (smaller first, different from each other) with the highest cosine similarity. Then explain in a text cell why the diagonal is 1.0.

# %%
best_pair = None   # YOUR CODE HERE, for example (0, 2)

# %%
off = [(S[i, j], (i, j)) for i in range(3) for j in range(i + 1, 3)]
true_pair = max(off)[1]
if best_pair is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("most similar pair", tuple(best_pair) == true_pair, "Ignore the diagonal and find the largest of the three off-diagonal values.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# best_pair = max(((S[i, j], (i, j)) for i in range(3) for j in range(i + 1, 3)))[1]
# ```
#
# The diagonal compares each sentence with itself, and the cosine of a vector with itself is 1. The cat sentence and the kitten sentence use different words but a similar meaning; the stock-market sentence is about something else. Check your matrix to see if the numbers agree.
# </details>

# %% [markdown]
# ### ✏️ Your turn: word analogy arithmetic
#
# In a 2-D embedding king = (5, 2), man = (2, 1) and woman = (2, 3). Compute king − man + woman as a list `[x, y]`. Then add three sentence pairs of your own in a new cell, predict the similarities, and run them. Include one pair with the same words in a different meaning. Remember that the famous analogy examples are chosen because they work.

# %%
analogy = None   # YOUR CODE HERE: [x, y]

# %%
true_analogy = list(np.array([5, 2]) - np.array([2, 1]) + np.array([2, 3]))
if analogy is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("king - man + woman", list(analogy) == true_analogy, "Subtract and add each coordinate separately.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# analogy = list(np.array([5, 2]) - np.array([2, 1]) + np.array([2, 3]))   # [5, 4]
# ```
#
# (5 − 2 + 2, 2 − 1 + 3) = (5, 4). Adding all three vectors instead gives (9, 6), a common mistake.
# </details>

# %% [markdown]
# ## Wrap-up
#
# - A bigram model estimates P(next | previous) from counts; it sees only one word back and gives zero to sequences it never saw.
# - BPE merges the most frequent neighbouring pair again and again, so a small vocabulary can still read new words by splitting them into known pieces.
# - Embeddings give each token or sentence a vector; cosine similarity measures how much two vectors point the same way, and embeddings can copy the biases of their training text.
#
# **Reflection:** double-click to write two or three sentences: (a) one thing a bigram model cannot do that BPE plus embeddings can, (b) one risk of embeddings learned from web text, (c) one limit of the king − man + woman example.
#
# Files in `/content` vanish when your Colab session ends, so download anything you want to keep (File → Download). Submit it the way your instructor asks.
