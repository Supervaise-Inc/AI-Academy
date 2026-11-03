# %% [markdown]
# # Week 16 — LLM Components & RAG: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** CPU is enough (Runtime → Change runtime type → None/CPU). A GPU is not needed.
# **Time:** about 3 hours (compute is under 15 minutes).
#
# **What you will do**
# - Split a few documents into overlapping chunks.
# - Retrieve the best chunks with TF-IDF and with sentence embeddings, and measure retrieval on its own.
# - Build the grounded prompt, including the "say you do not know" instruction.
# - Ask a small public instruct model the same questions with and without retrieval.
# - Break the pipeline on purpose and name the stage that broke (index, retrieve or generate).
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.
#
# All sample documents here are **FICTIONAL SAMPLE TEXT** for practice. They are not real engineering guidance.

# %%
#!pip install -q sentence-transformers
import os, random
import numpy as np
import torch

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
QUICK = bool(os.environ.get("NB_TEST"))

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# A fixed sanity check so a broken environment is noticed early
assert np.isclose(np.dot([1, 0], [0, 1]), 0.0)

# %% [markdown]
# ## Step 1 — Documents and chunking
#
# RAG starts with the **index** stage (see "How RAG works" in the lesson). We cut each document into small overlapping chunks, because a retriever works better on short pieces than on whole documents. Here a chunk is 30 words and neighbouring chunks share 10 words, so a sentence cut at a boundary still appears whole in one of them.

# %%
docs = {
 "cracks": "FICTIONAL SAMPLE TEXT. Cracks: record where each crack is on the structure, measure its length and its width with a ruler, and take a photo with a scale next to the crack. Note whether the crack is dry or wet. Cracks that cross a joint should be marked in the report with their own label.",
 "spalling": "FICTIONAL SAMPLE TEXT. Spalling: spalling is when concrete breaks off in flakes or chunks. Note the area that is affected, estimate how deep the missing layer is, and write down whether steel bars are visible underneath. Take photos from two angles.",
 "corrosion": "FICTIONAL SAMPLE TEXT. Corrosion: corrosion is rust on steel parts. Look at bolts, railings and exposed reinforcing bars, and note rust stains that run down the surface. Photograph each rusted part and record its position on the sketch.",
 "drainage": "FICTIONAL SAMPLE TEXT. Drainage: check that gutters and drain holes are clear of leaves and dirt. Standing water speeds up damage, so record any place where water collects after rain.",
 "reporting": "FICTIONAL SAMPLE TEXT. Reporting: every finding goes to the supervising engineer, who makes the pass or fail decision. The assistant only drafts a summary and must never declare a structure safe.",
}

def chunk(text, size=30, overlap=10):
    assert overlap < size, "overlap must be smaller than size"
    words = text.split()
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out

chunks = [(name, c) for name, t in docs.items() for c in chunk(t)]
print("chunks per document:", {n: len(chunk(t)) for n, t in docs.items()})
print("total chunks:", len(chunks))
assert len(chunks) == 10

# %% [markdown]
# ### ✏️ Your turn: change the chunk size
#
# The code above prints 10 chunks. Change the settings to `size=20` and `overlap=0`, count the chunks, and put the new total in `answer`. Then think about why `cracks` makes 3 chunks with the first settings and `drainage` makes only 1.

# %%
answer = None  # YOUR CODE HERE: the total number of chunks with size=20, overlap=0

# %%
true_count = sum(len(chunk(t, size=20, overlap=0)) for t in docs.values())
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("chunk count with size=20, overlap=0", answer == true_count, "count the chunks with a loop over docs and chunk(t, size=20, overlap=0)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = sum(len(chunk(t, size=20, overlap=0)) for t in docs.values())
# print(answer)   # 12 (3, 3, 2, 2 and 2 chunks per document)
# ```
#
# `cracks` has about 50 words, so with a window of 30 and a step of 20 it needs 3 windows. `drainage` has fewer than 30 words, so one window already covers it.
# </details>

# %% [markdown]
# ## Step 2 — TF-IDF retrieval (the no-download baseline)
#
# The **retrieve** stage turns the question and every chunk into vectors and ranks chunks by cosine similarity. TF-IDF is the simplest vector: it counts words and gives rare words more weight. It needs no model download, but it only matches words the question and the chunk **share**.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(stop_words="english")
M = vec.fit_transform([c for _, c in chunks])   # fit on the chunks only

def retrieve_tfidf(question, k=2):
    # transform (not fit_transform): the question must use the chunk vocabulary
    sims = cosine_similarity(vec.transform([question]), M)[0]
    top = sims.argsort()[::-1][:k]
    return [(chunks[i][0], chunks[i][1], float(sims[i])) for i in top]

for name, text, s in retrieve_tfidf("How do I measure a crack?"):
    print(round(s, 3), name, "|", text[:60])
assert retrieve_tfidf("How do I measure a crack?", k=1)[0][0] == "cracks"

# %% [markdown]
# ### ✏️ Your turn: a question with a score of 0
#
# Below are four questions. Two can be answered by the documents, one is a paraphrase with no shared keywords, and one is about something the documents never mention. Run `retrieve_tfidf` on each one (use `k=1`) and put in `answer` the question that gets a top score of exactly 0.

# %%
questions = [
    "How do I measure a crack?",
    "Which part of the bridge gets eroded by oxidation?",
    "What is the maximum allowed crack width in millimetres?",
    "Where does water collect?",
]
for q in questions:
    pass  # YOUR CODE HERE (optional): print the top score for each question
answer = None  # YOUR CODE HERE: the question string whose top score is 0

# %%
scores = {q: retrieve_tfidf(q, k=1)[0][2] for q in questions}
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    zero = [q for q, s in scores.items() if s == 0.0]
    check("you found a question with score 0", answer in zero, "print the top score of each question and look for 0.0")
    print("all scores:", {q[:30]: round(s, 3) for q, s in scores.items()})

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# for q in questions:
#     print(round(retrieve_tfidf(q, k=1)[0][2], 3), q)
# answer = "Which part of the bridge gets eroded by oxidation?"
# ```
#
# The paraphrase shares no content word with the corrosion chunk ("rust" is never written), so TF-IDF cannot match it and every score is 0. Embeddings (Step 4) fix this kind of miss. Note that the crack-width question is not 0: it shares "crack" with the documents even though the documents do not answer it.
# </details>

# %% [markdown]
# ## Step 3 — Measure retrieval on its own
#
# Before any language model is involved, check retrieval alone. If you judge only the final answer, you cannot tell which stage failed. Each test is a question and the document that should come first.

# %%
tests = [("How do I measure a crack?", "cracks"),
         ("What should I note about rust on bolts?", "corrosion"),
         ("Who makes the pass or fail decision?", "reporting"),
         ("Where does water collect?", "drainage"),
         ("What does a spalling inspection record?", "spalling")]

def top1_accuracy(retrieve_fn, tests):
    hits = sum(retrieve_fn(q, k=1)[0][0] == want for q, want in tests)
    return hits, len(tests)

h, n = top1_accuracy(retrieve_tfidf, tests)
print(f"TF-IDF top-1 accuracy: {h}/{n}")
assert (h, n) == (5, 5)

# %% [markdown]
# ### ✏️ Your turn: add two paraphrased questions
#
# Add two paraphrased questions of your own to `extra_tests`, each as a tuple `(question, expected_document)`. A paraphrase means different words than the document uses. The Check reports the new accuracy and tells you whether the right source is in the top 3 even when it is not first.

# %%
extra_tests = []   # YOUR CODE HERE: e.g. [("question one", "cracks"), ("question two", "drainage")]

# %%
if len(extra_tests) < 2:
    print("Fill in your answer above, then run this cell again.")
else:
    h, n = top1_accuracy(retrieve_tfidf, tests + extra_tests)
    print(f"TF-IDF top-1 accuracy with your questions: {h}/{n}")
    in_top3 = sum(want in [r[0] for r in retrieve_tfidf(q, k=3)] for q, want in extra_tests)
    print(f"right source inside the top 3 for your questions: {in_top3}/{len(extra_tests)}")
    check("expected documents exist", all(w in docs for _, w in extra_tests), "use one of: " + ", ".join(docs))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# extra_tests = [("How wide is the gap in the concrete and where is it?", "cracks"),
#                ("What do I look for where the pipes drain?", "drainage")]
# ```
#
# Your numbers depend on your questions. Paraphrases often lower top-1 accuracy for TF-IDF, and a larger k gives the right chunk more chances to appear, at the price of a longer prompt later.
# </details>

# %% [markdown]
# ## Step 4 — Embeddings with sentence-transformers
#
# An **embedding model** maps text to a vector so that texts with similar meaning are close, even without shared words. We use `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB download, 384 numbers per text). Because we normalise the vectors to length 1, the dot product equals the cosine similarity.

# %%
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
E = emb_model.encode([c for _, c in chunks], normalize_embeddings=True)
print("chunk embedding matrix:", E.shape)
assert E.shape == (len(chunks), 384)

def retrieve_emb(question, k=2):
    q = emb_model.encode([question], normalize_embeddings=True)[0]
    sims = E @ q                          # cosine similarity (unit vectors)
    top = sims.argsort()[::-1][:k]
    return [(chunks[i][0], chunks[i][1], float(sims[i])) for i in top]

h, n = top1_accuracy(retrieve_emb, tests)
print(f"embedding top-1 accuracy: {h}/{n}")
para = "Which part of the bridge gets eroded by oxidation?"
print("TF-IDF  :", [(s[0], round(s[2], 2)) for s in retrieve_tfidf(para, k=2)])
print("Embedding:", [(s[0], round(s[2], 2)) for s in retrieve_emb(para, k=2)])

# %% [markdown]
# Compare the two lines at the bottom. TF-IDF scores 0 on the paraphrase, while the embedding model gives every chunk a non-zero score, so it can still rank them. Whether it ranks the right one first is something you should look at, not assume.
#
# ### ✏️ Your turn: which retriever do you trust?
#
# Make a list `unanswerable` of two questions the documents **cannot** answer (for example about crack width limits or the weather). Print the top embedding score for each. A retriever always returns something, so the score alone does not tell you the answer is missing.

# %%
unanswerable = []   # YOUR CODE HERE: two questions the documents cannot answer
for q in unanswerable:
    print(round(retrieve_emb(q, k=1)[0][2], 3), q)

# %%
if len(unanswerable) < 2:
    print("Fill in your answer above, then run this cell again.")
else:
    top = [retrieve_emb(q, k=1)[0][2] for q in unanswerable]
    check("the retriever still returned a score for unanswerable questions", all(t > 0 for t in top),
          "cosine of real text is rarely exactly 0 for embeddings")
    print("Lesson: retrieval returns the nearest chunks, even when none of them answers the question.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# unanswerable = ["What is the maximum allowed crack width in millimetres?",
#                 "What is the weather forecast for tomorrow?"]
# ```
#
# Both still get a top chunk with a positive score. That is why the prompt needs the "say you do not know" instruction (next step): retrieval will never refuse by itself.
# </details>

# %% [markdown]
# ## Step 5 — Build the grounded prompt
#
# The **augment** stage puts the retrieved chunks into the prompt. This prompt has three instructions: use only the passages, quote the passage number, and reply exactly "I do not know." when the answer is not there (see "How RAG works").

# %%
def build_prompt(question, hits):
    context = "\n".join(f"[{i + 1}] ({name}) {text}" for i, (name, text, _) in enumerate(hits))
    return ("Answer using ONLY the passages below. Quote the passage number. "
            'If the answer is not in the passages, reply exactly: "I do not know."\n\n'
            f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:")

q = "What should I record for corrosion?"
print(build_prompt(q, retrieve_emb(q, k=2)))

# %% [markdown]
# ### ✏️ Your turn: what is each instruction for?
#
# Put in `answer` the exact sentence the model is told to reply when the passages do not contain the answer. Then, in a text cell or in a comment, explain in one line what each of the three instructions is for.

# %%
answer = None  # YOUR CODE HERE: the exact reply, as a string

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("exact reply for unknown answers", answer.strip() == "I do not know.", 'copy it from the prompt, including the full stop')

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = "I do not know."
# ```
#
# "ONLY the passages" keeps the model from answering out of memory. "Quote the passage number" makes the answer checkable. "Reply exactly I do not know" gives the model a safe way out, so it does not invent an answer when retrieval found nothing useful.
# </details>

# %% [markdown]
# ## Step 6 — Generate, with and without retrieval
#
# Now the **generate** stage. We use `Qwen/Qwen2.5-0.5B-Instruct`, a small public instruct model (about 1 GB download, runs on CPU in a few seconds per answer). It replaces the larger gated Llama model named in the lesson. A model this small may ignore the instructions, and that is a finding worth reporting, not a bug in your code.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
lm_name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(lm_name)
lm = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch.float32).to(device).eval()
print("loaded", lm_name, "parameters:", sum(p.numel() for p in lm.parameters()))

def ask(prompt, max_new_tokens=80):
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        out = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# %%
for q in ["What should I record for corrosion?", "What is the maximum allowed crack width in millimetres?"]:
    print("Q:    ", q)
    print("PLAIN:", ask(q))
    print("RAG:  ", ask(build_prompt(q, retrieve_emb(q, k=2))))
    print()

# %% [markdown]
# ### ✏️ Your turn: read the answers
#
# For each of the two questions, look at the plain answer and the RAG answer. Put in `grounded` which one is based on the documents for the corrosion question (`"plain"` or `"rag"`), and in `refused_unknown` whether the RAG answer to the crack-width question said "I do not know" (`True` or `False`). Write one sentence about what you see in a comment.

# %%
grounded = None          # YOUR CODE HERE: "plain" or "rag"
refused_unknown = None   # YOUR CODE HERE: True or False

# %%
if grounded is None or refused_unknown is None:
    print("Fill in your answers above, then run this cell again.")
else:
    check("the grounded answer is the RAG one", grounded == "rag", "the RAG prompt contains the corrosion passage")
    real = "i do not know" in ask(build_prompt("What is the maximum allowed crack width in millimetres?",
                                                retrieve_emb("What is the maximum allowed crack width in millimetres?", k=2))).lower()
    print("(for comparison, the model's actual behaviour this run:", real, ")")
    check("refused_unknown matches what the model did", refused_unknown == real, "re-read the RAG answer to the crack-width question")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# grounded = "rag"
# refused_unknown = False   # or True: a small model may or may not follow the instruction
# ```
#
# The RAG answer uses the corrosion passage (bolts, railings, rust stains). For the crack-width question the documents say nothing about limits. If the model guessed a number, it ignored the instruction, which a 0.5-billion-parameter model sometimes does. The honest result is whatever you saw.
# </details>

# %% [markdown]
# ## Step 7 — Break it on purpose
#
# Change one thing at a time and see what happens. The three changes below each break a different stage. Retrieval is checked directly, so you can see the effect without a language model.

# %%
# (b) k=1 for a question that needs two chunks
q = "How do I report a crack that crosses a joint?"
print("k=1:", [(s[0], round(s[2], 2)) for s in retrieve_emb(q, k=1)])
print("k=3:", [(s[0], round(s[2], 2)) for s in retrieve_emb(q, k=3)])

# (c) delete the corrosion document from the index, then ask about corrosion
chunks_small = [(n, c) for n, c in chunks if n != "corrosion"]
E_small = emb_model.encode([c for _, c in chunks_small], normalize_embeddings=True)
qv = emb_model.encode(["What should I note about rust on bolts?"], normalize_embeddings=True)[0]
best = (E_small @ qv).argmax()
print("without corrosion document, top source is:", chunks_small[best][0])
assert chunks_small[best][0] != "corrosion"

# (a) remove the "I do not know" instruction and compare the prompts
weak = build_prompt("What is the maximum allowed crack width in millimetres?", retrieve_emb("crack width", k=2)).replace(
    'If the answer is not in the passages, reply exactly: "I do not know."\n\n', "\n\n")
print("instruction removed:", 'I do not know' not in weak)

# %% [markdown]
# ### ✏️ Your turn: name the broken stage
#
# For each change, put the broken stage in a dictionary: `"index"`, `"retrieve"` or `"generate"`. The keys are `"a"` (removing the "I do not know" instruction), `"b"` (k=1 for a two-chunk question) and `"c"` (deleting the corrosion document).

# %%
stages = {"a": None, "b": None, "c": None}   # YOUR CODE HERE: replace each None

# %%
truth = {"a": "generate", "b": "retrieve", "c": "index"}
if any(v is None for v in stages.values()):
    print("Fill in your answer above, then run this cell again.")
else:
    for key in truth:
        check(f"change ({key})", stages[key] == truth[key], "ask yourself: was the information stored, found, or used?")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# stages = {"a": "generate", "b": "retrieve", "c": "index"}
# ```
#
# (a) The prompt no longer tells the model to refuse, so the model may guess: a **generate**-stage problem. (b) With k=1 only one of two needed chunks reaches the prompt: a **retrieve**-stage problem. (c) The document is missing from the index, so no retriever could find it: an **index**-stage problem.
# </details>

# %%
# Free the language model's memory when you are done
del lm
if device == "cuda":
    torch.cuda.empty_cache()
print("memory released")

# %% [markdown]
# ## Wrap-up
#
# - RAG has four stages (index, retrieve, augment, generate), and you can test each one on its own. Retrieval alone gave a measurable top-1 accuracy before any language model ran.
# - Embeddings can match meaning where TF-IDF needs shared words, but retrieval always returns something, so the prompt must tell the model to say "I do not know."
# - A small model may not follow the instructions, so a grounded prompt reduces but does not remove wrong answers.
#
# **Reflection.** Double-click to write two sentences: when did RAG help in this notebook, and when did it not?
#
# Files in `/content` vanish when the Colab session ends, so download anything you want to keep (File → Download .ipynb). Submit your notebook the way your instructor asks.
