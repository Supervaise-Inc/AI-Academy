# %% [markdown]
# # Week 19 — Capstone Build Week + Mock NCA-GENL Exam: practice notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive** so your answers are saved in your own copy.
#
# **Runtime:** CPU is enough (Runtime → Change runtime type → None/CPU). **Time:** about 2 hours.
#
# There is no lab this week, so this is a **practice notebook**: you compute answers in code instead of just reading them, and you get a starter for one stage of your capstone.
#
# **What you will do**
# - Use a mock-exam timing and self-scoring helper (50 questions, 60 minutes, pass at 35/50).
# - Practise one computed exercise from each term of the course.
# - Run a starter for the Structural Inspection Assistant: CLIP zero-shot classification of a synthetic image, with a structured result that always says an engineer must review it.
# - Run a tiny TF-IDF retrieval stage over fictional checklist text.
#
# **How this notebook works:** run cells top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open **Show a possible answer** only after you tried.

# %%
import os, math, random, textwrap
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Set to True only in the automated test; students keep the default.
QUICK = bool(os.environ.get("NB_TEST"))

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
except ImportError:
    device = "cpu"
print("Device:", device, "(CPU is fine for this notebook)")

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# A fixed case, so a broken environment is noticed straight away.
assert abs(math.sqrt(16) - 4) < 1e-9
check("setup works", True)

# %% [markdown]
# ## Part A — Mock exam timing and self-scoring
#
# The mock NCA-GENL exam in the course outline has **50 multiple-choice questions, 60 minutes, closed book, and a pass mark of 35/50 (70%)**. The lesson section *The mock NCA-GENL exam* works out the pace: 60 × 60 / 50 = 72 seconds per question on average. The helper below does that arithmetic for you and scores your own practice results. Nothing here is interactive, so you type your numbers into the cells.

# %%
EXAM_QUESTIONS = 50
EXAM_MINUTES = 60
PASS_MARK = 35   # 35 / 50 = 70%

def pace(questions=EXAM_QUESTIONS, minutes=EXAM_MINUTES, first_pass_seconds=55):
    """Seconds per question, and how much time a faster first pass leaves for review."""
    per_q = minutes * 60 / questions
    first_pass_total = questions * first_pass_seconds
    review_left = minutes * 60 - first_pass_total
    return {"seconds_per_question": per_q,
            "first_pass_total_s": first_pass_total,
            "review_time_s": review_left}

def score_mock(correct, total=EXAM_QUESTIONS, pass_mark=PASS_MARK, by_area=None):
    """Score a practice result. `correct` is a number, `by_area` an optional {area: (right, asked)}."""
    pct = 100 * correct / total
    out = {"correct": correct, "total": total, "percent": round(pct, 1),
           "exam_ready": correct >= pass_mark,
           "marks_short": max(0, pass_mark - correct)}
    if by_area:
        # weakest area first: that is where to revise
        ranked = sorted(by_area.items(), key=lambda kv: kv[1][0] / kv[1][1])
        out["weakest_areas"] = [f"{a}: {r}/{n}" for a, (r, n) in ranked[:3]]
    return out

print(pace())
print(score_mock(31, by_area={"Term 1": (8, 10), "Term 2": (6, 10), "Term 3": (9, 15), "Term 4": (8, 15)}))

# Fixed cases: these must always hold.
assert pace()["seconds_per_question"] == 72
assert pace()["review_time_s"] == 850
assert score_mock(35)["exam_ready"] and not score_mock(34)["exam_ready"]
assert score_mock(35)["percent"] == 70.0
check("timing helper matches the lesson (72 s per question, 850 s of review time)", True)

# %% [markdown]
# ### ✏️ Your turn: score your own practice
#
# Replace the numbers below with the result of one of your practice runs (for example a Mock Exam Prep module scaled up, or part of a full mock). Set `my_correct` to how many you got right out of 50. The cell must not crash if you leave it as `None`.

# %%
my_correct = None   # YOUR CODE HERE: a whole number from 0 to 50
my_areas = None     # optional: {"Term 1": (right, asked), ...}

if my_correct is None:
    print("Fill in my_correct above, then run this cell again.")
else:
    print(score_mock(my_correct, by_area=my_areas))

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_correct = 38
# my_areas = {"Term 1": (9, 10), "Term 2": (8, 10), "Term 3": (10, 15), "Term 4": (11, 15)}
# print(score_mock(my_correct, by_area=my_areas))
# ```
# 38 of 50 is 76%, above the 35 mark, so `exam_ready` is `True`. The weakest areas listed are where to spend your revision time before Week 20. These are made-up numbers; use your own.
# </details>

# %% [markdown]
# ## Part B — One computed exercise per term
#
# Each exercise comes from a calculation the lessons ask you to redo. Compute the answer in code, then run the Check.

# %% [markdown]
# ### Term 1: convolution output size (Week 4)
#
# The output side is `(n + 2p - k) / s + 1`, rounded down. Input side n = 32, kernel k = 3, padding p = 1, stride s = 2.
#
# ### ✏️ Your turn: what is the output side length?

# %%
answer_conv = None   # YOUR CODE HERE: an integer

# %%
true_conv = (32 + 2 * 1 - 3) // 2 + 1   # integer division rounds down
if answer_conv is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("convolution output side", answer_conv == true_conv, "use (n + 2p - k) // s + 1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_conv = (32 + 2*1 - 3) // 2 + 1   # = 31 // 2 + 1 = 15 + 1 = 16
# ```
# 32 + 2 − 3 = 31, and 31 divided by 2 rounds down to 15. Adding 1 gives 16, so the output is 16 × 16.
# </details>

# %% [markdown]
# ### Term 2: one attention weight (Weeks 7–8)
#
# One query compares against three keys with raw scores `[2.0, 1.0, 0.0]`. With key size d_k = 4, scaled dot-product attention divides the scores by √d_k and then applies softmax.
#
# ### ✏️ Your turn: what weight does the first key get? Round to 3 decimals.

# %%
answer_attn = None   # YOUR CODE HERE: a number between 0 and 1

# %%
scores = np.array([2.0, 1.0, 0.0]) / math.sqrt(4)
weights = np.exp(scores) / np.exp(scores).sum()
true_attn = float(weights[0])
if answer_attn is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("attention weight of the first key", abs(answer_attn - true_attn) < 0.001, "divide by sqrt(4)=2 first, then softmax")
assert abs(weights.sum() - 1) < 1e-9   # softmax weights always sum to 1

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# s = np.array([2.0, 1.0, 0.0]) / np.sqrt(4)    # [1.0, 0.5, 0.0]
# w = np.exp(s) / np.exp(s).sum()
# answer_attn = round(float(w[0]), 3)
# ```
# The scaled scores are 1.0, 0.5 and 0.0; softmax turns them into weights that sum to 1. The first key gets the largest weight because it had the largest score.
# </details>

# %% [markdown]
# ### Term 3: LoRA parameter count (Week 14)
#
# A LoRA adapter on one d × d matrix adds two matrices, A (d × r) and B (r × d). Take d = 768 and rank r = 16.
#
# ### ✏️ Your turn: how many trainable numbers does the adapter add, and what share of the full matrix is that (in percent)?

# %%
answer_lora = None        # YOUR CODE HERE: adapter parameter count
answer_lora_pct = None    # YOUR CODE HERE: percent of the full matrix

# %%
true_lora = 2 * 768 * 16
true_full = 768 * 768
true_pct = 100 * true_lora / true_full
print("(computed) adapter:", true_lora, "full matrix:", true_full, "percent:", round(true_pct, 2))
if answer_lora is None or answer_lora_pct is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("adapter parameter count", answer_lora == true_lora, "2 * d * r")
    check("percent of the full matrix", abs(answer_lora_pct - true_pct) < 0.01, "adapter / (d * d) * 100")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_lora = 2 * 768 * 16                 # 24,576
# answer_lora_pct = 100 * answer_lora / (768 * 768)   # about 4.17
# ```
# The adapter grows with d × r, not d × d, so a small rank trains far fewer numbers than the full matrix (589,824).
# </details>

# %% [markdown]
# ### Term 4: memory for model weights (Week 18)
#
# Weights take (number of parameters) × (bytes per number). A model has 7 billion parameters stored in 16-bit floats (2 bytes each). Use 1 GB = 10⁹ bytes. This counts weights only, not activations or a KV cache.
#
# ### ✏️ Your turn: how many GB for the weights in fp16? And how many if they were quantised to 4-bit (0.5 byte each)?

# %%
answer_mem16 = None   # YOUR CODE HERE: GB in fp16
answer_mem4 = None    # YOUR CODE HERE: GB at 4 bits

# %%
params = 7e9
true_mem16 = params * 2 / 1e9
true_mem4 = params * 0.5 / 1e9
if answer_mem16 is None or answer_mem4 is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("fp16 weights (GB)", abs(answer_mem16 - true_mem16) < 0.01, "parameters * bytes per number / 1e9")
    check("4-bit weights (GB)", abs(answer_mem4 - true_mem4) < 0.01, "4 bits = 0.5 byte")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_mem16 = 7e9 * 2 / 1e9     # 14.0 GB
# answer_mem4  = 7e9 * 0.5 / 1e9   # 3.5 GB
# ```
# Halving the bytes per number halves the memory; going from 16 bits to 4 bits is four times smaller.
# </details>

# %% [markdown]
# ## Part C — Capstone pipeline starter: Structural Inspection Assistant
#
# The capstone pipeline is **photo in → classify (crack / spalling / corrosion / none) → find checklist items (RAG) → draft summary for the engineer**. This notebook gives working code for two stages. Use it as a starting point and replace the pieces with your own.
#
# **Important:** the image below is a **synthetic picture drawn with PIL**. It is not a real inspection image, and every result is **illustrative only**. The AI-use rule of the project: **a licensed engineer makes every pass/fail decision; the assistant never signs off on safety.** That is why every result carries `needs_engineer_review: True`.

# %% [markdown]
# ### Step 1 — A synthetic test image
#
# We draw a grey, noisy "concrete" square with a dark jagged line. It is only a stand-in to test the code path.

# %%
from PIL import Image, ImageDraw

def make_synthetic_image(seed=SEED, size=224):
    rng = np.random.default_rng(seed)
    base = rng.normal(160, 12, (size, size)).clip(0, 255).astype(np.uint8)   # grey noisy "concrete"
    img = Image.fromarray(base).convert("RGB")
    d = ImageDraw.Draw(img)
    # a jagged dark line from top-left towards bottom-right
    x, y = 20, 30
    pts = [(x, y)]
    for _ in range(12):
        x += int(rng.integers(8, 18))
        y += int(rng.integers(10, 20))
        pts.append((min(x, size - 1), min(y, size - 1)))
    d.line(pts, fill=(40, 40, 40), width=3)
    return img

img = make_synthetic_image()
print("Synthetic image (NOT a real inspection photo):", img.size, img.mode)
assert img.size == (224, 224)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3, 3)); plt.imshow(img); plt.axis("off")
    plt.title("Synthetic test image"); plt.show()
except Exception as e:
    print("Plot skipped:", e)

# %% [markdown]
# ### Step 2 — The structured result
#
# Whatever model you use, the stage should return the same kind of dictionary, so the next stages do not care how the label was made. This function takes the four class scores and builds the record. It always sets `needs_engineer_review` to `True`; there is deliberately no way to switch it off.

# %%
LABELS = ["crack", "spalling", "corrosion", "none"]

def build_record(scores, source="clip-zero-shot (synthetic image, illustrative)"):
    """scores: dict label -> probability. Returns a structured, reviewable record."""
    assert set(scores) == set(LABELS), "need a score for each of the four labels"
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "label": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 3),
        "all_scores": {k: round(float(v), 3) for k, v in ranked},
        "status": "DRAFT",
        "needs_engineer_review": True,   # a licensed engineer makes every pass/fail decision
        "source": source,
    }

# A fixed, made-up example so the structure can be tested without any model.
demo = build_record({"crack": 0.6, "spalling": 0.2, "corrosion": 0.15, "none": 0.05})
print(demo)
assert demo["label"] == "crack" and demo["needs_engineer_review"] is True
check("record is a DRAFT that needs engineer review", demo["status"] == "DRAFT" and demo["needs_engineer_review"])

# %% [markdown]
# ### Step 3 — CLIP zero-shot classification
#
# CLIP (Radford et al., 2021) embeds images and text in the same space. For **zero-shot** classification we write one text prompt per class and pick the class whose text is closest to the image. We use `openai/clip-vit-base-patch32` (about 0.6 GB download, public, no key). Expect it to take a minute the first time. The synthetic line is crude, so do not be surprised if CLIP picks a different label than you expect: that is exactly why an engineer reviews every result.

# %%
# nb-test: skip
# (skipped in the automated test: the model download was not possible in the test environment)
import torch
from transformers import CLIPModel, CLIPProcessor

clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
clip_proc = CLIPProcessor.from_pretrained(clip_name)

PROMPTS = {
    "crack": "a photo of a crack in a concrete surface",
    "spalling": "a photo of spalling concrete with a broken, flaking surface",
    "corrosion": "a photo of rusted, corroded steel",
    "none": "a photo of a plain, undamaged concrete surface",
}

def classify_image(image):
    texts = [PROMPTS[l] for l in LABELS]
    inputs = clip_proc(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image[0]     # similarity of the image to each prompt
    probs = logits.softmax(dim=0).cpu().numpy()
    return build_record(dict(zip(LABELS, probs)))

record = classify_image(img)
print(record)
assert record["needs_engineer_review"] is True
assert abs(sum(record["all_scores"].values()) - 1) < 0.01

# %% [markdown]
# ### ✏️ Your turn: make the result safer to hand over
#
# Add a rule to `build_record`-style output: if the top confidence is below 0.5, add the field `"low_confidence": True` (and `False` otherwise). Write a small function `add_low_confidence(record)` that returns a new dictionary. Do not change `needs_engineer_review`.

# %%
def add_low_confidence(record):
    return None   # YOUR CODE HERE: return a copy of record with a "low_confidence" key

# %%
test_low = build_record({"crack": 0.4, "spalling": 0.3, "corrosion": 0.2, "none": 0.1})
test_high = build_record({"crack": 0.7, "spalling": 0.1, "corrosion": 0.1, "none": 0.1})
r_low, r_high = add_low_confidence(test_low), add_low_confidence(test_high)
if r_low is None:
    print("Fill in add_low_confidence above, then run this cell again.")
else:
    check("low confidence flagged", r_low.get("low_confidence") is True, "0.4 is below 0.5")
    check("high confidence not flagged", r_high.get("low_confidence") is False, "0.7 is not below 0.5")
    check("engineer review is still required", r_low["needs_engineer_review"] and r_high["needs_engineer_review"], "never change this field")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def add_low_confidence(record):
#     out = dict(record)                       # copy, do not change the original
#     out["low_confidence"] = record["confidence"] < 0.5
#     return out
# ```
# The flag helps the engineer see which results to look at first, but it never replaces the review: `needs_engineer_review` stays `True` for every record.
# </details>

# %% [markdown]
# ### Step 4 — Checklist retrieval with TF-IDF
#
# The next stage turns the label into a search query over checklist text. The passages below are **fictional sample text written for this notebook**, not a real inspection standard. TF-IDF scores words by how special they are to a passage, and cosine similarity ranks the passages (the retrieval idea behind RAG, Lewis et al., 2020; here the simplest version).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHECKLIST = [  # FICTIONAL SAMPLE TEXT, for practice only
    "Crack: measure the crack width with a gauge, note its length and location, and photograph it with a scale.",
    "Spalling: record the area of broken or flaking concrete, check whether reinforcement steel is exposed, and photograph the area.",
    "Corrosion: note rust stains on steel or concrete, record where the rust is, and check for loss of steel section.",
    "General: record the date, the structure and the element inspected, and the weather conditions.",
    "Escalation: flag anything unusual to the responsible engineer, who makes the pass or fail decision.",
]

vectorizer = TfidfVectorizer(stop_words="english")
doc_matrix = vectorizer.fit_transform(CHECKLIST)

def retrieve(query, k=2):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, doc_matrix)[0]
    order = np.argsort(-sims)[:k]
    return [(int(i), round(float(sims[i]), 3), CHECKLIST[i]) for i in order]

for rank, (i, s, text) in enumerate(retrieve("crack"), 1):
    print(f"{rank}. [{s}] {text[:70]}...")

# Fixed cases: each damage label should find its own passage first.
assert retrieve("crack")[0][0] == 0
assert retrieve("spalling")[0][0] == 1
assert retrieve("corrosion rust")[0][0] == 2
check("each label retrieves its own checklist passage first", True)

# %% [markdown]
# ### Step 5 — Join the stages
#
# This cell chains the stages for a single record: label in, checklist items out, a draft package for the engineer. We use the fixed example record so it runs without CLIP; swap in `classify_image(img)` once Step 3 works for you. There is no language model here. Adding a grounded summary (using only the retrieved text) is your job for the capstone.

# %%
def draft_package(record, k=2):
    items = retrieve(record["label"], k=k)
    return {
        **record,
        "checklist_items": [text for _, _, text in items],
        "note": "DRAFT. Illustrative only. A licensed engineer makes every pass/fail decision.",
    }

pkg = draft_package(demo)
for key, val in pkg.items():
    print(f"{key}: {val}")
assert pkg["needs_engineer_review"] is True and len(pkg["checklist_items"]) == 2

# %% [markdown]
# ### Capstone build checklist
#
# - [ ] The pipeline runs end to end on at least three photos.
# - [ ] Every output is labelled DRAFT and carries a "needs engineer review" field.
# - [ ] You have a confusion matrix and a failure log for your classifier.
# - [ ] Your README says what the assistant does and does not do, and which AI assistance you used.
# - [ ] Your team knows who is Team lead, Technical lead and Responsible-AI lead.

# %% [markdown]
# ## Wrap-up
#
# - You can time a 50-question, 60-minute mock (72 seconds per question) and check your score against the 35/50 mark.
# - You recomputed a convolution size, an attention weight, a LoRA parameter count and a weight-memory estimate.
# - You have a starter for two capstone stages, a structured result that always requires engineer review, and TF-IDF retrieval over fictional checklist text.
#
# **Reflect:** double-click to write two sentences: which term scored lowest for you, and what will you revise before Week 20?
#
# Files in `/content` vanish when the session ends, so download anything you want to keep. Submit this notebook the way your instructor asks.
