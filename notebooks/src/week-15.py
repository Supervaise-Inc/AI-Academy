# %% [markdown]
# # Week 15 — Pre-final Exam review + Capstone proposal: practice notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** CPU is enough, nothing is downloaded. **Time:** about 2 to 3 hours.
#
# This week has no lab. Instead, this is a **practice notebook**: for each of Weeks 11 to 14 you get a short reminder, a few small calculations to do in code, a Check against the true value, and a worked answer. At the end, a **capstone proposal helper** turns the proposal checklist into a small data structure and prints an outline for the Structural Inspection Assistant.
#
# ### What you will do
# - Week 11: count ViT patch tokens and patch dimensions, and rank labels by CLIP-style cosine similarity.
# - Week 12: compute the output size of a transpose convolution.
# - Week 13: compute ᾱ_t and the signal and noise weights.
# - Week 14: compute LoRA parameter counts and model memory.
# - Capstone: fill in a proposal checklist and print an outline using the marking criteria.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you have tried. This notebook never states exam dates, formats or weights: your instructor announces those.

# %% [markdown]
# ## Setup

# %%
import os, math
import numpy as np

SEED = 0
np.random.seed(SEED)
QUICK = bool(os.environ.get("NB_TEST"))   # unused here, kept for consistency with the other notebooks
device = "cpu"                            # this notebook only does arithmetic
print("Device:", device)

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# A fixed helper cases test so a broken environment is noticed
assert (224 // 16) ** 2 == 196 and 16 * 16 * 3 == 768, "environment sanity check failed"

# %% [markdown]
# ## Week 11: images become tokens
#
# **Reminder.** A Vision Transformer cuts an image into patches. With side `S` and patch size `P` there are (S ÷ P)² patches. Each patch of a colour image holds P × P × 3 numbers, which a linear layer projects to the model dimension. A learned CLS token is added in front.

# %%
def n_patches(side, patch):
    return (side // patch) ** 2

def patch_dim(patch, channels=3):
    return patch * patch * channels

print("224x224, patch 16:", n_patches(224, 16), "patches, each", patch_dim(16), "numbers; with CLS:", n_patches(224, 16) + 1)
print("8x8, patch 2:", n_patches(8, 2), "| patch 4:", n_patches(8, 4), "| patch 8:", n_patches(8, 8))

# %% [markdown]
# ### ✏️ Your turn: tokens and dimensions
#
# A ViT reads a **384×384** colour image with **32×32** patches. Store `answer = (patches, numbers_per_patch, tokens_including_CLS)`.

# %%
answer = None  # YOUR CODE HERE: a tuple of three integers

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    want = (n_patches(384, 32), patch_dim(32), n_patches(384, 32) + 1)
    check("patch count, patch size and token count", tuple(answer) == want, "patches = (side // patch)**2; numbers = patch*patch*3; tokens = patches + 1")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = (n_patches(384, 32), patch_dim(32), n_patches(384, 32) + 1)   # (144, 3072, 145)
# ```
# 384 ÷ 32 = 12 patches per side, 12 × 12 = 144; each patch has 32 × 32 × 3 = 3072 numbers; the CLS token makes 145 tokens.
# </details>

# %% [markdown]
# **CLIP-style zero-shot ranking.** CLIP puts an image and its captions in one space. For zero-shot classification you embed one prompt per label and pick the label with the highest **cosine similarity** to the image embedding: cos(a, b) = (a · b) / (‖a‖ ‖b‖). The vectors below are made-up 2-D numbers for practice (real embeddings have hundreds of dimensions).

# %%
def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

image_vec = np.array([2.0, 3.2])                     # illustrative
label_vecs = {"a photo of a cat": np.array([4.0, 1.6]),
              "a photo of a dog": np.array([2.4, 4.0]),
              "a photo of a car": np.array([-1.4, 4.0])}
for name, v in label_vecs.items():
    print(f"{name:<20} cosine = {cosine(image_vec, v):.3f}")
assert abs(cosine([1, 0], [0, 1])) < 1e-12 and abs(cosine([1, 1], [2, 2]) - 1) < 1e-12

# %% [markdown]
# ### ✏️ Your turn: rank the crack labels
#
# Here are made-up 2-D embeddings for an inspection photo and four labels. Find the label with the highest cosine similarity and store its name (a string from `labels`) in `answer`.

# %%
photo = np.array([3.0, 1.0])                         # illustrative
labels = {"crack": np.array([5.0, 1.0]), "spalling": np.array([1.0, 4.0]),
          "corrosion": np.array([-2.0, 3.0]), "none": np.array([-3.0, -1.0])}
answer = None  # YOUR CODE HERE: the best label as a string

# %% [markdown]
# ### Check

# %%
scores = {k: cosine(photo, v) for k, v in labels.items()}
best = max(scores, key=scores.get)
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("best label", answer == best, "compute cosine(photo, v) for every label and take the largest")
    print({k: round(s, 3) for k, s in scores.items()})

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = max(labels, key=lambda k: cosine(photo, labels[k]))   # "crack"
# ```
# Cosine similarity only compares the *direction* of the vectors, not their length. The photo points almost the same way as the "crack" label. These numbers are made up for practice; in the capstone a real CLIP model would give the embeddings.
# </details>

# %% [markdown]
# ## Week 12: transpose convolution
#
# **Reminder.** A transpose convolution is learnable upsampling. Without padding, the output side is **(n − 1) × stride + k**, where n is the input side and k the kernel side. A 2×2 input with a 3×3 kernel and stride 2 gives 5×5.

# %%
def tconv_side(n, k, stride=1, padding=0):
    return (n - 1) * stride + k - 2 * padding

print("2x2, k=3, stride 2:", tconv_side(2, 3, 2))
assert tconv_side(2, 3, 2) == 5

# We can also build the output by hand: stamp a scaled copy of the kernel for each input value
def tconv2d(x, kernel, stride=1):
    n, k = x.shape[0], kernel.shape[0]
    out = np.zeros((tconv_side(n, k, stride),) * 2)
    for i in range(n):
        for j in range(n):
            out[i * stride:i * stride + k, j * stride:j * stride + k] += x[i, j] * kernel
    return out

x = np.array([[1, 2], [3, 4]]); kern = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
print(tconv2d(x, kern, 2))

# %% [markdown]
# ### ✏️ Your turn: size of an upsampling layer
#
# A generator layer has a **4×4** input, a **3×3** kernel, stride **2** and no padding. What is the output side? Then a second layer takes that output with a **4×4** kernel and stride 2. What is the final side? Store `answer = (first_side, second_side)`.

# %%
answer = None  # YOUR CODE HERE: a tuple of two integers

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    s1 = tconv_side(4, 3, 2)
    want = (s1, tconv_side(s1, 4, 2))
    check("output sides", tuple(answer) == want, "side = (n - 1) * stride + k")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# s1 = (4 - 1) * 2 + 3      # 9
# s2 = (s1 - 1) * 2 + 4     # 20
# answer = (s1, s2)
# ```
# The first layer gives 9×9, and the second (9 − 1) × 2 + 4 = 20, so 20×20.
# </details>

# %% [markdown]
# ## Week 13: diffusion
#
# **Reminder.** With a linear β schedule from 0.0001 to 0.02 over T = 1000 steps, ᾱ_t is the running product of (1 − β). The forward step is x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε.

# %%
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alpha_bar = np.cumprod(1.0 - betas)       # alpha_bar[t-1] is step t
print("alpha_bar(300) =", round(alpha_bar[299], 4), "| alpha_bar(1000) =", alpha_bar[999])
assert abs(alpha_bar[299] - 0.3964) < 5e-4 and 3e-5 < alpha_bar[999] < 5e-5

# %% [markdown]
# ### ✏️ Your turn: signal and noise at t = 100
#
# From the schedule, compute ᾱ at **t = 100**, then the weight on x₀ (√ᾱ) and the weight on the noise (√(1 − ᾱ)). Store `answer = (alpha_bar_100, signal_weight, noise_weight)`.

# %%
answer = None  # YOUR CODE HERE: a tuple of three numbers

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    ab = alpha_bar[99]
    ok = all(abs(a - b) < 0.002 for a, b in zip(answer, (ab, np.sqrt(ab), np.sqrt(1 - ab))))
    check("alpha_bar, signal and noise at t = 100", ok, "remember alpha_bar[99] is step 100; signal = sqrt(ab); noise = sqrt(1 - ab)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# ab = alpha_bar[99]
# answer = (ab, np.sqrt(ab), np.sqrt(1 - ab))   # about (0.8970, 0.9471, 0.3209)
# ```
# At t = 100 most of the picture is still there. At t = 300 the weights are 0.63 and 0.78, so noise already outweighs the signal.
# </details>

# %% [markdown]
# ### ✏️ Your turn: which ᾱ?
#
# In the forward step suppose ᾱ_t = 0.25. How much of the clean image is kept (the multiplier on x₀), and what is the multiplier on the noise? Store `answer = (keep, noise)` and compare it with the quiz question in the lesson.

# %%
answer = None  # YOUR CODE HERE: a tuple (keep, noise)

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("multipliers for alpha_bar = 0.25", abs(answer[0] - 0.5) < 1e-6 and abs(answer[1] - math.sqrt(0.75)) < 1e-3,
          "keep = sqrt(0.25), noise = sqrt(1 - 0.25)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = (math.sqrt(0.25), math.sqrt(1 - 0.25))   # (0.5, 0.866...)
# ```
# The multiplier is √ᾱ, not ᾱ itself: √0.25 = 0.5, and the noise multiplier is √0.75 ≈ 0.87.
# </details>

# %% [markdown]
# ## Week 14: LoRA and memory
#
# **Reminder.** LoRA freezes W (d × d) and trains B (d × r) and A (r × d): **2 × d × r** numbers. Weights take parameters × bytes per parameter (FP32 4, FP16 2, INT8 1, INT4 0.5).

# %%
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)

print("4096x4096, r=8:", lora_params(4096, 4096, 8), "of", 4096 * 4096, f"= {lora_params(4096, 4096, 8) / 4096 ** 2:.4%}")
print("GPT-2 small, 12 layers of c_attn (768 -> 2304), r=8:", 12 * lora_params(768, 2304, 8))
assert lora_params(4096, 4096, 8) == 65_536 and 12 * lora_params(768, 2304, 8) == 294_912

# %% [markdown]
# ### ✏️ Your turn: LoRA on a bigger matrix
#
# A layer has a **2048 × 2048** weight matrix and you use rank **r = 16**. How many LoRA numbers does it add, and what percentage of the full matrix is that? Store `answer = (count, percent)` (percent as a number such as 1.5 for 1.5%).

# %%
answer = None  # YOUR CODE HERE: a tuple (count, percent)

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    cnt = lora_params(2048, 2048, 16)
    pct = 100 * cnt / (2048 * 2048)
    check("LoRA count and percentage", answer[0] == cnt and abs(answer[1] - pct) < 0.01, "count = 2 * d * r; percent = 100 * count / d**2")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# cnt = 2 * 2048 * 16                 # 65,536
# answer = (cnt, 100 * cnt / 2048**2) # 1.5625 %
# ```
# The share is 2r ÷ d = 32 ÷ 2048 = 1.5625%. It improves as the matrix gets larger.
# </details>

# %% [markdown]
# ### ✏️ Your turn: will it fit?
#
# A 7-billion-parameter model: how many GB do the weights need in **INT4**, and how many GB does full fine-tuning need with the 16-bytes-per-parameter Adam estimate? Store `answer = (int4_gb, adam_gb)`.

# %%
answer = None  # YOUR CODE HERE: a tuple (int4_gb, adam_gb)

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("7B memory", abs(answer[0] - 7e9 * 0.5 / 1e9) < 0.01 and abs(answer[1] - 7e9 * 16 / 1e9) < 0.1,
          "INT4 is half a byte per parameter; the Adam estimate is 16 bytes per parameter")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = (7e9 * 0.5 / 1e9, 7e9 * 16 / 1e9)   # (3.5, 112.0)
# ```
# 3.5 GB for 4-bit weights, and about 112 GB for full training before counting activations. This is why LoRA, which needs gradients only for the small adapter, makes fine-tuning possible on a single GPU.
# </details>

# %% [markdown]
# ### ✏️ Your turn: place each technique on the pipeline
#
# For each activity, choose `"pretrain"` (pre-training, including continued), `"sft"` (fine-tuning or instruction tuning) or `"rlhf"` (preference training). Fill in the dictionary.

# %%
activities = {
    "Predict the next token over a huge web-scale text collection": None,
    "Train on 10,000 prompt-answer pairs written by people": None,
    "Train a reward model from human rankings of two answers": None,
    "Keep training on a company's legal documents with the next-token goal": None,
    "Train only small adapter matrices A and B on 15 examples": None,   # LoRA is a method; which stage is it doing?
}  # YOUR CODE HERE: replace each None with "pretrain", "sft" or "rlhf"

# %% [markdown]
# ### Check

# %%
truth = {
    "Predict the next token over a huge web-scale text collection": "pretrain",
    "Train on 10,000 prompt-answer pairs written by people": "sft",
    "Train a reward model from human rankings of two answers": "rlhf",
    "Keep training on a company's legal documents with the next-token goal": "pretrain",
    "Train only small adapter matrices A and B on 15 examples": "sft",
}
if any(v is None for v in activities.values()):
    print("Fill in every answer above, then run this cell again.")
else:
    for k, v in truth.items():
        check(k[:60], activities[k] == v, "think about the data and the goal of the stage")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# activities = dict(truth)   # pretrain, sft, rlhf, pretrain, sft
# ```
# Continued pre-training keeps the same next-token goal, so it sits with pre-training. The reward model is step 2 of RLHF. LoRA is a cheap way of doing the training, not a stage: with 15 prompt-answer examples the stage is fine-tuning.
# </details>

# %% [markdown]
# ## Capstone proposal helper: the Structural Inspection Assistant
#
# The capstone is a team project: a Structural Inspection Assistant that triages inspection photos and drafts a summary for an engineer to review. The pipeline is: **photo in → classify (crack, spalling, corrosion or none) → find checklist items with RAG → draft a summary → the engineer decides**. One rule is fixed: **a licensed engineer makes every pass/fail decision, and the assistant never signs off on safety.**
#
# The code below stores the proposal checklist and the marking criteria as data, so you can see at a glance which parts of your proposal are still open and where the marks are. Your instructor announces the proposal format, length and deadline: use this only as a planning aid.

# %%
CRITERIA = {                        # marking criteria and weights from the course outline (they add up to 100%)
    "Problem and proposal": 15,
    "Technical implementation": 40,
    "Responsible AI": 15,
    "Presentation": 20,
    "Documentation": 10,
}
assert sum(CRITERIA.values()) == 100, "the criteria weights should add up to 100"

CHECKLIST = [                       # (question, which criterion it mainly serves)
    ("Problem: who has the problem, and what slows them down today?", "Problem and proposal"),
    ("Users and the human in the loop: who is the engineer, and what exactly do they review?", "Responsible AI"),
    ("Pipeline: which model does each step, and what goes in and out of it?", "Technical implementation"),
    ("Data plan: where do your test photos and checklist text come from, and are you allowed to use them? Mark sample text as fictional.", "Technical implementation"),
    ("Evaluation: how will you know the classifier and the retrieval work?", "Technical implementation"),
    ("Responsible AI: what can go wrong, and which guardrail or human check prevents each problem?", "Responsible AI"),
    ("Roles and plan: who is Team lead, Technical lead and Responsible-AI lead, and what is done by Week 19?", "Documentation"),
]
for crit in {c for _, c in CHECKLIST}:
    assert crit in CRITERIA

print("Marking criteria:")
for k, w in CRITERIA.items():
    print(f"  {w:>3}%  {k}")

# %% [markdown]
# ### ✏️ Your turn: draft your answers
#
# Fill in a short answer (one or two sentences) for each checklist item in `my_answers`, using the same order as `CHECKLIST`. Leave `None` for items you have not decided yet. Short and concrete is better than long and vague. Mark any example text as **fictional**. Do not paste real inspection photos, names or other real data into this notebook.

# %%
my_answers = [None] * len(CHECKLIST)
# YOUR CODE HERE, for example:
# my_answers[0] = "Site engineers spend a lot of time sorting photos before they can review them."
# my_answers[2] = "CLIP zero-shot classifies the photo; RAG finds checklist items; an LLM drafts the summary."

# %% [markdown]
# ### Check

# %%
def proposal_status(answers):
    done = [a is not None and str(a).strip() != "" for a in answers]
    return done, sum(done)

done, n_done = proposal_status(my_answers)
assert len(my_answers) == len(CHECKLIST)
print(f"{n_done} of {len(CHECKLIST)} checklist items answered.")
for ok, (q, c) in zip(done, CHECKLIST):
    print(("✓ " if ok else "✗ ") + q.split(":")[0] + "  (" + c + ")")
if n_done == 0:
    print("Fill in at least one answer above, then run this cell again.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_answers[0] = "Engineers lose time sorting inspection photos before the real review starts."
# my_answers[1] = "A licensed engineer reads every draft and makes each pass/fail decision."
# my_answers[2] = "CLIP zero-shot classifies the photo, RAG finds checklist items, an LLM drafts the summary."
# ```
# These are example sentences to show the style, not a complete proposal. Your own answers should name your real data, your real evaluation and your team members.
# </details>

# %% [markdown]
# Now print the proposal outline. It lists the pipeline, then every checklist item with your answer (or `TODO`), grouped by the criterion it mainly serves, with the weight next to each heading.

# %%
def proposal_outline(answers):
    lines = ["PROPOSAL OUTLINE: Structural Inspection Assistant", "",
             "Pipeline: photo -> classify (crack / spalling / corrosion / none) -> find checklist items (RAG) -> draft summary -> engineer decides",
             "AI-use rule: a licensed engineer makes every pass/fail decision; the assistant never signs off on safety.", ""]
    for crit, weight in CRITERIA.items():
        items = [(q, a) for (q, c), a in zip(CHECKLIST, answers) if c == crit]
        lines.append(f"[{weight}%] {crit}")
        if not items:
            lines.append("   (covered by your demo and the whole project, not by one checklist question)")
        for q, a in items:
            text = a if a is not None and str(a).strip() else "TODO"
            lines.append(f"   - {q.split(':')[0]}: {text}")
        lines.append("")
    return "\n".join(lines)

outline = proposal_outline(my_answers)
print(outline)
assert "licensed engineer" in outline and sum(CRITERIA.values()) == 100

# %% [markdown]
# **Where are the marks?** Technical implementation is the biggest share, so check that your plan says how you will test the classifier and the retrieval.

# %%
biggest = max(CRITERIA, key=CRITERIA.get)
print(f"Largest criterion: {biggest} ({CRITERIA[biggest]}%)")
print(f"Problem + Responsible AI together: {CRITERIA['Problem and proposal'] + CRITERIA['Responsible AI']}%")

# %% [markdown]
# ## Wrap-up
#
# - You can count ViT patch tokens ((side ÷ patch)²), rank CLIP labels by cosine similarity, and compute a transpose-convolution output size ((n − 1) × stride + k).
# - You can compute ᾱ_t and the signal and noise weights √ᾱ_t and √(1 − ᾱ_t), LoRA parameter counts (2 × d × r per square matrix) and model memory (parameters × bytes).
# - You have a capstone proposal outline with the marking criteria next to your answers, and a clear list of what is still TODO.
#
# **Reflection.** Double-click here to write two sentences: which of the four calculations (patches, transpose convolution, ᾱ, LoRA) do you still find hardest and why; and one way the Structural Inspection Assistant could go wrong, with one check that would catch it.
#
# **Before you leave:** files in `/content` vanish when the Colab session ends. Download anything you want to keep (File → Download .ipynb, or copy your outline into your team's document), and submit the notebook the way your instructor asks. Your instructor announces the Pre-final exam details and the proposal requirements.
