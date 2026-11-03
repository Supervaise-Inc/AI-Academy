# %% [markdown]
# # Week 20 — Final Exam Review + Demo Day: practice notebook
#
# Built for **Google Colab**. First choose **File → Save a copy in Drive** so your answers are saved in your own copy.
#
# **Runtime:** CPU only. **Time:** about 90 minutes.
#
# There is no lab this week, so this is a **practice notebook**: retrieval practice with answers you compute in code, followed by a Demo Day section that helps you time and rehearse your talk.
#
# **What you will do**
# - Compute cumulative practice answers, one or two per term of the course.
# - Use a timing helper that prints a 5-part demo schedule for the total length you choose and flags overruns.
# - Walk through a checklist of the capstone marking criteria and turn a self-assessment into points.
#
# **How this notebook works:** run cells top to bottom with ▶. **Your turn** cells are for you. A **Check** cell tells you if you are right. Open **Show a possible answer** only after you tried.

# %%
import math, random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
device = "cpu"
print("Device:", device, "(this notebook needs no GPU)")

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# A fixed case, so a broken environment is noticed straight away.
assert (4 + 0 - 2) // 1 + 1 == 3
check("setup works", True)

# %% [markdown]
# ## Part A — Prelim: how machines learn (Weeks 1–5)
#
# **Reminder.** A model is a function with adjustable numbers; training lowers a loss using gradients from backpropagation. A CNN slides a small kernel over an image with weights shared at every position.
#
# ### Exercise 1: convolution by hand
# Output side = (n + 2p − k) / s + 1. A 6 × 6 input, 3 × 3 kernel, stride 1, no padding.
#
# ### ✏️ Your turn: what is the output side length, and how many parameters does the kernel have (one channel, no bias)?

# %%
answer_side = None      # YOUR CODE HERE
answer_kparams = None   # YOUR CODE HERE

# %%
true_side = (6 + 0 - 3) // 1 + 1
true_kparams = 3 * 3
if answer_side is None or answer_kparams is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("output side", answer_side == true_side, "(n + 2p - k) // s + 1")
    check("kernel parameters", answer_kparams == true_kparams, "the kernel is 3 x 3, shared at every position")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_side = (6 + 0 - 3) // 1 + 1   # 4
# answer_kparams = 3 * 3               # 9
# ```
# The output is 4 × 4. The same 9 weights are reused at every position, which is the weight sharing that saves parameters.
# </details>

# %% [markdown]
# ### Exercise 2: sigmoid and the vanishing gradient
# The derivative of the sigmoid is at most 0.25 (at input 0). If the gradient passes back through five sigmoid layers, each time multiplied by at most 0.25, the factor is at most 0.25⁵.
#
# ### ✏️ Your turn: compute 0.25 to the power 5.

# %%
answer_vanish = None   # YOUR CODE HERE

# %%
sig = lambda x: 1 / (1 + math.exp(-x))
slope0 = sig(0) * (1 - sig(0))          # the sigmoid derivative at 0
assert abs(slope0 - 0.25) < 1e-12       # confirms the 0.25 above
true_vanish = slope0 ** 5
if answer_vanish is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("0.25 ** 5", abs(answer_vanish - true_vanish) < 1e-9, "multiply 0.25 by itself five times")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_vanish = 0.25 ** 5    # 0.0009765625
# ```
# About one thousandth: early layers receive a tiny gradient, so they learn very slowly. That is the vanishing gradient.
# </details>

# %% [markdown]
# ## Part B — Midterm: language and Transformers (Weeks 6–10)
#
# **Reminder.** Attention compares a query with keys and mixes the values by softmax weights. Scores are divided by √d_k so the softmax does not saturate. A causal mask stops a position from looking at later positions.
#
# ### Exercise 3: the attention scale and a masked row
# With d_k = 64, the scale factor is 1/√d_k. Then take scores `[2.0, 1.0, 0.5]` for position 1 of a three-token sentence, apply the causal mask (position 1 may see only positions 0 and 1) and softmax the remaining scores.
#
# ### ✏️ Your turn: give the scale factor and the attention weight on position 0.

# %%
answer_scale = None    # YOUR CODE HERE
answer_w0 = None       # YOUR CODE HERE: weight on position 0 after masking (3 decimals)

# %%
true_scale = 1 / math.sqrt(64)
row = np.array([2.0, 1.0, 0.5])
row = row * true_scale
masked = row.copy(); masked[2] = -np.inf          # position 2 is in the future: blocked
e = np.exp(masked - masked.max())
w = e / e.sum()
true_w0 = float(w[0])
assert abs(w.sum() - 1) < 1e-9 and w[2] == 0     # masked weight is exactly 0
if answer_scale is None or answer_w0 is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("scale factor", abs(answer_scale - true_scale) < 1e-6, "1 / sqrt(64)")
    check("weight on position 0", abs(answer_w0 - true_w0) < 0.001, "scale first, mask position 2, then softmax over two scores")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# scale = 1 / np.sqrt(64)                          # 0.125
# s = np.array([2.0, 1.0, 0.5]) * scale
# s[2] = -np.inf                                   # causal mask
# w = np.exp(s - s.max()); w = w / w.sum()
# answer_w0 = round(float(w[0]), 3)
# ```
# The masked position gets weight exactly 0, and the remaining weights still sum to 1.
# </details>

# %% [markdown]
# ### Exercise 4: Chinchilla-style token budget
# The Chinchilla result (Hoffmann et al., 2022) is often summarised as about 20 training tokens per parameter. This is a rule of thumb, not an exact law. For a 3-billion-parameter model:
#
# ### ✏️ Your turn: how many tokens (in billions) does the rule of thumb suggest?

# %%
answer_tokens_b = None   # YOUR CODE HERE (billions of tokens)

# %%
true_tokens_b = 3 * 20
if answer_tokens_b is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("tokens in billions", abs(answer_tokens_b - true_tokens_b) < 1e-9, "3 billion parameters * 20 tokens each")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_tokens_b = 3 * 20    # 60 billion tokens
# ```
# 3 × 10⁹ parameters × 20 = 60 × 10⁹ tokens.
# </details>

# %% [markdown]
# ## Part C — Pre-final: images and generation (Weeks 11–15)
#
# **Reminder.** A Vision Transformer cuts an image into patches and treats each as a token. LoRA trains small added matrices instead of the whole model.
#
# ### Exercise 5: ViT patches and LoRA
# A 224 × 224 image is cut into 16 × 16 patches. A LoRA adapter with rank r = 4 is added to a 1024 × 1024 matrix.
#
# ### ✏️ Your turn: how many patch tokens, and how many LoRA parameters (2 × d × r)?

# %%
answer_patches = None   # YOUR CODE HERE
answer_lora = None      # YOUR CODE HERE

# %%
true_patches = (224 // 16) ** 2
true_lora = 2 * 1024 * 4
if answer_patches is None or answer_lora is None:
    print("Fill in both answers above, then run this cell again.")
else:
    check("patch tokens", answer_patches == true_patches, "(224 / 16) squared")
    check("LoRA parameters", answer_lora == true_lora, "2 * d * r")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_patches = (224 // 16) ** 2   # 14 * 14 = 196
# answer_lora = 2 * 1024 * 4          # 8192
# ```
# The full matrix has 1,048,576 numbers, so the adapter is under 1% of it.
# </details>

# %% [markdown]
# ## Part D — Final: apps, agents and deployment (Weeks 16–20)
#
# **Reminder.** A model's weights take (parameters × bytes per number) of memory. Lower precision means fewer bytes per number. The mock exam is 50 questions in 60 minutes with a pass mark of 35/50 (70%).
#
# ### Exercise 6: memory per precision and the pass mark
# A 7-billion-parameter model. Use 1 GB = 10⁹ bytes and count weights only. Precisions: fp32 (4 bytes), fp16 (2), int8 (1), int4 (0.5).
#
# ### ✏️ Your turn: fill `answer_mem` with a dictionary of GB per precision, and `answer_pass` with the number of marks that equals 70% of 50.

# %%
answer_mem = None    # YOUR CODE HERE: {"fp32": ..., "fp16": ..., "int8": ..., "int4": ...}
answer_pass = None   # YOUR CODE HERE

# %%
bytes_per = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
true_mem = {k: 7e9 * b / 1e9 for k, b in bytes_per.items()}
true_pass = 0.70 * 50
print("(computed)", true_mem, true_pass)
if answer_mem is None or answer_pass is None:
    print("Fill in both answers above, then run this cell again.")
else:
    ok = all(abs(answer_mem.get(k, -1) - v) < 0.01 for k, v in true_mem.items())
    check("memory per precision", ok, "parameters * bytes per number / 1e9, for each of the four precisions")
    check("pass mark in marks", abs(answer_pass - true_pass) < 1e-9, "0.70 * 50")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer_mem = {k: 7e9 * b / 1e9 for k, b in {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}.items()}
# # {"fp32": 28.0, "fp16": 14.0, "int8": 7.0, "int4": 3.5}
# answer_pass = 0.70 * 50   # 35.0
# ```
# Each halving of the bytes halves the memory. 70% of 50 marks is 35.
# </details>

# %% [markdown]
# ## Part E — Demo Day
#
# Your demo is **a short story: the problem, a live demo, how it works, your responsible-AI choices and what you would do next**. Your instructor announces the length of each slot, so type that length below. The suggested split is only a starting point of ours; change the shares to fit your team.
#
# The helper prints a schedule and, if you give it the times from a rehearsal, flags every part that ran over.

# %%
PARTS = ["1. The problem", "2. Live demo", "3. How it works", "4. Responsible-AI choices", "5. What next"]
SHARES = [0.15, 0.35, 0.25, 0.15, 0.10]       # a suggestion (fractions of the total), not a rule
assert abs(sum(SHARES) - 1) < 1e-9

def fmt(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"

def demo_schedule(total_minutes, rehearsal_seconds=None, shares=SHARES):
    """Print the schedule. rehearsal_seconds: optional list of 5 times you measured."""
    assert len(shares) == len(PARTS) and abs(sum(shares) - 1) < 1e-9, "shares must sum to 1"
    total = total_minutes * 60
    budget = [s * total for s in shares]
    print(f"Total slot: {total_minutes} min ({int(total)} s)")
    start = 0
    over_any = False
    for i, (name, b) in enumerate(zip(PARTS, budget)):
        line = f"{fmt(start)}-{fmt(start + b)}  {name:<28} budget {fmt(b)}"
        if rehearsal_seconds is not None:
            used = rehearsal_seconds[i]
            flag = "OVER by " + fmt(used - b) if used > b else "ok"
            over_any = over_any or used > b
            line += f"   rehearsal {fmt(used)}  {flag}"
        print(line)
        start += b
    if rehearsal_seconds is not None:
        spent = sum(rehearsal_seconds)
        verdict = "OVER the slot by " + fmt(spent - total) if spent > total else "within the slot"
        print(f"Rehearsal total {fmt(spent)}: {verdict}")
    return budget

# Example with a made-up slot of 10 minutes and made-up rehearsal times.
b = demo_schedule(10, rehearsal_seconds=[80, 240, 170, 80, 50])
assert abs(sum(b) - 600) < 1e-9
assert abs(b[1] - 210) < 1e-9          # 35% of 600 s
check("the helper splits 10 minutes into 90, 210, 150, 90 and 60 seconds", [round(x) for x in b] == [90, 210, 150, 90, 60])

# %% [markdown]
# ### ✏️ Your turn: schedule your own slot
#
# Set `my_minutes` to the slot length your instructor gave you. After a rehearsal, set `my_times` to the five times (in seconds) you measured with a timer.

# %%
my_minutes = None   # YOUR CODE HERE: e.g. the slot length in minutes
my_times = None     # optional after a rehearsal: [problem, demo, how, responsible_ai, next] in seconds

if my_minutes is None:
    print("Fill in my_minutes above, then run this cell again.")
else:
    demo_schedule(my_minutes, my_times)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_minutes = 8
# my_times = [70, 190, 130, 60, 40]
# demo_schedule(my_minutes, my_times)
# ```
# For 8 minutes the budgets are 72, 168, 120, 72 and 48 seconds, so in this made-up rehearsal the live demo (190 s) and "How it works" (130 s) ran over, and the total of 490 s is 10 s over the 480 s slot. Cut or rehearse those parts. The numbers are made up; use your own.
# </details>

# %% [markdown]
# ### Capstone criteria checklist
#
# The capstone (the Structural Inspection Assistant) is marked on five criteria with these weights, from the course outline: problem and proposal 15%, technical implementation 40%, responsible AI 15%, presentation 20%, documentation 10%. Rate your team honestly from 0 to 100% on each criterion; the helper converts that to points out of 100. Each criterion's points are your rating times its weight.

# %%
CRITERIA = {
    "Problem and proposal": 15,
    "Technical implementation": 40,
    "Responsible AI": 15,
    "Presentation": 20,
    "Documentation": 10,
}
assert sum(CRITERIA.values()) == 100

QUESTIONS = {
    "Problem and proposal": "Can you say who inspects, what is slow or risky, and what the assistant does and does not do?",
    "Technical implementation": "Does one photo go through the whole pipeline live, and do you have a backup (recording or screenshots)?",
    "Responsible AI": "Is every output a DRAFT, does a licensed engineer make every pass/fail decision, and did you show one failure honestly?",
    "Presentation": "Does every team member speak, and did you rehearse with a timer?",
    "Documentation": "Does the README explain the pipeline, the tests and the AI assistance you used?",
}

def self_assess(ratings):
    """ratings: {criterion: 0-100 percent}. Returns points out of 100 per criterion and the total."""
    points = {c: round(ratings.get(c, 0) / 100 * w, 1) for c, w in CRITERIA.items()}
    return points, round(sum(points.values()), 1)

for c, q in QUESTIONS.items():
    print(f"{c} ({CRITERIA[c]}%): {q}")

pts, total = self_assess({c: 100 for c in CRITERIA})
assert total == 100
pts, total = self_assess({"Responsible AI": 60})
assert pts["Responsible AI"] == 9.0     # 0.60 * 15, the lesson's worked example

# %% [markdown]
# ### ✏️ Your turn: rate your team
#
# Fill in a percentage from 0 to 100 for each criterion. Then work out which criterion is worth the most points and where you would lose the most.

# %%
my_ratings = None   # YOUR CODE HERE: {"Problem and proposal": 80, "Technical implementation": 70, ...}

if my_ratings is None:
    print("Fill in my_ratings above, then run this cell again.")
else:
    pts, total = self_assess(my_ratings)
    print(pts, "total:", total)
    lost = {c: round(w - pts[c], 1) for c, w in CRITERIA.items()}
    worst = max(lost, key=lost.get)
    print("Most points lost in:", worst, "->", lost[worst])

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_ratings = {"Problem and proposal": 80, "Technical implementation": 70,
#               "Responsible AI": 60, "Presentation": 90, "Documentation": 50}
# ```
# These made-up ratings give 12.0 + 28.0 + 9.0 + 18.0 + 5.0 = 72.0 points. Technical implementation loses the most (12 points) because it carries the largest weight, which is why a working live demo matters. Use your own honest ratings.
# </details>

# %% [markdown]
# ### Demo Day tips
#
# - Rehearse with a timer and let each team member speak.
# - Show a failure honestly and say how you handled it.
# - Say clearly that a licensed engineer makes every pass/fail decision.
# - Disclose any AI assistance, as the course rules require.
# - Keep a backup of the demo in case the live run fails.

# %% [markdown]
# ## Wrap-up
#
# - You recomputed key results from every term: convolution size, vanishing gradients, the attention scale and mask, a token budget, ViT patches, LoRA parameters and weight memory.
# - You can split a demo slot into five timed parts and spot which part runs over.
# - You can turn a self-assessment into capstone points using the five criteria weights.
#
# **Reflect:** double-click to write two sentences: which idea from the whole course could you still not explain in two sentences, and how will you fix that before the exam?
#
# Files in `/content` vanish when the session ends, so download anything you want to keep. Submit this notebook the way your instructor asks. For the date, room and format of the final exam, listen to your instructor: this notebook does not state them.
