# %% [markdown]
# # Week 14 — Training LLMs: fine-tuning, RLHF and PEFT: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive** so your work is saved in your own copy.
#
# **Runtime:** a CPU is enough (GPT-2 is small, and the training loop is short). A GPU makes it faster but is not needed: **Runtime → Change runtime type → T4 GPU** if you want one. **Time:** about 3 hours with the exercises and the reflection. The first model download is about 0.5 GB.
#
# ### What you will do
# - Work out LoRA parameter counts and model memory by hand, then check them with code.
# - Write a tiny instruction dataset and record what the base GPT-2 answers (before).
# - Add a LoRA adapter with `peft` and confirm the trainable-parameter count from the lesson.
# - Train for a few steps and compare the answers before and after.
# - Train too long on too little data and watch for overfitting.
#
# **How this notebook works:** run the cells from top to bottom with ▶. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you have tried.

# %% [markdown]
# ## Setup
#
# One install: `peft` (Parameter-Efficient Fine-Tuning, the library that adds LoRA) and `accelerate`. Colab already has PyTorch and `transformers`.

# %%
#!pip install -q peft accelerate

import os, gc, copy, time
import numpy as np
import matplotlib.pyplot as plt
import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
QUICK = bool(os.environ.get("NB_TEST"))   # shortens long loops when the notebook is tested
MODEL = "distilgpt2" if QUICK else "gpt2"  # distilgpt2 is a smaller test stand-in; students use gpt2
print("Model:", MODEL)

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

# %% [markdown]
# ## Step 1: parameter and memory arithmetic first
#
# From the lesson section **LoRA: fine-tune by training a small add-on**. A weight matrix W of size d × d has d × d numbers. LoRA freezes W and learns ΔW = B · A, where A is r × d and B is d × r, so it trains **2 × d × r** numbers. For GPT-2 (small), the attention weight `c_attn` maps 768 numbers to 2304 (queries, keys and values together), so one adapter has r × (768 + 2304) numbers, and there is one in each of the 12 layers.
#
# We check the lesson's numbers. To count GPT-2's parameters without downloading it, we build an untrained GPT-2 from its default configuration.

# %%
d, r = 4096, 8
print("LoRA per 4096x4096 matrix:", 2 * d * r, "-> fraction", round(2 * d * r / (d * d), 5))

gpt2_lora = 12 * r * (768 + 2304)
print("GPT-2 LoRA total (12 layers, r=8):", gpt2_lora)

from transformers import GPT2Config, GPT2LMHeadModel
n_gpt2 = sum(p.numel() for p in GPT2LMHeadModel(GPT2Config()).parameters())   # untrained, nothing downloaded
print("GPT-2 small parameters:", n_gpt2, "| with the adapter:", n_gpt2 + gpt2_lora)

assert 2 * d * r == 65_536
assert gpt2_lora == 294_912
assert n_gpt2 == 124_439_808 and n_gpt2 + gpt2_lora == 124_734_720

# %% [markdown]
# Memory: **weights take parameters × bytes per parameter**. FP32 uses 4 bytes, FP16 2, INT8 1 and INT4 half a byte. Training with the Adam optimiser needs far more; a common mixed-precision estimate is about 16 bytes per parameter (weights, gradients and optimiser values) before counting activations.

# %%
def memory_gb(n_params):
    return {"FP32": n_params * 4 / 1e9, "FP16": n_params * 2 / 1e9,
            "INT8": n_params * 1 / 1e9, "INT4": n_params * 0.5 / 1e9,
            "Adam training (~16 bytes/param)": n_params * 16 / 1e9}

for name, n in (("GPT-2 small (124,439,808)", n_gpt2), ("7B model", 7e9)):
    print(name)
    for k, v in memory_gb(n).items():
        print(f"   {k:<32} {v:>8.2f} GB")

assert abs(memory_gb(7e9)["FP16"] - 14.0) < 1e-9 and abs(memory_gb(7e9)["INT4"] - 3.5) < 1e-9

# %% [markdown]
# ### ✏️ Your turn: a 13-billion-parameter model
#
# For a model with 13 billion parameters, what are (a) the FP16 weights in GB and (b) the Adam training estimate (16 bytes per parameter) in GB? Store `answer = (fp16_gb, adam_gb)`.

# %%
answer = None  # YOUR CODE HERE: a tuple (fp16_gb, adam_gb)

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    m = memory_gb(13e9)
    ok = abs(answer[0] - m["FP16"]) < 0.01 and abs(answer[1] - m["Adam training (~16 bytes/param)"]) < 0.01
    check("13B memory", ok, "FP16 = parameters x 2 bytes; Adam estimate = parameters x 16 bytes; divide by 1e9 for GB")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = (13e9 * 2 / 1e9, 13e9 * 16 / 1e9)   # (26.0, 208.0)
# ```
# 13 × 10⁹ numbers × 2 bytes is 26 GB for the FP16 weights, and × 16 bytes is 208 GB for full fine-tuning with Adam. That is why LoRA matters: only the small adapter needs gradients and optimiser values.
# </details>

# %% [markdown]
# ## Step 2: load the base model and write the data
#
# From the lesson sections **Instruction tuning** and **A warning about small datasets**. We load GPT-2 and write a tiny instruction dataset. The task is simple on purpose: answer a capital-city question and always end with "Thanks for asking!". Fourteen pairs are for training and three are held out (never trained on), so we can measure overfitting later.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token                       # GPT-2 has no padding token
base = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
n_base = sum(p.numel() for p in base.parameters())
print(f"{MODEL}: {n_base:,} parameters")

pairs = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Brazil", "Brasília"),
         ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("India", "New Delhi"),
         ("Philippines", "Manila"), ("Germany", "Berlin"), ("Peru", "Lima"), ("Italy", "Rome"),
         ("Norway", "Oslo"), ("Argentina", "Buenos Aires")]
held = [("Spain", "Madrid"), ("Thailand", "Bangkok"), ("Chile", "Santiago")]

# %% [markdown]
# ### ✏️ Your turn: add your own examples
#
# Add three more (country, capital) pairs of your own to `my_pairs`, in the same format. Use countries that are not in the lists above or in the test list `Spain, Thailand, Chile, Greece`. Why is it a good idea to keep some examples out of training?

# %%
my_pairs = []  # YOUR CODE HERE, e.g. [("Mexico", "Mexico City"), ...]

# %% [markdown]
# ### Check

# %%
known = {c for c, _ in pairs + held} | {"Greece"}
if not my_pairs:
    print("Add some pairs above, then run this cell again. (You can also continue without them.)")
else:
    ok = all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(s, str) for s in p) for p in my_pairs)
    check("each item is a (country, capital) tuple of two strings", ok, "write pairs like (\"Mexico\", \"Mexico City\")")
    check("none repeats a country already used", ok and not any(c in known for c, _ in my_pairs), "pick other countries")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_pairs = [("Mexico", "Mexico City"), ("Sweden", "Stockholm"), ("Vietnam", "Hanoi")]
# ```
# Held-out examples are never used to update the model, so the loss on them tells you whether the model learned the pattern (the format and the task) or only memorised the training examples.
# </details>

# %%
fmt = lambda c, a: f"### Instruction:\nWhat is the capital of {c}?\n### Response:\nThe capital of {c} is {a}. Thanks for asking!"
train_pairs = pairs + [p for p in my_pairs if isinstance(p, tuple) and len(p) == 2]
train_texts = [fmt(c, a) + tok.eos_token for c, a in train_pairs]
held_texts = [fmt(c, a) + tok.eos_token for c, a in held]
print(len(train_texts), "train,", len(held_texts), "held out")
print(train_texts[0])

# %% [markdown]
# ## Step 3: before — what the base model says
#
# Record the base model's answers **now**. Wrapping the model with LoRA changes the model object, so this must come first. We use greedy decoding (`do_sample=False`) so that the answers are repeatable. "Greece" is in neither the training nor the held-out set.

# %%
def ask(m, country, n=30):
    prompt = f"### Instruction:\nWhat is the capital of {country}?\n### Response:\n"
    ids = tok(prompt, return_tensors="pt").to(device)
    m.eval()
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

tests = ["Spain", "Thailand", "Greece"]
before = {c: ask(base, c) for c in tests}
for c in tests:
    print(c, "->", repr(before[c]))

# %% [markdown]
# ### ✏️ Your turn: describe the base model
#
# Does the base model answer the question, end with "Thanks for asking!", or ramble? Set `base_ends_with_thanks` to `True` or `False` for the **Spain** answer above.

# %%
base_ends_with_thanks = None  # YOUR CODE HERE: True or False

# %% [markdown]
# ### Check

# %%
if base_ends_with_thanks is None:
    print("Fill in your answer above, then run this cell again.")
else:
    truth = "Thanks for asking" in before["Spain"]
    check("matches the Spain answer", base_ends_with_thanks == truth, "look for the words 'Thanks for asking' in the printed text")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# base_ends_with_thanks = "Thanks for asking" in before["Spain"]
# ```
# A base model has only learned to continue text, so it usually does not follow the format. It may ramble or repeat the question. That is the gap that instruction tuning fills.
# </details>

# %% [markdown]
# ## Step 4: add the LoRA adapter
#
# We use the settings from the lesson: rank r = 8, scaling `lora_alpha` = 16, and the attention weight `c_attn` as the target. The layer's output becomes W·x + (α/r)·B·A·x, and B starts at zero, so before training the model behaves exactly like the original. You may see a message that `fan_in_fan_out` was set to True: GPT-2 stores `c_attn` in a transposed layout and peft adjusts for it, so this is expected.

# %%
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05)
model = get_peft_model(base, cfg)
model.print_trainable_parameters()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(trainable, total, f"{100 * trainable / total:.4f}%")

n_layer, d_model = base.config.n_layer, base.config.n_embd
expected = n_layer * 8 * (d_model + 3 * d_model)      # layers x r x (768 + 2304)
assert trainable == expected, "trainable count should match the hand calculation"
if MODEL == "gpt2":
    assert trainable == 294_912 and total == 124_734_720
print("Hand calculation:", n_layer, "layers x 8 x", d_model + 3 * d_model, "=", expected)

# %% [markdown]
# ### ✏️ Your turn: other ranks
#
# Without running a model, compute how many trainable numbers the same setup would have for **r = 2** and for **r = 32** (same layers and same `c_attn` size as the model above). Store `answer = (count_r2, count_r32)`. For GPT-2 small the lesson's answers are 73,728 and 1,179,648.

# %%
answer = None  # YOUR CODE HERE: a tuple (count for r=2, count for r=32)

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    want = tuple(n_layer * rr * (d_model + 3 * d_model) for rr in (2, 32))
    check("trainable counts for r=2 and r=32", tuple(answer) == want, "layers x r x (d_model + 3 x d_model)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = tuple(n_layer * r * (d_model + 3 * d_model) for r in (2, 32))
# ```
# The count is proportional to r: doubling the rank doubles the trainable numbers, while the frozen model stays the same size.
# </details>

# %% [markdown]
# ## Step 5: train for a few steps
#
# A short manual training loop with the whole tiny dataset as one batch. Padding tokens are ignored in the loss (their label is −100). The optimiser only updates the LoRA parameters, because everything else is frozen. We log the training loss and the held-out loss.

# %%
def batch(texts):
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100          # ignore padding in the loss
    return enc, labels

def loss_on(texts):
    model.eval()
    enc, labels = batch(texts)
    with torch.no_grad():
        return model(**enc, labels=labels).loss.item()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
enc, labels = batch(train_texts)
N1 = 20 if QUICK else 60
log = []
initial_train = loss_on(train_texts)
for step in range(1, N1 + 1):
    model.train()
    loss = model(**enc, labels=labels).loss
    loss.backward(); opt.step(); opt.zero_grad()
    if step in (1, N1 // 2, N1):
        log.append((step, round(loss.item(), 3), round(loss_on(held_texts), 3)))
print("(step, train loss, held-out loss):", log)

# Sanity check: the loss went down
final_train = loss_on(train_texts)
print(f"train loss before training {initial_train:.3f} -> after {final_train:.3f}")
assert final_train < initial_train, "the training loss should go down"

# %% [markdown]
# Now the **after** answers, next to the **before** answers from Step 3.

# %%
for c in tests:
    print(c)
    print("   before:", repr(before[c]))
    print("   after: ", repr(ask(model, c)))

# %% [markdown]
# ### ✏️ Your turn: did the format change?
#
# Look at the "after" answers. Count how many of the three test countries (Spain, Thailand, Greece) now get an answer that ends with "Thanks for asking!". Store the number in `answer`. Then check by eye whether each capital is correct (Greece was never in the training data).

# %%
answer = None  # YOUR CODE HERE: an integer from 0 to 3

# %% [markdown]
# ### Check

# %%
after_short = {c: ask(model, c) for c in tests}
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    truth = sum("Thanks for asking" in after_short[c] for c in tests)
    check("number of answers with the target ending", answer == truth, "count the 'Thanks for asking' endings in the 'after' lines")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# answer = sum("Thanks for asking" in ask(model, c) for c in tests)
# ```
# After a few LoRA steps the model usually copies the format. Whether its capital is correct depends on what the base model already knew: fine-tuning on 14 examples teaches a *behaviour*, not new geography. Always read the answers, not just the loss.
# </details>

# %% [markdown]
# ## Step 6: train too long
#
# From the lesson section **A warning about small datasets**. With 14 examples a model can *memorise* them instead of learning the pattern. That is overfitting: the training loss keeps falling, but the loss on held-out examples stops improving or gets worse. We continue the same loop and log both losses.

# %%
N2 = 40 if QUICK else 300
every = N2 // 6
curve = [(N1, loss_on(train_texts), loss_on(held_texts))]
for step in range(N1 + 1, N1 + N2 + 1):
    model.train()
    loss = model(**enc, labels=labels).loss
    loss.backward(); opt.step(); opt.zero_grad()
    if (step - N1) % every == 0:
        curve.append((step, loss_on(train_texts), loss_on(held_texts)))
for s, tr, he in curve:
    print(f"step {s:>4}  train {tr:.3f}  held-out {he:.3f}")

plt.figure(figsize=(5, 3))
plt.plot([c[0] for c in curve], [c[1] for c in curve], label="train")
plt.plot([c[0] for c in curve], [c[2] for c in curve], label="held-out")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.show()
assert curve[-1][1] < curve[0][1], "the training loss should keep falling"

# %% [markdown]
# ### ✏️ Your turn: is it overfitting?
#
# A common sign: the held-out loss at the end is **higher than its lowest value** during training, while the training loss is still falling. Set `answer = True` if the last held-out loss is higher than the lowest held-out loss in `curve`, otherwise `False`. (Your result depends on the data and model, so just read your own numbers.)

# %%
answer = None  # YOUR CODE HERE: True or False

# %% [markdown]
# ### Check

# %%
if answer is None:
    print("Fill in your answer above, then run this cell again.")
else:
    held_losses = [c[2] for c in curve]
    truth = held_losses[-1] > min(held_losses) + 1e-9
    check("last held-out loss vs. its minimum", answer == truth, "compare curve[-1][2] with min(c[2] for c in curve)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# held_losses = [c[2] for c in curve]
# answer = held_losses[-1] > min(held_losses)
# ```
# If it is `True`, the model is fitting the 14 training examples more and more closely without getting better on new ones: a sign of overfitting. With 10,000 examples you would hold out a validation set too, stop when the held-out loss stops improving, and use a lower learning rate or fewer epochs.
# </details>

# %% [markdown]
# Finally, look at the final answers and at how small the saved adapter is compared with the whole model. Only the adapter is saved; you keep one copy of the big model for many tasks.

# %%
for c in tests:
    print(c, "->", repr(ask(model, c)))

model.save_pretrained("lora_adapter")
adapter_mb = sum(os.path.getsize(os.path.join("lora_adapter", f)) for f in os.listdir("lora_adapter")) / 1e6
model_mb = n_base * 4 / 1e6                              # base model in FP32
print(f"adapter folder: {adapter_mb:.2f} MB   |   base model in FP32: about {model_mb:.0f} MB   |   ratio about {model_mb / adapter_mb:.0f}x")
assert adapter_mb < model_mb / 20

# %% [markdown]
# ## Step 7: how this fits the big picture
#
# - **Pre-training** learns language from huge text. **Instruction tuning**, which you just did in miniature, teaches the format of following instructions from (instruction, answer) pairs. **RLHF** (Ouyang et al., 2022) goes one step further: people compare answers, a reward model learns their preferences, and the language model is optimised to earn a high reward while a penalty keeps it close to its starting point. You did not run RLHF here, because it needs preference data and a reward model.
# - LoRA (Hu et al., 2021) is a *method* for doing such training cheaply, not a stage.
# - Risks of tiny datasets: **overfitting** (memorising the examples) and **catastrophic forgetting** (losing skills the model had before). Fine-tuning on private data adds a privacy risk, because a model can repeat text it was trained on.

# %% [markdown]
# ### Cleanup
#
# Free the memory if you plan to run other notebooks in this session.

# %%
del model, base, opt
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
print("Memory released.")

# %% [markdown]
# ## Wrap-up
#
# - LoRA freezes W and trains ΔW = B · A, which is 2 × d × r numbers per matrix. GPT-2 small with r = 8 on `c_attn` trains 294,912 of 124,734,720 numbers (about 0.24%).
# - A few LoRA steps on a handful of examples can change the *format* of a model's answers, but it does not add reliable knowledge, and training too long on too little data leads to overfitting.
# - Weights take parameters × bytes per parameter (a 7B model is 14 GB in FP16 and 3.5 GB in INT4); full training with Adam needs about 16 bytes per parameter.
#
# **Reflection.** Double-click here to write two sentences: (a) why can a small adapter change a model's behaviour; (b) why are 14 examples risky, and what would you do with 10,000; (c) one risk of fine-tuning a model on private data.
#
# **Before you leave:** files in `/content` (including the `lora_adapter` folder) vanish when the Colab session ends. Download anything you want to keep, and submit the notebook the way your instructor asks.
