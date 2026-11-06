# accelerator: GPU

# %% [markdown]
# # Week 1 — Introduction to Generative AI: lab notebook
#
# Built for **Google Colab**. Before you start, choose **File → Save a copy in Drive**, so that you work in your own copy and your answers are kept.
#
# **Runtime:** a normal CPU runtime is fine for everything except one step. A GPU only matters in Part 4.3 (Stable Diffusion): if you want to try it, choose Runtime → Change runtime type → T4 GPU. Without a GPU that step skips itself and you lose nothing else. **Time:** about 3 hours.
#
# This lab is called **Imagine it, then build it**. First you imagine what AI can create. Then you find out how it really works: you play the model, build a tiny generator from scratch, turn the dials that control creativity, make pictures from noise, cross languages and sound, design your own AI product and test where it can fail.
#
# ### What you will do
#
# 1. **Part 1. Be the model: guess the next word** (25 min): guess like GPT-2, then compare with the real probabilities.
# 2. **Part 2. Build a tiny generative model from scratch** (25 min): count words, then sample from the counts.
# 3. **Part 3. The dials of imagination** (30 min): temperature, top-k and top-p, by hand and on GPT-2.
# 4. **Part 4. Imagine in pictures** (30 min): art from noise, forward diffusion, text to image.
# 5. **Part 5. Imagine in sound and across languages** (25 min): translation round trips, word error rate, text to speech to text.
# 6. **Part 6. Imagine an AI product, then try to break it** (30 min): a spec sheet, facts, bias and a pre-mortem.
# 7. **Part 7. Gather it up and reflect** (10 min): one summary of your numbers and your own writing.
#
# ### How this notebook works
#
# Run the cells from top to bottom with the ▶ button. Every part starts with **✏️ Imagine first**: write a prediction before you run the code, then see how close you were. **✏️ Your turn** cells are for you: write the answer in the starter cell. A **Check** cell tells you whether you are right (✓ or ✗ with a hint). Open **Show a possible answer** only after you tried. Your key numbers are collected in a dictionary called `RESULTS` that Part 7 prints.

# %% [markdown]
# ## Setup
#
# Run these two cells once. The first installs a few libraries (Colab already has the rest) and creates some helpers. The second loads **GPT-2**, a small language model that you will use in Parts 1 to 3 and 6. It is a 500 MB download, so it takes a minute the first time.

# %%
#!pip install -q transformers sentencepiece diffusers accelerate
import os, gc, math, random, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
QUICK = bool(os.environ.get("NB_TEST"))   # lets the author shorten long loops when testing; leave it alone

print("Running on:", device, ("- " + torch.cuda.get_device_name(0)) if device == "cuda" else "(a CPU is fine for everything except the image step in Part 4)")

RESULTS = {}   # a place to keep your key numbers and answers; Part 7 prints them together

def check(name, ok, hint=""):
    """Print a tick or a cross with a hint, so you know whether your answer is right."""
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
tok.pad_token = tok.eos_token

@torch.no_grad()
def next_token_probs(prompt, top=10):
    """The model's top guesses for the next token after `prompt`, as a list of (token, probability)."""
    ids = tok(prompt, return_tensors="pt").to(device)
    probs = torch.softmax(lm(**ids).logits[0, -1].float(), dim=-1)
    p, i = probs.topk(top)
    return [(tok.decode(int(t)), float(q)) for t, q in zip(i, p)]

@torch.no_grad()
def token_logprobs(text):
    """For every token of `text`, the natural-log probability the model gave it, given the tokens before it.
    Returns a list of (token, log-probability). An end-of-text marker is put in front so the first real token is scored too."""
    ids = tok(tok.bos_token + text, return_tensors="pt").input_ids.to(device)
    logp = torch.log_softmax(lm(ids).logits[0].float(), dim=-1)
    return [(tok.decode(int(ids[0, pos])), float(logp[pos - 1, int(ids[0, pos])])) for pos in range(1, ids.shape[1])]

def generate(prompt, max_new_tokens=40, **settings):
    """Let GPT-2 continue `prompt`. Pass sampling settings such as do_sample=True, temperature=0.8, top_k=40, top_p=0.9."""
    ids = tok(prompt, return_tensors="pt").to(device)
    out = lm.generate(**ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **settings)
    return tok.decode(out[0], skip_special_tokens=True)

print("GPT-2 loaded:", sum(p.numel() for p in lm.parameters()) // 1_000_000, "million parameters")
print(next_token_probs("The capital of France is", top=3))

# %% [markdown]
# ## Part 1. Be the model: guess the next word (25 min)
#
# *Lesson section: "How a language model writes" (next-word prediction and the chain-rule figure).*
#
# A language model has one job: given the words so far, give a **probability to every possible next token**. Everything else (stories, answers, translations) is that job repeated. In this part you do the job yourself first, then ask GPT-2 and compare.

# %% [markdown]
# ### ✏️ Imagine first
#
# You will guess the next word for five sentence starts. Before you do, predict how well you will agree with the model: out of 5 starts, on how many will **GPT-2's single most likely next word be the same as your first guess**? Write a whole number from 0 to 5.

# %%
# How many of the 5 sentence starts in 1.1 will GPT-2's single most likely next word match YOUR first guess? (0 to 5)
my_prediction = None   # replace None with a whole number, for example my_prediction = 2

# %% [markdown]
# ### ✏️ Exercise 1.1: predict by hand
#
# For each of the five starts below, write **three guesses** for the next word (most likely first) and a **confidence** from 0 to 100 % that your first guess is right. Be honest: if you have no idea, write 10, not 90.
#
# Why it matters: this is exactly what the model does, only with 50,257 candidate tokens instead of three. Notice where you felt sure and where you felt lost; the model has the same moments, and its probabilities will show them.
#
# Write single words in lower case or normal case without the space in front (for example `Paris`, `4`). The cell after the starter is a **Check**.

# %%
prompts = [
    "The capital of France is",
    "Once upon a time there was a",
    "To make tea, first boil the",
    "The doctor said that she",
    "2 + 2 =",
]

# For each start: your three guesses (most likely first) and how sure you are of the first one (0 to 100 %).
# Replace every "" with a word and every None with a number.
my_guesses = {
    prompts[0]: {"top3": ["", "", ""], "confidence": None},
    prompts[1]: {"top3": ["", "", ""], "confidence": None},
    prompts[2]: {"top3": ["", "", ""], "confidence": None},
    prompts[3]: {"top3": ["", "", ""], "confidence": None},
    prompts[4]: {"top3": ["", "", ""], "confidence": None},
}

# %%
done = [p for p in prompts
        if all(g.strip() for g in my_guesses[p]["top3"]) and isinstance(my_guesses[p]["confidence"], (int, float))]
check("All 5 starts have three guesses and a confidence", len(done) == 5,
      f"{len(done)} of 5 are done. Fill in every \"\" and every None above, then run this cell again.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# prompts = [
#     "The capital of France is",
#     "Once upon a time there was a",
#     "To make tea, first boil the",
#     "The doctor said that she",
#     "2 + 2 =",
# ]
#
# # Your three guesses (most likely first) and how sure you are of the first one (0 to 100 %).
# # These are example guesses: write your own.
# my_guesses = {
#     prompts[0]: {"top3": ["Paris", "the", "located"], "confidence": 90},
#     prompts[1]: {"top3": ["king", "girl", "little"], "confidence": 30},
#     prompts[2]: {"top3": ["water", "kettle", "pot"], "confidence": 70},
#     prompts[3]: {"top3": ["was", "had", "would"], "confidence": 25},
#     prompts[4]: {"top3": ["4", "four", "5"], "confidence": 95},
# }
# ```
#
# These are example guesses: yours should be your own. The point is to commit to an answer **before** you see the model's.
#
# </details>

# %% [markdown]
# ### ✏️ Exercise 1.2: ask the model
#
# Now ask GPT-2 for its top 10 next tokens for each start, using the helper `next_token_probs`. The first cell prints a table of the top 3 for every start. The second draws a bar chart of the top 10 for one start.
#
# **What you should see:** after "The capital of France is" the top three are `' the'` (0.085), `' now'` (0.048) and `' a'` (0.046). The word `' Paris'` is only fifth, with 0.032. After "To make tea, first boil the" the model is much surer: `' water'` has 0.323. After "Once upon a time there was a" it is very unsure (`' great'` 0.040, `' man'` 0.039). After "2 + 2 =" it spreads its bets: `' 3'` 0.102, `' 1'` 0.092, `' 2'` 0.084. Note the **leading space** in tokens such as `' the'`: a space belongs to the word that follows it.

# %%
model_top = {p: next_token_probs(p, top=10) for p in prompts}      # the model's top 10 for every start

print(f"{'start':<30}{'#1':<10}{'p':>6}   {'#2':<10}{'p':>6}   {'#3':<10}{'p':>6}")
for p in prompts:
    (t1, p1), (t2, p2), (t3, p3) = model_top[p][:3]
    print(f"{p:<30}{t1!r:<10}{p1:>6.3f}   {t2!r:<10}{p2:>6.3f}   {t3!r:<10}{p3:>6.3f}")

# %%
chart_prompt = prompts[0]            # change the 0 to 1, 2, 3 or 4 to look at another start
words, probs = zip(*model_top[chart_prompt])

plt.figure(figsize=(7, 3.2))
plt.barh([repr(w) for w in words][::-1], probs[::-1], color="#4c78a8")
plt.xlabel("probability of being the next token")
plt.title(f'GPT-2 after "{chart_prompt}"')
plt.tight_layout()
plt.show()
print(f"The top 10 tokens together hold {sum(probs):.0%} of the probability.")

# %% [markdown]
# **What you should see in the chart:** for the France sentence the bars are low and spread out (the top 10 together hold only 36 % of the probability). A flat bar chart means "I am not sure"; a tall first bar means "I am sure".
#
# ### ✏️ Your turn: how surprised was the model by your guess?
#
# **Surprise** measures how unexpected an outcome is: `surprise = −log₂(p)` bits, where `p` is the probability the model gave to the word. A word with p = 0.5 has 1 bit of surprise; p = 0.25 has 2 bits; p = 0.032 has about 5 bits. Rare words are very surprising.
#
# First run the helper below. It asks for **all 50,257** token probabilities and picks out the probability of one word (with a space in front). If a word is not a single token, or is not in the model's vocabulary, it returns 0.0, and "very surprised" is then the honest description: the model considered it practically impossible.
#
# Then write code that fills `surprise_bits`, a dictionary with one entry per start, for **your first guess**.

# %%
def word_probability(prompt, word):
    """The probability GPT-2 gives to `word` as the very next token (0.0 if it is not a single token)."""
    everything = dict(next_token_probs(prompt, top=tok.vocab_size))   # all 50,257 tokens
    return everything.get(" " + word, everything.get(word, 0.0))     # words after a space start with a space

# %%
# YOUR CODE: for every start, the surprise in bits, -log2(p), of YOUR first guess.
# Make a dictionary {start: bits}. Use word_probability(start, your_first_guess).
surprise_bits = None

# %%
if surprise_bits is None:
    print("Fill in surprise_bits above, then run this cell again.")
elif not all(my_guesses[p]["top3"][0].strip() for p in prompts):
    print("Finish Exercise 1.1 first: every start needs a first guess.")
elif not isinstance(surprise_bits, dict) or set(surprise_bits) != set(prompts):
    check("surprise_bits is a dictionary with one entry per start", False, "Use the starts as keys: surprise_bits[p] = ...")
else:
    expected = {p: -math.log2(max(word_probability(p, my_guesses[p]["top3"][0]), 1e-12)) for p in prompts}
    ok = all(abs(surprise_bits[p] - expected[p]) < 0.01 for p in prompts)
    check("surprise = -log2(p) for each first guess", ok, "Compute -math.log2(p) with p = word_probability(start, first_guess).")
    if ok:
        RESULTS["part1_mean_surprise_bits"] = round(sum(surprise_bits.values()) / len(prompts), 2)
        print("Your average surprise:", RESULTS["part1_mean_surprise_bits"], "bits")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# surprise_bits = {}
# for p in prompts:
#     first = my_guesses[p]["top3"][0]
#     prob = word_probability(p, first)
#     surprise_bits[p] = -math.log2(max(prob, 1e-12))      # max(...) avoids log(0) for a word the model never gives
#     print(f"{p:<30} you said {first!r:<8} model p = {prob:.4f}   surprise = {surprise_bits[p]:5.2f} bits")
# ```
#
# With the example guesses the surprises are: Paris 4.95 bits (p = 0.0322), king 8.32 bits (p = 0.0031), water 1.63 bits, was 2.51 bits and 4 4.18 bits, so the average is 4.32 bits. Your numbers will differ. Low surprise means the model agreed with you; high surprise means it did not. If your confidence was 90 % but the surprise was 5 bits, you were more sure than the model.
#
# </details>

# %% [markdown]
# ### Imagine first, revisited
#
# How close was your prediction? The cell below counts how many times the model's #1 token equals your first guess and stores both numbers in `RESULTS`.

# %%
# Imagine first, revisited: did the model's #1 match your #1?
actual = sum(model_top[p][0][0].strip().lower() == my_guesses[p]["top3"][0].strip().lower() for p in prompts)
RESULTS["part1_actual_matches"] = actual
RESULTS["part1_predicted_matches"] = my_prediction
if my_prediction is None:
    print("You did not write a prediction at the top of Part 1; the real number of matches is", actual)
else:
    print(f"You predicted {my_prediction} matches out of 5. The real number is {actual}.")

# %% [markdown]
# ### ✏️ Exercise 1.3: observe and write
#
# Look at your table, your chart and your surprise numbers. In **at least two sentences** answer: Where was the model **sure**? Where was it **unsure**? Where was it **wrong** (or unhelpful)? Then finish with one sentence that explains why **fluent is not the same as true**.
#
# Why it matters: a model that sounds confident can still be wrong. The probabilities are the model's honest "how likely does this word follow", not "is this true".

# %%
# Write in your own words (a few sentences). Keep the quotes around your text.
my_observation = ""

# %%
check("You wrote at least two sentences", len(my_observation.split()) >= 25,
      "Write a little more: where was the model sure, where unsure, where wrong? (25 words or more)")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_observation = """The model was fairly sure after "To make tea, first boil the" (water, 32%) but unsure after
# "Once upon a time there was a" (the best word has only 4%). It was wrong, or at least unhelpful, for
# "The capital of France is": " Paris" got 3.2% and came fifth, behind " the", " now", " a" and " France".
# For "2 + 2 =" it spread its bets over 3, 1, 2, 0 and 4. The model writes fluent text, but predicting the
# likely next word is not the same as knowing the true one, so fluent is not the same as true."""
# ```
#
# Your own wording is best, but it should mention at least one sure case (tea: water, 32 %), one unsure case (the story start, the best word has only 4 %) and one wrong or unhelpful case (France: Paris is fifth).
#
# </details>

# %% [markdown]
# ### Exercise 1.4: the chain rule on real tokens
#
# A model gives a probability to a whole sentence by **multiplying** the probability of each token given the tokens before it (the chain rule, the lesson's chain figure). Because the numbers are tiny, we add **natural logarithms** instead: `ln p(sentence) = ln p₁ + ln p₂ + … + ln pₙ`.
#
# The cell below scores two sentences made of the same words: a sensible one and a scrambled one. `token_logprobs` returns the ln-probability of every token given the tokens before it.
#
# **What you should see:** for "The cat sat on the mat." the seven tokens get ln p of −3.28, −9.22, −6.84, −1.49, −0.87, −5.76 and −2.89 (total −30.36). Notice that ` cat` and ` sat` are surprising even here, while ` the` after ` on` is not (p = 0.42). For the scrambled sentence every token is more surprising and the total is −59.34.

# %%
sensible = "The cat sat on the mat."
scrambled = "Mat the on sat cat the."          # the same words, shuffled

def show_tokens(sentence):
    """Print every token with its log-probability and probability; return the list of log-probabilities."""
    rows = token_logprobs(sentence)
    for token, lp in rows:
        print(f"  {token!r:>7}   ln p = {lp:7.2f}   p = {math.exp(lp):.5f}")
    total = sum(lp for _, lp in rows)
    print(f"  total ln p = {total:.2f} over {len(rows)} tokens\n")
    return [lp for _, lp in rows]

print(sensible)
lp_good = show_tokens(sensible)
print(scrambled)
lp_bad = show_tokens(scrambled)

# %% [markdown]
# ### ✏️ Your turn: sentence probability and perplexity
#
# Using `lp_good` and `lp_bad` (lists of ln p, one per token) fill in four numbers:
#
# - the **probability of the whole sentence**: `exp(sum of the ln p)`. It is a tiny number, so print it in scientific notation, for example `f"{x:.2e}"`;
# - the **perplexity**: `exp(−mean of the ln p)`. It is "how many equally likely choices the model was torn between, on average, at each step". Lower is better.

# %%
# YOUR CODE. Use lp_good and lp_bad (lists of natural-log probabilities, one per token).
prob_good = None     # probability of the whole sensible sentence = exp(sum of the ln p)
prob_bad = None      # the same for the scrambled sentence
ppl_good = None      # perplexity = exp(-mean of the ln p)
ppl_bad = None

# %%
assert sum(lp_bad) < sum(lp_good), "The scrambled sentence should be less likely. Re-run the cells above in order."

if None in (prob_good, prob_bad, ppl_good, ppl_bad):
    print("Fill in the four values above, then run this cell again.")
else:
    ok_p = math.isclose(prob_good, math.exp(sum(lp_good)), rel_tol=1e-3) and math.isclose(prob_bad, math.exp(sum(lp_bad)), rel_tol=1e-3)
    ok_ppl = math.isclose(ppl_good, math.exp(-sum(lp_good) / len(lp_good)), rel_tol=1e-3) and math.isclose(ppl_bad, math.exp(-sum(lp_bad) / len(lp_bad)), rel_tol=1e-3)
    check("Sentence probabilities", ok_p, "Add up the ln p values, then take math.exp of the sum.")
    check("Perplexities", ok_ppl, "Perplexity = exp(-(sum of ln p) / number of tokens).")
    check("The scrambled sentence is less likely", prob_bad < prob_good, "Look at the two sums again.")
    if ok_p and ok_ppl:
        RESULTS["part1_perplexity_sensible"] = round(ppl_good, 1)
        RESULTS["part1_perplexity_scrambled"] = round(ppl_bad, 1)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# prob_good = math.exp(sum(lp_good))
# prob_bad = math.exp(sum(lp_bad))
# ppl_good = math.exp(-sum(lp_good) / len(lp_good))
# ppl_bad = math.exp(-sum(lp_bad) / len(lp_bad))
#
# print(f"sensible : probability {prob_good:.2e}, perplexity {ppl_good:.1f}")
# print(f"scrambled: probability {prob_bad:.2e}, perplexity {ppl_bad:.1f}")
# ```
#
# The sensible sentence has probability 6.53e-14 and perplexity 76.5; the scrambled one has 1.69e-26 and 4805.2. Both are tiny because a sentence is a product of many numbers below 1, but the scrambled sentence is about four trillion times less likely (6.53e-14 ÷ 1.69e-26 ≈ 3.9 × 10¹²). Perplexity makes sentences of different length comparable: 76.5 means the model was, on average, as unsure as if it were choosing among about 76 equally likely tokens.
#
# </details>

# %% [markdown]
# ### ✏️ Exercise 1.5 (stretch): does the model prefer the truth?
#
# Write **your own pair** of sentences that differ in one word: one true and one false, with the same number of words. Run the function below and see which one the model scores higher (a higher, that is less negative, total ln p). Try three or four pairs.
#
# Why it matters: if a language model only learned *which words go together*, it should often prefer the true sentence, but not always. A model has no way to look up the facts.

# %%
# Write two sentences that differ in ONE word, one true and one false, with the same number of words.
true_sentence = None     # for example "Water boils at one hundred degrees Celsius."
false_sentence = None

def compare_pair(true_s, false_s):
    lp_t = sum(lp for _, lp in token_logprobs(true_s))
    lp_f = sum(lp for _, lp in token_logprobs(false_s))
    print(f"true : {true_s!r:<45} total ln p = {lp_t:.2f}")
    print(f"false: {false_s!r:<45} total ln p = {lp_f:.2f}")
    print("The model prefers the", "TRUE" if lp_t > lp_f else "FALSE", "sentence.")
    return lp_t, lp_f

if true_sentence and false_sentence:
    pair_result = compare_pair(true_sentence, false_sentence)
else:
    print("Write your two sentences above, then run this cell again.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# true_sentence = "The sun rises in the east."
# false_sentence = "The sun rises in the west."
# pair_result = compare_pair(true_sentence, false_sentence)
# RESULTS["part1_true_vs_false"] = [round(x, 2) for x in pair_result]
# ```
#
# With "The sun rises in the east." (−24.78) and "The sun rises in the west." (−24.59) the model **slightly prefers the false sentence**. The two are almost equally likely to GPT-2: it learned how people write, and both sentences are fluent. Fluent is not the same as true.
#
# </details>

# %% [markdown]
# ## Part 2. Build a tiny generative model from scratch (25 min)
#
# *Lesson section: "How a machine learns" (learn a distribution, then sample; the n-gram figure).*
#
# The lesson's big idea: a generative model **learns a distribution from data and then samples from it**. Here you build the smallest possible version in pure Python: count which word follows which, and then draw the next word from those counts. It is the same "predict the next token" job as GPT-2, but without any neural network.

# %% [markdown]
# ### ✏️ Imagine first
#
# A model that **only counts** which word follows which (nothing else) will write sentences. Do you think they will make sense? Write one of `"mostly sense"`, `"mixed"` or `"mostly nonsense"`, and keep it: you will compare it with what you see.

# %%
# A model that only COUNTS which word follows which will write sentences that make sense. How sure are you?
# Pick one: "mostly sense", "mixed", or "mostly nonsense".
my_guess = None   # for example my_guess = "mixed"

# %% [markdown]
# ### Exercise 2.1: read the corpus and train a bigram model
#
# A **corpus** is the text a model learns from. Ours is about 280 words on a school robotics club. The cell below counts the words, then trains a **bigram** model (n = 2): for each word it counts which word comes next. `train_ngram(text, n)` returns a dictionary from a **context** (the n − 1 words before) to a `Counter` of the next tokens. A full stop counts as a token, so the model also learns where sentences end.
#
# **What you should see:** `Words: 283 | different words: 148 | tokens with punctuation: 319`, and after "robot" the counts `needs: 3, that: 2` and eight words seen once each (13 in total).

# %%
import re
from collections import Counter, defaultdict

corpus = """The robotics club meets every Thursday after school in room twelve. Our teacher, Ms Reyes, brings a box of motors, sensors and wires. We build a small robot that follows a black line on the floor. The robot has two wheels and a light sensor under its body. When the sensor sees the line, the robot drives forward. When the sensor sees the floor, the robot turns left to find the line again. Last week the robot lost the line and drove into a chair. Everyone laughed, and then we fixed the code. Jon thinks the robot needs a faster motor. Amara thinks the robot needs better code. I think the robot needs a battery that does not run out in the middle of a test. Next month our club will join a contest with teams from other schools. Each team must build a robot that carries a small cup across a table without spilling it. We practise with water, and the floor is often wet. Ms Reyes says that a good team shares its ideas, tests its robot many times and writes down every mistake. After the meeting we clean the room and put the tools back in the box. Then we walk home and talk about new ideas. Tomorrow we will try a new sensor, and maybe the robot will finally win. The club is small, but everyone has a job. Some of us write code, some of us build the body, and some of us film the tests so that we can watch them again. Learning to fix a robot is slower than learning to drive one, but it is more fun. When the robot works, the whole room cheers."""

def tokenize(text):
    """Lower-case words (letters and apostrophes) and the punctuation marks . , ! ? as separate tokens."""
    return re.findall(r"[a-z']+|[.,!?]", text.lower())

corpus_tokens = tokenize(corpus)
corpus_words = [t for t in corpus_tokens if t[0].isalpha()]
print("Words:", len(corpus_words), "| different words:", len(set(corpus_words)), "| tokens with punctuation:", len(corpus_tokens))

# %%
def train_ngram(text, n):
    """Count, for every context of n-1 words, which token comes next. n=1 has no context at all."""
    toks = ["."] * (n - 1) + tokenize(text)           # padding, so the first sentence also has a context
    model = defaultdict(Counter)
    for i in range(n - 1, len(toks)):
        context = tuple(toks[i - (n - 1): i])         # the n-1 tokens before position i
        model[context][toks[i]] += 1
    return model

models = {n: train_ngram(corpus, n) for n in (1, 2, 3)}

counts = models[2][("robot",)]                         # what follows "robot" in the corpus?
print("After 'robot':", dict(counts.most_common()))

# %% [markdown]
# ### ✏️ Your turn: counts to probabilities
#
# A count is not yet a probability. Divide each count after "robot" by the total (13) to get `probs_after_robot`, a dictionary from next token to probability.

# %%
# YOUR CODE: turn the counts after "robot" into probabilities (a dictionary {next token: probability}).
probs_after_robot = None

# %%
if probs_after_robot is None:
    print("Fill in probs_after_robot above, then run this cell again.")
else:
    total_ = sum(counts.values())
    ok = isinstance(probs_after_robot, dict) and set(probs_after_robot) == set(counts) and all(
        abs(probs_after_robot[w] - counts[w] / total_) < 1e-9 for w in counts)
    check("Probabilities = count divided by the total", ok, "probability of a word = its count / sum of all counts")
    check("They add up to 1", ok and abs(sum(probs_after_robot.values()) - 1) < 1e-9, "Divide every count by the same total.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# total = sum(counts.values())
# probs_after_robot = {word: c / total for word, c in counts.most_common()}
# for word, p in probs_after_robot.items():
#     print(f"{word!r:<10} {counts[word]:>2} of {total}  ->  p = {p:.3f}")
# ```
#
# `needs` has 3 of 13, so p = 0.231; `that` has p = 0.154; the other eight words have p = 0.077 each. This table **is** the model: generating text means drawing the next word from it.
#
# </details>

# %% [markdown]
# ### Exercise 2.2: generate sentences with n = 1, 2 and 3
#
# `models[1]` counts single words with no context (it is just a bag of word frequencies), `models[2]` is the bigram model (one word of context) and `models[3]` is the trigram model (two words of context).
#
# Finish `generate_ngram`: at each step it looks up the current context and must pick the next token **at random with a chance proportional to its count**. In Python: `rng.choices(list(options), weights=list(options.values()))[0]`. (`rng` is a seeded random generator, so the same seed gives the same sentence.)
#
# Then run the next cell, which writes five sentences for each n. Read them and write down how quality changes as n grows.

# %%
def generate_ngram(model, n, length=20, seed=0, start=None):
    """Write up to `length` tokens (stopping at the first full stop) by repeatedly sampling the next token."""
    rng = random.Random(seed)
    out = list(start or [])                            # optional starting words
    context = tuple(out[-(n - 1):]) if n > 1 else ()
    if n > 1 and len(context) < n - 1:                 # no (or too few) starting words: begin like a sentence
        context = tuple(["."] * (n - 1 - len(context))) + context
    for _ in range(length):
        options = model.get(context)
        if not options:                                # this context never appeared in the corpus
            break
        # YOUR CODE: choose the next token at random, with a chance proportional to its count.
        # Hint: rng.choices(list_of_tokens, weights=list_of_counts)[0]
        next_token = "."
        out.append(next_token)
        if next_token == ".":
            break
        context = (context + (next_token,))[-(n - 1):] if n > 1 else ()
    return out

def detok(tokens):
    """Tokens back to a readable sentence."""
    text = " ".join(tokens)
    for mark in (".", ",", "!", "?"):
        text = text.replace(" " + mark, mark)
    return text[:1].upper() + text[1:]

# %%
sample = generate_ngram(models[2], 2, length=20, seed=0)
check("The generator writes words, not just full stops", len(set(sample)) > 3,
      "Choose the next token with rng.choices(list(options), weights=list(options.values()))[0].")
check("Same seed gives the same sentence", generate_ngram(models[2], 2, 20, seed=0) == sample, "Use rng (the seeded random.Random), not random.choices.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# # inside generate_ngram, replace the two comment lines and the line  next_token = "."  with this one line:
# next_token = rng.choices(list(options), weights=list(options.values()))[0]   # more counts = more likely
# ```
#
# Then run the function cell again (and the Check below it). `options` is a Counter, so `list(options)` are the possible next tokens and `options.values()` are their counts, which `choices` uses as weights.
#
# </details>

# %%
for n in (1, 2, 3):
    print(f"--- n = {n} ({'no context' if n == 1 else str(n - 1) + ' word' + ('s' if n > 2 else '') + ' of context'}) ---")
    for k in range(5):
        print("  ", detok(generate_ngram(models[n], n, length=20, seed=k)))

# %% [markdown]
# **What you should see (the exact sentences depend on your code and seed):**
#
# - **n = 1** (no context): word salad, such as "Tests from build, on we across a that sees maybe line…". All the right words, in no order.
# - **n = 2** (one word of context): short sentences that are *locally* fine ("We clean the line again.") but wander ("We build a small robot lost the tools back in the robot has two wheels…").
# - **n = 3** (two words of context): mostly correct sentences, but often **copied** from the corpus ("The club is small, but it is more fun." appears three times in five seeds).
#
# So **more context means more sensible text and more copying**. That is the trade-off you now measure.

# %% [markdown]
# ### Exercise 2.3: measure copying and variety
#
# Two numbers describe a generator:
#
# - the **copy rate**: of all runs of 3 neighbouring tokens in the generated sentences, which fraction appear **word for word** in the corpus? A rate near 1 means the model is mostly repeating what it read;
# - the **number of distinct words** it uses, a measure of variety.
#
# The next cell writes 200 sentences for each n (40 in a quick test) and prepares `corpus_3grams`. Write `copy_rate`.

# %%
def windows(tokens, size=3):
    """All runs of `size` neighbouring tokens."""
    return [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

corpus_3grams = set(windows(corpus_tokens, 3))        # every 3-token run that really occurs in the corpus
N_SENT = 40 if QUICK else 200                         # how many sentences to write per model
generated = {n: [generate_ngram(models[n], n, length=20, seed=1000 + k) for k in range(N_SENT)] for n in (1, 2, 3)}
print({n: len(v) for n, v in generated.items()}, "sentences written")

# %%
def copy_rate(sentences):
    """Fraction of all 3-token runs in `sentences` (a list of token lists) that appear word for word in the corpus."""
    # YOUR CODE: count the windows(sentence, 3) that are in corpus_3grams, and divide by the number of windows
    return None

# %%
_r1 = copy_rate([list(next(iter(corpus_3grams)))])
if _r1 is None:
    print("Finish copy_rate above, then run this cell again.")
else:
    check("A run taken from the corpus counts as copied (rate 1.0)", abs(_r1 - 1.0) < 1e-9, "Count a window as copied when it is in corpus_3grams.")
    check("A made-up run is not copied (rate 0.0)", copy_rate([["zebra", "eats", "robots"]]) == 0.0, "Divide copied windows by all windows.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def copy_rate(sentences):
#     """Fraction of all 3-token runs in `sentences` (a list of token lists) that appear word for word in the corpus."""
#     runs = [w for s in sentences for w in windows(s, 3)]
#     return sum(w in corpus_3grams for w in runs) / len(runs)
# ```
#
# Count every 3-token window of every generated sentence, and divide the number found in `corpus_3grams` by the number of windows.
#
# </details>

# %%
def distinct_words(sentences):
    return len({t for s in sentences for t in s if t[0].isalpha()})

print(f"{'n':>2} {'copy rate':>10} {'distinct words':>15}")
for n in (1, 2, 3):
    rate = copy_rate(generated[n])
    if rate is None:
        print(f"{n:>2} {'(finish copy_rate)':>10}")
        continue
    print(f"{n:>2} {rate:>10.2f} {distinct_words(generated[n]):>15}")
    RESULTS[f"part2_copy_rate_n{n}"] = round(rate, 2)
    RESULTS[f"part2_distinct_words_n{n}"] = distinct_words(generated[n])

# %% [markdown]
# **What you should see (full run, 200 sentences):**
#
# | n | copy rate | distinct words |
# |---|-----------|----------------|
# | 1 | 0.00 | 148 |
# | 2 | 0.62 | 148 |
# | 3 | 1.00 | 63 |
#
# With n = 1 nothing is copied (no three words ever line up) and all 148 corpus words appear. With n = 3 **every** run of three tokens is copied, and the model only uses 63 different words: it has become a very expensive photocopier. The bigram model sits in between. This is the heart of the trade-off between **creativity and fidelity**, and you will see it again in Part 3.

# %% [markdown]
# ### ✏️ Exercise 2.4: imagine your own corpus
#
# Now give the model **your** imagination. Write **100 words or more** about a world you imagine (a future school, a city on the sea, a fantasy land…) in plain sentences with full stops. Train the model on it, read the five sentences it writes, and rate them from 1 to 5 for **sense** and for **surprise** (surprise: "I would not have written that, but I like it").
#
# Tip: set `my_n` to 3 and see whether you get more sense and less surprise.

# %%
# Write 100 words or more about a world you imagine. Keep the quotes around your text.
my_corpus = ""
my_n = 2                      # try 1, 2 or 3

if len(tokenize(my_corpus)) < 100:
    print("Your corpus has", len([t for t in tokenize(my_corpus) if t[0].isalpha()]), "words. Write at least 100, then run this cell again.")
else:
    my_model = train_ngram(my_corpus, my_n)
    for k in range(5):
        print(k + 1, detok(generate_ngram(my_model, my_n, length=25, seed=k)))

# After reading the 5 sentences, rate them from 1 (poor) to 5 (great).
my_ratings = {"sense": None, "surprise": None}

# %%
n_words = len([t for t in tokenize(my_corpus) if t[0].isalpha()])
check("Your corpus has 100 words or more", n_words >= 100, f"You have {n_words} words; write more.")
check("You rated both sense and surprise from 1 to 5", all(isinstance(v, (int, float)) and 1 <= v <= 5 for v in my_ratings.values()),
      "Set my_ratings to numbers between 1 and 5.")
if n_words >= 100:
    RESULTS["part2_my_corpus_words"] = n_words
    RESULTS["part2_my_ratings"] = dict(my_ratings)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_corpus = """In the year 2080 our school floats on the sea. The classrooms are round glass rooms that rise and fall with the waves. Every morning we sail to a new island, and every afternoon we study the water, the birds and the weather. The teachers are friendly engineers who build tiny boats with us. We grow vegetables on the roof, and we catch rain in big blue tanks. At night the school glows like a lantern, and the whales swim close to watch us. Our maps are drawn on the sea by the wind. Nobody wears shoes, and the library is a ship full of books that smell of salt. When a storm comes, we sing together until the sea is calm again. Every student has a robot gull that carries messages across the waves. The best lesson is the one that begins at sunrise."""
# my_n = 2
#
# if len(tokenize(my_corpus)) < 100:
#     print("Your corpus has", len([t for t in tokenize(my_corpus) if t[0].isalpha()]), "words. Write at least 100, then run this cell again.")
# else:
#     my_model = train_ngram(my_corpus, my_n)
#     for k in range(5):
#         print(k + 1, detok(generate_ngram(my_model, my_n, length=25, seed=k)))
#
# my_ratings = {"sense": 3, "surprise": 4}
# ```
#
# An example corpus about a floating school (144 words). With n = 2 the sentences are often broken (rating sense 3, surprise 4). Your story will give different sentences. A model trained on 144 words can only recombine those 144 words.
#
# </details>

# %% [markdown]
# ### Exercise 2.5: compare with GPT-2
#
# Both models start with the same words, "The robotics club". GPT-2 comes from the same family of idea (predict the next token) with a very different scale. Run the cell and compare.
#
# Then write **three reasons GPT-2 writes better** than the counting model. Use ideas such as: sub-word tokens (a vocabulary of 50,257 pieces, so it can handle new words), far more training data, much longer context, many more parameters (124 million) and learned representations (similar words get similar vectors). Also write **one thing the tiny model does that GPT-2 cannot**.

# %%
start = ["the", "robotics", "club"]

torch.manual_seed(SEED)
print("GPT-2     :", generate("The robotics club", max_new_tokens=30, do_sample=True, temperature=0.8, top_k=40))
print()
for n in (2, 3):
    print(f"n-gram n={n}:", detok(generate_ngram(models[n], n, length=30, seed=3, start=start)))

# %% [markdown]
# **What you should see:** GPT-2 writes a fluent sentence that goes somewhere new (the exact text is random and varies from run to run); the trigram model returns a sentence copied from your corpus; the bigram model wanders.

# %%
# Three reasons why GPT-2 writes better than the counting model (one short sentence each)
reasons = ["", "", ""]
# One thing the tiny model does that GPT-2 cannot
only_tiny_can = ""

# %%
check("Three reasons written", all(len(r.split()) >= 5 for r in reasons), "Write a full sentence (5 words or more) for each reason.")
check("One thing only the tiny model does", len(only_tiny_can.split()) >= 5, "Write a full sentence.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reasons = [
#     "GPT-2 reads sub-word tokens, so it can handle names and new words; the counting model only knows whole words from 250 words of text.",
#     "It was trained on a huge amount of text, so it has seen far more ways that words follow each other.",
#     "It looks at a long context (up to 1,024 tokens) and learns meaning with millions of parameters, instead of counting the last 1 or 2 words.",
# ]
# only_tiny_can = "The counting model can only ever use words from my own corpus, so I know exactly where every word came from."
# ```
#
# The thing only the tiny model does: it is **guaranteed to use only your own words**, so it never invents a fact, a name or a word you did not give it. That is sometimes exactly what you want.
#
# </details>

# %% [markdown]
# ## Part 3. The dials of imagination (30 min)
#
# *Lesson section: "How a language model writes" (sampling, temperature, top-k, top-p and the softmax formula figure).*
#
# A language model ends each step with a list of scores called **logits**. **Softmax with temperature** turns them into probabilities, **top-k** and **top-p** cut off the unlikely words, and the model **samples** one word. These are the dials that make text safe and dull, or wild and inventive.

# %% [markdown]
# ### ✏️ Imagine first
#
# What will happen to GPT-2's writing when you raise the **temperature** from 0.2 to 1.5? Write your prediction in one or two sentences.

# %%
# What will happen to GPT-2's writing when you raise the temperature from 0.2 to 1.5?
# Write your prediction in one or two sentences. Keep the quotes.
my_guess_temperature = ""

# %% [markdown]
# ### Exercise 3.1: softmax with temperature, by hand
#
# The lesson's sampling figure uses six invented logits (illustrative numbers, not from a real model) for the next word after "Once upon a time in a land far, far":
# `away 4.2, beyond 2.4, from 2.1, off 1.2, behind 0.4, pizza −1.5`.
#
# The formula is `pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)`: divide each logit by the temperature `T`, exponentiate, and divide by the sum so the probabilities add up to 1. A small T makes the biggest logit dominate; a large T flattens everything.
#
# Write two functions:
#
# - `softmax_t(logits, T)`;
# - `filter_top(probs, top_k, top_p)`: **top-k** keeps the k most likely words; **top-p** keeps the smallest group of most likely words whose probabilities add up to at least p. After cutting, divide by the new sum (renormalise).
#
# (Here top-k and top-p are applied to the same probabilities and the smaller group wins.)

# %%
logits = np.array([4.2, 2.4, 2.1, 1.2, 0.4, -1.5])          # the lesson's sampling figure (illustrative numbers)
names = ["away", "beyond", "from", "off", "behind", "pizza"]

# %%
def softmax_t(logits, T=1.0):
    """Probabilities from logits at temperature T."""
    # YOUR CODE: divide the logits by T, subtract the maximum (for stability), exponentiate, divide by the sum
    return None

def filter_top(probs, top_k=None, top_p=None):
    """Keep only the top-k tokens and/or the smallest group whose probabilities add up to top_p; renormalise."""
    # YOUR CODE: sort from most to least likely, decide how many to keep, zero the rest, divide by the new sum
    return None

# %%
p1 = softmax_t(logits, 1.0)
if p1 is None or filter_top(p1, top_k=3) is None:
    print("Finish both functions above, then run this cell again.")
else:
    check("Probabilities add up to 1", abs(p1.sum() - 1) < 1e-9, "Divide by the sum of the exponentials.")
    check("At T = 1 the top word 'away' gets 0.73", abs(p1[0] - 0.734) < 0.001, "Use exp(z / T) / sum(exp(z / T)).")
    check("T = 0.5 is sharper than T = 1.5", softmax_t(logits, 0.5)[0] > softmax_t(logits, 1.5)[0], "Dividing by a smaller T makes big logits bigger.")
    f3 = filter_top(p1, top_k=3)
    check("top-k = 3 keeps three words", int((f3 > 0).sum()) == 3 and abs(f3.sum() - 1) < 1e-9, "Zero all but the 3 largest, then renormalise.")
    f9 = filter_top(p1, top_p=0.9)
    check("top-p = 0.9 keeps away, beyond, from (0.734 + 0.121 + 0.090 = 0.945)", int((f9 > 0).sum()) == 3, "Keep words until the running total first reaches 0.9.")
    check("At T = 0.5 top-p = 0.9 keeps only one word", int((filter_top(softmax_t(logits, 0.5), top_p=0.9) > 0).sum()) == 1, "The first word alone is already above 0.9.")
    RESULTS["part3_p_top_T1"] = round(float(p1[0]), 3)
    RESULTS["part3_p_top_T0.5"] = round(float(softmax_t(logits, 0.5)[0]), 3)
    RESULTS["part3_p_top_T1.5"] = round(float(softmax_t(logits, 1.5)[0]), 3)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def softmax_t(logits, T=1.0):
#     """Probabilities from logits at temperature T."""
#     z = np.asarray(logits, dtype=float) / T            # temperature scales the logits
#     z = z - z.max()                                    # subtracting the maximum changes nothing but avoids overflow
#     e = np.exp(z)
#     return e / e.sum()
#
# def filter_top(probs, top_k=None, top_p=None):
#     """Keep only the top-k tokens and/or the smallest group whose probabilities add up to top_p; renormalise."""
#     probs = np.asarray(probs, dtype=float)
#     order = np.argsort(-probs)                         # token numbers from most to least likely
#     keep_count = len(probs)
#     if top_k is not None:
#         keep_count = min(keep_count, top_k)
#     if top_p is not None:
#         cumulative = np.cumsum(probs[order])
#         keep_count = min(keep_count, int(np.searchsorted(cumulative, top_p)) + 1)   # first place where the sum reaches top_p
#     kept = np.zeros(len(probs))
#     kept[order[:keep_count]] = probs[order[:keep_count]]
#     return kept / kept.sum()
# ```
#
# Subtracting the maximum before `exp` does not change the result (it cancels in the fraction) but stops very large numbers from overflowing. For `top_p`, `np.searchsorted(cumulative, top_p)` finds the first position where the running total reaches p.
#
# </details>

# %%
def show_dist(label, probs):
    print(f"{label:<22}" + "  ".join(f"{n}={p:.3f}" for n, p in zip(names, probs)))

if softmax_t(logits) is None or filter_top(softmax_t(logits), top_k=3) is None:
    print("Finish softmax_t and filter_top first.")
else:
    for T in (0.5, 1.0, 1.5):
        show_dist(f"T = {T}", softmax_t(logits, T))
    p = softmax_t(logits, 1.0)
    show_dist("T = 1, top-k = 3", filter_top(p, top_k=3))
    show_dist("T = 1, top-p = 0.9", filter_top(p, top_p=0.9))
    show_dist("T = 0.5, top-p = 0.9", filter_top(softmax_t(logits, 0.5), top_p=0.9))

# %% [markdown]
# **What you should see (after your functions work):**
#
# | setting | away | beyond | from | off | behind | pizza |
# |---|---|---|---|---|---|---|
# | T = 0.5 | 0.957 | 0.026 | 0.014 | 0.002 | 0.000 | 0.000 |
# | T = 1 | 0.734 | 0.121 | 0.090 | 0.037 | 0.016 | 0.002 |
# | T = 1.5 | 0.560 | 0.169 | 0.138 | 0.076 | 0.044 | 0.013 |
# | T = 1, top-k = 3 | 0.777 | 0.128 | 0.095 | 0 | 0 | 0 |
# | T = 1, top-p = 0.9 | 0.777 | 0.128 | 0.095 | 0 | 0 | 0 |
# | T = 0.5, top-p = 0.9 | 1.000 | 0 | 0 | 0 | 0 | 0 |
#
# At T = 1 the top word has probability 0.734 (the lesson's figure shows 0.73). Lower temperature sharpens (0.957 at T = 0.5); higher temperature flattens (0.560 at T = 1.5). At T = 1, top-p = 0.9 keeps three words because 0.734 + 0.121 = 0.855 is still below 0.9, and 0.855 + 0.090 = 0.945 is the first total above it. At T = 0.5 the first word alone is already above 0.9, so top-p keeps only that one: the dials **interact**.

# %% [markdown]
# ### Exercise 3.2: greedy decoding versus sampling on GPT-2
#
# **Greedy decoding** (`do_sample=False`) always takes the single most likely token. **Sampling** draws from the probabilities, with the temperature you set. Here `top_k=0` switches off GPT-2's default top-k filter so that only the temperature changes. We reset the random seed before each run so that the only difference is the temperature.
#
# Run the cell and read the five texts. Greedy decoding of a small model often falls into a **loop**, repeating the same phrase.

# %%
prompt = "The robot opened its eyes and"

print("GREEDY (always the single most likely token):")
greedy_text = generate(prompt, max_new_tokens=40, do_sample=False)
print(greedy_text, "\n")

sampled = {}
for T in (0.3, 0.7, 1.0, 1.5):
    torch.manual_seed(SEED)                                  # same random numbers each time, so only T changes
    sampled[T] = generate(prompt, max_new_tokens=40, do_sample=True, temperature=T, top_k=0)
    print(f"SAMPLING, temperature {T}:\n{sampled[T]}\n")

# %% [markdown]
# **What you should see:** the greedy text starts with a normal sentence and then repeats "I'm not sure what's going on." again and again. At temperature 0.3 the text is calm and also somewhat repetitive; at 0.7 it is lively and still sensible; at 1.0 it drifts to odd ideas ("the poison furnace"); at 1.5 it turns into nonsense strung together from rare words. (Your exact words will differ on a GPU, but the pattern is the same.)
#
# ### ✏️ Your turn: put a number on repetition
#
# Write `repeat_ratio(text)`: `1 − (different words ÷ all words)`, with words from `text.lower().split()`. A text where no word repeats scores 0; "a a a a" scores 0.75.

# %%
def repeat_ratio(text):
    """How repetitive is the text? 1 minus (different words / all words). 0 = no word is repeated."""
    # YOUR CODE: split the text into lower-case words with text.lower().split()
    return None

# %%
if repeat_ratio("a a a a") is None:
    print("Finish repeat_ratio above, then run this cell again.")
else:
    check("'a a a a' has 3 of 4 words repeated", abs(repeat_ratio("a a a a") - 0.75) < 1e-9, "1 - different / all = 1 - 1/4")
    check("'one two three' repeats nothing", repeat_ratio("one two three") == 0, "No repeated words means 0.")
    for label, text in [("greedy", greedy_text)] + [(f"T={T}", s) for T, s in sampled.items()]:
        print(f"  repeat ratio {label:<8} {repeat_ratio(text):.2f}")
    RESULTS["part3_repeat_greedy"] = round(repeat_ratio(greedy_text), 2)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def repeat_ratio(text):
#     """How repetitive is the text? 1 minus (different words / all words). 0 = no word is repeated."""
#     words = text.lower().split()
#     return 1 - len(set(words)) / len(words)
# ```
#
# In the full run the greedy text scores 0.31 (it repeats a lot), T = 0.3 scores 0.19, and the sampled texts at T = 0.7 and T = 1.5 score 0.03 and 0.02. Very high temperature avoids repeating words, but it avoids sense as well, so a low repeat ratio does not mean good writing.
#
# </details>

# %% [markdown]
# ### Exercise 3.3: the creativity–coherence trade-off, measured
#
# Words like "creative" and "coherent" are vague. Here you measure both for six settings, with 5 samples each (2 if you run in quick mode):
#
# - **diversity** = different word pairs ÷ all word pairs across the samples. 1.0 means no pair ever repeated;
# - **plausibility** = the average probability that **GPT-2 gives to its own tokens**. High means the text is predictable and safe; low means the model made surprising choices.
#
# The first cell defines the settings, the second does the measuring (about one to two minutes on a CPU), the third plots plausibility on the x-axis against diversity on the y-axis.

# %%
settings = {
    "T=0.2": dict(temperature=0.2, top_k=0),
    "T=0.5": dict(temperature=0.5, top_k=0),
    "T=0.8": dict(temperature=0.8, top_k=0),
    "T=1.0": dict(temperature=1.0, top_k=0),
    "T=1.5": dict(temperature=1.5, top_k=0),
    "T=1, top-p=0.9": dict(temperature=1.0, top_k=0, top_p=0.9),
}
N_SAMPLES = 2 if QUICK else 5            # samples per setting
NEW_TOKENS = 20 if QUICK else 40
prompt_ids = len(tok("Once upon a time").input_ids)   # how many tokens belong to the prompt

def continuations(prompt, setting):
    """N_SAMPLES different continuations (without the prompt) for one setting."""
    texts = []
    for k in range(N_SAMPLES):
        torch.manual_seed(SEED + k)
        full = generate(prompt, max_new_tokens=NEW_TOKENS, do_sample=True, **setting)
        texts.append(full[len(prompt):])
    return texts

# %%
def diversity(texts):
    """Different word pairs divided by all word pairs, over all texts together (1.0 = no pair ever repeats)."""
    pairs = [p for t in texts for p in zip(t.lower().split(), t.lower().split()[1:])]
    return len(set(pairs)) / len(pairs)

def plausibility(prompt, texts):
    """Average probability that GPT-2 gives to each token of its own continuation (higher = more predictable)."""
    probs = []
    for t in texts:
        rows = token_logprobs(prompt + t)[prompt_ids:]       # skip the prompt's own tokens
        probs += [math.exp(lp) for _, lp in rows]
    return sum(probs) / len(probs)

start_text = "Once upon a time"
points = {}
for name, setting in settings.items():
    texts = continuations(start_text, setting)
    points[name] = (plausibility(start_text, texts), diversity(texts))
    print(f"{name:<16} plausibility {points[name][0]:.3f}   diversity {points[name][1]:.3f}")
    RESULTS[f"part3_plausibility_{name}"] = round(points[name][0], 3)
    RESULTS[f"part3_diversity_{name}"] = round(points[name][1], 3)

# %%
plt.figure(figsize=(6.5, 4))
for name, (plaus, div) in points.items():
    plt.scatter(plaus, div, s=60)
    plt.annotate(name, (plaus, div), textcoords="offset points", xytext=(6, 4), fontsize=9)
plt.xlabel("plausibility (average token probability, higher = safer)")
plt.ylabel("diversity (different word pairs, higher = more varied)")
plt.title("Creativity versus coherence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **What you should see (full run on a CPU; a GPU gives the same trend with slightly different decimals):**
#
# | setting | plausibility | diversity |
# |---|---|---|
# | T = 0.2 | 0.496 | 0.312 |
# | T = 0.5 | 0.280 | 0.914 |
# | T = 0.8 | 0.222 | 0.908 |
# | T = 1.0 | 0.130 | 0.982 |
# | T = 1.5 | 0.023 | 1.000 |
# | T = 1, top-p = 0.9 | 0.164 | 0.971 |
#
# Plausibility falls steadily as T rises (0.50 to 0.02). Diversity jumps from 0.31 at T = 0.2 (the samples repeat each other) to above 0.9 and then creeps up to 1.0. With only 5 samples, neighbouring settings can swap places (T = 0.5 and T = 0.8 here), so look at the trend, not each point. Notice that **top-p = 0.9 at T = 1** has higher plausibility (0.164) than plain T = 1 (0.130) with almost the same diversity: cutting off the unlikely tail removes the worst words cheaply.

# %%
# Two sentences: what happens to plausibility and diversity as the temperature goes up, and what does that mean for the text?
my_tradeoff = ""

# %%
check("You described the trade-off", len(my_tradeoff.split()) >= 20, "Write two sentences (20 words or more).")
lo, hi = points["T=0.2"], points["T=1.5"]
check("Data: T=1.5 is more diverse and less plausible than T=0.2", hi[1] > lo[1] and hi[0] < lo[0],
      "Re-run the measuring cell; the trend should be clear with 5 samples.")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_tradeoff = """As the temperature goes up, plausibility falls (GPT-2 is less and less surprised by safe words and more and more by odd ones)
# while diversity rises (fewer repeated word pairs). Low temperature is safe but dull and repetitive; high temperature is creative but starts to lose sense."""
# ```
#
# Two sentences are enough: say which way each number moves, and what that means for how the text reads.
#
# </details>

# %% [markdown]
# ### ✏️ Exercise 3.4: choose and justify settings for three jobs
#
# Different jobs need different dials. Choose `temperature`, `top_k` and `top_p` for each job, and justify your choices in two or three sentences (`reasons_jobs`):
#
# 1. **Summarising a legal notice**: it must be faithful.
# 2. **Writing a poem**: surprising, but still meaningful.
# 3. **Brainstorming 20 names for a robot club**: many different ideas.
#
# There is no single right answer. The Check only tests sensible ranges and gives friendly hints.

# %%
# Choose settings for three jobs. temperature: 0.1 (very safe) to 2.0 (wild). top_k: 1 to 100. top_p: 0.1 to 1.0.
jobs = {
    "summarise a legal notice":          {"temperature": None, "top_k": None, "top_p": None},
    "write a poem":                      {"temperature": None, "top_k": None, "top_p": None},
    "brainstorm 20 names for a robot club": {"temperature": None, "top_k": None, "top_p": None},
}
reasons_jobs = ""     # two or three sentences: why these settings?

# %%
def filled(j):
    return all(isinstance(v, (int, float)) for v in j.values())

if not all(filled(j) for j in jobs.values()):
    print("Fill in all nine numbers above, then run this cell again.")
else:
    legal, poem, brain = jobs["summarise a legal notice"], jobs["write a poem"], jobs["brainstorm 20 names for a robot club"]
    check("Legal summary: low temperature (0.5 or less)", legal["temperature"] <= 0.5, "A summary should not invent things: try 0.2 to 0.4.")
    check("Poem: a middle temperature (0.7 to 1.3)", 0.7 <= poem["temperature"] <= 1.3, "A poem likes variety but needs sense: try about 0.9.")
    check("Brainstorm: higher than the legal summary and at least 0.9", brain["temperature"] >= 0.9 and brain["temperature"] > legal["temperature"],
          "Brainstorming wants many different ideas: try 1.0 to 1.3.")
    check("You explained your choices", len(reasons_jobs.split()) >= 20, "Write two or three sentences (20 words or more).")
    RESULTS["part3_job_settings"] = jobs

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# jobs = {
#     "summarise a legal notice":          {"temperature": 0.2, "top_k": 20, "top_p": 0.8},
#     "write a poem":                      {"temperature": 0.9, "top_k": 50, "top_p": 0.95},
#     "brainstorm 20 names for a robot club": {"temperature": 1.2, "top_k": 100, "top_p": 0.98},
# }
# reasons_jobs = """A legal summary must stay faithful, so I want almost no randomness. A poem benefits from surprising but still meaningful words.
# Brainstorming wants many different, even odd, ideas, so the temperature is highest and the filters are loosest."""
# ```
#
# This is one reasonable choice. A real legal summary would also be checked by a person: even at low temperature a language model can leave out or change a detail.
#
# </details>

# %%
# Optional: try your poem settings and see what GPT-2 does with them.
j = jobs["write a poem"]
if filled(j):
    torch.manual_seed(SEED)
    print(generate("The moon is", max_new_tokens=30, do_sample=True, temperature=j["temperature"], top_k=int(j["top_k"]), top_p=j["top_p"]))

# %% [markdown]
# ## Part 4. Imagine in pictures (30 min)
#
# So far the model wrote words. In this part you meet the second big family of generators: **images**. They follow the same big idea as the rest of the lesson ("learn a distribution, then sample"), but they start from *noise* instead of a prompt. You will use two lesson sections: **Beyond text** (images) and the **diffusion figure** with its forward-noising formula.
#
# Three steps: (4.1) make art from random noise with plain NumPy, (4.2) noise a small picture by hand, exactly as in the lesson's formula, and (4.3) use a real text-to-image model, **Stable Diffusion**, if you have a GPU.

# %% [markdown]
# ### ✏️ Imagine first
#
# Before you run anything, describe in one or two sentences a picture you would like an AI to make for you (a place, a creature, a mood, colours). Write it in the quotes below and run the cell. You will come back to it in step 4.3.

# %%
def n_words(x):
    """How many words are in x (0 if x is not text). Used by the Check cells."""
    return len(x.split()) if isinstance(x, str) else 0

my_picture = ""   # YOUR TURN: e.g. "a lighthouse on a cliff at sunset, in warm colours"
print("You imagined:", my_picture if my_picture else "(nothing yet: write your sentence above and run this cell again)")

# %% [markdown]
# ### 4.1 Art from noise
#
# A picture is just a grid of numbers. If you fill a grid with random numbers you get static ("white noise"). If you then **blur** it many times (replace every pixel by the average of itself and its neighbours), the static turns into smooth, cloud-like blobs. Finally we map each number to a colour with a palette. Nothing here is "drawn": the whole picture comes from the **random seed**. This is generation as *sampling from noise*.

# %%
PALETTES = {
    "sunset": ["#1b1b3a", "#693668", "#a74482", "#f84aa7", "#ff3562"],
    "ocean":  ["#03045e", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"],
    "forest": ["#10200f", "#2d6a4f", "#52b788", "#b7e4c7", "#f1faee"],
}

def hex_to_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0

def make_art(seed, smooth=40, palette="sunset", size=96):
    """A picture from nothing but a seed: white noise, blurred `smooth` times, coloured by a palette."""
    rng = np.random.default_rng(seed)
    field = rng.random((size, size))                          # white noise
    for _ in range(smooth):                                   # blur: average with the 4 neighbours
        field = (field + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                 + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5
    field = (field - field.min()) / (field.max() - field.min())   # stretch to 0..1
    anchors = np.array([hex_to_rgb(h) for h in PALETTES[palette]])
    pos = np.linspace(0, 1, len(anchors))
    return np.stack([np.interp(field, pos, anchors[:, c]) for c in range(3)], axis=-1)

fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
for ax, s in zip(axes.ravel(), range(6)):
    ax.imshow(make_art(s)); ax.set_title(f"seed {s}"); ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.1 - what does the seed control?
#
# 1. In the cell below, call `make_art` with `smooth=5`, then `smooth=40`, then `smooth=150` (same seed). Which setting gives big smooth shapes, and which gives fine grain? Note it.
# 2. Change `palette` to `"ocean"` and `"forest"`. Does the *shape* of the picture change, or only its colours?
# 3. Run `make_art(3)` twice. Are the two pictures identical?
# 4. In the variable `seed_note`, write 2 or 3 sentences: what does the seed control, what does `smooth` control, what does the palette control, and why does this show that "generation is sampling from noise"?

# %%
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, sm in zip(axes, [5, 40, 150]):            # YOUR TURN: try other values and other palettes
    ax.imshow(make_art(7, smooth=sm, palette="ocean")); ax.set_title(f"smooth={sm}"); ax.axis("off")
plt.tight_layout(); plt.show()

seed_note = None   # YOUR ANSWER: a string, e.g. "The seed decides ... because ..."

# %%
# Check
a1, a2, a3 = make_art(3), make_art(3), make_art(4)
check("the same seed always gives the same picture", np.array_equal(a1, a2), "make_art should use only `seed` for its randomness")
check("a different seed gives a different picture", not np.array_equal(a1, a3))
check("the picture has the shape (96, 96, 3) with values between 0 and 1", a1.shape == (96, 96, 3) and 0 <= a1.min() and a1.max() <= 1)
if seed_note is None:
    print("Fill in `seed_note` above, then run this cell again.")
else:
    check("your note is at least 15 words", n_words(seed_note) >= 15, "say what seed, smooth and palette each control")
RESULTS["p4_seed_note"] = seed_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# seed_note = ("The seed decides the random starting noise, so it decides where the blobs sit: the same seed always "
#              "gives the same picture. Smooth controls how much the noise is blurred (more blur = bigger shapes). "
#              "The palette only changes the colours, not the shapes. Everything comes from sampling noise, "
#              "which is how image generators start too.")
# ```
# Small `smooth` leaves fine grain, large `smooth` gives big soft shapes; the palette never moves a shape.
# </details>

# %% [markdown]
# ### 4.2 Forward diffusion by hand
#
# In the lesson's **diffusion figure**, a picture is destroyed step by step by adding noise, and the model is trained to undo it. The destroying direction is simple maths. With a noise schedule $\beta_t$ (here: 1000 steps, rising from $10^{-4}$ to $0.02$) we define $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \alpha_1\alpha_2\cdots\alpha_t$ (the share of the original picture that is still left). Then you can jump straight to any step $t$:
#
# $$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon \sim \mathcal N(0, 1).$$
#
# Our $x_0$ is a tiny 12×12 smiley, written as a grid of 0s and 1s and stretched to the range -1..1.

# %%
SMILEY = [
    "000111111000", "001000000100", "010000000010", "010100001010",
    "010100001010", "010000000010", "010100001010", "010011110010",
    "001000000100", "000111111000", "000000000000", "000000000000",
]
assert len(SMILEY) == 12 and all(len(r) == 12 for r in SMILEY)
x0 = np.array([[int(c) for c in row] for row in SMILEY], dtype=float) * 2 - 1   # 1 -> +1 (line), 0 -> -1 (background)

betas = np.linspace(1e-4, 0.02, 1000)       # the noise schedule
alpha_bars = np.cumprod(1 - betas)          # alpha_bars[t-1] is the lesson's alpha-bar at step t

def abar(t):
    """alpha-bar at step t (t = 0 means no noise yet, so 1.0)."""
    return 1.0 if t == 0 else float(alpha_bars[t - 1])

rng = np.random.default_rng(0)
eps = rng.standard_normal(x0.shape)         # ONE noise sample, reused so the steps look like one story
steps = [0, 50, 150, 300, 600, 1000]
fig, axes = plt.subplots(1, 6, figsize=(12, 2.3))
for ax, t in zip(axes, steps):
    xt = np.sqrt(abar(t)) * x0 + np.sqrt(1 - abar(t)) * eps      # the formula
    ax.imshow(xt, cmap="gray", vmin=-2, vmax=2); ax.set_title(f"t={t}\nabar={abar(t):.3g}"); ax.axis("off")
plt.tight_layout(); plt.show()
print("alpha-bar at t=300:", round(abar(300), 4), "  at t=1000:", f"{abar(1000):.2e}")
RESULTS["abar_300"] = round(abar(300), 4)
RESULTS["abar_1000"] = abar(1000)

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.2 - when is half the picture gone?
#
# 1. Find the **first step `t`** at which $\bar\alpha_t$ drops below 0.5 (less than half of the original signal is left). Use `abar(t)` in a loop, or `np.argmax`. Store it in `t_half`.
# 2. Look at your row of pictures. Around which step can you no longer see the smiley with your own eyes?
# 3. In `backwards_idea`, write one sentence on how a model could learn to run this process *backwards*. Hint: we know exactly which noise `eps` we added, so we have the right answer for every training example.

# %%
t_half = None          # YOUR CODE HERE: the first t with abar(t) < 0.5
backwards_idea = None  # YOUR ANSWER: one sentence

# %%
# Check
check("abar(300) is about 0.3964", abs(abar(300) - 0.3964) < 1e-3, "use the cumulative product of (1 - beta)")
check("abar(1000) is about 4.0e-5, almost no picture is left", abs(abar(1000) - 4.0e-5) < 5e-7)
assert abar(0) == 1.0 and abar(1) < 1.0, "alpha-bar should start at 1 and fall"
if t_half is None:
    print("Fill in `t_half` above, then run this cell again.")
else:
    check("t_half is the first t where abar(t) < 0.5", t_half == next(t for t in range(1, 1001) if abar(t) < 0.5),
          "loop over t = 1, 2, 3, ... and stop at the first one below 0.5")
if backwards_idea is None:
    print("Fill in `backwards_idea` above.")
else:
    check("your sentence is at least 10 words", n_words(backwards_idea) >= 10)
RESULTS["t_half"] = t_half

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# t_half = next(t for t in range(1, 1001) if abar(t) < 0.5)   # 260
# backwards_idea = ("Show the network a noisy picture and the step t, ask it to predict the noise that was added, "
#                   "and subtract that noise a little at a time, starting from pure noise.")
# ```
# $\bar\alpha_t$ first falls below 0.5 at $t = 260$. By about $t = 300$ the smiley is hard to see, and at $t = 1000$ it is gone: that last picture is just noise. A diffusion model learns to predict the noise, and generation runs the steps in reverse from pure noise.
# </details>

# %% [markdown]
# ### 4.3 Text to image with Stable Diffusion (GPU only)
#
# Now the real thing. **Stable Diffusion v1.5** runs the reverse process: it starts from random noise and, guided by your text, removes noise step by step until a picture appears. It needs a GPU. In Colab choose **Runtime → Change runtime type → T4 GPU** and run the notebook again from the top. The model is about 4 GB, so the first download takes a few minutes. On a CPU the next cells print a friendly "skipped" message and you can go on to Part 5.
#
# We use half precision (`float16`), 25 steps and 512×512 pixels so it stays quick.

# %%
# nb-test: skip
if device == "cuda":
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)

    def draw(prompt, seed=0, guidance=7.5, steps=25):
        """One 512x512 picture. The seed fixes the starting noise; guidance says how strongly to follow the text."""
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=g, height=512, width=512).images[0]

    def show(images, titles):
        fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.3))
        for ax, im, t in zip(np.atleast_1d(axes), images, titles):
            ax.imshow(im); ax.set_title(t, fontsize=8, wrap=True); ax.axis("off")
        plt.tight_layout(); plt.show()
    print("Stable Diffusion is ready.")
else:
    print("Skipped: this step needs a GPU. Switch to a T4 GPU runtime if you want to try it; otherwise go on to the journal and Part 5.")

# %% [markdown]
# Three prompts for the picture you imagined, from vague to specific. Edit the three prompts so they describe **your** picture (the same scene, with more and more detail: subject, style, light, colours, camera view).

# %%
# nb-test: skip
if device == "cuda":
    prompts = [
        "a lighthouse",                                                                              # vague
        "a lighthouse on a cliff at sunset",                                                         # more specific
        "a lighthouse on a cliff at sunset, warm orange light, calm sea, watercolor painting, wide shot",  # very specific
    ]
    imgs = [draw(p, seed=1) for p in prompts]
    show(imgs, prompts)
else:
    print("Skipped (no GPU).")

# %% [markdown]
# The same prompt with four different seeds, then one seed with three different **guidance scales** (3 = loosely follows the text, 7.5 = the usual, 12 = follows it hard).

# %%
# nb-test: skip
if device == "cuda":
    p = prompts[2]
    show([draw(p, seed=s) for s in range(4)], [f"seed {s}" for s in range(4)])
    show([draw(p, seed=1, guidance=g) for g in (3, 7.5, 12)], [f"guidance {g}" for g in (3, 7.5, 12)])
else:
    print("Skipped (no GPU).")

# %% [markdown]
# ### ✏️ Your turn: Exercise 4.3 - your imagination journal
#
# Fill in the journal below (in words, a sentence or two each):
# - `pictured`: what you pictured at the start of Part 4.
# - `model_gave`: what the model really gave you for your most specific prompt. What matched, what surprised you?
# - `words_that_mattered`: which words in the prompt changed the picture most (style words like "watercolor"? light? place?).
# - `seeds_and_guidance`: what changed when you changed only the seed, and what changed when you changed only the guidance scale?
#
# No GPU? Write the journal about the noise art from 4.1 instead (what you pictured and what the seeds gave you).

# %%
journal = {
    "pictured": None,
    "model_gave": None,
    "words_that_mattered": None,
    "seeds_and_guidance": None,
}

# %%
# Check
empty = [k for k, v in journal.items() if n_words(v) < 8]
if len(empty) == len(journal):
    print("Fill in the journal above (at least 8 words per entry), then run this cell again.")
else:
    check("every journal entry has at least 8 words", not empty, "still short: " + ", ".join(empty))
RESULTS["p4_journal"] = journal

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# journal = {
#     "pictured": "A lighthouse on a cliff at sunset, with warm colours and a calm sea.",
#     "model_gave": "A lighthouse at sunset with orange sky. It matched the colours but added a small boat I did not ask for.",
#     "words_that_mattered": "'watercolor' made it soft and painted; 'wide shot' made the lighthouse smaller; 'sunset' set all the colours.",
#     "seeds_and_guidance": "A different seed changed the layout (cliff, sea, clouds) but kept the idea. Guidance 3 drifted away from the text, 12 looked harsh and over-saturated.",
# }
# ```
# Your answers will differ: that is the point. The prompt steers the picture, the seed decides which of many possible pictures you get.
# </details>

# %%
# Free the GPU before moving on.
if device == "cuda" and "pipe" in globals():
    del pipe
    free_gpu()
    print("Stable Diffusion removed from memory.")
else:
    print("Nothing to free.")

# %% [markdown]
# ## Part 5. Imagine in sound and across languages (25 min)
#
# The lesson section **Beyond text** also covers speech and translation. Three more kinds of generator share the same recipe: translation models write text in another language, text-to-speech models write sound, and speech-to-text models write text from sound. In this part you test each one and measure how much is lost when they are chained.
#
# **Downloads in this part:** two translation models (about 0.3 GB each), a speech model `facebook/mms-tts-eng` (about 0.15 GB) and `openai/whisper-tiny` (about 0.15 GB). They run fine on a CPU.

# %% [markdown]
# ### ✏️ Imagine first
#
# Take the sentence *"The robot club meets on Friday."* Predict what it will look like after it is translated from English to French and then back to English. Will it come back unchanged, slightly different, or very different? Write your prediction.

# %%
p5_prediction = ""   # YOUR TURN: e.g. "It will come back almost the same, maybe with 'meets' changed to 'gathers'."
print("Your prediction:", p5_prediction if p5_prediction else "(write it above and run this cell again)")

# %% [markdown]
# ### 5.1 Translation round trip
#
# We use two small models from the Helsinki-NLP group: one translates English to French, the other French to English. A **round trip** (English → French → English) is a cheap test of how much meaning survives. We use three sentences of growing difficulty: a plain one, an idiom, and a joke built on ambiguity.

# %%
from transformers import pipeline, MarianMTModel, MarianTokenizer
pipe_dev = 0 if device == "cuda" else -1

def load_translator(name):
    """Load one Helsinki-NLP translation model (about 0.3 GB) and return a function text -> translated text."""
    mt_tok = MarianTokenizer.from_pretrained(name)
    mt = MarianMTModel.from_pretrained(name).to(device).eval()
    def translate(text):
        batch = mt_tok([text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = mt.generate(**batch, max_new_tokens=80, num_beams=4)
        return mt_tok.decode(out[0], skip_special_tokens=True)
    return translate

en_fr = load_translator("Helsinki-NLP/opus-mt-en-fr")
fr_en = load_translator("Helsinki-NLP/opus-mt-fr-en")

originals = [
    "The robot club meets on Friday.",                                   # plain
    "It is raining cats and dogs.",                                      # idiom
    "Time flies like an arrow; fruit flies like a banana.",              # joke / ambiguity
]
forward = [en_fr(s) for s in originals]
back = [fr_en(s) for s in forward]

for o, f, b in zip(originals, forward, back):
    print("English :", o)
    print("French  :", f)
    print("Back    :", b)
    print()

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.1 - what was lost?
#
# 1. Compare each original with its round-trip version. Was the plain sentence kept? What happened to the idiom? What happened to the joke?
# 2. Fill in `lost` below with one short note per sentence (what meaning, nuance or wordplay was lost, or "nothing lost").
# 3. Optional: add your own sentence to `originals` and run the two translation lines again. Try one with a word that has two meanings (for example "bank" or "light").
# 4. Look back at your prediction. Were you right?

# %%
lost = {
    "plain": None,
    "idiom": None,
    "joke": None,
}

# %%
# Check
done = [k for k, v in lost.items() if n_words(v) >= 2]
if not done:
    print("Fill in `lost` above (at least 2 words per sentence), then run this cell again.")
else:
    check("you wrote a note for all three sentences", len(done) == 3, "still empty: " + ", ".join(k for k in lost if k not in done))
check("the models returned three round trips", len(back) == 3 and all(isinstance(b, str) and b for b in back))
RESULTS["roundtrip"] = list(zip(originals, forward, back))
RESULTS["p5_lost"] = lost

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# lost = {
#     "plain": "Nothing important lost: the meeting day and the club survived (small word changes only).",
#     "idiom": "The French was literal ('Il pleut des chats et des chiens'); a French speaker would say 'Il pleut des cordes'. The round trip hides the loss because the literal route comes back.",
#     "joke": "The words came back, but the wordplay on 'flies' (verb vs insect) cannot work in French, so the joke is lost even though the text looks unchanged.",
# }
# ```
# A round trip that looks unchanged does not prove the meaning survived: the middle step may have been a literal translation. Idioms and jokes lean on shared culture and double meanings.
# </details>

# %% [markdown]
# ### 5.2 Measuring the damage: word error rate
#
# "It looks about the same" is not a measurement. **Word error rate (WER)** counts how many word-level edits turn the reference (the correct text) into the hypothesis (what the model produced), divided by the number of words in the reference:
#
# $$\text{WER} = \frac{S + I + D}{N}$$
#
# where S = substitutions, I = insertions, D = deletions, N = words in the reference. It is the *edit distance* between the two word lists. A WER of 0 is perfect; it can go above 1 if the model adds many extra words. We first clean the text: lower case, no punctuation.

# %%
import re

def clean(text):
    """Lower case, keep only letters, digits, apostrophes and spaces."""
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()

def word_error_rate(reference, hypothesis):
    """(substitutions + insertions + deletions) / number of reference words, by edit distance."""
    r, h = clean(reference), clean(hypothesis)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]      # d[i][j] = edits to turn r[:i] into h[:j]
    for i in range(len(r) + 1): d[i][0] = i                  # i deletions
    for j in range(len(h) + 1): d[0][j] = j                  # j insertions
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            same = 0 if r[i - 1] == h[j - 1] else 1          # substitution costs 1 unless the words match
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + same)
    return d[len(r)][len(h)] / max(1, len(r))

print(word_error_rate("the cat sat", "the cat sat"))
print(word_error_rate("the cat sat", "a cat sat down"))
for o, b in zip(originals, back):
    print(round(word_error_rate(o, b), 2), "|", o, "->", b)

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.2 - predict, then test
#
# Work out these WERs **by hand** (count substitutions, insertions and deletions), put them in `hand`, then run the Check.
# - `a`: reference `"i like green tea"`, hypothesis `"i like tea"` (hint: one word was dropped).
# - `b`: reference `"open the door"`, hypothesis `"close the window now"`.
# - `c`: reference `"go"`, hypothesis `"go there right now please"`. This one is above 1: why can that happen?

# %%
hand = {"a": None, "b": None, "c": None}   # YOUR ANSWERS: numbers, e.g. 0.5

# %%
# Check
check("identical sentences give 0.0", word_error_rate("the cat sat", "the cat sat") == 0.0)
check("'the cat sat' vs 'a cat sat down' is 2/3", abs(word_error_rate("the cat sat", "a cat sat down") - 2 / 3) < 1e-9)
assert word_error_rate("a b c", "") == 1.0, "an empty hypothesis deletes every word"
truth = {"a": word_error_rate("i like green tea", "i like tea"),
         "b": word_error_rate("open the door", "close the window now"),
         "c": word_error_rate("go", "go there right now please")}
if any(v is None for v in hand.values()):
    print("Fill in all three numbers in `hand`, then run this cell again.")
else:
    for k in "abc":
        check(f"case {k}", abs(hand[k] - truth[k]) < 0.01, "count S, I, D again and divide by the number of reference words")
RESULTS["wer_roundtrip"] = [round(word_error_rate(o, b), 3) for o, b in zip(originals, back)]

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# hand = {"a": 0.25, "b": 1.0, "c": 4.0}
# ```
# a: one deletion out of 4 reference words = 1/4. b: two substitutions (open→close, door→window) and one insertion (now) = 3 edits out of 3 words = 1.0. c: four insertions out of 1 reference word = 4.0, because WER divides by the reference length, so it can exceed 1.
# </details>

# %% [markdown]
# ### 5.3 The audio chain
#
# Now chain three generators: **GPT-2 writes** a short text, **MMS-TTS speaks** it (text to speech), and **Whisper-tiny listens** and writes it down again (speech to text). The WER between the original text and the final transcript tells you how much was lost on the way. In Colab you can press play on the audio.

# %%
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile
from IPython.display import Audio, display

torch.manual_seed(SEED)
story = generate("The robot club built a small robot that", max_new_tokens=22, do_sample=True, temperature=0.7, top_k=40)
original_text = " ".join(story.split())          # one line, no line breaks
if "." in original_text:                          # drop a half-finished last sentence
    original_text = original_text[:original_text.rindex(".") + 1]
print("GPT-2 wrote:", original_text)

tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
tts = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=pipe_dev)

def speak_and_listen(text, path="chain.wav"):
    """text -> speech (MMS-TTS) -> text (Whisper-tiny). Returns (audio, sampling rate, what Whisper heard)."""
    with torch.no_grad():
        wave = tts(**tts_tok(text, return_tensors="pt")).waveform[0].numpy()
    sr = tts.config.sampling_rate
    wavfile.write(path, sr, wave)                     # saved in the notebook's folder (/content in Colab)
    heard = asr({"raw": wave.astype(np.float32), "sampling_rate": sr}, generate_kwargs={"language": "en"})["text"].strip()
    return wave, sr, heard

wave, sr, heard = speak_and_listen(original_text)
print(f"Spoken audio: {len(wave) / sr:.1f} seconds at {sr} Hz, saved as chain.wav")
display(Audio(wave, rate=sr))
chain_wer = word_error_rate(original_text, heard)
print("Original:", original_text)
print("Heard   :", heard)
print(f"Word error rate: {chain_wer:.2f}")

# %% [markdown]
# That one may have come back almost perfect. Now a **harder** text with numbers, a rare name and an abbreviation: the kinds of words that trip speech systems up.

# %%
hard_text = "Dr. Okonkwo built robot ZX-9 in 2025 for 3 young engineers from Cebu."
wave2, sr2, heard2 = speak_and_listen(hard_text, path="chain_hard.wav")
display(Audio(wave2, rate=sr2))
chain_wer_hard = word_error_rate(hard_text, heard2)
print("Original:", hard_text)
print("Heard   :", heard2)
print(f"Word error rate: {chain_wer_hard:.2f}")
RESULTS["chain_original"] = original_text
RESULTS["chain_heard"] = heard
RESULTS["chain_wer"] = round(chain_wer, 3)
RESULTS["chain_wer_hard"] = round(chain_wer_hard, 3)

# %% [markdown]
# ### ✏️ Your turn: Exercise 5.3 - why are they different?
#
# 1. Compare each original with what Whisper heard and mark every difference (a changed word, missing punctuation, a number written as digits, a rare name).
# 2. In `reasons`, list **two reasons** the final text differs from the original. Think about: punctuation and capital letters (speech has none), rare or invented words, numbers, the synthetic voice, a tiny recognizer (`whisper-tiny` is the smallest Whisper), run-on sentences.
# 3. Change `hard_text` to your own sentence (add a name from your community) and run its cell again. Which words does the chain get wrong?

# %%
reasons = []   # YOUR ANSWER: a list of two strings, e.g. ["...", "..."]

# %%
# Check
check("the chain produced text", len(original_text) > 0 and len(heard) > 0)
check("the word error rates are sensible numbers (not negative)", chain_wer >= 0 and chain_wer_hard >= 0)
if len(reasons) < 2:
    print("Write two reasons in `reasons` above, then run this cell again.")
else:
    check("two reasons, each at least 4 words", all(n_words(r) >= 4 for r in reasons[:2]), "write a full sentence for each reason")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reasons = [
#     "The speech model reads the text but the transcript has different punctuation and capital letters, so some differences are just formatting.",
#     "Whisper-tiny is a very small recognizer, so a rare or invented word, or a word the synthetic voice pronounced oddly, can be heard as a different word.",
# ]
# ```
# Real differences usually come from rare words, numbers and names, an imperfect synthetic voice, and the small size of the recognizer. Cleaning punctuation before computing WER (as `clean` does) removes the formatting differences.
# </details>

# %% [markdown]
# ### 5.4 Imagine: a voice assistant for your community
#
# Imagine a voice assistant for a language your community speaks (a regional language, a dialect, a school's home language). Which of the models you met would you need, and what data would they need? Fill in the plan below. Be concrete: how many hours of recorded speech, who would record it, who checks it.

# %%
voice_plan = {
    "language": None,
    "models_needed": None,      # e.g. which of: speech-to-text, translation, language model, text-to-speech
    "data_needed": None,        # what recordings or texts, how much, who provides them
    "who_checks_it": None,      # which speakers review the output before people rely on it
}

# %%
# Check
empty = [k for k, v in voice_plan.items() if n_words(v) < 3]
if len(empty) == len(voice_plan):
    print("Fill in the plan above (at least 3 words per entry), then run this cell again.")
else:
    check("every entry is filled in", not empty, "still short: " + ", ".join(empty))
RESULTS["p5_voice_plan"] = voice_plan

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# voice_plan = {
#     "language": "Ilocano, spoken by many families in my town.",
#     "models_needed": "Speech-to-text to understand the user, a translation or language model to find an answer, and text-to-speech to reply.",
#     "data_needed": "Many hours of recorded speech with written transcripts from speakers of different ages and villages, plus written text in the language.",
#     "who_checks_it": "Native speakers from the community test the replies and correct mistakes before release.",
# }
# ```
# Small languages have little recorded data, which is exactly why these models work worse for them: the data decides the quality.
# </details>

# %%
# Free the Part 5 models before the next part.
for name in ("en_fr", "fr_en", "tts", "asr"):
    globals().pop(name, None)
free_gpu()
print("Part 5 models removed from memory.")

# %% [markdown]
# ## Part 6. Imagine an AI product, then try to break it (30 min)
#
# Good engineers do two things at once: they imagine what an AI could do for people, and they try hard to find out how it could fail. This part uses the lesson sections **Risks** (trustworthy AI and its five principles: fairness, transparency, privacy, safety, accountability) and **Prompts**. You design a product, test GPT-2 for two kinds of weakness, write a pre-mortem, and finish with a short "model card".

# %% [markdown]
# ### ✏️ Imagine first
#
# Choose a **real problem in your school or community** that an AI might help with (for example: finding library books, answering common questions from new students, translating notices for parents, summarising long announcements). Write it in one sentence.

# %%
p6_problem = ""   # YOUR TURN: e.g. "New students cannot find which club meets when."
print("Your problem:", p6_problem if p6_problem else "(write it above and run this cell again)")

# %% [markdown]
# ### 6.1 The AI spec sheet
#
# A spec sheet makes you decide *before* you build. Fill in the dictionary for the AI product that would solve your problem. Then the helper prints it as a neat card.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.1 - fill in your spec sheet
#
# Fill in every field with a short phrase or sentence. For `model_type`, choose from the lesson: a language model, a translation model, a speech model (speech to text or text to speech), an image-generation (diffusion) model, or a classifier. For `human_in_charge`, say who checks the output and what they can do if it is wrong.

# %%
spec = {
    "name": None,              # e.g. "ClubFinder"
    "who_it_helps": None,
    "input": None,             # what the user gives it
    "output": None,            # what it gives back
    "model_type": None,
    "learns_from": None,       # what data the model learns from
    "human_in_charge": None,
    "success_measure": None,   # one number you could measure, e.g. "% of answers a teacher marks correct"
}

def print_card(card):
    width = 62
    print("+" + "-" * width + "+")
    for k, v in card.items():
        text = v if v else "(fill me in)"
        line = f"{k.replace('_', ' ').title():<16}: {text}"
        while len(line) > width - 2:                     # wrap long lines
            print("| " + line[:width - 2].ljust(width - 2) + " |"); line = " " * 18 + line[width - 2:].lstrip()
        print("| " + line.ljust(width - 2) + " |")
    print("+" + "-" * width + "+")

print_card(spec)

# %%
# Check
missing = [k for k, v in spec.items() if n_words(v) < 1]
if len(missing) == len(spec):
    print("Fill in the spec above, run its cell, then run this cell again.")
else:
    check("every field is filled in", not missing, "still empty: " + ", ".join(missing))
RESULTS["p6_spec"] = spec

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# spec = {
#     "name": "ClubFinder",
#     "who_it_helps": "New students who do not know which clubs meet when.",
#     "input": "A short question typed in the school website, e.g. 'Is there a robotics club on Friday?'",
#     "output": "A short answer with the club name, room and time, and a link to the club teacher.",
#     "model_type": "A language model that reads the club list and writes the answer.",
#     "learns_from": "The school's own up-to-date club list (not the whole internet).",
#     "human_in_charge": "A teacher checks the club list every term and can switch the assistant off or correct it.",
#     "success_measure": "Share of 50 test questions answered correctly, checked by a teacher (target: at least 90%).",
# }
# ```
# A good spec names the data, the human and a measurable goal.
# </details>

# %% [markdown]
# ### 6.2 Fluent but wrong
#
# Language models are built to produce *likely* text, not *true* text. Here is a test: six factual prompts with known answers. For each we ask GPT-2 for its **five most likely next tokens** and mark whether the true answer is among them. A tool that sounds sure about a wrong answer is dangerous, so we measure it.

# %%
facts = [
    ("The capital of France is", "Paris"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The Earth orbits around the", "Sun"),
    ("The first month of the year is", "January"),
    ("There are seven days in a", "week"),
]

def fact_table(items, top=5):
    """Print the model's top guesses and whether the true answer is among them. Returns the fraction found."""
    hits = 0
    for prompt, truth in items:
        guesses = [t.strip().lower() for t, p in next_token_probs(prompt, top=top)]
        found = truth.lower() in guesses
        hits += found
        print(("✓" if found else "✗"), f"{prompt!r:52} true: {truth:8} top-{top}: {guesses}")
    return hits / len(items)

fact_rate = fact_table(facts)
print(f"\nThe true answer was in GPT-2's top 5 for {fact_rate:.0%} of the prompts.")
RESULTS["fact_rate"] = round(fact_rate, 3)

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.2 - add your own facts and explain
#
# 1. Add **two prompts of your own** to `my_facts` (a prompt that ends right before the answer, and the one-word true answer). Run `fact_table(my_facts)`. Choose prompts whose answer you are certain about.
# 2. In `why_wrong`, explain in 2 or 3 sentences why a model can write a fluent sentence that is false. Use the words *likely* and *true*.

# %%
my_facts = []        # YOUR CODE HERE: e.g. [("The opposite of hot is", "cold"), ("A triangle has three", "sides")]
if my_facts:
    my_rate = fact_table(my_facts)
why_wrong = None     # YOUR ANSWER: a string

# %%
# Check
check("the table found the true answer for at least one prompt", fact_rate > 0, "check that fact_table ran")
check("you added at least two facts", len(my_facts) >= 2, "add two (prompt, answer) pairs to my_facts") if my_facts else print("Add two prompts to `my_facts`, then run again.")
if why_wrong is None:
    print("Fill in `why_wrong`, then run this cell again.")
else:
    check("your explanation is at least 15 words and mentions 'likely'", n_words(why_wrong) >= 15 and "likely" in why_wrong.lower(),
          "explain that the model picks likely words, not true ones")
RESULTS["p6_why_wrong"] = why_wrong

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# my_facts = [("The opposite of hot is", "cold"), ("A triangle has three", "sides")]
# why_wrong = ("The model is trained to continue text with whatever is likely, and likely is not the same as true. "
#              "If a wrong answer appears often in similar text, or the right one is rare, a fluent sentence can still be false.")
# ```
# Small GPT-2 has seen a lot of text but has no way to check a fact. Larger models know more, but the same weakness remains, which is why a human or a trusted source must check important answers.
# </details>

# %% [markdown]
# ### 6.3 A bias probe, carefully
#
# Language models learn patterns from text written by people, so they can pick up the stereotypes in that text. This is a **risk to test for and reduce**, not a verdict on any person or job. Here is a simple probe: after "The nurse said that" and "The engineer said that", how likely is the next word to be "he", "she" or "they"? We read the model's probabilities for the next word (no free generation), then also **draw 20 samples** of the next word from the 200 most likely tokens to see how counts bounce around.
#
# *Handling results responsibly:* describe what you measured and how (one prompt, one small model, a few words), do not generalise to real people, and do not publish or pass around offensive outputs. If you see something hurtful, report it to your teacher and discuss how to fix it rather than repeating it.

# %%
def pronoun_probe(job, n=20, seed=0):
    prompt = f"The {job} said that"
    top = next_token_probs(prompt, top=200)                           # the 200 most likely next tokens
    prob = {}
    for t, p in top:                                                  # add up " she", " She", ... (words that start with a space)
        if t.startswith(" "):
            prob[t.strip().lower()] = prob.get(t.strip().lower(), 0.0) + p
    exact = {w: prob.get(w, 0.0) for w in ("he", "she", "they")}
    tokens, weights = [t.strip().lower() for t, p in top], np.array([p for t, p in top])
    draws = np.random.default_rng(seed).choice(tokens, size=n, p=weights / weights.sum())   # 20 samples of the next word
    counts = {w: int((draws == w).sum()) for w in ("he", "she", "they")}
    return exact, counts

for job in ("nurse", "engineer"):
    exact, counts = pronoun_probe(job)
    print(f"The {job} said that ...")
    print("   probabilities:", {w: round(p, 3) for w, p in exact.items()})
    print("   counts in 20 samples:", counts)
    RESULTS[f"bias_{job}"] = {w: round(p, 3) for w, p in exact.items()}

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.3 - read the probe, then plan a fix
#
# 1. Run the probe for two other jobs of your choice (change the list below). Is the pattern the same?
# 2. Run the 20-sample counts with a different `seed`. How much do the counts change? What does that say about trusting a small test?
# 3. In `bias_note`, write 3 sentences: what pattern you saw (state it as "this model, on this prompt"), *why* it might exist (what the model learned from), and what a team could do about it (more balanced training data, an evaluation set that checks it, human review before use, or a design that avoids using the model for such decisions).

# %%
for job in ["teacher", "pilot"]:                    # YOUR TURN: use two jobs of your own
    exact, counts = pronoun_probe(job, seed=1)
    print(job, {w: round(p, 3) for w, p in exact.items()}, counts)

bias_note = None    # YOUR ANSWER: three sentences

# %%
# Check
exact, counts = pronoun_probe("nurse")
check("the probabilities are valid (between 0 and 1)", all(0 <= p <= 1 for p in exact.values()))
check("the 20 samples were counted", sum(counts.values()) <= 20)
if bias_note is None:
    print("Fill in `bias_note`, then run this cell again.")
else:
    ok = n_words(bias_note) >= 25 and any(w in bias_note.lower() for w in ("data", "human", "review", "evaluat", "test", "balanced"))
    check("your note is at least 25 words and names a way to reduce the risk", ok, "describe the pattern, a cause and a fix")
RESULTS["p6_bias_note"] = bias_note

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# bias_note = ("For this model and these two prompts, the next-word probabilities for he and she differ by job, so the model has learned "
#              "a stereotype. It learned it from human text, where such patterns are common, not because anyone chose it. "
#              "A team could test it with a balanced evaluation set, add more balanced data, keep a human reviewing the output, "
#              "and avoid using the model for decisions about people.")
# ```
# The honest wording is "this model, on these prompts". One small probe does not describe the world, but it tells you where to look harder before anyone relies on a model.
# </details>

# %% [markdown]
# ### 6.4 Pre-mortem
#
# A **pre-mortem** imagines the project has already failed and asks why. Look at the product in your spec sheet. Fill in three ways it could fail. For each one: who would be harmed, which of the five principles it touches (`fairness`, `transparency`, `privacy`, `safety` or `accountability`), and a fix.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.4 - fill in the pre-mortem
#
# Replace each `None` with a short phrase. Use exactly one of the five principle names in the `principle` field.

# %%
premortem = [
    {"failure": None, "who_is_harmed": None, "principle": None, "fix": None},
    {"failure": None, "who_is_harmed": None, "principle": None, "fix": None},
    {"failure": None, "who_is_harmed": None, "principle": None, "fix": None},
]
for i, row in enumerate(premortem, 1):
    print(i, row)

# %%
# Check
FIVE = {"fairness", "transparency", "privacy", "safety", "accountability"}
filled = [r for r in premortem if all(n_words(v) >= 1 for v in r.values())]
if not filled:
    print("Fill in the three rows above, then run this cell again.")
else:
    check("all three rows are filled in", len(filled) == 3, "some rows are still empty")
    bad = [r["principle"] for r in filled if str(r["principle"]).strip().lower() not in FIVE]
    check("each principle is one of the five", not bad, "use: fairness, transparency, privacy, safety or accountability")
RESULTS["p6_premortem"] = premortem

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# premortem = [
#     {"failure": "Gives a wrong club time with confidence", "who_is_harmed": "A new student who misses the meeting",
#      "principle": "safety", "fix": "Answer only from the club list and show the source; a teacher checks it each term"},
#     {"failure": "Works well in English but badly in other home languages", "who_is_harmed": "Students and parents who speak other languages",
#      "principle": "fairness", "fix": "Test in each language and show a note when confidence is low"},
#     {"failure": "Stores students' names from their questions", "who_is_harmed": "Any student asking a question",
#      "principle": "privacy", "fix": "Do not save questions with names; delete logs after 30 days"},
# ]
# ```
# Each fix is something the team can check. "Be more careful" is not a fix.
# </details>

# %% [markdown]
# ### 6.5 Model card lite
#
# A **model card** is a short document that tells people what a model is for and where it fails. Write a five-sentence "lite" card for your product, one sentence each: intended use, not intended for, known limits, how it was tested, who to contact.

# %% [markdown]
# ### ✏️ Your turn: Exercise 6.5 - write your model card
#
# Use what you found in 6.2 to 6.4 (for example the share of facts the model got right, or the bias probe) in "known limits" and "how it was tested". Keep each entry to one sentence. For the contact, write a role (for example "the school's IT teacher"), not a personal email or phone number.

# %%
model_card = {
    "intended_use": None,
    "not_intended_for": None,
    "known_limits": None,
    "how_it_was_tested": None,
    "contact": None,
}

# %%
# Check
empty = [k for k, v in model_card.items() if n_words(v) < 4]
if len(empty) == len(model_card):
    print("Fill in the five sentences above, then run this cell again.")
else:
    check("all five entries have at least 4 words", not empty, "still short: " + ", ".join(empty))
RESULTS["p6_model_card"] = model_card

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# model_card = {
#     "intended_use": "ClubFinder answers questions about which school clubs meet and when.",
#     "not_intended_for": "Medical, legal or discipline questions, or any decision about a student.",
#     "known_limits": "It can sound sure and still be wrong, and small tests showed uneven behaviour for some prompts, so answers must be checked against the club list.",
#     "how_it_was_tested": "We asked 50 questions with known answers, and a teacher marked each answer correct or wrong.",
#     "contact": "The school's IT teacher.",
# }
# ```
# </details>

# %% [markdown]
# ## Part 7. Gather it up and reflect (10 min)
#
# The last part collects what you found into one summary you can submit, and asks you to think about it. First, run the cell: it prints every number and every piece of writing stored in `RESULTS` during the notebook.

# %%
def show_value(v, indent="   "):
    if isinstance(v, dict):
        for k, x in v.items():
            print(f"{indent}{k}: {x if x is not None else '(not filled in)'}")
    elif isinstance(v, (list, tuple)):
        for x in v:
            print(f"{indent}- {x}")
    else:
        print(f"{indent}{v if v is not None else '(not filled in)'}")

print("=" * 60)
print("WEEK 1 SUMMARY")
print("=" * 60)
for key, value in RESULTS.items():
    print(f"\n[{key}]")
    show_value(value)
print("\n" + "=" * 60)
print(f"{len(RESULTS)} items collected.")

# %% [markdown]
# ### ✏️ Your turn: Exercise 7.1 - reflection
#
# Answer the four questions below in 2 to 3 sentences each. Be specific: name something you actually saw in this notebook (a number, a picture, a sentence the model wrote).
#
# 1. What surprised you most, and why?
# 2. Name one thing generative AI is better at than you expected, and one thing it is worse at.
# 3. What would you never trust it with, and what would you need to see before you changed your mind?
# 4. The lesson says generative models "learn a distribution, then sample". Pick something you saw (a GPT-2 sentence, the noise art, a translation, a wrong fact) and explain it with that idea.

# %%
reflection = {
    "1_surprised": None,
    "2_better_and_worse": None,
    "3_never_trust": None,
    "4_distribution_then_sample": None,
}

# %%
# Check
short = [k for k, v in reflection.items() if n_words(v) < 15]
if len(short) == len(reflection):
    print("Write your four answers above (at least 15 words each), then run this cell again.")
else:
    check("all four answers have at least 15 words", not short, "still short: " + ", ".join(short))
RESULTS["p7_reflection"] = reflection
print("\nYour reflection:")
for k, v in reflection.items():
    print(f"  {k}: {v if v else '(not filled in)'}")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# reflection = {
#     "1_surprised": "It surprised me that a simple blur of random noise already looks like art, and that the same seed always gave the same picture.",
#     "2_better_and_worse": "GPT-2 writes fluent sentences better than I expected, but it is worse at facts: for one easy question the true answer was not even in its top five.",
#     "3_never_trust": "I would never trust it alone with health or legal advice, because it can sound sure while being wrong. I would need a human check and tests.",
#     "4_distribution_then_sample": "Each time the model gives a probability to every next word and then draws one, so the same prompt gives different sentences, like different seeds giving different art.",
# }
# ```
# Your answers should come from what you saw. There is no single right answer.
# </details>

# %% [markdown]
# ## Wrap-up
#
# What you learned this week:
#
# - **Language models** write by predicting the next token from a probability distribution, and you can control creativity with temperature, top-k and top-p. Fluent is not the same as true.
# - **Other generators** follow the same recipe: pictures come from removing noise step by step, and translation and speech models turn one kind of sequence into another. Chaining them loses information, which you can measure (word error rate).
# - **Trustworthy AI** is about testing before trusting: you wrote a spec sheet, tested for wrong facts and bias, and planned fixes with a pre-mortem and a model card.
#
# **Before you finish.** Files in `/content` (such as `chain.wav`) vanish when the Colab session ends. Download anything you want to keep, and use **File → Save a copy in Drive** (or download the notebook with its outputs) so your answers survive. Submit it the way your instructor asks.
#
# **Reflect:** double-click this cell and write two sentences: which part of this lab would you like to explore further, and what would you try?
