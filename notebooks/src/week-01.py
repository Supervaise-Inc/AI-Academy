# accelerator: GPU

# %% [markdown]
# # Week 1 — Introduction to Generative AI: lab notebook
#
# Built for **Google Colab**. First click **File → Save a copy in Drive**, so your work is saved in your own copy.
#
# **Runtime:** a GPU is recommended for the image step only: Runtime → Change runtime type → **T4 GPU**. Every other step runs on a CPU too, and the notebook tells you what it skips when there is no GPU. **Time:** about 3 hours.
#
# **What you will do**
# - Check your environment and the GPU.
# - Generate text with GPT-2 and compare two temperatures.
# - Translate a sentence into French, German and Spanish and back, and see what is lost.
# - Generate three images with Stable Diffusion (GPU only).
# - Turn text into speech, transcribe it again, and measure the difference.
# - Reflect on quality, uses and risks.
#
# **How this notebook works:** run the cells from top to bottom with the ▶ button. **Your turn** cells are for you to fill in. A **Check** cell tells you if you are right. Open the **Show a possible answer** block only after you tried.
#
# Downloads are public models from Hugging Face. No account or key is needed. The first run of each model downloads it, so those cells take longer.

# %% [markdown]
# ## Setup
#
# This cell installs packages Colab may not include (`diffusers` for images, `accelerate`, and `sentencepiece` for the translation tokenizer), imports what we need, fixes the random seed, picks the GPU if there is one, and defines a `check` helper that prints ✓ or ✗.

# %%
#!pip install -q diffusers accelerate sentencepiece
import os, gc, math, difflib
import numpy as np
import torch
from transformers import set_seed

SEED = 42
set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
QUICK = bool(os.environ.get("NB_TEST"))   # shortens long loops when the notebook is tested

def check(name, ok, hint=""):
    print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))

def free_memory():
    """Delete your model variables first, then call this to give the memory back."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# ## Step 1. Check the environment
#
# We add two small tensors and, if there is a GPU, print its name. If you see `cpu`, that is fine for every step except images: choose Runtime → Change runtime type → T4 GPU to enable them.

# %%
a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)
print("torch", torch.__version__, "| GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print(a + b)
assert (a + b).tolist() == [5.0, 7.0, 9.0]

# %% [markdown]
# ## Step 2. Text generation with GPT-2
#
# From the lesson section on how a language model writes: the model scores every possible next word and picks one, over and over. We use `gpt2` (about 0.5 GB to download). `do_sample=True` lets it pick randomly according to its scores, and `temperature` controls how adventurous the pick is. We load it once and reuse it in Step 5.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

def generate_text(prompt, new_tokens=50, seed=SEED, **kw):
    set_seed(seed)
    ids = tok(prompt, return_tensors="pt").to(device)
    out = lm.generate(**ids, max_new_tokens=new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0], skip_special_tokens=True)

print(generate_text("Once upon a time in a land far, far away", do_sample=True, temperature=0.7))

# %% [markdown]
# ### ✏️ Your turn: temperature 0.7 against 1.3
#
# Write your own prompt and generate a story of **100 new tokens** at temperature 0.7 and again at 1.3. Read both. Then, in the last line, write 3–4 sentences comparing them. Use what you saw in the temperature figure in the lesson.

# %%
my_prompt = "Once upon a time in a land far, far away"   # change this to your own prompt
story_low = None    # YOUR CODE HERE: generate_text(my_prompt, 100, do_sample=True, temperature=0.7)
story_high = None   # YOUR CODE HERE: the same with temperature=1.3
my_comparison = ""  # YOUR ANSWER: 3-4 sentences comparing the two stories
for name, s in [("0.7", story_low), ("1.3", story_high)]:
    print("--- temperature", name, "---\n", s)

# %%
# Check
def distinct_fraction(text):
    words = text.lower().split()
    return len(set(words)) / max(1, len(words))

if story_low is None or story_high is None:
    print("Fill in your answer above, then run this cell again.")
else:
    check("both stories generated", len(story_low) > len(my_prompt) and len(story_high) > len(my_prompt), "call generate_text with do_sample=True")
    n_sent = sum(my_comparison.count(c) for c in ".!?")
    check("comparison has at least 3 sentences", n_sent >= 3, "write 3-4 sentences in my_comparison")

# %% [markdown]
# A single story is a weak comparison. Let's measure over several samples. The cell below generates 5 stories at each temperature with different seeds and reports the **fraction of distinct words** (higher means more varied word choice) and how many stories contain a repeated three-word phrase.

# %%
def has_repeat(text, n=3):
    w = text.lower().split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(grams) != len(set(grams))

k = 2 if QUICK else 5
stats = {}
for temp in [0.7, 1.3]:
    outs = [generate_text(my_prompt, 100, seed=s, do_sample=True, temperature=temp) for s in range(k)]
    stats[temp] = (np.mean([distinct_fraction(o) for o in outs]), sum(has_repeat(o) for o in outs))
    print("temperature", temp, "| distinct-word fraction %.2f" % stats[temp][0], "| stories with a repeated 3-word phrase:", stats[temp][1], "of", k)

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# story_low = generate_text(my_prompt, 100, do_sample=True, temperature=0.7)
# story_high = generate_text(my_prompt, 100, do_sample=True, temperature=1.3)
# ```
# A possible comparison: "At 0.7 the story stays on topic and uses common words, but it can repeat itself. At 1.3 the word choice is more varied and surprising, and sometimes the sentences stop making sense. So a higher temperature gives more variety and less coherence." Your numbers above show the variety side; read the stories to judge the coherence side.
# </details>

# %% [markdown]
# ## Step 3. Translation with MarianMT
#
# The lesson's second pipeline: a translation model. We use the small `Helsinki-NLP/opus-mt-<src>-<tgt>` models, each about 0.3 GB. The function below loads a model the first time it is used and keeps it for later calls. Translating there and back is a **round trip**, a quick way to see what a model loses.

# %%
from transformers import MarianMTModel, MarianTokenizer
_marian = {}
def translate(text, src="en", tgt="fr"):
    name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    if name not in _marian:
        _marian[name] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).to(device))
    t, m = _marian[name]
    batch = t([text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = m.generate(**batch, max_new_tokens=100)
    return t.decode(out[0], skip_special_tokens=True)

fr = translate("Hello, how are you?")
print(fr)
assert "Bonjour" in fr

# %% [markdown]
# ### ✏️ Your turn: round trip in three languages
#
# Pick one English sentence with some nuance (an idiom works well, such as "It's raining cats and dogs."). Translate it to French, German and Spanish and back to English. Fill in `round_trip` for each language, then write down in `lost` what meaning or nuance was lost. (In the test version of this notebook only French is run, because each language pair downloads two models.)

# %%
sentence = "It's raining cats and dogs, so I'll take the bus."   # change it if you like
languages = ["fr"] if QUICK else ["fr", "de", "es"]
round_trip = {}   # YOUR CODE HERE: for each language l, round_trip[l] = translate(translate(sentence, "en", l), l, "en")
lost = ""         # YOUR ANSWER: what meaning or nuance was lost? Be specific.

# %%
# Check
if not round_trip:
    print("Fill in your answer above, then run this cell again.")
else:
    for l in languages:
        back = round_trip.get(l)
        if back is None:
            check(f"round trip for {l}", False, "missing: add it to round_trip")
            continue
        sim = difflib.SequenceMatcher(None, sentence.lower(), back.lower()).ratio()
        print(f"{l}: {back}   (similarity to the original: {sim:.2f})")
    check("lost-meaning note written", len(lost.split()) >= 8, "write at least one full sentence in `lost`")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# for l in languages:
#     round_trip[l] = translate(translate(sentence, "en", l), l, "en")
# ```
# Idioms such as "raining cats and dogs" are usually translated word by word or replaced, so the back-translation often loses the figure of speech or changes the word order. Compare each result with the original and write down the exact words that changed.
# </details>

# %% [markdown]
# ## Step 4. Image generation with Stable Diffusion (GPU only)
#
# The third pipeline creates a picture from a text prompt. We use `stable-diffusion-v1-5/stable-diffusion-v1-5` in half precision (fp16) with 30 steps, which fits a free T4 GPU. **This downloads about 4 GB** and needs a GPU; on a CPU the cell prints what it skips and moves on. Free the GPU memory when you are done, because the T4 is small.

# %%
# nb-test: skip
images = []
if device == "cuda":
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    gen = torch.Generator("cuda").manual_seed(SEED)
    img = pipe("A futuristic city with flying cars", num_inference_steps=30, guidance_scale=7.5, generator=gen).images[0]
    img.save("city.png")
    display(img)
else:
    print("Skipped: no GPU. Use Runtime → Change runtime type → T4 GPU to run the image step.")

# %% [markdown]
# ### ✏️ Your turn: three prompts
#
# Write three creative prompts. For each one, look at the image and note whether it matches your intention and one surprising element. Fill in `prompts` and `notes`, then run the first cell to generate, and the second cell to check.

# %%
# nb-test: skip
prompts = ["", "", ""]   # YOUR PROMPTS, e.g. "a watercolor of a jeepney in the rain"
notes = ["", "", ""]     # YOUR NOTES: matches intention? one surprising element?
if device == "cuda":
    images = []
    for i, p in enumerate(prompts):
        if not p.strip():
            print("Prompt", i + 1, "is empty: fill in `prompts`, then run this cell again.")
            continue
        g = torch.Generator("cuda").manual_seed(SEED + i)
        im = pipe(p, num_inference_steps=30, guidance_scale=7.5, generator=g).images[0]
        im.save(f"image_{i + 1}.png"); images.append(im)
        print(i + 1, p); display(im)

# %%
# Check
if device != "cuda":
    print("No GPU in this session, so this exercise is skipped. Switch to a T4 GPU runtime to do it.")
elif len(images) < 3:
    print("Fill in all three prompts above, generate the images, then run this cell again.")
else:
    check("three images generated", len(images) == 3, "fill in all three prompts")
    check("each image is 512 x 512", all(im.size == (512, 512) for im in images), "the default size is 512 x 512")
    check("notes written", all(len(n.split()) >= 5 for n in notes), "write a short note for every image")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# prompts = ["a watercolor of a jeepney in the rain", "a robot reading a book in a library", "a bowl of sinigang, studio photo"]
# notes = ["Matches; surprise: the jeepney has the wrong number of wheels.",
#          "Mostly matches; the book text is unreadable.",
#          "Matches the mood; the bowl looks too perfect."]
# ```
# Typical surprises are distorted hands, unreadable text and objects that blend together. Write what you actually saw, not this example.
# </details>

# %%
# Free the GPU when you are finished with images
if "pipe" in globals():
    del pipe
free_memory()
print("memory freed")

# %% [markdown]
# ## Step 5. Speech: text to speech and back
#
# The fourth pipeline works with audio. First text-to-speech with `facebook/mms-tts-eng` (a VITS model, about 0.14 GB), then speech-to-text with `openai/whisper-tiny` (a small Whisper model through `transformers`). In the lesson's edge-case figure, Whisper was the example that is "either": it generates text, but the text is a transcript of what was said.

# %%
from transformers import VitsModel, pipeline
from IPython.display import Audio
tts_tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
tts = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
sr = tts.config.sampling_rate

def speak(text):
    set_seed(SEED)    # VITS adds random variation to the voice
    inp = tts_tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        return tts(**inp).waveform[0].cpu().numpy()

wave = speak("Hello, this is a test of text to speech conversion.")
print("sampling rate:", sr, "| seconds of audio:", round(len(wave) / sr, 1))
Audio(wave, rate=sr)

# %%
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=0 if device == "cuda" else -1)

def transcribe(wave, sr):
    out = asr({"raw": wave.astype(np.float32), "sampling_rate": sr}, generate_kwargs={"language": "english", "task": "transcribe"})
    return out["text"].strip()

heard = transcribe(wave, sr)
print(heard)
assert len(wave) > sr, "the audio should be longer than one second"

# %% [markdown]
# ### ✏️ Your turn: the audio chain
#
# Generate text with GPT-2, speak it, transcribe it, and compare. To compare, write a function `wer(reference, hypothesis)`: the **word error rate**, which is the minimum number of word insertions, deletions and substitutions needed to turn the reference into the hypothesis, divided by the number of reference words. For example, `wer("a b c", "a x c")` is 1/3. Fill in the function first and run the Check; then run the chain cell below it.

# %%
def wer(reference, hypothesis):
    # YOUR CODE HERE: split both into words (lowercase, no punctuation), compute the edit distance, divide by len(reference words)
    return None

# %%
# Check
import re
def words(s): return re.findall(r"[a-z0-9']+", s.lower())
cases = [("a b c", "a x c", 1 / 3), ("hello world", "hello world", 0.0), ("one two", "one two three", 0.5)]
if wer("a b c", "a x c") is None:
    print("Fill in your answer above, then run this cell again.")
else:
    for ref, hyp, expected in cases:
        check(f"wer({ref!r}, {hyp!r}) = {expected:.2f}", abs(wer(ref, hyp) - expected) < 1e-6, "count edits with dynamic programming, divide by the reference length")

# %%
# The chain: GPT-2 text -> speech -> transcript
original = generate_text("The weather in Manila today is", 20, do_sample=True, temperature=0.7)
heard2 = transcribe(speak(original), sr)
print("original  :", original)
print("transcript:", heard2)
rate = wer(original, heard2)
if rate is None:
    print("Define `wer` above to see the word error rate.")
else:
    print("word error rate: %.2f" % rate)
print("Now list two reasons for the differences in a text cell (for example punctuation, rare words, numbers, synthetic pronunciation).")

# %% [markdown]
# <details><summary>Show a possible answer</summary>
#
# ```python
# def wer(reference, hypothesis):
#     r, h = words(reference), words(hypothesis)
#     d = list(range(len(h) + 1))
#     for i in range(1, len(r) + 1):
#         prev, d[0] = d[0], i
#         for j in range(1, len(h) + 1):
#             cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
#             prev, d[j] = d[j], cur
#     return d[len(h)] / len(r)
# ```
# The function keeps one row of the edit-distance table. Typical reasons for differences: punctuation and capital letters are not spoken, rare or cut-off words are mispronounced by the synthetic voice, and numbers or names can be written in another way by the transcriber. The final word of the GPT-2 text may also be cut in the middle because we stopped at a fixed number of tokens.
# </details>

# %% [markdown]
# ## Step 6. Clean up
#
# On a shared or small GPU you free memory when you are done. Delete the models, free the memory, and (in Colab) you may use Runtime → Restart session afterwards.

# %%
for name in ["lm", "tts", "asr"]:
    if name in globals():
        del globals()[name]
_marian.clear()
free_memory()
print("models released")

# %% [markdown]
# ## Wrap-up
#
# - You ran four generative pipelines: text (GPT-2), translation (MarianMT), images (Stable Diffusion) and speech (text-to-speech and Whisper).
# - You saw that temperature trades coherence for variety, that round trips and transcripts lose small details, and that a fluent output can still be wrong.
# - You measured a difference between two texts with a number (word error rate) instead of only reading them.
#
# **Reflection.** Double-click here and write short answers: (a) the most surprising result, (b) two real-world applications, (c) two ethical concerns and one mitigation for each. Use the five trustworthy-AI principles from the lesson: fairness, transparency, privacy, safety and accountability.
#
# Files in `/content` (such as `city.png`) vanish when the session ends, so download anything you want to keep. Submit your notebook the way your instructor asks.
