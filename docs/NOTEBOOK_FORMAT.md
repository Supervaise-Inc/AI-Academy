# Writing a lesson notebook (`notebooks/src/week-NN.py` → `notebooks/week-NN.ipynb`)

Every lesson has a hands-on Jupyter notebook that students open in **Google Colab** from the lesson page ("Open in Colab"). Students click the button, Colab opens the notebook from GitHub on Google's servers, and they **File → Save a copy in Drive** to keep their own copy. Nothing is installed on the student's computer and nothing depends on the DGX Spark.

You write the notebook as a plain Python file in "percent format" (easy to review and diff); a script turns it into `.ipynb`.

```bash
node tools/build-notebooks.mjs week-NN                 # must print ✓ (structure checks)
python3 tools/test-notebooks.py --syntax week-NN       # every code cell compiles
<nbvenv>/bin/python tools/test-notebooks.py --run week-NN   # actually runs it (see §6)
```

Read first: `backend/content/week-NN.md` (the lesson; the notebook is its hands-on part), `COURSE_OUTLINE.md`, `docs/LESSON_FORMAT.md` §5–§8 (lab rules and accuracy rules apply here too). The existing lab section of the lesson is your starting point, but the notebook is for Colab, not the offline Spark, so you may `pip install`, download small public models and datasets, and use the internet.

## 1. Source format

```python
# accelerator: GPU            ← optional, FIRST line only, when a GPU makes it much nicer (omit for CPU-friendly notebooks)

# %% [markdown]
# # Week 4 — Convolutional Neural Networks: lab notebook
#
# Built for **Google Colab**. … (markdown: every line starts with "# ")

# %%
#!pip install -q peft            ← shell command: "#!" at the start of a line becomes "!pip install -q peft"
import numpy as np              ← ordinary code; `#%matplotlib inline` becomes the line magic `%matplotlib inline`
```

- `# %% [markdown]` starts a markdown cell: every following line is `# text` (or `#` for a blank line). Colab renders GitHub-flavoured Markdown plus `<details>`.
- `# %%` starts a code cell.
- Magics stay comments in the `.py`, so the same file also runs as a normal script.
- Nothing may appear before the first `# %%` except the optional `# accelerator:` line.

## 2. Required shape of every notebook

1. **Title cell** (markdown): `# Week N — <lesson title>: lab notebook`, then a short paragraph that says it is **built for Google Colab** and tells the student to open **File → Save a copy in Drive** first; the runtime (CPU is enough, or "GPU recommended: Runtime → Change runtime type → T4 GPU"); an estimated time; a bullet list "What you will do" matching the steps; one line "How this notebook works": run cells top to bottom with ▶; **Your turn** cells are for you; a **Check** cell tells you if you are right; open the **Show a possible answer** block only after you tried.
2. **Setup cell**: installs (`#!pip install -q …` only for packages Colab lacks, such as `peft`, `diffusers`, `sentence-transformers`, `accelerate`; Colab already has torch, numpy, scikit-learn, matplotlib, pandas and transformers), imports, `SEED`, `device = "cuda" if torch.cuda.is_available() else "cpu"`, a print of the device, `QUICK = bool(os.environ.get("NB_TEST"))` (used to shorten long loops when testing), and a tiny helper:
   ```python
   def check(name, ok, hint=""):
       print(("✓ " if ok else "✗ ") + name + ("" if ok else "  → " + hint))
   ```
3. **Steps**: one block per lab step in the lesson, each with
   - a markdown cell: what we do and why (2–5 sentences, plain language, a mention of the lesson section by name (no URLs needed));
   - a code cell with complete, working, commented code (≤ 30 lines per cell; prefer several small cells);
   - a **Your turn** exercise: markdown cell "### ✏️ Your turn: …" with the exercise from the lesson, then a code cell with a starter (e.g. `answer = None  # YOUR CODE HERE`) that **does not crash when left unfinished**, then a **Check** code cell that uses `check(...)` and handles `answer is None` ("Fill in your answer above, then run this cell again");
   - a markdown cell with `<details><summary>Show a possible answer</summary> … a fenced ```python solution plus one sentence of explanation … </details>`.
4. At least one `assert` somewhere (for example in the Check helper's fixed cases, or a "Sanity check: the loss went down" cell) so a broken environment is noticed.
5. **Wrap-up cell** (last cell, markdown): 3 bullets of what the student learned, a short reflection prompt ("Double-click to write two sentences: …"), a reminder that `/content` files vanish when the session ends (download anything to keep), "submit it the way your instructor asks" (never invent a portal URL), and, where the lesson lists one, the UvA "Go further" link with credit.

Aim for 8–16 code cells and 3 hours of work in total for a lab week. Runtime on a free Colab should be under 15 minutes of compute.

## 3. Colab rules

- Must run **top to bottom in a fresh Colab runtime without editing**, on CPU (and on a T4 GPU if you set `# accelerator: GPU`). Use `device` everywhere; never hard-code `"cuda"` unless the cell is guarded by `if device == "cuda":` with a printed explanation of what was skipped.
- Set seeds. Keep downloads small and say how big they are before a download cell.
- Only **public, ungated** Hugging Face models and datasets, no API keys. Verified IDs (all exist, ungated): `gpt2` (= `openai-community/gpt2`, 0.5 GB), `gpt2-medium`, `distilgpt2`, `bert-base-uncased`, `t5-small`, `openai/clip-vit-base-patch32`, `google/vit-base-patch16-224`, `sentence-transformers/all-MiniLM-L6-v2`, `Helsinki-NLP/opus-mt-en-fr` (and other `opus-mt` pairs), `facebook/mms-tts-eng`, `Qwen/Qwen2.5-0.5B-Instruct` (use this small instruct model wherever the lesson mentions the gated Llama-3.1-8B-Instruct, which needs approval and a token), and `stable-diffusion-v1-5/stable-diffusion-v1-5` (the old `runwayml/…` ID no longer exists; T4 GPU only, fp16, about 4 GB download: guard with `device == "cuda"`).
- Datasets that need no account: `sklearn.datasets` (`load_digits`, `load_iris`, `make_moons`, `make_blobs`), `torchvision.datasets` (FashionMNIST, CIFAR-10 download automatically; only if a lesson step really needs them), and synthetic data you generate.
- Plots: `matplotlib` inline (`plt.show()`). Keep figures small.
- Do not use `input()`, widgets that need extra installs, Google Drive mounting, `google.colab` APIs, or secrets.
- Never put real student data, passwords or tokens in a notebook.

## 4. What good looks like

- The code cells are the lesson's lab code, improved: complete, correct, with comments saying what each line is for. The exercises are the lesson's exercises, each with a runnable starter, a Check, and a worked answer.
- Every number the notebook prints or asserts has been computed (run it).
- Explanations stay faithful to the lesson text and its definitions; no new claims you cannot back up. Follow the accuracy rules in `docs/LESSON_FORMAT.md` §8.
- Friendly tone, second person, short sentences. One idea per markdown cell. No emoji except ✏️ on "Your turn" headings and ✓/✗ in Check output.

## 5. Exam weeks (5, 10, 15, 20) and Week 19

There is no lab in those weeks, so the notebook is a **practice notebook**: retrieval practice with *computed* answers. Each section is one covered week or theme: a short reminder cell, then 2–4 small exercises the student does in code (compute a convolution output size, an attention row, a BPE merge, a positional-encoding value, a LoRA parameter count, an ᾱ_t, a memory estimate, …), a Check against the true value, and a worked answer in `<details>`. For Week 19 add a "capstone build checklist" section with working starter code for one pipeline stage from COURSE_OUTLINE.md (for example a skeleton that classifies an image with CLIP zero-shot and returns a structured result) and for Week 20 a "Demo Day" section that helps the student time and rehearse a demo (a small timing helper). Never state exam dates, formats or weights.

## 6. Testing

- Always run `node tools/build-notebooks.mjs week-NN` and `python3 tools/test-notebooks.py --syntax week-NN`.
- A shared Python environment with PyTorch (CPU), scikit-learn, matplotlib, scipy, transformers, sentence-transformers, peft, diffusers and accelerate is at **`/tmp/claude-1001/-home-jeanettepao-Documents-AI-Academy/0c8aa091-6193-4de5-aae9-8ec19988357d/scratchpad/nbvenv`** (it may still be installing when you start; wait until `…/scratchpad/nbvenv-install.log` ends with `FINISHED`). Use `<that>/bin/python tools/test-notebooks.py --run week-NN`. `--run` skips `!`/`%` lines and executes the cells in order; set `QUICK` loops short via `NB_TEST` (set automatically).
- **The disk is almost full (≈ 7 GB free). Be careful.** Before running anything that downloads from Hugging Face, `export HF_HOME=<scratchpad>/hfcache-<your week>` and delete that folder when you finish. Only download models of about 600 MB or less (gpt2, distilgpt2, bert-base-uncased, MiniLM, CLIP-B/32, opus-mt, Qwen2.5-0.5B ≈ 1 GB is the upper limit). **Never download** Stable Diffusion, Llama, Whisper-large, or any multi-GB model, and do not download datasets larger than 100 MB (skip CIFAR/FashionMNIST in tests). For any cell you cannot run for these reasons, make its first line `# nb-test: skip` (it is then not executed by `--run`) and say so in your report.
- If the network is not reachable, say which cells you could not test.
- Fix every failure. A notebook that raises in Colab is the worst outcome.

## 7. Where notebooks plug into the app

Nothing for you to do: if `notebooks/week-NN.ipynb` exists, the lesson page shows "Open in Colab" and "Download .ipynb" buttons in its lab (or study-plan) section. Do not edit the lesson Markdown, `frontend/` or `backend/`.
