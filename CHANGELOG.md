# Changelog

All notable changes to this project are recorded here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Add new entries under **Unreleased** as you work, then rename that heading to a date when you commit or release.

## [Unreleased]

### Added
- **Four new lesson figures.** `nested` (AI ⊃ machine learning ⊃ deep learning ⊃ generative AI), `sampling` (softmax with temperature plus top-k and top-p, with the formula's live numbers and a draw-100-times tally), `chain` (the chain rule: a sentence's probability as the product of next-word probabilities, with perplexity) and `logbars` (sizes on a log scale). Documented in `docs/LESSON_FORMAT.md` and checked by the validator, which now allows quizzes of up to 15 questions.
- **Colab notebooks for every lesson.** `notebooks/week-01.ipynb` … `week-20.ipynb` (20 notebooks, including Week 7). Each lesson's lab (or study-plan) section shows **Open in Colab** and **Download .ipynb** buttons; students run the notebook on Google's servers and save their own copy to Drive. Weeks 1–4, 6–9, 11–14 and 16–18 are lab notebooks; Weeks 5, 10, 15, 19 and 20 are practice notebooks with computed exercises. Every notebook has Your turn exercises, Check cells and worked answers. The built-in Week 7 lab's "Open Laboratory 3 notebook" button now opens its notebook.
- **Notebook toolchain.** Notebooks are written as percent-format Python in `notebooks/src/` and built with `npm run build:notebooks` (`tools/build-notebooks.mjs`, with structure checks). `npm run test:notebooks` compiles every cell; `python3 tools/test-notebooks.py --run week-NN` executes a notebook. Format and rules: `docs/NOTEBOOK_FORMAT.md`.
- **API:** `GET /api/notebooks/:slug` (Colab link and download path) and `GET /api/notebooks/:slug/download`; the lesson response also carries `notebook`. New settings `NOTEBOOKS_DIR`, `COLAB_REPO` (default `Supervaise-Inc/AI-Academy`) and `COLAB_BRANCH` (default `main`). One more backend test.
- **Week-driven Overview.** `/dashboard/:week` shows the selected week's term, learning activities, sessions and "next action". `/dashboard` still shows the current week (7).
- **`frontend/src/data/courseWeeks.js`**: one data file holding the term, activities, sessions, dates and hero for all 20 weeks. Weeks other than 7 use generated placeholder sessions (08:00 lecture, 10:30 lab, 14:00 consultation) and due dates (Saturday, 11:59 PM PHT); replace them with the real timetable.
- **Week overview** button on the Course Map, next to **Open lesson**.
- **`attention` lesson figure**: query, keys, values, dot products, softmax weights α and the weighted-sum output, computed live. Choose the query word and toggle √d scaling.
- **`equation` lesson figure**: a formula whose symbols can be selected for a plain-language meaning, with an optional worked example.
- Both figures are documented in `docs/LESSON_FORMAT.md` and checked by `tools/validate-lessons.mjs`.
- Week 8 lesson: an equation figure (scaled dot-product attention, worked example for "apple") and an attention diagram for "I am eating an apple".
- **Lab steps are easier to scan.** In lesson plans, numbered lab steps show a large green number badge (the numbering still restarts correctly after each code block) and a bold step title. Code boxes and exercises sit indented under their step.
- **Exercise callouts.** A paragraph that starts with `*Exercise N:*` in a lesson file becomes an amber callout with an "Exercise N" badge, so it stands out when skimming.

### Changed
- **Week 4 rewritten around convolutional neural networks.** About 1,450 lines of lesson text, 38 figures of 14 types (ten formula figures: dense parameters, the convolution sum, channels, output size, pooled size, receptive field, kernel gradient, input gradient, pooling backward, gradient check), 15 quiz questions (9 calculations) and six rail sections (Images, Convolution, Output size, Stacking, Backward, Training, plus Laboratory).
- **Four new figure types:** `convgrad` (the backward pass of a convolution: select an input or kernel cell and see the terms that make its gradient), `filters` (edge, blur and sharpen kernels or your own on a small picture), `pool` (max and average pooling) and `cnnstack` (shapes, parameters and receptive field through a LeNet-style network, with editable layers). The validator recomputes `convgrad` and `cnnstack` checks.
- **New Week 4 lab, "Imagine a CNN, then build one".** Seven parts: pictures as numbers and why dense layers struggle (including a shift test and a pixel-shuffle test), convolution by hand, filters and pooling, the backward pass of a convolution checked by finite differences and autograd, building and training a CNN against an MLP, looking inside the network, and designing your own. Uses generated shape pictures, so nothing is downloaded. Rubric and deliverables included.
- **Week 4 notebook rebuilt** (141 code cells), run end to end on CPU in about a minute.
- **Week 3 rewritten around deep learning and backpropagation.** About 1,300 lines of lesson text, 30 figures of 14 types (ten formula figures: layered forward pass, chain rule, the four backprop equations, powers of a factor, clipping, initialization, normalization, residual connection), 15 quiz questions (8 calculations) and seven rail sections (Why depth, Forward, Chain rule, Backward, Vanishing, Exploding, Remedies, plus Laboratory).
- **Three new figure types:** `backprop` (forward and backward pass on a small network with δ, dL/dw and gradient-descent updates), `gradflow` (gradient size per layer for sigmoid, tanh and ReLU at any depth and weight scale) and `fold` (depth doubles the straight pieces). The validator recomputes `backprop` and `gradflow` checks.
- **New Week 3 lab, "Imagine a deep network, then trace its gradients".** Seven parts: why depth, forward pass, chain rule and a mini autograd engine from scratch, backpropagation in NumPy checked against PyTorch and finite differences, watching gradients vanish and explode, repairing deep networks, and designing and diagnosing your own. Rubric and deliverables included. The lab time table sums to 200 minutes.
- **Week 3 notebook rebuilt** (140 code cells), run end to end on CPU in about 2.5 minutes with no downloads.
- **Week 2 rewritten around supervised learning and the artificial neuron.** About 1,150 lines of lesson text, 24 figures of 15 types (every formula is a figure: MSE, −ln p, neuron, layer, softmax, collapse of stacked linear layers, update rule), 15 quiz questions (7 calculations) and seven rail sections (Supervised, Loss, Neuron, Layers, Hidden layers, Training, Generalization, plus Laboratory).
- **Five new figure types for Week 2:** `network` (click any circle to see z and a, edit weights, switch the activation), `fit` (polynomial degree slider with training and test error), `trainer` (a small network learning XOR, circles, moons or a line in the browser), `mlp` (parameter counter) and `calc` kind `split` (train/validation/test). The validator recomputes the `network` and `trainer` checks.
- **New Week 2 lab, "Imagine a neural network, then build one".** Seven parts (about 3 hours): be the model on a labelled table, loss, one neuron by hand, layers and a hand-built XOR network, gradient descent you can watch (including NumPy backpropagation), PyTorch with overfitting, and designing your own classifier. Rubric and deliverables included.
- **Week 2 notebook rebuilt** (109 code cells). Run end to end on CPU in about 36 seconds; no downloads needed.
- **Week 1 rewritten as a full introduction to generative AI.** About 1,300 lines of lesson text with 20 figures of 12 types and every formula shown as a figure (P(y|x), P(x|prompt), the −log p loss, the chain rule, softmax with temperature, forward diffusion), 15 quiz questions, and eight rail sections (Concept, Learning, Writing, Beyond text, History, Risks, Laboratory, Check).
- **New Week 1 lab, "Imagine it, then build it".** Seven parts (about 3 hours): be the model and guess the next word, build a tiny generator from scratch, the dials of imagination (temperature, top-k, top-p), art from noise and diffusion, sound and languages, designing an AI product and testing where it fails, and a reflection. Each part starts with an "Imagine first" prediction and has detailed exercises, a 100-point rubric and deliverables.
- **Week 1 notebook rebuilt to match** (93 code cells): Your turn exercises, Check cells and worked answers for every exercise. Run end to end on CPU (90 cells; the 3 Stable Diffusion cells are GPU-only and were not run).
- Dockerfile copies `notebooks/` into the image (`NOTEBOOKS_DIR=/app/notebooks`).
- **Lesson text width matches Week 7.** In the Markdown lessons the reading column is now 60rem (the page width on narrower screens), the same as the built-in Week 7 lesson, so text runs about as wide as the figures. It was 41rem, which left text noticeably narrower than the figures. Lab steps and exercise callouts follow the same width.
- **Larger type in the Markdown lessons (Weeks 1–6, 8–20).** Body text 17 → 18.5 px, lab steps and lists 16 → 17.5 px, captions and detail text 15 → 16.5 px, subheadings 18.5 → 20.5 px, code 12.5 → 14 px, and 1–2 px more on the interactive figures, tables and quiz. The text column is a little wider (37 → 41 rem). The built-in Week 7 lesson keeps its original sizes.
- Overview page (`Main.jsx`) no longer hardcodes Week 7: header, hero, activities, sessions, schedule pop-up and lesson card all read from the week data.
- Lesson page header shows "Week N of 20" and the term for `/lesson/N`.

### Known gaps
- **Colab needs the notebooks on GitHub.** The buttons open `Supervaise-Inc/AI-Academy` on `main`; the notebooks only work there once they are pushed (and the repository is public). Until then use the download button.
- **Not run in a real Colab.** All notebooks were executed on CPU with PyTorch here. Not exercised: the Stable Diffusion cells (Weeks 1 and 13), the GPU benchmark cells (Week 18), the German and Spanish translation steps (Week 1), full-length `gpt2` fine-tuning (Week 14), and the `pip install` lines in a fresh Colab runtime.
- The lesson lab sections still describe the offline DGX Spark workflow (portal and JupyterLab); the notebooks are the Colab alternative.
- The Upcoming requirements and Alerts pop-ups on the Overview are still Week 7 content.
- The "You are here" roadmap progress is fixed at Week 7 (`CURRENT_WEEK`).
- The built-in Week 7 lesson (`/lesson`) has no attention diagram yet.

## [0.1.0] - 2026-10-27

### Added
- Initial commit: AI Academy app (React + Vite frontend, Express backend, weekly lesson Markdown content).
