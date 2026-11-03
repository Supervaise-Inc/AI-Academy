# Changelog

All notable changes to this project are recorded here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Add new entries under **Unreleased** as you work, then rename that heading to a date when you commit or release.

## [Unreleased]

### Added
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
