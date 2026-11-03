# Supervaise × NVIDIA — GenAI & LLM Learning Studio

A working web application built from the Claude Design file *"Supervaise × NVIDIA — GenAI & LLM Learning Studio"*.

- **frontend/** — React 18 + Vite single-page app. All 10 screens from the design, pixel-for-pixel.
- **backend/** — Node.js + Express API: accounts, sign-in sessions, per-user saved progress. It can also serve the built frontend, so the whole site can run as one process.

| Route | Screen | Who |
|---|---|---|
| `/` | Landing page (hero, tokeniser demo, programme, certifications, log in / sign up) | Public |
| `/intro` | Short landing variant | Public |
| `/dashboard` | Student · Overview | Signed in |
| `/course-map` | Student · Course Map | Signed in |
| `/lesson` | Student · Lesson Workspace (the built-in Week 7 lesson) | Signed in |
| `/lesson/:week` | Lesson Workspace for that week, e.g. `/lesson/1`. Reached from **Course Map → Open lesson** | Signed in |
| `/prompt-workshop` | Student · Prompt Engineering Workshop | Signed in |
| `/capstone` | Student · Capstone Project | Signed in |
| `/mock-exam` | Student · Mock Exam Prep | Signed in |
| `/certification` | Student · Achievements & Certification | Signed in |
| `/educator` | Educator · Faculty Console | Educators only |

## What is real, and what is sample content

**Real (backed by the server):**
- Sign up, log in, log out. Passwords are hashed with bcrypt; the session is an HTTP-only cookie holding a signed token that lasts 7 days.
- Route protection. Signed-out visitors are sent to the landing page, and students cannot open the Faculty Console.
- The signed-in user's name and initials in the sidebar.
- Light/dark theme saved to the user's profile.
- Saved progress per user: mock-exam attempt history, Prompt Workshop drafts and notes, Course Map selection, Lesson quiz position, and the Faculty Console's grades, reminders and lab-session plans.

**Sample content from the design (not yet data-driven):**
- Course content (the 20 weeks, lessons, exam questions, the prompt "model" simulator) is written into the page files, as in the design.
- The Faculty Console's learner roster and submissions are the design's demo data. It is labelled *DEMO DATA* on screen. Grades you enter are saved, but the roster is not built from registered users.
- Mock exams are scored in the browser.

## Adding weekly lessons

Pick a week in the **Course Map** and press **Open lesson**. It opens the **Lesson Workspace** for that week (`/lesson/1`, `/lesson/2` …). The lesson text comes from one Markdown file per week in `backend/content/`, named `week-01.md`, `week-02.md` … `week-20.md`. All weeks except Week 7 are included (see below). **To write or edit a lesson, read [docs/LESSON_FORMAT.md](docs/LESSON_FORMAT.md)** (the full format, every figure type with its JSON, the quiz and lab rules) and check it with `npm run validate:lessons`. [docs/WEEK_BRIEFS.md](docs/WEEK_BRIEFS.md) holds the plan for each week.

- Lessons use the same look as the Week 7 lesson (serif reading text, section rail, key-takeaway blocks, code boxes). Layout comes from a few Markdown conventions; see `week-01.md` for a full example:
  - `> source note` on the first line, then `# Week N — Title`, then `**Purpose:** one sentence` and `**Term:** … · **Time:** …` (these fill the header and its "Lesson details").
  - `==important idea==` gives the highlighter marker. `[[term|short definition]]` makes a dotted-underline term that opens its definition when selected.
  - `## By the end, you can…` becomes the outcomes checklist, `## Key takeaways` becomes takeaway blocks, and a `##` heading starting with `Lab` is styled as the lab. Add `{rail=Short name}` to a `##` heading to put it in the sticky section rail.
  - Start a paragraph with `{lede}` for the larger main-idea line of a section, or `{sub}` for the small line under a heading.
  - Python code blocks get syntax colours and a Copy button. Checklist ticks are remembered in the browser.
- A lesson can embed **interactive visuals and a scored quiz** with fenced blocks containing JSON. `week-01.md` has working examples of each:
  - ```` ```viz ```` with a `"type"`: `hook` (the animated opener under the header; scene kinds `text`, `image`, `speech`, `tokens`, `bars`, `denoise`, `conv`, `loss`), `flow` (input → model → output tabs), `sorter` (sort examples into groups with feedback), `timeline`, `temperature`, `pipeline` (step through a process), `curve` (activation and gradient plots), `descent` (gradient descent with a learning-rate slider), `neuron` (a decision line on 2-D data), `conv` (convolution and transpose convolution by hand), `calc` (parameter, LoRA, memory and scaling calculators), `bpe` (byte-pair encoding learned live), `ngram`, `scatter` (embedding map with analogies), `heatmap` (attention maps, masks), `posenc`, `patch` (image to ViT tokens), `diffusion` and `retrieval` (TF-IDF search and the RAG prompt). Every figure computes its numbers from its JSON.
  - ```` ```quiz ```` with `id`, `passMark` and `questions` (each: `q`, `options`, `answer` as the option's index starting at 0, and `why`). Each student's best score and attempt count are saved to their account.
  - A block with a JSON mistake shows a red notice in place of the visual, so typos are easy to spot.
- Instructor-only material (prep checklists, answer keys) should not go in `week-NN.md`. The Week 1 and Week 7 instructor plans are kept in `backend/content/instructor/` and are not shown in the app.
- Lab sections link to the [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/) (Phillip Lippe, MIT licence) as optional "go further" notebooks for Weeks 2, 3, 4, 7, 8, 11, 12 and 17. Those notebooks download data and checkpoints, so students run them with internet access or the instructor pre-downloads them for the Spark.
- Files are read on every request, so saving a file and refreshing the page is enough. No restart or rebuild.
- A week with no file shows "Week N lesson not published yet". Week 7 keeps the lesson that came with the design (it has no `week-07.md`). Opening **Lesson Workspace** from the sidebar (no week) also shows that built-in Week 7 lesson.
- Everything in a `week-NN.md` file is shown to every signed-in user.

## Lesson notebooks (Google Colab)

Every lesson has a hands-on Jupyter notebook, `notebooks/week-NN.ipynb`. In the lesson page the lab (or study-plan) section shows **Open in Colab** and **Download .ipynb** buttons. Students click **Open in Colab**: the notebook opens in a new tab and runs on Google's servers, with nothing to install, and they choose **File → Save a copy in Drive** to keep their own copy. They need a Google account. The notebooks are written for Colab, so they may `pip install` and download small public models; they do not run on the offline DGX Spark.

- **How the button works:** the server builds the link `https://colab.research.google.com/github/<COLAB_REPO>/blob/<COLAB_BRANCH>/notebooks/week-NN.ipynb`. The default is `Supervaise-Inc/AI-Academy` on `main`. **Colab reads the notebooks from GitHub, so they must be pushed to that branch of a public repository** (a private repository asks each student to sign in to GitHub). Set `COLAB_REPO` and `COLAB_BRANCH` in `backend/.env` to point elsewhere, or set `COLAB_REPO=` (empty) to show only the download button, which students can upload to Colab by hand.
- **Write or edit a notebook:** edit `notebooks/src/week-NN.py` (a plain Python file in "percent format"), then run `npm run build:notebooks` to regenerate the `.ipynb` files and `npm run test:notebooks` to check that every code cell compiles. The format and rules are in [docs/NOTEBOOK_FORMAT.md](docs/NOTEBOOK_FORMAT.md). Commit the generated `.ipynb` files, because Colab opens those.
- **Exam weeks (5, 10, 15, 20) and Week 19** have practice notebooks with computed exercises instead of a lab.
- **API:** `GET /api/notebooks/:slug` returns the Colab link and download path, and `GET /api/notebooks/:slug/download` returns the file.

## Repository layout

```
frontend/        React app (src/pages = the 10 screens, src/dc = runtime helpers)
backend/         Express API (src/app.js routes, src/db.js storage, scripts/create-user.js)
deploy/          Example nginx configs
design-source/   The original Claude Design files (.dc.html) the pages were generated from
tools/           One-time converter: design-source -> frontend/src/pages
Dockerfile       Builds frontend + backend into one image
docker-compose.yml
```

## Run it locally (development)

Requirements: **Node.js 20 or newer** (18.18+ works) and npm.

```bash
npm install                 # installs the root helper (concurrently)
npm run install:all         # installs frontend and backend dependencies
npm run dev                 # API on :4000, website on :5173
```

Open **http://localhost:5173**. In development the backend creates two demo accounts (password `academy-demo`):

| Email | Role |
|---|---|
| `maya@supervaise.io` | Student |
| `faculty@supervaise.io` | Educator (opens the Faculty Console) |

Demo accounts are never created when `NODE_ENV=production`.

Other commands: `npm test` (API tests), `npm run build` (production frontend into `frontend/dist`).

## Configuration (backend/.env)

Copy `backend/.env.example` to `backend/.env` and edit it. The file is optional for local development (the defaults work); on a server you must set `NODE_ENV=production` and `JWT_SECRET`.

| Variable | Required | Meaning |
|---|---|---|
| `NODE_ENV` | yes, in production | Set to `production`. |
| `JWT_SECRET` | **yes, in production** | Random string of 32+ characters. The server refuses to start without it. Generate: `node -e "console.log(require('crypto').randomBytes(48).toString('hex'))"` |
| `PORT` | no | Default `4000`. |
| `EDUCATOR_EMAILS` | no | Comma-separated emails. Anyone who signs up with one of these becomes an educator. |
| `CONTENT_DIR` | no | Folder of weekly lessons. Default `backend/content`. |
| `NOTEBOOKS_DIR` | no | Folder of the Colab notebooks. Default `../notebooks`. |
| `COLAB_REPO` / `COLAB_BRANCH` | no | GitHub repository and branch the "Open in Colab" buttons use. Default `Supervaise-Inc/AI-Academy` and `main`. Empty `COLAB_REPO` hides the button. |
| `DATA_DIR` | no | Folder for `db.json` (users and saved progress). Default `backend/data`. Put it on persistent storage. |
| `FRONTEND_DIST` | no | Folder of the built frontend to serve. Default `../frontend/dist`. |
| `TRUST_PROXY` | no | `1` when behind nginx, Render, Railway, Fly, etc. |
| `CORS_ORIGIN` | split hosting only | Public origin of the website, e.g. `https://academy.example.com`. |
| `COOKIE_SAMESITE` / `COOKIE_SECURE` | split hosting only | See *Option C*. Defaults are right for single-origin hosting over HTTPS. |

The frontend has one optional build-time variable, `VITE_API_URL` (`frontend/.env.example`). Leave it empty unless the API is on a different domain.

## Hosting

Pick one option. **Option A or B (one origin) is recommended** because there are no CORS or cross-site cookie problems.

You need HTTPS in production, because sign-in cookies are marked `Secure`. The steps below use Let's Encrypt.

### Option A — one server (VPS such as DigitalOcean, Hetzner, AWS Lightsail)

The Node process serves the API and the built website together. Ubuntu 22.04/24.04 shown.

```bash
# 1. Node 20, nginx, pm2
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs nginx git
sudo npm install -g pm2

# 2. Get the code onto the server (git clone, or scp/rsync this folder), then:
cd /opt/ai-academy
npm run install:all
npm run build                              # creates frontend/dist

# 3. Configure
cp backend/.env.example backend/.env
nano backend/.env                          # set NODE_ENV=production, JWT_SECRET=..., EDUCATOR_EMAILS=you@school.edu

# 4. Start it and keep it running across reboots
cd backend
pm2 start src/server.js --name ai-academy
pm2 save && pm2 startup                    # run the command it prints

# 5. HTTPS in front (replace academy.example.com; point its DNS A record at the server first)
sudo cp ../deploy/nginx.conf /etc/nginx/sites-available/academy
sudo nano /etc/nginx/sites-available/academy        # set server_name
sudo ln -s /etc/nginx/sites-available/academy /etc/nginx/sites-enabled/academy
sudo nginx -t && sudo systemctl reload nginx
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d academy.example.com
```

Visit `https://academy.example.com`. Then create your faculty account (see *Creating accounts*).

**Updating:** `git pull && npm run install:all && npm run build && pm2 restart ai-academy`.

### Option B — Docker

```bash
echo "JWT_SECRET=$(node -e "console.log(require('crypto').randomBytes(48).toString('hex'))")" > .env
echo "EDUCATOR_EMAILS=you@school.edu" >> .env
docker compose up -d --build
```

The site is on port 4000. Users and progress live in the `academy-data` volume, so they survive rebuilds. Put an HTTPS reverse proxy in front (nginx + certbot as above, Caddy, Traefik, or your cloud's load balancer). The same image runs on Render, Railway, Fly.io, Google Cloud Run and similar. Mount a persistent disk at `/data` and set `JWT_SECRET`. On plain-http test boxes only, set `COOKIE_SECURE=false`.

> The Dockerfile was written but **not build-tested** in the environment this project was created in (no Docker access). Run `docker compose up --build` once to confirm before relying on it.

### Option C — frontend and backend on different hosts

Use this for a static host (Netlify, Vercel, Cloudflare Pages, S3 + CloudFront) plus an API host (Render, Railway, Fly.io, a VPS).

**C1 (recommended): keep `/api` on the website's own domain** by having the static host proxy it, so nothing about cookies changes.

1. Deploy the backend anywhere (Option A without the frontend, or Docker). Set `NODE_ENV=production`, `JWT_SECRET`, `TRUST_PROXY=1`.
2. Build the frontend with `VITE_API_URL` empty: `cd frontend && npm ci && npm run build`. Publish `frontend/dist`.
3. Add a proxy rule so `/api/*` goes to your API host:
   - **Netlify / Cloudflare Pages:** add to `frontend/public/_redirects`, above the last line: `/api/*  https://YOUR-API-HOST/api/:splat  200`
   - **Vercel:** in `frontend/vercel.json`, add `{ "source": "/api/(.*)", "destination": "https://YOUR-API-HOST/api/$1" }` before the catch-all rewrite.
   - **Your own nginx:** use `deploy/nginx-static-frontend.conf`.

**C2: truly cross-origin** (website on `academy.example.com`, API on `api.example.com`):

1. Backend env: `CORS_ORIGIN=https://academy.example.com`, `COOKIE_SAMESITE=none`, `COOKIE_SECURE=true`, plus the usual `NODE_ENV`, `JWT_SECRET`, `TRUST_PROXY=1`.
2. Build the frontend with `VITE_API_URL=https://api.example.com` (set it in `frontend/.env` or the host's build settings).
3. Publish `frontend/dist`. The single-page-app fallback (`_redirects` / `vercel.json`) is already included.

Some browsers (Safari, Brave, Chrome with third-party cookies blocked) refuse cross-site cookies, so sign-in can fail for those visitors. Prefer C1 or a shared parent domain.

## Creating accounts

Students sign up themselves from the landing page. For faculty:

```bash
cd backend
npm run create-user -- --email prof@school.edu --name "Dr. Jose" --password 'a-long-password' --role educator
```

or list their email in `EDUCATOR_EMAILS` before they sign up. (Use `--role student` for a student.)

## Backups and scaling

All data is one file, `DATA_DIR/db.json`. Copy it to back up; writes are atomic, so copying while the server runs is safe. This JSON-file store suits a single server and a cohort of a few hundred learners. To run several instances or a much larger cohort, replace `backend/src/db.js` with a Postgres or SQLite implementation of the same small interface.

## API reference

All routes are under `/api`. Bodies are JSON.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/auth/register` | `{ name, email, password }` (8+ chars). Signs the user in. Rate limited. |
| POST | `/auth/login` | `{ email, password }`. Rate limited. |
| POST | `/auth/logout` | Clears the session |
| GET | `/auth/me` | `{ user }` or `{ user: null }` |
| PUT | `/me/theme` | `{ theme: "light" \| "dark" }` |
| GET | `/content` | List of weekly lessons (title, term) |
| GET | `/content/:slug` | One lesson, e.g. `week-01`, as Markdown (404 if not written yet) |
| GET | `/notebooks/:slug` | Colab link and download path for a week's notebook (404 if none) |
| GET | `/notebooks/:slug/download` | The `.ipynb` file |
| GET / PUT | `/state/:key` | The signed-in user's saved state for a page. PUT body `{ data: {…} }`. |

## Notes on how the pages were built

The design file is a set of `.dc.html` artboards (kept in `design-source/`). `tools/convert.mjs` turned each one into a React page in `frontend/src/pages/` (a `.jsx` with the markup and the design's own logic, plus a `.css` with its styles). `tools/patches.mjs` adds the real sign-in forms, user name, sign-out and the persistence list.

Treat `frontend/src/pages/*` as the source from now on and edit it directly. Re-running `cd tools && npm install && node convert.mjs` regenerates and overwrites those files, so only do that if you are changing the design and are happy to redo any hand edits.

Fonts load from Google Fonts (`index.html`). To self-host them for an offline or restricted network, download the four families (Plus Jakarta Sans, Rubik, JetBrains Mono, Source Serif 4) into `frontend/public/fonts`, then replace the Google link in `frontend/index.html`. Update the `font-src` / `style-src` entries in `backend/src/app.js` to match.

## Troubleshooting

- **Server exits with "JWT_SECRET must be set …"**: either you are developing locally, so set `NODE_ENV=development` in `backend/.env` (or delete the file); or you are on a server, so set a 32+ character `JWT_SECRET`.
- **Refreshing a page like `/dashboard` gives 404**: the host lacks the single-page-app fallback. The Express server and the provided nginx/Netlify/Vercel configs include it; other hosts need a "rewrite everything to `/index.html`" rule.
- **Log in succeeds but you land back on the landing page** (production): the site isn't on HTTPS, so the `Secure` cookie is dropped. Enable HTTPS, or for a plain-http test box set `COOKIE_SECURE=false`.
- **Log in works locally but not when hosted behind a proxy**: set `TRUST_PROXY=1` and pass `X-Forwarded-Proto` (the nginx example does).
- **"Cannot reach the server"** in the login box: the browser cannot reach `/api`. Check the proxy rule, or `VITE_API_URL` and `CORS_ORIGIN` for split hosting.
- **Progress doesn't persist after a redeploy**: `DATA_DIR` is not on persistent storage.
