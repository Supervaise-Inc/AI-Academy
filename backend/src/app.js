import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import rateLimit from 'express-rate-limit';

const COOKIE = 'aia_session';
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
const STATE_KEY = /^[a-z0-9-]{1,40}$/;
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function createApp({ db, config }) {
  const { contentDir, notebooksDir, colabRepo, colabBranch, jwtSecret, isProd, corsOrigins, educatorEmails, frontendDist, sameSite, secure, trustProxy } = config;
  const app = express();
  if (trustProxy) app.set('trust proxy', 1);
  app.disable('x-powered-by');

  // The pages load Google Fonts and inline styles; everything else is same-origin.
  app.use(
    helmet({
      contentSecurityPolicy: {
        useDefaults: true,
        directives: {
          'style-src': ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
          'font-src': ["'self'", 'https://fonts.gstatic.com', 'data:'],
          'img-src': ["'self'", 'data:'],
          'connect-src': ["'self'", ...corsOrigins],
          // Only force https upgrades when running behind HTTPS; on a plain-http LAN box it would break every asset.
          'upgrade-insecure-requests': secure ? [] : null,
        },
      },
      crossOriginEmbedderPolicy: false,
    })
  );
  if (corsOrigins.length) app.use(cors({ origin: corsOrigins, credentials: true }));
  app.use(express.json({ limit: '200kb' }));
  app.use(cookieParser());

  const cookieOpts = { httpOnly: true, sameSite, secure, maxAge: WEEK_MS, path: '/' };
  const issue = (res, user) => res.cookie(COOKIE, jwt.sign({ sub: user.id }, jwtSecret, { expiresIn: '7d' }), cookieOpts);

  const requireAuth = (req, res, next) => {
    try {
      const { sub } = jwt.verify(req.cookies[COOKIE] || '', jwtSecret);
      const user = db.findUserById(sub);
      if (!user) throw new Error('unknown user');
      req.user = user;
      next();
    } catch (e) {
      res.status(401).json({ error: 'Please sign in.' });
    }
  };

  const authLimiter = rateLimit({ windowMs: 15 * 60 * 1000, limit: 30, standardHeaders: true, legacyHeaders: false, message: { error: 'Too many attempts. Try again in a few minutes.' } });
  const api = express.Router();

  api.get('/health', (req, res) => res.json({ ok: true }));

  api.post('/auth/register', authLimiter, async (req, res) => {
    const name = String(req.body.name || '').trim().slice(0, 80);
    const email = String(req.body.email || '').trim().toLowerCase();
    const password = String(req.body.password || '');
    if (!name) return res.status(400).json({ error: 'Enter your name.' });
    if (!EMAIL.test(email)) return res.status(400).json({ error: 'Enter a valid email address.' });
    if (password.length < 8 || password.length > 200) return res.status(400).json({ error: 'Use a password of at least 8 characters.' });
    if (db.findUserByEmail(email)) return res.status(409).json({ error: 'An account with this email already exists. Try logging in.' });
    const user = await db.createUser({
      name,
      email,
      passwordHash: await bcrypt.hash(password, 10),
      role: educatorEmails.includes(email) ? 'educator' : 'student',
    });
    issue(res, user);
    res.status(201).json({ user: db.publicUser(user) });
  });

  api.post('/auth/login', authLimiter, async (req, res) => {
    const email = String(req.body.email || '').trim().toLowerCase();
    const password = String(req.body.password || '');
    const user = db.findUserByEmail(email);
    // Always run bcrypt so response time doesn't reveal whether the email exists.
    const ok = await bcrypt.compare(password, user ? user.passwordHash : '$2a$10$invalidinvalidinvalidinvalidinvalidinvalidinvalidinvalidinva');
    if (!user || !ok) return res.status(401).json({ error: 'Incorrect email or password.' });
    issue(res, user);
    res.json({ user: db.publicUser(user) });
  });

  api.post('/auth/logout', (req, res) => {
    res.clearCookie(COOKIE, { ...cookieOpts, maxAge: undefined });
    res.json({ ok: true });
  });

  api.get('/auth/me', (req, res) => {
    try {
      const { sub } = jwt.verify(req.cookies[COOKIE] || '', jwtSecret);
      const user = db.findUserById(sub);
      return res.json({ user: db.publicUser(user) });
    } catch (e) {
      return res.json({ user: null });
    }
  });

  api.put('/me/theme', requireAuth, async (req, res) => {
    const { theme } = req.body;
    if (theme !== 'light' && theme !== 'dark') return res.status(400).json({ error: 'theme must be "light" or "dark".' });
    await db.setTheme(req.user.id, theme);
    res.json({ ok: true });
  });

  // Per-user saved UI state for a page (answers, drafts, grades …), keyed by page.
  api.get('/state/:key', requireAuth, (req, res) => {
    if (!STATE_KEY.test(req.params.key)) return res.status(400).json({ error: 'Invalid key.' });
    res.json({ data: db.getState(req.user.id, req.params.key) });
  });
  api.put('/state/:key', requireAuth, async (req, res) => {
    if (!STATE_KEY.test(req.params.key)) return res.status(400).json({ error: 'Invalid key.' });
    const { data } = req.body;
    if (data === undefined || data === null || typeof data !== 'object') return res.status(400).json({ error: 'data must be an object.' });
    await db.setState(req.user.id, req.params.key, data);
    res.json({ ok: true });
  });

  // Weekly lesson plans: plain Markdown files in backend/content (week-01.md, week-02.md …).
  // Read on every request, so adding or editing a file shows up on refresh without a restart.
  const SLUG = /^week-\d{2}$/;
  const readWeek = (slug) => {
    const markdown = fs.readFileSync(path.join(contentDir, `${slug}.md`), 'utf8');
    const title = (markdown.match(/^#\s+(.+)$/m) || [])[1] || slug;
    const term = (markdown.match(/\*\*Term:\*\*\s*([^·\n]+)/) || [])[1];
    return { slug, title: title.trim(), term: term ? term.trim() : null, markdown };
  };
  api.get('/content', requireAuth, (req, res) => {
    const files = fs.existsSync(contentDir) ? fs.readdirSync(contentDir) : [];
    const weeks = files
      .map((f) => f.replace(/\.md$/, ''))
      .filter((slug) => SLUG.test(slug))
      .sort()
      .map((slug) => {
        const { title, term } = readWeek(slug);
        return { slug, title, term };
      });
    res.json({ weeks });
  });
  api.get('/content/:slug', requireAuth, (req, res) => {
    if (!SLUG.test(req.params.slug) || !fs.existsSync(path.join(contentDir, `${req.params.slug}.md`))) {
      return res.status(404).json({ error: 'That lesson plan does not exist.' });
    }
    res.json({ ...readWeek(req.params.slug), notebook: notebookInfo(req.params.slug) });
  });

  // Hands-on Jupyter notebooks (notebooks/week-NN.ipynb). Students open them in Google Colab from the public
  // GitHub copy, or download the file and upload it to Colab themselves.
  const notebookFile = (slug) => path.join(notebooksDir, `${slug}.ipynb`);
  function notebookInfo(slug) {
    if (!SLUG.test(slug) || !fs.existsSync(notebookFile(slug))) return null;
    return {
      colabUrl: colabRepo ? `https://colab.research.google.com/github/${colabRepo}/blob/${colabBranch}/notebooks/${slug}.ipynb` : null,
      downloadUrl: `/api/notebooks/${slug}/download`,
    };
  }
  api.get('/notebooks/:slug', requireAuth, (req, res) => {
    const info = notebookInfo(req.params.slug);
    if (!info) return res.status(404).json({ error: 'There is no notebook for that week yet.' });
    res.json({ slug: req.params.slug, ...info });
  });
  api.get('/notebooks/:slug/download', requireAuth, (req, res) => {
    if (!notebookInfo(req.params.slug)) return res.status(404).json({ error: 'There is no notebook for that week yet.' });
    res.type('application/x-ipynb+json');
    res.download(notebookFile(req.params.slug), `${req.params.slug}.ipynb`);
  });

  app.use('/api', api);
  app.use('/api', (req, res) => res.status(404).json({ error: 'Not found.' }));

  // Serve the built frontend (single-process hosting). Skipped when the folder doesn't exist.
  if (frontendDist && fs.existsSync(path.join(frontendDist, 'index.html'))) {
    app.use(express.static(frontendDist, { index: false, maxAge: '1h', setHeaders: (res, p) => { if (p.includes(`${path.sep}assets${path.sep}`) && /-[A-Za-z0-9_-]{8}\./.test(p)) res.setHeader('Cache-Control', 'public, max-age=31536000, immutable'); } }));
    app.get('*', (req, res) => res.sendFile(path.join(frontendDist, 'index.html')));
  }

  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (err.type === 'entity.parse.failed') return res.status(400).json({ error: 'Invalid JSON.' });
    if (err.type === 'entity.too.large') return res.status(413).json({ error: 'Payload too large.' });
    console.error(err);
    res.status(500).json({ error: 'Something went wrong.' });
  });

  return app;
}
