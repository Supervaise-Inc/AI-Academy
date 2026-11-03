import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

const here = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(here, '..', '.env') });

const list = (v) => String(v || '').split(',').map((s) => s.trim()).filter(Boolean);
const root = path.join(here, '..');

export function loadConfig(env = process.env) {
  const isProd = env.NODE_ENV === 'production';
  const jwtSecret = env.JWT_SECRET || (isProd ? '' : 'dev-only-insecure-secret');
  if (!jwtSecret || jwtSecret === 'change-me' || (isProd && jwtSecret.length < 32)) {
    throw new Error('JWT_SECRET must be set to a random string of at least 32 characters in production.');
  }
  const sameSite = (env.COOKIE_SAMESITE || 'lax').toLowerCase();
  const secure = env.COOKIE_SECURE ? env.COOKIE_SECURE === 'true' : isProd;
  if (sameSite === 'none' && !secure) throw new Error('COOKIE_SAMESITE=none requires COOKIE_SECURE=true (HTTPS).');
  return {
    isProd,
    port: Number(env.PORT) || 4000,
    jwtSecret,
    corsOrigins: list(env.CORS_ORIGIN),
    educatorEmails: list(env.EDUCATOR_EMAILS).map((e) => e.toLowerCase()),
    contentDir: path.resolve(root, env.CONTENT_DIR || './content'),
    notebooksDir: path.resolve(root, env.NOTEBOOKS_DIR || '../notebooks'),
    // "Open in Colab" links point at the public GitHub copy of the notebooks. Set COLAB_REPO to "" to hide the button.
    colabRepo: env.COLAB_REPO === undefined ? 'Supervaise-Inc/AI-Academy' : env.COLAB_REPO.trim(),
    colabBranch: (env.COLAB_BRANCH || 'main').trim(),
    dataDir: path.resolve(root, env.DATA_DIR || './data'),
    frontendDist: path.resolve(root, env.FRONTEND_DIST || '../frontend/dist'),
    sameSite,
    secure,
    trustProxy: env.TRUST_PROXY === '1' || env.TRUST_PROXY === 'true',
  };
}
