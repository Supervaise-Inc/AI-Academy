// Tiny framework-agnostic store for the signed-in user + API client.
// The ported design pages are class components, so they read this store instead of React context.
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '') + '/api';
const THEME_KEY = 'supervaise-theme';
const THEME_EVENT = 'supervaise-theme-change';

// Absolute URL for a server path such as "/api/notebooks/week-01/download" (honours VITE_API_URL).
export const apiUrl = (p) => (import.meta.env.VITE_API_URL || '').replace(/\/$/, '') + p;

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function api(path, { method = 'GET', body } = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method,
      credentials: 'include',
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError('Cannot reach the server. Check your connection and try again.', 0);
  }
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    /* empty body */
  }
  if (!res.ok) throw new ApiError((data && data.error) || `Request failed (${res.status})`, res.status);
  return data;
}

let state = { ready: false, user: null };
const listeners = new Set();
let navigateFn = null;
let afterLogin = null;

const emit = () => listeners.forEach((fn) => fn(state));
const set = (patch) => {
  state = { ...state, ...patch };
  emit();
};

export const getSession = () => state;
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
export const setNavigate = (fn) => {
  navigateFn = fn;
};
export const navigate = (to) => navigateFn && navigateFn(to);
export const setAfterLogin = (path) => {
  afterLogin = path;
};

function applyTheme(user) {
  if (user && (user.theme === 'light' || user.theme === 'dark')) {
    try {
      localStorage.setItem(THEME_KEY, user.theme);
    } catch (e) {
      /* storage unavailable */
    }
  }
}

export async function init() {
  try {
    const { user } = await api('/auth/me');
    applyTheme(user);
    set({ ready: true, user });
  } catch (e) {
    set({ ready: true, user: null });
  }
}

export async function login(email, password) {
  const { user } = await api('/auth/login', { method: 'POST', body: { email, password } });
  applyTheme(user);
  set({ user });
  return user;
}

export async function register({ name, email, password }) {
  const { user } = await api('/auth/register', { method: 'POST', body: { name, email, password } });
  set({ user });
  return user;
}

export async function logout() {
  try {
    await api('/auth/logout', { method: 'POST' });
  } finally {
    set({ user: null });
  }
}

export function homeFor(user) {
  return user && user.role === 'educator' ? '/educator' : '/dashboard';
}

// Where to go after a successful sign-in: the page the visitor was sent away from, else their home.
export function afterLoginPath(user) {
  const p = afterLogin;
  afterLogin = null;
  if (p && !(p === '/educator' && user.role !== 'educator')) return p;
  return homeFor(user);
}

// Per-user, per-page persisted UI state (answers, drafts, grades …).
export const loadPageState = (key) => api(`/state/${encodeURIComponent(key)}`).then((r) => r.data);
export const savePageState = (key, data) => api(`/state/${encodeURIComponent(key)}`, { method: 'PUT', body: { data } });

// The pages announce theme changes on window; mirror them to the user's profile.
if (typeof window !== 'undefined') {
  window.addEventListener(THEME_EVENT, (e) => {
    const theme = e.detail;
    if (state.user && (theme === 'light' || theme === 'dark') && state.user.theme !== theme) {
      state = { ...state, user: { ...state.user, theme } };
      api('/me/theme', { method: 'PUT', body: { theme } }).catch(() => {});
    }
  });
}
