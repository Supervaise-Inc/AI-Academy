import React, { Suspense, lazy, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import * as session from './dc/session.js';

// One chunk per screen: the 10 pages are large and a visitor only needs one at a time.
const pages = {
  Landing: () => import('./pages/Landing.jsx'),
  LandingFull: () => import('./pages/LandingFull.jsx'),
  Main: () => import('./pages/Main.jsx'),
  CourseMap: () => import('./pages/CourseMap.jsx'),
  Lesson: () => import('./pages/Lesson.jsx'),
  PromptWorkshop: () => import('./pages/PromptWorkshop.jsx'),
  Capstone: () => import('./pages/Capstone.jsx'),
  PracticeExam: () => import('./pages/PracticeExam.jsx'),
  Certificates: () => import('./pages/Certificates.jsx'),
  Educator: () => import('./pages/Educator.jsx'),
};

// Student Activities / Announcements / My grades. Plain React pages (not design-converted), so they mount without PageHost.
const lmsPages = {
  Activities: lazy(() => import('./lms/Activities.jsx')),
  Announcements: lazy(() => import('./lms/Announcements.jsx')),
  Grades: lazy(() => import('./lms/Grades.jsx')),
};

const useSession = () => useSyncExternalStore(session.subscribe, session.getSession);

const Blank = () => <div style={{ minHeight: '100vh', background: '#090A0F' }} aria-busy="true" />;

// Loads a page's chunk, then mounts it.
function LazyPage({ name }) {
  const { week } = useParams();
  const [Page, setPage] = useState(null);
  useEffect(() => {
    let alive = true;
    pages[name]().then((m) => alive && setPage(() => m.default));
    return () => {
      alive = false;
    };
  }, [name]);
  return Page ? <PageHost key={week || 'default'} Page={Page} week={week} /> : <Blank />;
}

// Mounts a design page. Loads the user's saved state for that page before first paint of saved data.
function PageHost({ Page, week }) {
  const { user } = useSession();
  const ref = useRef(null);
  useEffect(() => {
    let alive = true;
    const inst = ref.current;
    if (!inst) return undefined;
    if (!user || !Page.persist || !Page.persist.length) {
      inst.__hydrated = !!user; // nothing to load; still allow saving for signed-in users
      return undefined;
    }
    session
      .loadPageState(Page.pageKey)
      .then((saved) => alive && inst.hydrate(saved))
      .catch(() => alive && inst.hydrate({}));
    return () => {
      alive = false;
    };
  }, [Page, user && user.id]);
  return <Page ref={ref} user={user} week={week} />;
}

// Signed-in screens. Educators may also view student screens; only educators see /educator.
function Protected({ role, children }) {
  const { ready, user } = useSession();
  const location = useLocation();
  if (!ready) return <Blank />;
  if (!user) {
    session.setAfterLogin(location.pathname);
    return <Navigate to="/" replace />;
  }
  if (role && user.role !== role) return <Navigate to={session.homeFor(user)} replace />;
  return children;
}

function Bridge() {
  const navigate = useNavigate();
  const { pathname, hash } = useLocation();
  useEffect(() => {
    session.setNavigate((to) => navigate(to));
  }, [navigate]);
  useEffect(() => {
    if (!hash) window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

export default function App() {
  useEffect(() => {
    session.init();
  }, []);
  const guard = (el, role) => <Protected role={role}>{el}</Protected>;
  return (
    <BrowserRouter>
      <Bridge />
      <Routes>
        <Route path="/" element={<LazyPage name="LandingFull" />} />
        <Route path="/intro" element={<LazyPage name="Landing" />} />
        <Route path="/dashboard" element={guard(<LazyPage name="Main" />)} />
        <Route path="/dashboard/:week" element={guard(<LazyPage name="Main" />)} />
        <Route path="/course-map" element={guard(<LazyPage name="CourseMap" />)} />
        <Route path="/lesson" element={guard(<LazyPage name="Lesson" />)} />
        <Route path="/lesson/:week" element={guard(<LazyPage name="Lesson" />)} />
        <Route path="/prompt-workshop" element={guard(<LazyPage name="PromptWorkshop" />)} />
        <Route path="/prompt-workshop/:week" element={guard(<LazyPage name="PromptWorkshop" />)} />
        <Route path="/capstone" element={guard(<LazyPage name="Capstone" />)} />
        <Route path="/mock-exam" element={guard(<LazyPage name="PracticeExam" />)} />
        <Route path="/certification" element={guard(<LazyPage name="Certificates" />)} />
        {Object.entries({ '/activities': 'Activities', '/announcements': 'Announcements', '/grades': 'Grades' }).map(([path, name]) => {
          const Page = lmsPages[name];
          return (
            <Route
              key={path}
              path={path}
              element={guard(
                <Suspense fallback={<Blank />}>
                  <Page />
                </Suspense>,
              )}
            />
          );
        })}
        <Route path="/educator" element={guard(<LazyPage name="Educator" />, 'educator')} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
