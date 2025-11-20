'use client';

import { startTransition, useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'chantiflow-theme';
type Theme = 'light' | 'dark';

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>('light');

  const applyTheme = useCallback((next: Theme) => {
    document.documentElement.dataset.theme = next;
    document.documentElement.classList.toggle('dark', next === 'dark');
    localStorage.setItem(STORAGE_KEY, next);
  }, []);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initial: Theme = stored ?? (prefersDark ? 'dark' : 'light');
    startTransition(() => setTheme(initial));
    applyTheme(initial);
  }, [applyTheme]);

  function toggleTheme() {
    setTheme((prev) => {
      const next: Theme = prev === 'light' ? 'dark' : 'light';
      applyTheme(next);
      return next;
    });
  }

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-white/80 px-4 py-2 text-sm font-medium text-zinc-700 shadow-lg shadow-black/10 transition hover:border-zinc-400 hover:text-zinc-900 dark:border-white/30 dark:bg-zinc-900/80 dark:text-white dark:hover:border-white/60"
    >
      <span role="img" aria-hidden="true">
        {theme === 'light' ? '🌙' : '☀️'}
      </span>
      Mode {theme === 'light' ? 'sombre' : 'clair'}
    </button>
  );
}

