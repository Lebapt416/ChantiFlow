export default function Loading() {
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
      {/* Header skeleton */}
      <header className="sticky top-0 z-50 border-b border-zinc-200/50 bg-white/80 backdrop-blur-md dark:border-zinc-800/50 dark:bg-zinc-950/80">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
              <div className="h-6 w-32 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
            </div>
            <div className="flex items-center gap-6">
              <div className="h-6 w-20 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
              <div className="h-10 w-24 rounded-full bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
            </div>
          </div>
        </div>
      </header>

      {/* Hero section skeleton */}
      <section className="relative mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-4xl text-center space-y-6">
          <div className="h-6 w-48 mx-auto rounded-full bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          <div className="h-16 w-full max-w-3xl mx-auto rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          <div className="h-6 w-full max-w-2xl mx-auto rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          <div className="flex justify-center gap-4 mt-10">
            <div className="h-12 w-48 rounded-xl bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
            <div className="h-12 w-40 rounded-xl bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          </div>
          <div className="mt-16 h-[400px] w-full max-w-6xl mx-auto rounded-2xl bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
        </div>
      </section>
    </div>
  );
}

