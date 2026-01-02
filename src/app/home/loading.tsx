export default function HomeLoading() {
  return (
    <div className="min-h-screen bg-white dark:bg-zinc-950">
      {/* Header skeleton */}
      <div className="border-b border-zinc-200 dark:border-zinc-800">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="h-8 w-32 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
            <div className="h-10 w-24 rounded-full bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          </div>
        </div>
      </div>

      {/* Content skeleton */}
      <div className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <div className="h-7 w-48 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
            <div className="h-6 w-8 rounded-full bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          </div>
          <div className="flex gap-4 overflow-x-auto pb-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="min-w-[320px] rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 space-y-4"
              >
                <div className="h-6 w-3/4 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
                <div className="h-4 w-full rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
                <div className="h-4 w-2/3 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
                <div className="flex gap-2 mt-4">
                  <div className="h-8 w-20 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
                  <div className="h-8 w-20 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
                  <div className="h-8 w-20 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

