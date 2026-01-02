export default function HomeLoading() {
  return (
    <div className="space-y-8">
      {/* Skeleton pour les chantiers en cours */}
      <div className="mb-8">
        <div className="mb-4 flex items-center gap-2">
          <div className="h-7 w-48 rounded bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
          <div className="h-6 w-12 rounded-full bg-zinc-200 dark:bg-zinc-800 animate-pulse" />
        </div>
        <div className="overflow-x-auto pb-4">
          <div className="flex gap-4 min-w-max">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-64 w-80 rounded-2xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 animate-pulse"
              >
                <div className="p-6 space-y-4">
                  <div className="h-6 w-3/4 rounded bg-zinc-200 dark:bg-zinc-800" />
                  <div className="h-4 w-1/2 rounded bg-zinc-200 dark:bg-zinc-800" />
                  <div className="h-20 w-full rounded bg-zinc-200 dark:bg-zinc-800" />
                  <div className="flex gap-2">
                    <div className="h-6 w-16 rounded-full bg-zinc-200 dark:bg-zinc-800" />
                    <div className="h-6 w-16 rounded-full bg-zinc-200 dark:bg-zinc-800" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Skeleton pour la carte de création */}
      <div className="overflow-x-auto pb-4">
        <div className="flex gap-4 min-w-max">
          <div className="h-64 w-80 rounded-2xl border-2 border-dashed border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900/50 animate-pulse">
            <div className="p-6 flex flex-col items-center justify-center h-full space-y-4">
              <div className="h-12 w-12 rounded-full bg-zinc-200 dark:bg-zinc-800" />
              <div className="h-6 w-32 rounded bg-zinc-200 dark:bg-zinc-800" />
              <div className="h-4 w-48 rounded bg-zinc-200 dark:bg-zinc-800" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
