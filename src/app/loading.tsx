export default function Loading() {
  return (
    <div className="min-h-screen bg-paper">
      {/* Header skeleton */}
      <header className="sticky top-0 z-50 border-b border-rule-soft bg-paper">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded bg-paper-2 animate-pulse" />
              <div className="h-6 w-32 rounded bg-paper-2 animate-pulse" />
            </div>
            <div className="flex items-center gap-6">
              <div className="h-6 w-20 rounded bg-paper-2 animate-pulse" />
              <div className="h-10 w-24 rounded bg-paper-2 animate-pulse" />
            </div>
          </div>
        </div>
      </header>

      {/* Hero section skeleton */}
      <section className="relative mx-auto max-w-7xl px-6 py-20">
        <div className="mx-auto max-w-4xl text-center space-y-6">
          <div className="h-6 w-48 mx-auto rounded bg-paper-2 animate-pulse" />
          <div className="h-16 w-full max-w-3xl mx-auto rounded bg-paper-2 animate-pulse" />
          <div className="h-6 w-full max-w-2xl mx-auto rounded bg-paper-2 animate-pulse" />
          <div className="flex justify-center gap-4 mt-10">
            <div className="h-12 w-48 rounded bg-paper-2 animate-pulse" />
            <div className="h-12 w-40 rounded bg-paper-2 animate-pulse" />
          </div>
          <div className="mt-16 h-[400px] w-full max-w-6xl mx-auto rounded bg-paper-2 animate-pulse" />
        </div>
      </section>
    </div>
  );
}
