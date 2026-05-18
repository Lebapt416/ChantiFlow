export default function HomeLoading() {
  return (
    <div className="space-y-8">
      {/* Skeleton pour les chantiers en cours */}
      <div className="mb-8">
        <div className="mb-4 flex items-center gap-2">
          <div className="h-7 w-48 rounded bg-paper-2 animate-pulse" />
          <div className="h-6 w-12 rounded-full bg-paper-2 animate-pulse" />
        </div>
        <div className="overflow-x-auto pb-4">
          <div className="flex gap-4 min-w-max">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-64 w-80 rounded border border-rule-soft bg-paper animate-pulse"
              >
                <div className="p-6 space-y-4">
                  <div className="h-6 w-3/4 rounded bg-paper-2" />
                  <div className="h-4 w-1/2 rounded bg-paper-2" />
                  <div className="h-20 w-full rounded bg-paper-2" />
                  <div className="flex gap-2">
                    <div className="h-6 w-16 rounded-full bg-paper-2" />
                    <div className="h-6 w-16 rounded-full bg-paper-2" />
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
          <div className="h-64 w-80 rounded border-2 border-dashed border-rule-soft bg-paper-2  animate-pulse">
            <div className="p-6 flex flex-col items-center justify-center h-full space-y-4">
              <div className="h-12 w-12 rounded-full bg-paper-2" />
              <div className="h-6 w-32 rounded bg-paper-2" />
              <div className="h-4 w-48 rounded bg-paper-2" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
