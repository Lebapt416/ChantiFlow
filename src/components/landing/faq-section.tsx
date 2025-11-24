type FaqEntry = {
  question: string;
  answer: string;
};

type FaqSectionProps = {
  entries: FaqEntry[];
};

export function FaqSection({ entries }: FaqSectionProps) {
  return (
    <section className="mx-auto max-w-7xl px-6 py-20">
      <div className="mx-auto max-w-3xl text-center">
        <p className="text-sm font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400">
          FAQ ChantiFlow
        </p>
        <h2 className="mt-4 text-3xl font-bold tracking-tight text-zinc-900 dark:text-white sm:text-4xl">
          Tout ce qu’il faut savoir avant de déployer ChantiFlow
        </h2>
        <p className="mt-4 text-lg text-zinc-600 dark:text-zinc-400">
          Nous accompagnons chaque client avec un plan de mise en œuvre adapté à la taille de l’entreprise.
        </p>
      </div>
      <div className="mx-auto mt-12 max-w-4xl space-y-4">
        {entries.map((entry) => (
          <div
            key={entry.question}
            className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
          >
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">{entry.question}</h3>
            <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{entry.answer}</p>
          </div>
        ))}
      </div>
    </section>
  );
}


