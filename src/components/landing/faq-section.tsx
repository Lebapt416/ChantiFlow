type FaqEntry = { question: string; answer: string; };
type FaqSectionProps = { entries: FaqEntry[]; };

export function FaqSection({ entries }: FaqSectionProps) {
  return (
    <section className="border-t border-rule py-20">
      <div className="mx-auto max-w-7xl px-8">
        {/* Chapter header */}
        <div className="grid grid-cols-[auto_1fr] gap-20 items-end pb-16 border-b border-rule-soft mb-16">
          <div className="font-mono text-[12px] text-ink-2 tracking-widest">
            <span className="block font-serif text-[96px] italic font-light text-ink leading-none mb-2" style={{fontVariationSettings: '"opsz" 144, "SOFT" 100'}}>04</span>
            Chapitre · Questions
          </div>
          <div>
            <h2 className="font-serif font-normal text-ink leading-[1.02] tracking-tight" style={{fontSize: 'clamp(32px, 4vw, 48px)', fontVariationSettings: '"opsz" 144, "SOFT" 30'}}>
              Tout ce qu&apos;il faut savoir<br/>avant de <em className="italic font-light text-orange">déployer.</em>
            </h2>
          </div>
        </div>

        {/* Accordion */}
        <div className="divide-y divide-rule-soft max-w-3xl">
          {entries.map((entry) => (
            <details key={entry.question} className="group py-5">
              <summary className="flex items-center justify-between cursor-pointer list-none gap-4">
                <span className="font-sans font-medium text-[16px] text-ink">{entry.question}</span>
                <span className="font-mono text-[16px] text-ink-2 transition-transform group-open:rotate-45 flex-shrink-0">+</span>
              </summary>
              <p className="mt-4 text-[15px] text-ink-2 leading-relaxed pr-8">{entry.answer}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
