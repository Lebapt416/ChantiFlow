'use client';

type DataPoint = {
  label: string;
  value: number;
  color?: string;
};

type Props = {
  data: DataPoint[];
  title?: string;
  height?: number;
};

export function SimpleChart({ data, title, height = 200 }: Props) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className="space-y-2">
      {title && (
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{title}</h3>
      )}
      <div className="space-y-2">
        {data.map((point, index) => (
          <div key={index} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-zinc-600 dark:text-zinc-400">{point.label}</span>
              <span className="font-semibold text-zinc-900 dark:text-white">{point.value}</span>
            </div>
            <div className="h-2 rounded-full bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${(point.value / maxValue) * 100}%`,
                  backgroundColor: point.color || '#10b981',
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

