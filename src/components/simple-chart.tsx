'use client';

import { useEffect, useState } from 'react';

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

export function SimpleChart({ data, title }: Props) {
  const targetValues = data.map((d) => d.value);
  const maxValue = Math.max(...targetValues, 1);
  const [animatedValues, setAnimatedValues] = useState(() => data.map(() => 0));

  useEffect(() => {
    setAnimatedValues(data.map(() => 0));
    const timeout = setTimeout(() => {
      setAnimatedValues(data.map((point) => point.value));
    }, 50);
    return () => clearTimeout(timeout);
  }, [data]);

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
              <span className="font-semibold text-zinc-900 dark:text-white">
                {Math.round((animatedValues[index] + Number.EPSILON) * 100) / 100}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
              <div
                className="h-full rounded-full transition-[width] duration-500 ease-out"
                style={{
                  width: `${(animatedValues[index] / maxValue) * 100}%`,
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

