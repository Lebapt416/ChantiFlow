'use client';

import { useState, useRef, useCallback, useMemo } from 'react';
import { formatDateISO } from '@/lib/ai/jours-ouvres';

// ─── Types ───────────────────────────────────────────────────────────────────

export type PlanningTask = {
  taskId: string;
  taskTitle: string;
  startDate: string;   // ISO YYYY-MM-DD
  endDate: string;     // ISO YYYY-MM-DD
  status: string | null;
  requiredRole?: string | null;
  assignedWorkerId?: string | null;
  durationHours?: number;
  order?: number;
  priority?: 'high' | 'medium' | 'low';
};

export type PlanningWorker = {
  id: string;
  name: string;
  role?: string | null;
};

export type PlanningViewEditorialProps = {
  siteId: string;
  siteName: string;
  siteAddress?: string | null;
  siteDeadline?: string | null;
  initialTasks: PlanningTask[];
  workers: PlanningWorker[];
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function getTaskStyle(task: PlanningTask): { bg: string; text: string; border?: string; dashed?: boolean } {
  const status = (task.status || '').toLowerCase();
  const now = new Date();
  const endDate = task.endDate ? new Date(task.endDate) : null;
  const isOverdue =
    endDate &&
    endDate < now &&
    status !== 'done' &&
    status !== 'completed' &&
    status !== 'validated' &&
    status !== 'terminé';

  if (
    status === 'done' ||
    status === 'completed' ||
    status === 'validated' ||
    status === 'terminé'
  ) {
    return { bg: '#2F5D3A', text: '#F1EBDF' };
  }
  if (isOverdue) {
    return { bg: '#E85D1F', text: '#F1EBDF' };
  }
  if (status === 'in_progress' || status === 'en cours') {
    return { bg: '#1E3A8A', text: '#F1EBDF' };
  }
  if (status === 'recalculated' || status === 'replanifié') {
    return { bg: 'transparent', text: '#1E3A8A', border: '#1E3A8A', dashed: true };
  }
  // Default: "à planifier" / pending
  return { bg: 'transparent', text: '#5A554B', border: '#5A554B', dashed: false };
}

function getTaskIcon(task: PlanningTask): string {
  const status = (task.status || '').toLowerCase();
  const now = new Date();
  const endDate = task.endDate ? new Date(task.endDate) : null;
  const isOverdue =
    endDate &&
    endDate < now &&
    status !== 'done' &&
    status !== 'completed' &&
    status !== 'validated';
  if (status === 'done' || status === 'completed' || status === 'validated') return '✓ ';
  if (isOverdue) return '⚠ ';
  return '';
}

function buildDateRange(tasks: PlanningTask[], paddingDays = 3): Date[] {
  if (tasks.length === 0) {
    const start = new Date();
    return Array.from({ length: 14 }, (_, i) => {
      const d = new Date(start);
      d.setDate(d.getDate() + i);
      return d;
    });
  }
  const dates = tasks.flatMap(t => [new Date(t.startDate), new Date(t.endDate)]);
  const minDate = new Date(Math.min(...dates.map(d => d.getTime())));
  const maxDate = new Date(Math.max(...dates.map(d => d.getTime())));
  minDate.setDate(minDate.getDate() - paddingDays);
  maxDate.setDate(maxDate.getDate() + paddingDays);
  const result: Date[] = [];
  const cur = new Date(minDate);
  while (cur <= maxDate) {
    result.push(new Date(cur));
    cur.setDate(cur.getDate() + 1);
  }
  return result;
}

const DAY_WIDTH = { day: 80, week: 40, month: 20 } as const;
const ROW_HEIGHT = 52; // px per task row
const LABEL_COL_WIDTH = 140; // px for the left label column

// ─── Component ───────────────────────────────────────────────────────────────

export function PlanningViewEditorial({
  siteId,
  siteName,
  siteAddress,
  siteDeadline,
  initialTasks,
  workers,
}: PlanningViewEditorialProps) {
  const [tasks, setTasks] = useState<PlanningTask[]>(initialTasks);
  const [zoom, setZoom] = useState<'day' | 'week' | 'month'>('week');
  const [selectedTask, setSelectedTask] = useState<PlanningTask | null>(null);
  const [dragging, setDragging] = useState<{
    taskId: string;
    type: 'move' | 'resize-end';
    startX: number;
    originalStart: string;
    originalEnd: string;
  } | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  // Refs to avoid stale closures in pointer handlers
  const draggingRef = useRef(dragging);
  draggingRef.current = dragging;

  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;

  const dayWidth = DAY_WIDTH[zoom];
  const dates = useMemo(() => buildDateRange(tasks), [tasks]);
  const today = formatDateISO(new Date());
  const todayIndex = dates.findIndex(d => formatDateISO(d) === today);

  const workerMap = useMemo(() => new Map(workers.map(w => [w.id, w])), [workers]);

  const DAY_LABELS = ['D', 'L', 'M', 'M', 'J', 'V', 'S'];

  const dateToX = useCallback(
    (dateStr: string): number => {
      const idx = dates.findIndex(d => formatDateISO(d) === dateStr);
      return idx >= 0 ? idx * dayWidth : 0;
    },
    [dates, dayWidth],
  );

  const dateRangeToWidth = useCallback(
    (startStr: string, endStr: string): number => {
      const startIdx = dates.findIndex(d => formatDateISO(d) === startStr);
      const endIdx = dates.findIndex(d => formatDateISO(d) === endStr);
      if (startIdx < 0 || endIdx < 0) return dayWidth;
      return (endIdx - startIdx + 1) * dayWidth;
    },
    [dates, dayWidth],
  );

  const handlePointerDown = useCallback(
    (
      e: React.PointerEvent<HTMLDivElement>,
      task: PlanningTask,
      type: 'move' | 'resize-end',
    ) => {
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      const next = {
        taskId: task.taskId,
        type,
        startX: e.clientX,
        originalStart: task.startDate,
        originalEnd: task.endDate,
      };
      setDragging(next);
      draggingRef.current = next;
    },
    [],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const drag = draggingRef.current;
      if (!drag) return;
      const dx = e.clientX - drag.startX;
      const daysDelta = Math.round(dx / dayWidth);

      setTasks(prev =>
        prev.map(t => {
          if (t.taskId !== drag.taskId) return t;
          const origStart = new Date(drag.originalStart);
          const origEnd = new Date(drag.originalEnd);

          if (drag.type === 'move') {
            const newStart = new Date(origStart);
            const newEnd = new Date(origEnd);
            newStart.setDate(newStart.getDate() + daysDelta);
            newEnd.setDate(newEnd.getDate() + daysDelta);
            return { ...t, startDate: formatDateISO(newStart), endDate: formatDateISO(newEnd) };
          } else {
            // resize-end: only move end date
            const newEnd = new Date(origEnd);
            newEnd.setDate(newEnd.getDate() + daysDelta);
            if (newEnd <= origStart) return t;
            return { ...t, endDate: formatDateISO(newEnd) };
          }
        }),
      );
    },
    [dayWidth],
  );

  const handlePointerUp = useCallback(async () => {
    const drag = draggingRef.current;
    if (!drag) return;
    setDragging(null);
    draggingRef.current = null;

    const updatedTask = tasksRef.current.find(t => t.taskId === drag.taskId);
    if (!updatedTask) return;

    try {
      await fetch(`/api/planning/update-task`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          siteId,
          taskId: updatedTask.taskId,
          startDate: updatedTask.startDate,
          endDate: updatedTask.endDate,
        }),
      });
    } catch (err) {
      console.error('Failed to persist task update', err);
    }
  }, [siteId]);

  const totalGridWidth = dates.length * dayWidth;

  return (
    <div className="flex flex-col h-full bg-paper font-sans select-none">

      {/* TOP BAR */}
      <div className="h-12 bg-ink flex items-center px-4 justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-orange" />
          <div className="w-2.5 h-2.5 rounded-full bg-warn" />
          <div className="w-2.5 h-2.5 rounded-full bg-green" />
          <span className="ml-4 font-mono text-[12px] tracking-[0.08em] text-paper uppercase">
            ChantiFlow / Planning —{' '}
            {new Date()
              .toLocaleDateString('fr-FR', { month: 'long', year: 'numeric' })
              .toUpperCase()}
          </span>
        </div>
        <div className="flex items-center gap-4">
          {/* Zoom toggle */}
          <div className="flex border border-paper/20">
            {(['day', 'week', 'month'] as const).map(z => (
              <button
                key={z}
                onClick={() => setZoom(z)}
                className={`px-3 py-1 font-mono text-[11px] uppercase tracking-widest transition-colors ${
                  zoom === z ? 'bg-paper text-ink' : 'text-paper/60 hover:text-paper'
                }`}
              >
                {z === 'day' ? 'Jour' : z === 'week' ? 'Sem.' : 'Mois'}
              </button>
            ))}
          </div>
          <span className="font-mono text-[12px] text-paper/60">⌘K</span>
        </div>
      </div>

      {/* SITE HEADER */}
      <div className="px-6 py-3 border-b border-rule flex items-end justify-between flex-shrink-0">
        <div>
          <p
            className="font-serif text-[22px] italic text-ink leading-tight"
            style={{ fontVariationSettings: '"opsz" 60, "SOFT" 60' }}
          >
            {siteName}
          </p>
          <p className="font-mono text-[10px] uppercase tracking-widest text-ink-2 mt-0.5">
            {siteAddress || 'Adresse non renseignée'}
            {siteDeadline &&
              ` · Livraison ${new Date(siteDeadline).toLocaleDateString('fr-FR')}`}
          </p>
        </div>
        {/* Legend */}
        <div className="flex items-center gap-5 pb-0.5">
          {[
            { color: '#2F5D3A', label: 'Validé' },
            { color: '#E85D1F', label: 'Retard' },
            { color: '#1E3A8A', label: 'Prévu' },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className="w-3 h-3" style={{ background: color }} />
              <span className="font-mono text-[10px] uppercase tracking-widest text-ink-2">
                {label}
              </span>
            </div>
          ))}
          <div className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 border border-dashed"
              style={{ borderColor: '#1E3A8A' }}
            />
            <span className="font-mono text-[10px] uppercase tracking-widest text-ink-2">
              Recalc.
            </span>
          </div>
        </div>
      </div>

      {/* GANTT GRID */}
      <div
        className="flex-1 overflow-auto"
        ref={gridRef}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <div style={{ minWidth: LABEL_COL_WIDTH + totalGridWidth }}>

          {/* DATE HEADER */}
          <div className="flex sticky top-0 z-10 bg-paper border-b border-rule-soft">
            <div style={{ width: LABEL_COL_WIDTH, flexShrink: 0 }} />
            <div className="flex relative" style={{ width: totalGridWidth }}>
              {dates.map((date, i) => {
                const dow = date.getDay();
                const isWeekend = dow === 0 || dow === 6;
                return (
                  <div
                    key={i}
                    style={{ width: dayWidth, flexShrink: 0 }}
                    className={`border-r border-rule-soft/30 py-2 text-center ${isWeekend ? 'opacity-40' : ''}`}
                  >
                    <span className="font-mono text-[10px] uppercase tracking-wide text-ink-2">
                      {DAY_LABELS[dow]} {date.getDate()}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* TASK ROWS */}
          {tasks.map(task => {
            const style = getTaskStyle(task);
            const icon = getTaskIcon(task);
            const barLeft = dateToX(task.startDate);
            const barWidth = dateRangeToWidth(task.startDate, task.endDate);
            const assignedWorker = task.assignedWorkerId
              ? workerMap.get(task.assignedWorkerId)
              : null;

            return (
              <div
                key={task.taskId}
                className="flex border-b border-rule-soft/40 hover:bg-paper-2/50 transition-colors duration-100 relative"
                style={{ height: ROW_HEIGHT }}
              >
                {/* Row label */}
                <div
                  style={{ width: LABEL_COL_WIDTH, flexShrink: 0 }}
                  className="flex flex-col justify-center px-4 border-r border-rule-soft"
                >
                  <span className="font-sans text-[12px] text-ink-2 truncate leading-tight">
                    {task.requiredRole || task.taskTitle}
                  </span>
                  {assignedWorker && (
                    <span className="font-mono text-[9px] text-ink-3 uppercase tracking-wide truncate mt-0.5">
                      {assignedWorker.name}
                    </span>
                  )}
                </div>

                {/* Task bars area */}
                <div className="flex-1 relative" style={{ width: totalGridWidth }}>
                  {/* Vertical dashed grid lines */}
                  {dates.map((_, i) => (
                    <div
                      key={i}
                      className="absolute top-0 bottom-0 border-r border-dashed border-rule-soft/20"
                      style={{ left: i * dayWidth }}
                    />
                  ))}

                  {/* Today line */}
                  {todayIndex >= 0 && (
                    <div
                      className="absolute top-0 bottom-0 z-10"
                      style={{
                        left: todayIndex * dayWidth + dayWidth / 2,
                        width: '1.5px',
                        background: '#E85D1F',
                      }}
                    />
                  )}

                  {/* Task bar */}
                  <div
                    className="absolute top-3 bottom-3 cursor-grab active:cursor-grabbing z-20 flex items-center px-3 overflow-hidden"
                    style={{
                      left: barLeft,
                      width: Math.max(barWidth, 20),
                      background: style.bg,
                      border: style.border
                        ? `${style.dashed ? '1.5px dashed' : '1px solid'} ${style.border}`
                        : undefined,
                    }}
                    onPointerDown={e => handlePointerDown(e, task, 'move')}
                    onClick={() => setSelectedTask(task)}
                    title={`${icon}${task.taskTitle}${assignedWorker ? ` · ${assignedWorker.name}` : ''}`}
                  >
                    <span
                      className="font-sans text-[12px] font-medium whitespace-nowrap overflow-hidden text-ellipsis flex-1"
                      style={{ color: style.text }}
                    >
                      {icon}
                      {task.taskTitle}
                    </span>

                    {/* Resize handle on right edge */}
                    <div
                      className="absolute right-0 top-0 bottom-0 w-2 cursor-col-resize z-30 hover:bg-white/20"
                      onPointerDown={e => {
                        e.stopPropagation();
                        handlePointerDown(e, task, 'resize-end');
                      }}
                    />
                  </div>
                </div>
              </div>
            );
          })}

          {/* Today label at bottom */}
          {todayIndex >= 0 && (
            <div
              className="sticky bottom-0 left-0 z-10 flex"
              style={{ height: 24, minWidth: LABEL_COL_WIDTH + totalGridWidth }}
            >
              <div style={{ width: LABEL_COL_WIDTH, flexShrink: 0 }} />
              <div className="relative flex-1" style={{ width: totalGridWidth }}>
                <div
                  className="absolute flex items-center justify-center"
                  style={{
                    left: todayIndex * dayWidth,
                    width: dayWidth,
                    top: 0,
                    bottom: 0,
                  }}
                >
                  <div className="bg-orange px-1.5 py-0.5">
                    <span className="font-mono text-[10px] uppercase tracking-widest text-paper">
                      Auj.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* TASK DETAIL DRAWER */}
      {selectedTask && (
        <div
          className="fixed inset-0 z-50 flex justify-end"
          onClick={() => setSelectedTask(null)}
        >
          <div
            className="w-full max-w-sm h-full bg-paper border-l border-rule overflow-y-auto p-6"
            style={{ boxShadow: 'none' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-start justify-between mb-6">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">
                  Détail tâche
                </p>
                <h2
                  className="font-serif text-[22px] text-ink leading-tight"
                  style={{ fontVariationSettings: '"opsz" 60' }}
                >
                  {selectedTask.taskTitle}
                </h2>
              </div>
              <button
                onClick={() => setSelectedTask(null)}
                className="font-mono text-[20px] text-ink-3 hover:text-ink transition-colors leading-none"
              >
                ×
              </button>
            </div>

            <div className="space-y-4 divide-y divide-rule-soft">
              <div className="pb-4">
                <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">
                  Période
                </p>
                <p className="text-[14px] text-ink">
                  {new Date(selectedTask.startDate).toLocaleDateString('fr-FR')} →{' '}
                  {new Date(selectedTask.endDate).toLocaleDateString('fr-FR')}
                </p>
              </div>
              {selectedTask.requiredRole && (
                <div className="py-4">
                  <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">
                    Corps de métier
                  </p>
                  <p className="text-[14px] text-ink">{selectedTask.requiredRole}</p>
                </div>
              )}
              {selectedTask.assignedWorkerId &&
                workerMap.get(selectedTask.assignedWorkerId) && (
                  <div className="py-4">
                    <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">
                      Assigné à
                    </p>
                    <p className="text-[14px] text-ink">
                      {workerMap.get(selectedTask.assignedWorkerId)?.name}
                    </p>
                  </div>
                )}
              <div className="py-4">
                <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-1">
                  Statut
                </p>
                <div className="flex items-center gap-2">
                  <div
                    className="w-2.5 h-2.5"
                    style={{
                      background:
                        getTaskStyle(selectedTask).bg ||
                        getTaskStyle(selectedTask).border ||
                        '#8A8478',
                    }}
                  />
                  <p className="text-[14px] text-ink capitalize">
                    {selectedTask.status || 'À planifier'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
