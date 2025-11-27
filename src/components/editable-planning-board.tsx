'use client';

import { useEffect, useMemo, useState, useTransition } from 'react';
import {
  DndContext,
  PointerSensor,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Clock, Edit3 } from 'lucide-react';
import { updateTaskPlanningAction } from '@/app/actions/update-task-planning';
import { MAX_WORKING_HOURS_PER_DAY, LUNCH_BREAK_DURATION_HOURS } from '@/lib/ai/work-rules';

type Worker = {
  id: string;
  name: string;
  role: string | null;
};

type PlanningTask = {
  taskId: string;
  taskTitle: string;
  order: number;
  startDate: string;
  endDate: string;
  durationHours: number;
  assignedWorkerId: string | null;
  priority: 'high' | 'medium' | 'low';
};

type Props = {
  siteId: string;
  initialPlanning: PlanningTask[];
  workers: Worker[];
};

type EditableTask = PlanningTask;

type TaskWithDistributedHours = EditableTask & {
  hoursForDay: number; // Heures distribuées pour ce jour spécifique
  dayStartTime: string; // Heure de début pour ce jour (format HH:mm)
  dayEndTime: string; // Heure de fin pour ce jour (format HH:mm)
};

function getDayKey(date: string) {
  return new Date(date).toISOString().split('T')[0];
}

function DroppableColumn({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  const { setNodeRef, isOver } = useDroppable({ id });
  return (
    <div
      ref={setNodeRef}
      className={`flex-1 rounded-2xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900 ${
        isOver ? 'ring-2 ring-emerald-400' : ''
      }`}
    >
      {children}
    </div>
  );
}

function SortableTaskCard({
  task,
  onEdit,
}: {
  task: TaskWithDistributedHours;
  onEdit: (task: EditableTask) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({
    id: task.taskId,
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={`rounded-lg border px-3 py-2 text-xs shadow-sm ${
        task.priority === 'high'
          ? 'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-900/60 dark:bg-rose-900/20 dark:text-rose-200'
          : task.priority === 'medium'
            ? 'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-900/20 dark:text-amber-200'
            : 'border-zinc-200 bg-white text-zinc-900 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-semibold">{task.taskTitle}</p>
          <p className="mt-0.5 text-[10px] text-zinc-500 dark:text-zinc-400">
            {task.dayStartTime} - {task.dayEndTime}
          </p>
        </div>
        <button
          type="button"
          onClick={() => onEdit(task)}
          className="rounded-full p-1 text-zinc-400 transition hover:bg-white hover:text-zinc-700 dark:hover:bg-zinc-700"
        >
          <Edit3 className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="mt-1 flex items-center gap-1 text-[10px] text-zinc-500 dark:text-zinc-400">
        <Clock className="h-3 w-3" />
        {task.hoursForDay}h
        {task.durationHours > task.hoursForDay && (
          <span className="text-[9px] opacity-75">
            ({task.durationHours}h total)
          </span>
        )}
      </div>
    </div>
  );
}

function TaskEditModal({
  task,
  workers,
  onClose,
  onSave,
  isSaving,
}: {
  task: EditableTask | null;
  workers: Worker[];
  onClose: () => void;
  onSave: (payload: {
    startDate: string;
    endDate: string;
    durationHours: number;
    workerId: string | null;
  }) => void;
  isSaving: boolean;
}) {
  const [startInput, setStartInput] = useState('');
  const [duration, setDuration] = useState(8);
  const [workerId, setWorkerId] = useState<string | null>(null);

  useEffect(() => {
    if (task) {
      const start = new Date(task.startDate);
      const end = new Date(task.endDate);
      const diffHours = Math.max(1, Math.round((end.getTime() - start.getTime()) / 36e5));

      // eslint-disable-next-line react-hooks/exhaustive-deps
      setTimeout(() => {
        setStartInput(start.toISOString().slice(0, 16));
        setDuration(diffHours);
        setWorkerId(task.assignedWorkerId);
      }, 0);
    }
  }, [task]);

  if (!task) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">
          Modifier {task.taskTitle}
        </h3>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Ajustez la durée et les horaires puis sauvegardez.
        </p>

        <div className="mt-4 space-y-4 text-sm">
          <div className="space-y-1">
            <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-300">
              Début
            </label>
            <input
              type="datetime-local"
              value={startInput}
              onChange={(e) => setStartInput(e.target.value)}
              className="w-full rounded-lg border border-zinc-200 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-800"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-300">
              Durée (heures)
            </label>
            <input
              type="number"
              min={1}
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full rounded-lg border border-zinc-200 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-800"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-zinc-600 dark:text-zinc-300">
              Employé assigné
            </label>
            <select
              value={workerId ?? ''}
              onChange={(e) => setWorkerId(e.target.value || null)}
              className="w-full rounded-lg border border-zinc-200 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-800"
            >
              <option value="">Non assigné</option>
              {workers.map((worker) => (
                <option key={worker.id} value={worker.id}>
                  {worker.name} {worker.role ? `(${worker.role})` : ''}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-6 flex items-center justify-end gap-2 text-sm">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-zinc-200 px-4 py-2 text-zinc-600 transition hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            Annuler
          </button>
          <button
            type="button"
            disabled={isSaving}
            onClick={() => {
              if (!startInput) return;
              const startDate = new Date(startInput);
              const endDate = new Date(startDate.getTime() + duration * 60 * 60 * 1000);
              onSave({
                startDate: startDate.toISOString(),
                endDate: endDate.toISOString(),
                durationHours: duration,
                workerId,
              });
            }}
            className="rounded-lg bg-emerald-600 px-4 py-2 font-semibold text-white transition hover:bg-emerald-700 disabled:opacity-50"
          >
            {isSaving ? 'Sauvegarde…' : 'Enregistrer'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function EditablePlanningBoard({ siteId, initialPlanning, workers }: Props) {
  const [tasks, setTasks] = useState<EditableTask[]>(initialPlanning);
  const [editingTask, setEditingTask] = useState<EditableTask | null>(null);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    setTasks(initialPlanning);
  }, [initialPlanning]);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
  );

  const weekDays = useMemo(() => {
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    const dayOfWeek = start.getDay();
    const mondayOffset = dayOfWeek === 0 ? -6 : 1;
    start.setDate(start.getDate() - dayOfWeek + mondayOffset);

    return Array.from({ length: 7 }).map((_, index) => {
      const date = new Date(start);
      date.setDate(start.getDate() + index);
      return date;
    });
  }, []);

  // Fonction pour calculer les heures de début et de fin pour une journée
  const calculateDayHours = (
    hoursForDay: number,
    defaultStartHour: string = '07:00',
  ): { startTime: string; endTime: string } => {
    if (hoursForDay <= 0) {
      return { startTime: defaultStartHour, endTime: defaultStartHour };
    }

    const [startHour, startMin] = defaultStartHour.split(':').map(Number);
    const startMinutes = startHour * 60 + startMin;

    // Si 5h ou moins, tout le matin (pas besoin de pause déjeuner)
    if (hoursForDay <= 5) {
      const endMinutes = startMinutes + hoursForDay * 60;
      const endHour = Math.floor(endMinutes / 60);
      const endMin = endMinutes % 60;
      return {
        startTime: `${String(startHour).padStart(2, '0')}:${String(startMin).padStart(2, '0')}`,
        endTime: `${String(endHour).padStart(2, '0')}:${String(endMin).padStart(2, '0')}`,
      };
    }

    // Si plus de 5h, inclure la pause déjeuner de 1h
    // Exemple pour 8h effectives: 7h-12h (5h) + pause 12h-13h (1h) + 13h-17h (4h) = 8h effectives, fin à 17h
    // Structure: matin jusqu'à 12h (5h), pause 1h, puis après-midi
    const morningHours = 5; // 7h-12h = 5h
    const afternoonHours = hoursForDay - morningHours; // Reste pour l'après-midi
    
    // Heure de fin = 13h (après pause) + heures après-midi
    const afternoonStartMinutes = 13 * 60; // 13h00
    const endMinutes = afternoonStartMinutes + afternoonHours * 60;
    const endHour = Math.floor(endMinutes / 60);
    const endMin = endMinutes % 60;
    
    return {
      startTime: `${String(startHour).padStart(2, '0')}:${String(startMin).padStart(2, '0')}`,
      endTime: `${String(endHour).padStart(2, '0')}:${String(endMin).padStart(2, '0')}`,
    };
  };

  // Fonction pour distribuer les heures sur plusieurs jours
  const distributeHoursForDay = (
    totalHours: number,
    startDate: Date,
    endDate: Date,
    currentDay: Date,
  ): number => {
    if (totalHours <= MAX_WORKING_HOURS_PER_DAY) {
      // Si la tâche fait 8h ou moins, tout est sur le premier jour
      if (currentDay.toDateString() === startDate.toDateString()) {
        return totalHours;
      }
      return 0;
    }

    // Calculer le nombre de jours entre startDate et endDate (inclus)
    const daysDiff = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)) + 1;
    
    // Calculer le jour actuel dans la séquence (0 = premier jour)
    const currentDayIndex = Math.floor((currentDay.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
    
    if (currentDayIndex < 0 || currentDayIndex >= daysDiff) {
      return 0;
    }

    // Distribuer les heures : 8h par jour sauf le dernier jour
    if (currentDayIndex < daysDiff - 1) {
      return MAX_WORKING_HOURS_PER_DAY;
    } else {
      // Dernier jour : reste des heures
      const remainingHours = totalHours - (MAX_WORKING_HOURS_PER_DAY * (daysDiff - 1));
      return Math.max(0, remainingHours);
    }
  };

  const tasksByDay = useMemo(() => {
    const map: Record<string, TaskWithDistributedHours[]> = {};
    
    weekDays.forEach((day) => {
      const key = day.toISOString().split('T')[0];
      const dayDate = new Date(day);
      dayDate.setHours(0, 0, 0, 0);
      
      const tasksForDay: TaskWithDistributedHours[] = [];
      
      tasks.forEach((task) => {
        const taskStart = new Date(task.startDate);
        taskStart.setHours(0, 0, 0, 0);
        const taskEnd = new Date(task.endDate);
        taskEnd.setHours(0, 0, 0, 0);
        
        // Vérifier si le jour est dans la plage de la tâche (inclus)
        if (dayDate >= taskStart && dayDate <= taskEnd) {
          const hoursForDay = distributeHoursForDay(
            task.durationHours,
            taskStart,
            taskEnd,
            dayDate,
          );
          
          if (hoursForDay > 0) {
            // Déterminer l'heure de début par défaut selon le rôle
            // Pour l'instant, on utilise 07:00 par défaut (maçon, charpentier, etc.)
            // On pourrait améliorer en utilisant getWorkRule(task.required_role)
            const defaultStartHour = '07:00';
            
            // Calculer les heures de début et de fin pour ce jour
            const { startTime, endTime } = calculateDayHours(hoursForDay, defaultStartHour);
            
            tasksForDay.push({
              ...task,
              hoursForDay,
              dayStartTime: startTime,
              dayEndTime: endTime,
            });
          }
        }
      });
      
      map[key] = tasksForDay;
    });
    
    return map;
  }, [tasks, weekDays]);

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over) return;

    const taskId = String(active.id);
    const task = tasks.find((t) => t.taskId === taskId);
    if (!task) return;

    let targetDay: string | null = null;

    if (String(over.id).startsWith('day:')) {
      targetDay = String(over.id).replace('day:', '');
    } else {
      const overTask = tasks.find((t) => t.taskId === over.id);
      if (overTask) {
        targetDay = getDayKey(overTask.startDate);
      }
    }

    if (!targetDay) return;

    const currentDay = getDayKey(task.startDate);
    if (currentDay === targetDay) return;

    const newStart = new Date(targetDay);
    const startTime = new Date(task.startDate);
    newStart.setHours(startTime.getHours(), startTime.getMinutes(), 0, 0);

    const newEnd = new Date(newStart.getTime() + task.durationHours * 60 * 60 * 1000);

    setTasks((prev) =>
      prev.map((t) =>
        t.taskId === taskId
          ? { ...t, startDate: newStart.toISOString(), endDate: newEnd.toISOString() }
          : t,
      ),
    );

    startTransition(() =>
      updateTaskPlanningAction({
        siteId,
        taskId,
        plannedStart: newStart.toISOString(),
        plannedEnd: newEnd.toISOString(),
        workerId: task.assignedWorkerId,
        plannedOrder: task.order,
      }),
    );
  };

  const dayNames = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'];

  return (
    <div className="space-y-4">
      <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
        <div className="flex flex-col gap-4 md:flex-row">
          {weekDays.map((day, index) => {
            const key = day.toISOString().split('T')[0];
            const tasksForDay = tasksByDay[key] ?? [];
            const isToday = new Date().toDateString() === day.toDateString();

            return (
              <DroppableColumn key={key} id={`day:${key}`}>
                <div className="mb-3 flex items-center justify-between text-xs">
                  <div>
                    <p
                      className={`font-semibold ${
                        isToday ? 'text-emerald-600' : 'text-zinc-600 dark:text-zinc-300'
                      }`}
                    >
                      {dayNames[index]}
                    </p>
                    <p className="text-zinc-400 dark:text-zinc-500">{day.getDate()}</p>
                  </div>
                </div>

                {tasksForDay.length > 0 ? (
                  <SortableContext
                    items={tasksForDay.map((task) => task.taskId)}
                    strategy={verticalListSortingStrategy}
                  >
                    <div className="space-y-2">
                      {tasksForDay.map((task) => (
                        <SortableTaskCard key={task.taskId} task={task} onEdit={setEditingTask} />
                      ))}
                    </div>
                  </SortableContext>
                ) : (
                  <div className="rounded-lg border border-dashed border-zinc-200 p-4 text-center text-xs text-zinc-400 dark:border-zinc-700 dark:text-zinc-500">
                    Glissez une tâche ici
                  </div>
                )}
              </DroppableColumn>
            );
          })}
        </div>
      </DndContext>

      {editingTask && (
        <TaskEditModal
          task={editingTask}
          workers={workers}
          isSaving={isPending}
          onClose={() => setEditingTask(null)}
          onSave={({ startDate, endDate, durationHours, workerId }) => {
            setEditingTask(null);
            setTasks((prev) =>
              prev.map((task) =>
                task.taskId === editingTask.taskId
                  ? {
                      ...task,
                      startDate,
                      endDate,
                      durationHours,
                      assignedWorkerId: workerId,
                    }
                  : task,
              ),
            );

            startTransition(() =>
              updateTaskPlanningAction({
                siteId,
                taskId: editingTask.taskId,
                plannedStart: startDate,
                plannedEnd: endDate,
                durationHours,
                workerId,
                plannedOrder: editingTask.order,
              }),
            );
          }}
        />
      )}
    </div>
  );
}

