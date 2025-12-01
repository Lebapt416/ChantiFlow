'use client';

import { useEffect, useMemo, useState, useTransition } from 'react';
import {
  DndContext,
  DragOverlay,
  PointerSensor,
  useDroppable,
  useSensor,
  useSensors,
  defaultDropAnimationSideEffects,
  type DragEndEvent,
  type DragStartEvent,
  type Modifier,
} from '@dnd-kit/core';
import {
  SortableContext,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { AlertTriangle, CalendarDays, ChevronLeft, ChevronRight, Clock, Edit3, MoreHorizontal } from 'lucide-react';
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
  assignedWorkerIds?: string[];
  priority: 'high' | 'medium' | 'low';
  status?: string | null;
  requiredRole?: string | null;
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
  uniqueId: string; // ID unique pour cette instance (taskId::dayKey)
  dayKey: string; // Clé du jour (YYYY-MM-DD)
};

function getDayKey(date: string) {
  return new Date(date).toISOString().split('T')[0];
}

/**
 * Modificateur personnalisé pour centrer la carte sur le curseur dans le DragOverlay
 */
const snapCenterToCursor: Modifier = ({ transform, draggingNodeRect }) => {
  if (!draggingNodeRect) {
    return transform;
  }

  return {
    ...transform,
    x: transform.x - draggingNodeRect.width / 2,
    y: transform.y - draggingNodeRect.height / 2,
  };
};

/**
 * Calcule le pourcentage d'avancement théorique d'une tâche en temps réel,
 * en tenant compte de la pause déjeuner française (12h-13h).
 * @param dayStartTime Heure de début au format "HH:mm" (ex: "08:00")
 * @param hoursForDay Durée de travail effectif en heures pour ce jour
 * @returns Pourcentage d'avancement théorique (0-100)
 */
function calculateTheoreticalProgress(dayStartTime: string, hoursForDay: number): number {
  const now = new Date();
  // Note: Dans une vraie app, vérifier si la colonne correspond à "Aujourd'hui". Ici on assume pour la démo.

  const [startH, startM] = dayStartTime.split(':').map(Number);
  const startMinutes = startH * 60 + startM;
  const currentMinutes = now.getHours() * 60 + now.getMinutes();

  // Règle Pause Déjeuner : 12h00 (720min) - 13h00 (780min)
  const LUNCH_START = 12 * 60;
  const LUNCH_END = 13 * 60;

  if (currentMinutes < startMinutes) return 0;

  // Calcul du temps écoulé effectif (hors pause)
  let effectiveElapsed = 0;

  if (startMinutes < LUNCH_START) {
    // Si commence le matin
    if (currentMinutes < LUNCH_START) {
      effectiveElapsed = currentMinutes - startMinutes;
    } else if (currentMinutes < LUNCH_END) {
      effectiveElapsed = LUNCH_START - startMinutes; // Gelé pendant la pause
    } else {
      effectiveElapsed = (LUNCH_START - startMinutes) + (currentMinutes - LUNCH_END);
    }
  } else if (startMinutes >= LUNCH_END) {
    // Si commence l'après-midi
    effectiveElapsed = currentMinutes - startMinutes;
  } else {
    // Si commence pendant la pause
    effectiveElapsed = Math.max(0, currentMinutes - LUNCH_END);
  }

  const totalMinutes = hoursForDay * 60;
  const progress = Math.min(100, Math.max(0, (effectiveElapsed / totalMinutes) * 100));
  return Math.round(progress);
}

function DroppableColumn({
  id,
  children,
  isToday = false,
  label,
  dayNumber,
}: {
  id: string;
  children: React.ReactNode;
  isToday?: boolean;
  label?: string;
  dayNumber?: number;
}) {
  const { setNodeRef, isOver } = useDroppable({ id });
  return (
    <div
      ref={setNodeRef}
      className={`flex-1 rounded-2xl border border-zinc-200/50 bg-zinc-50/50 p-4 shadow-sm backdrop-blur-sm transition ${
        isOver ? 'ring-2 ring-emerald-400/60' : ''
      } dark:border-zinc-700/30 dark:bg-zinc-900/30`}
    >
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          {label}
        </h3>
        <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-zinc-600 shadow-sm dark:bg-zinc-800 dark:text-zinc-300">
          {dayNumber}
        </span>
      </div>
      <div className="space-y-2">{children}</div>
      <button
        type="button"
        className="mt-3 w-full rounded-lg border border-dashed border-zinc-300/50 px-3 py-2 text-[11px] text-zinc-400 transition-all hover:border-emerald-300 hover:bg-white/50 hover:text-emerald-600 dark:border-zinc-700/50 dark:hover:bg-zinc-800/50"
      >
        + Ajouter une tâche
      </button>
    </div>
  );
}

function SortableTaskCard({
  task,
  onEdit,
  workerLookup,
  isOverlay = false,
}: {
  task: TaskWithDistributedHours;
  onEdit: (task: EditableTask) => void;
  workerLookup: Map<string, Worker>;
  isOverlay?: boolean;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: task.uniqueId || task.taskId,
  });

  const style = isOverlay
    ? undefined
    : {
        transform: CSS.Transform.toString(transform),
        transition,
      };

  const statusKey = (() => {
    const normalized = (task.status ?? '').toLowerCase();
    if (['done', 'completed', 'terminé', 'validé', 'finished'].includes(normalized)) return 'done';
    if (['in_progress', 'progress', 'en cours'].includes(normalized)) return 'in_progress';
    return 'todo';
  })();

  const statusStyles: Record<'todo' | 'in_progress' | 'done', { border: string; progress: number; progressColor: string }> = {
    todo: { border: 'border-l-zinc-300', progress: 0, progressColor: 'bg-zinc-300 dark:bg-zinc-600' },
    in_progress: { border: 'border-l-blue-500', progress: 50, progressColor: 'bg-blue-500' },
    done: { border: 'border-l-emerald-500', progress: 100, progressColor: 'bg-emerald-500' },
  };

  const config = statusStyles[statusKey];

  const endDate = new Date(task.endDate);
  const isOverdue = statusKey !== 'done' && endDate.getTime() < Date.now();

  const badgeLabel = task.requiredRole || 'Polyvalent';

  const workerIds =
    task.assignedWorkerIds?.length && task.assignedWorkerIds.length > 0
      ? task.assignedWorkerIds
      : task.assignedWorkerId
        ? [task.assignedWorkerId]
        : [];
  const assignedWorkers = workerIds
    .map((id) => workerLookup.get(id))
    .filter((w): w is Worker => Boolean(w));

  // Calcul de la progression théorique en temps réel
  const [theoreticalProgress, setTheoreticalProgress] = useState(() =>
    calculateTheoreticalProgress(task.dayStartTime, task.hoursForDay),
  );

  // Mise à jour de la progression toutes les minutes
  useEffect(() => {
    const updateProgress = () => {
      setTheoreticalProgress(calculateTheoreticalProgress(task.dayStartTime, task.hoursForDay));
    };

    updateProgress(); // Mise à jour immédiate
    const interval = setInterval(updateProgress, 60000); // Mise à jour toutes les minutes

    return () => clearInterval(interval);
  }, [task.dayStartTime, task.hoursForDay]);

  // Déterminer la couleur de la barre selon le pourcentage
  const getProgressColor = () => {
    if (theoreticalProgress < 10) return 'bg-zinc-300 dark:bg-zinc-600';
    if (theoreticalProgress < 90) return 'bg-blue-500';
    return 'bg-emerald-500';
  };

  const progressColor = getProgressColor();

  // Couleur de l'indicateur de statut basé sur la priorité
  const priorityColors: Record<'high' | 'medium' | 'low', { bg: string; shadow: string }> = {
    high: { bg: 'bg-rose-500', shadow: 'shadow-rose-500/50' },
    medium: { bg: 'bg-amber-500', shadow: 'shadow-amber-500/50' },
    low: { bg: 'bg-emerald-500', shadow: 'shadow-emerald-500/50' },
  };

  const priorityConfig = priorityColors[task.priority || 'medium'];

  // Styles conditionnels selon le mode
  const getCardClassName = () => {
    if (isOverlay) {
      // Mode overlay : carte qui flotte
      return 'group relative overflow-hidden rounded-xl border border-white/20 bg-white/100 p-4 shadow-2xl ring-1 ring-emerald-500/50 rotate-2 scale-105 cursor-grabbing dark:border-zinc-700/50 dark:bg-zinc-900/100';
    }
    if (isDragging) {
      // Mode dragging : carte originale (fantôme)
      return 'group relative overflow-hidden rounded-xl border-0 bg-white/30 p-4 shadow-none backdrop-blur-md transition-all duration-300 dark:bg-zinc-900/30 opacity-30';
    }
    // Mode normal
    return 'group relative overflow-hidden rounded-xl border border-white/20 bg-white/80 p-4 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg dark:border-zinc-700/50 dark:bg-zinc-900/80';
  };

  return (
    <div
      ref={isOverlay ? undefined : setNodeRef}
      style={style}
      {...(isOverlay ? {} : { ...attributes, ...listeners })}
      className={getCardClassName()}
    >
      {/* Indicateur de statut latéral (néon) */}
      <div
        className={`absolute left-0 top-0 bottom-0 w-1 ${priorityConfig.bg} ${priorityConfig.shadow} shadow-lg`}
      />

      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 space-y-2">
          <p className="text-sm font-bold text-zinc-900 dark:text-zinc-100">{task.taskTitle}</p>

          {/* Métadonnées & Alertes */}
          {isOverdue ? (
            <div className="flex items-center gap-1 rounded border border-red-500/20 bg-red-500/10 px-2 py-1 text-[10px] font-bold text-red-600 dark:text-red-400">
              <AlertTriangle className="h-3 w-3" />
              <span>En retard</span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-1 rounded-md bg-zinc-100/50 px-2 py-1 text-[10px] font-medium text-zinc-600 dark:bg-zinc-800/50 dark:text-zinc-400">
              <CalendarDays className="h-3 w-3" />
              <span>
                {task.dayStartTime} - {task.dayEndTime}
              </span>
            </div>
          )}
        </div>

        {/* Bouton Options (apparaît au hover) */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onEdit(task);
          }}
          className="opacity-0 transition-opacity group-hover:opacity-100"
        >
          <MoreHorizontal className="h-4 w-4 text-zinc-400 hover:text-zinc-600 dark:text-zinc-500 dark:hover:text-zinc-300" />
        </button>
      </div>

      {/* Barre de progression temps réel (Footer) */}
      <div className="mt-4 space-y-1.5">
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200/50 dark:bg-zinc-700/50">
          <div
            className={`h-full rounded-full transition-all duration-1000 ${progressColor}`}
            style={{ width: `${theoreticalProgress}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-[10px] text-zinc-500 dark:text-zinc-400">
          <span>{theoreticalProgress}%</span>
          <span className="text-zinc-400 dark:text-zinc-500">
            {task.hoursForDay}h / {task.durationHours}h
          </span>
        </div>
      </div>

      {/* Avatar en bas à droite */}
      {assignedWorkers.length > 0 && (
        <div className="absolute bottom-3 right-3">
          <div className="h-7 w-7 rounded-full border-2 border-white bg-gradient-to-br from-emerald-400 to-emerald-600 text-[10px] font-semibold uppercase text-white shadow-lg shadow-emerald-500/30 dark:border-zinc-900">
            <div className="flex h-full w-full items-center justify-center">
              {assignedWorkers[0]?.name ? assignedWorkers[0].name.slice(0, 2) : 'OU'}
            </div>
          </div>
        </div>
      )}
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
  const [activeTask, setActiveTask] = useState<TaskWithDistributedHours | null>(null);
  const [isPending, startTransition] = useTransition();

  // State pour la navigation de semaine
  const [currentWeekStart, setCurrentWeekStart] = useState(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const dayOfWeek = today.getDay();
    const mondayOffset = dayOfWeek === 0 ? -6 : 1;
    const monday = new Date(today);
    monday.setDate(today.getDate() - dayOfWeek + mondayOffset);
    return monday;
  });

  useEffect(() => {
    setTasks(initialPlanning);
  }, [initialPlanning]);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
  );

  const workerLookup = useMemo(() => new Map<string, Worker>(workers.map((worker) => [worker.id, worker])), [workers]);

  const weekDays = useMemo(() => {
    const start = new Date(currentWeekStart);
    start.setHours(0, 0, 0, 0);

    return Array.from({ length: 7 }).map((_, index) => {
      const date = new Date(start);
      date.setDate(start.getDate() + index);
      return date;
    });
  }, [currentWeekStart]);

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
            
            // Générer un ID unique pour cette instance (taskId::dayKey)
            const uniqueId = `${task.taskId}::${key}`;
            
            tasksForDay.push({
              ...task,
              hoursForDay,
              dayStartTime: startTime,
              dayEndTime: endTime,
              uniqueId,
              dayKey: key,
            });
          }
        }
      });
      
      map[key] = tasksForDay;
    });
    
    return map;
  }, [tasks, weekDays]);

  const handleDragStart = (event: DragStartEvent) => {
    const { active } = event;
    const uniqueId = String(active.id);
    
    // Trouver la tâche dans tasksByDay en utilisant uniqueId
    for (const dayTasks of Object.values(tasksByDay)) {
      const foundTask = dayTasks.find((t) => t.uniqueId === uniqueId);
      if (foundTask) {
        setActiveTask(foundTask);
        return;
      }
    }
  };

  const handleDragEnd = (event: DragEndEvent) => {
    setActiveTask(null); // Réinitialiser la tâche active
    const { active, over } = event;
    if (!over) return;

    // Parser l'ID unique pour retrouver la vraie tâche
    const uniqueId = String(active.id);
    const [originalTaskId] = uniqueId.split('::');
    const task = tasks.find((t) => t.taskId === originalTaskId);
    if (!task) return;

    let targetDay: string | null = null;

    if (String(over.id).startsWith('day:')) {
      targetDay = String(over.id).replace('day:', '');
    } else {
      // Parser l'ID unique de la carte cible
      const overUniqueId = String(over.id);
      const [overTaskId] = overUniqueId.split('::');
      const overTask = tasks.find((t) => t.taskId === overTaskId);
      if (overTask) {
        // Trouver le dayKey de la carte cible dans tasksByDay
        for (const [dayKey, dayTasks] of Object.entries(tasksByDay)) {
          const foundTask = dayTasks.find((t) => t.uniqueId === overUniqueId);
          if (foundTask) {
            targetDay = dayKey;
            break;
          }
        }
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
        t.taskId === originalTaskId
          ? { ...t, startDate: newStart.toISOString(), endDate: newEnd.toISOString() }
          : t,
      ),
    );

    startTransition(() =>
      updateTaskPlanningAction({
        siteId,
        taskId: originalTaskId,
        plannedStart: newStart.toISOString(),
        plannedEnd: newEnd.toISOString(),
        workerId: task.assignedWorkerId,
        plannedOrder: task.order,
      }),
    );
  };

  const dayNames = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'];

  // Fonctions de navigation de semaine
  const goToPreviousWeek = () => {
    const newWeekStart = new Date(currentWeekStart);
    newWeekStart.setDate(currentWeekStart.getDate() - 7);
    setCurrentWeekStart(newWeekStart);
  };

  const goToNextWeek = () => {
    const newWeekStart = new Date(currentWeekStart);
    newWeekStart.setDate(currentWeekStart.getDate() + 7);
    setCurrentWeekStart(newWeekStart);
  };

  const goToToday = () => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const dayOfWeek = today.getDay();
    const mondayOffset = dayOfWeek === 0 ? -6 : 1;
    const monday = new Date(today);
    monday.setDate(today.getDate() - dayOfWeek + mondayOffset);
    setCurrentWeekStart(monday);
  };

  // Configuration de l'animation de drop
  const dropAnimation = {
    sideEffects: defaultDropAnimationSideEffects({
      styles: {
        active: { opacity: '0.5' },
      },
    }),
  };

  return (
    <div className="relative space-y-4 overflow-hidden rounded-3xl border border-white/10 bg-white/40 p-4 shadow-inner backdrop-blur-3xl dark:bg-zinc-900/40">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.15),_transparent_40%),_radial-gradient(circle_at_bottom,_rgba(16,185,129,0.15),_transparent_45%)]" />
      
      {/* Barre de navigation Liquid Glass */}
      <div className="flex items-center justify-center">
        <div className="flex items-center gap-4 rounded-full border border-white/20 bg-white/40 px-4 py-2 text-sm font-medium text-zinc-700 shadow-sm backdrop-blur-md dark:bg-zinc-800/40 dark:text-zinc-200">
          <button
            type="button"
            onClick={goToPreviousWeek}
            className="rounded-full p-1.5 transition hover:bg-white/50 dark:hover:bg-zinc-700/50"
            aria-label="Semaine précédente"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={goToToday}
            className="rounded-full px-3 py-1.5 text-xs transition hover:bg-white/50 dark:hover:bg-zinc-700/50"
          >
            Aujourd'hui
          </button>
          <button
            type="button"
            onClick={goToNextWeek}
            className="rounded-full p-1.5 transition hover:bg-white/50 dark:hover:bg-zinc-700/50"
            aria-label="Semaine suivante"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      <DndContext sensors={sensors} onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
        <div className="flex flex-col gap-4 md:flex-row">
          {weekDays.map((day, index) => {
            const key = day.toISOString().split('T')[0];
            const tasksForDay = tasksByDay[key] ?? [];
            const isToday = new Date().toDateString() === day.toDateString();

            return (
              <DroppableColumn
                key={key}
                id={`day:${key}`}
                isToday={isToday}
                label={dayNames[index]}
                dayNumber={day.getDate()}
              >
                {tasksForDay.length > 0 ? (
                  <SortableContext
                    items={tasksForDay.map((task) => task.uniqueId)}
                    strategy={verticalListSortingStrategy}
                  >
                    <div>
                      {tasksForDay.map((task) => (
                        <SortableTaskCard
                          key={task.uniqueId}
                          task={task}
                          onEdit={setEditingTask}
                          workerLookup={workerLookup}
                        />
                      ))}
                    </div>
                  </SortableContext>
                ) : (
                  <div className="rounded-lg border border-dashed border-zinc-300/50 p-4 text-center text-xs text-zinc-400 dark:border-zinc-700/50 dark:text-zinc-500">
                    Glissez une tâche ici
                  </div>
                )}
              </DroppableColumn>
            );
          })}
        </div>
        <DragOverlay dropAnimation={dropAnimation} modifiers={[snapCenterToCursor]}>
          {activeTask ? (
            <div className="w-[280px]">
              <SortableTaskCard
                task={activeTask}
                onEdit={() => {}}
                workerLookup={workerLookup}
                isOverlay={true}
              />
            </div>
          ) : null}
        </DragOverlay>
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

