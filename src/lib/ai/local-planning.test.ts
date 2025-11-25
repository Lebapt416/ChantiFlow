import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { generateLocalAIPlanning } from './local-planning';

describe('generateLocalAIPlanning', () => {
  const mockTasks = [
    {
      id: 'task-1',
      title: 'Fondation',
      required_role: 'maçon',
      duration_hours: 8,
      status: 'pending' as const,
    },
    {
      id: 'task-2',
      title: 'Structure',
      required_role: 'charpentier',
      duration_hours: 16,
      status: 'pending' as const,
    },
    {
      id: 'task-3',
      title: 'Peinture',
      required_role: 'peintre',
      duration_hours: 8,
      status: 'pending' as const,
    },
  ];

  const mockWorkers = [
    {
      id: 'worker-1',
      name: 'Jean Maçon',
      email: 'jean@example.com',
      role: 'maçon',
    },
    {
      id: 'worker-2',
      name: 'Pierre Charpentier',
      email: 'pierre@example.com',
      role: 'charpentier',
    },
  ];

  beforeEach(() => {
    // Mock de la date pour des tests prévisibles
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('devrait générer un planning vide si aucune tâche', async () => {
    const result = await generateLocalAIPlanning(
      [],
      mockWorkers,
      null,
      'Test Site',
    );

    expect(result.orderedTasks).toHaveLength(0);
    expect(result.warnings).toContain('Aucune tâche à planifier');
  });

  it('devrait respecter les dépendances entre tâches', async () => {
    const result = await generateLocalAIPlanning(
      mockTasks,
      mockWorkers,
      null,
      'Test Site',
    );

    // La peinture doit venir après la structure
    const structureTask = result.orderedTasks.find((t) => t.taskId === 'task-2');
    const peintureTask = result.orderedTasks.find((t) => t.taskId === 'task-3');

    expect(structureTask).toBeDefined();
    expect(peintureTask).toBeDefined();

    if (structureTask && peintureTask) {
      const structureEnd = new Date(structureTask.endDate);
      const peintureStart = new Date(peintureTask.startDate);
      
      // La peinture doit commencer après la fin de la structure
      expect(peintureStart.getTime()).toBeGreaterThanOrEqual(structureEnd.getTime());
    }
  });

  it('devrait éviter les week-ends dans le planning', async () => {
    // Définir une date de début un vendredi
    const friday = new Date('2024-01-05'); // Vendredi 5 janvier 2024
    vi.setSystemTime(friday);

    const result = await generateLocalAIPlanning(
      mockTasks,
      mockWorkers,
      null,
      'Test Site',
    );

    // Vérifier qu'aucune tâche ne commence un samedi ou dimanche
    result.orderedTasks.forEach((task) => {
      const startDate = new Date(task.startDate);
      const dayOfWeek = startDate.getDay(); // 0 = Dimanche, 6 = Samedi
      
      expect(dayOfWeek).not.toBe(0); // Pas de dimanche
      expect(dayOfWeek).not.toBe(6); // Pas de samedi
    });
  });

  it('devrait répartir les tâches longues sur plusieurs jours', async () => {
    const longTask = {
      id: 'task-long',
      title: 'Tâche longue',
      required_role: null,
      duration_hours: 20, // Plus de 8h, doit être répartie
      status: 'pending' as const,
    };

    const result = await generateLocalAIPlanning(
      [longTask],
      mockWorkers,
      null,
      'Test Site',
    );

    expect(result.warnings.some((w) => w.includes('répartie sur'))).toBe(true);
    
    const task = result.orderedTasks.find((t) => t.taskId === 'task-long');
    expect(task).toBeDefined();
    
    if (task) {
      const start = new Date(task.startDate);
      const end = new Date(task.endDate);
      const daysDiff = Math.ceil((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
      
      // 20h / 8h par jour = 3 jours minimum
      expect(daysDiff).toBeGreaterThanOrEqual(3);
    }
  });

  it('devrait assigner les workers selon leur rôle', async () => {
    const result = await generateLocalAIPlanning(
      mockTasks,
      mockWorkers,
      null,
      'Test Site',
    );

    const maçonTask = result.orderedTasks.find(
      (t) => t.taskId === 'task-1',
    );
    expect(maçonTask?.assignedWorkerId).toBe('worker-1'); // Maçon assigné à la tâche maçon

    const charpentierTask = result.orderedTasks.find(
      (t) => t.taskId === 'task-2',
    );
    expect(charpentierTask?.assignedWorkerId).toBe('worker-2'); // Charpentier assigné
  });

  it('devrait générer un warning si la deadline est irréaliste', async () => {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);

    const result = await generateLocalAIPlanning(
      mockTasks,
      mockWorkers,
      tomorrow.toISOString().split('T')[0],
      'Test Site',
    );

    expect(result.warnings.some((w) => w.includes('deadline'))).toBe(true);
  });

  it('devrait générer un raisonnement explicatif', async () => {
    const result = await generateLocalAIPlanning(
      mockTasks,
      mockWorkers,
      null,
      'Test Site',
    );

    expect(result.reasoning).toBeDefined();
    expect(result.reasoning.length).toBeGreaterThan(0);
    expect(result.reasoning).toContain('analysé');
  });
});

