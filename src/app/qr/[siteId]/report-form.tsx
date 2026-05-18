"use client";

import { startTransition, useActionState, useEffect, useMemo, useState, useRef } from "react";
import { useFormStatus } from "react-dom";
import { submitReportAction, type ReportState } from './actions';
import { capitalizeRoleWords } from '@/lib/utils/role-formatting';
import { useOfflineSync } from '@/hooks/useOfflineSync';

const initialState: ReportState = {};

type TaskOption = {
  id: string;
  title: string;
};

type WorkerOption = {
  name: string | null;
  email: string | null;
  role: string | null;
};

type Props = {
  siteId: string;
  tasks: TaskOption[];
  workers: WorkerOption[];
};

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      className="w-full rounded border border-rule-soft py-2 text-sm font-semibold text-ink transition hover:bg-paper-2 disabled:cursor-not-allowed disabled:opacity-70 dark:border-rule dark:text-paper"
      disabled={pending}
    >
      {pending ? 'Envoi...' : 'Envoyer le rapport'}
    </button>
  );
}

export function ReportForm({ siteId, tasks, workers }: Props) {
  const [state, formAction] = useActionState(submitReportAction, initialState);
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('');
  const photoFileRef = useRef<File | null>(null);
  const { isOnline, savePendingReport, syncAllPendingReports } = useOfflineSync();

  const matchedWorker = useMemo(
    () =>
      workers.find(
        (worker) =>
          worker.email?.toLowerCase() === email.trim().toLowerCase() &&
          worker.email,
      ),
    [email, workers],
  );

  // Synchroniser automatiquement quand on revient en ligne
  useEffect(() => {
    if (isOnline) {
      syncAllPendingReports(submitReportAction);
    }
  }, [isOnline, syncAllPendingReports]);

  useEffect(() => {
    if (state?.success) {
      const form = document.getElementById('employee-report-form') as HTMLFormElement | null;
      form?.reset();
      photoFileRef.current = null;
      startTransition(() => {
        setEmail('');
        setName('');
        setRole('');
      });
    }
  }, [state?.success]);

  // Gérer la soumission du formulaire avec support offline
  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);

    const taskId = formData.get('taskId') as string;
    const emailValue = formData.get('email') as string;
    const nameValue = formData.get('name') as string;
    const roleValue = formData.get('role') as string;
    const description = formData.get('description') as string;
    const markDone = formData.get('mark_done') === 'on';
    const photo = photoFileRef.current;

    if (!taskId || !emailValue) {
      return;
    }

    // Si on est offline, sauvegarder localement avec priorité
    if (!isOnline) {
      try {
        // Déterminer la priorité: high si tâche marquée comme faite, medium sinon
        const priority: 'high' | 'medium' | 'low' = markDone ? 'high' : 'medium';
        
        await savePendingReport({
          siteId,
          taskId,
          email: emailValue.trim().toLowerCase(),
          name: nameValue.trim(),
          role: roleValue.trim(),
          description: description.trim(),
          photo: photo || undefined,
          markDone,
        }, priority);

        // Réinitialiser le formulaire
        form.reset();
        photoFileRef.current = null;
        startTransition(() => {
          setEmail('');
          setName('');
          setRole('');
        });

        // Afficher un message de succès local
        alert('✅ Rapport sauvegardé localement. Il sera synchronisé automatiquement dès que vous serez en ligne.');
      } catch (error) {
        console.error('Erreur lors de la sauvegarde locale:', error);
        alert('❌ Erreur lors de la sauvegarde locale du rapport.');
      }
      return;
    }

    // Si on est online, soumettre normalement
    formAction(formData);
  };

  return (
    <form
      id="employee-report-form"
      onSubmit={handleSubmit}
      className="space-y-4 rounded border border-zinc-200 bg-paper p-6  "
    >
      <input type="hidden" name="siteId" value={siteId} />
      <div className="space-y-2">
        <label
          className="text-sm font-semibold text-ink dark:text-paper"
          htmlFor="taskId"
        >
          Tâche
        </label>
        <select
          name="taskId"
          id="taskId"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
          required
        >
          <option value="">Sélectionne une tâche</option>
          {tasks.map((task) => (
            <option key={task.id} value={task.id}>
              {task.title}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-2">
        <label
          className="text-sm font-semibold text-ink dark:text-paper"
          htmlFor="email"
        >
          Email professionnel
        </label>
        <input
          id="email"
          name="email"
          type="email"
          placeholder="prenom@entreprise.com"
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
          required
          value={email}
          onChange={(event) => {
            const value = event.target.value;
            setEmail(value);
            const worker = workers.find(
              (candidate) =>
                candidate.email?.toLowerCase() === value.toLowerCase(),
            );
            if (worker) {
              setName(worker.name ?? '');
              setRole(worker.role ?? '');
            } else {
              setName('');
              setRole('');
            }
          }}
          list="worker-emails"
        />
        <datalist id="worker-emails">
          {workers
            .filter((worker) => worker.email)
            .map((worker) => (
              <option key={worker.email ?? ''} value={worker.email ?? ''}>
                {worker.name ?? ''}
              </option>
            ))}
        </datalist>
        {matchedWorker ? (
          <p className="text-xs text-green dark:text-green">
            {matchedWorker.name} reconnu ({matchedWorker.role ?? 'rôle inconnu'})
          </p>
        ) : (
          <p className="text-xs text-zinc-500 dark:text-zinc-300">
            Si tu n’apparais pas, complète les champs ci-dessous.
          </p>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <label
            className="text-sm font-semibold text-ink dark:text-paper"
            htmlFor="name"
          >
            Nom complet
          </label>
          <input
            id="name"
            name="name"
            placeholder="Camille Dupont"
            className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        <div className="space-y-2">
          <label
            className="text-sm font-semibold text-ink dark:text-paper"
            htmlFor="role"
          >
            Métier / rôle
          </label>
          <input
            id="role"
            name="role"
            placeholder="Électricien·ne"
            className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
            value={role}
            onChange={(event) => setRole(event.target.value)}
            onBlur={(event) => {
              const capitalized = capitalizeRoleWords(event.target.value);
              setRole(capitalized);
              event.target.value = capitalized;
            }}
          />
        </div>
      </div>

      <div className="space-y-2">
        <label
          className="text-sm font-semibold text-ink dark:text-paper"
          htmlFor="description"
        >
          Rapport texte
        </label>
        <textarea
          id="description"
          name="description"
          rows={4}
          placeholder="Décris brièvement l’avancement, les incidents, le matériel utilisé..."
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
        />
      </div>

      <div className="space-y-2">
        <label
          className="text-sm font-semibold text-ink dark:text-paper"
          htmlFor="photo"
        >
          Photo (optionnel)
        </label>
        <input
          id="photo"
          name="photo"
          type="file"
          accept="image/*"
          className="w-full rounded border border-dashed border-rule-soft px-3 py-2 text-sm text-ink-3 dark:border-rule dark:bg-ink dark:text-ink-2"
          onChange={(e) => {
            photoFileRef.current = e.target.files?.[0] || null;
          }}
        />
        <p className="text-xs text-ink-3">
          Formats acceptés: JPG, PNG. Taille max définie par Supabase Storage.
        </p>
      </div>

      <label className="flex items-center gap-2 text-sm text-ink-2">
        <input
          type="checkbox"
          name="mark_done"
          className="rounded border-rule-soft dark:border-rule"
        />
        Marquer la tâche comme terminée
      </label>

      <SubmitButton />

      {!isOnline && (
        <div className="rounded border border-warn bg-paper-2 p-3">
          <p className="text-sm font-semibold text-warn">
            ⚠️ Mode Hors-ligne
          </p>
          <p className="text-xs text-ink-2">
            Votre rapport sera sauvegardé localement et synchronisé automatiquement dès que vous serez en ligne.
          </p>
        </div>
      )}
      {state?.error ? (
        <p className="text-sm text-danger">{state.error}</p>
      ) : null}
      {state?.success ? (
        <p className="text-sm text-green">Rapport envoyé ✅</p>
      ) : null}
    </form>
  );
}

