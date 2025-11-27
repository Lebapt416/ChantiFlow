"use client";

import { useState, useActionState, useEffect } from "react";
import { useFormStatus } from "react-dom";
import { addWorkerAction, type ActionState } from '@/app/site/[id]/actions';

const initialState: ActionState = {};

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
      disabled={pending}
    >
      {pending ? 'Ajout...' : 'Ajouter au chantier'}
    </button>
  );
}

type Worker = {
  id: string;
  name: string;
  email: string | null;
  role: string | null;
};

type Site = {
  id: string;
  name: string;
};

type Props = {
  sites: Site[];
  availableWorkers: Worker[];
};

export function AddWorkerToSiteForm({ sites, availableWorkers }: Props) {
  const [selectedSiteId, setSelectedSiteId] = useState('');
  const [useExisting, setUseExisting] = useState(false);
  const [state, formAction] = useActionState(addWorkerAction, initialState);

  useEffect(() => {
    if (state?.success) {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      setTimeout(() => {
        const form = document.getElementById('add-worker-to-site-form') as HTMLFormElement | null;
        form?.reset();
        setUseExisting(false);
      }, 0);
      // Ne pas réinitialiser le site sélectionné
    }
  }, [state?.success]);

  // Filtrer les workers disponibles (ceux qui ne sont pas déjà assignés au chantier sélectionné)
  const filteredWorkers = selectedSiteId
    ? availableWorkers.filter((worker) => {
        // Vérifier si le worker n'est pas déjà assigné à ce chantier
        // (on ne peut pas vérifier côté client, donc on affiche tous les workers disponibles)
        return true;
      })
    : availableWorkers;

  return (
    <form
      id="add-worker-to-site-form"
      action={formAction}
      className="space-y-4 rounded-2xl border border-zinc-100 bg-white p-6 shadow-lg shadow-black/5 dark:border-zinc-800 dark:bg-zinc-900"
    >
      <input type="hidden" name="siteId" value={selectedSiteId} />
      
      <div className="space-y-2">
        <label
          htmlFor="siteSelect"
          className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
        >
          Chantier *
        </label>
        <select
          id="siteSelect"
          value={selectedSiteId}
          onChange={(e) => setSelectedSiteId(e.target.value)}
          className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
          required
        >
          <option value="">-- Sélectionner un chantier --</option>
          {sites.map((site) => (
            <option key={site.id} value={site.id}>
              {site.name}
            </option>
          ))}
        </select>
      </div>

      {selectedSiteId && (
        <>
          {filteredWorkers.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="useExisting"
                  checked={useExisting}
                  onChange={(e) => setUseExisting(e.target.checked)}
                  className="h-4 w-4 rounded border-zinc-300 text-emerald-600 focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600"
                />
                <label
                  htmlFor="useExisting"
                  className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
                >
                  Sélectionner un membre existant de mon équipe
                </label>
              </div>
            </div>
          )}

          {useExisting && filteredWorkers.length > 0 ? (
            <div className="space-y-2">
              <label
                htmlFor="existingWorkerId"
                className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
              >
                Membre de l&apos;équipe
              </label>
              <select
                id="existingWorkerId"
                name="existingWorkerId"
                className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
                required
              >
                <option value="">-- Sélectionner un membre --</option>
                {filteredWorkers.map((worker) => (
                  <option key={worker.id} value={worker.id}>
                    {worker.name} {worker.role ? `(${worker.role})` : ''} {worker.email ? `- ${worker.email}` : ''}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <label
                  htmlFor="name"
                  className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
                >
                  Nom *
                </label>
                <input
                  id="name"
                  name="name"
                  placeholder="Camille Dupont"
                  className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
                  required={!useExisting}
                />
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="email"
                  className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
                >
                  Email
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="camille@chantier.com"
                  className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
                />
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="role"
                  className="text-sm font-medium text-zinc-600 dark:text-zinc-300"
                >
                  Métier
                </label>
                <input
                  id="role"
                  name="role"
                  placeholder="Chef d'équipe"
                  className="w-full rounded-md border border-zinc-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-white"
                />
              </div>
            </div>
          )}

          <div className="flex items-end">
            <SubmitButton />
          </div>
          {state?.error ? (
            <p className="text-sm text-rose-400">{state.error}</p>
          ) : null}
          {state?.success ? (
            <p className="text-sm text-emerald-400">
              Membre ajouté au chantier avec succès !
            </p>
          ) : null}
        </>
      )}

      {!selectedSiteId && (
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Sélectionnez un chantier pour ajouter un membre.
        </p>
      )}
    </form>
  );
}

