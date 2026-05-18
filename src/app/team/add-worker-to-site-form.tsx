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
      className="border border-orange px-4 py-2 font-mono text-[11px] uppercase tracking-widest text-orange transition hover:bg-paper-2 disabled:cursor-not-allowed disabled:opacity-70"
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
      setTimeout(() => {
        const form = document.getElementById('add-worker-to-site-form') as HTMLFormElement | null;
        form?.reset();
        setUseExisting(false);
      }, 0);
    }
  }, [state?.success]);

  const filteredWorkers = selectedSiteId
    ? availableWorkers.filter(() => true)
    : availableWorkers;

  return (
    <form
      id="add-worker-to-site-form"
      action={formAction}
      className="space-y-4 rounded border border-rule-soft bg-paper p-6 dark:border-rule dark:bg-ink"
    >
      <input type="hidden" name="siteId" value={selectedSiteId} />

      <div className="space-y-2">
        <label
          htmlFor="siteSelect"
          className="text-sm font-medium text-ink-2"
        >
          Chantier *
        </label>
        <select
          id="siteSelect"
          value={selectedSiteId}
          onChange={(e) => setSelectedSiteId(e.target.value)}
          className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
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
                  className="h-4 w-4 border-rule-soft text-orange focus:ring-2 focus:ring-orange dark:border-rule"
                />
                <label
                  htmlFor="useExisting"
                  className="text-sm font-medium text-ink-2"
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
                className="text-sm font-medium text-ink-2"
              >
                Membre de l&apos;équipe
              </label>
              <select
                id="existingWorkerId"
                name="existingWorkerId"
                className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
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
                  className="text-sm font-medium text-ink-2"
                >
                  Nom *
                </label>
                <input
                  id="name"
                  name="name"
                  placeholder="Camille Dupont"
                  className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
                  required={!useExisting}
                />
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="email"
                  className="text-sm font-medium text-ink-2"
                >
                  Email
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  placeholder="camille@chantier.com"
                  className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
                />
              </div>
              <div className="space-y-2">
                <label
                  htmlFor="role"
                  className="text-sm font-medium text-ink-2"
                >
                  Métier
                </label>
                <input
                  id="role"
                  name="role"
                  placeholder="Chef d'équipe"
                  className="w-full rounded border border-rule-soft px-3 py-2 focus:outline-none focus:ring-2 focus:ring-orange text-ink bg-paper dark:border-rule dark:bg-ink dark:text-paper"
                />
              </div>
            </div>
          )}

          <div className="flex items-end">
            <SubmitButton />
          </div>
          {state?.error ? (
            <p className="text-sm text-danger">{state.error}</p>
          ) : null}
          {state?.success ? (
            <p className="text-sm text-green">
              Membre ajouté au chantier avec succès !
            </p>
          ) : null}
        </>
      )}

      {!selectedSiteId && (
        <p className="text-sm text-ink-3">
          Sélectionnez un chantier pour ajouter un membre.
        </p>
      )}
    </form>
  );
}
