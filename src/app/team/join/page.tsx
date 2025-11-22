'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';

export default function JoinTeamPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (!name.trim()) {
      setError('Le nom est requis.');
      return;
    }

    startTransition(async () => {
      // Récupérer l'ID de l'utilisateur depuis l'URL ou les paramètres
      const urlParams = new URLSearchParams(window.location.search);
      const userId = urlParams.get('userId');

      if (!userId) {
        setError('URL invalide. Veuillez scanner le QR code fourni.');
        return;
      }

      const supabase = createSupabaseBrowserClient();

      // Vérifier si un worker avec le même email existe déjà pour ce compte
      if (email) {
        const { data: existingWorker } = await supabase
          .from('workers')
          .select('id')
          .eq('created_by', userId)
          .is('site_id', null)
          .eq('email', email.toLowerCase())
          .maybeSingle();

        if (existingWorker) {
          setError('Un membre avec cet email existe déjà dans l\'équipe.');
          return;
        }
      }

      // Récupérer les infos du manager pour l'email (via une action serveur)
      // Pour l'instant, on passe undefined pour managerName

      // Créer un worker au niveau du compte (sans site_id) avec status 'pending'
      // Essayer d'abord avec status, puis sans si la colonne n'existe pas
      let insertData: any = {
        created_by: userId,
        name: name.trim(),
        email: email.trim() || null,
        role: role.trim() || null,
        site_id: null, // Worker au niveau du compte
      };

      // Essayer d'ajouter avec status 'pending'
      insertData.status = 'pending';
      let { error: insertError } = await supabase
        .from('workers')
        .insert(insertData);

      // Si l'erreur est liée à la colonne status, on ne peut pas créer le worker correctement
      // Il faut que la migration SQL soit exécutée
      if (insertError && (insertError.message.includes('status') || insertError.message.includes('column'))) {
        setError('La colonne status n\'existe pas dans la base de données. Veuillez exécuter la migration SQL migration-worker-status.sql dans Supabase pour activer le système de validation.');
        return;
      }

      if (insertError) {
        if (insertError.message.includes('unique') || insertError.message.includes('duplicate')) {
          setError('Un membre avec cet email existe déjà dans l\'équipe.');
        } else {
          setError(`Erreur lors de l'ajout: ${insertError.message}`);
        }
        return;
      }

      // Envoyer un email de confirmation si l'email est fourni
      if (email.trim()) {
        try {
          console.log('📧 Tentative d\'envoi email de confirmation à:', email.trim());
          // Appel à l'API pour envoyer l'email (car c'est une fonction serveur)
          const emailResponse = await fetch('/api/team/join-confirmation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              workerEmail: email.trim(),
              workerName: name.trim(),
              userId,
            }),
          });

          const emailResult = await emailResponse.json();
          
          if (!emailResponse.ok || !emailResult.success) {
            console.error('❌ Erreur envoi email confirmation:', emailResult.error || 'Erreur inconnue');
            // Ne pas bloquer l'ajout, mais logger l'erreur
          } else {
            console.log('✅ Email de confirmation envoyé avec succès');
          }
        } catch (emailError) {
          console.error('❌ Exception lors de l\'envoi email confirmation:', emailError);
          // Ne pas bloquer l'ajout si l'email échoue
        }
      } else {
        console.warn('⚠️ Pas d\'email fourni, email de confirmation non envoyé');
      }

      setSuccess(true);
      setTimeout(() => {
        router.push('/team/join/success');
      }, 1500);
    });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-100 via-white to-zinc-50 p-6 text-zinc-900 dark:from-zinc-900 dark:via-zinc-800 dark:to-black dark:text-white flex items-center justify-center">
      <div className="w-full max-w-md">
        <div className="rounded-3xl border border-zinc-200 bg-white/80 p-8 backdrop-blur dark:border-white/10 dark:bg-white/5 shadow-xl">
          <h1 className="text-3xl font-bold text-center mb-2 dark:text-white">
            Rejoindre l'équipe
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 text-center mb-8">
            Remplissez vos informations pour rejoindre l'équipe. Vous serez ajouté au catalogue et pourrez être assigné à des chantiers.
          </p>

          {success ? (
            <div className="rounded-lg bg-emerald-50 p-4 text-sm text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300 text-center">
              <p className="font-semibold mb-2">✅ Inscription réussie !</p>
              <p>Vous allez être redirigé vers la page équipe...</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="name" className="block text-sm font-semibold text-zinc-800 dark:text-zinc-100 mb-2">
                  Nom complet *
                </label>
                <input
                  type="text"
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jean Dupont"
                  className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
                  required
                  autoFocus
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-semibold text-zinc-800 dark:text-zinc-100 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="jean.dupont@example.com"
                  className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
                />
              </div>

              <div>
                <label htmlFor="role" className="block text-sm font-semibold text-zinc-800 dark:text-zinc-100 mb-2">
                  Métier
                </label>
                <input
                  type="text"
                  id="role"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  placeholder="Chef d'équipe"
                  className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-3 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 dark:border-zinc-600 dark:bg-zinc-800 dark:text-white"
                />
              </div>

              {error && (
                <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800 dark:bg-rose-900/30 dark:text-rose-300">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={isPending || !name.trim()}
                className="w-full rounded-lg bg-emerald-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {isPending ? 'Ajout en cours...' : 'Rejoindre l\'équipe'}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

