'use client';

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { createSupabaseBrowserClient } from '@/lib/supabase/client';

export default function JoinTeamPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('');
  const [error, setError] = useState('');
  const [emailError, setEmailError] = useState('');
  const [success, setSuccess] = useState(false);
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setEmailError('');
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const insertData: any = {
        created_by: userId,
        name: name.trim(),
        email: email.trim() || null,
        role: role.trim() || null,
        site_id: null, // Worker au niveau du compte
      };

      // Essayer d'ajouter avec status 'pending'
      insertData.status = 'pending';
      const { error: insertError } = await supabase
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
      if (email && email.trim()) {
        try {
          console.log('📧 Tentative d\'envoi email de confirmation à:', email.trim());
          console.log('📧 Données envoyées:', { workerEmail: email.trim(), workerName: name.trim(), userId });
          
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

          console.log('📧 Réponse API status:', emailResponse.status, emailResponse.statusText);

          if (!emailResponse.ok) {
            const errorText = await emailResponse.text();
            console.error('❌ Erreur HTTP envoi email confirmation:', emailResponse.status, errorText);
            setEmailError(`Erreur lors de l'envoi de l'email de confirmation (${emailResponse.status}). Votre inscription a été enregistrée, mais l'email n'a pas pu être envoyé.`);
          } else {
            const emailResult = await emailResponse.json();
            console.log('📧 Résultat API:', emailResult);
            
            if (!emailResult.success) {
              console.error('❌ Erreur envoi email confirmation:', emailResult.error || 'Erreur inconnue');
              setEmailError(`Erreur lors de l'envoi de l'email: ${emailResult.error || 'Erreur inconnue'}. Votre inscription a été enregistrée.`);
            } else {
              console.log('✅ Email de confirmation envoyé avec succès');
            }
          }
        } catch (emailError) {
          console.error('❌ Exception lors de l\'envoi email confirmation:', emailError);
          setEmailError('Erreur lors de l\'envoi de l\'email de confirmation. Votre inscription a été enregistrée, mais l\'email n\'a pas pu être envoyé.');
        }
      } else {
        console.warn('⚠️ Pas d\'email fourni, email de confirmation non envoyé. Email:', email);
      }

      setSuccess(true);
      setTimeout(() => {
        router.push('/team/join/success');
      }, 1500);
    });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-900 via-zinc-800 to-zinc-900 p-6 text-white flex items-center justify-center">
      <div className="w-full max-w-md">
        <div className="rounded border border-zinc-700/50 bg-zinc-800/90 p-8  ">
          <h1 className="text-3xl font-bold text-center mb-2 text-zinc-100">
            Rejoindre l&apos;équipe
          </h1>
          <p className="text-sm text-zinc-400 text-center mb-8">
            Remplissez vos informations pour rejoindre l&apos;équipe. Vous serez ajouté au catalogue et pourrez être assigné à des chantiers.
          </p>

          {success ? (
            <div className="rounded-lg bg-paper-2 p-4 text-sm text-green text-center">
              <p className="font-semibold mb-2">✅ Inscription réussie !</p>
              <p>Vous allez être redirigé vers la page équipe...</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label htmlFor="name" className="block text-sm font-semibold text-zinc-300 mb-2">
                  Nom complet *
                </label>
                <input
                  type="text"
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jean Dupont"
                  className="w-full rounded-lg border-2 border-orange bg-ink px-4 py-3 text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-orange focus:border-orange transition-colors"
                  required
                  autoFocus
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-semibold text-zinc-300 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="jean.dupont@example.com"
                  className="w-full rounded-lg border border-zinc-600 bg-ink px-4 py-3 text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-orange focus:border-orange transition-colors"
                />
              </div>

              <div>
                <label htmlFor="role" className="block text-sm font-semibold text-zinc-300 mb-2">
                  Métier
                </label>
                <input
                  type="text"
                  id="role"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  placeholder="Chef d'équipe"
                  className="w-full rounded-lg border border-zinc-600 bg-ink px-4 py-3 text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-orange focus:border-orange transition-colors"
                />
              </div>

              {error && (
                <div className="rounded-lg bg-rose-900/30 p-3 text-sm text-rose-300">
                  {error}
                </div>
              )}

              {emailError && (
                <div className="rounded-lg bg-amber-900/30 p-3 text-sm text-amber-300">
                  ⚠️ {emailError}
                </div>
              )}

              <button
                type="submit"
                disabled={isPending || !name.trim()}
                className="w-full rounded-lg bg-orange px-6 py-3 text-base font-semibold text-white transition hover:bg-orange-dark disabled:cursor-not-allowed disabled:opacity-70 shadow-orange/20"
              >
                {isPending ? 'Ajout en cours...' : 'Rejoindre l&apos;équipe'}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

