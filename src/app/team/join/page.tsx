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
      const urlParams = new URLSearchParams(window.location.search);
      const userId = urlParams.get('userId');

      if (!userId) {
        setError('URL invalide. Veuillez scanner le QR code fourni.');
        return;
      }

      const supabase = createSupabaseBrowserClient();

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

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const insertData: any = {
        created_by: userId,
        name: name.trim(),
        email: email.trim() || null,
        role: role.trim() || null,
        site_id: null,
      };

      insertData.status = 'pending';
      const { error: insertError } = await supabase
        .from('workers')
        .insert(insertData);

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

      if (email && email.trim()) {
        try {
          console.log('📧 Tentative d\'envoi email de confirmation à:', email.trim());
          console.log('📧 Données envoyées:', { workerEmail: email.trim(), workerName: name.trim(), userId });

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
    <div className="min-h-screen bg-ink p-6 text-paper flex items-center justify-center">
      <div className="w-full max-w-md">
        <div className="border border-rule bg-ink p-8">
          <h1 className="font-serif text-[32px] text-paper text-center mb-2">
            Rejoindre l&apos;équipe
          </h1>
          <p className="font-mono text-[10px] uppercase tracking-widest text-ink-3 text-center mb-8">
            Remplissez vos informations pour rejoindre l&apos;équipe. Vous serez ajouté au catalogue et pourrez être assigné à des chantiers.
          </p>

          {success ? (
            <div className="border border-green bg-paper-2 p-4 font-mono text-[10px] uppercase tracking-widest text-green text-center">
              <p className="font-semibold mb-2">✅ Inscription réussie !</p>
              <p>Vous allez être redirigé vers la page équipe...</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label htmlFor="name" className="block font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-2">
                  Nom complet *
                </label>
                <input
                  type="text"
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jean Dupont"
                  className="w-full border-2 border-orange bg-ink px-4 py-3 text-paper placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-orange transition-colors"
                  required
                  autoFocus
                />
              </div>

              <div>
                <label htmlFor="email" className="block font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="jean.dupont@example.com"
                  className="w-full border border-rule bg-ink px-4 py-3 text-paper placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-orange transition-colors"
                />
              </div>

              <div>
                <label htmlFor="role" className="block font-mono text-[10px] uppercase tracking-widest text-ink-3 mb-2">
                  Métier
                </label>
                <input
                  type="text"
                  id="role"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  placeholder="Chef d'équipe"
                  className="w-full border border-rule bg-ink px-4 py-3 text-paper placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-orange transition-colors"
                />
              </div>

              {error && (
                <div className="border border-danger bg-paper-2 p-3 font-mono text-[10px] text-danger">
                  {error}
                </div>
              )}

              {emailError && (
                <div className="border border-warn bg-paper-2 p-3 font-mono text-[10px] text-warn">
                  ⚠️ {emailError}
                </div>
              )}

              <button
                type="submit"
                disabled={isPending || !name.trim()}
                className="w-full border border-orange bg-orange px-6 py-3 font-mono text-[11px] uppercase tracking-widest text-paper transition hover:bg-orange-dark disabled:cursor-not-allowed disabled:opacity-70"
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
