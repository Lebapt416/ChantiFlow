'use client';

import { useState } from 'react';
import { Send } from 'lucide-react';
import { AppShell } from '@/components/app-shell';

export default function ContactPage() {
  const [email, setEmail] = useState('');
  const [subject, setSubject] = useState('');
  const [message, setMessage] = useState('');
  const [isSent, setIsSent] = useState(false);
  const [isPending, setIsPending] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setIsPending(true);
    
    // Créer un lien mailto avec les informations
    const mailtoLink = `mailto:contact@chantiflow.com?subject=${encodeURIComponent(subject || 'Contact depuis ChantiFlow')}&body=${encodeURIComponent(`Email: ${email}\n\n${message}`)}`;
    
    // Ouvrir le client email
    window.location.href = mailtoLink;
    setIsSent(true);
    setIsPending(false);
    
    // Reset après 3 secondes
    setTimeout(() => {
      setIsSent(false);
      setEmail('');
      setSubject('');
      setMessage('');
    }, 3000);
  }

  return (
    <AppShell heading="Contact" subheading="Envoyez-nous un message" userEmail={null} primarySite={null}>
      <div className="mx-auto max-w-2xl">
        <div className="rounded-3xl border border-zinc-200 bg-white p-8 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block text-sm font-semibold text-zinc-900 dark:text-white mb-2">
                Votre email
              </label>
              <input
                type="email"
                id="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-2 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-black dark:border-zinc-600 dark:bg-zinc-800 dark:text-white dark:focus:ring-white"
                placeholder="votre@email.com"
              />
            </div>

            <div>
              <label htmlFor="subject" className="block text-sm font-semibold text-zinc-900 dark:text-white mb-2">
                Sujet
              </label>
              <input
                type="text"
                id="subject"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-2 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-black dark:border-zinc-600 dark:bg-zinc-800 dark:text-white dark:focus:ring-white"
                placeholder="Sujet de votre message"
              />
            </div>

            <div>
              <label htmlFor="message" className="block text-sm font-semibold text-zinc-900 dark:text-white mb-2">
                Message
              </label>
              <textarea
                id="message"
                required
                rows={6}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                className="w-full rounded-lg border border-zinc-300 bg-white px-4 py-2 text-zinc-900 focus:outline-none focus:ring-2 focus:ring-black dark:border-zinc-600 dark:bg-zinc-800 dark:text-white dark:focus:ring-white"
                placeholder="Votre message..."
              />
            </div>

            {isSent ? (
              <div className="rounded-lg bg-emerald-50 p-4 text-sm text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300">
                ✓ Votre client email va s&apos;ouvrir. Le message sera envoyé à contact@chantiflow.com
              </div>
            ) : null}

            <button
              type="submit"
              disabled={isPending}
              className="w-full rounded-lg bg-black px-6 py-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-white dark:text-black dark:hover:bg-zinc-200"
            >
              {isPending ? 'Envoi...' : (
                <>
                  <Send className="mr-2 inline h-4 w-4" />
                  Envoyer à contact@chantiflow.com
                </>
              )}
            </button>
          </form>

          <div className="mt-8 border-t border-zinc-200 pt-6 dark:border-zinc-800">
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Vous pouvez aussi nous contacter directement :
            </p>
            <ul className="mt-2 space-y-1 text-sm text-zinc-600 dark:text-zinc-400">
              <li>
                <a href="mailto:contact@chantiflow.com" className="hover:text-zinc-900 dark:hover:text-white">
                  contact@chantiflow.com
                </a>
              </li>
              <li>
                <a href="mailto:admin@chantiflow.com" className="hover:text-zinc-900 dark:hover:text-white">
                  admin@chantiflow.com (support)
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

