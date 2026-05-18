'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { Mail, Send, CheckCircle2, AlertCircle, ArrowLeft } from 'lucide-react';

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    company: '',
    message: '',
  });
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setErrorMessage('');

    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Erreur lors de l\'envoi du message');
      }

      setStatus('success');
      setFormData({ name: '', email: '', company: '', message: '' });
    } catch (error) {
      console.error('Erreur contact:', error);
      setStatus('error');
      setErrorMessage(
        error instanceof Error ? error.message : 'Une erreur est survenue. Veuillez réessayer.'
      );
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  return (
    <div className="min-h-screen bg-paper">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-zinc-200/50 bg-paper   ">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3">
              <Image
                src="/logo.svg"
                alt="ChantiFlow"
                width={40}
                height={40}
                priority
                className="h-10 w-10"
              />
              <span className="text-xl font-bold text-ink">
                ChantiFlow
              </span>
            </Link>
            <nav className="flex items-center gap-6">
              <Link
                href="/login"
                className="text-sm font-medium text-ink-2 transition hover:text-ink"
              >
                Connexion
              </Link>
              <Link
                href="/login"
                className="rounded bg-ink px-5 py-2 text-sm font-semibold text-paper transition hover:bg-rule"
              >
                Commencer
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-4xl px-6 py-16">
        <div className="mb-6">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-sm font-medium text-ink-2 hover:text-ink transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Retour à l&apos;accueil
          </Link>
        </div>
        
        <div className="text-center mb-12">
          <h1 className="text-4xl font-extrabold tracking-tight text-ink sm:text-5xl">
            Contactez-nous
          </h1>
          <p className="mt-4 text-lg text-ink-2">
            Une question ? Un projet ? Nous sommes là pour vous aider.
          </p>
        </div>

        {/* Success Message */}
        {status === 'success' && (
          <div className="mb-8 rounded border border-rule-soft bg-paper-2 p-6 dark:border-rule dark:bg-paper-2">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="h-6 w-6 text-orange dark:text-green flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-ink dark:text-paper">
                  Message envoyé avec succès !
                </h3>
                <p className="mt-2 text-sm text-ink dark:text-green">
                  Nous vous répondrons dans les plus brefs délais.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {status === 'error' && (
          <div className="mb-8 rounded border border-rose-200 bg-rose-50 p-6 dark:border-rose-800 dark:bg-rose-900/20">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-6 w-6 text-rose-600 dark:text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-lg font-semibold text-rose-900 dark:text-rose-100">
                  Erreur lors de l&apos;envoi
                </h3>
                <p className="mt-2 text-sm text-rose-700 dark:text-rose-300">
                  {errorMessage || 'Une erreur est survenue. Veuillez réessayer.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Contact Form */}
        <div className="rounded border border-zinc-200 bg-white p-8 dark:border-zinc-800 dark:bg-zinc-900">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              <div>
                <label
                  htmlFor="name"
                  className="block text-sm font-medium text-ink-2 mb-2"
                >
                  Nom complet *
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  required
                  value={formData.name}
                  onChange={handleChange}
                  className="w-full rounded-lg border border-zinc-300 px-4 py-3 text-zinc-900 shadow-sm focus:border-orange focus:outline-none focus:ring-2 focus:ring-paper-2 dark:border-zinc-700 dark:bg-zinc-800 dark:text-white"
                  placeholder="Jean Dupont"
                />
              </div>

              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium text-ink-2 mb-2"
                >
                  Email *
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  required
                  value={formData.email}
                  onChange={handleChange}
                  className="w-full rounded-lg border border-zinc-300 px-4 py-3 text-zinc-900 shadow-sm focus:border-orange focus:outline-none focus:ring-2 focus:ring-paper-2 dark:border-zinc-700 dark:bg-zinc-800 dark:text-white"
                  placeholder="jean.dupont@example.com"
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="company"
                className="block text-sm font-medium text-ink-2 mb-2"
              >
                Entreprise (optionnel)
              </label>
              <input
                type="text"
                id="company"
                name="company"
                value={formData.company}
                onChange={handleChange}
                className="w-full rounded-lg border border-zinc-300 px-4 py-3 text-zinc-900 shadow-sm focus:border-orange focus:outline-none focus:ring-2 focus:ring-paper-2 dark:border-zinc-700 dark:bg-zinc-800 dark:text-white"
                placeholder="Nom de votre entreprise"
              />
            </div>

            <div>
              <label
                htmlFor="message"
                className="block text-sm font-medium text-ink-2 mb-2"
              >
                Message *
              </label>
              <textarea
                id="message"
                name="message"
                required
                rows={6}
                value={formData.message}
                onChange={handleChange}
                className="w-full rounded-lg border border-zinc-300 px-4 py-3 text-zinc-900 shadow-sm focus:border-orange focus:outline-none focus:ring-2 focus:ring-paper-2 dark:border-zinc-700 dark:bg-zinc-800 dark:text-white resize-none"
                placeholder="Décrivez votre projet ou votre question..."
              />
            </div>

            <div className="flex items-center justify-between pt-4">
              <p className="text-sm text-ink-3">
                * Champs obligatoires
              </p>
              <button
                type="submit"
                disabled={status === 'loading'}
                className="flex items-center gap-2 rounded bg-orange px-6 py-3 text-paper font-semibold transition-colors duration-200 hover:bg-orange-dark disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {status === 'loading' ? (
                  <>
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    Envoi en cours...
                  </>
                ) : (
                  <>
                    <Send className="h-5 w-5" />
                    Envoyer le message
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Alternative Contact */}
        <div className="mt-12 text-center">
          <p className="text-sm text-ink-3 mb-4">
            Vous préférez nous écrire directement ?
          </p>
          <a
            href="mailto:contact@chantiflow.com"
            className="inline-flex items-center gap-2 text-orange dark:text-green font-medium hover:text-ink dark:hover:text-green transition"
          >
            <Mail className="h-5 w-5" />
            contact@chantiflow.com
          </a>
        </div>
      </main>
    </div>
  );
}
