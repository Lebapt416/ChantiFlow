import { redirect } from 'next/navigation';

export const dynamic = 'force-dynamic';

export default function Home() {
  // Rediriger toujours vers la landing page
  redirect('/landing');
}
