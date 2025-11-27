'use server';

import { cookies } from 'next/headers';

const WORKER_SESSION_COOKIE = 'worker_session';

export type WorkerSessionPayload = {
  workerId: string;
  token: string;
  siteId?: string | null;
  name?: string | null;
  email?: string | null;
};

function serializeSession(payload: WorkerSessionPayload) {
  return JSON.stringify({
    workerId: payload.workerId,
    token: payload.token,
    siteId: payload.siteId ?? null,
    name: payload.name ?? null,
    email: payload.email ?? null,
  });
}

export async function readWorkerSession(): Promise<WorkerSessionPayload | null> {
  const cookieStore = await cookies();
  const cookie = cookieStore.get(WORKER_SESSION_COOKIE);
  if (!cookie?.value) return null;
  try {
    return JSON.parse(cookie.value) as WorkerSessionPayload;
  } catch {
    return null;
  }
}

export async function writeWorkerSession(payload: WorkerSessionPayload) {
  const cookieStore = await cookies();
  cookieStore.set(WORKER_SESSION_COOKIE, serializeSession(payload), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 60, // 60 jours
    path: '/',
  });
}

export async function updateWorkerSession(partial: Partial<WorkerSessionPayload>) {
  const current = (await readWorkerSession()) || { workerId: '', token: '' };
  const next = {
    ...current,
    ...partial,
  } as WorkerSessionPayload;
  if (!next.workerId || !next.token) {
    throw new Error('Session worker invalide');
  }
  await writeWorkerSession(next);
}

export async function clearWorkerSession() {
  const cookieStore = await cookies();
  cookieStore.set(WORKER_SESSION_COOKIE, '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 0,
    path: '/',
  });
}

