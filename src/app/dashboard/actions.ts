'use server';

import { revalidatePath } from 'next/cache';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { canCreateSite } from '@/lib/plans';

export type CreateSiteState = {
  error?: string;
  success?: boolean;
};

export async function createSiteAction(
  _prevState: CreateSiteState,
  formData: FormData,
): Promise<CreateSiteState> {
  const name = String(formData.get('name') ?? '').trim();
  const deadline = String(formData.get('deadline') ?? '');
  const postalCode = String(formData.get('postal_code') ?? '').trim();

  if (!name || !deadline) {
    return { error: 'Nom et deadline sont requis.' };
  }

  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  if (userError || !user) {
    return { error: 'Session expirée, reconnecte-toi.' };
  }

  // Vérifier les limites du plan
  const { allowed, reason } = await canCreateSite(user.id);
  if (!allowed) {
    return { error: reason || 'Limite de chantiers atteinte pour votre plan.' };
  }

  // Préparer les données à insérer
  const insertData: {
    name: string;
    deadline: string;
    created_by: string;
    postal_code?: string | null;
  } = {
    name,
    deadline,
    created_by: user.id,
  };

  // Ajouter le code postal seulement s'il est fourni et valide (5 chiffres)
  if (postalCode && /^\d{5}$/.test(postalCode)) {
    insertData.postal_code = postalCode;
  } else if (postalCode && postalCode.trim() !== '') {
    // Si ce n'est pas un code postal valide, ne pas l'ajouter
    console.warn('⚠️ Code postal invalide:', postalCode);
  }

  const { error } = await supabase.from('sites').insert(insertData);

  if (error) {
    // Si l'erreur est liée à la colonne postal_code (n'existe pas), réessayer sans
    if (error.message.includes('postal_code') || error.message.includes('column')) {
      console.warn('⚠️ Colonne postal_code non trouvée - migration SQL non exécutée');
      const { error: retryError } = await supabase.from('sites').insert({
        name,
        deadline,
        created_by: user.id,
      });
      
      if (retryError) {
        return { error: `Erreur: ${retryError.message}. Veuillez exécuter la migration SQL (migration-site-address.sql)` };
      }
    } else {
      return { error: error.message };
    }
  }

  revalidatePath('/dashboard');
  revalidatePath('/sites');
  return { success: true };
}

