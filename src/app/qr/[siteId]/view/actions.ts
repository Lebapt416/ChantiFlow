'use server';

import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { generateAccessCode } from '@/lib/access-code';

export type TestWorkerResult = {
  success: boolean;
  workerId?: string;
  accessCode?: string;
  error?: string;
};

export async function createTestWorkerAction(siteId: string): Promise<TestWorkerResult> {
  try {
    const admin = createSupabaseAdminClient();

    // Chercher un worker de test existant
    const { data: existingTestWorker } = await admin
      .from('workers')
      .select('id, access_code')
      .eq('site_id', siteId)
      .eq('email', 'test@chantiflow.com')
      .maybeSingle();

    if (existingTestWorker) {
      return {
        success: true,
        workerId: existingTestWorker.id,
        accessCode: existingTestWorker.access_code || 'TEST1234',
      };
    }

    // Créer un nouveau worker de test
    const accessCode = generateAccessCode();

    // Essayer d'insérer avec access_code
    const insertData: any = {
      site_id: siteId,
      name: 'Test Employé',
      email: 'test@chantiflow.com',
      role: 'Test',
    };

    // Ajouter access_code si possible
    try {
      insertData.access_code = accessCode;
    } catch (e) {
      // Ignorer si la colonne n'existe pas
    }

    const { data: newWorker, error: insertError } = await admin
      .from('workers')
      .insert(insertData)
      .select('id')
      .single();

    if (insertError || !newWorker) {
      console.error('Erreur création worker test:', insertError);
      return {
        success: false,
        error: insertError?.message || 'Impossible de créer le worker de test.',
      };
    }

    return {
      success: true,
      workerId: newWorker.id,
      accessCode: accessCode,
    };
  } catch (error) {
    console.error('Erreur createTestWorkerAction:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Erreur inconnue',
    };
  }
}

