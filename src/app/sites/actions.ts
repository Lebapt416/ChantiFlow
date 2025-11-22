'use server';

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';
import { sendSiteCompletedEmail } from '@/lib/email';

export type SiteActionState = {
  error?: string;
  success?: boolean;
  message?: string;
};

export async function deleteSiteAction(siteId: string): Promise<SiteActionState> {
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return { error: 'Non autorisé.' };
    }

    // Vérifier que le chantier appartient à l'utilisateur
    const { data: site } = await supabase
      .from('sites')
      .select('id, name, created_by')
      .eq('id', siteId)
      .single();

    if (!site || site.created_by !== user.id) {
      return { error: 'Chantier non trouvé ou non autorisé.' };
    }

    const admin = createSupabaseAdminClient();

    // Supprimer toutes les données associées (workers, tasks, reports)
    // Note: Les contraintes de clé étrangère avec ON DELETE CASCADE devraient gérer cela automatiquement
    // Mais on supprime explicitement pour être sûr

    // Supprimer les workers du chantier
    await admin.from('workers').delete().eq('site_id', siteId);

    // Supprimer les rapports liés aux tâches du chantier
    const { data: tasks } = await admin
      .from('tasks')
      .select('id')
      .eq('site_id', siteId);

    if (tasks && tasks.length > 0) {
      const taskIds = tasks.map((t) => t.id);
      await admin.from('reports').delete().in('task_id', taskIds);
    }

    // Supprimer les tâches
    await admin.from('tasks').delete().eq('site_id', siteId);

    // Supprimer le chantier
    const { error: deleteError } = await admin.from('sites').delete().eq('id', siteId);

    if (deleteError) {
      console.error('Erreur suppression chantier:', deleteError);
      return { error: deleteError.message };
    }

    revalidatePath('/sites');
    revalidatePath('/dashboard');

    return {
      success: true,
      message: 'Chantier supprimé avec succès.',
    };
  } catch (error) {
    console.error('Erreur suppression chantier:', error);
    return {
      error: error instanceof Error ? error.message : 'Erreur lors de la suppression.',
    };
  }
}

export async function completeSiteFromListAction(siteId: string): Promise<SiteActionState> {
  try {
    const supabase = await createSupabaseServerClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return { error: 'Non autorisé.' };
    }

    // Vérifier que le chantier appartient à l'utilisateur
    const { data: site } = await supabase
      .from('sites')
      .select('id, name, created_by')
      .eq('id', siteId)
      .single();

    if (!site || site.created_by !== user.id) {
      return { error: 'Chantier non trouvé ou non autorisé.' };
    }

    const admin = createSupabaseAdminClient();

    // Marquer le chantier comme terminé
    const { error: updateError } = await admin
      .from('sites')
      .update({ completed_at: new Date().toISOString() })
      .eq('id', siteId);

    if (updateError) {
      console.error('Erreur mise à jour chantier:', updateError);
      // Fallback: mettre à jour user_metadata
      try {
        const { data: currentUser } = await admin.auth.admin.getUserById(user.id);
        const completedSites = Array.isArray(currentUser?.user?.user_metadata?.completedSites)
          ? (currentUser.user.user_metadata.completedSites as string[])
          : [];
        
        if (!completedSites.includes(siteId)) {
          await admin.auth.admin.updateUserById(user.id, {
            user_metadata: {
              ...currentUser?.user?.user_metadata,
              completedSites: [...completedSites, siteId],
            },
          });
        }
      } catch (metaError) {
        console.error('Erreur fallback metadata:', metaError);
      }
    }

    // Récupérer tous les workers du chantier
    const { data: workers } = await admin
      .from('workers')
      .select('id, name, email')
      .eq('site_id', siteId);

    // Envoyer un email à chaque worker
    if (workers && workers.length > 0) {
      const emailPromises = workers.map((worker) => {
        if (worker.email) {
          return sendSiteCompletedEmail({
            workerEmail: worker.email,
            workerName: worker.name || 'Employé',
            siteName: site.name,
          });
        }
        return Promise.resolve({ success: false });
      });

      await Promise.allSettled(emailPromises);
    }

    // Envoyer un email au créateur du chantier
    if (user.email) {
      try {
        await sendSiteCompletedEmail({
          workerEmail: user.email,
          workerName: user.email,
          siteName: site.name,
        });
      } catch (emailError) {
        console.error('Erreur envoi email créateur:', emailError);
      }
    }

    // Supprimer les workers du chantier
    if (workers && workers.length > 0) {
      await admin.from('workers').delete().eq('site_id', siteId);
    }

    revalidatePath('/sites');
    revalidatePath('/dashboard');
    revalidatePath(`/site/${siteId}`);

    return {
      success: true,
      message: 'Chantier terminé avec succès. Les employés ont été notifiés.',
    };
  } catch (error) {
    console.error('Erreur terminaison chantier:', error);
    return {
      error: error instanceof Error ? error.message : 'Erreur lors de la terminaison.',
    };
  }
}

