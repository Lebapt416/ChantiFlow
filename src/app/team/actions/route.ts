import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { sendTeamApprovalEmail, sendTeamRejectionEmail } from '@/lib/email';
import { generateAccessCode } from '@/lib/access-code';
import { createSupabaseAdminClient } from '@/lib/supabase/admin';

export async function POST(request: NextRequest) {
  const supabase = await createSupabaseServerClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: 'Non authentifié' }, { status: 401 });
  }

  const formData = await request.formData();
  const action = formData.get('action') as string;
  const workerId = formData.get('workerId') as string;

  if (!action || !workerId) {
    return NextResponse.json({ error: 'Paramètres manquants' }, { status: 400 });
  }

  // Vérifier que le worker appartient à l'utilisateur et récupérer ses infos
  const { data: worker, error: fetchError } = await supabase
    .from('workers')
    .select('id, name, email, role, created_by, status, access_code')
    .eq('id', workerId)
    .single();

  if (fetchError || !worker || worker.created_by !== user.id) {
    return NextResponse.json({ error: 'Worker non trouvé' }, { status: 404 });
  }

  if (action === 'approve') {
    // Générer un code d'accès unique si le worker n'en a pas déjà un
    let accessCode = worker.access_code;
    const admin = createSupabaseAdminClient();
    
    if (!accessCode) {
      // Générer un code d'accès unique
      let newAccessCode = generateAccessCode();
      let attempts = 0;
      const maxAttempts = 10;
      
      // Vérifier que le code n'existe pas déjà
      while (attempts < maxAttempts) {
        const { data: existingWorker } = await admin
          .from('workers')
          .select('id')
          .eq('access_code', newAccessCode)
          .maybeSingle();
        
        if (!existingWorker) {
          accessCode = newAccessCode;
          break;
        }
        
        newAccessCode = generateAccessCode();
        attempts++;
      }
      
      if (!accessCode) {
        return NextResponse.json({ error: 'Impossible de générer un code d\'accès unique' }, { status: 500 });
      }
      
      // Mettre à jour le worker avec le code d'accès et le statut
      const updateData: { status: string; access_code?: string } = { status: 'approved' };
      if (accessCode) {
        updateData.access_code = accessCode;
      }
      
      const { error: updateError } = await admin
        .from('workers')
        .update(updateData)
        .eq('id', workerId);
      
      if (updateError) {
        // Si l'erreur est liée à access_code ou status, essayer sans
        if (updateError.message.includes('access_code') || updateError.message.includes('status') || updateError.message.includes('column')) {
          console.warn('Colonne access_code ou status non trouvée, mise à jour sans code');
          // Essayer de mettre à jour seulement le status
          const { error: statusError } = await supabase
            .from('workers')
            .update({ status: 'approved' })
            .eq('id', workerId);
          
          if (statusError && !statusError.message.includes('status') && !statusError.message.includes('column')) {
            return NextResponse.json({ error: statusError.message }, { status: 500 });
          }
        } else {
          return NextResponse.json({ error: updateError.message }, { status: 500 });
        }
      }
    } else {
      // Le worker a déjà un code, juste mettre à jour le statut
      const { error: updateError } = await supabase
        .from('workers')
        .update({ status: 'approved' })
        .eq('id', workerId);
      
      if (updateError) {
        // Si l'erreur est liée à la colonne status, essayer sans
        if (updateError.message.includes('status') || updateError.message.includes('column')) {
          console.warn('Colonne status non trouvée, worker déjà approuvé par défaut');
        } else {
          return NextResponse.json({ error: updateError.message }, { status: 500 });
        }
      }
    }

    // Envoyer un email d'approbation si l'email est fourni
    if (worker.email) {
      try {
        await sendTeamApprovalEmail({
          workerEmail: worker.email,
          workerName: worker.name,
          managerName: user.email || undefined,
          accessCode: accessCode || undefined,
        });
      } catch (emailError) {
        console.error('Erreur envoi email approbation:', emailError);
        // Ne pas bloquer l'approbation si l'email échoue
      }
    }
  } else if (action === 'reject') {
    // Envoyer un email de refus AVANT de supprimer le worker
    if (worker.email) {
      try {
        await sendTeamRejectionEmail({
          workerEmail: worker.email,
          workerName: worker.name,
          managerName: user.email || undefined,
        });
      } catch (emailError) {
        console.error('Erreur envoi email refus:', emailError);
        // Ne pas bloquer le refus si l'email échoue
      }
    }

    const { error: deleteError } = await supabase
      .from('workers')
      .delete()
      .eq('id', workerId);

    if (deleteError) {
      return NextResponse.json({ error: deleteError.message }, { status: 500 });
    }
  } else {
    return NextResponse.json({ error: 'Action invalide' }, { status: 400 });
  }

  revalidatePath('/team');
  revalidatePath('/dashboard');
  
  // Rediriger vers la page d'origine (team ou dashboard)
  const referer = request.headers.get('referer');
  const redirectUrl = referer && (referer.includes('/team') || referer.includes('/dashboard'))
    ? referer
    : '/team';
  
  return NextResponse.redirect(new URL(redirectUrl, request.url));
}

