import { NextRequest, NextResponse } from 'next/server';
import { createSupabaseServerClient } from '@/lib/supabase/server';
import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';

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

  // Vérifier que le worker appartient à l'utilisateur
  const { data: worker, error: fetchError } = await supabase
    .from('workers')
    .select('id, created_by, status')
    .eq('id', workerId)
    .single();

  if (fetchError || !worker || worker.created_by !== user.id) {
    return NextResponse.json({ error: 'Worker non trouvé' }, { status: 404 });
  }

  if (action === 'approve') {
    const { error: updateError } = await supabase
      .from('workers')
      .update({ status: 'approved' })
      .eq('id', workerId);

    if (updateError) {
      return NextResponse.json({ error: updateError.message }, { status: 500 });
    }
  } else if (action === 'reject') {
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
  return NextResponse.redirect(new URL('/team', request.url));
}

