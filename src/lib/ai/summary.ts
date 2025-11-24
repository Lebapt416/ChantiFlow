'use server';

const API_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL || 'http://localhost:8000';

export async function generateGlobalSummary(sites: any[]) {
  if (!API_URL) return null;
  
  try {
    // Transformer les données pour l'API Python
    const payload = {
      sites: sites.map(s => ({
        name: s.name,
        tasks_total: s.tasks?.length || 0,
        tasks_done: s.tasks?.filter((t: any) => t.status === 'done').length || 0,
        complexity: 5.0, // Valeur par défaut ou calculée
        days_elapsed: Math.ceil((new Date().getTime() - new Date(s.created_at).getTime()) / (1000 * 3600 * 24))
      }))
    };

    const res = await fetch(`${API_URL}/summary/global`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      next: { revalidate: 60 } // Cache 1 minute
    });
    
    if (!res.ok) return null;
    return await res.json();
    
  } catch (e) {
    console.error("Erreur résumé global:", e);
    return null;
  }
}

export async function generateSiteSummary(site: any, tasks: any[]) {
  if (!API_URL) return null;

  try {
    const totalTasks = tasks.length;
    const pendingTasks = tasks.filter(t => t.status === 'pending').length;
    
    // Estimation deadline théorique (simplifiée)
    const plannedDays = site.deadline 
      ? Math.ceil((new Date(site.deadline).getTime() - new Date(site.created_at).getTime()) / (1000 * 3600 * 24))
      : 30;

    const payload = {
      site_name: site.name,
      tasks_total: totalTasks,
      tasks_pending: pendingTasks,
      complexity: 5.5, // On pourrait affiner ce calcul
      days_elapsed: Math.ceil((new Date().getTime() - new Date(site.created_at).getTime()) / (1000 * 3600 * 24)),
      planned_duration: plannedDays
    };

    const res = await fetch(`${API_URL}/summary/site`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      next: { revalidate: 60 }
    });

    if (!res.ok) return null;
    return await res.json();

  } catch (e) {
    console.error("Erreur résumé chantier:", e);
    return null;
  }
}
