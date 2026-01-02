import { NextRequest, NextResponse } from 'next/server';

/**
 * API route pour générer une description IA d'une tâche
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskTitle, requiredRole, durationHours, reportCount } = body;

    if (!taskTitle) {
      return NextResponse.json(
        { error: 'Titre de tâche requis' },
        { status: 400 }
      );
    }

    // Générer une description basée sur les informations de la tâche
    let description = `Cette tâche consiste à ${taskTitle.toLowerCase()}.`;

    if (requiredRole) {
      description += ` Elle nécessite les compétences d'un ${requiredRole}.`;
    }

    if (durationHours) {
      description += ` La durée estimée est de ${durationHours} heure${durationHours > 1 ? 's' : ''}.`;
    }

    if (reportCount > 0) {
      description += ` ${reportCount} rapport${reportCount > 1 ? 's' : ''} ${reportCount > 1 ? 'ont été' : 'a été'} envoyé${reportCount > 1 ? 's' : ''} pour cette tâche, montrant un suivi actif de l'avancement.`;
    } else {
      description += ` Aucun rapport n'a encore été envoyé pour cette tâche.`;
    }

    // Si Google Gemini est configuré, utiliser l'API pour une description plus riche
    const { isGeminiConfigured, generateWithGemini } = await import('@/lib/ai/gemini');
    
    if (await isGeminiConfigured()) {
      try {
        const systemInstruction = 'Tu es un assistant expert en gestion de chantiers de construction. Génère une description claire et professionnelle d\'une tâche de chantier en français.';
        
        const prompt = `Génère une description détaillée pour cette tâche de chantier:
- Titre: ${taskTitle}
${requiredRole ? `- Rôle requis: ${requiredRole}` : ''}
${durationHours ? `- Durée estimée: ${durationHours} heures` : ''}
${reportCount > 0 ? `- Nombre de rapports: ${reportCount}` : 'Aucun rapport encore'}

La description doit être professionnelle, claire et aider l'employé à comprendre ce qu'il doit faire.`;

        const aiDescription = await generateWithGemini(prompt, systemInstruction, {
          temperature: 0.7,
          maxOutputTokens: 200,
        });

        if (aiDescription) {
          return NextResponse.json({ description: aiDescription });
        }
      } catch (error) {
        console.error('Erreur Gemini:', error);
        // Continuer avec la description générée localement
      }
    }

    // Retourner la description générée localement
    return NextResponse.json({ description });
  } catch (error) {
    console.error('Erreur génération description:', error);
    return NextResponse.json(
      { error: 'Erreur lors de la génération de la description' },
      { status: 500 }
    );
  }
}

