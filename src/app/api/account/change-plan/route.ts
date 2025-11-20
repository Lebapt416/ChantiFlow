import { NextResponse } from 'next/server';
import { changePlanAction } from '@/app/account/actions';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const result = await changePlanAction({}, formData);

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 400 });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Erreur inconnue' },
      { status: 500 },
    );
  }
}

