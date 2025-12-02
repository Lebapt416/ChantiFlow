import { describe, it, expect, beforeEach, vi } from 'vitest';
import { addTaskAction, addWorkerAction } from './actions';

// Mock des dépendances avant les imports
vi.mock('@/lib/supabase/server', () => ({
  createSupabaseServerClient: vi.fn(),
}));

vi.mock('@/lib/utils/role-formatting', () => ({
  capitalizeRoleWords: vi.fn((role: string) => {
    return role
      .split(' ')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }),
}));

vi.mock('@/lib/plans', () => ({
  canAddWorker: vi.fn().mockResolvedValue({ allowed: true, reason: null }),
}));

vi.mock('@/lib/email', () => ({
  sendWorkerWelcomeEmail: vi.fn().mockResolvedValue({ success: true }),
}));

vi.mock('@/lib/access-code', () => ({
  generateAccessCode: vi.fn(() => 'ABC123'),
}));

vi.mock('next/cache', () => ({
  revalidatePath: vi.fn(),
}));

describe('Server Actions - Site Actions', () => {
  let mockSupabase: any;

  beforeEach(async () => {
    // Reset des mocks
    vi.clearAllMocks();

    // Mock Supabase client avec chaînage de méthodes
    const createChainableMock = () => {
      const chain = {
        from: vi.fn().mockReturnThis(),
        insert: vi.fn().mockReturnThis(),
        select: vi.fn().mockReturnThis(),
        eq: vi.fn().mockReturnThis(),
        is: vi.fn().mockReturnThis(),
        single: vi.fn(),
        maybeSingle: vi.fn(),
        update: vi.fn().mockReturnThis(),
      };
      // Faire en sorte que toutes les méthodes retournent l'objet chainable
      Object.keys(chain).forEach((key) => {
        if (key !== 'single' && key !== 'maybeSingle') {
          (chain as any)[key] = vi.fn().mockReturnValue(chain);
        }
      });
      return chain;
    };

    mockSupabase = {
      ...createChainableMock(),
      auth: {
        getUser: vi.fn(),
      },
    };

    const { createSupabaseServerClient } = await import('@/lib/supabase/server');
    vi.mocked(createSupabaseServerClient).mockResolvedValue(mockSupabase as any);
  });

  describe('addTaskAction', () => {
    it('devrait créer une tâche avec succès', async () => {
      // Mock de la réponse Supabase
      mockSupabase.single.mockResolvedValue({
        data: { id: 'task-123', title: 'Nouvelle tâche' },
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      formData.append('title', 'Nouvelle tâche');
      formData.append('required_role', 'plombier');
      formData.append('duration_hours', '8');

      const result = await addTaskAction({}, formData);

      expect(result.success).toBe(true);
      expect(mockSupabase.from).toHaveBeenCalledWith('tasks');
      expect(mockSupabase.insert).toHaveBeenCalledWith(
        expect.objectContaining({
          site_id: 'site-123',
          title: 'Nouvelle tâche',
          required_role: 'Plombier',
          duration_hours: 8,
          status: 'pending',
        }),
      );
    });

    it('devrait retourner une erreur si le siteId est manquant', async () => {
      const formData = new FormData();
      formData.append('title', 'Nouvelle tâche');

      const result = await addTaskAction({}, formData);

      expect(result.error).toBe('Site et titre requis.');
      expect(result.success).toBeUndefined();
    });

    it('devrait retourner une erreur si le titre est manquant', async () => {
      const formData = new FormData();
      formData.append('siteId', 'site-123');

      const result = await addTaskAction({}, formData);

      expect(result.error).toBe('Site et titre requis.');
    });

    it('devrait gérer les erreurs Supabase', async () => {
      // Mock de l'insert qui retourne une erreur
      mockSupabase.insert.mockResolvedValue({
        error: { message: 'Erreur de base de données' },
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      formData.append('title', 'Nouvelle tâche');

      const result = await addTaskAction({}, formData);

      expect(result.error).toBe('Erreur de base de données');
      expect(result.success).toBeUndefined();
    });

    it('devrait capitaliser le rôle correctement', async () => {
      mockSupabase.single.mockResolvedValue({
        data: { id: 'task-123' },
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      formData.append('title', 'Tâche');
      formData.append('required_role', 'plombier');

      await addTaskAction({}, formData);

      expect(mockSupabase.insert).toHaveBeenCalledWith(
        expect.objectContaining({
          required_role: 'Plombier',
        }),
      );
    });
  });

  describe('addWorkerAction', () => {
    beforeEach(() => {
      // Mock de l'utilisateur authentifié
      mockSupabase.auth.getUser.mockResolvedValue({
        data: {
          user: {
            id: 'user-123',
            email: 'chef@example.com',
            user_metadata: { full_name: 'Chef de Chantier' },
          },
        },
        error: null,
      });

      // Mock du site
      mockSupabase.maybeSingle.mockResolvedValue({
        data: {
          id: 'site-123',
          name: 'Chantier Test',
          created_by: 'user-123',
        },
        error: null,
      });
    });

    it('devrait créer un nouveau worker avec succès', async () => {
      // Mock pour vérifier que le code d'accès n'existe pas
      mockSupabase.maybeSingle
        .mockResolvedValueOnce({
          // Première vérification : site existe
          data: { id: 'site-123', name: 'Chantier Test' },
          error: null,
        })
        .mockResolvedValueOnce({
          // Deuxième vérification : code d'accès n'existe pas
          data: null,
          error: null,
        });

      // Mock de l'insertion du worker
      mockSupabase.single.mockResolvedValue({
        data: {
          id: 'worker-123',
          access_code: 'ABC123',
        },
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      formData.append('name', 'Jean Dupont');
      formData.append('email', 'jean@example.com');
      formData.append('role', 'plombier');

      const result = await addWorkerAction({}, formData);

      expect(result.success).toBe(true);
      expect(mockSupabase.from).toHaveBeenCalledWith('workers');
      expect(mockSupabase.insert).toHaveBeenCalledWith(
        expect.objectContaining({
          site_id: 'site-123',
          name: 'Jean Dupont',
          email: 'jean@example.com',
          role: 'Plombier',
        }),
      );
    });

    it('devrait retourner une erreur si l\'utilisateur n\'est pas authentifié', async () => {
      mockSupabase.auth.getUser.mockResolvedValue({
        data: { user: null },
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      formData.append('name', 'Jean Dupont');

      const result = await addWorkerAction({}, formData);

      expect(result.error).toBe('Non authentifié.');
    });

    it('devrait retourner une erreur si le site n\'existe pas', async () => {
      // Mock pour le site qui n'existe pas (utilise .single() dans le code)
      mockSupabase.single.mockResolvedValueOnce({
        data: null,
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-inexistant');
      formData.append('name', 'Jean Dupont');

      const result = await addWorkerAction({}, formData);

      expect(result.error).toBe('Chantier non trouvé ou accès refusé.');
    });

    it('devrait retourner une erreur si le nom est manquant pour un nouveau worker', async () => {
      // Mock pour le site (utilise .single() dans le code)
      mockSupabase.single.mockResolvedValueOnce({
        data: { id: 'site-123', name: 'Chantier Test' },
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      // Pas de nom et pas d'existingWorkerId

      const result = await addWorkerAction({}, formData);

      expect(result.error).toBe('Nom requis.');
    });

    it('devrait lier un worker existant au chantier', async () => {
      // Mock pour le site (première requête: vérifier que le site existe)
      mockSupabase.single.mockResolvedValueOnce({
        data: { id: 'site-123', name: 'Chantier Test' },
        error: null,
      });

      // Mock pour le worker existant (deuxième requête: vérifier que le worker existe)
      mockSupabase.single.mockResolvedValueOnce({
        data: {
          id: 'worker-existing',
          name: 'Jean Dupont',
          email: 'jean@example.com',
          role: 'Plombier',
          status: 'approved',
        },
        error: null,
      });

      // Mock pour vérifier que le code d'accès n'existe pas (troisième requête)
      mockSupabase.maybeSingle.mockResolvedValueOnce({
        data: null,
        error: null,
      });

      // Mock pour vérifier que le worker n'est pas déjà assigné (quatrième requête)
      mockSupabase.maybeSingle.mockResolvedValueOnce({
        data: null,
        error: null,
      });

      // Mock de l'insertion (cinquième requête)
      mockSupabase.single.mockResolvedValueOnce({
        data: {
          id: 'worker-new',
          access_code: 'ABC123',
        },
        error: null,
      });

      const formData = new FormData();
      formData.append('siteId', 'site-123');
      formData.append('existingWorkerId', 'worker-existing');

      const result = await addWorkerAction({}, formData);

      expect(result.success).toBe(true);
      // Vérifier que l'insert a été appelé
      expect(mockSupabase.insert).toHaveBeenCalled();
    });
  });
});

