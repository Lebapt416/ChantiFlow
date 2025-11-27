import { test, expect } from '@playwright/test';

test.describe('Page de connexion', () => {
  test('devrait afficher le formulaire de connexion', async ({ page }) => {
    await page.goto('/login');

    // Vérifier que la page contient les éléments essentiels
    await expect(page).toHaveTitle(/ChantiFlow|Connexion/i);
    
    // Vérifier la présence du formulaire
    const emailInput = page.getByLabel(/email|e-mail/i).or(page.getByPlaceholder(/email/i));
    const passwordInput = page.getByLabel(/mot de passe|password/i).or(page.getByPlaceholder(/mot de passe|password/i));
    // Cibler spécifiquement le bouton de soumission (type="submit") avec le texte "Se connecter"
    const submitButton = page.getByRole('button', { name: 'Se connecter' });

    await expect(emailInput).toBeVisible();
    await expect(passwordInput).toBeVisible();
    await expect(submitButton).toBeVisible();
  });

  test('devrait afficher une erreur avec des identifiants invalides', async ({ page }) => {
    await page.goto('/login');

    // Remplir le formulaire avec des données invalides
    const emailInput = page.getByLabel(/email|e-mail/i).or(page.getByPlaceholder(/email/i));
    const passwordInput = page.getByLabel(/mot de passe|password/i).or(page.getByPlaceholder(/mot de passe|password/i));
    // Cibler spécifiquement le bouton de soumission (type="submit") avec le texte "Se connecter"
    const submitButton = page.getByRole('button', { name: 'Se connecter' });

    await emailInput.fill('invalid@example.com');
    await passwordInput.fill('wrongpassword');
    await submitButton.click();

    // Attendre qu'un message d'erreur apparaisse (peut prendre un peu de temps)
    await page.waitForTimeout(1000);
    
    // Vérifier qu'un message d'erreur est affiché (format flexible)
    const errorMessage = page.getByText(/erreur|invalid|incorrect|échec/i);
    await expect(errorMessage.first()).toBeVisible({ timeout: 5000 });
  });

  test('devrait avoir des champs requis', async ({ page }) => {
    await page.goto('/login');

    const emailInput = page.getByLabel(/email|e-mail/i).or(page.getByPlaceholder(/email/i));
    const passwordInput = page.getByLabel(/mot de passe|password/i).or(page.getByPlaceholder(/mot de passe|password/i));
    // Cibler spécifiquement le bouton de soumission (type="submit") avec le texte "Se connecter"
    const submitButton = page.getByRole('button', { name: 'Se connecter' });

    // Tenter de soumettre sans remplir les champs
    await submitButton.click();

    // Vérifier que les champs sont marqués comme requis (HTML5 validation ou message d'erreur)
    const isRequired = await emailInput.evaluate((el: HTMLInputElement) => {
      return el.hasAttribute('required') || el.validity.valueMissing;
    });
    
    expect(isRequired).toBeTruthy();
  });
});

