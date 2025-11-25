import '@testing-library/jest-dom';
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Cleanup après chaque test
afterEach(() => {
  cleanup();
});

// Extend Vitest's expect avec jest-dom matchers
expect.extend({
  // Ajouter des matchers personnalisés si nécessaire
});

