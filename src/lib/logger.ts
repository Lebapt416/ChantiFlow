/**
 * Logger structuré pour l'application ChantiFlow
 * Format JSON prêt pour intégration avec Sentry, Datadog, etc.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogContext {
  [key: string]: unknown;
}

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: LogContext;
  error?: {
    name: string;
    message: string;
    stack?: string;
  };
  userId?: string;
  siteId?: string;
  requestId?: string;
}

class Logger {
  private isDevelopment = process.env.NODE_ENV === 'development';

  /**
   * Log un message avec un niveau donné
   */
  private log(
    level: LogLevel,
    message: string,
    context?: LogContext,
    error?: Error,
  ): void {
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
    };

    if (error) {
      entry.error = {
        name: error.name,
        message: error.message,
        stack: error.stack,
      };
    }

    // En production, logger en JSON pour faciliter l'agrégation
    // En développement, logger de manière lisible
    if (this.isDevelopment) {
      const colorMap: Record<LogLevel, string> = {
        debug: '\x1b[36m', // Cyan
        info: '\x1b[32m', // Green
        warn: '\x1b[33m', // Yellow
        error: '\x1b[31m', // Red
      };
      const reset = '\x1b[0m';
      const color = colorMap[level] || reset;

      console.log(
        `${color}[${level.toUpperCase()}]${reset} ${message}`,
        context ? JSON.stringify(context, null, 2) : '',
        error ? `\n${error.stack}` : '',
      );
    } else {
      // Format JSON structuré pour production
      console.log(JSON.stringify(entry));
    }

    // Intégration Sentry (si installé)
    if (level === 'error') {
      // Côté serveur (Node.js)
      if (typeof window === 'undefined' && typeof process !== 'undefined') {
        try {
          // eslint-disable-next-line @typescript-eslint/no-var-requires
          const Sentry = require('@sentry/nextjs');
          if (Sentry && error) {
            Sentry.captureException(error, {
              contexts: { custom: context },
              tags: {
                userId: entry.userId,
                siteId: entry.siteId,
              },
            });
          } else if (Sentry && !error) {
            Sentry.captureMessage(message, {
              level: 'error',
              contexts: { custom: context },
              tags: {
                userId: entry.userId,
                siteId: entry.siteId,
              },
            });
          }
        } catch {
          // Sentry non installé ou erreur d'import - ignorer silencieusement
        }
      }
      
      // Côté client (browser)
      if (typeof window !== 'undefined') {
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const Sentry = (window as any).Sentry;
          if (Sentry) {
            if (error) {
              Sentry.captureException(error, {
                contexts: { custom: context },
                tags: {
                  userId: entry.userId,
                  siteId: entry.siteId,
                },
              });
            } else {
              Sentry.captureMessage(message, {
                level: 'error',
                contexts: { custom: context },
                tags: {
                  userId: entry.userId,
                  siteId: entry.siteId,
                },
              });
            }
          }
        } catch {
          // Sentry non disponible - ignorer silencieusement
        }
      }
    }
  }

  /**
   * Log un message de debug
   */
  debug(message: string, context?: LogContext): void {
    if (this.isDevelopment) {
      this.log('debug', message, context);
    }
  }

  /**
   * Log un message d'information
   */
  info(message: string, context?: LogContext): void {
    this.log('info', message, context);
  }

  /**
   * Log un avertissement
   */
  warn(message: string, context?: LogContext, error?: Error): void {
    this.log('warn', message, context, error);
  }

  /**
   * Log une erreur
   */
  error(message: string, context?: LogContext, error?: Error): void {
    this.log('error', message, context, error);
  }

  /**
   * Log une erreur avec contexte utilisateur
   */
  errorWithUser(
    message: string,
    userId: string,
    context?: LogContext,
    error?: Error,
  ): void {
    this.log('error', message, { ...context, userId }, error);
  }

  /**
   * Log une erreur avec contexte chantier
   */
  errorWithSite(
    message: string,
    siteId: string,
    context?: LogContext,
    error?: Error,
  ): void {
    this.log('error', message, { ...context, siteId }, error);
  }
}

// Export d'une instance singleton
export const logger = new Logger();

