'use client';

import { useState, useEffect } from 'react';
import { Cloud, CloudRain, Sun, Lock } from 'lucide-react';
import Link from 'next/link';

type WeatherData = {
  temperature: number;
  precipitation: number;
  weatherCode: number;
};

type WeatherWidgetProps = {
  location?: string; // Adresse ou ville du chantier
  isLocked?: boolean; // Si true, affiche flouté avec cadenas
};

const getWeatherIcon = (code: number, precipitation: number) => {
  if (precipitation > 0.5) {
    return <CloudRain className="h-8 w-8 text-blue-500" />;
  }
  if (code >= 1 && code <= 3) {
    return <Cloud className="h-8 w-8 text-gray-400" />;
  }
  return <Sun className="h-8 w-8 text-yellow-500" />;
};

const getWeatherLabel = (code: number, precipitation: number) => {
  if (precipitation > 0.5) {
    return 'Pluie';
  }
  if (code >= 1 && code <= 3) {
    return 'Nuageux';
  }
  return 'Ensoleillé';
};

export function WeatherWidget({ location, isLocked = false }: WeatherWidgetProps) {
  const [weather, setWeather] = useState<{
    today: WeatherData | null;
    tomorrow: WeatherData | null;
  }>({ today: null, tomorrow: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [city, setCity] = useState<string>(location || 'Paris');
  const [showCityInput, setShowCityInput] = useState(!location);

  // Géocodage de la ville en coordonnées
  const geocodeCity = async (cityName: string): Promise<{ lat: number; lon: number } | null> => {
    try {
      // Utiliser l'API de géocodage gratuite Nominatim
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(cityName)}&limit=1`,
        {
          headers: {
            'User-Agent': 'ChantiFlow Weather Widget',
          },
        },
      );
      const data = await response.json();
      if (data && data.length > 0) {
        return {
          lat: parseFloat(data[0].lat),
          lon: parseFloat(data[0].lon),
        };
      }
      return null;
    } catch (err) {
      console.error('Erreur géocodage:', err);
      return null;
    }
  };

  // Récupérer la météo depuis OpenMeteo
  const fetchWeather = async (cityName: string) => {
    setLoading(true);
    setError(null);

    try {
      const coords = await geocodeCity(cityName);
      if (!coords) {
        setError('Ville non trouvée');
        setLoading(false);
        return;
      }

      const response = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${coords.lat}&longitude=${coords.lon}&daily=temperature_2m_max,precipitation_sum,weathercode&timezone=Europe/Paris&forecast_days=2`,
      );

      if (!response.ok) {
        throw new Error('Erreur API météo');
      }

      const data = await response.json();
      const daily = data.daily;

      setWeather({
        today: {
          temperature: Math.round(daily.temperature_2m_max[0]),
          precipitation: daily.precipitation_sum[0],
          weatherCode: daily.weathercode[0],
        },
        tomorrow: {
          temperature: Math.round(daily.temperature_2m_max[1]),
          precipitation: daily.precipitation_sum[1],
          weatherCode: daily.weathercode[1],
        },
      });
    } catch (err) {
      console.error('Erreur météo:', err);
      setError('Impossible de charger la météo');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (city && !showCityInput) {
      fetchWeather(city);
    }
  }, [city, showCityInput]);

  const handleCitySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (city.trim()) {
      setShowCityInput(false);
      fetchWeather(city.trim());
    }
  };

  if (isLocked) {
    return (
      <div className="relative rounded-2xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-zinc-900/80 backdrop-blur-sm">
          <div className="text-center">
            <Lock className="mx-auto h-8 w-8 text-white" />
            <p className="mt-2 text-sm font-semibold text-white">Fonctionnalité Plus</p>
            <Link
              href="/account"
              className="mt-2 inline-block rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white transition hover:bg-emerald-700"
            >
              Passer à Plus
            </Link>
          </div>
        </div>
        <div className="opacity-30 blur-sm">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">Météo du chantier</h3>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
              <p className="text-xs text-zinc-500">Aujourd'hui</p>
              <div className="mt-2 flex items-center gap-2">
                <Sun className="h-6 w-6 text-yellow-500" />
                <span className="text-2xl font-bold">20°C</span>
              </div>
              <p className="mt-1 text-xs text-zinc-500">0% pluie</p>
            </div>
            <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
              <p className="text-xs text-zinc-500">Demain</p>
              <div className="mt-2 flex items-center gap-2">
                <Cloud className="h-6 w-6 text-gray-400" />
                <span className="text-2xl font-bold">18°C</span>
              </div>
              <p className="mt-1 text-xs text-zinc-500">10% pluie</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (showCityInput) {
    return (
      <div className="rounded-2xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">Météo du chantier</h3>
        <form onSubmit={handleCitySubmit} className="mt-4">
          <input
            type="text"
            value={city}
            onChange={(e) => setCity(e.target.value)}
            placeholder="Entrez la ville du chantier"
            className="w-full rounded-lg border border-zinc-300 px-4 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-800 dark:text-white"
          />
          <button
            type="submit"
            className="mt-2 w-full rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-emerald-700"
          >
            Charger la météo
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">Météo du chantier</h3>
        <button
          onClick={() => setShowCityInput(true)}
          className="text-xs text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
        >
          Changer
        </button>
      </div>

      {loading && (
        <div className="py-8 text-center text-sm text-zinc-500">Chargement de la météo...</div>
      )}

      {error && (
        <div className="py-8 text-center text-sm text-rose-600">
          {error}
          <button
            onClick={() => fetchWeather(city)}
            className="ml-2 text-emerald-600 hover:underline"
          >
            Réessayer
          </button>
        </div>
      )}

      {!loading && !error && weather.today && weather.tomorrow && (
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Aujourd'hui</p>
            <div className="mt-2 flex items-center gap-2">
              {getWeatherIcon(weather.today.weatherCode, weather.today.precipitation)}
              <span className="text-2xl font-bold text-zinc-900 dark:text-white">
                {weather.today.temperature}°C
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-500">
              {getWeatherLabel(weather.today.weatherCode, weather.today.precipitation)}
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              Risque pluie: {weather.today.precipitation > 0.5 ? 'Élevé' : 'Faible'} (
              {Math.round(weather.today.precipitation * 10)}%)
            </p>
          </div>

          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-700 dark:bg-zinc-800">
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Demain</p>
            <div className="mt-2 flex items-center gap-2">
              {getWeatherIcon(weather.tomorrow.weatherCode, weather.tomorrow.precipitation)}
              <span className="text-2xl font-bold text-zinc-900 dark:text-white">
                {weather.tomorrow.temperature}°C
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-500">
              {getWeatherLabel(weather.tomorrow.weatherCode, weather.tomorrow.precipitation)}
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              Risque pluie: {weather.tomorrow.precipitation > 0.5 ? 'Élevé' : 'Faible'} (
              {Math.round(weather.tomorrow.precipitation * 10)}%)
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

