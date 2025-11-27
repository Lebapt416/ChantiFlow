'use client';

import { useState, useEffect, useRef } from 'react';

interface City {
  nom: string;
  code: string;
  codeDepartement: string;
  codeRegion: string;
  codesPostaux: string[];
  population?: number;
}

interface CityAutocompleteProps {
  value?: string;
  onChange?: (value: string) => void;
  name?: string;
  id?: string;
  placeholder?: string;
  className?: string;
  required?: boolean;
}

export function CityAutocomplete({
  value = '',
  onChange,
  name = 'address',
  id = 'address',
  placeholder = 'Ex: Paris, Lyon, Marseille, Toulouse...',
  className = '',
  required = false,
}: CityAutocompleteProps) {
  const [inputValue, setInputValue] = useState(value);
  const [suggestions, setSuggestions] = useState<City[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  // Synchroniser la valeur externe
  useEffect(() => {
    setInputValue(value);
  }, [value]);

  // Fermer les suggestions quand on clique ailleurs
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Rechercher les codes postaux avec l'API Géo
  const searchCities = async (query: string) => {
    // Si c'est un code postal (5 chiffres), rechercher directement
    const isPostalCode = /^\d{5}$/.test(query.trim());
    
    if (query.length < 2 && !isPostalCode) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    setIsLoading(true);
    try {
      let url: string;
      
      if (isPostalCode) {
        // Recherche par code postal
        url = `https://geo.api.gouv.fr/communes?codePostal=${encodeURIComponent(query.trim())}&limit=20&fields=nom,code,codeDepartement,codeRegion,codesPostaux,population`;
      } else {
        // Recherche par nom de ville
        url = `https://geo.api.gouv.fr/communes?nom=${encodeURIComponent(query)}&limit=20&fields=nom,code,codeDepartement,codeRegion,codesPostaux,population`;
      }

      const response = await fetch(url, {
        headers: {
          Accept: 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Erreur lors de la recherche');
      }

      const data: City[] = await response.json();
      
      // Trier par population (plus grandes villes en premier)
      const sortedData = data.sort((a, b) => {
        const popA = a.population || 0;
        const popB = b.population || 0;
        return popB - popA;
      });

      setSuggestions(sortedData);
      setShowSuggestions(true);
    } catch (error) {
      console.error('Erreur recherche codes postaux:', error);
      setSuggestions([]);
      setShowSuggestions(false);
    } finally {
      setIsLoading(false);
    }
  };

  // Gérer le changement de saisie
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    onChange?.(newValue);
    searchCities(newValue);
  };

  // Sélectionner un code postal
  const handleSelectCity = (city: City) => {
    // Utiliser le premier code postal de la ville
    const postalCode = city.codesPostaux?.[0] || '';
    if (postalCode) {
      setInputValue(postalCode);
      onChange?.(postalCode);
    } else {
      // Fallback sur le nom si pas de code postal
      setInputValue(city.nom);
      onChange?.(city.nom);
    }
    setShowSuggestions(false);
    setSuggestions([]);
  };

  // Formater l'affichage d'une ville avec son code postal
  const formatCityDisplay = (city: City) => {
    const postalCode = city.codesPostaux?.[0] || '';
    return `${postalCode} - ${city.nom}`;
  };

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        id={id}
        name={name}
        value={inputValue}
        onChange={handleInputChange}
        onFocus={() => {
          if (suggestions.length > 0) {
            setShowSuggestions(true);
          }
        }}
        placeholder={placeholder}
        className={className}
        required={required}
        autoComplete="off"
        aria-autocomplete="list"
        aria-expanded={showSuggestions}
        aria-controls="city-suggestions"
      />
      
      {isLoading && (
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600 dark:border-zinc-600 dark:border-t-emerald-400" />
        </div>
      )}

      {showSuggestions && suggestions.length > 0 && (
        <div
          ref={suggestionsRef}
          id="city-suggestions"
          className="absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border border-zinc-200 bg-white shadow-lg dark:border-zinc-700 dark:bg-zinc-900"
          role="listbox"
        >
          {suggestions.map((city, index) => (
            <button
              key={`${city.code}-${index}`}
              type="button"
              onClick={() => handleSelectCity(city)}
              className="w-full px-4 py-2 text-left text-sm text-zinc-900 hover:bg-emerald-50 dark:text-zinc-100 dark:hover:bg-emerald-900/20"
              role="option"
              aria-selected={false}
            >
              <div className="font-medium">{formatCityDisplay(city)}</div>
              {city.population && (
                <div className="text-xs text-zinc-500 dark:text-zinc-400">
                  {city.population.toLocaleString('fr-FR')} habitants
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {showSuggestions && suggestions.length === 0 && inputValue.length >= 2 && !isLoading && (
        <div className="absolute z-50 mt-1 w-full rounded-md border border-zinc-200 bg-white p-4 text-sm text-zinc-600 shadow-lg dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-400">
          Aucun code postal trouvé. Vérifiez l&apos;orthographe ou entrez un code postal à 5 chiffres.
        </div>
      )}
    </div>
  );
}

