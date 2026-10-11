'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LocationPoint, LocationSearchResult } from '../types';
import { searchLocations } from '../lib/api';
import { Search, Loader2, X, MapPin, AlertCircle } from 'lucide-react';

interface LocationSearchProps {
  label: string;
  icon?: React.ReactNode;
  placeholder?: string;
  selectedLocation: LocationPoint | null;
  selectedLabel?: string;
  onSelect: (point: LocationPoint, displayName: string) => void;
  onClear: () => void;
  disabled?: boolean;
}

export default function LocationSearch({
  label,
  icon,
  placeholder = 'Search place, landmark, station...',
  selectedLocation,
  selectedLabel,
  onSelect,
  onClear,
  disabled = false,
}: LocationSearchProps) {
  const [inputValue, setInputValue] = useState<string>('');
  const [suggestions, setSuggestions] = useState<LocationSearchResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);

  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeAbortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef<number>(0);

  // Sync displayed input when selectedLabel changes from outside (e.g., map click or swap)
  useEffect(() => {
    if (selectedLabel) {
      setInputValue(selectedLabel);
    } else if (selectedLocation) {
      setInputValue(`${selectedLocation.lat.toFixed(4)}, ${selectedLocation.lng.toFixed(4)}`);
    } else {
      setInputValue('');
    }
  }, [selectedLabel, selectedLocation]);

  // Click outside listener to dismiss dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (activeAbortRef.current) {
        activeAbortRef.current.abort();
      }
    };
  }, []);

  // Debounced search logic
  const performSearch = useCallback(async (query: string) => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setSuggestions([]);
      setIsOpen(false);
      setIsLoading(false);
      setErrorMessage(null);
      setHasSearched(false);
      return;
    }

    // Abort previous in-flight request
    if (activeAbortRef.current) {
      activeAbortRef.current.abort();
    }
    const abortController = new AbortController();
    activeAbortRef.current = abortController;

    const currentReqId = ++requestIdRef.current;
    setIsLoading(true);
    setErrorMessage(null);
    setHasSearched(true);
    setIsOpen(true);
    setHighlightedIndex(-1);

    try {
      const results = await searchLocations(trimmed, abortController.signal);
      // Ensure only the latest response updates state
      if (currentReqId === requestIdRef.current) {
        setSuggestions(results);
        setIsLoading(false);
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return; // Ignore aborted requests
      }
      if (currentReqId === requestIdRef.current) {
        setIsLoading(false);
        setErrorMessage(err.message || 'Failed to fetch location suggestions');
        setSuggestions([]);
      }
    }
  }, []);

  // Debounce input change
  useEffect(() => {
    // If the input matches the currently selected label, don't trigger search
    if (selectedLabel && inputValue === selectedLabel) {
      return;
    }
    if (selectedLocation && inputValue === `${selectedLocation.lat.toFixed(4)}, ${selectedLocation.lng.toFixed(4)}`) {
      return;
    }

    if (!inputValue || inputValue.trim().length < 2) {
      setSuggestions([]);
      setIsOpen(false);
      setIsLoading(false);
      return;
    }

    const timer = setTimeout(() => {
      performSearch(inputValue);
    }, 350);

    return () => clearTimeout(timer);
  }, [inputValue, selectedLabel, selectedLocation, performSearch]);

  const handleSelectResult = (item: LocationSearchResult) => {
    const point: LocationPoint = {
      lat: parseFloat(item.latitude.toFixed(6)),
      lng: parseFloat(item.longitude.toFixed(6)),
    };
    setInputValue(item.name || item.display_name);
    setIsOpen(false);
    setSuggestions([]);
    setErrorMessage(null);
    onSelect(point, item.display_name);
  };

  const handleClear = () => {
    if (activeAbortRef.current) {
      activeAbortRef.current.abort();
    }
    setInputValue('');
    setSuggestions([]);
    setIsOpen(false);
    setErrorMessage(null);
    setHasSearched(false);
    onClear();
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' && suggestions.length > 0) {
        setIsOpen(true);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : suggestions.length - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightedIndex >= 0 && highlightedIndex < suggestions.length) {
        handleSelectResult(suggestions[highlightedIndex]);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <div ref={containerRef} className="relative space-y-1">
      <label className="block text-xs font-semibold uppercase text-slate-400 flex items-center justify-between">
        <span className="flex items-center">
          {icon || <MapPin className="w-3.5 h-3.5 mr-1 text-sky-400" />} {label}
        </span>
        {selectedLocation && (
          <span className="text-[10px] text-slate-500 font-mono lowercase font-normal">
            {selectedLocation.lat.toFixed(4)}, {selectedLocation.lng.toFixed(4)}
          </span>
        )}
      </label>

      <div className="relative flex items-center">
        <div className="absolute left-3 text-slate-500 pointer-events-none">
          <Search className="w-4 h-4" />
        </div>

        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          disabled={disabled}
          placeholder={placeholder}
          aria-label={label}
          aria-autocomplete="list"
          aria-expanded={isOpen}
          role="combobox"
          onFocus={() => {
            if (suggestions.length > 0 || (hasSearched && inputValue.trim().length >= 2)) {
              setIsOpen(true);
            }
          }}
          onChange={(e) => {
            setInputValue(e.target.value);
            // If user changes text away from the selected location, clear current selection
            if (selectedLocation && e.target.value !== selectedLabel) {
              // User is editing the query
            }
          }}
          onKeyDown={handleKeyDown}
          className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-9 py-2.5 text-sm text-white placeholder-slate-500 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 focus:outline-none transition-all disabled:opacity-50"
        />

        <div className="absolute right-2.5 flex items-center space-x-1">
          {isLoading && (
            <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
          )}
          {inputValue && !disabled && (
            <button
              type="button"
              onClick={handleClear}
              aria-label={`Clear ${label}`}
              className="text-slate-500 hover:text-slate-300 p-1 rounded-md transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Dropdown Suggestions */}
      {isOpen && (
        <div
          role="listbox"
          className="absolute z-50 left-0 right-0 mt-1 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl max-h-64 overflow-y-auto divide-y divide-slate-800/60"
        >
          {isLoading && suggestions.length === 0 && (
            <div className="p-4 text-center text-xs text-slate-400 flex items-center justify-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-sky-400" />
              <span>Searching locations via OpenStreetMap...</span>
            </div>
          )}

          {errorMessage && (
            <div className="p-3 text-xs text-rose-300 bg-rose-950/30 flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          )}

          {!isLoading && !errorMessage && suggestions.length === 0 && hasSearched && (
            <div className="p-4 text-center text-xs text-slate-400">
              <p className="font-semibold text-slate-300">No matching locations found</p>
              <p className="text-[11px] text-slate-500 mt-1">Try entering a landmark, road, station, or city name.</p>
            </div>
          )}

          {suggestions.map((item, index) => {
            const isHighlighted = index === highlightedIndex;
            return (
              <button
                key={`${item.latitude}-${item.longitude}-${index}`}
                type="button"
                role="option"
                aria-selected={isHighlighted}
                onMouseEnter={() => setHighlightedIndex(index)}
                onClick={() => handleSelectResult(item)}
                className={`w-full text-left px-3.5 py-2.5 transition-colors flex items-start space-x-2.5 ${
                  isHighlighted ? 'bg-sky-950/60 text-white' : 'hover:bg-slate-800/50 text-slate-200'
                }`}
              >
                <MapPin className="w-4 h-4 text-sky-400 flex-shrink-0 mt-1" />
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-semibold text-white truncate">
                    {item.name}
                  </div>
                  <div className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                    {item.display_name}
                  </div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                    {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
