'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { LocationPoint, ComparisonResponse, GraphStatusResponse } from '../types';
import { compareRoutes, fetchGraphStatus } from '../lib/api';
import RouteCard from '../components/RouteCard';
import ComparisonTable from '../components/ComparisonTable';
import LocationSearch from '../components/LocationSearch';
import { Play, Sparkles, Sliders, MapPin, Gauge, ArrowUpDown, AlertCircle } from 'lucide-react';

const MapComponent = dynamic(() => import('../components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center text-slate-400 border border-slate-800 animate-pulse">
      Loading Leaflet Interactive Map...
    </div>
  ),
});

export default function DashboardPage() {
  const [pickup, setPickup] = useState<LocationPoint | null>({ lat: 30.8610, lng: 75.8173 });
  const [destination, setDestination] = useState<LocationPoint | null>({ lat: 30.9330, lng: 75.8893 });
  const [pickupLabel, setPickupLabel] = useState<string>('Gill Road / Model Town Area, Ludhiana');
  const [destinationLabel, setDestinationLabel] = useState<string>('Ludhiana Junction Railway Station');

  const [trafficLevel, setTrafficLevel] = useState<string>('MEDIUM');
  const [objective, setObjective] = useState<string>('balanced');
  
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [graphStatus, setGraphStatus] = useState<GraphStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGraphStatus()
      .then(setGraphStatus)
      .catch((err) => console.error('Failed to load graph status:', err));
  }, []);

  const handleSwapLocations = () => {
    const tempPickup = pickup;
    const tempPickupLabel = pickupLabel;
    setPickup(destination);
    setPickupLabel(destinationLabel);
    setDestination(tempPickup);
    setDestinationLabel(tempPickupLabel);
  };

  const handleMapSelectPickup = (loc: LocationPoint) => {
    setPickup(loc);
    setPickupLabel(`Map Pin (${loc.lat.toFixed(4)}, ${loc.lng.toFixed(4)})`);
  };

  const handleMapSelectDestination = (loc: LocationPoint) => {
    setDestination(loc);
    setDestinationLabel(`Map Pin (${loc.lat.toFixed(4)}, ${loc.lng.toFixed(4)})`);
  };

  const handleOptimize = async () => {
    if (!pickup || !destination) {
      setError('Please select both a valid pickup location and destination before optimizing.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await compareRoutes(pickup, destination, trafficLevel, objective);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to compute routes. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-sky-950/40 p-6 rounded-2xl border border-slate-800 shadow-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white">
            TaxiRoute <span className="text-sky-400">RL</span>
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Intelligent Taxi Route Optimization using Reinforcement Learning (DQN) & Classical Baselines (Dijkstra, A*)
          </p>
        </div>
        {graphStatus && (
          <div className="flex items-center space-x-3 bg-slate-950/80 px-4 py-2.5 rounded-xl border border-slate-800 text-xs">
            <Gauge className={`w-4 h-4 ${graphStatus.is_synthetic ? 'text-amber-400' : 'text-emerald-400'}`} />
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-slate-400">Network:</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wide uppercase ${
                  graphStatus.is_synthetic 
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' 
                    : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                }`}>
                  {graphStatus.source_type || (graphStatus.is_synthetic ? 'SYNTHETIC SIMULATION' : 'REAL OSM ROAD NETWORK')}
                </span>
              </div>
              <div className="text-slate-300 text-[11px] mt-0.5">
                <strong className="text-white">{graphStatus.city_name}</strong> &bull; {graphStatus.num_nodes.toLocaleString()} nodes, {graphStatus.num_edges.toLocaleString()} edges
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Control Bar & Map Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls Column */}
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-5">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Sliders className="w-5 h-5 text-sky-400" /> Route Controls
          </h2>

          {/* Searchable Pickup Location */}
          <LocationSearch
            label="Pickup Location"
            icon={<MapPin className="w-3.5 h-3.5 mr-1 text-emerald-400" />}
            placeholder="Search pickup address, landmark..."
            selectedLocation={pickup}
            selectedLabel={pickupLabel}
            onSelect={(point, name) => {
              setPickup(point);
              setPickupLabel(name);
              setError(null);
            }}
            onClear={() => {
              setPickup(null);
              setPickupLabel('');
            }}
          />

          {/* Swap Locations Button */}
          <div className="flex justify-center -my-1">
            <button
              type="button"
              onClick={handleSwapLocations}
              className="px-3 py-1 bg-slate-950 hover:bg-slate-800 border border-slate-800 rounded-full text-xs text-slate-400 hover:text-white flex items-center space-x-1.5 transition-colors shadow-sm"
              title="Swap pickup and destination"
            >
              <ArrowUpDown className="w-3.5 h-3.5 text-sky-400" />
              <span>Swap Locations</span>
            </button>
          </div>

          {/* Searchable Destination Location */}
          <LocationSearch
            label="Destination Location"
            icon={<MapPin className="w-3.5 h-3.5 mr-1 text-rose-400" />}
            placeholder="Search destination, station, landmark..."
            selectedLocation={destination}
            selectedLabel={destinationLabel}
            onSelect={(point, name) => {
              setDestination(point);
              setDestinationLabel(name);
              setError(null);
            }}
            onClear={() => {
              setDestination(null);
              setDestinationLabel('');
            }}
          />

          {/* Simulated Traffic State Selector */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-400 mb-2">Simulated Traffic State</label>
            <select
              value={trafficLevel}
              onChange={(e) => setTrafficLevel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
            >
              <option value="LOW">LOW Traffic (Free Flow 1.0x)</option>
              <option value="MEDIUM">MEDIUM Traffic (Moderate 1.3x)</option>
              <option value="HIGH">HIGH Congestion (Heavy 1.8x)</option>
            </select>
          </div>

          {/* Optimization Objective Selector */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-400 mb-2">Optimization Objective</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
            >
              <option value="balanced">Balanced (Cost & Compute)</option>
              <option value="time">Minimize Travel Time (ETA)</option>
              <option value="distance">Minimize Travel Distance</option>
              <option value="cost">Minimize Fuel & Operational Cost</option>
            </select>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleOptimize}
            disabled={loading || !pickup || !destination}
            className="w-full bg-gradient-to-r from-sky-500 to-sky-600 hover:from-sky-600 hover:to-sky-700 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-sky-500/25 flex items-center justify-center space-x-2 transition-all disabled:opacity-50"
          >
            {loading ? (
              <span>Calculating Optimal Routes...</span>
            ) : (
              <>
                <Play className="w-5 h-5 fill-current" />
                <span>OPTIMIZE ROUTE</span>
              </>
            )}
          </button>

          {error && (
            <div className="bg-rose-950/50 border border-rose-800 text-rose-300 text-xs p-3 rounded-lg flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Map Column */}
        <div className="lg:col-span-2">
          <MapComponent
            pickup={pickup || { lat: 30.8610, lng: 75.8173 }}
            destination={destination || { lat: 30.9330, lng: 75.8893 }}
            pickupLabel={pickupLabel}
            destinationLabel={destinationLabel}
            onSelectPickup={handleMapSelectPickup}
            onSelectDestination={handleMapSelectDestination}
            routes={result?.algorithms}
            recommended_algorithm={result?.recommended_algorithm}
          />
        </div>
      </div>

      {/* Results Cards & AI Explanation */}
      {result && (
        <div className="space-y-8 animate-fadeIn">
          {/* AI Explainability Panel */}
          <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/30 p-6 rounded-2xl border border-emerald-500/40 shadow-2xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
              <div className="flex items-center space-x-2 text-emerald-400 font-bold text-lg">
                <Sparkles className="w-5 h-5" />
                <span>AI Route Explanation</span>
              </div>
              <span className={`inline-flex items-center self-start sm:self-auto text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wider ${
                (result.graph_source || (graphStatus?.is_synthetic ? 'SYNTHETIC SIMULATION' : 'REAL OSM ROAD NETWORK')).includes('REAL')
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
              }`}>
                {result.graph_source || (graphStatus?.is_synthetic ? 'SYNTHETIC SIMULATION' : 'REAL OSM ROAD NETWORK')}
              </span>
            </div>
            <p className="text-slate-300 text-sm leading-relaxed">{result.explanation}</p>
          </div>

          {/* Algorithm Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(result.algorithms).map(([name, data]) => (
              <RouteCard
                key={name}
                algorithm={name}
                routeData={data}
                isRecommended={name === result.recommended_algorithm}
              />
            ))}
          </div>

          {/* Quantitative Comparison Table */}
          <div>
            <h3 className="text-lg font-bold text-white mb-4">Quantitative Algorithm Benchmark</h3>
            <ComparisonTable algorithms={result.algorithms} recommended_algorithm={result.recommended_algorithm} />
          </div>
        </div>
      )}
    </div>
  );
}
