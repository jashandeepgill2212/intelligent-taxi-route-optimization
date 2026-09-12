'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { LocationPoint, ComparisonResponse, GraphStatusResponse } from '../types';
import { compareRoutes, fetchGraphStatus } from '../lib/api';
import RouteCard from '../components/RouteCard';
import ComparisonTable from '../components/ComparisonTable';
import { Play, Sparkles, Sliders, ShieldCheck, MapPin, Gauge } from 'lucide-react';

const MapComponent = dynamic(() => import('../components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center text-slate-400 border border-slate-800 animate-pulse">
      Loading Leaflet Interactive Map...
    </div>
  ),
});

export default function DashboardPage() {
  const [pickup, setPickup] = useState<LocationPoint>({ lat: 30.8610, lng: 75.8173 });
  const [destination, setDestination] = useState<LocationPoint>({ lat: 30.9330, lng: 75.8893 });
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

  const handleOptimize = async () => {
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
          <div className="flex items-center space-x-2 bg-slate-950/80 px-4 py-2 rounded-xl border border-slate-800 text-xs">
            <Gauge className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="text-slate-400 block">Network Graph:</span>
              <strong className="text-white">{graphStatus.city_name}</strong> ({graphStatus.num_nodes} nodes, {graphStatus.num_edges} edges)
            </div>
          </div>
        )}
      </div>

      {/* Control Bar & Map Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls Column */}
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-6">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Sliders className="w-5 h-5 text-sky-400" /> Route Controls
          </h2>

          {/* Pickup Input */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-400 mb-2 flex items-center">
              <MapPin className="w-3.5 h-3.5 mr-1 text-emerald-400" /> Pickup Location
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                step="0.001"
                value={pickup.lat}
                onChange={(e) => setPickup({ ...pickup, lat: parseFloat(e.target.value) || 0 })}
                className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
              <input
                type="number"
                step="0.001"
                value={pickup.lng}
                onChange={(e) => setPickup({ ...pickup, lng: parseFloat(e.target.value) || 0 })}
                className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
            </div>
          </div>

          {/* Destination Input */}
          <div>
            <label className="block text-xs font-semibold uppercase text-slate-400 mb-2 flex items-center">
              <MapPin className="w-3.5 h-3.5 mr-1 text-rose-400" /> Destination Location
            </label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                step="0.001"
                value={destination.lat}
                onChange={(e) => setDestination({ ...destination, lat: parseFloat(e.target.value) || 0 })}
                className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
              <input
                type="number"
                step="0.001"
                value={destination.lng}
                onChange={(e) => setDestination({ ...destination, lng: parseFloat(e.target.value) || 0 })}
                className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-sky-500 focus:outline-none"
              />
            </div>
          </div>

          {/* Traffic Condition Selector */}
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
            disabled={loading}
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
            <div className="bg-rose-950/50 border border-rose-800 text-rose-300 text-xs p-3 rounded-lg">
              {error}
            </div>
          )}
        </div>

        {/* Map Column */}
        <div className="lg:col-span-2">
          <MapComponent
            pickup={pickup}
            destination={destination}
            onSelectPickup={setPickup}
            onSelectDestination={setDestination}
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
            <div className="flex items-center space-x-2 text-emerald-400 font-bold text-lg mb-2">
              <Sparkles className="w-5 h-5" />
              <span>AI Route Explanation</span>
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
