'use client';

import React, { useState, useEffect } from 'react';
import { ComparisonResponse } from '../../types';
import { compareRoutes } from '../../lib/api';
import ComparisonTable from '../../components/ComparisonTable';
import { GitCompare, RefreshCw, Layers } from 'lucide-react';

export default function ComparePage() {
  const [source, setSource] = useState({ lat: 30.8610, lng: 75.8173 });
  const [destination, setDestination] = useState({ lat: 30.9330, lng: 75.8893 });
  const [traffic, setTraffic] = useState('HIGH');
  const [objective, setObjective] = useState('balanced');
  
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);

  const runBenchmark = async () => {
    setLoading(true);
    try {
      const res = await compareRoutes(source, destination, traffic, objective);
      setComparison(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runBenchmark();
  }, []);

  return (
    <div className="space-y-8">
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <GitCompare className="w-6 h-6 text-sky-400" /> Deep Algorithm Comparison
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Empirical benchmark of Dijkstra's Algorithm, A* Search, and Deep Q-Network (DQN) under identical road topologies.
          </p>
        </div>
        <button
          onClick={runBenchmark}
          disabled={loading}
          className="bg-sky-600 hover:bg-sky-700 text-white font-bold py-2.5 px-4 rounded-xl flex items-center gap-2 text-sm shadow-lg disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>RUN BENCHMARK</span>
        </button>
      </div>

      {/* Filter Matrix */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase mb-2">Simulated Traffic Condition</label>
          <select
            value={traffic}
            onChange={(e) => setTraffic(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white"
          >
            <option value="LOW">LOW Traffic (Free Flow 1.0x)</option>
            <option value="MEDIUM">MEDIUM Traffic (Moderate 1.3x)</option>
            <option value="HIGH">HIGH Traffic (Heavy 1.8x)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-400 uppercase mb-2">Optimization Objective</label>
          <select
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white"
          >
            <option value="balanced">Balanced (Cost & Compute)</option>
            <option value="time">Minimize Travel Time (ETA)</option>
            <option value="distance">Minimize Distance</option>
            <option value="cost">Minimize Operational Fuel Cost</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={runBenchmark}
            className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold py-2 px-4 rounded-lg border border-slate-700"
          >
            Apply Filters & Re-evaluate
          </button>
        </div>
      </div>

      {comparison && (
        <div className="space-y-8">
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800">
            <h3 className="text-lg font-bold text-white mb-2">Algorithm Theoretical Breakdown</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4 text-xs">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <h4 className="font-bold text-sky-400 text-sm mb-1">Dijkstra's Algorithm</h4>
                <p className="text-slate-400">Guarantees exact mathematical shortest path on graph with non-negative edge weights. Serves as exact lower bound for cost comparison.</p>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <h4 className="font-bold text-amber-400 text-sm mb-1">A* Search Algorithm</h4>
                <p className="text-slate-400">Uses Haversine distance heuristic to guide state space search toward destination node, accelerating path computation.</p>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <h4 className="font-bold text-emerald-400 text-sm mb-1">Deep Q-Network (DQN)</h4>
                <p className="text-slate-400">Reinforcement Learning agent trained via Q-learning with Experience Replay. Learns generalized routing policy across dynamic traffic states.</p>
              </div>
            </div>
          </div>

          <ComparisonTable algorithms={comparison.algorithms} recommended_algorithm={comparison.recommended_algorithm} />
        </div>
      )}
    </div>
  );
}
