import React from 'react';
import { RouteResponse } from '../types';
import { Clock, MapPin, DollarSign, Zap, CheckCircle2, AlertTriangle } from 'lucide-react';

interface RouteCardProps {
  algorithm: string;
  routeData?: RouteResponse;
  isRecommended?: boolean;
}

export default function RouteCard({ algorithm, routeData, isRecommended }: RouteCardProps) {
  if (!routeData) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
        <h3 className="font-bold text-lg text-slate-200">{algorithm}</h3>
        <p className="text-slate-500 text-sm mt-2">No route calculated yet.</p>
      </div>
    );
  }

  if (!routeData.success) {
    return (
      <div className="bg-slate-900 border border-amber-900/50 rounded-xl p-5 shadow-lg">
        <div className="flex items-center space-x-2 text-amber-400 font-bold text-lg mb-2">
          <AlertTriangle className="w-5 h-5" />
          <span>{algorithm}</span>
        </div>
        <p className="text-slate-400 text-xs">{routeData.error || 'Model requires training or no valid path exists.'}</p>
      </div>
    );
  }

  return (
    <div
      className={`relative bg-slate-900 rounded-xl p-5 border transition-all duration-300 shadow-xl ${
        isRecommended
          ? 'border-emerald-500/80 bg-gradient-to-b from-slate-900 via-slate-900 to-emerald-950/20 ring-2 ring-emerald-500/20'
          : 'border-slate-800 hover:border-slate-700'
      }`}
    >
      {isRecommended && (
        <span className="absolute -top-3 right-4 bg-emerald-500 text-slate-950 font-extrabold text-xs px-3 py-1 rounded-full flex items-center gap-1 shadow-md">
          <CheckCircle2 className="w-3.5 h-3.5" /> RECOMMENDED
        </span>
      )}

      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-xl font-bold text-white tracking-wide">{algorithm}</h3>
          <span className="text-xs text-slate-400">
            Efficiency: <strong className="text-sky-400">{routeData.route_efficiency_pct}%</strong>
          </span>
        </div>
        <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2.5 py-1 rounded-md border border-slate-700">
          {routeData.execution_time_ms} ms
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center text-slate-400 text-xs mb-1">
            <MapPin className="w-3.5 h-3.5 mr-1 text-sky-400" /> Distance
          </div>
          <span className="text-lg font-bold text-white">{routeData.total_distance_km} <span className="text-xs font-normal text-slate-400">km</span></span>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center text-slate-400 text-xs mb-1">
            <Clock className="w-3.5 h-3.5 mr-1 text-amber-400" /> Travel Time (ETA)
          </div>
          <span className="text-lg font-bold text-white">{routeData.total_time_min} <span className="text-xs font-normal text-slate-400">min</span></span>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center text-slate-400 text-xs mb-1">
            <DollarSign className="w-3.5 h-3.5 mr-1 text-emerald-400" /> Operational Cost
          </div>
          <span className="text-lg font-bold text-white">₹{routeData.operational_cost}</span>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center text-slate-400 text-xs mb-1">
            <Zap className="w-3.5 h-3.5 mr-1 text-purple-400" /> Route Steps
          </div>
          <span className="text-lg font-bold text-white">{routeData.step_count} <span className="text-xs font-normal text-slate-400">nodes</span></span>
        </div>
      </div>
    </div>
  );
}
