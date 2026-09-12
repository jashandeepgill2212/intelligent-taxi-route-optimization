import React from 'react';
import { RouteResponse } from '../types';

interface ComparisonTableProps {
  algorithms: Record<string, RouteResponse>;
  recommended_algorithm?: string;
}

export default function ComparisonTable({ algorithms, recommended_algorithm }: ComparisonTableProps) {
  return (
    <div className="overflow-x-auto bg-slate-900 border border-slate-800 rounded-xl shadow-xl">
      <table className="w-full text-left text-sm text-slate-300">
        <thead className="bg-slate-950 text-slate-400 uppercase text-xs tracking-wider border-b border-slate-800">
          <tr>
            <th className="py-3.5 px-4 font-semibold">Algorithm</th>
            <th className="py-3.5 px-4 font-semibold">Status</th>
            <th className="py-3.5 px-4 font-semibold">Distance (km)</th>
            <th className="py-3.5 px-4 font-semibold">Travel Time (min)</th>
            <th className="py-3.5 px-4 font-semibold">Cost (₹)</th>
            <th className="py-3.5 px-4 font-semibold">Route Steps</th>
            <th className="py-3.5 px-4 font-semibold">Efficiency</th>
            <th className="py-3.5 px-4 font-semibold">Compute (ms)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {Object.entries(algorithms).map(([name, data]) => {
            const isWinner = name === recommended_algorithm;
            return (
              <tr
                key={name}
                className={`transition-colors ${
                  isWinner ? 'bg-emerald-950/30 text-emerald-300 font-semibold' : 'hover:bg-slate-800/50'
                }`}
              >
                <td className="py-3.5 px-4 flex items-center space-x-2">
                  <span>{name}</span>
                  {isWinner && (
                    <span className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 text-[10px] px-2 py-0.5 rounded-full font-bold">
                      BEST
                    </span>
                  )}
                </td>
                <td className="py-3.5 px-4">
                  {data.success ? (
                    <span className="text-emerald-400 font-medium">Completed</span>
                  ) : (
                    <span className="text-amber-400 font-medium">Requires Training</span>
                  )}
                </td>
                <td className="py-3.5 px-4">{data.success ? `${data.total_distance_km} km` : '—'}</td>
                <td className="py-3.5 px-4">{data.success ? `${data.total_time_min} min` : '—'}</td>
                <td className="py-3.5 px-4">{data.success ? `₹${data.operational_cost}` : '—'}</td>
                <td className="py-3.5 px-4">{data.success ? data.step_count : '—'}</td>
                <td className="py-3.5 px-4">{data.success ? `${data.route_efficiency_pct}%` : '—'}</td>
                <td className="py-3.5 px-4 font-mono">{data.execution_time_ms} ms</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
