'use client';

import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { TrainingMetrics } from '../types';

interface TrainingChartProps {
  metrics: TrainingMetrics;
}

export default function TrainingChart({ metrics }: TrainingChartProps) {
  if (!metrics || !metrics.episodes || metrics.episodes.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
        <p>No training metrics available yet. Trigger DQN training to see live analytics curves.</p>
      </div>
    );
  }

  const chartData = metrics.episodes.map((ep, i) => ({
    episode: ep,
    reward: metrics.rewards[i],
    length: metrics.lengths[i],
    loss: metrics.losses[i],
    epsilon: metrics.epsilons[i]
  }));

  return (
    <div className="space-y-6">
      {/* Reward Trajectory Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
        <h4 className="text-sm font-semibold text-slate-300 mb-4">Episode Cumulative Reward Trajectory</h4>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="episode" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
              <Legend />
              <Line type="monotone" dataKey="reward" stroke="#10b981" strokeWidth={2} dot={false} name="Reward" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Episode Length & Loss Dual Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">Episode Steps / Path Length</h4>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="episode" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
                <Line type="monotone" dataKey="length" stroke="#3b82f6" strokeWidth={2} dot={false} name="Steps" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">DQN Loss & Epsilon Decay</h4>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="episode" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }} />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="MSE Loss" />
                <Line type="monotone" dataKey="epsilon" stroke="#a855f7" strokeWidth={2} dot={false} name="Epsilon (ε)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
