'use client';

import React, { useState, useEffect } from 'react';
import { TrainingStatusResponse } from '../../types';
import { fetchTrainingStatus, startDqnTraining } from '../../lib/api';
import TrainingChart from '../../components/TrainingChart';
import { Activity, Play, CheckCircle2, AlertCircle, RefreshCw, Cpu } from 'lucide-react';

export default function TrainingPage() {
  const [statusData, setStatusData] = useState<TrainingStatusResponse | null>(null);
  const [episodes, setEpisodes] = useState(150);
  const [batchSize, setBatchSize] = useState(32);
  const [isTriggering, setIsTriggering] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = async () => {
    try {
      const data = await fetchTrainingStatus();
      setStatusData(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadStatus();
    const timer = setInterval(loadStatus, 4000);
    return () => clearInterval(timer);
  }, []);

  const handleStartTraining = async () => {
    setIsTriggering(true);
    setError(null);
    try {
      await startDqnTraining(episodes, batchSize);
      await loadStatus();
    } catch (err: any) {
      setError(err.message || 'Failed to start training.');
    } finally {
      setIsTriggering(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Activity className="w-6 h-6 text-emerald-400" /> Reinforcement Learning Training Analytics
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Monitor real-time Deep Q-Network (DQN) policy optimization, episode reward trajectories, and checkpoint state.
          </p>
        </div>

        <button
          onClick={loadStatus}
          className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3.5 py-2 rounded-xl text-xs flex items-center gap-1.5 border border-slate-700"
        >
          <RefreshCw className="w-3.5 h-3.5" /> REFRESH
        </button>
      </div>

      {/* Metrics Cards Row */}
      {statusData && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 font-semibold uppercase">Training Status</span>
            <div className="flex items-center space-x-2 mt-2">
              {statusData.is_training ? (
                <span className="flex items-center text-amber-400 font-bold text-lg animate-pulse">
                  <RefreshCw className="w-5 h-5 mr-1.5 animate-spin" /> Training...
                </span>
              ) : statusData.model_checkpoint_exists ? (
                <span className="flex items-center text-emerald-400 font-bold text-lg">
                  <CheckCircle2 className="w-5 h-5 mr-1.5" /> Checkpoint Ready
                </span>
              ) : (
                <span className="flex items-center text-rose-400 font-bold text-lg">
                  <AlertCircle className="w-5 h-5 mr-1.5" /> Not Trained
                </span>
              )}
            </div>
          </div>

          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 font-semibold uppercase">Latest Win Rate</span>
            <div className="text-2xl font-extrabold text-white mt-1">
              {statusData.metrics?.final_success_rate !== undefined
                ? `${statusData.metrics.final_success_rate}%`
                : '—'}
            </div>
          </div>

          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 font-semibold uppercase">Total Training Time</span>
            <div className="text-2xl font-extrabold text-white mt-1">
              {statusData.metrics?.training_time_sec !== undefined
                ? `${statusData.metrics.training_time_sec}s`
                : '—'}
            </div>
          </div>

          <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shadow-lg">
            <span className="text-xs text-slate-400 font-semibold uppercase">Saved Checkpoint</span>
            <div className="text-sm font-mono text-slate-300 mt-2 truncate">
              {statusData.model_checkpoint_exists ? 'dqn_model.pt' : 'None'}
            </div>
          </div>
        </div>
      )}

      {/* Training Controls Box */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-6">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-sky-400" /> Trigger Model Training
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase mb-2">Training Episodes</label>
            <input
              type="number"
              min="10"
              max="2000"
              value={episodes}
              onChange={(e) => setEpisodes(parseInt(e.target.value) || 150)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white"
            />
            <span className="text-xs text-slate-500 mt-1 block">Recommended: 150 (Fast Dev Mode) or 500 (Full Mode)</span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase mb-2">Replay Batch Size</label>
            <input
              type="number"
              min="8"
              max="256"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 32)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white"
            />
            <span className="text-xs text-slate-500 mt-1 block">Batch size for experience replay gradient steps</span>
          </div>
        </div>

        <button
          onClick={handleStartTraining}
          disabled={isTriggering || statusData?.is_training}
          className="bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-6 rounded-xl flex items-center justify-center gap-2 shadow-lg disabled:opacity-50"
        >
          <Play className="w-5 h-5 fill-current" />
          <span>{statusData?.is_training ? 'Training in Progress...' : 'START DQN TRAINING'}</span>
        </button>

        {error && <p className="text-rose-400 text-xs">{error}</p>}
      </div>

      {/* Analytics Charts */}
      {statusData?.metrics && (
        <div>
          <h3 className="text-lg font-bold text-white mb-4">Training Curves & Trajectory</h3>
          <TrainingChart metrics={statusData.metrics} />
        </div>
      )}
    </div>
  );
}
