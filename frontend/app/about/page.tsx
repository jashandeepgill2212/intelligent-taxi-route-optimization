import React from 'react';
import { BookOpen, Users, Cpu, Layers, Target, ShieldCheck } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8 py-4">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-sky-950/40 p-8 rounded-2xl border border-slate-800 shadow-2xl">
        <span className="text-xs font-semibold uppercase text-sky-400 tracking-wider">B.Tech Major Project</span>
        <h1 className="text-3xl font-extrabold text-white mt-1">
          INTELLIGENT TAXI ROUTE OPTIMIZATION USING REINFORCEMENT LEARNING
        </h1>
        <p className="text-slate-400 text-sm mt-3 leading-relaxed">
          An end-to-end intelligent transportation system that combines Deep Q-Networks (DQN) with classical graph search algorithms (Dijkstra and A*) to discover optimal taxi routes under dynamic urban traffic conditions.
        </p>
      </div>

      {/* Team Info */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Users className="w-5 h-5 text-sky-400" /> Project Team
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-500 uppercase block font-semibold">Team Member 1</span>
            <strong className="text-lg text-white font-bold block mt-1">Jashandeep Singh</strong>
            <span className="text-sky-400 text-xs font-mono">Roll No: 72510438</span>
          </div>
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-500 uppercase block font-semibold">Team Member 2</span>
            <strong className="text-lg text-white font-bold block mt-1">Harshpreet Singh Malhi</strong>
            <span className="text-sky-400 text-xs font-mono">Roll No: 72520287</span>
          </div>
        </div>
      </div>

      {/* Problem & Objectives */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Target className="w-4 h-4 text-rose-400" /> Problem Statement
          </h3>
          <p className="text-slate-300 text-xs leading-relaxed">
            Conventional routing engines rely strictly on static shortest-path heuristics that often fail under dynamic traffic bottlenecks, time-of-day variations, and non-linear fuel costs. This project formulates urban taxi navigation as a Markov Decision Process (MDP) to learn adaptive policies via Reinforcement Learning.
          </p>
        </div>

        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" /> Key Objectives
          </h3>
          <ul className="text-slate-300 text-xs space-y-1.5 list-disc list-inside">
            <li>Formulate Gymnasium-compatible road network environment with action masking.</li>
            <li>Implement PyTorch Deep Q-Network (DQN) with Replay Buffer & Target Network.</li>
            <li>Benchmark RL performance against exact Dijkstra & heuristic A* algorithms.</li>
            <li>Provide interactive Next.js Leaflet GIS dashboard with deterministic AI explainability.</li>
          </ul>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-purple-400" /> Technology Stack
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
            <strong className="text-sky-400 block font-bold">Frontend</strong>
            <span className="text-slate-300">Next.js 14, Tailwind CSS, Leaflet</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
            <strong className="text-emerald-400 block font-bold">Backend</strong>
            <span className="text-slate-300">FastAPI, Pydantic, Uvicorn</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
            <strong className="text-amber-400 block font-bold">RL & ML</strong>
            <span className="text-slate-300">PyTorch, Gymnasium, Stable-Baselines3</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
            <strong className="text-purple-400 block font-bold">GIS & Graphs</strong>
            <span className="text-slate-300">NetworkX, OSMnx, OpenStreetMap</span>
          </div>
        </div>
      </div>
    </div>
  );
}
