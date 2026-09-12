import React from 'react';

export default function Footer() {
  return (
    <footer className="bg-slate-950 border-t border-slate-800 text-slate-400 py-6 text-sm">
      <div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <p className="font-semibold text-slate-200">TaxiRoute RL — Reinforcement Learning Route Optimizer</p>
          <p className="text-xs text-slate-400">B.Tech CSE / AI&DS Major Project</p>
        </div>
        <div className="text-xs text-right">
          <p><span className="text-sky-400 font-medium">Jashandeep Singh</span> (72510438)</p>
          <p><span className="text-sky-400 font-medium">Harshpreet Singh Malhi</span> (72520287)</p>
        </div>
      </div>
    </footer>
  );
}
