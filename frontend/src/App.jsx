import { useState, useEffect } from "react";
import { fetchHealth } from "./api";
import SearchTab  from "./tabs/SearchTab";
import CompareTab from "./tabs/CompareTab";
import RagTab     from "./tabs/RagTab";
import MetricsTab from "./tabs/MetricsTab";

const TABS = [
  {
    id: "search",
    name: "Search",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/>
        <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-4.35-4.35"/>
      </svg>
    ),
  },
  {
    id: "compare",
    name: "Compare",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"/>
      </svg>
    ),
  },
  {
    id: "rag",
    name: "RAG",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"/>
      </svg>
    ),
  },
  {
    id: "metrics",
    name: "Metrics",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"/>
      </svg>
    ),
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("search");
  const [health, setHealth]       = useState({ status: "checking", data: null });

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const data = await fetchHealth();
        if (mounted) setHealth({ status: "ok", data });
      } catch {
        if (mounted) setHealth({ status: "error", data: null });
      }
    };
    check();
    const id = setInterval(check, 10_000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-600/30 selection:text-indigo-200">
      {/* ── Navigation Bar ── */}
      <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-[1400px] mx-auto px-4 h-16 flex items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-black text-lg select-none">
              Q
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">Quest</span>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  IR &amp; RAG
                </span>
              </div>
              <p className="text-[11px] text-slate-500 leading-none mt-0.5">LoTTE Lifestyle Corpus</p>
            </div>
          </div>

          {/* Tab Navigation */}
          <nav className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800" role="tablist">
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  role="tab"
                  aria-selected={isActive}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                    isActive
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  {tab.icon}
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </nav>

          {/* Backend health badge */}
          <div className="flex-shrink-0">
            {health.status === "ok" && (
              <div
                id="backend-status-badge"
                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-950/40 border border-emerald-800/50 text-xs font-medium text-emerald-300"
                title={`Backend online · TF-IDF: ${health.data?.indexes?.tfidf?.num_docs ?? 0}, BM25: ${health.data?.indexes?.bm25?.num_docs ?? 0}`}
              >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"/>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"/>
                </span>
                Backend Online
              </div>
            )}
            {health.status === "checking" && (
              <div id="backend-status-badge" className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-950/40 border border-amber-800/50 text-xs font-medium text-amber-300">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"/>
                Connecting…
              </div>
            )}
            {health.status === "error" && (
              <div id="backend-status-badge" className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-950/40 border border-red-800/50 text-xs font-medium text-red-300">
                <span className="w-2 h-2 rounded-full bg-red-500"/>
                Backend Offline
              </div>
            )}
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 py-8">
        {activeTab === "search"  && <SearchTab />}
        {activeTab === "compare" && <CompareTab />}
        {activeTab === "rag"     && <RagTab />}
        {activeTab === "metrics" && <MetricsTab />}
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-slate-800/60 py-4 text-center text-xs text-slate-600">
        <div className="max-w-[1400px] mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Quest — Information Retrieval &amp; RAG · LoTTE lifestyle dev benchmark</span>
          <span>FastAPI · React + Vite · Tailwind · Recharts</span>
        </div>
      </footer>
    </div>
  );
}
