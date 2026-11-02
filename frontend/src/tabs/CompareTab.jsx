import { useState, useEffect, useCallback } from "react";
import { fetchCompare, fetchSampleQueries } from "../api";

const RANKERS = [
  { id: "tfidf",  label: "TF-IDF",  color: "text-sky-400",    border: "border-sky-500/30",    bg: "bg-sky-500/5",    badge: "bg-sky-950 border-sky-700/50 text-sky-300" },
  { id: "bm25",   label: "BM25",    color: "text-violet-400", border: "border-violet-500/30", bg: "bg-violet-500/5", badge: "bg-violet-950 border-violet-700/50 text-violet-300" },
  { id: "dense",  label: "Dense",   color: "text-emerald-400",border: "border-emerald-500/30",bg: "bg-emerald-500/5",badge: "bg-emerald-950 border-emerald-700/50 text-emerald-300" },
  { id: "hybrid", label: "Hybrid",  color: "text-amber-400",  border: "border-amber-500/30",  bg: "bg-amber-500/5",  badge: "bg-amber-950 border-amber-700/50 text-amber-300" },
];

function buildOverlapSet(data) {
  // Count how many ranker lists contain each PID
  const counts = {};
  for (const r of RANKERS) {
    const hits = data[r.id] || [];
    for (const hit of hits) {
      counts[hit.pid] = (counts[hit.pid] || 0) + 1;
    }
  }
  return counts;
}

export default function CompareTab() {
  const [query, setQuery]           = useState("");
  const [data, setData]             = useState(null);
  const [latency, setLatency]       = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [samples, setSamples]       = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [overlapCounts, setOverlapCounts]   = useState({});

  useEffect(() => {
    fetchSampleQueries()
      .then((d) => setSamples(d.queries || []))
      .catch(() => {});
  }, []);

  const runCompare = useCallback(async (q, sample) => {
    const target = (q ?? query).trim();
    if (!target) return;
    setLoading(true);
    setError(null);
    setData(null);
    setOverlapCounts({});
    setSelectedSample(sample ?? null);
    try {
      const res = await fetchCompare(target, 10);
      setData(res);
      setLatency(res.latency_ms);
      setOverlapCounts(buildOverlapSet(res));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [query]);

  const handleSampleChange = (e) => {
    const qid = Number(e.target.value);
    if (!qid) return;
    const s = samples.find((x) => x.qid === qid);
    if (s) {
      setQuery(s.text);
      runCompare(s.text, s);
    }
  };

  return (
    <div className="w-full max-w-[1400px] mx-auto space-y-6">
      {/* ── Controls ── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
          {/* Sample query dropdown */}
          {samples.length > 0 && (
            <div className="relative flex-shrink-0">
              <select
                id="compare-sample-select"
                onChange={handleSampleChange}
                className="appearance-none pl-3 pr-8 py-2.5 bg-slate-950 border border-slate-700 rounded-lg text-sm text-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
                defaultValue=""
              >
                <option value="" disabled>Sample queries…</option>
                {samples.map((s) => (
                  <option key={s.qid} value={s.qid}>
                    QID {s.qid}: {s.text.length > 55 ? s.text.slice(0, 52) + "…" : s.text}
                  </option>
                ))}
              </select>
              <div className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500">▾</div>
            </div>
          )}

          {/* Free-text input */}
          <div className="relative flex-1 flex items-center gap-2">
            <div className="relative flex-1">
              <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <circle cx="11" cy="11" r="8"/><path strokeLinecap="round" strokeLinejoin="round" d="m21 21-4.35-4.35"/>
                </svg>
              </div>
              <input
                id="compare-input"
                type="text"
                className="w-full pl-9 pr-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-lg text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                placeholder="Or type a custom query…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runCompare()}
              />
            </div>
            <button
              id="compare-submit"
              disabled={loading || !query.trim()}
              onClick={() => runCompare()}
              className="flex-shrink-0 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-all flex items-center gap-2"
            >
              {loading ? (
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"/>
                </svg>
              )}
              Compare
            </button>
          </div>

          {latency !== null && !loading && (
            <div className="flex-shrink-0 flex items-center gap-1.5 px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block"/>
              <span className="font-mono font-semibold text-emerald-400">{latency.toFixed(1)} ms</span>
            </div>
          )}
        </div>

        {/* Legend */}
        {data && (
          <div className="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap gap-4 text-xs text-slate-400">
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-amber-400/20 border border-amber-400/50"/>
              <span>Appears in ≥2 rankers</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm bg-green-400/15 border border-green-400/50"/>
              <span>Known relevant (ground truth)</span>
            </div>
            {selectedSample && (
              <div className="ml-auto text-slate-500">
                QID {selectedSample.qid}: {selectedSample.relevant_pids.length} known relevant passages
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="p-4 bg-red-950/50 border border-red-800/70 rounded-xl text-red-200 text-sm flex gap-3">
          <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <span><strong className="text-red-100">Error: </strong>{error}</span>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {loading && (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          {RANKERS.map((r) => (
            <div key={r.id} className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="h-6 w-20 bg-slate-800 rounded animate-pulse"/>
              {[1,2,3,4,5].map((i) => (
                <div key={i} className="bg-slate-800/60 rounded-lg p-3 space-y-2 animate-pulse">
                  <div className="h-3.5 w-16 bg-slate-700 rounded"/>
                  <div className="h-3 bg-slate-700 rounded w-full"/>
                  <div className="h-3 bg-slate-700 rounded w-4/5"/>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {/* ── Initial state ── */}
      {!loading && !data && !error && (
        <div className="bg-slate-900/30 border border-dashed border-slate-800 rounded-2xl p-14 text-center">
          <div className="w-12 h-12 mx-auto mb-3 text-indigo-400/60">
            <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"/>
            </svg>
          </div>
          <h3 className="text-base font-semibold text-slate-200">Select a sample query or type one above</h3>
          <p className="text-sm text-slate-500 mt-1 max-w-md mx-auto">
            All four rankers will run in parallel and results appear side-by-side.
            Overlapping passages and ground-truth relevance are highlighted.
          </p>
        </div>
      )}

      {/* ── Results Grid ── */}
      {!loading && data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {RANKERS.map((ranker) => {
            const hits = data[ranker.id] || [];
            const relevantPids = new Set(selectedSample?.relevant_pids || []);

            return (
              <div
                key={ranker.id}
                id={`compare-col-${ranker.id}`}
                className={`bg-slate-900/60 border ${ranker.border} rounded-xl overflow-hidden flex flex-col`}
              >
                {/* Column Header */}
                <div className={`px-4 py-3 ${ranker.bg} border-b ${ranker.border} flex items-center justify-between`}>
                  <span className={`font-bold text-sm ${ranker.color}`}>{ranker.label}</span>
                  {hits.length === 0 && (
                    <span className="text-xs text-slate-500 italic">unavailable</span>
                  )}
                </div>

                {/* Hit Cards */}
                <div className="flex-1 divide-y divide-slate-800/60 overflow-y-auto max-h-[72vh]">
                  {hits.length === 0 ? (
                    <div className="p-6 text-center text-sm text-slate-500">
                      Ranker not loaded
                    </div>
                  ) : (
                    hits.map((hit, i) => {
                      const inMultiple  = (overlapCounts[hit.pid] || 0) >= 2;
                      const isRelevant  = relevantPids.has(hit.pid);
                      const truncated   = hit.text?.length > 180;
                      const display     = truncated ? hit.text.slice(0, 180) + "…" : hit.text;

                      return (
                        <div
                          key={hit.pid}
                          id={`compare-${ranker.id}-${i}`}
                          className={`p-3 text-xs space-y-1.5 transition-colors ${
                            isRelevant  ? "bg-green-950/25 hover:bg-green-950/40" :
                            inMultiple  ? "bg-amber-950/20 hover:bg-amber-950/35" :
                                          "hover:bg-slate-800/40"
                          }`}
                        >
                          {/* Row metadata */}
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className={`font-bold font-mono ${ranker.color}`}>#{hit.rank}</span>
                            <span className="font-mono text-slate-500 bg-slate-950/70 px-1.5 py-0.5 rounded border border-slate-800/60">
                              PID {hit.pid}
                            </span>
                            <span className="font-mono text-slate-500 ml-auto">{hit.score.toFixed(3)}</span>

                            {/* Overlap badge */}
                            {inMultiple && !isRelevant && (
                              <span className="px-1.5 py-0.5 rounded bg-amber-900/40 border border-amber-700/40 text-amber-300 font-semibold" title="Appears in multiple ranker lists">
                                ×{overlapCounts[hit.pid]}
                              </span>
                            )}
                            {/* Ground truth badge */}
                            {isRelevant && (
                              <span className="px-1.5 py-0.5 rounded bg-green-900/50 border border-green-700/50 text-green-300 font-semibold" title="Known relevant passage">
                                ✓ GT
                              </span>
                            )}
                          </div>
                          <p className="text-slate-400 leading-relaxed">{display}</p>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
