import { useState, useEffect, useCallback, useRef } from "react";
import { fetchSearch, fetchSampleQueries } from "../api";

const MODES = [
  { id: "tfidf",  label: "TF-IDF",  desc: "Term Frequency - Inverse Document Frequency" },
  { id: "bm25",   label: "BM25",    desc: "Probabilistic relevance model (bm25s)" },
  { id: "dense",  label: "Dense",   desc: "Semantic vector retrieval (sentence-transformers)" },
  { id: "hybrid", label: "Hybrid",  desc: "Reciprocal Rank Fusion (RRF)" },
];

export default function SearchTab() {
  const [query, setQuery]         = useState("");
  const [mode, setMode]           = useState("bm25");
  const [results, setResults]     = useState(null);
  const [latency, setLatency]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [expanded, setExpanded]   = useState({});
  const [samples, setSamples]     = useState([]);
  const inputRef                  = useRef(null);

  // Load sample queries on mount
  useEffect(() => {
    fetchSampleQueries()
      .then((d) => setSamples(d.queries ?? []))
      .catch(() => {});
  }, []);

  const runSearch = useCallback(async (q, m) => {
    const targetQuery = (q ?? query).trim();
    const targetMode = m ?? mode;
    if (!targetQuery) return;

    setLoading(true);
    setError(null);
    setExpanded({});

    try {
      const data = await fetchSearch(targetQuery, targetMode, 10);
      setResults(data.results || []);
      setLatency(data.latency_ms);
    } catch (e) {
      setError(e.message || "Failed to fetch results from the backend.");
      setResults(null);
      setLatency(null);
    } finally {
      setLoading(false);
    }
  }, [query, mode]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      runSearch();
    }
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    if (query.trim()) {
      runSearch(query, newMode);
    }
  };

  const handleSampleClick = (sampleText) => {
    setQuery(sampleText);
    runSearch(sampleText, mode);
    inputRef.current?.focus();
  };

  const toggleExpand = (idx) => {
    setExpanded((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      {/* ── Search Input & Controls ── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-sm">
        <div className="relative flex items-center">
          <div className="absolute left-4 text-slate-400 pointer-events-none">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <circle cx="11" cy="11" r="8" />
              <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-4.35-4.35" />
            </svg>
          </div>
          <input
            ref={inputRef}
            id="search-input"
            type="text"
            className="w-full pl-12 pr-28 py-3.5 bg-slate-950/80 border border-slate-700/80 rounded-xl text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/80 focus:border-indigo-500 transition-all text-base"
            placeholder="Search passages in the LoTTE corpus... (e.g. coffee brewing, fitness routines)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            autoComplete="off"
            spellCheck={false}
          />
          {query && (
            <button
              onClick={() => { setQuery(""); setResults(null); setError(null); }}
              className="absolute right-24 text-slate-400 hover:text-slate-200 p-1 text-sm transition-colors"
              title="Clear search"
            >
              ✕
            </button>
          )}
          <button
            id="search-submit"
            onClick={() => runSearch()}
            disabled={loading || !query.trim()}
            className="absolute right-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium text-sm rounded-lg shadow-sm transition-all flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                </svg>
                <span>Searching</span>
              </>
            ) : (
              <span>Search</span>
            )}
          </button>
        </div>

        {/* ── Mode Selector ── */}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/80">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold tracking-wider uppercase text-slate-400">Ranker:</span>
            <div className="inline-flex p-1 bg-slate-950/90 rounded-xl border border-slate-800" role="group">
              {MODES.map((m) => {
                const isActive = mode === m.id;
                return (
                  <button
                    key={m.id}
                    id={`mode-${m.id}`}
                    onClick={() => handleModeChange(m.id)}
                    title={m.desc}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 ${
                      isActive
                        ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                    }`}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Latency & count badge */}
          {latency !== null && !loading && (
            <div className="flex items-center gap-2 px-3 py-1 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-300">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
              <span><strong className="text-emerald-400 font-mono">{latency.toFixed(1)} ms</strong> latency</span>
              <span className="text-slate-600">•</span>
              <span className="text-slate-400">{results?.length ?? 0} hits</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Sample Queries ── */}
      {samples.length > 0 && !results && !loading && (
        <div className="bg-slate-900/50 border border-slate-800/70 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2 text-xs font-medium text-slate-400 uppercase tracking-wider">
            <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>Sample Benchmark Queries:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {samples.slice(0, 5).map((s) => (
              <button
                key={s.qid}
                onClick={() => handleSampleClick(s.text)}
                className="text-left text-xs bg-slate-800/80 hover:bg-slate-700 border border-slate-700/60 text-slate-300 hover:text-white px-3 py-1.5 rounded-lg transition-colors duration-150"
              >
                "{s.text.length > 60 ? s.text.slice(0, 57) + "…" : s.text}"
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Error Banner ── */}
      {error && (
        <div className="p-4 bg-red-950/50 border border-red-800/70 rounded-xl text-red-200 text-sm flex items-start gap-3">
          <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <span className="font-semibold text-red-100">Search Error: </span>
            {error}
          </div>
        </div>
      )}

      {/* ── Loading Skeleton ── */}
      {loading && (
        <div className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 animate-pulse space-y-3">
              <div className="flex items-center gap-3">
                <div className="h-5 w-10 bg-slate-800 rounded"></div>
                <div className="h-5 w-20 bg-slate-800 rounded"></div>
                <div className="h-5 w-16 bg-slate-800 rounded ml-auto"></div>
              </div>
              <div className="h-4 bg-slate-800 rounded w-full"></div>
              <div className="h-4 bg-slate-800 rounded w-5/6"></div>
              <div className="h-4 bg-slate-800 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      )}

      {/* ── Empty State ── */}
      {!loading && results && results.length === 0 && (
        <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-12 text-center">
          <div className="w-12 h-12 mx-auto mb-3 text-slate-600">
            <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
            </svg>
          </div>
          <h3 className="text-base font-medium text-slate-300">No matching passages found</h3>
          <p className="text-sm text-slate-500 mt-1">Try rephrasing your search query or switching to another ranker mode.</p>
        </div>
      )}

      {/* ── Initial State ── */}
      {!loading && !results && !error && (
        <div className="bg-slate-900/30 border border-dashed border-slate-800 rounded-2xl p-12 text-center">
          <div className="w-12 h-12 mx-auto mb-3 text-indigo-400/70">
            <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 0 0 1.5-.189m-1.5.189a6.01 6.01 0 0 1-1.5-.189m3.75 7.478a12.06 12.06 0 0 1-4.5 0m3.75 2.383a14.406 14.406 0 0 1-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 1 0-7.516 0c.85.493 1.508 1.333 1.508 2.316V18" />
            </svg>
          </div>
          <h3 className="text-base font-semibold text-slate-200">Ready to Search</h3>
          <p className="text-sm text-slate-400 mt-1 max-w-md mx-auto">
            Type a query above and hit <kbd className="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-xs font-mono text-slate-300">Enter</kbd> to rank passages across the 269,597 documents in LoTTE.
          </p>
        </div>
      )}

      {/* ── Results List ── */}
      {!loading && results && results.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span>Showing top {results.length} passages ranked by <strong>{MODES.find((m) => m.id === mode)?.label}</strong>:</span>
            <span>Click any passage to expand</span>
          </div>

          <ol className="space-y-3">
            {results.map((hit, i) => {
              const isLong = (hit.text || "").length > 280;
              const isExpanded = expanded[i];
              const textContent = hit.text || "(Passage text not loaded)";
              const displayText = isLong && !isExpanded ? textContent.slice(0, 280) + "…" : textContent;

              return (
                <li
                  key={hit.pid || i}
                  id={`result-${i}`}
                  className="bg-slate-900/80 border border-slate-800 hover:border-slate-700/90 rounded-xl p-4 transition-all duration-150 shadow-sm hover:shadow-md"
                >
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-md bg-indigo-600/20 text-indigo-400 text-xs font-bold font-mono border border-indigo-500/30">
                        #{hit.rank}
                      </span>
                      <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                        PID: {hit.pid}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">Score:</span>
                      <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40">
                        {typeof hit.score === "number" ? hit.score.toFixed(4) : hit.score}
                      </span>
                    </div>
                  </div>

                  <p className="text-sm text-slate-300 leading-relaxed font-normal whitespace-pre-line">
                    {displayText}
                  </p>

                  {isLong && (
                    <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex justify-end">
                      <button
                        id={`show-more-${i}`}
                        onClick={() => toggleExpand(i)}
                        className="text-xs font-medium text-indigo-400 hover:text-indigo-300 hover:underline flex items-center gap-1 transition-colors"
                      >
                        {isExpanded ? (
                          <>
                            <span>Show less</span>
                            <span>↑</span>
                          </>
                        ) : (
                          <>
                            <span>Show full passage</span>
                            <span>↓</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}
