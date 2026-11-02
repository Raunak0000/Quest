import { useState, useEffect } from "react";
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from "recharts";
import { fetchMetrics, fetchPrCurves } from "../api";

// Color palette per ranker
const RANKER_COLORS = {
  "TF-IDF": "#38bdf8",  // sky-400
  "BM25":   "#a78bfa",  // violet-400
  "Dense":  "#34d399",  // emerald-400
  "Hybrid": "#fbbf24",  // amber-400
};

const METRIC_LABELS = {
  "P@10":      "P@10",
  "R@100":     "R@100",
  "MAP":       "MAP",
  "nDCG@10":   "nDCG@10",
  "Success@5": "Success@5",
};

// Find the best ranker per metric column
function findBest(metrics) {
  const best = {};
  const rankers = Object.keys(metrics);
  for (const metric of Object.keys(METRIC_LABELS)) {
    let bestVal = -Infinity;
    let bestRanker = null;
    for (const ranker of rankers) {
      const v = metrics[ranker]?.[metric];
      if (v !== undefined && v > bestVal) {
        bestVal = v;
        bestRanker = ranker;
      }
    }
    best[metric] = bestRanker;
  }
  return best;
}

// Build data for grouped bar chart: one entry per metric
function buildBarData(metrics) {
  return Object.keys(METRIC_LABELS).map((metric) => {
    const entry = { metric: METRIC_LABELS[metric] };
    for (const [ranker, vals] of Object.entries(metrics)) {
      entry[ranker] = vals?.[metric] ?? null;
    }
    return entry;
  });
}

// Build data for PR curve line chart
function buildPrData(prCurves) {
  if (!prCurves) return [];
  // Get recall levels from first ranker
  const firstRanker = Object.values(prCurves)[0];
  if (!firstRanker?.recall) return [];
  return firstRanker.recall.map((r, i) => {
    const entry = { recall: r.toFixed(1) };
    for (const [ranker, data] of Object.entries(prCurves)) {
      entry[ranker] = data.precision?.[i] ?? null;
    }
    return entry;
  });
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 shadow-2xl text-xs space-y-1.5">
      <p className="font-bold text-slate-200 mb-2">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: p.color }}/>
          <span className="text-slate-400 font-medium">{p.name}:</span>
          <span className="font-mono font-semibold" style={{ color: p.color }}>
            {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function MetricsTab() {
  const [metrics, setMetrics]   = useState(null);
  const [prCurves, setPrCurves] = useState(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.allSettled([fetchMetrics(), fetchPrCurves()]).then(([mRes, prRes]) => {
      if (cancelled) return;
      if (mRes.status === "fulfilled") {
        setMetrics(mRes.value);
      } else {
        setError(mRes.reason?.message || "Failed to load metrics.");
      }
      if (prRes.status === "fulfilled") {
        setPrCurves(prRes.value);
      }
      setLoading(false);
    });

    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="w-full max-w-5xl mx-auto space-y-6">
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-8 animate-pulse space-y-4">
          <div className="h-6 bg-slate-800 rounded w-48"/>
          <div className="h-48 bg-slate-800 rounded-xl"/>
        </div>
      </div>
    );
  }

  if (error && !metrics) {
    return (
      <div className="w-full max-w-3xl mx-auto p-6 bg-red-950/40 border border-red-800/60 rounded-2xl text-sm text-red-200 space-y-2">
        <div className="flex items-center gap-2 font-semibold text-red-100">
          <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          Metrics unavailable
        </div>
        <p>{error}</p>
        <p className="text-xs text-red-300/60 mt-1">Run <code className="font-mono bg-red-950/50 px-1.5 py-0.5 rounded">python scripts/run_eval.py</code> to generate evaluation results, then restart the server.</p>
      </div>
    );
  }

  const rankers = Object.keys(metrics || {});
  const bestPerMetric = metrics ? findBest(metrics) : {};
  const barData = metrics ? buildBarData(metrics) : [];
  const prData  = prCurves ? buildPrData(prCurves) : [];
  const metricKeys = Object.keys(METRIC_LABELS);

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8">
      {/* ── Header ── */}
      <div>
        <h1 className="text-xl font-bold text-white">Evaluation Metrics</h1>
        <p className="text-sm text-slate-400 mt-1">
          LoTTE lifestyle · dev split · search queries · top-100 retrieved passages
        </p>
      </div>

      {/* ── How to read the table ── */}
      <div className="bg-slate-900/50 border border-slate-800/70 rounded-xl p-4 text-xs text-slate-400 space-y-1">
        <p className="font-semibold text-slate-300 mb-1.5">📖 How to read this table</p>
        <p><strong className="text-slate-200">P@10</strong> — Precision at 10: fraction of top-10 results that are relevant. Higher = better precision.</p>
        <p><strong className="text-slate-200">R@100</strong> — Recall at 100: fraction of all relevant passages found in top-100. Higher = better coverage.</p>
        <p><strong className="text-slate-200">MAP</strong> — Mean Average Precision: area under the precision-recall curve, averaged over all queries.</p>
        <p><strong className="text-slate-200">nDCG@10</strong> — Normalized Discounted Cumulative Gain at 10: accounts for rank position of relevant results.</p>
        <p><strong className="text-slate-200">Success@5</strong> — Fraction of queries where at least one relevant passage appears in top-5.</p>
        <p className="mt-1.5 text-slate-500">Bold values indicate the <span className="text-amber-400">best</span> result per column.</p>
      </div>

      {/* ── Comparison Table ── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18M10 3v18M14 3v18"/>
          </svg>
          <h2 className="font-semibold text-sm text-white">Performance Comparison Table</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-950/60">
                <th className="text-left px-5 py-3 text-xs font-semibold uppercase tracking-wider text-slate-400 border-b border-slate-800">Ranker</th>
                {metricKeys.map((m) => (
                  <th key={m} className="text-right px-5 py-3 text-xs font-semibold uppercase tracking-wider text-slate-400 border-b border-slate-800">
                    {METRIC_LABELS[m]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {rankers.map((ranker, ri) => (
                <tr key={ranker} className={`transition-colors hover:bg-slate-800/30 ${ri % 2 === 0 ? "" : "bg-slate-900/20"}`}>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2.5">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ background: RANKER_COLORS[ranker] || "#64748b" }}
                      />
                      <span className="font-semibold text-slate-200">{ranker}</span>
                    </div>
                  </td>
                  {metricKeys.map((metric) => {
                    const val = metrics[ranker]?.[metric];
                    const isBest = bestPerMetric[metric] === ranker;
                    return (
                      <td key={metric} className="text-right px-5 py-4">
                        <span className={`font-mono text-sm ${isBest ? "font-bold text-amber-400" : "text-slate-300"}`}>
                          {val !== undefined ? val.toFixed(4) : "—"}
                        </span>
                        {isBest && (
                          <span className="ml-1 text-amber-400 text-xs" title="Best result">★</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Grouped Bar Chart ── */}
      {barData.length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
            <h2 className="font-semibold text-sm text-white">Metric Comparison — Bar Chart</h2>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false}/>
              <XAxis
                dataKey="metric"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                axisLine={{ stroke: "#334155" }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
                width={44}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(99,102,241,0.06)" }}/>
              <Legend
                wrapperStyle={{ fontSize: "12px", paddingTop: "12px", color: "#94a3b8" }}
              />
              {rankers.map((ranker) => (
                <Bar
                  key={ranker}
                  dataKey={ranker}
                  fill={RANKER_COLORS[ranker] || "#64748b"}
                  radius={[3, 3, 0, 0]}
                  maxBarSize={32}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── PR Curve ── */}
      {prData.length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <div className="flex items-center gap-2 mb-4">
            <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"/>
            </svg>
            <h2 className="font-semibold text-sm text-white">Interpolated Precision–Recall Curve</h2>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={prData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
              <XAxis
                dataKey="recall"
                label={{ value: "Recall", position: "insideBottomRight", offset: -5, fill: "#64748b", fontSize: 12 }}
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={{ stroke: "#334155" }}
                tickLine={false}
              />
              <YAxis
                label={{ value: "Precision", angle: -90, position: "insideLeft", offset: 10, fill: "#64748b", fontSize: 12 }}
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => v.toFixed(2)}
                width={44}
              />
              <Tooltip content={<CustomTooltip />}/>
              <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "12px", color: "#94a3b8" }}/>
              {Object.keys(prData[0] || {}).filter((k) => k !== "recall").map((ranker) => (
                <Line
                  key={ranker}
                  type="monotone"
                  dataKey={ranker}
                  stroke={RANKER_COLORS[ranker] || "#64748b"}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 5, stroke: RANKER_COLORS[ranker], strokeWidth: 2, fill: "#0f172a" }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* No PR curves notice */}
      {!prData.length && !loading && (
        <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-xl p-6 text-center text-xs text-slate-500">
          <p>Precision-Recall curve data not found (<code className="font-mono">results/pr_curves.json</code>).</p>
          <p className="mt-1">Run <code className="font-mono bg-slate-800 px-1.5 py-0.5 rounded">python scripts/run_eval.py</code> to generate it.</p>
        </div>
      )}
    </div>
  );
}
