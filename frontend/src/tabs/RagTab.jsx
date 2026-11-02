import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { Marked } from "marked";
import { streamRag } from "../api";

// Configure Marked instance with inline citation chip extension
const customMarked = new Marked({
  gfm: true,
  breaks: true,
  extensions: [
    {
      name: "citation",
      level: "inline",
      start(src) {
        return src.indexOf("[");
      },
      tokenizer(src) {
        const match = /^\[\s*(\d+)\s*\]/.exec(src);
        if (match) {
          return {
            type: "citation",
            raw: match[0],
            num: Number(match[1]),
          };
        }
      },
      renderer(token) {
        return `<button type="button" data-cite="${token.num}" class="cite-chip" title="Jump to passage ${token.num}">[${token.num}]</button>`;
      },
    },
  ],
});

export default function RagTab() {
  const [question, setQuestion]   = useState("");
  const [passages, setPassages]   = useState([]);
  const [answer, setAnswer]       = useState("");
  const [status, setStatus]       = useState("idle"); // idle | streaming | done | error
  const [errorMsg, setErrorMsg]   = useState(null);
  const [highlighted, setHighlighted] = useState(null);
  const abortRef                  = useRef(null);
  const passageRefs               = useRef({});
  const answerEndRef              = useRef(null);

  // Auto-scroll answer area as tokens arrive
  useEffect(() => {
    if (status === "streaming") {
      answerEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [answer, status]);

  const handleCiteClick = (n) => {
    const idx = n - 1;
    setHighlighted(idx);
    passageRefs.current[idx]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    // Clear highlight after 2.5 s
    setTimeout(() => setHighlighted((h) => (h === idx ? null : h)), 2500);
  };

  const handleSubmit = useCallback(() => {
    const q = question.trim();
    if (!q) return;

    // Abort any existing stream
    abortRef.current?.();
    abortRef.current = null;

    setPassages([]);
    setAnswer("");
    setStatus("streaming");
    setErrorMsg(null);
    setHighlighted(null);

    abortRef.current = streamRag(q, 5, {
      onPassages: (hits) => setPassages(hits),
      onToken:    (t)    => setAnswer((prev) => prev + t),
      onDone:     ()     => setStatus("done"),
      onError:    (msg)  => { setStatus("error"); setErrorMsg(msg); },
    });
  }, [question]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleStop = () => {
    abortRef.current?.();
    abortRef.current = null;
    setStatus("done");
  };

  const handleReset = () => {
    abortRef.current?.();
    abortRef.current = null;
    setPassages([]);
    setAnswer("");
    setStatus("idle");
    setErrorMsg(null);
    setHighlighted(null);
  };

  // Convert streamed LLM text to formatted HTML with clean markdown & interactive citations
  const renderedAnswerHtml = useMemo(() => {
    if (!answer) return "";
    // Normalize common LLM markdown formatting artifacts:
    // 1. Ensure space after heading hashes: '###Heading' -> '### Heading'
    let cleaned = answer.replace(/^(\s*)#{1,6}(?=[^\s#])/gm, "$& ");
    // 2. Fix triple asterisks at start of lines: '***Title:**' -> '* **Title:**'
    cleaned = cleaned.replace(/^(\s*)\*\*\*([^*]+)\*\*/gm, "$1* **$2**");
    // 3. Ensure bullet has space after asterisk: '*Item' -> '* Item'
    cleaned = cleaned.replace(/^(\s*)\*(?!\s|\*)(.+)$/gm, "$1* $2");

    return customMarked.parse(cleaned);
  }, [answer]);

  // Event delegation to catch clicks on citation buttons inside the rendered HTML
  const handleAnswerClick = (e) => {
    const btn = e.target.closest("[data-cite]");
    if (!btn) return;
    const n = Number(btn.getAttribute("data-cite"));
    if (n >= 1 && n <= passages.length) {
      handleCiteClick(n);
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-5">
      {/* ── Question Input ── */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-start gap-3">
          <div className="flex-1 relative">
            <div className="absolute left-3 top-3 text-slate-400 pointer-events-none">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"/>
              </svg>
            </div>
            <textarea
              id="rag-input"
              rows={2}
              className="w-full pl-9 pr-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-xl text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
              placeholder="Ask a question about the LoTTE lifestyle corpus… (Enter to submit)"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={status === "streaming"}
            />
          </div>

          <div className="flex flex-col gap-2 flex-shrink-0">
            {status !== "streaming" ? (
              <button
                id="rag-submit"
                onClick={handleSubmit}
                disabled={!question.trim() || status === "streaming"}
                className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-xl transition-all flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                Ask LLM
              </button>
            ) : (
              <button
                id="rag-stop"
                onClick={handleStop}
                className="px-5 py-2.5 bg-red-700 hover:bg-red-600 text-white text-sm font-semibold rounded-xl transition-all flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <rect x="4" y="4" width="12" height="12" rx="2"/>
                </svg>
                Stop
              </button>
            )}
            {(status === "done" || status === "error") && (
              <button
                onClick={handleReset}
                className="px-5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium rounded-xl transition-all"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-block w-2 h-2 rounded-full bg-indigo-400"/>
          <span>Retrieves top-5 passages via Hybrid ranker, then streams an LLM answer with citations.</span>
          <span className="ml-auto">Press <kbd className="px-1 py-0.5 bg-slate-800 border border-slate-700 rounded font-mono text-slate-400">Enter</kbd> to submit</span>
        </div>
      </div>

      {/* ── Error ── */}
      {status === "error" && errorMsg && (
        <div className="p-4 bg-red-950/50 border border-red-800/70 rounded-xl text-red-200 text-sm flex gap-3">
          <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <span><strong className="text-red-100">Backend error: </strong>{errorMsg}</span>
        </div>
      )}

      {/* ── Streaming Answer ── */}
      {(answer || status === "streaming") && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/80">
            <div className="flex items-center gap-2.5 text-sm font-semibold text-white">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-[10px] font-black text-white shadow-lg tracking-wider">
                LLM
              </div>
              <span>LLM Answer</span>
            </div>
            {status === "streaming" && (
              <div className="flex items-center gap-2 text-xs text-indigo-400 animate-pulse">
                <svg className="animate-spin w-3.5 h-3.5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                <span>Streaming…</span>
              </div>
            )}
            {status === "done" && (
              <span className="text-xs text-emerald-400 flex items-center gap-1">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7"/>
                </svg>
                Complete
              </span>
            )}
          </div>

          <div
            className="markdown-content text-sm leading-relaxed font-normal min-h-[60px]"
            onClick={handleAnswerClick}
          >
            <div dangerouslySetInnerHTML={{ __html: renderedAnswerHtml }} />
            {status === "streaming" && (
              <span className="inline-block w-1.5 h-4 bg-indigo-400 animate-pulse mt-1 align-middle rounded-sm" />
            )}
          </div>
          <div ref={answerEndRef} />
        </div>
      )}

      {/* ── Source Passages ── */}
      {passages.length > 0 && (
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 px-1">
            <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
            </svg>
            Source Passages ({passages.length})
          </div>

          <div className="space-y-3">
            {passages.map((hit, i) => {
              const isHighlighted = highlighted === i;
              return (
                <div
                  key={hit.pid}
                  ref={(el) => (passageRefs.current[i] = el)}
                  id={`rag-passage-${i}`}
                  className={`rounded-xl border p-4 transition-all duration-500 ${
                    isHighlighted
                      ? "bg-indigo-950/50 border-indigo-500/70 shadow-lg shadow-indigo-500/10 scale-[1.005]"
                      : "bg-slate-900/70 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-md text-xs font-bold font-mono border transition-all ${
                      isHighlighted
                        ? "bg-indigo-600 border-indigo-500 text-white"
                        : "bg-indigo-600/20 border-indigo-500/30 text-indigo-400"
                    }`}>
                      {i + 1}
                    </span>
                    <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                      PID: {hit.pid}
                    </span>
                    <span className="ml-auto text-xs font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40">
                      score: {typeof hit.score === "number" ? hit.score.toFixed(4) : hit.score}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 leading-relaxed">{hit.text || "(text not available)"}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Idle state ── */}
      {status === "idle" && passages.length === 0 && (
        <div className="bg-slate-900/30 border border-dashed border-slate-800 rounded-2xl p-14 text-center">
          <div className="w-12 h-12 mx-auto mb-3 text-violet-400/60">
            <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"/>
            </svg>
          </div>
          <h3 className="text-base font-semibold text-slate-200">RAG with LLM</h3>
          <p className="text-sm text-slate-400 mt-1 max-w-md mx-auto">
            Ask any question. The hybrid ranker fetches 5 relevant passages, then the LLM streams a cited answer.
            Click <span className="font-mono text-indigo-400 text-xs">[1]</span>–<span className="font-mono text-indigo-400 text-xs">[5]</span> chips in the answer to jump to the cited passage.
          </p>
        </div>
      )}
    </div>
  );
}
