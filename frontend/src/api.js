const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// ─── Health ───────────────────────────────────────────────────────────────────

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// ─── Search ───────────────────────────────────────────────────────────────────

/**
 * @param {string} q
 * @param {"tfidf"|"bm25"|"dense"|"hybrid"} mode
 * @param {number} k
 */
export async function fetchSearch(q, mode = "bm25", k = 10) {
  const url = new URL(`${BASE_URL}/search`);
  url.searchParams.set("q", q);
  url.searchParams.set("mode", mode);
  url.searchParams.set("k", k);
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Search failed: ${res.status}`);
  }
  return res.json();
}

// ─── Compare ──────────────────────────────────────────────────────────────────

export async function fetchCompare(q, k = 10) {
  const url = new URL(`${BASE_URL}/compare`);
  url.searchParams.set("q", q);
  url.searchParams.set("k", k);
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Compare failed: ${res.status}`);
  }
  return res.json();
}

// ─── Sample Queries ───────────────────────────────────────────────────────────

export async function fetchSampleQueries() {
  const res = await fetch(`${BASE_URL}/queries/sample`);
  if (!res.ok) throw new Error(`Failed to load sample queries: ${res.status}`);
  return res.json(); // { queries: [...] }
}

// ─── RAG (SSE streaming) ─────────────────────────────────────────────────────

/**
 * Streams the RAG endpoint. Calls callbacks as SSE events arrive.
 * @param {string} question
 * @param {number} k
 * @param {{ onPassages: (hits: object[]) => void, onToken: (t: string) => void, onDone: () => void, onError: (msg: string) => void }} callbacks
 * @returns {() => void} abort function
 */
export function streamRag(question, k = 5, { onPassages, onToken, onDone, onError }) {
  const controller = new AbortController();

  (async () => {
    let res;
    try {
      res = await fetch(`${BASE_URL}/rag`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k }),
        signal: controller.signal,
      });
    } catch (e) {
      if (e.name !== "AbortError") onError(e.message);
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      onError(err.detail ?? `RAG failed: ${res.status}`);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse SSE lines from buffer
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete last part

      for (const block of parts) {
        let event = null, data = null;
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:"))  data  = line.slice(5).trim();
        }
        if (event === "passages") {
          try { onPassages(JSON.parse(data)); } catch { /* ignore */ }
        } else if (event === "token") {
          onToken(data.replace(/\\n/g, "\n"));
        } else if (event === "done") {
          onDone();
        } else if (event === "error") {
          onError(data);
        }
      }
    }
  })();

  return () => controller.abort();
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

export async function fetchMetrics() {
  const res = await fetch(`${BASE_URL}/metrics`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Metrics failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchPrCurves() {
  const res = await fetch(`${BASE_URL}/pr-curves`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `PR Curves failed: ${res.status}`);
  }
  return res.json();
}
