"""FastAPI application — loads all indexes at startup and exposes:
  GET  /health
  GET  /search?q=...&ranker=...&top_k=...
  GET  /compare?q=...&top_k=...
  POST /rag  (streaming, Server-Sent Events)
  GET  /metrics
"""
