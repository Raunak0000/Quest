"""RAG layer: build prompt from top-5 hybrid passages and stream LLM answer.

Public API
----------
build_prompt(question, passages)
    Build a system+user prompt string from the retrieved passages.

generate(prompt)
    Async generator that yields string tokens from the configured LLM provider.
    Reads LLM_PROVIDER and LLM_API_KEY from the environment (loaded via .env).

Supported providers
-------------------
openai    — uses openai>=1.0 AsyncOpenAI client
anthropic — uses anthropic AsyncAnthropic client
gemini    — uses google-generativeai GenerativeModel
groq      — uses groq AsyncGroq client (OpenAI-compatible)
ollama    — uses httpx streaming against local ollama REST API (no key needed)
"""

from __future__ import annotations

import os
from typing import AsyncIterator

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise, citation-aware research assistant. "
    "Answer the user's question using ONLY the passages provided below. "
    "For every factual claim, cite the relevant passage by its number, e.g. [1]. "
    "If no passage is relevant, say you don't have enough information. "
    "Be concise and do not fabricate information."
)


def build_prompt(question: str, passages: list[tuple[int, str]]) -> str:
    """Build a self-contained prompt that includes numbered passages.

    Parameters
    ----------
    question : str
        The user's raw question.
    passages : list of (pid, text)
        Ordered list of (passage_id, passage_text) to include as context.
        Typically the top-5 results from the hybrid ranker.

    Returns
    -------
    str
        A single string containing the system instructions, numbered passages,
        and the question, ready to be sent to any chat-completion API.
    """
    passage_block = "\n\n".join(
        f"[{i + 1}] (PID {pid}): {text.strip()}"
        for i, (pid, text) in enumerate(passages)
    )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"--- Passages ---\n{passage_block}\n\n"
        f"--- Question ---\n{question}"
    )


# ---------------------------------------------------------------------------
# Provider-agnostic streaming generator
# ---------------------------------------------------------------------------

async def generate(prompt: str) -> AsyncIterator[str]:
    """Async generator that yields string tokens from the configured LLM.

    Configuration is read from environment variables (populated from .env):
      LLM_PROVIDER  — one of: openai, anthropic, gemini, groq, ollama
      LLM_API_KEY   — API key (leave blank for ollama)

    Raises
    ------
    EnvironmentError
        If LLM_PROVIDER is not set or is not a supported value.
    ValueError
        If LLM_API_KEY is required but missing for the chosen provider.
    """
    # Load .env if not already loaded (idempotent)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass  # python-dotenv optional; rely on env vars being set directly

    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    api_key  = os.environ.get("LLM_API_KEY",  "").strip()

    if not provider:
        raise EnvironmentError(
            "LLM_PROVIDER is not set. "
            "Add LLM_PROVIDER=openai|anthropic|gemini|groq|ollama to your .env file."
        )

    if provider == "openai":
        async for token in _openai(prompt, api_key):
            yield token

    elif provider == "anthropic":
        async for token in _anthropic(prompt, api_key):
            yield token

    elif provider == "gemini":
        async for token in _gemini(prompt, api_key):
            yield token

    elif provider == "groq":
        async for token in _groq(prompt, api_key):
            yield token

    elif provider == "ollama":
        async for token in _ollama(prompt):
            yield token

    else:
        raise EnvironmentError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            "Must be one of: openai, anthropic, gemini, groq, ollama."
        )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _require_key(provider: str, api_key: str) -> str:
    if not api_key:
        raise ValueError(
            f"LLM_API_KEY must be set for provider '{provider}'. "
            "Add it to your .env file."
        )
    return api_key


async def _openai(prompt: str, api_key: str) -> AsyncIterator[str]:
    """Stream tokens from OpenAI chat completions (gpt-4o-mini)."""
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is not installed. Install it with: pip install openai"
        ) from exc

    client = AsyncOpenAI(api_key=_require_key("openai", api_key))
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.2,
        max_tokens=512,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def _anthropic(prompt: str, api_key: str) -> AsyncIterator[str]:
    """Stream tokens from Anthropic Messages API (claude-3-haiku)."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is not installed. Install it with: pip install anthropic"
        ) from exc

    client = anthropic.AsyncAnthropic(api_key=_require_key("anthropic", api_key))
    async with client.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _gemini(prompt: str, api_key: str) -> AsyncIterator[str]:
    """Stream tokens from Google Gemini (defaults to gemini-3.5-flash, supports GEMINI_MODEL env)."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "The 'google-generativeai' package is not installed. Install it with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=_require_key("gemini", api_key))
    preferred_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"
    candidate_models = [preferred_model]
    for m in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
        if m not in candidate_models:
            candidate_models.append(m)

    last_exc = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = await model.generate_content_async(prompt, stream=True)
            async for chunk in response:
                try:
                    text = chunk.text
                except Exception:
                    text = ""
                if text:
                    yield text
            return
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc:
        raise last_exc


async def _groq(prompt: str, api_key: str) -> AsyncIterator[str]:
    """Stream tokens from Groq (llama-3.1-8b-instant via OpenAI-compatible API)."""
    try:
        from groq import AsyncGroq  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'groq' package is not installed. Install it with: pip install groq"
        ) from exc

    client = AsyncGroq(api_key=_require_key("groq", api_key))
    stream = await client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.2,
        max_tokens=512,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def _ollama(prompt: str) -> AsyncIterator[str]:
    """Stream tokens from a local Ollama instance (llama3 model)."""
    import httpx
    import json

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    url = f"{ollama_url}/api/generate"
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "llama3"),
        "prompt": prompt,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
