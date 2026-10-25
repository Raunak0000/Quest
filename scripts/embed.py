"""Encode raw passage texts with sentence-transformers and save embeddings + pids.

Model: all-MiniLM-L6-v2 (dim=384)
Settings: batch_size=256, max_seq_length=256, normalize_embeddings=True

Outputs:
  indexes/embeddings.npy  (float32 array, shape: num_passages x 384)
  indexes/pids.json       (JSON list of integer passage IDs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if sys.platform == "win32":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.data_loader import load_lotte


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode LoTTE corpus passages with SentenceTransformer."
    )
    parser.add_argument("--domain", type=str, default="lifestyle", help="LoTTE domain (default: lifestyle)")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split (default: dev)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding (default: 256)")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max token sequence length (default: 256)")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--output-dir", type=str, default=os.path.join(_REPO_ROOT, "indexes"), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of passages (for testing)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', etc.)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("LoTTE Dense Passage Embedding")
    print(f"Domain: {args.domain} | Split: {args.split} | Device: {device}")
    print(f"Model: {args.model_name} (batch: {args.batch_size}, max_seq_length: {args.max_seq_length})")
    print("=" * 80)

    # 1. Load passages
    t0 = time.time()
    passages, _, _ = load_lotte(args.domain, args.split)
    pids = list(passages.keys())
    if args.limit:
        pids = pids[:args.limit]
    texts = [passages[pid] for pid in pids]
    print(f"[1/3] Loaded {len(texts):,} passages in {time.time()-t0:.2f}s")

    # 2. Load model
    print(f"[2/3] Loading SentenceTransformer '{args.model_name}' on {device}...")
    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = args.max_seq_length

    # 3. Encode passages
    print(f"[3/3] Encoding {len(texts):,} passages (batch_size={args.batch_size})...")
    t1 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    encode_time = time.time() - t1
    print(f"      Encoding complete in {encode_time:.2f}s ({len(texts)/encode_time:.1f} passages/s)")
    print(f"      Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # 4. Save embeddings and pids
    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    pids_path = os.path.join(args.output_dir, "pids.json")

    print(f"Saving embeddings -> {emb_path}")
    np.save(emb_path, embeddings)

    print(f"Saving passage IDs -> {pids_path}")
    with open(pids_path, "w", encoding="utf-8") as f:
        json.dump(pids, f)

    file_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
    print(f"Done! {emb_path} ({file_size_mb:.1f} MB) and {pids_path} generated successfully.")


if __name__ == "__main__":
    main()
