#!/usr/bin/env python3
"""Prepare WikiText-2 (raw) token data for the perplexity harness.

Downloads the raw test split from the Hugging Face dataset mirror, joins the
lines into one document exactly as llama.cpp's perplexity tool does, encodes
it with the checkpoint's own tokenizer.json, and writes the token ids as
little-endian uint32. The output feeds the CLI's --ppl mode.

Usage:
    python3 tools/prepare_wikitext.py \
        --tokenizer models/qwen3-4b-base/source/tokenizer.json \
        --output bench/data/wikitext2-test.tokens
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="path to the checkpoint's tokenizer.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset-repo", default="Salesforce/wikitext")
    ap.add_argument("--dataset-file",
                    default="wikitext-2-raw-v1/test-00000-of-00001.parquet")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    parquet_path = hf_hub_download(repo_id=args.dataset_repo,
                                   filename=args.dataset_file,
                                   repo_type="dataset")
    table = pq.read_table(parquet_path)
    text = "".join(table.column("text").to_pylist())

    tokenizer = Tokenizer.from_file(args.tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False).ids

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        for token in ids:
            f.write(struct.pack("<I", token))
    print(f"{len(ids)} tokens -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
