#!/usr/bin/env python3
"""Capture HF chat-template renderings as golden vectors for Kipp's native
ChatML renderer. One file per checkpoint variant; the C renderer must
reproduce each rendered string byte-for-byte."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from transformers import AutoTokenizer

import checkpoints

# Message scripts exercised for every checkpoint. Each is (messages,
# add_generation_prompt, enable_thinking). enable_thinking is passed only
# when the template accepts it; hybrid instruct models honor it.
CASES = [
    ("user-only", [{"role": "user", "content": "Hi there"}], True, None),
    ("system-user",
     [{"role": "system", "content": "You are terse."},
      {"role": "user", "content": "Say hi"}], True, None),
    ("multiturn",
     [{"role": "user", "content": "1+1?"},
      {"role": "assistant", "content": "2"},
      {"role": "user", "content": "times 3?"}], True, None),
    ("no-genprompt",
     [{"role": "user", "content": "Hello"}], False, None),
    ("think-off",
     [{"role": "user", "content": "Hello"}], True, False),
    ("think-on",
     [{"role": "user", "content": "Hello"}], True, True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="qwen3-4b-instruct-2507")
    parser.add_argument("--source", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    spec = checkpoints.get(args.checkpoint)
    source = args.source or pathlib.Path("models") / spec.id / "source"
    output = (
        args.output
        or pathlib.Path("tests/test-vectors") / spec.id / "chat-cases.json"
    )
    source = source.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    cases = []
    for name, messages, add_gen, enable_thinking in CASES:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_gen,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as error:  # noqa: BLE001
            print(f"skip {name}: {error}", file=sys.stderr)
            continue
        cases.append({
            "name": name,
            "messages": messages,
            "add_generation_prompt": add_gen,
            "enable_thinking": enable_thinking,
            "rendered": rendered,
        })
    output.write_text(
        json.dumps({"checkpoint": spec.id, "variant": spec.variant,
                    "cases": cases}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output} ({len(cases)} cases)")
    for case in cases:
        print(f"--- {case['name']} ---")
        print(repr(case["rendered"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"generate_chat_vectors.py: {error}", file=sys.stderr)
        raise SystemExit(1) from error
