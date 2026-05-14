from __future__ import annotations

import argparse
from pathlib import Path

from src.pyramidkv.data import load_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a small dataset sample to a text file.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_documents", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = load_text(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_documents=args.max_documents,
        streaming=args.streaming,
    )
    if not text.strip():
        raise ValueError("The selected dataset sample is empty.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} characters to {out}")


if __name__ == "__main__":
    main()
