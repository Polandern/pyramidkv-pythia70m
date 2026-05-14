from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    rows = []
    for path in sorted(Path("results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "file": path.name,
                "method": data.get("method"),
                "ppl": data.get("ppl"),
                "ttft": data.get("ttft_sec"),
                "tpot": data.get("tpot_sec"),
                "throughput": data.get("throughput_tok_per_sec"),
                "memory_mb": data.get("peak_rss_mb"),
                "avg_cache": data.get("avg_cache_length"),
            }
        )

    if not rows:
        print("No result files found in results/.")
        return

    headers = ["file", "method", "ppl", "ttft", "tpot", "throughput", "memory_mb", "avg_cache"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append("" if value is None else str(value))
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()

