from __future__ import annotations

import subprocess
import sys


def main() -> None:
    commands = [
        [
            sys.executable,
            "eval_ppl.py",
            "--method",
            "pyramidkv",
            "--sample_text",
            "samples/tiny_pg19.txt",
            "--max_tokens",
            "128",
            "--kv_budget",
            "64",
            "--device",
            "cpu",
        ],
        [
            sys.executable,
            "benchmark_speed.py",
            "--method",
            "pyramidkv",
            "--prompt_file",
            "samples/tiny_pg19.txt",
            "--prompt_tokens",
            "128",
            "--generate_tokens",
            "16",
            "--kv_budget",
            "64",
            "--device",
            "cpu",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

