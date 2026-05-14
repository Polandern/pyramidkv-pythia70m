from __future__ import annotations

import subprocess
import sys


def main() -> None:
    commands = [
        [
            sys.executable,
            "eval_ppl.py",
            "--method",
            "dense",
            "--sample_text",
            "samples/tiny_pg19.txt",
            "--max_tokens",
            "128",
            "--device",
            "cpu",
        ],
        [
            sys.executable,
            "benchmark_speed.py",
            "--method",
            "dense",
            "--prompt_file",
            "samples/tiny_pg19.txt",
            "--prompt_tokens",
            "128",
            "--generate_tokens",
            "16",
            "--device",
            "cpu",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

