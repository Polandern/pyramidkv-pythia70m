from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.pyramidkv.cache import build_layer_budgets, cache_stats, compress_past_key_values
from src.pyramidkv.data import load_text
from src.pyramidkv.modeling import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CPU-friendly streaming perplexity.")
    parser.add_argument("--method", choices=["dense", "pyramidkv"], default="dense")
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample_text", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_documents", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--kv_budget", type=int, default=128)
    parser.add_argument("--budget_mode", choices=["pyramid", "uniform"], default="pyramid")
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_tokens", type=int, default=32)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    text = load_text(
        sample_text=args.sample_text,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_documents=args.max_documents,
        streaming=args.streaming,
    )
    token_ids = tokenizer(text, return_tensors="pt").input_ids[:, : args.max_tokens].to(device)
    if token_ids.shape[1] < 2:
        raise ValueError("Need at least two tokens to compute perplexity.")

    budgets = build_layer_budgets(
        num_layers=model.config.num_hidden_layers,
        kv_budget=args.kv_budget,
        mode=args.budget_mode,
    )

    past_key_values = None
    total_nll = 0.0
    total_targets = 0
    cache_lengths = []
    started = time.perf_counter()

    iterator = range(token_ids.shape[1] - 1)
    for pos in tqdm(iterator, desc=f"ppl:{args.method}"):
        current = token_ids[:, pos : pos + 1]
        target = token_ids[:, pos + 1]
        model_inputs = dict(
            input_ids=current,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=args.method == "pyramidkv",
            return_dict=True,
        )
        if args.method == "dense":
            model_inputs["position_ids"] = torch.tensor([[pos]], device=device, dtype=torch.long)
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :]
        total_nll += float(F.cross_entropy(logits, target, reduction="sum").item())
        total_targets += int(target.numel())
        past_key_values = outputs.past_key_values

        if args.method == "pyramidkv":
            past_key_values = compress_past_key_values(
                past_key_values,
                outputs.attentions,
                budgets=budgets,
                sink_tokens=args.sink_tokens,
                recent_tokens=args.recent_tokens,
            )
        cache_lengths.append(cache_stats(past_key_values).average_length)

    elapsed = time.perf_counter() - started
    ppl = math.exp(total_nll / max(1, total_targets))
    result = {
        "method": args.method,
        "model_name": args.model_name,
        "device": args.device,
        "max_tokens": int(token_ids.shape[1]),
        "kv_budget": args.kv_budget if args.method == "pyramidkv" else None,
        "budget_mode": args.budget_mode if args.method == "pyramidkv" else None,
        "nll": total_nll,
        "ppl": ppl,
        "tokens_evaluated": total_targets,
        "elapsed_sec": elapsed,
        "avg_cache_length": sum(cache_lengths) / len(cache_lengths),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.method
    out_file = out_dir / f"ppl_{run_name}_{args.method}_tokens{token_ids.shape[1]}_budget{args.kv_budget}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
