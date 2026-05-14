from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import psutil
import torch

from src.pyramidkv.cache import build_layer_budgets, cache_stats, compress_past_key_values
from src.pyramidkv.modeling import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dense vs PyramidKV generation.")
    parser.add_argument("--method", choices=["dense", "pyramidkv"], default="dense")
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--prompt_file", default="samples/tiny_pg19.txt")
    parser.add_argument("--prompt_tokens", type=int, default=128)
    parser.add_argument("--generate_tokens", type=int, default=16)

    parser.add_argument("--kv_budget", type=int, default=128)
    parser.add_argument("--budget_mode", choices=["pyramid", "uniform"], default="pyramid")
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_tokens", type=int, default=32)

    parser.add_argument(
        "--selection_strategy",
        choices=["attention", "recent"],
        default="attention",
        help=(
            "KV selection strategy for pyramidkv. "
            "'attention' uses the latest attention map; "
            "'recent' avoids output_attentions and keeps sink + recent-biased tokens."
        ),
    )

    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    process = psutil.Process()
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.prompt_tokens].to(device)

    budgets = build_layer_budgets(
        num_layers=model.config.num_hidden_layers,
        kv_budget=args.kv_budget,
        mode=args.budget_mode,
    )

    # Only the attention-based reproduction needs attention tensors.
    # The recent strategy is a lower-overhead ablation and should not request
    # output_attentions=True.
    need_attn = args.method == "pyramidkv" and args.selection_strategy == "attention"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = time.perf_counter()

    # Prefill stage.
    prefill_inputs = dict(
        input_ids=input_ids,
        use_cache=True,
        output_attentions=need_attn,
        return_dict=True,
    )

    # Dense keeps full cache, so absolute positions are safe.
    # PyramidKV physically deletes cache entries; for GPT-NeoX/Pythia this can
    # interact badly with RoPE if absolute position_ids are passed manually.
    if args.method == "dense":
        prefill_inputs["position_ids"] = torch.arange(
            input_ids.shape[1], device=device
        ).unsqueeze(0)

    outputs = model(**prefill_inputs)

    if is_cuda:
        torch.cuda.synchronize(device)

    prefill_done = time.perf_counter()

    past_key_values = outputs.past_key_values

    if args.method == "pyramidkv":
        past_key_values = compress_past_key_values(
            past_key_values,
            outputs.attentions if need_attn else None,
            budgets=budgets,
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
        )

        if is_cuda:
            torch.cuda.synchronize(device)

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated: list[int] = []

    peak_rss = process.memory_info().rss
    first_token_done = None

    # Decode stage.
    for step in range(args.generate_tokens):
        decode_inputs = dict(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=need_attn,
            return_dict=True,
        )

        if args.method == "dense":
            decode_inputs["position_ids"] = torch.tensor(
                [[input_ids.shape[1] + step]], device=device, dtype=torch.long
            )

        outputs = model(**decode_inputs)

        if is_cuda:
            torch.cuda.synchronize(device)

        if first_token_done is None:
            first_token_done = time.perf_counter()

        past_key_values = outputs.past_key_values

        if args.method == "pyramidkv":
            past_key_values = compress_past_key_values(
                past_key_values,
                outputs.attentions if need_attn else None,
                budgets=budgets,
                sink_tokens=args.sink_tokens,
                recent_tokens=args.recent_tokens,
            )

            if is_cuda:
                torch.cuda.synchronize(device)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(int(next_token.item()))

        peak_rss = max(peak_rss, process.memory_info().rss)

    if is_cuda:
        torch.cuda.synchronize(device)

    ended = time.perf_counter()

    ttft = (first_token_done or ended) - started
    decode_time = ended - prefill_done
    tpot = decode_time / max(1, args.generate_tokens)
    throughput = args.generate_tokens / max(1e-9, decode_time)

    peak_cuda_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if is_cuda
        else None
    )

    result = {
        "method": args.method,
        "model_name": args.model_name,
        "device": args.device,
        "prompt_tokens": int(input_ids.shape[1]),
        "generate_tokens": args.generate_tokens,
        "kv_budget": args.kv_budget if args.method == "pyramidkv" else None,
        "budget_mode": args.budget_mode if args.method == "pyramidkv" else None,
        "selection_strategy": args.selection_strategy if args.method == "pyramidkv" else None,
        "sink_tokens": args.sink_tokens if args.method == "pyramidkv" else None,
        "recent_tokens": args.recent_tokens if args.method == "pyramidkv" else None,
        "ttft_sec": ttft,
        "tpot_sec": tpot,
        "throughput_tok_per_sec": throughput,
        "elapsed_sec": ended - started,
        "peak_rss_mb": peak_rss / (1024**2),
        "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
        "avg_cache_length": cache_stats(past_key_values).average_length,
        "generated_text": tokenizer.decode(generated),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or args.method
    out_file = (
        out_dir
        / f"speed_{run_name}_{args.method}_prompt{input_ids.shape[1]}_gen{args.generate_tokens}_budget{args.kv_budget}.json"
    )

    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()