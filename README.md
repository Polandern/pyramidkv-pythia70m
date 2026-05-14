# PyramidKV on Pythia-70M

CPU-friendly reproduction code for KV-cache compression experiments on
`EleutherAI/pythia-70m`.

This repository is prepared for the course project "Efficient Inference for
Language Models". It contains a dense baseline and a PyramidKV-style KV cache
compression baseline. The default commands are intentionally small so they can
run on a CPU-only laptop; increase the sequence lengths when running on a GPU.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If model or dataset downloads are slow, set a Hugging Face mirror/cache before
running the scripts.

## Quick CPU Smoke Test

Dense baseline:

```powershell
python eval_ppl.py --method dense --sample_text samples/tiny_pg19.txt --max_tokens 128 --device cpu
python benchmark_speed.py --method dense --prompt_file samples/tiny_pg19.txt --prompt_tokens 128 --generate_tokens 16 --device cpu
```

PyramidKV reproduction:

```powershell
python eval_ppl.py --method pyramidkv --sample_text samples/tiny_pg19.txt --max_tokens 128 --kv_budget 64 --device cpu
python benchmark_speed.py --method pyramidkv --prompt_file samples/tiny_pg19.txt --prompt_tokens 128 --generate_tokens 16 --kv_budget 64 --device cpu
```

Equivalent convenience commands:

```powershell
python run_baseline.py
python run_pyramidkv.py
```

The scripts write JSON results into `results/`.

## CPU Reproduction

This repository includes local copies of the lightweight evaluation inputs:

- `samples/wikitext2_test.raw`: WikiText-2 raw test split.
- `samples/pg19_validation_35816.txt`: one PG-19 validation book sample.

Run the reported CPU reproduction:

```powershell
.\run_cpu_reproduction.ps1
```

Or run commands individually:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python eval_ppl.py --run_name wikitext2 --method dense --sample_text samples/wikitext2_test.raw --max_tokens 256 --device cpu
python eval_ppl.py --run_name wikitext2 --method pyramidkv --sample_text samples/wikitext2_test.raw --max_tokens 256 --kv_budget 128 --device cpu

python eval_ppl.py --run_name pg19_35816 --method dense --sample_text samples/pg19_validation_35816.txt --max_tokens 256 --device cpu
python eval_ppl.py --run_name pg19_35816 --method pyramidkv --sample_text samples/pg19_validation_35816.txt --max_tokens 256 --kv_budget 128 --device cpu

python benchmark_speed.py --run_name wikitext2 --method dense --prompt_file samples/wikitext2_test.raw --prompt_tokens 256 --generate_tokens 32 --device cpu
python benchmark_speed.py --run_name wikitext2 --method pyramidkv --prompt_file samples/wikitext2_test.raw --prompt_tokens 256 --generate_tokens 32 --kv_budget 128 --device cpu
```

### CPU Results

Environment: Windows CPU-only laptop, Python 3.11.9, PyTorch 2.12.0+cpu,
Transformers 4.44.2, model `EleutherAI/pythia-70m`.

Perplexity, 256-token evaluation:

| Dataset | Method | KV budget | PPL | Avg. KV length/layer | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| WikiText-2 raw test | Dense | full | 57.98 | 128.00 | 2.46 |
| WikiText-2 raw test | PyramidKV | 128 | 263.47 | 77.19 | 2.37 |
| PG-19 validation sample 35816 | Dense | full | 26.99 | 128.00 | 2.07 |
| PG-19 validation sample 35816 | PyramidKV | 128 | 79.97 | 77.19 | 2.24 |

Generation speed on WikiText-2 prompt, 256 prompt tokens and 32 generated
tokens:

| Method | KV budget | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak RSS (MB) | Avg. KV length/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | full | 0.183 | 0.0123 | 81.20 | 639.00 | 288.00 |
| PyramidKV | 128 | 0.162 | 0.0127 | 78.86 | 650.73 | 96.00 |

On this CPU setting, PyramidKV clearly reduces the retained KV length, but it
does not improve throughput because the Python-level attention collection and
cache compression overhead dominate. This CPU run is therefore a reproducibility
check rather than a final performance claim.

Implementation note: the current Hugging Face GPT-NeoX legacy tuple cache does
not expose sparse absolute cache positions. For CPU reproduction, PyramidKV uses
cache-local positions after compression so the baseline remains runnable with
`transformers==4.44.2`. GPU follow-up experiments should revisit this with a
cache implementation that supports sparse RoPE positions.

## Suggested Full Experiments

On a GPU machine, use the same commands but increase `--max_tokens`,
`--prompt_tokens`, and `--generate_tokens`.

Recommended sweep:

```powershell
python eval_ppl.py --method dense --dataset wikitext --dataset_config wikitext-2-raw-v1 --max_tokens 2048 --device cuda
python eval_ppl.py --method pyramidkv --dataset wikitext --dataset_config wikitext-2-raw-v1 --max_tokens 2048 --kv_budget 128 --device cuda
python eval_ppl.py --method pyramidkv --dataset wikitext --dataset_config wikitext-2-raw-v1 --max_tokens 2048 --kv_budget 256 --device cuda
python eval_ppl.py --method pyramidkv --dataset wikitext --dataset_config wikitext-2-raw-v1 --max_tokens 2048 --kv_budget 512 --device cuda
```

For PG-19, use a single long validation sample:

```powershell
python eval_ppl.py --method pyramidkv --dataset pg19 --split validation --streaming --max_tokens 4096 --kv_budget 512 --device cuda
```

## What Is Implemented

- Dense baseline with normal Hugging Face KV cache.
- PyramidKV-style layer-wise budgets.
- Token retention policy:
  - attention sink tokens at the beginning,
  - recent window tokens,
  - high-attention historical tokens selected from the latest attention map.
- CPU-compatible perplexity and speed benchmark scripts.
- JSON logging for reproducibility.

## Output Metrics

`eval_ppl.py` reports:

- negative log-likelihood,
- perplexity,
- token count,
- elapsed time,
- average retained KV length per layer.

`benchmark_speed.py` reports:

- TTFT,
- TPOT,
- throughput,
- generated token count,
- peak process RSS memory.

## Notes

CPU runs are only for correctness and reproducibility checks. They are expected
to be slow and may not show realistic acceleration because Python overhead and
CPU attention kernels can dominate. Use GPU results for the final paper tables.
