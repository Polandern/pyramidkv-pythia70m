$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

.\.venv\Scripts\python.exe eval_ppl.py --run_name wikitext2 --method dense --sample_text samples/wikitext2_test.raw --max_tokens 256 --device cpu
.\.venv\Scripts\python.exe eval_ppl.py --run_name wikitext2 --method pyramidkv --sample_text samples/wikitext2_test.raw --max_tokens 256 --kv_budget 128 --device cpu

.\.venv\Scripts\python.exe eval_ppl.py --run_name pg19_35816 --method dense --sample_text samples/pg19_validation_35816.txt --max_tokens 256 --device cpu
.\.venv\Scripts\python.exe eval_ppl.py --run_name pg19_35816 --method pyramidkv --sample_text samples/pg19_validation_35816.txt --max_tokens 256 --kv_budget 128 --device cpu

.\.venv\Scripts\python.exe benchmark_speed.py --run_name wikitext2 --method dense --prompt_file samples/wikitext2_test.raw --prompt_tokens 256 --generate_tokens 32 --device cpu
.\.venv\Scripts\python.exe benchmark_speed.py --run_name wikitext2 --method pyramidkv --prompt_file samples/wikitext2_test.raw --prompt_tokens 256 --generate_tokens 32 --kv_budget 128 --device cpu

.\.venv\Scripts\python.exe scripts\summarize_results.py
