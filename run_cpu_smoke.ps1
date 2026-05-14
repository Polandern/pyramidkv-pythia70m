$ErrorActionPreference = "Stop"

python eval_ppl.py --method dense --sample_text samples/tiny_pg19.txt --max_tokens 128 --device cpu
python eval_ppl.py --method pyramidkv --sample_text samples/tiny_pg19.txt --max_tokens 128 --kv_budget 64 --device cpu
python benchmark_speed.py --method dense --prompt_file samples/tiny_pg19.txt --prompt_tokens 128 --generate_tokens 16 --device cpu
python benchmark_speed.py --method pyramidkv --prompt_file samples/tiny_pg19.txt --prompt_tokens 128 --generate_tokens 16 --kv_budget 64 --device cpu

