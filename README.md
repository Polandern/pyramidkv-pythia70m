# PyramidKV 在 Pythia-70M 上的复现实验

本仓库用于课程大作业“语言模型高效推理”的个人部分。目标是在不修改模型参数、不重新训练模型的前提下，复现一种 KV Cache 压缩方法，并与 Dense baseline 进行对比。

本项目选择复现 **PyramidKV 风格的逐层 KV Cache 压缩方法**。实验模型为 `EleutherAI/pythia-70m`，评估内容包括：

- WikiText-2 上的 perplexity；
- PG-19 单样本上的 perplexity；
- 生成阶段的 TTFT、TPOT、throughput；
- 平均 KV Cache 长度；
- CPU 内存和 CUDA 显存占用。

需要提前说明：本仓库中的 PyramidKV 是一个面向课程复现的轻量版本，而不是完整工程级优化实现。实验结果显示，当前 attention-based 简化实现虽然能显著降低平均 KV Cache 长度，但会造成明显 PPL 退化，并且没有带来端到端推理加速。因此，本 README 不把实验包装成“成功加速”，而是完整记录 baseline、失败现象、失败原因分析，以及后续 recent/uniform ablation 的改进结果。

---

## 1. 方法简介

Dense baseline 使用 Hugging Face 默认的 KV Cache，保留所有历史 token 的 key/value 状态。

PyramidKV-style 方法为每一层设置一个 KV budget。当当前层 cache 长度超过 budget 后，只保留一部分 token 的 KV：

1. 开头的 attention sink tokens；
2. 最近的 recent tokens；
3. 中间区域中根据策略选择的历史 tokens。

本仓库目前支持两种选择策略：

### 1.1 `attention` 策略

这是最接近原始复现思路的版本。它会开启：

```bash
output_attentions=True
```

然后根据最新 token 的 attention map，从中间历史区域选择高 attention 的 token 保留。

优点：

- 更接近 attention-score based KV selection 的直觉；
- 可以观察 PyramidKV-style 压缩是否能降低 KV Cache 长度。

缺点：

- 每一步都要额外输出 attention；
- 需要 Python 层执行 top-k、index_select 和 cache reconstruction；
- 在 Pythia/GPT-NeoX + Hugging Face legacy tuple cache 接口下，容易受到 RoPE 位置处理影响；
- 实验中 PPL 明显变差，速度也没有提升。

### 1.2 `recent` 策略

这是基于失败分析后加入的 ablation。它不再开启 `output_attentions=True`，而是在没有 attention score 的情况下保留 sink tokens 和最近的历史 tokens。

它不是严格论文版 PyramidKV，而是一个用于诊断工程开销和压缩策略的对照实验：

- 如果 `recent` 比 `attention` 更快，说明 `output_attentions=True` 和 Python 层 attention selection 是速度瓶颈之一；
- 如果 `recent` 的 PPL 更好，说明当前 attention-based 简化选择策略不适合 Pythia-70M 的这一设置；
- 如果 larger budget 能改善 PPL，说明当前方法存在明显的 compression-quality trade-off。

---

### 1.3 改进算法：SC-PyramidKV

小组创新部分中，本仓库进一步实现了一个独立的 PyramidKV 改进版：
**SC-PyramidKV**，即 Sensitivity-Calibrated PyramidKV。它不与
StreamingLLM 或 SnapKV 融合，而是专门改进 PyramidKV 自身。

SC-PyramidKV 的动机来自当前复现实验的失败分析：简单 attention top-k
压缩在 Pythia/GPT-NeoX 上会造成明显 PPL 退化，且每 token 压缩带来较大
Python 开销。因此，SC-PyramidKV 做了四个改动：

1. **Layer sensitivity calibration**：先用一小段校准文本估计每层 attention
   entropy 和 long-range attention mass。更敏感的层获得更多 KV budget。
2. **Sensitivity-calibrated budget**：在 PyramidKV 的逐层预算基础上重新分配
   budget，但保持总体预算规模相近。
3. **Warmup + block-wise compression**：前若干 token 不压缩，之后每隔固定
   interval 压缩一次，避免每 token 重建 KV cache。
4. **Sink + recent + landmark retention**：保留 attention sink、最近窗口和
   中间区域均匀 landmark tokens，比单纯 attention top-k 更稳定。

相关文件：

- `src/pyramidkv/sc_pyramidkv.py`
- `eval_sc_pyramidkv.py`
- `benchmark_sc_pyramidkv.py`
- `run_sc_pyramidkv.py`

GPU 运行示例：

```bash
python eval_sc_pyramidkv.py \
  --run_name gpu4090_wikitext2_t2048_sc_b1024 \
  --model_name $MODEL \
  --sample_text samples/wikitext2_test.raw \
  --max_tokens 2048 \
  --kv_budget 1024 \
  --recent_tokens 256 \
  --warmup_tokens 1024 \
  --compress_interval 128 \
  --device cuda \
  --out_dir results_sc

python benchmark_sc_pyramidkv.py \
  --run_name gpu4090_wikitext2_p1536_g128_sc_b1024 \
  --model_name $MODEL \
  --prompt_file samples/wikitext2_test.raw \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --kv_budget 1024 \
  --recent_tokens 256 \
  --warmup_tokens 1024 \
  --compress_interval 128 \
  --device cuda \
  --out_dir results_sc
```

建议消融：

- `--sensitivity_alpha 0`：去掉 sensitivity calibration；
- `--budget_mode uniform`：去掉 PyramidKV layer-wise budget；
- `--middle_strategy recent`：去掉 landmark tokens；
- `--compress_interval 1`：退回每 token 压缩。

### 1.4 单文件版 improved PyramidKV

为了便于小组仓库集成，本仓库另外提供了一个单文件版本：

```text
improve_pyramidkv.py
```

它将模型加载、数据读取、PPL 评测、速度测试、KV 压缩逻辑和 CSV 写入都放在一个文件中，格式与组内其他方法实现保持一致。该文件主要用于后续小组项目中作为 `methods/improve_pyramidkv.py` 迁移。

推荐运行命令：

```bash
python improve_pyramidkv.py \
  --model_id $MODEL \
  --dataset wikitext \
  --wikitext_local_path samples/wikitext2_test.raw \
  --split test \
  --max_eval_tokens 2048 \
  --max_context_tokens 1536 \
  --max_new_tokens 128 \
  --kv_budget 1536 \
  --recent_tokens 512 \
  --warmup_tokens 1536 \
  --compress_interval 256 \
  --sensitivity_alpha 0 \
  --output_csv results/pyramidkv_improved_metrics.csv
```

当前 RTX 4090 上的 WikiText-2 单文件版结果如下：

| Method | Dataset | Eval tokens | Context tokens | KV budget | PPL | Throughput (tok/s) | PPL Avg. KV | Speed Avg. KV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| improved PyramidKV | WikiText-2 | 2048 | 1536 | 1536 | 74.63 | 204.84 | 863.92 | 1151.00 |

这一结果说明：相比 attention-based PyramidKV 复现版本，改进版显著降低了 PPL 退化，并且在单文件评测脚本中取得了接近或略高于 Dense baseline 的生成吞吐。不过，PPL 仍然高于 Dense，因此该方法更适合作为“质量退化明显改善 + 保持一定 KV 压缩”的 PyramidKV 改进版本，而不是无损压缩方法。

---

## 2. 仓库结构

```text
PyramidKV/
├── improve_pyramidkv.py        # 单文件版改进 PyramidKV，便于小组仓库集成
├── benchmark_speed.py          # 生成速度测试：TTFT、TPOT、throughput、显存
├── benchmark_sc_pyramidkv.py   # SC-PyramidKV 速度测试 CLI 包装器
├── eval_ppl.py                 # PPL 测试
├── eval_sc_pyramidkv.py        # SC-PyramidKV PPL 测试 CLI 包装器
├── methods/
│   ├── __init__.py
│   ├── improve_pyramidkv.py    # 单文件版改进 PyramidKV：加载、评测、测速、写 CSV
│   └── pyramidkv.py            # 小组框架形式的 PyramidKV/SC-PyramidKV 方法模块
├── run_baseline.py             # Dense baseline 快速入口
├── run_pyramidkv.py            # PyramidKV 快速入口
├── run_sc_pyramidkv.py         # SC-PyramidKV 快速入口
├── run_cpu_reproduction.ps1    # CPU 复现实验脚本
├── requirements.txt
├── samples/
│   ├── tiny_pg19.txt
│   ├── wikitext2_test.raw
│   └── pg19_validation_35816.txt
├── scripts/
│   └── summarize_results.py
├── src/
│   └── pyramidkv/
│       ├── cache.py            # KV budget 和 cache 压缩逻辑
│       ├── data.py             # 数据读取
│       ├── sc_pyramidkv.py     # 改进版 SC-PyramidKV 算法
│       └── modeling.py         # 模型和 tokenizer 加载
├── results/                    # CPU 实验结果
├── results_4090/               # RTX 4090 个人复现实验结果
└── results_sc/                 # SC-PyramidKV 结果，运行后生成
```

---

## 3. 环境配置

### 3.1 CPU 环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.2 RTX 4090 云端环境

Linux / Ubuntu：

```bash
conda create -n pyramidkv python=3.11 -y
conda activate pyramidkv

python -m pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 "datasets>=2.18" "accelerate>=0.28" "psutil>=5.9" "tqdm>=4.66" fsspec
```

检查 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

如果服务器无法直接访问 Hugging Face，可以使用镜像或提前下载模型。推荐下载到本地路径：

```bash
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "EleutherAI/pythia-70m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

tok.save_pretrained("/KV/models/pythia-70m")
model.save_pretrained("/KV/models/pythia-70m")

print("saved to /KV/models/pythia-70m")
PY
```

之后设置：

```bash
MODEL=/KV/models/pythia-70m
```

后续所有命令都可以使用：

```bash
--model_name $MODEL
```

---

## 4. 快速 smoke test

Dense：

```bash
python eval_ppl.py \
  --run_name smoke_dense \
  --method dense \
  --model_name $MODEL \
  --sample_text samples/tiny_pg19.txt \
  --max_tokens 128 \
  --device cuda \
  --out_dir results_4090
```

PyramidKV attention 策略：

```bash
python eval_ppl.py \
  --run_name smoke_pyramid_attention \
  --method pyramidkv \
  --selection_strategy attention \
  --model_name $MODEL \
  --sample_text samples/tiny_pg19.txt \
  --max_tokens 128 \
  --kv_budget 64 \
  --device cuda \
  --out_dir results_4090
```

PyramidKV recent 策略：

```bash
python eval_ppl.py \
  --run_name smoke_pyramid_recent \
  --method pyramidkv \
  --selection_strategy recent \
  --model_name $MODEL \
  --sample_text samples/tiny_pg19.txt \
  --max_tokens 128 \
  --kv_budget 64 \
  --device cuda \
  --out_dir results_4090
```

---

## 5. CPU 复现实验

CPU 实验主要用于证明代码可运行、结果可复现，不作为最终加速结论。

```powershell
.\run_cpu_reproduction.ps1
```

或单独运行：

```powershell
python eval_ppl.py --run_name wikitext2 --method dense --sample_text samples/wikitext2_test.raw --max_tokens 256 --device cpu
python eval_ppl.py --run_name wikitext2 --method pyramidkv --sample_text samples/wikitext2_test.raw --max_tokens 256 --kv_budget 128 --device cpu

python eval_ppl.py --run_name pg19_35816 --method dense --sample_text samples/pg19_validation_35816.txt --max_tokens 256 --device cpu
python eval_ppl.py --run_name pg19_35816 --method pyramidkv --sample_text samples/pg19_validation_35816.txt --max_tokens 256 --kv_budget 128 --device cpu

python benchmark_speed.py --run_name wikitext2 --method dense --prompt_file samples/wikitext2_test.raw --prompt_tokens 256 --generate_tokens 32 --device cpu
python benchmark_speed.py --run_name wikitext2 --method pyramidkv --prompt_file samples/wikitext2_test.raw --prompt_tokens 256 --generate_tokens 32 --kv_budget 128 --device cpu
```

### 5.1 CPU 结果

环境：Windows CPU-only laptop，Python 3.11.9，PyTorch CPU，Transformers 4.44.2，模型 `EleutherAI/pythia-70m`。

PPL，256-token evaluation：

| Dataset | Method | KV budget | PPL | Avg. KV length/layer | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| WikiText-2 raw test | Dense | full | 57.98 | 128.00 | 2.46 |
| WikiText-2 raw test | PyramidKV attention | 128 | 263.47 | 77.19 | 2.37 |
| PG-19 validation sample 35816 | Dense | full | 26.99 | 128.00 | 2.07 |
| PG-19 validation sample 35816 | PyramidKV attention | 128 | 79.97 | 77.19 | 2.24 |

WikiText-2 prompt，256 prompt tokens，32 generated tokens：

| Method | KV budget | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak RSS (MB) | Avg. KV length/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | full | 0.183 | 0.0123 | 81.20 | 639.00 | 288.00 |
| PyramidKV attention | 128 | 0.162 | 0.0127 | 78.86 | 650.73 | 96.00 |

CPU 结果说明：PyramidKV-style 压缩可以降低平均 KV 长度，但在 CPU 上没有带来可靠吞吐提升。原因是 CPU attention kernel 和 Python 层 cache 压缩开销占比较高。

---

## 6. RTX 4090 实验：Dense baseline 与 attention-based PyramidKV 失败记录

### 6.1 实验环境

- GPU: NVIDIA GeForce RTX 4090 24GB
- Python: 3.11
- PyTorch: CUDA 12.1 wheel
- Transformers: 4.44.2
- Model: `EleutherAI/pythia-70m`
- 本地模型路径：`/KV/models/pythia-70m`
- 数据：
  - `samples/wikitext2_test.raw`
  - `samples/pg19_validation_35816.txt`

### 6.2 PPL 实验命令

WikiText-2，1024 tokens：

```bash
python eval_ppl.py \
  --run_name gpu4090_wikitext2_t1024 \
  --method dense \
  --model_name $MODEL \
  --sample_text samples/wikitext2_test.raw \
  --max_tokens 1024 \
  --device cuda \
  --out_dir results_4090

for B in 256 512 1024; do
  python eval_ppl.py \
    --run_name gpu4090_wikitext2_t1024_b${B} \
    --method pyramidkv \
    --selection_strategy attention \
    --model_name $MODEL \
    --sample_text samples/wikitext2_test.raw \
    --max_tokens 1024 \
    --kv_budget ${B} \
    --device cuda \
    --out_dir results_4090
done
```

WikiText-2，2048 tokens：

```bash
python eval_ppl.py \
  --run_name gpu4090_wikitext2_t2048 \
  --method dense \
  --model_name $MODEL \
  --sample_text samples/wikitext2_test.raw \
  --max_tokens 2048 \
  --device cuda \
  --out_dir results_4090

for B in 256 512 1024; do
  python eval_ppl.py \
    --run_name gpu4090_wikitext2_t2048_b${B} \
    --method pyramidkv \
    --selection_strategy attention \
    --model_name $MODEL \
    --sample_text samples/wikitext2_test.raw \
    --max_tokens 2048 \
    --kv_budget ${B} \
    --device cuda \
    --out_dir results_4090
done
```

PG-19，2048 tokens：

```bash
python eval_ppl.py \
  --run_name gpu4090_pg19_35816_t2048 \
  --method dense \
  --model_name $MODEL \
  --sample_text samples/pg19_validation_35816.txt \
  --max_tokens 2048 \
  --device cuda \
  --out_dir results_4090

for B in 256 512 1024; do
  python eval_ppl.py \
    --run_name gpu4090_pg19_35816_t2048_b${B} \
    --method pyramidkv \
    --selection_strategy attention \
    --model_name $MODEL \
    --sample_text samples/pg19_validation_35816.txt \
    --max_tokens 2048 \
    --kv_budget ${B} \
    --device cuda \
    --out_dir results_4090
done
```

### 6.3 PPL 结果

WikiText-2，1024 tokens：

| Method | KV budget | PPL | Avg. KV length/layer | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| Dense | full | 27.54 | 512.00 | 5.01 |
| PyramidKV attention | 256 | 488.18 | 173.15 | 6.19 |
| PyramidKV attention | 512 | 345.88 | 308.37 | 6.05 |
| PyramidKV attention | 1024 | 139.77 | 465.13 | 5.51 |

WikiText-2，2048 tokens：

| Method | KV budget | PPL | Avg. KV length/layer | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| Dense | full | 35.13 | 1024.00 | 9.85 |
| PyramidKV attention | 256 | 724.39 | 182.58 | 12.40 |
| PyramidKV attention | 512 | 653.37 | 346.20 | 12.25 |
| PyramidKV attention | 1024 | 354.07 | 616.64 | 11.74 |

PG-19，2048 tokens：

| Method | KV budget | PPL | Avg. KV length/layer | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| Dense | full | 29.57 | 1024.00 | 9.92 |
| PyramidKV attention | 256 | 320.92 | 182.58 | 12.52 |
| PyramidKV attention | 512 | 273.48 | 346.20 | 12.07 |
| PyramidKV attention | 1024 | 170.18 | 616.64 | 11.78 |

### 6.4 速度实验命令

WikiText-2：

```bash
python benchmark_speed.py \
  --run_name gpu4090_wikitext2_p1536_g128 \
  --method dense \
  --model_name $MODEL \
  --prompt_file samples/wikitext2_test.raw \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --device cuda \
  --out_dir results_4090

for B in 256 512 1024; do
  python benchmark_speed.py \
    --run_name gpu4090_wikitext2_p1536_g128_b${B} \
    --method pyramidkv \
    --selection_strategy attention \
    --model_name $MODEL \
    --prompt_file samples/wikitext2_test.raw \
    --prompt_tokens 1536 \
    --generate_tokens 128 \
    --kv_budget ${B} \
    --device cuda \
    --out_dir results_4090
done
```

PG-19：

```bash
python benchmark_speed.py \
  --run_name gpu4090_pg19_p1536_g128 \
  --method dense \
  --model_name $MODEL \
  --prompt_file samples/pg19_validation_35816.txt \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --device cuda \
  --out_dir results_4090

for B in 256 512 1024; do
  python benchmark_speed.py \
    --run_name gpu4090_pg19_p1536_g128_b${B} \
    --method pyramidkv \
    --selection_strategy attention \
    --model_name $MODEL \
    --prompt_file samples/pg19_validation_35816.txt \
    --prompt_tokens 1536 \
    --generate_tokens 128 \
    --kv_budget ${B} \
    --device cuda \
    --out_dir results_4090
done
```

### 6.5 速度结果

WikiText-2，prompt 1536，generate 128：

| Method | KV budget | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak CUDA MB | Avg. KV length/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | full | 0.260 | 0.00496 | 201.65 | 410.23 | 1664 |
| PyramidKV attention | 256 | 0.302 | 0.00687 | 145.62 | 590.23 | 192 |
| PyramidKV attention | 512 | 0.323 | 0.00690 | 144.98 | 596.66 | 384 |
| PyramidKV attention | 1024 | 0.292 | 0.00684 | 146.18 | 611.09 | 768 |

PG-19，prompt 1536，generate 128：

| Method | KV budget | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak CUDA MB | Avg. KV length/layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | full | 0.266 | 0.00503 | 198.89 | 410.23 | 1664 |
| PyramidKV attention | 256 | 0.330 | 0.00681 | 146.82 | 590.23 | 192 |
| PyramidKV attention | 512 | 0.329 | 0.00683 | 146.34 | 596.66 | 384 |
| PyramidKV attention | 1024 | 0.323 | 0.00683 | 146.34 | 611.09 | 768 |

### 6.6 失败分析

这组 RTX 4090 实验说明：

1. **KV Cache 压缩是有效的。**  
   例如在 speed 实验中，Dense 的平均 KV length/layer 为 1664，而 PyramidKV attention 在 budget 256、512、1024 下分别降低到 192、384、768。

2. **PPL 退化明显。**  
   在 PG-19 2048-token 实验中，Dense PPL 为 29.57，而 PyramidKV attention 1024 的 PPL 仍然达到 170.18。说明当前简化压缩策略会丢失对预测有用的上下文。

3. **端到端速度没有提升。**  
   Dense 在 RTX 4090 上约为 199--202 tok/s，而 PyramidKV attention 约为 145--147 tok/s。原因是当前实现每步需要 `output_attentions=True`，并在 Python 层执行 cache selection 和 `index_select`，这些额外开销抵消了 KV 变短的收益。

4. **GPU 显存峰值没有下降，反而上升。**  
   Dense 的 peak CUDA allocated 约为 410 MB，而 PyramidKV attention 约为 590--611 MB。主要原因可能是 attention tensor 输出和压缩过程中的临时张量增加了显存占用。

5. **RoPE 位置处理是一个关键限制。**  
   Pythia-70M 属于 GPT-NeoX 架构，使用 rotary position embedding。当前代码采用物理删除 KV 的方式压缩 cache。若在压缩后继续手动传入绝对 `position_ids`，会触发 RoPE 相关 index overflow；若不传绝对 `position_ids`，模型会使用 cache-local positions，程序可以跑通，但上下文位置语义不严格，可能造成质量退化。因此当前实现更适合作为复现 scaffold 和诊断 baseline，而不是最终优化实现。

---

## 7. 改进 ablation：recent + uniform

为了进一步分析 attention-based PyramidKV 失败的原因，本仓库加入了 `--selection_strategy recent`。

该策略不打开 `output_attentions=True`，而是保留 sink tokens 和最近的历史 tokens。它不是严格论文版 PyramidKV，而是一个工程 ablation，用来验证：

- attention 输出是否是主要速度和显存瓶颈；
- uniform budget 是否比当前手写 pyramid budget 更稳定；
- 更宽松的 budget 是否能降低 PPL 退化；
- KV budget 与 PPL、throughput、cache length 之间的 trade-off。

### 7.1 PG-19 PPL ablation 命令

```bash
python eval_ppl.py \
  --run_name gpu4090_pg19_t2048_recent_uniform1024 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --sample_text samples/pg19_validation_35816.txt \
  --max_tokens 2048 \
  --kv_budget 1024 \
  --device cuda \
  --out_dir results_4090

python eval_ppl.py \
  --run_name gpu4090_pg19_t2048_recent_uniform1536 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --sample_text samples/pg19_validation_35816.txt \
  --max_tokens 2048 \
  --kv_budget 1536 \
  --device cuda \
  --out_dir results_4090

python eval_ppl.py \
  --run_name gpu4090_pg19_t2048_recent_uniform1792 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --sample_text samples/pg19_validation_35816.txt \
  --max_tokens 2048 \
  --kv_budget 1792 \
  --device cuda \
  --out_dir results_4090

python eval_ppl.py \
  --run_name gpu4090_pg19_t2048_recent_uniform1920 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --sample_text samples/pg19_validation_35816.txt \
  --max_tokens 2048 \
  --kv_budget 1920 \
  --device cuda \
  --out_dir results_4090
```

### 7.2 PG-19 PPL ablation 结果

| Method | KV budget | Budget mode | Selection | PPL | Avg. KV length/layer | Time (s) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Dense | full | full | full | 29.57 | 1024.00 | 9.92 |
| PyramidKV attention | 1024 | pyramid | attention | 170.18 | 616.64 | 11.78 |
| PyramidKV recent | 1024 | uniform | recent | 91.70 | 768.13 | 10.63 |
| PyramidKV recent | 1536 | uniform | recent | 49.63 | 960.09 | 10.17 |
| PyramidKV recent | 1792 | uniform | recent | 36.27 | 1008.05 | 10.01 |
| PyramidKV recent | 1920 | uniform | recent | 32.55 | 1020.03 | 10.01 |

### 7.3 PG-19 speed ablation 命令

```bash
python benchmark_speed.py \
  --run_name gpu4090_pg19_p1536_g128_recent_uniform1024 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --prompt_file samples/pg19_validation_35816.txt \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --kv_budget 1024 \
  --device cuda \
  --out_dir results_4090

python benchmark_speed.py \
  --run_name gpu4090_pg19_p1536_g128_recent_uniform1536 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --prompt_file samples/pg19_validation_35816.txt \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --kv_budget 1536 \
  --device cuda \
  --out_dir results_4090

python benchmark_speed.py \
  --run_name gpu4090_pg19_p1536_g128_recent_uniform1792 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --prompt_file samples/pg19_validation_35816.txt \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --kv_budget 1792 \
  --device cuda \
  --out_dir results_4090

python benchmark_speed.py \
  --run_name gpu4090_pg19_p1536_g128_recent_uniform1920 \
  --method pyramidkv \
  --selection_strategy recent \
  --budget_mode uniform \
  --model_name $MODEL \
  --prompt_file samples/pg19_validation_35816.txt \
  --prompt_tokens 1536 \
  --generate_tokens 128 \
  --kv_budget 1920 \
  --device cuda \
  --out_dir results_4090
```

### 7.4 PG-19 speed ablation 结果

| Method | KV budget | Budget mode | Selection | TTFT (s) | TPOT (s) | Throughput (tok/s) | Peak CUDA MB | Avg. KV length/layer |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense | full | full | full | 0.266 | 0.00503 | 198.89 | 410.23 | 1664 |
| PyramidKV attention | 1024 | pyramid | attention | 0.323 | 0.00683 | 146.34 | 611.09 | 768 |
| PyramidKV recent | 1024 | uniform | recent | 0.277 | 0.00579 | 172.57 | 410.23 | 1024 |
| PyramidKV recent | 1536 | uniform | recent | 0.266 | 0.00606 | 165.11 | 410.23 | 1536 |
| PyramidKV recent | 1792 | uniform | recent | 0.264 | 0.00496 | 201.65 | 410.23 | 1664 |
| PyramidKV recent | 1920 | uniform | recent | 0.265 | 0.00501 | 199.62 | 410.23 | 1664 |

### 7.5 Ablation 分析

`recent + uniform` 相比 attention-based 简化 PyramidKV 有明显改善，但不同 budget 的意义不同。

1. **PPL 明显改善。**  
   PG-19 2048-token 下，PyramidKV attention-1024 的 PPL 为 170.18，而 recent-uniform1536 降到 49.63，recent-uniform1792 降到 36.27，recent-uniform1920 进一步降到 32.55。Dense baseline 为 29.57，因此大 budget 的 recent 策略已经接近 Dense 质量。

2. **显存峰值恢复到 Dense 同级别。**  
   PyramidKV attention-1024 的 peak CUDA allocated 为 611.09 MB，而 recent-uniform 系列均为约 410.23 MB，与 Dense 基本一致。这说明 `output_attentions=True` 和 attention tensor 是显存峰值上升的重要原因。

3. **中等 budget 有真实压缩，但速度仍未超过 Dense。**  
   recent-uniform1024 的平均 KV length/layer 为 1024，低于 Dense 的 1664，throughput 为 172.57 tok/s，明显好于 attention-1024 的 146.34 tok/s，但仍低于 Dense 的 198.89 tok/s。这说明去掉 attention 输出后确实减少了开销，但 Python 层 cache 压缩仍然抵消了部分收益。

4. **大 budget 是近似无损上界，而不是有效压缩加速。**  
   recent-uniform1792 和 recent-uniform1920 的 PPL 分别为 36.27 和 32.55，非常接近 Dense；但在 prompt 1536 + generate 128 的 speed 实验中，它们的平均 KV length/layer 都是 1664，与 Dense 相同。这说明这两组主要展示“更大 budget 可恢复质量”，不能被解释为实际压缩加速。

5. **存在明显 trade-off。**  
   KV budget 越小，压缩越明显，但 PPL 越差；KV budget 越大，PPL 越接近 Dense，但压缩收益越弱。当前实现中，recent-uniform1024 是较有压缩意义的配置，recent-uniform1792/1920 更适合作为近似无损上界。

---

## 8. 输出指标说明

`eval_ppl.py` 输出：

- `nll`: negative log-likelihood；
- `ppl`: perplexity；
- `tokens_evaluated`: 实际评估 token 数；
- `elapsed_sec`: 运行时间；
- `avg_cache_length`: 每层平均保留 KV 长度；
- `selection_strategy`: `attention` 或 `recent`。

`benchmark_speed.py` 输出：

- `ttft_sec`: Time To First Token；
- `tpot_sec`: Time Per Output Token；
- `throughput_tok_per_sec`: 每秒生成 token 数；
- `peak_rss_mb`: 进程峰值 RSS 内存；
- `peak_cuda_allocated_mb`: PyTorch 记录的 CUDA 峰值显存；
- `avg_cache_length`: 每层平均保留 KV 长度；
- `generated_text`: 生成文本片段。

---

## 9. 结论

本项目复现了一个 PyramidKV-style KV Cache 压缩流程，并在 Pythia-70M 上完成了 Dense baseline、PPL 测试、速度测试、失败记录和改进 ablation。

主要结论如下：

1. 当前实现可以显著降低平均 KV Cache 长度。
2. attention-based 简化 PyramidKV 在 RTX 4090 上没有获得端到端加速。
3. attention-based 简化 PyramidKV 会导致明显 PPL 退化。
4. 失败原因主要包括：
   - `output_attentions=True` 带来的额外开销；
   - Python 层 top-k 和 cache compression 开销；
   - GPT-NeoX/Pythia 的 RoPE 位置编码与物理删除 KV 的简化实现不完全匹配；
   - Pythia-70M 是小模型，Dense baseline 本身已经很快，KV 压缩收益不容易覆盖额外开销。
5. `recent + uniform` ablation 明显改善了失败结果：
   - PG-19 PPL 从 attention-1024 的 170.18 降到 recent-uniform1536 的 49.63；
   - 更大的 recent-uniform1792 和 recent-uniform1920 进一步将 PPL 降到 36.27 和 32.55，接近 Dense 的 29.57；
   - Peak CUDA memory 从 attention-1024 的 611.09 MB 降回 410.23 MB；
   - Throughput 从 attention-1024 的 146.34 tok/s 提升到 recent-uniform1024 的 172.57 tok/s。
6. 但是，当前实现仍然没有在有效压缩设置下超过 Dense baseline：
   - Dense PG-19 PPL 为 29.57；
   - Dense throughput 为 198.89 tok/s；
   - recent-uniform1792/1920 的 throughput 接近 Dense，但它们的平均 KV length/layer 为 1664，与 Dense 相同，因此这两组应视为近似无损上界，而不是压缩加速结果。
7. 因此，本仓库不把当前结果包装成“成功加速”，而是将其作为一个透明的课程复现、失败分析和 ablation study。

最终结论是：**KV Cache 压缩本身并不自动等于推理加速。要获得真实收益，需要 RoPE-aware 的 cache position 处理、更低开销的 attention score 收集方式，以及更接近底层 kernel 的压缩实现。**
