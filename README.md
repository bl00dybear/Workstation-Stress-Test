# Workstation Stress Test

A GPU stress test and benchmark suite for multi-GPU workstations. Covers compute throughput, VRAM capacity, memory bandwidth, and LLM inference parallelism.

**Target hardware:** 2× NVIDIA RTX A2000 12 GB · Xeon E5-2640 v3 · 128 GB RAM

---

## Quick Start

```bash
git clone https://github.com/bl00dybear/Workstation-Stress-Test.git
cd Workstation-Stress-Test
chmod +x run.sh
./run.sh
```

`run.sh` installs `uv` if absent, creates the virtual environment, pulls all dependencies (including `torch+cu121`), verifies CUDA, and runs the full suite.

**Prerequisites (manual, one-time):**
- NVIDIA driver ≥ 525 installed and `nvidia-smi` functional
- `curl` available on the system

---

## Test Suite

| Script | Description |
|---|---|
| `gpu_full_stress_v2.py` | **Main entry point** — all tests, full CLI |
| `gpu_full_stress.py` | Original 5-test suite (no LLM) |
| `gpu_stress_compute.py` | Compute-only: GEMM FP16 + neural net AMP |
| `gpu_stress_vram.py` | VRAM fill + bandwidth |
| `gpu_monitor.py` | Real-time GPU monitor (temp, util, power, clocks) |
| `llm_model_parallel.py` | Standalone LLM parallelism benchmark |

### Tests included in `gpu_full_stress_v2.py`

| # | Test | What it stresses |
|---|---|---|
| 1 | Compute FP16 GEMM | Tensor Cores via large matrix multiply |
| 2 | Neural Network AMP | Forward + backward pass with mixed precision |
| 3 | VRAM Fill | Allocates 90% VRAM, measures fill bandwidth |
| 4 | VRAM Bandwidth | Sustained read/write between large buffers |
| 5 | Combined | Simultaneous compute + VRAM pressure |
| 6 | LLM Pipeline Parallel | HuggingFace `device_map="auto"` across 2 GPUs |
| 7 | LLM Tensor Parallel | `infer_auto_device_map` balanced split |

---

## Usage

```bash
./run.sh [test] [duration_seconds] [--quick]

./run.sh                      # full suite, 120s per test
./run.sh all 300              # full suite, 5 min per test
./run.sh all 60 --quick       # quick mode, 60s per test
./run.sh compute 180          # compute only, 3 min
./run.sh full 120             # 5 stress tests + both LLM tests
./run.sh llm_all 120          # LLM tests only
./run.sh llm_pipeline 120     # pipeline parallelism only
./run.sh llm_tensor 120       # tensor parallelism only
```

Or run directly with uv:

```bash
uv run python gpu_full_stress_v2.py --help

uv run python gpu_full_stress_v2.py --test full --duration 120
uv run python gpu_full_stress_v2.py --test compute --duration 300 --matrix-size 8192
uv run python gpu_full_stress_v2.py --llm --llm-model mistralai/Mistral-7B-Instruct-v0.3
uv run python gpu_full_stress_v2.py --llm --llm-max-new-tokens 256 --llm-n-prompts 40
uv run python gpu_monitor.py --interval 1
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--test` | `all` | Test to run: `compute`, `nn`, `vram_fill`, `bandwidth`, `combined`, `all`, `full`, `llm_all`, `llm_pipeline`, `llm_tensor` |
| `--duration` | `120` | Seconds per non-LLM test |
| `--quick` | off | Halves duration (60 s per test) |
| `--matrix-size` | `8192` | GEMM matrix dimension |
| `--llm` | off | Enable LLM tests alongside the standard 5 |
| `--llm-model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `--llm-n-prompts` | `20` | Number of inference prompts |
| `--llm-max-new-tokens` | `128` | Tokens generated per prompt |
| `--llm-batch-size` | `1` | Inference batch size |

---

## Output

Each run saves a timestamped JSON report:

```
gpu_stress_report_20260315_170000.json
```

The report contains system info, per-GPU results, temperatures before/after each test, and (if run) LLM throughput/latency metrics.

---

## Performance Targets (RTX A2000 12 GB)

| Metric | Good | Warning | Problem |
|---|---|---|---|
| GPU Utilization | ≥ 95% | 80–95% | < 80% |
| Temperature | < 75 °C | 75–85 °C | > 85 °C |
| VRAM Used | > 10 GB | 8–10 GB | < 8 GB |
| FP16 TFLOPS | > 15 | 10–15 | < 10 |
| VRAM Bandwidth | > 150 GB/s | 100–150 | < 100 |
| LLM Throughput | > 30 tok/s | — | < 30 tok/s |
| LLM Latency avg | < 5000 ms | — | > 5000 ms |

**RTX A2000 12 GB nominal:** 70 W TDP · 288 GB/s bandwidth · 31.2 TFLOPS FP16

---

## LLM Models Compatible with 2× 12 GB

| Model | Size | Notes |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 7 B | Default, ~14 GB split across 2 GPUs |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7 B | ~14 GB |
| `Qwen/Qwen2.5-3B-Instruct` | 3 B | Fits on a single GPU |
| `microsoft/Phi-3.5-mini-instruct` | 3.8 B | Fits on a single GPU |

---

## Dependencies

Managed by [uv](https://github.com/astral-sh/uv). Defined in `pyproject.toml`, locked in `uv.lock`.

| Package | Purpose |
|---|---|
| `torch >= 2.1` (cu121) | GPU compute |
| `transformers >= 4.40` | LLM loading and inference |
| `accelerate >= 0.30` | Multi-GPU dispatch |
| `pynvml >= 11` | Temperature / power monitoring |
| `sentencepiece`, `protobuf` | Tokenizer support |