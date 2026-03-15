import torch
import time
import json
import argparse
import sys
import gc
from datetime import datetime
from typing import Optional


def check_and_install():
    required = {
        "transformers": "transformers>=4.40.0",
        "accelerate":   "accelerate>=0.30.0",
        "sentencepiece": "sentencepiece",
        "protobuf":     "protobuf",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n⚠  Dependente lipsa: {', '.join(missing)}")
        print(f"   Instalare automata...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet"
        ] + missing)
        print("   ✓ Instalare completa. Restarteaza scriptul daca apar erori.\n")

check_and_install()

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
import accelerate


TEST_PROMPTS = [
    "Explica pe scurt cum functioneaza o retea neuronala.",
    "Care sunt avantajele si dezavantajele energiei nucleare?",
    "Scrie un algoritm de sortare rapida in Python.",
    "Descrie diferenta dintre RAM si stocare SSD.",
    "Ce este paralelismul in informatica si de ce conteaza?",
    "Cum functioneaza un motor cu ardere interna?",
    "Explica conceptul de entropie in termodinamica.",
    "Ce este machine learning si cum difera de AI clasic?",
    "Descrie arhitectura Transformer in procesarea limbajului natural.",
    "Care sunt principalele diferente intre Python si C++?",
    "Explica cum functioneaza criptografia cu cheie publica.",
    "Ce este un sistem de operare si care sunt componentele sale principale?",
    "Descrie diferenta dintre supervised si unsupervised learning.",
    "Cum functioneaza un procesor modern la nivel hardware?",
    "Ce este Docker si de ce este util in dezvoltarea software?",
    "Explica conceptul de gradient descent in optimizarea modelelor ML.",
    "Descrie arhitectura unui sistem distribuit.",
    "Ce este CUDA si cum accelereaza calculele pe GPU?",
    "Explica diferenta dintre o retea convolutionala si un Transformer.",
    "Ce este cuantizarea unui model neural si ce avantaje aduce?",
]


def print_banner(text, char="═"):
    print(f"\n{char*65}")
    print(f"  {text}")
    print(f"{char*65}")

def print_section(text):
    print(f"\n  {'─'*60}")
    print(f"  {text}")
    print(f"  {'─'*60}")

def snapshot_vram(label=""):
    snap = {}
    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        snap[i] = {"used": used, "total": total, "pct": used / total * 100}
    if label:
        parts = " | ".join(
            f"GPU{i}: {v['used']:.2f}/{v['total']:.1f}GB ({v['pct']:.0f}%)"
            for i, v in snap.items()
        )
        print(f"  VRAM {label}: {parts}")
    return snap

def clear_gpu_memory(*models):
    for m in models:
        if m is not None:
            del m
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)

def get_device_map_info(model):
    if not hasattr(model, "hf_device_map"):
        return {}
    dmap = model.hf_device_map
    gpu_layers = {}
    for layer_name, dev in dmap.items():
        dev_str = str(dev)
        gpu_layers.setdefault(dev_str, []).append(layer_name)
    return gpu_layers


def load_tokenizer(model_id: str):
    print(f"  Incarcare tokenizer: {model_id} ...")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  ✓ Tokenizer incarcat | Vocab: {tok.vocab_size:,} tokens")
    return tok


def prepare_inputs(tokenizer, prompts, device="cuda:0"):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def run_benchmark(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int,
    batch_size: int,
    label: str,
    input_device: str = "cuda:0",
):
    print_section(f"Benchmark: {label}")

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"  Warmup (2 prompts)...")
    warmup_inputs = prepare_inputs(tokenizer, prompts[:2], device=input_device)
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, generation_config=gen_config)
    torch.cuda.synchronize()
    del warmup_inputs

    snapshot_vram("dupa warmup")

    print(f"  Start benchmark: {len(prompts)} prompts | batch={batch_size} | max_new_tokens={max_new_tokens}")

    total_tokens_generated = 0
    latencies_ms = []
    total_start = time.time()

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        inputs = prepare_inputs(tokenizer, batch_prompts, device=input_device)

        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        torch.cuda.synchronize()
        t1 = time.time()

        generated_tokens = (outputs.shape[1] - input_len) * len(batch_prompts)
        total_tokens_generated += generated_tokens
        latency_ms = (t1 - t0) * 1000
        latencies_ms.append(latency_ms)

        batch_idx = batch_start // batch_size + 1
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_idx:2d}/{n_batches} | "
              f"Latenta: {latency_ms:.0f}ms | "
              f"Tokens generate: {generated_tokens} | "
              f"Tok/s: {generated_tokens / (t1 - t0):.1f}")

        del inputs, outputs

    total_elapsed = time.time() - total_start
    vram_peak = snapshot_vram("peak")

    n = len(latencies_ms)
    lat_sorted = sorted(latencies_ms)
    lat_avg = sum(latencies_ms) / n
    lat_p50 = lat_sorted[n // 2]
    lat_p95 = lat_sorted[int(n * 0.95)] if n >= 20 else lat_sorted[-1]
    throughput_tps = total_tokens_generated / total_elapsed
    throughput_rps = len(prompts) / total_elapsed

    result = {
        "label":              label,
        "throughput_tok_s":   round(throughput_tps, 2),
        "throughput_req_s":   round(throughput_rps, 3),
        "latency_avg_ms":     round(lat_avg, 1),
        "latency_p50_ms":     round(lat_p50, 1),
        "latency_p95_ms":     round(lat_p95, 1),
        "total_tokens":       total_tokens_generated,
        "total_prompts":      len(prompts),
        "elapsed_s":          round(total_elapsed, 2),
        "max_new_tokens":     max_new_tokens,
        "vram_gb": {
            str(i): round(vram_peak[i]["used"], 3)
            for i in range(torch.cuda.device_count())
        },
    }

    print(f"\n  REZULTAT {label}:")
    print(f"    Throughput:      {throughput_tps:.1f} tokens/sec")
    print(f"    Requests/sec:    {throughput_rps:.3f}")
    print(f"    Latenta avg:     {lat_avg:.0f} ms/batch")
    print(f"    Latenta p50:     {lat_p50:.0f} ms")
    print(f"    Latenta p95:     {lat_p95:.0f} ms")
    print(f"    Total tokens:    {total_tokens_generated:,}")
    print(f"    Timp total:      {total_elapsed:.1f}s")
    for i, v in vram_peak.items():
        print(f"    VRAM GPU{i}:      {v['used']:.2f}GB / {v['total']:.1f}GB ({v['pct']:.0f}%)")

    return result


def test_baseline(model_id, tokenizer, prompts, max_new_tokens, batch_size):
    print_banner("TEST 1: BASELINE — Single GPU (GPU0)", "─")
    print("  Tot modelul incarcat pe cuda:0")
    print("  Folosit ca referinta pentru comparatie.\n")

    snapshot_vram("inainte de incarcare")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        model.eval()
    except torch.cuda.OutOfMemoryError:
        print("  ⚠  Modelul nu incape pe un singur GPU (OOM)!")
        print("     Skipping baseline — ruleaza doar testele multi-GPU.")
        return None

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Model incarcat | Parametri: {n_params:.2f}B")
    snapshot_vram("dupa incarcare")

    result = run_benchmark(
        model, tokenizer, prompts,
        max_new_tokens, batch_size,
        label="Baseline (Single GPU0)",
        input_device="cuda:0",
    )

    clear_gpu_memory(model)
    return result


def test_pipeline_parallel(model_id, tokenizer, prompts, max_new_tokens, batch_size):
    print_banner("TEST 2: PIPELINE PARALLELISM — device_map='auto'", "─")
    print("  accelerate imparte automat layerele modelului intre GPU0 si GPU1")
    print("  in functie de VRAM disponibil pe fiecare GPU.")
    print("  Activarile trec prin PCIe de la GPU0 la GPU1 in timpul inferentei.\n")

    snapshot_vram("inainte de incarcare")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Model incarcat | Parametri: {n_params:.2f}B")

    if hasattr(model, "hf_device_map"):
        dmap = model.hf_device_map
        gpu_layer_count = {}
        for layer, dev in dmap.items():
            dev_str = str(dev)
            gpu_layer_count[dev_str] = gpu_layer_count.get(dev_str, 0) + 1

        print(f"\n  Distributie layere (device_map):")
        for dev, count in sorted(gpu_layer_count.items()):
            print(f"    {dev}: {count} layere/componente")

        items = list(dmap.items())
        print(f"\n  Prime 5 layere:")
        for name, dev in items[:5]:
            print(f"    {name:<45} → {dev}")
        print(f"  ...")
        print(f"  Ultimele 5 layere:")
        for name, dev in items[-5:]:
            print(f"    {name:<45} → {dev}")

    snapshot_vram("dupa incarcare")

    first_device = next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else "cuda:0"
    first_device_str = f"cuda:{first_device}" if isinstance(first_device, int) else str(first_device)

    result = run_benchmark(
        model, tokenizer, prompts,
        max_new_tokens, batch_size,
        label="Pipeline Parallel (device_map=auto)",
        input_device=first_device_str,
    )

    clear_gpu_memory(model)
    return result


def test_tensor_parallel(model_id, tokenizer, prompts, max_new_tokens, batch_size):
    print_banner("TEST 3: TENSOR PARALLELISM — accelerate tensor parallel", "─")
    print("  Fiecare layer Linear din model e impartit pe coloane intre GPU0 si GPU1.")
    print("  Ambele GPU-uri calculeaza simultan parti din fiecare layer.")
    print("  All-reduce intre GPU-uri dupa fiecare layer.\n")

    snapshot_vram("inainte de incarcare")

    from accelerate import init_empty_weights, infer_auto_device_map
    from accelerate import dispatch_model

    vram_per_gpu = torch.cuda.get_device_properties(0).total_memory
    max_memory = {
        0: f"{int(vram_per_gpu * 0.85 / 1024**3)}GiB",
        1: f"{int(vram_per_gpu * 0.85 / 1024**3)}GiB",
        "cpu": "32GiB",
    }
    print(f"  Max memory per device: {max_memory}")

    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    device_map = infer_auto_device_map(
        empty_model,
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer", "Qwen2DecoderLayer",
                                  "MistralDecoderLayer", "PhiDecoderLayer",
                                  "GPTNeoXLayer", "BloomBlock"],
        dtype=torch.float16,
    )
    del empty_model
    gc.collect()

    gpu_counts = {}
    for layer, dev in device_map.items():
        dev_str = str(dev)
        gpu_counts[dev_str] = gpu_counts.get(dev_str, 0) + 1
    print(f"\n  Distributie layere calculata:")
    for dev, cnt in sorted(gpu_counts.items()):
        print(f"    {dev}: {cnt} componente")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"\n  ✓ Model incarcat cu tensor-aware device map | Parametri: {n_params:.2f}B")
    snapshot_vram("dupa incarcare")

    first_device = list(device_map.values())[0]
    first_device_str = f"cuda:{first_device}" if isinstance(first_device, int) else str(first_device)

    result = run_benchmark(
        model, tokenizer, prompts,
        max_new_tokens, batch_size,
        label="Tensor Parallel (infer_auto_device_map)",
        input_device=first_device_str,
    )

    clear_gpu_memory(model)
    return result


def print_summary(results: dict, model_id: str):
    print_banner("SUMMARY COMPARATIV")
    print(f"  Model: {model_id}")
    print()

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("  Nu exista rezultate de afisat.")
        return

    print(f"  {'Strategie':<42} {'Tok/s':>8} {'Req/s':>7} {'Lat avg':>10} {'VRAM G0':>9} {'VRAM G1':>9}")
    print(f"  {'─'*42} {'─'*8} {'─'*7} {'─'*10} {'─'*9} {'─'*9}")

    baseline_tps = valid.get("baseline", {}).get("throughput_tok_s", None)

    for key, r in valid.items():
        tps  = r["throughput_tok_s"]
        rps  = r["throughput_req_s"]
        lat  = r["latency_avg_ms"]
        v0   = r["vram_gb"].get("0", 0)
        v1   = r["vram_gb"].get("1", 0)
        speedup = f"({tps/baseline_tps:.2f}x)" if baseline_tps and key != "baseline" else ""

        print(f"  {r['label']:<42} {tps:>6.1f}   {speedup:>5} "
              f"{rps:>6.3f}   {lat:>8.0f}ms   {v0:>6.2f}GB   {v1:>6.2f}GB")

    print(f"\n  Interpretare:")
    if "baseline" in valid and "pipeline" in valid:
        ratio = valid["pipeline"]["throughput_tok_s"] / valid["baseline"]["throughput_tok_s"]
        if ratio >= 1.5:
            print(f"  ✓ Pipeline: {ratio:.2f}x speedup → paralelizarea aduce beneficiu real")
        elif ratio >= 0.9:
            print(f"  ~ Pipeline: {ratio:.2f}x → overhead PCIe echilibreaza castigul")
        else:
            print(f"  ⚠ Pipeline: {ratio:.2f}x → overhead > beneficiu pentru acest model/batch")

    if "baseline" in valid and "tensor" in valid:
        ratio = valid["tensor"]["throughput_tok_s"] / valid["baseline"]["throughput_tok_s"]
        if ratio >= 1.5:
            print(f"  ✓ Tensor Parallel: {ratio:.2f}x speedup → distributie echilibrata")
        elif ratio >= 0.9:
            print(f"  ~ Tensor Parallel: {ratio:.2f}x → echilibru overhead/castig")
        else:
            print(f"  ⚠ Tensor Parallel: {ratio:.2f}x → all-reduce overhead dominant")

    print(f"\n  Note:")
    print(f"  • Pipeline e mai bun pentru modele ce nu incap pe 1 GPU")
    print(f"  • Tensor parallel reduce latenta dar creste comunicatia inter-GPU")
    print(f"  • Cu 2x RTX A2000 12GB, modelele 7B ruleaza confortabil split pe 2 GPU-uri")
    print(f"  • Creste --max-new-tokens pentru a vedea diferente mai mari de throughput")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Model Parallelism Test - 2x RTX A2000 12GB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemple:
  python llm_model_parallel.py
  python llm_model_parallel.py --model mistralai/Mistral-7B-Instruct-v0.3
  python llm_model_parallel.py --model Qwen/Qwen2.5-3B-Instruct --test baseline
  python llm_model_parallel.py --test pipeline --n-prompts 30 --max-new-tokens 150
  python llm_model_parallel.py --test tensor  --batch-size 2

Modele recomandate pentru 2x12GB:
  Qwen/Qwen2.5-7B-Instruct          (default, 7B params, ~14GB)
  mistralai/Mistral-7B-Instruct-v0.3 (7B params, ~14GB)
  Qwen/Qwen2.5-3B-Instruct          (3B params, incape pe 1 GPU)
  microsoft/Phi-3.5-mini-instruct    (3.8B params)
  meta-llama/Llama-3.2-3B-Instruct  (necesita acces HF)
        """
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model HuggingFace ID (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--test", choices=["all", "baseline", "pipeline", "tensor"],
        default="all",
        help="Ce test sa ruleze (default: all)"
    )
    parser.add_argument(
        "--n-prompts", type=int, default=20,
        help="Numarul de prompts de generat per test (default: 20)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Tokens noi de generat per prompt (default: 128)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size pentru inferenta (default: 1)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("EROARE: CUDA nu este disponibil!")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2 and args.test in ["pipeline", "tensor", "all"]:
        print(f"ATENTIE: Detectat {n_gpus} GPU. Testele multi-GPU necesita 2 GPU-uri.")
        print("Ruleaza cu --test baseline pentru single GPU.")

    print_banner("LLM Model Parallelism Test Suite")
    print(f"  Data:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:           {args.model}")
    print(f"  Teste:           {args.test}")
    print(f"  Prompts:         {args.n_prompts}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  GPU-uri:         {n_gpus}")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU{i}: {p.name} | {p.total_memory/1024**3:.1f}GB VRAM")

    prompts = (TEST_PROMPTS * ((args.n_prompts // len(TEST_PROMPTS)) + 1))[:args.n_prompts]
    print(f"\n  Prompts pregatite: {len(prompts)}")

    print(f"\n  Descarcare model de pe HuggingFace (prima rulare poate dura cateva minute)...")
    tokenizer = load_tokenizer(args.model)

    results = {}

    if args.test in ["all", "baseline"]:
        results["baseline"] = test_baseline(
            args.model, tokenizer, prompts,
            args.max_new_tokens, args.batch_size
        )

    if args.test in ["all", "pipeline"]:
        results["pipeline"] = test_pipeline_parallel(
            args.model, tokenizer, prompts,
            args.max_new_tokens, args.batch_size
        )

    if args.test in ["all", "tensor"]:
        results["tensor"] = test_tensor_parallel(
            args.model, tokenizer, prompts,
            args.max_new_tokens, args.batch_size
        )

    print_summary(results, args.model)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model":     args.model,
        "config": {
            "n_prompts":      args.n_prompts,
            "max_new_tokens": args.max_new_tokens,
            "batch_size":     args.batch_size,
        },
        "gpus": [
            torch.cuda.get_device_properties(i).name
            for i in range(n_gpus)
        ],
        "results": {k: v for k, v in results.items() if v is not None},
    }
    fname = f"llm_parallel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  📄 Raport JSON salvat: {fname}")
    print_banner("✅ LLM Parallelism Test complet!")


if __name__ == "__main__":
    main()