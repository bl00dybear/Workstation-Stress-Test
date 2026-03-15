import torch
import torch.nn as nn
import threading
import time
import sys
import argparse
import json
import gc
from datetime import datetime
from typing import Optional


def check_cuda():
    if not torch.cuda.is_available():
        print("\n❌ CUDA nu este disponibil!")
        print("   Verifica: nvidia-smi, driverele, CUDA toolkit")
        sys.exit(1)

def gpu_info():
    n = torch.cuda.device_count()
    gpus = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gpus.append({
            "id": i,
            "name": p.name,
            "vram_gb": p.total_memory / 1024**3,
            "sm_count": p.multi_processor_count,
            "cuda_capability": f"{p.major}.{p.minor}"
        })
    return gpus

def print_banner(text):
    print(f"\n{'━'*65}")
    print(f"  {text}")
    print(f"{'━'*65}")

def print_result(label, value, ok=True):
    icon = "✓" if ok else "⚠"
    print(f"  {icon}  {label:<35} {value}")

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

    return {i: v["used"] for i, v in snap.items()}

def try_get_temp(device_id):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception:
        return None


def test_compute(device_id, duration, matrix_size, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    size = matrix_size
    A = torch.randn(size, size, dtype=torch.float16, device=device)
    B = torch.randn(size, size, dtype=torch.float16, device=device)

    ops = 0
    start = time.time()
    peak_vram = 0

    while time.time() - start < duration:
        C = torch.mm(A, B)
        C = torch.relu(C)
        C = torch.mm(C, A)
        torch.cuda.synchronize(device_id)
        ops += 1
        v = torch.cuda.memory_allocated(device_id) / 1024**3
        peak_vram = max(peak_vram, v)

    elapsed = time.time() - start
    tflops = (2 * size**3 * ops) / elapsed / 1e12
    results[device_id] = {"ops": ops, "tflops": tflops, "peak_vram_gb": peak_vram, "elapsed": elapsed}
    del A, B
    torch.cuda.empty_cache()


def test_nn(device_id, duration, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    model = nn.Sequential(
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 2048), nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device).half()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    iters = 0
    peak_vram = 0
    start = time.time()

    while time.time() - start < duration:
        x = torch.randn(512, 4096, device=device, dtype=torch.float16)
        target = torch.randn(512, 1024, device=device, dtype=torch.float16)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = nn.functional.mse_loss(out, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        iters += 1
        v = torch.cuda.memory_allocated(device_id) / 1024**3
        peak_vram = max(peak_vram, v)

    elapsed = time.time() - start
    results[device_id] = {"iters": iters, "it_per_sec": iters / elapsed,
                           "peak_vram_gb": peak_vram, "elapsed": elapsed}
    del model
    torch.cuda.empty_cache()


def test_vram_fill(device_id, duration, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device_id).total_memory
    safe = int(total * 0.90)
    chunk = 256 * 1024 * 1024
    tensors = []
    allocated = 0

    while allocated + chunk < safe:
        try:
            t = torch.zeros(chunk // 2, dtype=torch.float16, device=device)
            tensors.append(t)
            allocated += chunk
        except torch.cuda.OutOfMemoryError:
            break

    peak_vram = torch.cuda.memory_allocated(device_id) / 1024**3
    ops = 0
    start = time.time()

    while time.time() - start < duration:
        for t in tensors:
            t.fill_(0.5)
        torch.cuda.synchronize(device_id)
        ops += 1

    elapsed = time.time() - start
    results[device_id] = {"peak_vram_gb": peak_vram, "ops": ops, "elapsed": elapsed,
                           "total_vram_gb": total / 1024**3}
    del tensors
    torch.cuda.empty_cache()


def test_bandwidth(device_id, duration, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device_id).total_memory
    buf_size = min(int(total * 0.40), 4 * 1024**3)
    buf_el = buf_size // 2

    src = torch.ones(buf_el, dtype=torch.float16, device=device)
    dst = torch.zeros(buf_el, dtype=torch.float16, device=device)

    ops = 0
    start = time.time()

    while time.time() - start < duration:
        dst.copy_(src)
        src.copy_(dst)
        src.add_(0.001)
        torch.cuda.synchronize(device_id)
        ops += 1

    elapsed = time.time() - start
    bytes_moved = buf_size * 3 * ops
    bw_gbs = bytes_moved / elapsed / 1e9

    results[device_id] = {"ops": ops, "bandwidth_gbs": bw_gbs,
                           "buf_gb": buf_size / 1024**3, "elapsed": elapsed}
    del src, dst
    torch.cuda.empty_cache()


def test_combined(device_id, duration, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(device_id).total_memory
    fill_size = int(total * 0.55) // 2
    ballast = torch.zeros(fill_size, dtype=torch.float16, device=device)

    sz = 4096
    A = torch.randn(sz, sz, dtype=torch.float16, device=device)
    B = torch.randn(sz, sz, dtype=torch.float16, device=device)

    ops = 0
    peak_vram = 0
    start = time.time()

    while time.time() - start < duration:
        C = torch.mm(A, B)
        C = torch.relu(C)
        ballast.fill_(float(ops % 10) / 10.0)
        torch.cuda.synchronize(device_id)
        ops += 1
        v = torch.cuda.memory_allocated(device_id) / 1024**3
        peak_vram = max(peak_vram, v)

    elapsed = time.time() - start
    results[device_id] = {"ops": ops, "peak_vram_gb": peak_vram, "elapsed": elapsed}
    del ballast, A, B
    torch.cuda.empty_cache()


def run_test(name, fn_per_gpu, duration, report):
    n_gpus = torch.cuda.device_count()
    results = {}
    threads = []

    temps_before = {i: try_get_temp(i) for i in range(n_gpus)}

    print_banner(f"TEST: {name} | {duration}s per GPU")

    for i in range(n_gpus):
        t = threading.Thread(target=fn_per_gpu, args=(i, duration, results))
        threads.append(t)

    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_elapsed = time.time() - t_start

    temps_after = {i: try_get_temp(i) for i in range(n_gpus)}

    print(f"\n  Rezultate ({t_elapsed:.1f}s total):")
    for i, res in results.items():
        print(f"  GPU {i}:", end="")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k}={v:.3f}", end="")
            else:
                print(f"  {k}={v}", end="")
        tb = temps_before.get(i)
        ta = temps_after.get(i)
        if tb and ta:
            print(f"  temp={tb}→{ta}°C", end="")
        print()

    report[name] = {
        "duration_s": t_elapsed,
        "results": results,
        "temps_before": temps_before,
        "temps_after": temps_after
    }


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


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
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


def run_benchmark(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int,
    batch_size: int,
    label: str,
    input_device: str = "cuda:0",
):
    from transformers import GenerationConfig

    def _print_section(text):
        print(f"\n  {'─'*60}")
        print(f"  {text}")
        print(f"  {'─'*60}")

    _print_section(f"Benchmark: {label}")

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
    from transformers import AutoModelForCausalLM

    print_banner("LLM TEST: BASELINE — Single GPU (GPU0)")
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
    from transformers import AutoModelForCausalLM

    print_banner("LLM TEST: PIPELINE PARALLELISM — device_map='auto'")
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
    from transformers import AutoModelForCausalLM
    from accelerate import init_empty_weights, infer_auto_device_map

    print_banner("LLM TEST: TENSOR PARALLELISM — accelerate tensor parallel")
    print("  Fiecare layer Linear din model e impartit pe coloane intre GPU0 si GPU1.")
    print("  Ambele GPU-uri calculeaza simultan parti din fiecare layer.")
    print("  All-reduce intre GPU-uri dupa fiecare layer.\n")

    snapshot_vram("inainte de incarcare")

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
    print_banner("LLM SUMMARY COMPARATIV")
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
    print(f"  • Creste --llm-max-new-tokens pentru a vedea diferente mai mari de throughput")


def run_llm_tests(args, report):
    check_and_install()

    test_choice = args.test

    run_pipeline = test_choice in ("llm_pipeline", "llm_all", "full") or args.llm
    run_tensor   = test_choice in ("llm_tensor",   "llm_all", "full") or args.llm

    if args.llm and test_choice not in ("llm_pipeline", "llm_tensor", "llm_all", "full"):
        run_pipeline = True
        run_tensor   = True

    n = args.llm_n_prompts
    prompts = (TEST_PROMPTS * ((n // len(TEST_PROMPTS)) + 1))[:n]
    print(f"\n  Prompts pregatite: {len(prompts)}")

    print(f"\n  Descarcare tokenizer pentru {args.llm_model} ...")
    tokenizer = load_tokenizer(args.llm_model)

    llm_results = {}

    if run_pipeline:
        n_gpus = torch.cuda.device_count()
        temps_before = {i: try_get_temp(i) for i in range(n_gpus)}
        t0 = time.time()

        result = test_pipeline_parallel(
            args.llm_model, tokenizer, prompts,
            args.llm_max_new_tokens, args.llm_batch_size,
        )

        elapsed = time.time() - t0
        temps_after = {i: try_get_temp(i) for i in range(n_gpus)}

        llm_results["pipeline"] = result
        report["tests"]["llm_pipeline"] = {
            "duration_s":   elapsed,
            "results":      result,
            "temps_before": temps_before,
            "temps_after":  temps_after,
        }

        if run_tensor:
            clear_gpu_memory()

    if run_tensor:
        n_gpus = torch.cuda.device_count()
        temps_before = {i: try_get_temp(i) for i in range(n_gpus)}
        t0 = time.time()

        result = test_tensor_parallel(
            args.llm_model, tokenizer, prompts,
            args.llm_max_new_tokens, args.llm_batch_size,
        )

        elapsed = time.time() - t0
        temps_after = {i: try_get_temp(i) for i in range(n_gpus)}

        llm_results["tensor"] = result
        report["tests"]["llm_tensor"] = {
            "duration_s":   elapsed,
            "results":      result,
            "temps_before": temps_before,
            "temps_after":  temps_after,
        }

    print_summary(llm_results, args.llm_model)


def main():
    parser = argparse.ArgumentParser(description="GPU Full Stress Suite v2")
    parser.add_argument("--duration", type=int, default=120,
                        help="Durata per test in secunde (default: 120)")
    parser.add_argument("--quick", action="store_true",
                        help="Mod rapid: 60s per test")
    parser.add_argument(
        "--test",
        choices=[
            "compute", "nn", "vram_fill", "bandwidth", "combined",
            "all",
            "llm_pipeline", "llm_tensor", "llm_all",
            "full",
        ],
        default="all",
        help=(
            "Care test sa ruleze (default: all). "
            "'all' = original 5 tests. "
            "'full' = original 5 + both LLM tests. "
            "'llm_all' = only LLM tests. "
            "'llm_pipeline' / 'llm_tensor' = only one LLM strategy."
        ),
    )
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Dimensiunea matricei pentru GEMM (default: 8192)")

    parser.add_argument("--llm", action="store_true",
                        help="Enable LLM parallelism tests (pipeline + tensor)")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID for LLM tests (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--llm-n-prompts", type=int, default=20,
                        help="Number of prompts for LLM tests (default: 20)")
    parser.add_argument("--llm-max-new-tokens", type=int, default=128,
                        help="Max new tokens per prompt for LLM tests (default: 128)")
    parser.add_argument("--llm-batch-size", type=int, default=1,
                        help="Batch size for LLM inference (default: 1)")

    args = parser.parse_args()

    check_cuda()

    duration = 60 if args.quick else args.duration
    gpus = gpu_info()

    print(f"\n{'═'*65}")
    print(f"  🖥  GPU Full Stress Suite v2")
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU-uri: {len(gpus)}")
    for g in gpus:
        print(f"    GPU {g['id']}: {g['name']} | {g['vram_gb']:.1f}GB VRAM | CUDA {g['cuda_capability']}")
    print(f"  Durata per test (non-LLM): {duration}s")
    print(f"{'═'*65}")

    report = {"system": gpus, "timestamp": datetime.now().isoformat(), "tests": {}}

    _llm_only = args.test in ("llm_pipeline", "llm_tensor", "llm_all")

    if not _llm_only:
        tests = {
            "compute":   lambda: run_test("Compute FP16 GEMM",      lambda d, t, r: test_compute(d, t, args.matrix_size, r), duration, report["tests"]),
            "nn":        lambda: run_test("Neural Net AMP",         test_nn,         duration, report["tests"]),
            "vram_fill": lambda: run_test("VRAM Fill (capacitate)", test_vram_fill,  duration, report["tests"]),
            "bandwidth": lambda: run_test("VRAM Bandwidth",         test_bandwidth,  duration, report["tests"]),
            "combined":  lambda: run_test("Combined (Compute+VRAM)",test_combined,   duration, report["tests"]),
        }

        if args.test in ("all", "full"):
            for fn in tests.values():
                fn()
        elif args.test in tests:
            tests[args.test]()

    _run_llm = (
        args.llm
        or args.test in ("llm_pipeline", "llm_tensor", "llm_all", "full")
    )
    if _run_llm:
        run_llm_tests(args, report)

    print_banner("RAPORT FINAL")
    for name, data in report["tests"].items():
        print(f"\n  [{name}]")

        if name in ("llm_pipeline", "llm_tensor"):
            res = data["results"]
            if res is None:
                print("    N/A (test skipped)")
                continue

            tps  = res.get("throughput_tok_s", 0)
            lat  = res.get("latency_avg_ms", 0)
            vram = res.get("vram_gb", {})

            print_result("LLM Throughput (tok/s)", f"{tps:.1f}", tps > 30)
            print_result("LLM Latency avg (ms)",   f"{lat:.0f}", lat < 5000)
            for gpu_id_str, gb in vram.items():
                print_result(f"VRAM GPU{gpu_id_str} used",
                             f"{gb:.2f} GB",
                             gb > 5.0)
        else:
            for gpu_id, res in data["results"].items():
                print(f"    GPU {gpu_id}:", end="")
                if "tflops" in res:
                    ok = res["tflops"] > 10
                    print_result(f"TFLOPS (FP16)", f"{res['tflops']:.2f}", ok)
                if "it_per_sec" in res:
                    print_result(f"Iteratii/sec (NN)", f"{res['it_per_sec']:.1f}", True)
                if "bandwidth_gbs" in res:
                    ok = res["bandwidth_gbs"] > 100
                    print_result(f"Bandwidth VRAM", f"{res['bandwidth_gbs']:.1f} GB/s", ok)
                if "peak_vram_gb" in res:
                    total_vram = gpus[gpu_id]["vram_gb"]
                    pct = (res["peak_vram_gb"] / total_vram) * 100
                    print_result(f"Peak VRAM",
                                 f"{res['peak_vram_gb']:.2f}GB / {total_vram:.1f}GB ({pct:.0f}%)",
                                 pct > 80)

    report_file = f"gpu_stress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)
        json.dump(convert(report), f, indent=2)

    print(f"\n  📄 Raport salvat: {report_file}")
    print_banner("✅ Stress suite complet!")


if __name__ == "__main__":
    main()
