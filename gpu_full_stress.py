import torch
import torch.nn as nn
import threading
import time
import sys
import argparse
import json
from datetime import datetime


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

def snapshot_vram():
    snap = {}
    for i in range(torch.cuda.device_count()):
        snap[i] = torch.cuda.memory_allocated(i) / 1024**3
    return snap

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


def main():
    parser = argparse.ArgumentParser(description="GPU Full Stress Suite")
    parser.add_argument("--duration", type=int, default=120,
                        help="Durata per test in secunde (default: 120)")
    parser.add_argument("--quick", action="store_true",
                        help="Mod rapid: 60s per test")
    parser.add_argument("--test", choices=["compute", "nn", "vram_fill", "bandwidth", "combined", "all"],
                        default="all", help="Care test sa ruleze (default: all)")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Dimensiunea matricei pentru GEMM (default: 8192)")
    args = parser.parse_args()

    check_cuda()

    duration = 60 if args.quick else args.duration
    gpus = gpu_info()

    print(f"\n{'═'*65}")
    print(f"  🖥  GPU Full Stress Suite")
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  GPU-uri: {len(gpus)}")
    for g in gpus:
        print(f"    GPU {g['id']}: {g['name']} | {g['vram_gb']:.1f}GB VRAM | CUDA {g['cuda_capability']}")
    print(f"  Durata per test: {duration}s")
    print(f"{'═'*65}")

    report = {"system": gpus, "timestamp": datetime.now().isoformat(), "tests": {}}

    tests = {
        "compute":    lambda: run_test("Compute FP16 GEMM",      lambda d,t,r: test_compute(d, t, args.matrix_size, r), duration, report["tests"]),
        "nn":         lambda: run_test("Neural Net AMP",         test_nn,         duration, report["tests"]),
        "vram_fill":  lambda: run_test("VRAM Fill (capacitate)", test_vram_fill,  duration, report["tests"]),
        "bandwidth":  lambda: run_test("VRAM Bandwidth",         test_bandwidth,  duration, report["tests"]),
        "combined":   lambda: run_test("Combined (Compute+VRAM)",test_combined,   duration, report["tests"]),
    }

    if args.test == "all":
        for fn in tests.values():
            fn()
    else:
        tests[args.test]()

    print_banner("RAPORT FINAL")
    for name, data in report["tests"].items():
        print(f"\n  [{name}]")
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
                print_result(f"Peak VRAM", f"{res['peak_vram_gb']:.2f}GB / {total_vram:.1f}GB ({pct:.0f}%)", pct > 80)

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