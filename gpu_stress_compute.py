import torch
import threading
import time
import sys
import argparse


def get_gpu_info():
    if not torch.cuda.is_available():
        print("EROARE: CUDA nu este disponibil! Verifica driverele NVIDIA.")
        sys.exit(1)
    n = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  GPU-uri detectate: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} | VRAM: {mem:.1f} GB | SM: {props.multi_processor_count}")
    print(f"{'='*60}\n")
    return n


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


def print_banner(text):
    print(f"\n{'━'*65}")
    print(f"  {text}")
    print(f"{'━'*65}")


def print_result(label, value, ok=True):
    icon = "✓" if ok else "⚠"
    print(f"  {icon}  {label:<35} {value}")


def test_compute(device_id, duration_sec, matrix_size, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

    print(f"  [GPU {device_id}] Start compute stress | matrice {matrix_size}x{matrix_size} | FP16")

    ops_count = 0
    start = time.time()
    elapsed = 0

    try:
        while elapsed < duration_sec:
            C = torch.mm(A, B)
            C = torch.relu(C)
            C = torch.mm(C, A)
            C = torch.sigmoid(C)
            torch.cuda.synchronize(device_id)
            ops_count += 4
            elapsed = time.time() - start

            if ops_count % 100 == 0:
                mem_used = torch.cuda.memory_allocated(device_id) / 1024**3
                mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                tflops = (2 * matrix_size**3 * ops_count) / elapsed / 1e12
                print(f"  [GPU {device_id}] {elapsed:6.1f}s | Ops: {ops_count:5d} | "
                      f"VRAM: {mem_used:.2f}/{mem_total:.1f}GB | ~{tflops:.2f} TFLOPS")

    except Exception as e:
        print(f"  [GPU {device_id}] EROARE: {e}")

    results[device_id] = ops_count
    print(f"  [GPU {device_id}] ✓ Compute stress finalizat | {ops_count} ops in {elapsed:.1f}s")


def stress_mixed_precision(device_id, duration_sec, results):
    import torch.nn as nn
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device).half()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    print(f"  [GPU {device_id}] Start neural net stress | batch 512 | FP16 AMP")

    batch_size = 512
    iters = 0
    start = time.time()
    elapsed = 0

    try:
        while elapsed < duration_sec:
            x = torch.randn(batch_size, 4096, device=device, dtype=torch.float16)
            target = torch.randn(batch_size, 1024, device=device, dtype=torch.float16)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = nn.functional.mse_loss(out, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iters += 1
            elapsed = time.time() - start

            if iters % 50 == 0:
                mem_used = torch.cuda.memory_allocated(device_id) / 1024**3
                mem_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                it_per_sec = iters / elapsed
                print(f"  [GPU {device_id}] {elapsed:6.1f}s | Iter: {iters:5d} | "
                      f"Loss: {loss.item():.4f} | VRAM: {mem_used:.2f}/{mem_total:.1f}GB | "
                      f"{it_per_sec:.1f} it/s")

    except Exception as e:
        print(f"  [GPU {device_id}] EROARE: {e}")

    results[device_id] = iters
    print(f"  [GPU {device_id}] ✓ Neural net stress finalizat | {iters} iteratii in {elapsed:.1f}s")


def run_stress(test_type="compute", duration=120, matrix_size=8192):
    n_gpus = get_gpu_info()

    print(f"Test: {test_type.upper()} | Durata: {duration}s | GPU-uri: {n_gpus}")
    print(f"{'='*60}")

    results = {}
    threads = []

    if test_type == "compute":
        for i in range(n_gpus):
            t = threading.Thread(target=test_compute, args=(i, duration, matrix_size, results))
            threads.append(t)
    elif test_type == "nn":
        for i in range(n_gpus):
            t = threading.Thread(target=stress_mixed_precision, args=(i, duration, results))
            threads.append(t)

    start_all = time.time()
    for t in threads:
        t.start()

    while any(t.is_alive() for t in threads):
        time.sleep(10)
        elapsed = time.time() - start_all
        for i in range(n_gpus):
            try:
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                util_info = f"VRAM rezervata: {mem_reserved:.2f}GB"
                print(f"  >> GPU {i} | {elapsed:.0f}s | {util_info}")
            except:
                pass

    for t in threads:
        t.join()

    total_time = time.time() - start_all
    print(f"\n{'='*60}")
    print(f"  ✓ Test finalizat in {total_time:.1f}s")
    for gpu_id, count in results.items():
        print(f"  GPU {gpu_id}: {count} operatii/iteratii")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Compute Stress Test")
    parser.add_argument("--test", choices=["compute", "nn"], default="compute",
                        help="Tipul testului: compute (GEMM) sau nn (neural network)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Durata in secunde (default: 120)")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Dimensiunea matricei pentru test compute (default: 8192)")
    args = parser.parse_args()

    run_stress(args.test, args.duration, args.matrix_size)