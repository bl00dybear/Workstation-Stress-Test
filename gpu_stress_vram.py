import torch
import threading
import time
import sys


def get_safe_vram_limit(device_id, fraction=0.92):
    total = torch.cuda.get_device_properties(device_id).total_memory
    return int(total * fraction)


def stress_vram_fill(device_id, duration_sec, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    safe_limit = get_safe_vram_limit(device_id, fraction=0.90)
    total_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

    print(f"  [GPU {device_id}] VRAM total: {total_gb:.1f}GB | Tinta fill: {safe_limit/1024**3:.2f}GB")

    tensors = []
    allocated = 0
    chunk_bytes = 512 * 1024 * 1024
    chunk_elements = chunk_bytes // 2

    print(f"  [GPU {device_id}] Alocare VRAM...")
    while allocated + chunk_bytes < safe_limit:
        try:
            t = torch.zeros(chunk_elements, dtype=torch.float16, device=device)
            tensors.append(t)
            allocated += chunk_bytes
        except torch.cuda.OutOfMemoryError:
            print(f"  [GPU {device_id}] OOM la {allocated/1024**3:.2f}GB - oprim alocarea")
            break

    used_gb = torch.cuda.memory_allocated(device_id) / 1024**3
    print(f"  [GPU {device_id}] ✓ VRAM alocat: {used_gb:.2f}GB / {total_gb:.1f}GB")

    ops = 0
    start = time.time()
    elapsed = 0

    print(f"  [GPU {device_id}] Start bandwidth stress pe {len(tensors)} chunksuri...")

    try:
        while elapsed < duration_sec:
            for i, t in enumerate(tensors):
                t.fill_(float(ops % 100) / 100.0)
                _ = t.sum()
                if i + 1 < len(tensors):
                    tensors[i + 1].copy_(t)

            torch.cuda.synchronize(device_id)
            ops += 1
            elapsed = time.time() - start

            bytes_per_op = allocated * 3
            bandwidth_gbs = (bytes_per_op * ops) / elapsed / 1e9

            if ops % 10 == 0:
                print(f"  [GPU {device_id}] {elapsed:6.1f}s | Ops: {ops:4d} | "
                      f"VRAM: {used_gb:.2f}GB | Bandwidth: ~{bandwidth_gbs:.1f} GB/s")

    except Exception as e:
        print(f"  [GPU {device_id}] EROARE: {e}")
    finally:
        del tensors
        torch.cuda.empty_cache()

    results[device_id] = {"ops": ops, "vram_used_gb": used_gb}
    print(f"  [GPU {device_id}] ✓ VRAM stress finalizat | {ops} ops | VRAM eliberat")


def stress_vram_bandwidth_only(device_id, duration_sec, results):
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    total = torch.cuda.get_device_properties(device_id).total_memory
    buf_size = min(int(total * 0.40), 4 * 1024**3)
    buf_elements = buf_size // 2

    print(f"  [GPU {device_id}] Bandwidth test | 2x buffer {buf_size/1024**3:.1f}GB")

    try:
        src = torch.ones(buf_elements, dtype=torch.float16, device=device)
        dst = torch.zeros(buf_elements, dtype=torch.float16, device=device)
    except torch.cuda.OutOfMemoryError:
        print(f"  [GPU {device_id}] OOM - reducem buffer la 2GB")
        buf_elements = (2 * 1024**3) // 2
        src = torch.ones(buf_elements, dtype=torch.float16, device=device)
        dst = torch.zeros(buf_elements, dtype=torch.float16, device=device)

    buf_gb = (buf_elements * 2) / 1024**3
    ops = 0
    start = time.time()
    elapsed = 0

    try:
        while elapsed < duration_sec:
            dst.copy_(src)
            src.copy_(dst)
            src.add_(0.001)
            dst.mul_(0.9999)
            torch.cuda.synchronize(device_id)
            ops += 1
            elapsed = time.time() - start

            if ops % 20 == 0:
                bytes_moved = buf_gb * 1024**3 * 4 * ops
                bw = bytes_moved / elapsed / 1e9
                mem_used = torch.cuda.memory_allocated(device_id) / 1024**3
                print(f"  [GPU {device_id}] {elapsed:6.1f}s | Ops: {ops:5d} | "
                      f"VRAM: {mem_used:.2f}GB | BW: {bw:.1f} GB/s")

    except Exception as e:
        print(f"  [GPU {device_id}] EROARE: {e}")
    finally:
        del src, dst
        torch.cuda.empty_cache()

    results[device_id] = ops
    print(f"  [GPU {device_id}] ✓ Bandwidth test finalizat | {ops} iteratii")


def run_vram_stress(mode="fill", duration=120):
    if not torch.cuda.is_available():
        print("EROARE: CUDA nu este disponibil!")
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  VRAM Stress Test | Mode: {mode} | Durata: {duration}s | GPU-uri: {n_gpus}")
    print(f"{'='*60}")

    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory/1024**3:.1f}GB")
    print()

    results = {}
    threads = []

    for i in range(n_gpus):
        if mode == "fill":
            t = threading.Thread(target=stress_vram_fill, args=(i, duration, results))
        else:
            t = threading.Thread(target=stress_vram_bandwidth_only, args=(i, duration, results))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(f"\n{'='*60}")
    print("  REZULTATE FINALE:")
    for gpu_id, res in results.items():
        print(f"  GPU {gpu_id}: {res}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU VRAM Stress Test")
    parser.add_argument("--mode", choices=["fill", "bandwidth"], default="fill",
                        help="fill: umple VRAM-ul | bandwidth: test pur de bandwidth")
    parser.add_argument("--duration", type=int, default=120,
                        help="Durata in secunde (default: 120)")
    args = parser.parse_args()

    run_vram_stress(args.mode, args.duration)