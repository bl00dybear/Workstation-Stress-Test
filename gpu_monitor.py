import time
import sys
import argparse


def check_dependencies():
    try:
        import pynvml
        return True
    except ImportError:
        print("pynvml nu este instalat. Instaleaza cu: pip install pynvml")
        print("Alternativ, ruleaza: nvidia-smi dmon -s pucvmet -d 1")
        return False


def monitor_gpus(interval=1.0, duration=None):
    try:
        import pynvml
    except ImportError:
        print("EROARE: pip install pynvml")
        sys.exit(1)

    pynvml.nvmlInit()
    n_gpus = pynvml.nvmlDeviceGetCount()

    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]
    names = [pynvml.nvmlDeviceGetName(h) for h in handles]
    names = [n.decode() if isinstance(n, bytes) else n for n in names]

    print(f"\n{'='*100}")
    print(f"  GPU Monitor | {n_gpus} GPU-uri detectate | Interval: {interval}s")
    print(f"{'='*100}")

    header = f"{'Timp':>8} | "
    for i in range(n_gpus):
        short_name = names[i].replace("NVIDIA ", "").replace("RTX ", "RTX")[:20]
        header += f"{'GPU'+str(i)+' '+short_name:^50} | "
    print(header)

    sub_header = f"{'':>8} | "
    for i in range(n_gpus):
        sub_header += f"{'Util%':>5} {'Temp°C':>6} {'VRAM':>12} {'Power W':>8} {'Clk MHz':>8} {'MemClk':>7} | "
    print(sub_header)
    print("-" * 100)

    start = time.time()
    sample = 0

    stats = {i: {"max_temp": 0, "max_util": 0, "max_power": 0, "max_vram": 0} for i in range(n_gpus)}

    try:
        while True:
            elapsed = time.time() - start
            if duration and elapsed > duration:
                break

            line = f"{elapsed:>7.1f}s | "

            for i, h in enumerate(handles):
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    clocks = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
                    mem_clocks = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)

                    mem_used_gb = mem.used / 1024**3
                    mem_total_gb = mem.total / 1024**3

                    stats[i]["max_temp"] = max(stats[i]["max_temp"], temp)
                    stats[i]["max_util"] = max(stats[i]["max_util"], util.gpu)
                    stats[i]["max_power"] = max(stats[i]["max_power"], power)
                    stats[i]["max_vram"] = max(stats[i]["max_vram"], mem_used_gb)

                    util_str = f"{util.gpu:3d}%"
                    if util.gpu >= 95:
                        util_str = f"[{util.gpu:3d}%]"
                    elif util.gpu >= 80:
                        util_str = f" {util.gpu:3d}% "

                    temp_str = f"{temp:3d}°C"
                    if temp >= 85:
                        temp_str = f"!{temp:3d}!"
                    elif temp >= 75:
                        temp_str = f"*{temp:3d}*"

                    vram_str = f"{mem_used_gb:.2f}/{mem_total_gb:.1f}G"

                    line += (f"{util_str:>5} {temp_str:>6} {vram_str:>12} "
                             f"{power:>7.1f}W {clocks:>7}M {mem_clocks:>6}M | ")

                except pynvml.NVMLError as e:
                    line += f"{'EROARE: ' + str(e):^50} | "

            print(line)
            sample += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        pass

    finally:
        elapsed_total = time.time() - start
        print(f"\n{'='*100}")
        print(f"  STATISTICI ({sample} samples in {elapsed_total:.1f}s)")
        print(f"{'='*100}")
        for i in range(n_gpus):
            s = stats[i]
            print(f"  GPU {i} ({names[i]}):")
            print(f"    Max Temp:    {s['max_temp']}°C {'⚠ PERICOL!' if s['max_temp'] >= 90 else '✓ OK' if s['max_temp'] < 80 else '! Atentie'}")
            print(f"    Max Util:    {s['max_util']}%  {'✓ Maxim atins!' if s['max_util'] >= 95 else '⚠ Sub maxim'}")
            print(f"    Max Power:   {s['max_power']:.1f}W")
            print(f"    Max VRAM:    {s['max_vram']:.2f}GB")
        print(f"{'='*100}\n")

        pynvml.nvmlShutdown()


def print_nvidia_smi_command():
    print("\nDaca nu ai pynvml, foloseste direct in terminal:")
    print("  nvidia-smi dmon -s pucvmet -d 1")
    print("  sau:")
    print("  watch -n 1 nvidia-smi")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Monitor in timp real")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Interval intre samples in secunde (default: 1.0)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Durata monitorizare in secunde (default: infinit, Ctrl+C pentru stop)")
    args = parser.parse_args()

    if not check_dependencies():
        print_nvidia_smi_command()
        print("Instaleaza cu: pip install pynvml")
        print("Apoi ruleaza din nou scriptul.")
        sys.exit(1)

    print("Pornire monitor GPU... (Ctrl+C pentru a opri)")
    monitor_gpus(args.interval, args.duration)