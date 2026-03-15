# GPU Stress Test Suite
### Pentru: 2x NVIDIA RTX A2000 12GB + Xeon E5-2640 v3 + 128GB RAM

---

## Instalare

```bash
# 1. PyTorch cu suport CUDA (alege versiunea potrivita cu CUDA-ul tau)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Monitor GPU (optional, pentru temperaturi)
pip install pynvml

# 3. Verifica instalarea
python -c "import torch; print(torch.cuda.device_count(), 'GPU-uri')"
```

---

## Scripturi disponibile

| Fisier                  | Descriere                                      |
|-------------------------|------------------------------------------------|
| `gpu_full_stress.py`    | **START AICI** - Suite completa, toate testele |
| `gpu_stress_compute.py` | Compute intens (GEMM FP16, Neural Net)         |
| `gpu_stress_vram.py`    | VRAM fill + bandwidth                          |
| `gpu_monitor.py`        | Monitor timp real (temperatura, util%, VRAM)   |

---

## Utilizare rapida

### Suite completa (recomandat, ~11 min)
```bash
python gpu_full_stress.py
```

### Run rapid de verificare (~5 min)
```bash
python gpu_full_stress.py --quick
```

### Stress prelungit pentru burn-in (30 min per test)
```bash
python gpu_full_stress.py --duration 1800
```

### Test specific
```bash
python gpu_full_stress.py --test compute
python gpu_full_stress.py --test nn
python gpu_full_stress.py --test vram_fill
python gpu_full_stress.py --test bandwidth
python gpu_full_stress.py --test combined
```

---

## Monitorizare in paralel

Deschide **doua terminale**:

**Terminal 1 - stress test:**
```bash
python gpu_full_stress.py --duration 300
```

**Terminal 2 - monitor:**
```bash
# Daca ai pynvml instalat:
python gpu_monitor.py

# Alternativ, direct cu nvidia-smi:
nvidia-smi dmon -s pucvmet -d 1
# sau:
watch -n 1 nvidia-smi
```

---

## Ce sa urmaresti

| Metrica              | Target (OK)     | Atentie         | Problema        |
|----------------------|-----------------|-----------------|-----------------|
| GPU Utilization      | 95-100%         | 80-95%          | < 80%           |
| Temperatura          | < 75°C          | 75-85°C         | > 85°C          |
| VRAM utilizat        | > 10GB / 12GB   | 8-10GB          | < 8GB           |
| TFLOPS FP16          | > 15 TFLOPS     | 10-15           | < 10            |
| Bandwidth VRAM       | > 150 GB/s      | 100-150         | < 100           |

**RTX A2000 12GB specs nominale:**
- TDP: 70W
- VRAM bandwidth teoretic: ~288 GB/s
- FP16 Tensor perf teoretic: ~31.2 TFLOPS

---

## Rezultate asteptate pentru workload normal

Dupa testele de burn-in, GPU-urile tale ar trebui sa:
- Mentina temperaturi stabile sub 80°C in operare continua
- Atinga 95%+ utilizare in teste compute
- Nu aiba erori CUDA sau OOM neasteptate
- Sa nu throttle-ze (clock-urile sa ramana stabile)

---

## Depanare

```bash
# Verifica drivere
nvidia-smi

# Verifica versiunea CUDA
nvcc --version

# Daca primesti CUDA out of memory:
python gpu_full_stress.py --matrix-size 4096  # matrice mai mica

# Daca un GPU nu e detectat:
nvidia-smi -L  # listeaza toate GPU-urile
```