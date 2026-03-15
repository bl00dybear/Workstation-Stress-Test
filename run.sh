#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">>> uv $(uv --version)"

uv sync --no-install-project

uv run python -c "
import torch, sys
n = torch.cuda.device_count()
if n == 0:
    print('ERROR: No CUDA GPUs found. Check drivers / nvidia-smi.')
    sys.exit(1)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} | {p.total_memory/1024**3:.1f} GB VRAM | CUDA {p.major}.{p.minor}')
"

TEST="${1:-all}"
DURATION="${2:-120}"
QUICK="${3:-}"

QUICK_FLAG=""
if [[ "$QUICK" == "--quick" ]]; then
    QUICK_FLAG="--quick"
fi

uv run python gpu_full_stress_v2.py \
    --test "$TEST" \
    --duration "$DURATION" \
    $QUICK_FLAG \
    "${@:4}"
