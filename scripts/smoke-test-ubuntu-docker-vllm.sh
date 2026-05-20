#!/usr/bin/env bash
set -euo pipefail

echo "== Device nodes =="
test -e /dev/kfd
test -d /dev/dri
ls -l /dev/kfd /dev/dri || true

echo
echo "== Groups =="
id

echo
echo "== ROCm tools =="
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi || true
else
    echo "rocm-smi is not present in this image or not on PATH."
fi

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | sed -n '1,120p'
else
    echo "rocminfo is not present in this image or not on PATH."
fi

echo
echo "== Python, PyTorch, and vLLM =="
python - <<'PY'
import importlib
import os
import sys

print("python:", sys.version.replace("\n", " "))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("VLLM_CACHE_ROOT:", os.environ.get("VLLM_CACHE_ROOT"))

for module_name in ("torch", "vllm"):
    module = importlib.import_module(module_name)
    print(f"{module_name}:", getattr(module, "__version__", "unknown"))

import torch

print("torch.version.hip:", torch.version.hip)
print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch ROCm backend did not report an available GPU.")

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(
        f"device[{index}]: name={props.name}, "
        f"total_memory={props.total_memory // (1024 ** 2)} MiB"
    )

x = torch.ones((1024,), device="cuda")
print("cuda_tensor_sum:", float(x.sum().item()))
PY

echo
echo "Smoke test passed. Start vLLM only after these checks are clean."
