#!/usr/bin/env bash
# Lightweight banner with machine/GPU and ROCm version (vLLM edition)
# No Triton env sourcing, same info/format as the image/video banner.

# Only show for interactive shells
case $- in *i*) ;; *) return 0 ;; esac

oem_info() {
  local v="" m="" d lv lm
  for d in /sys/class/dmi/id /sys/devices/virtual/dmi/id; do
    [[ -r "$d/sys_vendor" ]] && v=$(<"$d/sys_vendor")
    [[ -r "$d/product_name" ]] && m=$(<"$d/product_name")
    [[ -n "$v" || -n "$m" ]] && break
  done
  # ARM/SBC fallback
  if [[ -z "$v" && -z "$m" && -r /proc/device-tree/model ]]; then
    tr -d '\0' </proc/device-tree/model
    return
  fi
  lv=$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')
  lm=$(printf '%s' "$m" | tr '[:upper:]' '[:lower:]')
  if [[ -n "$m" && "$lm" == "$lv "* ]]; then
    printf '%s\n' "$m"
  else
    printf '%s %s\n' "${v:-Unknown}" "${m:-Unknown}"
  fi
}

gpu_name() {
  local name=""
  if command -v rocm-smi >/dev/null 2>&1; then
    name=$(rocm-smi --showproductname --csv 2>/dev/null | tail -n1 | cut -d, -f2)
    [[ -z "$name" ]] && name=$(rocm-smi --showproductname 2>/dev/null | grep -m1 -E 'Product Name|Card series' | sed 's/.*: //')
  fi
  if [[ -z "$name" ]] && command -v rocminfo >/dev/null 2>&1; then
    name=$(rocminfo 2>/dev/null | awk -F': ' '/^[[:space:]]*Name:/{print $2; exit}')
  fi
  if [[ -z "$name" ]] && command -v lspci >/dev/null 2>&1; then
    name=$(lspci -nn 2>/dev/null | grep -Ei 'vga|display|gpu' | grep -i amd | head -n1 | cut -d: -f3-)
  fi
  # trim
  name=$(printf '%s' "$name" | sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//' -e 's/[[:space:]]\{2,\}/ /g')
  printf '%s\n' "${name:-Unknown AMD GPU}"
}

rocm_version() {
  # Prefer the PyTorch HIP version from the venv, fallback to rocm pkg metadata
  local PY="/torch-therock/.venv/bin/python"
  [[ -x "$PY" ]] || PY="python"
  "$PY" - <<'PY' 2>/dev/null || true
try:
    import torch
    v = getattr(getattr(torch, "version", None), "hip", "") or ""
    if v:
        print(v)
    else:
        raise Exception("no torch.version.hip")
except Exception:
    try:
        import importlib.metadata as im
        try:
            print(im.version("_rocm_sdk_core"))
        except Exception:
            print(im.version("rocm"))
    except Exception:
        print("")
PY
}

MACHINE="$(oem_info)"
GPU="$(gpu_name)"
ROCM_VER="$(rocm_version)"

echo
cat <<'ASCII'
███████╗████████╗██████╗ ██╗██╗  ██╗      ██╗  ██╗ █████╗ ██╗      ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██║╚██╗██╔╝      ██║  ██║██╔══██╗██║     ██╔═══██╗
███████╗   ██║   ██████╔╝██║ ╚███╔╝       ███████║███████║██║     ██║   ██║
╚════██║   ██║   ██╔══██╗██║ ██╔██╗       ██╔══██║██╔══██║██║     ██║   ██║
███████║   ██║   ██║  ██║██║██╔╝ ██╗      ██║  ██║██║  ██║███████╗╚██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ 

                               v L L M                                      
ASCII
echo
printf 'AMD STRIX HALO — vLLM Toolbox (gfx1151, ROCm via %s)\n' "${TOOLBOX_ROCM_SOURCE:-TheRock nightly}"
[[ -n "$ROCM_VER" ]] && printf 'ROCm: %s\n' "$ROCM_VER"
echo
printf 'Machine: %s\n' "$MACHINE"
printf 'GPU    : %s\n\n' "$GPU"
printf 'Repo   : https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes\n'
printf 'Image  : docker.io/kyuz0/vllm-therock-gfx1151:latest\n\n'
printf 'Included:\n'
printf '  - %-16s → %s\n' "start-vllm (TUI)" "Interactive launcher: Model select, Multi-GPU & Cache handling"
printf '  - %-16s → %s\n' "start-vllm-cluster" "Cluster launcher: Setup Ray Head/Worker & Launch vLLM RCCL"
printf '  - %-16s → %s\n' "vllm-cluster-bench" "Cluster Benchmark: TP=2, Auto-detected Env, JSON Results"
printf '  - %-16s → %s\n' "vLLM server" "vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct"
printf '  - %-16s → %s\n' "API test"    "curl localhost:8000/v1/chat/completions"
echo
printf 'SSH tip: ssh -L 8000:localhost:8000 user@host\n\n'

unset PROMPT_COMMAND
PS1='\u@\h:\w\$ '
