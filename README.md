# AMD Strix Halo — vLLM Toolbox/Container (gfx1151, PyTorch + AOTriton)

An **Arch-based** Docker/Podman container that is **Toolbx-compatible** (usable as a Fedora toolbox) for serving LLMs with **vLLM** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. Built on the PyTorch + AOTriton base to make ROCm on Strix Halo practical for day‑to‑day use.

> **Built on:** [https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton](https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton)
> **Credits:** **lhl** (build tools/scripts), **ssweens** (Arch‑based Dockerfiles), and the **AMD Strix Halo Home Lab Discord** for testing/support.

---

## ⚠️ Status & Expectations (Experimental)

This setup is **highly experimental** on ROCm/Strix Halo. Some models work; **many fail** due to missing custom kernels, unsupported quant types, or TorchInductor/AOTriton limitations on gfx1151. The matrix below lists combinations tested so far. **Please contribute fixes** or additional working recipes (see *Contributing*).

---

## Tested Models (Experimental Matrix)

> **Legend:** ✅ Works (with flags) · ❌ Fails · ⚠️ Notes include the *exact* error/symptom seen.

| Model (Hugging Face)               | Params / Quant |               Status | Required flags (if any)                              | Notes / Errors                                                                                       |
| ---------------------------------- | -------------- | -------------------: | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `Qwen/Qwen2.5-7B-Instruct`          | 7B FP16        |              ✅ Works | (recommended) `--dtype float16`                      | Good baseline; simple serve works.                                                                   |
| `meta-llama/Llama-2-7b-chat-hf`     | 7B FP16        |              ✅ Works | (recommended) `--dtype float16`                      | Stable.                                                                                              |
| `Qwen/Qwen3-30B-A3B-Instruct-2507`  | 30B (A3B) FP16 |              ✅ Works | (recommended) `--dtype float16`                      |  |
| `Qwen/Qwen3-4B-Instruct-2507`       | 4B FP16        |              ✅ Works | (recommended) `--dtype float16`                      |  |
| `Google/Gemma3-27B-Instruct`        | 27B FP16       |              ✅ Works | (recommended) `--dtype float16`                      | Slow         |
| `Google/Gemma3-12B-Instruct`        | 12B FP16       |              ✅ Works | (recommended) `--dtype float16`                      |          |
| `Google/Gemma3-4B-Instruct`         | 4B FP16        |              ✅ Works | (recommended) `--dtype float16`                      |          |
| `Qwen/Qwen3-14B-AWQ`                | 14B AWQ        | ✅ Works (with flags) | `--quantization awq --dtype float16 --enforce-eager` | On ROCm, eager avoids missing `awq_dequantize` during compile; vLLM auto‑sets `VLLM_USE_TRITON_AWQ`. |
| `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ`| 4B AWQ         | ✅ Works (with flags) | `--quantization awq --dtype float16 --enforce-eager` | On ROCm, eager avoids missing `awq_dequantize` during compile; vLLM auto‑sets `VLLM_USE_TRITON_AWQ`. |
| `openai/gpt-oss-20b`                | 20B MXFP4      |              ❌ Fails | —                                                    | `ModuleNotFoundError: triton_kernels.matmul_ogs` (MXFP4 path not available in this image).           |
| `zai-org/GLM-4.5-Air-FP8`           | FP8            |              ❌ Fails | —                                                    | `ValueError: type fp8e4nv not supported (only 'fp8e5')`.                                             |
| `cpatonn/GLM-4.5-Air-AWQ-4bit`      | AWQ-4bit (MoE) |              ❌ Fails | —                                                    | Missing custom op: `torch.ops._C.gptq_marlin_repack` (Marlin kernels).                               |

> If you get a model to work, please PR a new row with: **model name**, **exact flags**, vLLM version, `torch` & `triton` versions, and a note on **gfx1151** driver/kernel stack.

---

## Table of Contents

* [1) Toolbx vs Docker/Podman](#1-toolbx-vs-dockerpodman)
* [2) Quickstart — Fedora Toolbx (development)](#2-quickstart--fedora-toolbx-development)
* [3) Testing the API](#3-testing-the-api)
* [4) Quickstart — Podman/Docker](#4-quickstart--podmandocker)
* [5) Models, dtypes & storage](#5-models-dtypes--storage)
* [6) Performance notes (short)](#6-performance-notes-short)
* [7) Requirements (host)](#7-requirements-host)
* [8) Acknowledgements & Links](#8-acknowledgements--links)
* [Tested Models](#tested-models)
* [Contributing](#contributing)


## 1) Toolbx vs Docker/Podman

The `kyuz0/vllm-therock-gfx1151-aotriton:latest` image can be used both as: 

* **Fedora Toolbx (recommended for development):** Toolbx shares your **HOME** and user, so models/configs live on the host. Great for iterating quickly while keeping the host clean. 
* **Docker/Podman (recommended for deployment/perf):** Use for running vLLM as a service (host networking, IPC tuning, etc.). Always **mount a host directory** for model weights so they stay outside the container.

---

## 2) Quickstart — Fedora Toolbx (development)

Create a toolbox that exposes the GPU and relaxes seccomp to avoid ROCm syscall issues:

```bash
toolbox create vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151-aotriton:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

Enter it:

```bash
toolbox enter vllm
```

**Model storage (Toolbx):** keep weights **outside** the toolbox under your HOME so they persist. Recommended path:

```bash
mkdir -p ~/vllm-models
```

Serve a model using the helper script **`start-vllm`** (it prints the exact `vllm serve` command and then runs it). Models download to `~/vllm-models` by default; if a model isn't present, it will be fetched from Hugging Face automatically:

```bash
start-vllm
# pick a model from the menu; the script prints the serve command and launches it
```

> Defaults: `0.0.0.0:8000` and `~/vllm-models` for weights. You can still run `vllm serve` manually if you prefer.

> Toolbx shares HOME by design, so `~/vllm-models` stays on the host and survives toolbox updates.
>
> **Cache note (Toolbx):** vLLM will also write compiled kernels to `~/.cache/vllm/torch_compile_cache/` in your HOME. For example:
>
> ```bash
> du -sh ~/.cache/vllm/torch_compile_cache/
> # e.g., 138M  /home/you/.cache/vllm/torch_compile_cache/
> ```

---

## 3) Testing the API

Once the server is up, hit the OpenAI‑compatible endpoint:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hello! Test the performance."}]}'
```

You should receive a JSON response with a `choices[0].message.content` reply.

If you don't want to bother specifying the model name, you can run this which will query the currently deployed model:

```bash
MODEL=$(curl -s http://localhost:8000/v1/models | jq -r '.data[0].id') curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Hello! Test the performance.\"}]
  }"
```

---

## 4) Quickstart — Podman/Docker

Prefer this for persistent services. **Always mount a host directory for weights** so they live outside the container. If the model isn't present, vLLM will fetch it from **Hugging Face** into the mapped directory.

**Qwen2.5 7B Instruct**

```bash
podman run -d --name vllm-qwen2p5-7b \
  --ipc=host \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/vllm-models:/models \
  -v ~/.cache/vllm:/root/.cache/vllm \
  docker.io/kyuz0/vllm-therock-gfx1151-aotriton:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve Qwen/Qwen2.5-7B-Instruct --dtype float16 \
      --host 0.0.0.0 --port 8000 --download-dir /models'
```

> Not using `--network host`? Map a port instead: `-p 8000:8000`.

For other models, you can try:


**Qwen3 30B A3B Instruct (2507)**

```bash
podman run -d --name vllm-qwen3-30b-a3b \
  --ipc=host \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/vllm-models:/models \
  -v ~/.cache/vllm:/root/.cache/vllm \
  docker.io/kyuz0/vllm-therock-gfx1151-aotriton:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --dtype float16 \
      --host 0.0.0.0 --port 8000 --download-dir /models'
```

**Qwen3 14B AWQ**  *(requires extra flags on ROCm)*

```bash
podman run -d --name vllm-qwen3-14b-awq \
  --ipc=host \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  -v ~/vllm-models:/models \
  -v ~/.cache/vllm:/root/.cache/vllm \
  docker.io/kyuz0/vllm-therock-gfx1151-aotriton:latest \
  bash -lc 'source /torch-therock/.venv/bin/activate; \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    vllm serve Qwen/Qwen3-14B-AWQ --quantization awq --dtype float16 --enforce-eager \
      --host 0.0.0.0 --port 8000 --download-dir /models'
```

---

## 5) Models, dtypes & storage

* Start with **Qwen/Qwen2.5-7B-Instruct**; larger models may work but are less forgiving on unified memory.
* Use `--dtype float16` unless you have a reason to change.
* **Storage discipline:**

  * **Toolbx:** `--download-dir ~/vllm-models` (lives in your HOME on the host).
  * **Podman/Docker:** `-v ~/vllm-models:/models` and `--download-dir /models`.

---

## 6) Performance notes (short)

* The image is built on the PyTorch + **AOTriton** base; enabling `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` can improve startup/throughput on some models.
* vLLM flags you might tune later: `--gpu-memory-utilization`, `--max-num-seqs`, `--max-model-len`. Start simple; add knobs only if needed.

---

## 7) Requirements (host)

**Hardware & drivers**

* AMD Strix Halo APU (gfx1151).
* Working amdgpu stack with `/dev/kfd` (ROCm compute) and `/dev/dri` (graphics).
* Your user in the **video** and **render** groups.

**Unified memory setup (HIGHLY recommended)**
Enable large GTT/unified memory so the iGPU can borrow system RAM for bigger models:

1. **Kernel parameters** (append to your GRUB cmdline):

   ```
   amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432
   ```

   | Parameter                  | Purpose                      |
   | -------------------------- | ---------------------------- |
   | `amd_iommu=off`            | Reduces latency              |
   | `amdgpu.gttsize=131072`    | 128 GiB GTT (unified memory) |
   | `ttm.pages_limit=33554432` | Large pinned allocations     |

2. **BIOS**: allocate **minimal VRAM** to the iGPU (e.g., **512 MB**) and rely on unified memory.

3. **Fedora example** (GRUB): edit `/etc/default/grub` → `GRUB_CMDLINE_LINUX=...` then:

   ```bash
   sudo grub2-mkconfig -o /boot/grub2/grub.cfg
   sudo reboot
   ```

**Container runtime**

* Podman or Docker installed (examples use Podman; replace with Docker if preferred).

---

## 8) Contributing

Spotted a fix, a working flag combo, or a model that should be on the list? **PRs welcome!** Please include:

* Model repo + exact version tag (if any)
* Full `vllm serve` command/flags that work
* vLLM version, `torch` & `triton` versions (`python -c "import torch, triton; print(torch.__version__, triton.__version__)"`)
* Short log snippet of success/failure (especially the **first** error)
* Any relevant kernel/AOTriton env vars (e.g., `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`)

---

## 9) Acknowledgements & Links

* Base images & docs: [https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton](https://github.com/kyuz0/amd-strix-halo-pytorch-gfx1151-aotriton)
* Upstreams: [vLLM](https://github.com/vllm-project/vllm), [ROCm/TheRock](https://github.com/ROCm/TheRock), [AOTriton](https://github.com/ROCm/aotriton)
* Community: **AMD Strix Halo Home Lab Discord** — [https://discord.gg/pnPRyucNrG](https://discord.gg/pnPRyucNrG)
* Big thanks to **lhl** and **ssweens** for doing the actual heavy lifting for this.
