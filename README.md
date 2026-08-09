# AMD Strix Halo (gfx1151) — vLLM Toolbox/Container

An **Fedora 43** Docker/Podman container that is **Toolbx-compatible** (usable as a Fedora toolbox) for serving LLMs with **vLLM** on **AMD Ryzen AI Max “Strix Halo” (gfx1151)**. Built on the **TheRock nightly builds** for ROCm.

---

## 🚀 High-Performance Clustering Support (New!)

**Update:** This toolbox now ships with a **custom build of ROCm/RCCL** that enables **native RDMA/RoCE v2 support for Strix Halo (gfx1151)**. This allows you to connect two nodes via a low-latency interconnect (e.g., Intel E810) and run vLLM with Tensor Parallelism (TP=2) effectively acting as a single 256GB Unified Memory GPU.

👉 **[Read the Full RDMA Cluster Setup Guide](rdma_cluster/setup_guide.md)** for hardware requirements and configuration instructions.

---

### 📦 Project Context

This repository is part of the **[Strix Halo AI Toolboxes](https://strix-halo-toolboxes.com)** project. Check out the website for an overview of all toolboxes, tutorials, and host configuration guides.

### ❤️ Support

This is a hobby project maintained in my spare time. If you find these toolboxes and tutorials useful, you can **[buy me a coffee](https://buymeacoffee.com/dcapitella)** to support the work! ☕

## 🙏 Acknowledgments

* **Adrian ([@Lafunamor](https://github.com/Lafunamor))**: Huge thanks for all the help, PRs, and testing to get this project stabilized!
* **Patrick Audley ([paudley/ai-notes](https://github.com/paudley/ai-notes))**: Thanks for the `strix-halo` build notes. This toolbox relies on that research (specifically the Triton patches and `aiter` compilation strategy) to successfully run vLLM and AITER Flash-Attention on Strix Halo.

---

## Table of Contents

* [Tested Models (Benchmarks)](#tested-models-benchmarks)
* [1) Toolbx vs Docker/Podman](#1-toolbx-vs-dockerpodman)
* [2) Quickstart — Fedora Toolbx](#2-quickstart--fedora-toolbx)
* [3) Quickstart — Ubuntu (Distrobox)](#3-quickstart--ubuntu-distrobox)
* [4) Testing the API](#4-testing-the-api)
* [5) Use a Web UI for Chatting](#5-use-a-web-ui-for-chatting)
* [6) Host Configuration](#6-host-configuration)
* [7) Distributed Clustering (RDMA/RoCE)](#7-distributed-clustering-rdmaroce)


## Tested Models (Benchmarks)

> [!IMPORTANT]
> **Note on Throughput:** These benchmarks measure **Peak Multi-User Throughput** (Tokens/Second) at high concurrency (batching multiple sequences simultaneously to saturate the Strix Halo's memory bandwidth). If you are testing with a single request (Concurrency = 1), your individual generation speed will be lower than these maximum hardware-saturation numbers. These metrics represent the total capacity of the system under heavy load.

View full benchmarks at: [https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/](https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/)

| Model | Params / Quant | GPU Requirement |
| :--- | :--- | :--- |
| [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | 8B / BF16 | 1 GPU (TP=1, 2) |
| [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | 26B / BF16 | 1 GPU (TP=1, 2) |
| [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) | 31B / BF16 | 1 GPU (TP=1, 2) |
| [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | 20B / BF16 | 1 GPU (TP=1, 2) |
| [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | 120B / BF16 | 1 GPU (TP=1) |
| [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B / BF16 | 1 GPU (TP=1) |
| [`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit) | 35B / AWQ 4-bit | 1 GPU (TP=1) |
| [`cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit) | 122B / AWQ 4-bit | 1 GPU (TP=1, 2) |
| [`cyankiwi/Qwen3.5-122B-A10B-AWQ-8bit`](https://huggingface.co/cyankiwi/Qwen3.5-122B-A10B-AWQ-8bit) | 122B / AWQ 8-bit | **2 GPUs (TP=2 Only)** |
| [`cyankiwi/MiniMax-M2.7-AWQ-4bit`](https://huggingface.co/cyankiwi/MiniMax-M2.7-AWQ-4bit) | N/A / AWQ 4-bit | **2 GPUs (TP=2 Only)** |
| [`ayysasha/MiniMax-M2.7-AWQ-G32-STRIX-2H`](https://huggingface.co/ayysasha/MiniMax-M2.7-AWQ-G32-STRIX-2H) | N/A / Mixed BF16+INT4 AWQ | **2 GPUs (TP=2 Only)** |


---

## 1) Toolbx vs Docker/Podman

The `kyuz0/vllm-therock-gfx1151` image is available in two channels:

| Tag | Description |
| :--- | :--- |
| **`:latest`** | Last verified working build. **Recommended for most users.** |
| **`:dev`**    | Absolute latest build. May contain upstream regressions. |

The image can be used both as:

* **Fedora Toolbx (recommended for development):** Toolbx shares your **HOME** and user, so models/configs live on the host. Great for iterating quickly while keeping the host clean.
* **Docker/Podman (recommended for deployment/perf):** Use for running vLLM as a service (host networking, IPC tuning, etc.). Always **mount a host directory** for model weights so they stay outside the container.


---

## 2) Quickstart — Fedora Toolbx

**Recommended:** Use the included `refresh_toolbox.sh` script. It pulls the image and creates the toolbox with the correct parameters:

```bash
# Interactive — prompts you to choose latest (default) or dev
./refresh_toolbox.sh

# Or specify directly:
./refresh_toolbox.sh latest   # verified working build
./refresh_toolbox.sh dev      # bleeding edge
```

> **InfiniBand / RDMA Support:** The script automatically detects if a fast InfiniBand link is active (checks `/dev/infiniband`). If found, it correctly sets up the container to expose these devices, enabling high-performance clustering.

**Manual Creation:**

To manually create a toolbox that exposes the GPU and relaxes seccomp:

```bash
toolbox create vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add keep-groups --security-opt seccomp=unconfined
```

> [!IMPORTANT]
> Use `--group-add keep-groups`, **not** `--group-add video --group-add render`. See
> [GPU device permissions](#61-gpu-device-permissions) — the named groups do not grant GPU
> access under rootless podman and can stop the container from starting at all.

Enter it:

```bash
toolbox enter vllm
```

**Model storage:** Models are downloaded to `~/.cache/huggingface` by default. This directory is shared with the host if you created the toolbox correctly, so downloads persist.

### Serving a Model (Easiest Way)

The toolbox includes a TUI wizard called **`start-vllm`** which includes pre-configured models and handles the launch flags for you. This is the easiest way to get started.

```bash
start-vllm
```

> **Cache note:** vLLM writes compiled kernels to `~/.cache/vllm/`.

---

## 3) Quickstart — Ubuntu (Distrobox)

Ubuntu’s toolbox package still breaks GPU access, so use Distrobox instead:

```bash
distrobox create -n vllm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined"

distrobox enter vllm
```

The canonical `vllm-therock-gfx1151` image is built on Ubuntu with the stable ROCm stack.

> **Verification:** Run `rocm-smi` to check GPU status. It should print your GPU name (e.g.
> `Radeon 8060S Graphics`). If it reports `get_name, Failed to load a library` or no device at
> all, see [GPU device permissions](#61-gpu-device-permissions).

### Serving a Model (Easiest Way)

The toolbox includes a TUI wizard called **`start-vllm`** which includes pre-configured models and handles the launch flags for you. This is the easiest way to get started.

```bash
start-vllm
```

---

## 4) Testing the API

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

## 5) Use a Web UI for Chatting

If vLLM is on a remote server, expose port 8000 via SSH port forwarding:

```bash
ssh -L 0.0.0.0:8000:localhost:8000 <vllm-host>
```

Then, you can start HuggingFace ChatUI like this (on your host):

```bash
docker run -p 3000:3000 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=dummy \
  -v chat-ui-data:/data \
  ghcr.io/huggingface/chat-ui-db
```

## 6) Host Configuration

This should work on any Strix Halo. For a complete list of available hardware, see: [Strix Halo Hardware Database](https://strixhalo-homelab.d7.wtf/Hardware)

### 6.1 GPU Device Permissions

The container needs access to `/dev/kfd` (the ROCm compute device) and `/dev/dri/renderD*`. On
most distributions these ship as mode **0660 `root:render`**, so access is granted by group
membership. Add yourself to the groups once (log out and back in afterwards):

```bash
sudo usermod -aG render,video "$USER"
```

**Then pass those groups into the container with `--group-add keep-groups`.**

> [!WARNING]
> Do **not** use `--group-add video --group-add render`. Under **rootless podman** — the default
> for both toolbx and distrobox — a *named* `--group-add` is resolved against the **container's**
> `/etc/group`, and the resulting gid lives inside the user namespace. It never maps to the
> host's `render`/`video` gid, so it grants **no** access to `/dev/kfd`. Two failure modes follow:
>
> * **Container refuses to start**, if the image has no such group:
>   `Error: ... unable to find group render: no matching entries in group file`
> * **Container starts but has no GPU**: `/dev/kfd` returns `EACCES` and `rocm-smi` reports
>   `get_name, Failed to load a library` with no device name.
>
> `keep-groups` (crun's `keep_original_groups`) passes your **real host** supplementary groups
> through, which is what actually authorises the device.

**Rootful Docker** has no user namespace, so numeric **host** gids work there instead:

```bash
docker run --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" ...
```

**Headless / multi-user hosts** may prefer relaxing the device modes instead of managing groups:

```bash
# /etc/udev/rules.d/70-kfd.rules
SUBSYSTEM=="kfd", KERNEL=="kfd", MODE="0666"
SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"
```

Reload with `sudo udevadm control --reload && sudo udevadm trigger`. This makes the devices
world-accessible, which removes the group requirement entirely — convenient on a single-user
box, but it does grant every local user GPU access.

**Quick diagnosis** from inside the container:

```bash
id                                  # do you actually have the host render/video gids?
ls -l /dev/kfd /dev/dri/renderD128  # 0660 needs group access; 0666 needs nothing
rocm-smi --showproductname          # should print your GPU name
```

### 6.2 Test Configuration

| Component         | Specification                                               |
| :---------------- | :---------------------------------------------------------- |
| **Test Machine**  | Framework Desktop                                           |
| **CPU**           | Ryzen AI MAX+ 395 "Strix Halo"                              |
| **System Memory** | 128 GB RAM                                                  |
| **GPU Memory**    | 512 MB allocated in BIOS                                    |
| **Host OS**       | Fedora 43, Linux 6.18.5-200.fc43.x86_64            |

### 6.3 Kernel Parameters (tested on Fedora 42)

Add these boot parameters to enable unified memory while reserving a minimum of 4 GiB for the OS (max 124 GiB for iGPU):

> [!WARNING]
> Based on [benchmarking by Lars Urban (@urbanswelt)](https://github.com/urbanswelt), there is definitive indication that setting `amd_iommu=off` performs better than the previously recommended `iommu=pt`. Key result: `amd_iommu=off` is 5-12% faster than either IOMMU-enabled mode. See [Issue #66](https://github.com/kyuz0/amd-strix-halo-toolboxes/issues/66#issuecomment-4460612951) for details.

`amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856`

| Parameter                   | Purpose                                                                                    |
|-----------------------------|--------------------------------------------------------------------------------------------|
| `amd_iommu=off`             | Disables the AMD IOMMU. This improves performance over `iommu=pt`, reducing overhead for both the RDMA NIC and the iGPU unified memory access. |
| `amdgpu.gttsize=126976`     | Caps GPU unified memory to 124 GiB; 126976 MiB ÷ 1024 = 124 GiB                            |
| `ttm.pages_limit=32505856`  | Caps pinned memory to 124 GiB; 32505856 × 4 KiB = 126976 MiB = 124 GiB                     |

Source: https://www.reddit.com/r/LocalLLaMA/comments/1m9wcdc/comment/n5gf53d/?context=3&utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button


**Apply the changes:**

```
# Edit /etc/default/grub to add parameters to GRUB_CMDLINE_LINUX
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

## 7) Distributed Clustering (RDMA/RoCE)

This toolbox supports high-performance clustering of multiple Strix Halo nodes using Infiniband or RoCE v2 (e.g., Intel E810). This enables **Tensor Parallelism** across machines with extremely low latency (~5µs).

**Detailed Documentation:** [RDMA Cluster Setup Guide](rdma_cluster/setup_guide.md)

**Key Features:**
*   **Custom RCCL Patch:** Use of a custom-built `librccl.so` to support RDMA on `gfx1151`.
*   **Easy Setup:** `refresh_toolbox.sh` automatically detects and exposes RDMA devices.
*   **Cluster Management:** Included `start-vllm-cluster` TUI for managing Ray and vLLM.

## 8) AITER on Strix Halo Support Status

This toolbox uses only the AITER paths verified on Strix Halo (gfx1151). The broad `VLLM_ROCM_USE_AITER` toggle stays disabled by default because it also enables unsupported operators such as the AITER sampler.

To bypass this limitation, `scripts/patch_strix.py` applies a few APU-specific guards (building on the work from `ai-notes` linked above):
* **Patch 2 (`vllm/_aiter_ops.py`)**: Intercepts the MoE gate (`is_fused_moe_enabled()`) forcing it to disable AITER MoE and Linear FP8 on `gfx1x` architectures.
* **Patch 3.5 (`vllm/model_executor/layers/fused_moe/oracle/unquantized.py`)**: Blocks the `VLLM_ROCM_USE_AITER_MOE` environment variable from forcing a JIT compile override.
* **Patch 5 (`vllm/platforms/rocm.py`)**: Bypasses the RMSNorm custom op registration on `gfx1x` to prevent CUDA Graph capture crashes during model initialization.

The launcher exposes the exact generic backend names: `TRITON_ATTN`, `ROCM_ATTN`, and `ROCM_AITER_UNIFIED_ATTN`. `Qwen/Qwen3.6-35B-A3B` defaults to the gfx1151-verified unified AITER backend with the broad AITER toggle disabled. The legacy `ROCM_AITER_FA` backend is not offered because its paged-attention decode kernel has no Navi implementation.

DeepSeek V4 is separate: its ROCm model implementation hardwires the model-specific `ROCM_FLASHMLA_SPARSE_DSV4` backend, so the launcher does not pass `--attention-backend`. Its model entry enables AITER for the tested sparse-indexer MQA-logits helper, disables AITER linear, and uses `--logprobs-mode processed_logprobs` to avoid the unsupported sampler.
