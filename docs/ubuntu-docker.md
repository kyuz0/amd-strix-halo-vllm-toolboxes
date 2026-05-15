# Ubuntu 26.04 Docker Engine Path for Strix Halo

This path is for Ubuntu hosts using Docker Engine directly. It does not use
Podman, Distrobox, Fedora Toolbx, Ubuntu toolbox, or LXC.

The container image is the same Strix Halo `gfx1151` image used elsewhere in
this repository:

```bash
docker.io/kyuz0/vllm-therock-gfx1151:stable
```

Use `:stable` first. Use `:latest` only when you intentionally want to test a
newer image that may have upstream regressions.

## Tested Host Shape

This document targets Ubuntu 26.04 LTS on AMD Strix Halo / Radeon 8050S or
8060S class hardware, with Docker Engine already installed and active.

Known-good host expectations:

- `/dev/kfd` exists and is accessible to the `render` group.
- `/dev/dri/card*` exists and is accessible to the `video` group.
- `/dev/dri/renderD*` exists and is accessible to the `render` group.
- The user running Docker is already in `docker`, `video`, and `render`.
- ROCm tools do not need to be installed on the Ubuntu host. ROCm is validated
  inside the container.

## Host Kernel Notes

For Ubuntu, update GRUB with `update-grub`, not `grub2-mkconfig`.

The upstream Strix Halo guidance commonly uses:

```text
iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

On Ubuntu, that means editing `/etc/default/grub`, then applying with:

```bash
sudo update-grub
sudo reboot
```

Do not run those commands from this repository's scripts. They are host state
changes and should be handled manually by the system owner.

If the current kernel command line already contains `iommu=pt` and
`ttm.pages_limit=32505856` but does not contain `amdgpu.gttsize=126976`, treat
that as a host configuration difference to review before large-model testing.
If it also contains `ttm.page_pool_size=32505856`, note that this differs from
the upstream README recommendation above and should be evaluated separately.

## Required Docker Runtime Flags

The Docker container needs direct access to the AMD GPU devices and relaxed
runtime isolation for ROCm/vLLM:

- `--device /dev/kfd`
- `--device /dev/dri`
- `--group-add video`
- `--group-add render`
- `--security-opt seccomp=unconfined`
- `--ipc=host`
- `--network=host`

The helper script in this repo uses those flags by default.

## Cache Mounts

Keep model weights and compiled kernels on the host so they survive container
recreation:

- Hugging Face cache: `~/.cache/huggingface` mounted to
  `/root/.cache/huggingface`
- vLLM cache: `~/.cache/vllm` mounted to `/root/.cache/vllm`

Override them when needed:

```bash
HF_HOME=/data/hf-cache VLLM_CACHE_ROOT=/data/vllm-cache \
  ./scripts/run-ubuntu-docker-vllm.sh shell
```

## Hugging Face Authentication

Do not paste Hugging Face tokens into docs, issues, pull requests, or chat.
Inside this image, `huggingface-cli login` is deprecated. Use:

```bash
hf auth login
hf auth whoami
```

The Docker runner mounts the host Hugging Face cache into
`/root/.cache/huggingface`, so authentication persists for future runs. The git
credential helper warning can be ignored for model downloads; it matters only
for Git-based Hugging Face pushes.

## First Run: Conservative Validation

Start with an interactive shell:

```bash
./scripts/run-ubuntu-docker-vllm.sh shell
```

Inside the container, run the smoke test:

```bash
/workspace/scripts/smoke-test-ubuntu-docker-vllm.sh
```

The smoke test checks device nodes, ROCm visibility, Python package imports,
PyTorch ROCm availability, and basic GPU discovery before any vLLM server is
started.

## Memory Exposure Observations

Ubuntu 26.04 validation on kernel `7.0.0-15-generic` used the
`docker.io/kyuz0/vllm-therock-gfx1151:stable` image. The Docker ROCm/vLLM smoke
test passed on that host.

The host kernel command line included:

```text
iommu=pt ttm.pages_limit=32505856 ttm.page_pool_size=32505856 amdgpu.gttsize=126976
```

With that configuration, `dmesg` reported:

```text
126976M of GTT memory ready.
```

Treat this as kernel GTT readiness, not as proof that a single userspace
allocation can consume all of that memory. In the same validation, PyTorch still
reported `reported_total_memory_mib` around `62890`, so PyTorch's reported total
memory should not be presented as the whole usable GTT story either.

High-memory allocation tests must be run with other GPU/model containers
stopped. With existing Strix model containers still running, a 48 GiB PyTorch
allocation caused global OOM or hung-system behavior. After stopping
`qwen3-coder` and `qwen3-6` and disabling their restart policy, single PyTorch
`uint8` allocations of 40 GiB, 44 GiB, and 48 GiB passed cleanly.

The current proven clean single-allocation size from this validation is 48 GiB.
Do not claim that 126 GiB usable allocation has been proven from these results.

Only after the smoke test passes should you launch vLLM:

```bash
start-vllm
```

For a direct server launch instead of the TUI, pass the command after `--`.
Port `8000` may already be occupied by another service or legacy container; in
one validation it was occupied by `legacy-printer`. Check before launching, or
use another port:

```bash
ss -ltnp | grep ':8000'
lsof -iTCP:8000 -sTCP:LISTEN
```

The first conservative API validation model was
`Qwen/Qwen2.5-7B-Instruct`. This is a small validation target for proving the
Docker/vLLM API path, not the final performance target and not proof that
larger models are validated.

```bash
./scripts/run-ubuntu-docker-vllm.sh run -- \
  python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8010 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.60 \
    --enforce-eager
```

Do not make this direct-launch example depend on `VLLM_ATTENTION_BACKEND`; the
validated `docker.io/kyuz0/vllm-therock-gfx1151:stable` build did not recognize
that environment variable.

## Helper Script

The Docker runner is repo-local:

```bash
./scripts/run-ubuntu-docker-vllm.sh shell
./scripts/run-ubuntu-docker-vllm.sh smoke
./scripts/run-ubuntu-docker-vllm.sh run -- start-vllm
```

It intentionally does not use or require Podman, Distrobox, Fedora Toolbx,
Ubuntu toolbox, LXC, host ROCm tools, package installation, or `sudo`.

## API Check After vLLM Starts

Ubuntu 26.04 validation on kernel `7.0.0-15-generic` with
`docker.io/kyuz0/vllm-therock-gfx1151:stable` launched
`Qwen/Qwen2.5-7B-Instruct` successfully on port `8010` using:

- `--dtype bfloat16`
- `--max-model-len 8192`
- `--gpu-memory-utilization 0.60`
- `--enforce-eager`

During startup, vLLM reported that model loading took `14.34 GiB` of memory and
reported `52.87 GiB` of available KV cache memory.

Once vLLM is listening, query the port selected at launch:

```bash
curl -s http://localhost:8010/v1/models
```

Then send a small OpenAI-compatible request using the model ID returned by the
server. In the validation above, `/v1/models` returned HTTP `200` and listed
`Qwen/Qwen2.5-7B-Instruct`; `/v1/chat/completions` returned a valid assistant
response.

### Authenticated Conservative API Validation

After Hugging Face authentication with `hf auth login`, the same Ubuntu Docker
vLLM validation was rerun against the cached model data. The authenticated run
used `docker.io/kyuz0/vllm-therock-gfx1151:stable` with
`Qwen/Qwen2.5-7B-Instruct` on port `8010` and the same conservative flags:

- `--dtype bfloat16`
- `--max-model-len 8192`
- `--gpu-memory-utilization 0.60`
- `--enforce-eager`

In that authenticated run, `/v1/models` returned HTTP `200`.
`/v1/chat/completions` returned HTTP `200` and produced exactly:

```text
authenticated Docker vLLM validation passed
```

The rerun did not show the previous unauthenticated Hugging Face warning. vLLM
reported that model loading took `14.34 GiB` and that available KV cache memory
was `52.87 GiB`. While vLLM was running, host memory showed about `46 GiB`
available.

This validates only the conservative authenticated Docker/vLLM API path above.
It is not larger-model validation and is not performance validation.

## Fedora Toolbx Is Separate

The main README still documents Fedora Toolbx and RDMA-oriented workflows. This
Ubuntu path is Docker-native and should be used on Ubuntu hosts where the goal
is to run Docker Engine directly without Podman, Distrobox, toolbox, or LXC.
