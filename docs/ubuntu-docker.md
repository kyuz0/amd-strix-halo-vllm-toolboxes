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

Only after the smoke test passes should you launch vLLM:

```bash
start-vllm
```

For a direct server launch instead of the TUI, pass the command after `--`:

```bash
./scripts/run-ubuntu-docker-vllm.sh run -- \
  python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

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

Once vLLM is listening on port 8000:

```bash
curl -s http://localhost:8000/v1/models
```

Then send a small OpenAI-compatible request using the model ID returned by the
server.

## Fedora Toolbx Is Separate

The main README still documents Fedora Toolbx and RDMA-oriented workflows. This
Ubuntu path is Docker-native and should be used on Ubuntu hosts where the goal
is to run Docker Engine directly without Podman, Distrobox, toolbox, or LXC.
