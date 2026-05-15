#!/usr/bin/env bash
set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-docker.io/kyuz0/vllm-therock-gfx1151}"
IMAGE_TAG="${IMAGE_TAG:-stable}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-strix-halo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
WORKSPACE_MOUNT="${WORKSPACE_MOUNT:-$REPO_ROOT}"

usage() {
    cat <<EOF
Usage:
  $0 shell
  $0 smoke
  $0 run -- <command> [args...]
  $0 print -- <command> [args...]

Environment:
  IMAGE_REPO          default: docker.io/kyuz0/vllm-therock-gfx1151
  IMAGE_TAG           default: stable
  CONTAINER_NAME      default: vllm-strix-halo
  HF_HOME             default: \$HOME/.cache/huggingface
  VLLM_CACHE_ROOT     default: \$HOME/.cache/vllm
  WORKSPACE_MOUNT     default: repository root

This script uses Docker Engine only. It does not use Podman, Distrobox,
toolbox, LXC, sudo, host ROCm tools, or package installation.
EOF
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Error: docker is not available in PATH." >&2
        exit 1
    fi
}

require_device() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "Error: required device is missing: $path" >&2
        exit 1
    fi
}

build_docker_args() {
    local image="${IMAGE_REPO}:${IMAGE_TAG}"

    DOCKER_ARGS=(
        run
        --rm
        -it
        --name "$CONTAINER_NAME"
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --group-add render
        --security-opt seccomp=unconfined
        --ipc=host
        --network=host
        -e HF_HOME=/root/.cache/huggingface
        -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
        -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
        -e VLLM_CACHE_ROOT=/root/.cache/vllm
        -v "${HF_HOME}:/root/.cache/huggingface"
        -v "${VLLM_CACHE_ROOT}:/root/.cache/vllm"
        -v "${WORKSPACE_MOUNT}:/workspace:ro"
        -w /workspace
        "$image"
    )
}

print_command() {
    printf 'docker'
    printf ' %q' "${DOCKER_ARGS[@]}"
    printf ' %q' "$@"
    printf '\n'
}

main() {
    local mode="${1:-}"
    shift || true

    case "$mode" in
        shell|smoke|run|print)
            ;;
        -h|--help|help|"")
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown mode: $mode" >&2
            usage >&2
            exit 1
            ;;
    esac

    require_docker
    require_device /dev/kfd
    require_device /dev/dri

    mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"
    build_docker_args

    case "$mode" in
        shell)
            docker "${DOCKER_ARGS[@]}" bash -l
            ;;
        smoke)
            docker "${DOCKER_ARGS[@]}" /workspace/scripts/smoke-test-ubuntu-docker-vllm.sh
            ;;
        run)
            if [[ "${1:-}" == "--" ]]; then
                shift
            fi
            if [[ "$#" -eq 0 ]]; then
                echo "Error: run mode requires a command after --." >&2
                exit 1
            fi
            docker "${DOCKER_ARGS[@]}" "$@"
            ;;
        print)
            if [[ "${1:-}" == "--" ]]; then
                shift
            fi
            if [[ "$#" -eq 0 ]]; then
                set -- bash -l
            fi
            print_command "$@"
            ;;
    esac
}

main "$@"
