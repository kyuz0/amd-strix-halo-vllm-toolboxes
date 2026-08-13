#!/usr/bin/env bash

set -e

# Override only when intentionally testing another image repository.
IMAGE_REPO="${IMAGE_REPO:-docker.io/kyuz0/vllm-therock-gfx1151}"

# --- Channel selection (latest / dev) ---
resolve_channel() {
    local arg="${1:-}"
    case "$arg" in
        latest)  echo "latest" ;;
        dev)     echo "dev" ;;
        "")
            # Interactive menu
            echo "" >&2
            echo "Which image channel do you want?" >&2
            echo "  1) latest  — Last verified working build (recommended)" >&2
            echo "  2) dev     — Absolute latest build (may be unstable)" >&2
            echo "" >&2
            read -rp "Choice [1]: " choice
            case "${choice:-1}" in
                1|latest)  echo "latest" ;;
                2|dev)     echo "dev" ;;
                *)
                    echo "Invalid choice: $choice" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "Usage: $0 [latest|dev]" >&2
            echo "  latest  — Pull the last verified working build (default)" >&2
            echo "  dev     — Pull the absolute latest build" >&2
            exit 1
            ;;
    esac
}

CHANNEL="$(resolve_channel "${1:-}")"
IMAGE="${IMAGE_REPO}:${CHANNEL}"

# Keep stable and development toolboxes side by side by default. An explicit
# TOOLBOX_NAME still overrides this mapping.
if [ "$CHANNEL" = "dev" ]; then
    DEFAULT_TOOLBOX_NAME="vllm-therock-gfx1151-dev"
else
    DEFAULT_TOOLBOX_NAME="vllm-therock-gfx1151"
fi
TOOLBOX_NAME="${TOOLBOX_NAME:-$DEFAULT_TOOLBOX_NAME}"

# Base options.
#
# GPU access uses --group-add keep-groups, NOT --group-add video/render. Under rootless podman
# (what toolbx and distrobox use by default) a named --group-add resolves against the
# CONTAINER's /etc/group and the resulting gid lives inside the user namespace — it never maps
# to the host's video/render gid, so it grants no access to /dev/kfd or /dev/dri at all. Worse,
# if the name is absent from the image podman refuses to start the container outright
# ("unable to find group <name>: no matching entries in group file").
# keep-groups (crun's keep_original_groups) passes your real HOST supplementary groups through,
# which is what actually authorises /dev/kfd on a stock 0660 root:render device.
OPTIONS="--device /dev/dri --device /dev/kfd --group-add keep-groups --security-opt seccomp=unconfined"

# Check for InfiniBand devices
if [ -d "/dev/infiniband" ]; then
    echo "🔎 InfiniBand devices detected! Adding RDMA support..."
    # No --group-add rdma needed: keep-groups already carries the host's rdma group through.
    OPTIONS="$OPTIONS --device /dev/infiniband --ulimit memlock=-1"
else
    echo "ℹ️  No InfiniBand devices detected."
fi

# Detect container manager (toolbox requires podman; distrobox works with either)
if command -v toolbox &>/dev/null && command -v podman &>/dev/null; then
    MANAGER="toolbox"
elif command -v distrobox &>/dev/null; then
    MANAGER="distrobox"
else
    echo "Error: neither 'toolbox' (with podman) nor 'distrobox' is installed." >&2
    exit 1
fi

# Detect container runtime for image pull and cleanup
if command -v podman &>/dev/null; then
    RUNTIME="podman"
elif command -v docker &>/dev/null; then
    RUNTIME="docker"
else
    echo "Error: neither 'podman' nor 'docker' is installed." >&2
    exit 1
fi

echo "🔄 Refreshing $TOOLBOX_NAME via $MANAGER (channel: $CHANNEL, image: $IMAGE)"

# Remove existing container if it exists
if $MANAGER list 2>/dev/null | grep -q "$TOOLBOX_NAME"; then
    echo "🧹 Removing existing $MANAGER: $TOOLBOX_NAME"
    $MANAGER rm -f "$TOOLBOX_NAME"
fi

echo "⬇️ Pulling image: $IMAGE"
$RUNTIME pull "$IMAGE"

# Identify current image ID/digest for cleanup
new_id="$($RUNTIME image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
new_digest="$($RUNTIME image inspect --format '{{.Digest}}' "$IMAGE" 2>/dev/null || true)"

echo "📦 Recreating $MANAGER: $TOOLBOX_NAME"
echo "   Options: $OPTIONS"

if [ "$MANAGER" = "toolbox" ]; then
    # toolbox passes extra flags to podman via '--'
    toolbox create "$TOOLBOX_NAME" --image "$IMAGE" -- $OPTIONS
else
    # distrobox passes extra flags via --additional-flags
    distrobox create -n "$TOOLBOX_NAME" --image "$IMAGE" --additional-flags "$OPTIONS"
fi

# --- Cleanup: keep only the most recent image for this tag ---
repo="${IMAGE%:*}"

while read -r id ref dig; do
    if [[ "$id" != "$new_id" ]]; then
        $RUNTIME image rm -f "$id" >/dev/null 2>&1 || true
    fi
done < <($RUNTIME images --digests --format '{{.ID}} {{.Repository}}:{{.Tag}} {{.Digest}}' \
         | awk -v ref="$IMAGE" -v ndig="$new_digest" '$2==ref && $3!=ndig')

while read -r id; do
    $RUNTIME image rm -f "$id" >/dev/null 2>&1 || true
done < <($RUNTIME images --format '{{.ID}} {{.Repository}}:{{.Tag}}' \
         | awk -v r="$repo" '$2==r":<none>" {print $1}')
# --- end cleanup ---

echo "✅ $TOOLBOX_NAME refreshed (channel: $CHANNEL)"
