#!/bin/bash

# Defaults
HEAD_IP="192.168.100.1"
WORKER_IP="192.168.100.2"
MODE=""

# Help function
usage() {
    echo "Usage: $0 [mode] [options]"
    echo "Modes:"
    echo "  head      Configure and start Ray head node"
    echo "  worker    Configure and start Ray worker node"
    echo "  run-vllm  Run vLLM serve"
    echo ""
    echo "Options:"
    echo "  --head-ip <ip>    Set Head Node IP (default: 192.168.100.1)"
    echo "  --worker-ip <ip>  Set Worker Node IP (default: 192.168.100.2)"
    echo "  -h, --help        Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        head|worker|run-vllm)
            MODE="$1"
            shift
            ;;
        --head-ip)
            HEAD_IP="$2"
            shift 2
            ;;
        --worker-ip)
            WORKER_IP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$MODE" ]; then
    usage
fi

setup_head() {
    echo "Configuring Head Node..."
    ray stop --force

    # Ray workers must not inherit model-specific AITER policy. vLLM propagates
    # the selected model's values from the driver after the actors are created.
    unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR

    # Critical Config
    export RAY_DISABLE_METRICS=1
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export VLLM_HOST_IP=$HEAD_IP
    # Dynamic interface detection based on subnet of HEAD_IP
    SUBNET=$(echo $HEAD_IP | awk -F. '{print $1"."$2"."$3".0/24"}')
    export RDMA_IFACE=$(ip -o addr show to $SUBNET | awk '{print $2}' | head -n1)
    
    if [ -z "$RDMA_IFACE" ]; then
        echo "Warning: Could not detect interface for $SUBNET. Defaulting to eth0."
        export RDMA_IFACE="eth0"
    fi

    export NCCL_SOCKET_IFNAME=$RDMA_IFACE
    export GLOO_SOCKET_IFNAME=$RDMA_IFACE

    echo "Starting Ray Head on $HEAD_IP (Interface: $RDMA_IFACE)..."
    ray start --head --port=6379 --node-ip-address=$HEAD_IP --num-gpus=1 --num-cpus=8 --disable-usage-stats
}

setup_worker() {
    echo "Configuring Worker Node..."
    ray stop --force

    # Ray workers must not inherit model-specific AITER policy. vLLM propagates
    # the selected model's values from the driver after the actors are created.
    unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR

    # Critical Config
    export RAY_DISABLE_METRICS=1
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export VLLM_HOST_IP=$WORKER_IP
    # Dynamic interface detection based on subnet of WORKER_IP
    SUBNET=$(echo $WORKER_IP | awk -F. '{print $1"."$2"."$3".0/24"}')
    export RDMA_IFACE=$(ip -o addr show to $SUBNET | awk '{print $2}' | head -n1)

    if [ -z "$RDMA_IFACE" ]; then
        echo "Warning: Could not detect interface for $SUBNET. Defaulting to eth0."
        export RDMA_IFACE="eth0"
    fi

    export NCCL_SOCKET_IFNAME=$RDMA_IFACE
    export GLOO_SOCKET_IFNAME=$RDMA_IFACE

    echo "Starting Ray Worker on $WORKER_IP connecting to $HEAD_IP:6379 (Interface: $RDMA_IFACE)..."
    ray start --address="$HEAD_IP:6379" --num-gpus=1 --num-cpus=8 --disable-usage-stats
}

run_vllm() {
    echo "Running vLLM..."
    
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export VLLM_HOST_IP=$HEAD_IP
    
    # Dynamic interface detection based on subnet of HEAD_IP
    SUBNET=$(echo $HEAD_IP | awk -F. '{print $1"."$2"."$3".0/24"}')
    export NCCL_SOCKET_IFNAME=$(ip -o addr show to $SUBNET | awk '{print $2}' | head -n1)
    
    if [ -z "$NCCL_SOCKET_IFNAME" ]; then
        echo "Warning: Could not detect interface for $SUBNET. Defaulting to eth0."
        export NCCL_SOCKET_IFNAME="eth0"
    fi
    
    export NCCL_IB_GID_INDEX=1
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=0

    echo "Launching vLLM Serve..."
    vllm serve facebook/opt-125m \
      --tensor-parallel-size 2 \
      --distributed-executor-backend ray \
      --trust-remote-code \
      --enforce-eager \
      --gpu-memory-utilization 0.90
}

# Execute Mode
if [ "$MODE" == "head" ]; then
    setup_head
elif [ "$MODE" == "worker" ]; then
    setup_worker
elif [ "$MODE" == "run-vllm" ]; then
    run_vllm
fi
