import subprocess
import time
import os

def get_net_iface(ip_prefix=None):
    """
    Auto-detects the interface that serves the cluster network.
    Assumes standard 192.168.100.x setup from start_vllm_cluster.py, but parameterizable.
    """
    if ip_prefix is None:
        head_ip = os.getenv("VLLM_HEAD_IP", "192.168.100.1")
        ip_prefix = ".".join(head_ip.split('.')[:3])
        
    try:
        # ip -o addr show | grep <ip_prefix>
        cmd = f"ip -o addr show | grep {ip_prefix}"
        res = subprocess.check_output(cmd, shell=True, text=True).strip()
        # Output format: 2: eth0    inet 192.168.100.1/24 ...
        parts = res.split()
        if len(parts) >= 2:
            return parts[1] # Interface name
    except:
        pass
    return "eth0" # Fallback

def get_local_ip(iface):
    try:
        cmd = f"ip -o -4 addr show {iface} | awk '{{print $4}}' | cut -d/ -f1"
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except:
        return "127.0.0.1"

def get_subnet_from_ip(ip):
    """Accurately gets the /24 subnet string for the given IP."""
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def stop_cluster(worker_ip=None, toolbox_name="vllm-therock-gfx1151"):
    """
    Stops Ray locally and on the worker node if provided.
    """
    print("Stopping Ray cluster locally...")
    subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if worker_ip:
        print(f"Stopping Ray cluster on worker ({worker_ip})...")
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", worker_ip,
            "toolbox", "run", "-c", toolbox_name, "--", "ray", "stop", "--force"
        ]
        try:
            subprocess.run(ssh_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to stop worker node completely: {e}")

def setup_worker_node(worker_ip, head_ip, toolbox_name):
    subnet = get_subnet_from_ip(worker_ip)
    
    # Read overrides from current env
    nccl_disable_val = os.getenv("NCCL_IB_DISABLE", "0")
    nccl_debug_val = os.getenv("NCCL_DEBUG", "")
    
    script = f"""
    source /etc/profile
    # Silence the kill command
    ray stop --force > /dev/null 2>&1 || true

    # RayExecutorV2 applies driver model variables with setdefault. Keep the
    # daemon model-neutral so the selected model can supply its own AITER policy.
    unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR
    
    # Calculate Interface dynamically
    RDMA_IFACE=$(ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1)
    
    echo "\\n--- Ray Worker Environment ({worker_ip}) ---"
    echo "export RAY_DISABLE_METRICS=1"
    echo "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1"
    echo "export RAY_memory_monitor_refresh_ms=0"
    echo "unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR"
    echo "export TRITON_CACHE_DIR=$HOME/.cache/triton"
    echo "export VLLM_HOST_IP={worker_ip}"
    echo "export RDMA_IFACE=$RDMA_IFACE"
    echo "export NCCL_SOCKET_IFNAME=$RDMA_IFACE"
    echo "export GLOO_SOCKET_IFNAME=$RDMA_IFACE"
    echo "export NCCL_IB_TIMEOUT=23"
    echo "export NCCL_IB_RETRY_CNT=7"
    echo "export NCCL_IB_DISABLE={nccl_disable_val}"
    
    export RAY_DISABLE_METRICS=1
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export TRITON_CACHE_DIR="$HOME/.cache/triton"
    mkdir -p "$TRITON_CACHE_DIR"
    export VLLM_HOST_IP={worker_ip}
    export RDMA_IFACE=$RDMA_IFACE
    export NCCL_SOCKET_IFNAME=$RDMA_IFACE
    export GLOO_SOCKET_IFNAME=$RDMA_IFACE
    # Stability for RDMA
    export NCCL_IB_TIMEOUT=23
    export NCCL_IB_RETRY_CNT=7
    export NCCL_IB_DISABLE={nccl_disable_val}
    """
    if nccl_debug_val:
        script += f"""
    echo "export NCCL_DEBUG={nccl_debug_val}"
    echo "export NCCL_DEBUG_SUBSYS=INIT,NET"
    export NCCL_DEBUG={nccl_debug_val}
    export NCCL_DEBUG_SUBSYS=INIT,NET
    """
    
    script += f"""
    echo "\\nStarting Ray Worker on {worker_ip} connecting to {head_ip}..."
    if [ "{nccl_disable_val}" = "1" ]; then
        echo "Note: Worker is configured with NCCL_IB_DISABLE=1 (Ethernet Forced)"
    fi
    ray start --address='{head_ip}:6379' --num-gpus=1 --num-cpus=8 --disable-usage-stats
    """
    
    print(f"Setting up Worker Node ({worker_ip})...")
    
    # Use bash -s to read script from stdin
    # Command: ssh user@host "toolbox run -c <selected toolbox> -- bash -s"
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", worker_ip,
        "toolbox", "run", "-c", toolbox_name, "--", "bash", "-s"
    ]
    
    try:
        subprocess.run(ssh_cmd, input=script.encode(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup worker: {e}")
        return False

def setup_head_node(head_ip):
    subnet = get_subnet_from_ip(head_ip)
    
    print(f"Setting up Head Node ({head_ip})...")
    
    # Read overrides from current env
    nccl_disable_val = os.getenv("NCCL_IB_DISABLE", "0")
    nccl_debug_val = os.getenv("NCCL_DEBUG", "")
    
    script = f"""
    # Silence the kill command
    ray stop --force > /dev/null 2>&1 || true

    # RayExecutorV2 applies driver model variables with setdefault. Keep the
    # daemon model-neutral so the selected model can supply its own AITER policy.
    unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR
    
    # Calculate Interface dynamically
    RDMA_IFACE=$(ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1)
    
    echo "\\n--- Ray Head Environment ({head_ip}) ---"
    echo "export RAY_DISABLE_METRICS=1"
    echo "export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1"
    echo "export RAY_memory_monitor_refresh_ms=0"
    echo "unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR"
    echo "export TRITON_CACHE_DIR=$HOME/.cache/triton"
    echo "export VLLM_HOST_IP={head_ip}"
    echo "export RDMA_IFACE=$RDMA_IFACE"
    echo "export NCCL_SOCKET_IFNAME=$RDMA_IFACE"
    echo "export GLOO_SOCKET_IFNAME=$RDMA_IFACE"
    echo "export NCCL_IB_TIMEOUT=23"
    echo "export NCCL_IB_RETRY_CNT=7"
    echo "export NCCL_IB_DISABLE={nccl_disable_val}"

    export RAY_DISABLE_METRICS=1
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export TRITON_CACHE_DIR="$HOME/.cache/triton"
    mkdir -p "$TRITON_CACHE_DIR"
    export VLLM_HOST_IP={head_ip}
    export RDMA_IFACE=$RDMA_IFACE
    export NCCL_SOCKET_IFNAME=$RDMA_IFACE
    export GLOO_SOCKET_IFNAME=$RDMA_IFACE
    # Stability for RDMA
    export NCCL_IB_TIMEOUT=23
    export NCCL_IB_RETRY_CNT=7
    export NCCL_IB_DISABLE={nccl_disable_val}
    """
    
    if nccl_debug_val:
        script += f"""
    echo "export NCCL_DEBUG={nccl_debug_val}"
    echo "export NCCL_DEBUG_SUBSYS=INIT,NET"
    export NCCL_DEBUG={nccl_debug_val}
    export NCCL_DEBUG_SUBSYS=INIT,NET
    """
    
    script += f"""
    echo "\\nStarting Ray Head on {head_ip}..."
    if [ "{nccl_disable_val}" = "1" ]; then
        echo "Note: Head is configured with NCCL_IB_DISABLE=1 (Ethernet Forced)"
    fi
    ray start --head --port=6379 --node-ip-address={head_ip} --num-gpus=1 --num-cpus=8 --disable-usage-stats --include-dashboard=false
    """
    
    try:
        # Run locally
        subprocess.run(["bash", "-s"], input=script.encode(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup head: {e}")
        return False

def get_ray_nodes():
    """Returns a list of active Ray node IPs."""
    try:
        res = subprocess.run(["ray", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return []
            
        nodes = []
        in_active_section = False
        import re
        for line in res.stdout.splitlines():
            if "Active:" in line:
                in_active_section = True
                continue
            if "Pending:" in line or "Recent failures:" in line:
                in_active_section = False
            
            if in_active_section:
                # Match "1 node_<ID_OR_IP>"
                # We relax regex to accept hex IDs or IPs
                match = re.search(r"node_([a-zA-Z0-9\.\-_]+)", line)
                if match:
                    nodes.append(match.group(1))

                
        return nodes
    except:
        return []

def check_ray_status():
    """Returns (active_nodes, total_gpus) parsing 'ray status' output roughly."""
    nodes = get_ray_nodes()
    # Assume 1 GPU per node for now as per strix halo setup
    return len(nodes), len(nodes)

def wait_for_cluster(expected_nodes=2, timeout=60):
    print(f"Waiting for Ray cluster to initialize (expecting {expected_nodes} nodes)...")
    for i in range(timeout):
        nodes, gpus = check_ray_status()
        if i % 5 == 0:
             print(f"Check {i}/{timeout}: Active Nodes={nodes}")
        if nodes >= expected_nodes:
            print("Cluster is Ready!")
            time.sleep(2)
            return True
        time.sleep(1)
        
    print("Timeout waiting for cluster.")
    return False

def nuke_vllm_cache_on_node(ip, is_local=False):
    """Clears vLLM cache on a specific node."""
    cmd_str = f"Locally" if is_local else f"on {ip}"
    print(f"Clearing vLLM cache {cmd_str}...", end="", flush=True)
    
    try:
        if is_local:
            from pathlib import Path
            cache = Path.home() / ".cache" / "vllm"
            if cache.exists():
                subprocess.run(["rm", "-rf", str(cache)], check=True)
                cache.mkdir(parents=True, exist_ok=True)
        else:
            # Remote SSH
            ssh_cmd = [
                "ssh", "-o", "StrictHostKeyChecking=no", ip,
                "rm -rf ~/.cache/vllm && mkdir -p ~/.cache/vllm"
            ]
            subprocess.run(ssh_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        print(" Done.")
    except Exception as e:
        print(f" Failed ({e}).")

def nuke_vllm_cache_cluster(nodes=None):
    """
    Clears vLLM cache on cluster nodes.
    If 'nodes' (list of IPs) is provided, uses those.
    Otherwise attempts to discover from ray status (which may fail if status shows Hex IDs and not IPs).
    """
    if nodes is None:
        nodes = get_ray_nodes()
    
    # Check if nodes look like IPs before trying SSH
    # If we only have Hex IDs, we can't SSH unless we map them.
    # For now, we filter for things that look like IPs if we are relying on discovery
    # But if user passed explicit list, we assume they are IPs.
    
    rdma_iface = get_net_iface()
    local_ip = get_local_ip(rdma_iface)
    
    if not nodes:
        # Fallback to just local?
        nuke_vllm_cache_on_node(local_ip, is_local=True)
        return

    import re
    ip_pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

    for node_ip in nodes:
        # If discovered node is NOT an IP (e.g. Hex ID), we warn and skip remote nuke
        # unless it is '127.0.0.1' or we can determine it is local.
        
        is_ip = ip_pattern.match(node_ip) or node_ip == "localhost"
        
        if not is_ip:
            # Maybe it's a Hex ID. We can't SSH to a Hex ID.
            print(f"Skipping cache clear on '{node_ip}' (Not an IP address).")
            continue
            
        is_local = (node_ip == local_ip) or (node_ip == "127.0.0.1")
        nuke_vllm_cache_on_node(node_ip, is_local)

    time.sleep(2)
