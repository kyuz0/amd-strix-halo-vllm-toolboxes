import subprocess
import time
import os

# Transport choices for the two-host RCCL cluster. "auto" resolves at runtime
# in order: native InfiniBand when an active IB-link-layer HCA is present, else
# RDMA/RoCE when an active Ethernet-link-layer RDMA port exists, else plain
# Ethernet. "ethernet" forces TCP sockets with the IB/RDMA data path disabled.
TRANSPORTS = ("auto", "infiniband", "roce", "ethernet")

TRANSPORT_LABELS = {
    "auto": "Auto-detect (InfiniBand, else RDMA/RoCE, else Ethernet)",
    "infiniband": "InfiniBand (native IB, GID index 0)",
    "roce": "RDMA/RoCE (Ethernet-based RDMA, GID index 1)",
    "ethernet": "Ethernet (TCP sockets, RDMA disabled)",
}


def active_rdma_ports():
    """Return [(device:port, link_layer)] for every ACTIVE RDMA port.

    link_layer is "InfiniBand" (native IB) or "Ethernet" (RoCE). Used both to
    pick the transport in "auto" mode and to warn when several rails exist.
    """
    import glob

    ports = []
    for state_path in sorted(glob.glob("/sys/class/infiniband/*/ports/*/state")):
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                state = fh.read().strip().upper()
        except OSError:
            continue
        if "ACTIVE" not in state:
            continue
        parts = state_path.split("/")
        # /sys/class/infiniband/<device>/ports/<port>/state
        dev = parts[-4]
        port = parts[-2]
        link_path = f"/sys/class/infiniband/{dev}/ports/{port}/link_layer"
        try:
            with open(link_path, "r", encoding="utf-8") as fh:
                actual = fh.read().strip()
        except OSError:
            actual = ""
        ports.append((f"{dev}:{port}", actual))
    return ports


def _active_rdma_port(link_layer):
    """Return '<device>:<port>' of the first ACTIVE port with the given
    link-layer type ("InfiniBand" for native IB, "Ethernet" for RoCE), or None."""
    for device_port, actual in active_rdma_ports():
        if link_layer.lower() in actual.lower():
            return device_port
    return None


def detect_ib_hca():
    """Return '<device>:<port>' of an active native InfiniBand HCA, or None."""
    return _active_rdma_port("InfiniBand")


def detect_roce_hca():
    """Return '<device>:<port>' of an active RoCE (Ethernet link-layer) RDMA
    port, or None."""
    return _active_rdma_port("Ethernet")


def resolve_transport(transport=None):
    """Resolve a transport choice (or the persisted one) to a concrete mode."""
    transport = (
        transport or os.getenv("VLLM_CLUSTER_TRANSPORT") or "auto"
    ).strip().lower()
    if transport in ("infiniband", "roce", "ethernet"):
        return transport
    # auto: prefer native InfiniBand, then RDMA/RoCE, then plain Ethernet
    if detect_ib_hca():
        return "infiniband"
    if detect_roce_hca():
        return "roce"
    return "ethernet"


def _transport_vars(transport, net_iface_expr, hca_expr):
    """Return [(key, value)] export pairs. Values may be shell expressions."""
    pairs = [
        ("NCCL_SOCKET_IFNAME", net_iface_expr),
        ("GLOO_SOCKET_IFNAME", net_iface_expr),
        ("NCCL_IB_TIMEOUT", "23"),
        ("NCCL_IB_RETRY_CNT", "7"),
        ("NCCL_NET_GDR_LEVEL", "0"),
    ]
    if transport == "ethernet":
        pairs.append(("NCCL_IB_DISABLE", "1"))
        return pairs
    pairs.append(("NCCL_IB_DISABLE", "0"))
    if transport == "infiniband":
        if hca_expr:
            pairs.append(("NCCL_IB_HCA", hca_expr))
        # Native IB uses the link-local GID (index 0); index 1 is RoCEv2-IPv4
        # only and does not exist on an IB fabric.
        pairs.append(("NCCL_IB_GID_INDEX", "0"))
        pairs.append(("NCCL_PROTO", "LL"))
        pairs.append(("NCCL_ALGO", "Ring"))
    else:  # roce (RDMA/RoCE)
        pairs.append(("NCCL_IB_GID_INDEX", "1"))
        if hca_expr:
            pairs.append(("NCCL_IB_HCA", hca_expr))
    return pairs


def transport_env(transport, net_iface):
    """Return the canonical NCCL/RCCL env dict for in-process use (vllm launch).

    HCA selection honors an explicit ``VLLM_IB_HCA`` / ``VLLM_ROCE_HCA``
    override, falling back to auto-detection. For RoCE the HCA is only pinned
    when it is ambiguous (several active RDMA rails) so single-NIC behavior is
    unchanged.
    """
    transport = resolve_transport(transport)
    hca = None
    if transport == "infiniband":
        hca = os.getenv("VLLM_IB_HCA") or detect_ib_hca()
    elif transport == "roce":
        hca = os.getenv("VLLM_ROCE_HCA")
        if not hca and len(active_rdma_ports()) > 1:
            # Multiple active rails: pin one so RCCL never sees a prefix that
            # matches more than one device (ncclCommInitRank "internal error").
            hca = detect_roce_hca()
    return {
        key: value
        for key, value in _transport_vars(transport, net_iface, hca)
        if value not in (None, "")
    }


def transport_script_exports(transport, net_iface_expr="$RDMA_IFACE"):
    """Return bash 'export' lines for a generated node-setup script.

    ``net_iface_expr`` defaults to the ``$RDMA_IFACE`` the script computes; for
    InfiniBand (and ambiguous multi-rail RoCE) the HCA is detected inside the
    script so per-node device names never have to match the head node's. An
    explicit ``VLLM_IB_HCA`` / ``VLLM_ROCE_HCA`` override wins over detection.
    """
    transport = resolve_transport(transport)
    if transport == "infiniband":
        override = os.getenv("VLLM_IB_HCA")
        hca_expr = (
            override if override else "$( _detect_rdma_hca 'InfiniBand' || true )"
        )
    elif transport == "roce":
        override = os.getenv("VLLM_ROCE_HCA")
        if override:
            hca_expr = override
        elif len(active_rdma_ports()) > 1:
            hca_expr = "$( _detect_rdma_hca 'Ethernet' || true )"
        else:
            hca_expr = None
    else:
        hca_expr = None
    lines = []
    for key, value in _transport_vars(transport, net_iface_expr, hca_expr):
        if value in (None, ""):
            continue
        lines.append(f"export {key}={value}")
    return "\n".join(lines)


def _ib_hca_detect_snippet():
    return r'''
# Detect the first ACTIVE RDMA port of a given link layer ("InfiniBand" for
# native IB, "Ethernet" for RoCE). Used by the infiniband / multi-rail roce
# transports so per-node device names never have to match the head node's.
_detect_rdma_hca() {
    local _kind="${1:-InfiniBand}"
    for _state in /sys/class/infiniband/*/ports/*/state; do
        [ -r "$_state" ] || continue
        grep -qi active "$_state" || continue
        _dev="${_state%%/ports/*}"
        _dev="${_dev##*/}"
        _port="$(basename "$(dirname "$_state")")"
        _ll="/sys/class/infiniband/${_dev}/ports/${_port}/link_layer"
        if [ -r "$_ll" ] && grep -qi "$_kind" "$_ll"; then
            echo "${_dev}:${_port}"
            return 0
        fi
    done
    return 1
}
'''


def warn_multi_rdma(transport):
    """Warn when several active RDMA ports exist so a silent pin is visible.

    Emits the candidate list plus the effective pin for the transport, and how
    to override it (``VLLM_IB_HCA`` / ``VLLM_ROCE_HCA``). No-op with <2 ports.
    """
    ports = active_rdma_ports()
    if len(ports) < 2:
        return
    active = [dp for dp, _ in ports]
    print(f"[cluster] {len(active)} active RDMA ports detected: {', '.join(active)}")
    if transport == "infiniband":
        hca = os.getenv("VLLM_IB_HCA") or detect_ib_hca()
        print(
            f"[cluster] InfiniBand transport will use HCA {hca}; "
            "set VLLM_IB_HCA to override"
        )
    elif transport == "roce":
        hca = os.getenv("VLLM_ROCE_HCA")
        if hca:
            print(f"[cluster] RDMA/RoCE transport will use HCA {hca} (from VLLM_ROCE_HCA)")
        else:
            print(
                "[cluster] RDMA/RoCE transport pins the first active RoCE rail; "
                "set VLLM_ROCE_HCA to choose another"
            )


def _transport_note(transport):
    if transport == "ethernet":
        return "Ethernet transport (NCCL_IB_DISABLE=1, TCP sockets)"
    if transport == "infiniband":
        return "InfiniBand transport (native IB, GID index 0)"
    return "RDMA/RoCE transport (GID index 1)"


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
    transport = resolve_transport()
    transport_label = TRANSPORT_LABELS.get(transport, transport)
    warn_multi_rdma(transport)
    nccl_debug_val = os.getenv("NCCL_DEBUG", "")
    
    script = f"""
    source /etc/profile
    # Silence the kill command
    ray stop --force > /dev/null 2>&1 || true

    # RayExecutorV2 applies driver model variables with setdefault. Keep the
    # daemon model-neutral so the selected model can supply its own AITER policy.
    unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR

{_ib_hca_detect_snippet()}
    # Calculate Interface dynamically
    RDMA_IFACE=$(ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1)
    
    echo "\\n--- Ray Worker Environment ({worker_ip}) ---"
    echo "transport={transport} ({transport_label})"

    export RAY_DISABLE_METRICS=1
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export TRITON_CACHE_DIR="$HOME/.cache/triton"
    mkdir -p "$TRITON_CACHE_DIR"
    export VLLM_HOST_IP={worker_ip}
    export RDMA_IFACE=$RDMA_IFACE
{transport_script_exports(transport)}
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
    echo "Note: {_transport_note(transport)}"
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
    transport = resolve_transport()
    transport_label = TRANSPORT_LABELS.get(transport, transport)
    warn_multi_rdma(transport)

    print(f"Setting up Head Node ({head_ip})...")
    
    nccl_debug_val = os.getenv("NCCL_DEBUG", "")
    
    script = f"""
    # Silence the kill command
    ray stop --force > /dev/null 2>&1 || true

    # RayExecutorV2 applies driver model variables with setdefault. Keep the
    # daemon model-neutral so the selected model can supply its own AITER policy.
    unset VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_LINEAR

{_ib_hca_detect_snippet()}
    # Calculate Interface dynamically
    RDMA_IFACE=$(ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1)
    
    echo "\\n--- Ray Head Environment ({head_ip}) ---"
    echo "transport={transport} ({transport_label})"

    export RAY_DISABLE_METRICS=1
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export TRITON_CACHE_DIR="$HOME/.cache/triton"
    mkdir -p "$TRITON_CACHE_DIR"
    export VLLM_HOST_IP={head_ip}
    export RDMA_IFACE=$RDMA_IFACE
{transport_script_exports(transport)}
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
    echo "Note: {_transport_note(transport)}"
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
