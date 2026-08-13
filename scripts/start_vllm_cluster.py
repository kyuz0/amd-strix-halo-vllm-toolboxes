#!/usr/bin/env python3
import sys
import os
import json
import shutil
import tempfile
import subprocess
import time
from pathlib import Path

# Add benchmarks dir to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR = SCRIPT_DIR.parent / "benchmarks"
OPT_DIR = Path("/opt")


# Check /opt first (Container), then local fallback
if (OPT_DIR / "run_vllm_bench.py").exists():
    sys.path.append(str(OPT_DIR))
else:
    sys.path.append(str(BENCH_DIR))
    sys.path.append(str(SCRIPT_DIR))

try:
    import models
    MODEL_TABLE = models.MODEL_TABLE
    MODELS_TO_RUN = models.MODELS_TO_RUN
except ImportError:
    print("Error: Could not import models.py config.")
    sys.exit(1)

if (OPT_DIR / "max_context_results.json").exists():
    RESULTS_FILE = OPT_DIR / "max_context_results.json"
else:
    RESULTS_FILE = BENCH_DIR / "max_context_results.json"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")

ATTENTION_BACKENDS = (
    "TRITON_ATTN",
    "ROCM_ATTN",
    "ROCM_AITER_UNIFIED_ATTN",
)

REMOTE_TOOLBOXES = (
    ("vllm-therock-gfx1151", "latest"),
    ("vllm-therock-gfx1151-dev", "dev"),
)

def get_discovered_models():
    """
    Overrides the hardcoded MODELS_TO_RUN by looking at what we actually have results for.
    """
    # Bypass verification check for Cluster Launcher
    # We want to see ALL models, including those that require TP > 1 (which find_max_context might have skipped)
    return MODELS_TO_RUN

# Refresh the list of models to run based on what we found
MODELS_TO_RUN = get_discovered_models()

def check_dependencies():
    missing = []
    if not shutil.which("dialog"):
        missing.append("dialog")
    if not shutil.which("ssh"):
        missing.append("ssh")
    if not shutil.which("ray"):
        missing.append("ray")
        
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}.")
        print("Please install them (e.g., sudo dnf install dialog openssh-clients).")
        print("Ensure 'ray' is in your PATH (pip install ray).")
        sys.exit(1)

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def show_info(title, msg):
    run_dialog(["--title", title, "--msgbox", msg, "12", "60"])


# Import Shared Cluster Manager
try:
    import cluster_manager
except ImportError:
    # Try importing from current directory if script is run directly
    sys.path.append(str(Path(__file__).parent))
    import cluster_manager

# Delegate Functions to Cluster Manager
def get_subnet_from_ip(ip):
    return cluster_manager.get_subnet_from_ip(ip)

def check_ray_status():
    return cluster_manager.check_ray_status()

def wait_for_cluster():
    return cluster_manager.wait_for_cluster()

def nuke_vllm_cache(head_ip):
    # Only nukes local cache on the head node for now, or use cluster nuke?
    # The original script just did local nuke.
    # cluster_manager has nuke_vllm_cache_on_node and nuke_vllm_cache_cluster
    # Let's use the local ip one effectively
    prefix = ".".join(head_ip.split('.')[:3])
    rdma = cluster_manager.get_net_iface(prefix)
    local = cluster_manager.get_local_ip(rdma)
    cluster_manager.nuke_vllm_cache_on_node(local, is_local=True)

def setup_worker_node(worker_ip, head_ip, toolbox_name):
    return cluster_manager.setup_worker_node(worker_ip, head_ip, toolbox_name)

def setup_head_node(head_ip):
    return cluster_manager.setup_head_node(head_ip)


def get_verified_config(model_id, tp_size, max_seqs):
    """Reads max_context_results.json."""
    model_ctx_default = MODEL_TABLE[model_id].get("ctx", "auto")
    default_config = {
        "ctx": model_ctx_default,
        "util": 0.90
    }
    
    if not RESULTS_FILE.exists():
        return default_config

    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
            
        matches = [r for r in data 
                  if r["model"] == model_id 
                  and r["tp"] == tp_size 
                  and r["max_seqs"] == max_seqs 
                  and r["status"] == "success"]
        
        if not matches:
            return default_config
            
        matches.sort(key=lambda x: (float(x["util"]), x["max_context_1_user"]), reverse=True)
        best = matches[0]
        # Cap util to 0.90 max. Due to recent changes in vLLM/ROCm UMA
        # available memory calculations (psutil.virtual_memory().available vs hipMemGetInfo),
        # 0.95 often leads to OOM at startup on Strix Halo APUs.
        util = float(best["util"])
        return {
            "ctx": best["max_context_1_user"],
            "util": min(0.90, util)
        }
        
    except Exception as e:
        return default_config

def configure_and_launch_vllm(model_idx, head_ip, remote_toolbox):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]
    name = model_id.split("/")[-1]
    
    # Defaults
    current_tp = 2 # Forced default for Cluster
    current_seqs = 1
    
    # Lookup Config
    verified = get_verified_config(model_id, current_tp, current_seqs if isinstance(current_seqs, int) else 1)
    current_ctx = verified["ctx"]
    current_util = verified["util"]
    
    clear_cache = True  # Default ON: stale graphs from version upgrades cause crashes
    # Default to eager mode for stability in cluster situations, unless explicitly disabled
    use_eager = config.get("enforce_eager", True)
    trust_remote = True # Default True as per request
    configured_attn_backend = config.get("attention_backend", "TRITON_ATTN")
    attention_backend_locked = configured_attn_backend is None
    if attention_backend_locked:
        current_attn_backend = config.get(
            "attention_backend_label", "Model-specific"
        )
        attn_backends = [current_attn_backend]
    else:
        if configured_attn_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unsupported attention backend {configured_attn_backend!r} "
                f"for {model_id}"
            )
        attn_backends = list(ATTENTION_BACKENDS)
        current_attn_backend = configured_attn_backend
    current_extra_flags = list(config.get("extra_flags", []))  # Copy so edits don't mutate config

    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"
        trust_status = "YES" if trust_remote else "NO"

        extra_flags_display = ' '.join(current_extra_flags) if current_extra_flags else '(none)'
        # Truncate display for menu readability
        extra_flags_short = (extra_flags_display[:40] + '...') if len(extra_flags_display) > 43 else extra_flags_display

        menu_args = [
            "--clear", "--backtitle",
            f"AMD VLLM CLUSTER Launcher (Head: {head_ip}, Worker: {remote_toolbox})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "24", "70", "11",
            "1", f"Tensor Parallelism:   {current_tp} (Fixed)",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx}",
            "4", f"GPU Utilization:      {current_util}",
            "5", f"Trust Remote Code:    {trust_status}",
            "6", f"Attention Backend:    {current_attn_backend}",
            "7", f"Erase vLLM Cache:     {cache_status}",
            "8", f"Force Eager Mode:     {eager_status}",
            "9", f"Extra vLLM Flags:     {extra_flags_short}",
            "10", "LAUNCH SERVER"
        ]
        
        choice = run_dialog(menu_args)
        if not choice: return False
        
        if choice == "1":
            # TP Selection - Allow change but warn?
             new_tp = run_dialog([
                "--title", "Tensor Parallelism",
                "--rangebox", "Set TP Size:", "10", "40", "1", "8", str(current_tp)
            ])
             if new_tp: current_tp = int(new_tp)
             
        elif choice == "2":
            new_seqs = run_dialog([
                "--title", "Concurrent Requests",
                "--inputbox", "Enter Max Concurrent Requests (or 'auto'):", "10", "40", str(current_seqs)
            ])
            if new_seqs: 
                if new_seqs.lower().strip() == "auto":
                    current_seqs = "auto"
                else:
                    try:
                        current_seqs = int(new_seqs)
                    except ValueError:
                        pass
            
        elif choice == "3":
            new_ctx = run_dialog([
                "--title", "Context Length",
                "--inputbox", f"Enter Context Length (or 'auto'):", "10", "40", str(current_ctx)
            ])
            if new_ctx:
                if new_ctx.lower().strip() == "auto":
                    current_ctx = "auto"
                else:
                    try:
                        current_ctx = int(new_ctx)
                    except ValueError:
                        pass

        elif choice == "4":
             new_util = run_dialog([
                "--title", "GPU Utilization",
                "--inputbox", "Enter GPU Utilization (0.1 - 1.0):", "10", "40", str(current_util)
            ])
             if new_util: current_util = float(new_util)
             
        elif choice == "5":
            trust_remote = not trust_remote

        elif choice == "6":
            if attention_backend_locked:
                run_dialog([
                    "--title", "Attention Backend",
                    "--msgbox",
                    f"{current_attn_backend}\n\n"
                    "This model provides its own attention implementation, so "
                    "start-vllm-cluster does not pass --attention-backend.",
                    "10", "68"
                ])
            else:
                idx = attn_backends.index(current_attn_backend)
                current_attn_backend = attn_backends[(idx + 1) % len(attn_backends)]

        elif choice == "7":
            clear_cache = not clear_cache

        elif choice == "8":
            use_eager = not use_eager

        elif choice == "9":
            # Edit Extra vLLM Flags
            current_str = ' '.join(current_extra_flags)
            new_flags = run_dialog([
                "--title", "Extra vLLM Flags",
                "--inputbox",
                "Edit extra flags (space-separated, passed directly to vllm serve).\n"
                "Clear the field to remove all extra flags.",
                "12", "70", current_str
            ])
            if new_flags is not None:  # None = cancelled
                current_extra_flags = new_flags.split() if new_flags.strip() else []

        elif choice == "10":
            break
            
    # Build Command
    subprocess.run(["clear"])
    
    if clear_cache:
        nuke_vllm_cache(head_ip)
    
    # Environment Setup
    # We need to set these variables in the current process before exec or pass them in env
    subnet = get_subnet_from_ip(head_ip)
    
    # Compute RDMA IFACE dynamically
    # Note: we need to run logical command to get the iface name
    try:
        iface_cmd = f"ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1"
        rdma_iface = subprocess.check_output(iface_cmd, shell=True, text=True).strip()
    except:
        rdma_iface = "eth0" # Fallback
        print("Warning: Could not detect RDMA IFACE, defaulting to eth0")

    print(f"Detected RDMA Interface: {rdma_iface}")
    
    env = os.environ.copy()
    env.pop("VLLM_ROCM_USE_AITER", None)
    env.update(config.get("env", {}))
    env["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
    env["VLLM_HOST_IP"] = head_ip
    env["NCCL_SOCKET_IFNAME"] = rdma_iface
    env["NCCL_IB_GID_INDEX"] = "1"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    
    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--gpu-memory-utilization", str(current_util),
        "--distributed-executor-backend", "ray",
        "--dtype", "auto"
    ]

    if not attention_backend_locked:
        cmd.extend(["--attention-backend", current_attn_backend])

    cmd.extend(["--mm-encoder-attn-backend", "TRITON_ATTN"])
            
    if str(current_seqs) != "auto":
        cmd.extend(["--max-num-seqs", str(current_seqs)])
        
    if str(current_ctx) != "auto":
        cmd.extend(["--max-model-len", str(current_ctx)])
    
    if trust_remote: cmd.append("--trust-remote-code")
    if use_eager: cmd.append("--enforce-eager")

    # Extra vLLM flags (from models.py defaults + user edits)
    if current_extra_flags:
        cmd.extend(current_extra_flags)
    
    print("\n" + "="*60)
    print(f" Launching VLLM Cluster on Head: {head_ip}")
    print(f" Worker:    {remote_toolbox}")
    print(f" Model:     {name}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx}")
    print(f" Backend:   {current_attn_backend}")
    if use_eager:
        print(" Note:      Eager Mode Enabled (Recommended for Cluster Stability)")
    if current_extra_flags:
        print(f" Extras:    {' '.join(current_extra_flags)}")
        
    print("\n --- Environment Variables ---")
    vars_to_print = [
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "VLLM_HOST_IP",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_GID_INDEX",
        "NCCL_NET_GDR_LEVEL"
    ]
    for key in config.get("env", {}):
        if key not in vars_to_print:
            vars_to_print.append(key)
        
    for k in vars_to_print:
        if k in env:
            print(f" export {k}={env[k]}")
            
    print(f"\n Command:   {' '.join(cmd)}")
    print("="*60 + "\n")
    
    # Exec
    os.execvpe("vllm", cmd, env)

def setup_ips_dialog(current_head, current_worker):
    # Using a form to edit both IPs
    # label y x item y x flen ilen
    form_args = [
        "--title", "Cluster Configuration",
        "--form", "Enter IP addresses for Head and Worker nodes:",
        "10", "60", "2",
        "Head Node IP:", "1", "1",  current_head, "1", "20", "20", "0",
        "Worker Node IP:", "2", "1", current_worker, "2", "20", "20", "0"
    ]
    
    result = run_dialog(form_args)
    if not result:
        return None
        
    lines = result.splitlines()
    if len(lines) >= 2:
        return lines[0].strip(), lines[1].strip()
    return None


def main():
    check_dependencies()
    
    # Default IPs
    head_ip = os.getenv("VLLM_HEAD_IP", "192.168.100.1")
    worker_ip = os.getenv("VLLM_WORKER_IP", "192.168.100.2")
    remote_toolbox = os.getenv("VLLM_REMOTE_TOOLBOX", REMOTE_TOOLBOXES[0][0])
    
    while True:
        # Main Menu
        # 1. Select worker toolbox
        # 2. Configure IPs
        # 3. Start Cluster (Ray)
        # 4. Stop Ray Cluster
        # 5. Ray Cluster Status
        # 6. Launch VLLM Serve
        # 7. Exit
        
        choice = run_dialog([
            "--clear", "--backtitle", "AMD VLLM RCCL Cluster Manager",
            "--title", "Main Menu",
            "--menu", "Select Action:", "18", "76", "7",
            "1", f"Select Target Toolbox ({remote_toolbox})",
            "2", f"Configure IPs (Head: {head_ip}, Worker: {worker_ip})",
            "3", "Start Ray Cluster",
            "4", "Stop Ray Cluster",
            "5", "Ray Cluster Status",
            "6", "Launch VLLM Serve",
            "7", "Exit"
        ])
        
        if not choice or choice == "7":
            subprocess.run(["clear"])
            sys.exit(0)
            
        if choice == "1":
            toolbox_items = []
            for toolbox_name, channel in REMOTE_TOOLBOXES:
                toolbox_items.extend([toolbox_name, channel])
            selected_toolbox = run_dialog([
                "--title", "Select Target Toolbox",
                "--default-item", remote_toolbox,
                "--menu", "Choose the toolbox on the worker host:",
                "12", "68", str(len(REMOTE_TOOLBOXES)),
            ] + toolbox_items)
            if selected_toolbox:
                remote_toolbox = selected_toolbox

        elif choice == "2":
            res = setup_ips_dialog(head_ip, worker_ip)
            if res:
                head_ip, worker_ip = res

        elif choice == "3":
            force_ethernet = False
            enable_nccl_debug = False
            
            while True:
                eth_status = "YES" if force_ethernet else "NO"
                debug_status = "YES" if enable_nccl_debug else "NO"
                
                c_choice = run_dialog([
                    "--clear", "--backtitle", "AMD VLLM RCCL Cluster Manager",
                    "--title", "Cluster Network Configuration",
                    "--menu", f"Worker toolbox: {remote_toolbox}", "15", "72", "3",
                    "1", f"Force Ethernet (Disable RDMA/RoCE):  {eth_status}",
                    "2", f"Enable NCCL Debug Logging:           {debug_status}",
                    "3", "START CLUSTER"
                ])
                if not c_choice: break
                
                if c_choice == "1":
                    force_ethernet = not force_ethernet
                elif c_choice == "2":
                    enable_nccl_debug = not enable_nccl_debug
                elif c_choice == "3":
                    os.environ["NCCL_IB_DISABLE"] = "1" if force_ethernet else "0"
                    if enable_nccl_debug:
                        os.environ["NCCL_DEBUG"] = "INFO"
                        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
                    else:
                        os.environ.pop("NCCL_DEBUG", None)
                        os.environ.pop("NCCL_DEBUG_SUBSYS", None)
                    
                    subprocess.run(["clear"])
                    print("= Starting Ray Cluster Setup =")
                    # 1. Start Head
                    if setup_head_node(head_ip):
                        print("Head node started successfully. Waiting 5s before worker connection...")
                        time.sleep(5)
                        # 2. Start Worker
                        if setup_worker_node(worker_ip, head_ip, remote_toolbox):
                            # 3. Wait for full cluster
                            wait_for_cluster()
                    input("Press Enter to continue...")
                    break

            print("= Ray Cluster Status =")
            subprocess.run(["ray", "status"])
            input("\nPress Enter to continue...")
            
        elif choice == "4":
            subprocess.run(["clear"])
            print("= Stopping Ray Cluster =")
            cluster_manager.stop_cluster(worker_ip, remote_toolbox)
            input("\nPress Enter to continue...")
            
        elif choice == "5":
            subprocess.run(["clear"])
            print("= Ray Cluster Status =")
            res = subprocess.run(["ray", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                print("\n[!] Cluster is Offline or Unreachable.")
                print("Please start the cluster first via Option 3 (Start Ray Cluster).")
            else:
                print(res.stdout)
            input("\nPress Enter to continue...")
            
        elif choice == "6":
            # Select Model
            menu_items = []
            for i, m_id in enumerate(MODELS_TO_RUN):
                name = m_id.split("/")[-1]
                menu_items.extend([str(i), name])
                
            m_choice = run_dialog([
                "--title", "Select Model",
                "--menu", "Choose a model to serve:", "20", "60", "10"
            ] + menu_items)
            
            if m_choice:
                configure_and_launch_vllm(int(m_choice), head_ip, remote_toolbox)
                # Note: execvpe replaces process, so we won't return here.

if __name__ == "__main__":
    main()
