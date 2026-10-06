#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, argparse, re, shutil
from pathlib import Path

try:
    import bench_utils
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    import bench_utils

# Import models immediately to access globals
try:
    import models
except ImportError:
    # If in /opt, this should work if path includes ., otherwise:
    sys.path.append(os.getcwd())
    try:
        import models
        # Also try parent/scripts for local dev if above failed?
    except ImportError:
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        import models

# =========================
# ⚙️ GLOBAL SETTINGS
# =========================

# CLUSTER CONFIG: 2x Strix Halo (TP=2)
# User requested specifically to test with TP=2 on the cluster.
CLUSTER_TP = 2
GPU_UTIL = "0.90" 
FORCE_ETH = False
FORCE_DEBUG_NCCL = False

# THROUGHPUT CONFIG (Imported from models.py)
OFF_NUM_PROMPTS      = models.OFF_NUM_PROMPTS
OFF_FORCED_OUTPUT    = models.OFF_FORCED_OUTPUT
DEFAULT_BATCH_TOKENS = models.DEFAULT_BATCH_TOKENS

RESULTS_DIR = Path("~/vllm_benchmark_results").expanduser()
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Reuse the model table from the main benchmark script
# We can just import it or copy it. Importing is cleaner but might rely on path.
# For standalone robustness, I will copy the minimal needed config or import if possible.
# Since this is a new file in root/benchmarks? No, likely scripts/ or same dir.
# Let's assume it's in the same dir as run_vllm_bench.py.


MODEL_TABLE = models.MODEL_TABLE
MODELS_TO_RUN = models.MODELS_TO_RUN


# =========================
# UTILS (Adapted for Cluster)
# =========================


# =========================
# CLUSTER MANAGER INTEGRATION
# =========================
try:
    import cluster_manager
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    import cluster_manager

# Defaults for Cluster
HEAD_IP = os.getenv("VLLM_HEAD_IP", "192.168.100.1")
WORKER_IP = os.getenv("VLLM_WORKER_IP", "192.168.100.2")

def log(msg): print(f"\n[CLUSTER-BENCH] {msg}")

def restart_cluster():
    log("Restarting Ray Cluster (Clean State)...")
    
    # Push config to env so cluster_manager picks it up for daemon injection
    os.environ["NCCL_IB_DISABLE"] = "1" if FORCE_ETH else "0"
    if FORCE_DEBUG_NCCL:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    else:
        os.environ.pop("NCCL_DEBUG", None)
        os.environ.pop("NCCL_DEBUG_SUBSYS", None)
        
    # 1. Stop Cluster (Best Effort)
    cluster_manager.stop_cluster()
    
    # 2. Start Head
    if not cluster_manager.setup_head_node(HEAD_IP):
        log("ERROR: Failed to start HEAD node.")
        sys.exit(1)
        
    # 3. Start Worker
    # Give head a moment
    time.sleep(5)
    if not cluster_manager.setup_worker_node(WORKER_IP, HEAD_IP):
        log("ERROR: Failed to start WORKER node.")
        sys.exit(1)
        
    # 4. Wait
    if not cluster_manager.wait_for_cluster():
        log("ERROR: Cluster failed to initialize.")
        sys.exit(1)
        
    log("Cluster Ready.")

def get_net_iface():
    prefix = ".".join(HEAD_IP.split('.')[:3])
    return cluster_manager.get_net_iface(prefix)

def get_local_ip(iface):
    return cluster_manager.get_local_ip(iface)

def nuke_vllm_cache():
    # We use explicit IPs because ray status might return Hex IDs which we can't SSH to.
    cluster_manager.nuke_vllm_cache_cluster(nodes=[HEAD_IP, WORKER_IP])


def get_dataset():
    # Same as original
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists(): return str(data_path)
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(data_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None

def get_cluster_env():
    # Detect Interface and IP
    rdma_iface = get_net_iface()
    host_ip = get_local_ip(rdma_iface)
    
    env = os.environ.copy()
    env.pop("VLLM_ROCM_USE_AITER", None)
    
    # Critical Cluster Envs (Match start_vllm_cluster.py)
    env["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
    env["VLLM_HOST_IP"] = host_ip
    env["NCCL_SOCKET_IFNAME"] = rdma_iface
    env["GLOO_SOCKET_IFNAME"] = rdma_iface
    # RCCL specific
    env["NCCL_IB_GID_INDEX"] = "1"
    env["NCCL_IB_DISABLE"] = "1" if FORCE_ETH else "0"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    
    # Stability for RDMA (Fix for high-throughput models like Gemma 3)
    env["NCCL_IB_TIMEOUT"] = "23"  # ~32 seconds (default is 18/~1s)
    env["NCCL_IB_RETRY_CNT"] = "7" # Default is 3, increase for lossy networks
    
    if FORCE_DEBUG_NCCL:
        env["NCCL_DEBUG"] = "INFO"
        env["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
    
    return env

def get_model_args(model, overrides=None):
    config = MODEL_TABLE.get(model, {})
    overrides = overrides or {}
    util = overrides.get("gpu_util", config.get("gpu_util", GPU_UTIL))
    max_seq_override = overrides.get("max_num_seqs", config.get("max_num_seqs", "16"))

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", str(util),
        "--dtype", "auto",
        "--tensor-parallel-size", str(CLUSTER_TP),
        "--max-num-seqs", str(max_seq_override),
        "--distributed-executor-backend", "ray"
    ]
    
    # Optional ctx
    ctx_override = overrides.get("ctx", config.get("ctx"))
    if ctx_override:
        cmd.extend(["--max-model-len", str(ctx_override)])
        
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    
    # Force eager mode for cluster stability
    cmd.append("--enforce-eager")
    
    return cmd

def get_benchmark_output_file(model, output_dir, tag=""):
    model_safe = model.replace("/", "_")
    output_dir_path = Path(output_dir)
    eth_suffix = "_eth" if FORCE_ETH else ""
    tag_suffix = f"_{tag}" if tag else ""
    return output_dir_path / f"{model_safe}_cluster_tp{CLUSTER_TP}{eth_suffix}{tag_suffix}_throughput.json"

def run_bench_set(model, backend_name, output_dir, extra_env=None, overrides=None):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    
    tag = overrides.get("tag", "").strip()
    output_file = get_benchmark_output_file(model, output_dir, tag)
    
    if output_file.exists():
        log(f"SKIP {model} [{backend_name}] (Result exists)")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    batch_tokens = str(overrides.get("max_tokens", MODEL_TABLE.get(model, {}).get("max_tokens", DEFAULT_BATCH_TOKENS)))

    log(f"START {model} [TP={CLUSTER_TP} | {backend_name}]...")
    
    nuke_vllm_cache()

    vllm_path = shutil.which("vllm") or "vllm"
    cmd = ["python", "-W", "ignore", vllm_path, "bench", "throughput"] + get_model_args(model, overrides)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    # Explicitly set Attention Backend for every run
    if backend_name == "AITER-Attn":
        cmd.extend(["--attention-backend", "ROCM_AITER_FA"])
    elif backend_name == "ROCm-Attn":
        cmd.extend(["--attention-backend", "ROCM_ATTN"])
    else:
        cmd.extend(["--attention-backend", "TRITON_ATTN"])

    cmd.extend(["--mm-encoder-attn-backend", "TRITON_ATTN"])

    env = get_cluster_env()
    
    # Model specific envs
    model_env = models.get_model_env(MODEL_TABLE[model], CLUSTER_TP)
    env.update(model_env)
    
    # Run specific envs (e.g. ROCm attention)
    if extra_env:
        env.update(extra_env)

    try: 
        log(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        log(f"ERROR: Failed {model} [{backend_name}] (Exit {e.returncode})")
    except Exception as e:
        log(f"ERROR: System error: {e}")

def run_cluster_throughput(model, overrides=None):
    overrides = overrides or {}
    tag = overrides.get("tag", "").strip()
    
    # 1. Triton Attention (explicit)
    if get_benchmark_output_file(model, RESULTS_DIR / "triton", tag).exists():
        log(f"SKIP {model} [Triton-Attn] (Result exists)")
    else:
        restart_cluster()
        run_bench_set(
            model, 
            "Triton-Attn", 
            RESULTS_DIR / "triton",
            overrides=overrides
        )
    
    # 2. ROCm Attention Run
    if get_benchmark_output_file(model, RESULTS_DIR / "rocm", tag).exists():
        log(f"SKIP {model} [ROCm-Attn] (Result exists)")
    else:
        restart_cluster()
        run_bench_set(
            model,
            "ROCm-Attn",
            RESULTS_DIR / "rocm",
            extra_env={},
            overrides=overrides
        )

    # 3. AITER Attention Run
    if get_benchmark_output_file(model, RESULTS_DIR / "aiter", tag).exists():
        log(f"SKIP {model} [AITER-Attn] (Result exists)")
    else:
        restart_cluster()
        run_bench_set(
            model,
            "AITER-Attn",
            RESULTS_DIR / "aiter",
            extra_env={},
            overrides=overrides
        )


def print_summary():
    eth_suffix = "_eth" if FORCE_ETH else ""
    title_suffix = " (Ethernet ONLY)" if FORCE_ETH else ""
    print(f"\n{f'MODEL (TP={CLUSTER_TP}){title_suffix}':<50} | {'Tag':<15} | {'Triton':<8} | {'ROCm':<8} | {'AITER':<8}")
    print("-" * 103)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        name_cell = m.split('/')[-1]
        
        # Find all tags used for this model by looking at the files in RESULTS_DIR
        prefix = f"{msafe}_cluster_tp{CLUSTER_TP}{eth_suffix}"
        
        # Gather all unique tags from both directories
        tags = set()
        for p in (RESULTS_DIR / "triton").glob(f"{prefix}*_throughput.json"):
            # Extract tag: {prefix}_{tag}_throughput.json or {prefix}_throughput.json
            name_part = p.name[len(prefix):-len("_throughput.json")]
            tag = name_part.lstrip("_")
            tags.add(tag)
            
        for p in (RESULTS_DIR / "rocm").glob(f"{prefix}*_throughput.json"):
            name_part = p.name[len(prefix):-len("_throughput.json")]
            tag = name_part.lstrip("_")
            tags.add(tag)
            
        for p in (RESULTS_DIR / "aiter").glob(f"{prefix}*_throughput.json"):
            name_part = p.name[len(prefix):-len("_throughput.json")]
            tag = name_part.lstrip("_")
            tags.add(tag)
            
        if not tags:
            tags.add("") # Default empty tag if no files found
            
        # Sort so empty tag (Default) comes first
        for tag in sorted(list(tags)):
            tag_suffix = f"_{tag}" if tag else ""
            
            # Default (Triton)
            try: 
                p1 = (RESULTS_DIR / "triton") / f"{prefix}{tag_suffix}_throughput.json"
                if p1.exists():
                    d1 = json.loads(p1.read_text())
                    val1 = f"{d1.get('tokens_per_second', 0):.1f}"
                else:
                    val1 = "N/A"
            except: val1 = "N/A"
            
            # ROCm
            try:
                p2 = (RESULTS_DIR / "rocm") / f"{prefix}{tag_suffix}_throughput.json"
                if p2.exists():
                    d2 = json.loads(p2.read_text())
                    val2 = f"{d2.get('tokens_per_second', 0):.1f}"
                else:
                    val2 = "N/A"
            except: val2 = "N/A"

            # AITER
            try:
                p3 = (RESULTS_DIR / "aiter") / f"{prefix}{tag_suffix}_throughput.json"
                if p3.exists():
                    d3 = json.loads(p3.read_text())
                    val3 = f"{d3.get('tokens_per_second', 0):.1f}"
                else:
                    val3 = "N/A"
            except: val3 = "N/A"

            display_tag = tag if tag else "(Default)"
            print(f"{name_cell:<50} | {display_tag:<15} | {val1:<8} | {val2:<8} | {val3:<8}")
            
    print("-" * 103)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM High-Concurrency Throughput Benchmark Suite (Cluster)")
    parser.add_argument("--eth-only", action="store_true", help="Run benchmark using only Ethernet (disable RDMA/RoCE)")
    parser.add_argument("--debug-nccl", action="store_true", help="Enable NCCL Debug logging (INFO level for Transport tracking)")
    parser.add_argument("--tui", action="store_true", help="Launch interactive configuration UI")
    args = parser.parse_args()
    
    FORCE_ETH = args.eth_only
    FORCE_DEBUG_NCCL = args.debug_nccl

    selected_models = MODELS_TO_RUN
    
    if args.tui:
        # 1. Cluster IPs Configuration
        form_args = [
            "--clear", "--backtitle", "AMD VLLM Cluster Configuration",
            "--title", "Cluster Network Details",
            "--form", "Verify Head and Worker IPs for this run:",
            "10", "60", "2",
            "Head Node IP:", "1", "1", HEAD_IP, "1", "20", "20", "0",
            "Worker Node IP:", "2", "1", WORKER_IP, "2", "20", "20", "0"
        ]
        res = bench_utils.run_dialog(form_args)
        if res is None:
            subprocess.run(["clear"])
            print("Cancelled by user.")
            sys.exit(0)
            
        lines = res.splitlines()
        if len(lines) >= 2:
            HEAD_IP = lines[0].strip()
            WORKER_IP = lines[1].strip()
            os.environ["VLLM_HEAD_IP"] = HEAD_IP
            os.environ["VLLM_WORKER_IP"] = WORKER_IP
            
        # 2. Network Options (ETH / Debug)
        eth_status = "on" if FORCE_ETH else "off"
        debug_status = "on" if FORCE_DEBUG_NCCL else "off"
        check_args = [
            "--title", "Network Overrides",
            "--checklist", "Select custom backend flags:", "10", "60", "2",
            "ETH_ONLY", "Force Ethernet (Disable RDMA/RoCE)", eth_status,
            "DEBUG_NCCL", "Enable NCCL debug logs", debug_status
        ]
        flags_res = bench_utils.run_dialog(check_args)
        if flags_res is not None:
            FORCE_ETH = "ETH_ONLY" in flags_res
            FORCE_DEBUG_NCCL = "DEBUG_NCCL" in flags_res

        # 3. Model Selection
        checklist_args = [
            "--title", "Model Selection",
            "--checklist", "Select models to benchmark:", "20", "65", "10"
        ]
        for m in MODELS_TO_RUN:
            m_name = m.split("/")[-1]
            checklist_args.extend([m, m_name, "on"])
            
        choice = bench_utils.run_dialog(checklist_args)
        if choice is None:
            subprocess.run(["clear"])
            print("Cancelled by user.")
            sys.exit(0)
            
        import shlex
        selected_models = [m for m in shlex.split(choice)]
        if not selected_models:
            subprocess.run(["clear"])
            print("No models selected. Exiting.")
            sys.exit(0)

    log("Ray Cluster Detected. Starting Benchmarks (Dual Backend)...")
    log("NOTE: Running Peak Throughput Benchmark. This simulates high-concurrency batching to saturate hardware bandwidth.")
    log("This does NOT represent single-request user generation speed (Concurrency=1).")
    if FORCE_ETH:
        log("Note: Ethernet ONLY mode enabled. RDMA/RoCE disabled.")
    if FORCE_DEBUG_NCCL:
        log("Note: NCCL Debug mode enabled (Transport Logging).")
    log("Note: Eager Mode (--enforce-eager) is ENABLED for cluster stability.")
    
    for m in selected_models:
        overrides = {}
        if args.tui:
            config = MODEL_TABLE.get(m, {})
            default_seqs = config.get("max_num_seqs", "16")
            default_tokens = config.get("max_tokens", DEFAULT_BATCH_TOKENS)
            default_util = config.get("gpu_util", GPU_UTIL)
            default_ctx = config.get("ctx", "auto")
            
            form_args = [
                "--clear", "--backtitle", f"AMD VLLM Cluster Benchmark Configuration (TP: {CLUSTER_TP})",
                "--title", f"Tune Parameters: {m.split('/')[-1]}",
                "--form", "Edit cluster model options. Leave tag empty for no suffix.",
                "15", "70", "5",
                "Max Concurrent Seqs:", "1", "1",  str(default_seqs), "1", "25", "15", "0",
                "Max Batched Tokens:", "2", "1", str(default_tokens), "2", "25", "15", "0",
                "GPU Utilization (0-1):", "3", "1", str(default_util), "3", "25", "15", "0",
                "Max Context Length:", "4", "1", str(default_ctx), "4", "25", "15", "0",
                "Filename Tag (Optional):", "5", "1", "", "5", "25", "15", "0"
            ]
            
            form_res = bench_utils.run_dialog(form_args)
            if form_res is None:
                subprocess.run(["clear"])
                print(f"Skipping {m} due to user cancellation.")
                continue
                
            lines = form_res.splitlines()
            if len(lines) >= 5:
                overrides["max_num_seqs"] = lines[0].strip()
                overrides["max_tokens"] = lines[1].strip()
                overrides["gpu_util"] = lines[2].strip()
                
                ctx_val = lines[3].strip()
                if ctx_val and ctx_val.lower() != "auto":
                    overrides["ctx"] = ctx_val
                    
                overrides["tag"] = lines[4].strip()

        run_cluster_throughput(m, overrides=overrides)
        
    print_summary()
