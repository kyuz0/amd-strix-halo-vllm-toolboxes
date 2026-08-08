#!/usr/bin/env python3
import subprocess
import time
import socket
import json
import sys
import os
import requests
import re
import argparse
from pathlib import Path
try:
    from transformers import AutoConfig
except ImportError:
    print("Error: 'transformers' not found. Please install it or run in vLLM environment.")
    sys.exit(1)


# Import path handling for scripts/models.py
try:
    import sys, os
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    import models
    import cluster_manager # Import shared cluster logic
except ImportError:
    print("Error: Could not import scripts/models.py or cluster_manager.py.")
    sys.exit(1)

# Import Utils from run_vllm_bench (keep utils shared)
try:
    from run_vllm_bench import kill_vllm
    # We do NOT import get_gpu_count because we are overriding it for cluster awareness
except ImportError:
    print("Error: Could not import run_vllm_bench.py.")
    sys.exit(1)

MODEL_TABLE = models.MODEL_TABLE
MODELS_TO_RUN = models.MODELS_TO_RUN

# =========================
# 🧠 GROUNDING & METHODOLOGY
# =========================
# This script finds the Maximum Working Context (MWC) for vLLM models.
#
# Methodology:
# 1. **Inspect**: Use `transformers.AutoConfig` to determine the model's theoretical limit 
#    (e.g., `max_position_embeddings`). 
# 2. **Probe**: Launch `vllm serve` at this limit.
# 3. **React**: 
#    - If stable ("Application startup complete"): Success.
#    - If OOM ("KV cache capacity... is X"): Retry with vLLM's suggested X.
#    - If Config Error ("max_model_len... is Y"): Retry with vLLM's suggested Y.

# =========================
# ⚙️ CONFIG
# =========================
HOST = "127.0.0.1"
PORT = 8000
RESULTS_FILE = Path("max_context_results.json")
REPORT_FILE = Path("max_context_report.md")

# We test these GPU Utilizations steps to see how much we can squeeze
# 0.90 is default, but we want MAX context.
# 0.58 is our target high. 0.90 is the fallback.
GPU_UTIL_STEPS = ["0.95", "0.90"]
# We test these concurrency settings
CONCURRENCY_STEPS = [1, 4, 8, 16]

def log(msg):    print(f"[MAX-CTX] {msg}", flush=True)

def get_gpu_count(use_cluster=False):
    """
    Returns total GPUs. 
    If --use-cluster is passed AND Ray Cluster is active, returns cluster GPU count.
    Otherwise returns 1 (local single-GPU mode, no Ray).
    """
    if use_cluster and cluster_manager.check_ray_status():
        log("Ray Cluster Detected: Assuming 2 GPUs available.")
        return 2
        
    # Local mode: always 1 GPU. Even if rocm-smi reports multiple entries,
    # we don't use Ray for TP>1 without --use-cluster.
    return 1


def get_hf_context_limit(model_name, trust_remote=False):
    # ... (Keep existing implementation)
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote)

        # Gemma 3 and similar multi-config models
        if hasattr(cfg, "text_config"):
            tc = cfg.text_config
            if hasattr(tc, "max_position_embeddings"):
                return int(tc.max_position_embeddings)

        # Standard HF attributes
        for attr in (
            "max_position_embeddings",
            "seq_length",
            "max_seq_len",
            "n_positions",
        ):
            val = getattr(cfg, attr, None)
            if val is not None:
                return int(val)

        return 8192

    except Exception as e:
        log(f"Warning: Could not read config for {model_name}: {e}. Defaulting to 32768.")
        return 32768

def get_vllm_server_cmd(model, tp_size, util, max_len, max_seqs, use_cluster=False):
    """
    Constructs the vLLM serve command.
    Only uses Ray Backend if use_cluster=True AND tp_size > 1.
    """
    config = MODEL_TABLE[model]
    
    cmd = [
        "vllm", "serve", model,
        "--gpu-memory-utilization", str(util),
        "--max-model-len", str(max_len),
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", str(max_seqs),
        "--dtype", "auto",
        "--attention-backend", "TRITON_ATTN",
        # "--disable-log-stats" 
    ]
    
    # Env Setup
    env = os.environ.copy()
    env.update(config.get("env", {}))

    # CLUSTER / RAY LOGIC
    # Only engage Ray if explicitly requested via --use-cluster
    if use_cluster and tp_size > 1:
        log(f"TP={tp_size} > 1: Using Ray Distributed Backend")
        cmd.extend(["--distributed-executor-backend", "ray"])
        
        rdma_iface = cluster_manager.get_net_iface()
        head_ip = cluster_manager.get_local_ip(rdma_iface)
        cmd.extend(["--host", head_ip])
        
        env["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
        env["VLLM_HOST_IP"] = head_ip
        env["NCCL_SOCKET_IFNAME"] = rdma_iface
        env["NCCL_IB_GID_INDEX"] = "1"
        env["NCCL_IB_DISABLE"] = "0"
        env["NCCL_NET_GDR_LEVEL"] = "0"
    else:
        # Default Localhost bind for single node safety
        cmd.extend(["--host", "127.0.0.1"])
        
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    
    return cmd, env

def is_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def force_cleanup(hard=False):
    """
    Kills vLLM using multiple methods and ensures port is free.
    BLOCKS until processes are definitely gone.
    """
    timeout = 20 if hard else 10
    start_time = time.time()
    
    while True:
        # 1. Aggressive Kill Commands
        # We send these EVERY loop iteration until they die.
        subprocess.run("pkill -9 -f 'vllm.entrypoints.api_server'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f 'vllm serve'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f 'VLLM::'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f 'multiprocessing.spawn'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f ray::", shell=True, stderr=subprocess.DEVNULL)

        # 2. Check if they are still there
        # We check specifically for the persistence of any vllm-related process
        # We use explicit list to know WHICH one triggered it
        # CRITICAL FIX: We MUST use shell=False otherwise 'pgrep -f pattern' 
        # matches the 'sh -c pgrep ... pattern' command content itself!
        dirty = False
        
        # Check 1: vllm serve
        if subprocess.run(["pgrep", "-f", "vllm serve"], stdout=subprocess.DEVNULL).returncode == 0:
            # Double check it's not us (Python script)
            # But simpler to just proceed if we trust shell=False works
            log("Clean waiting: Found 'vllm serve' process:")
            subprocess.run("pgrep -a -f 'vllm serve'", shell=True) # debug
            dirty = True
            
        # Check 2: api_server
        if subprocess.run(["pgrep", "-f", "vllm.entrypoints.api_server"], stdout=subprocess.DEVNULL).returncode == 0:
            log("Clean waiting: Found 'vllm.entrypoints.api_server' process:")
            subprocess.run("pgrep -a -f 'vllm.entrypoints.api_server'", shell=True) # debug
            dirty = True
            
        # Check 3: VLLM:: (Ray workers)
        if subprocess.run(["pgrep", "-f", "VLLM::"], stdout=subprocess.DEVNULL).returncode == 0:
            log("Clean waiting: Found 'VLLM::' process:")
            subprocess.run("pgrep -a -f 'VLLM::'", shell=True) # debug
            dirty = True

        if not dirty:
            # Processes are gone. Now check port.
            if is_port_free(PORT):
                time.sleep(1) # Final safety buffer
                return # Clean!
            else:
                log("Clean: Processes gone, but Port 8000 still held. Waiting...")
        else:
            log("Clean: Processes still detected. Retrying kill...")
        
        if time.time() - start_time > timeout:
            log("CRITICAL: Cleanup timed out! Force attempting `killall -9 vllm` as last resort.")
            subprocess.run("killall -9 vllm", shell=True, stderr=subprocess.DEVNULL)
            break
            
        time.sleep(1.5) # Wait a bit before hammering again


def wait_for_server_and_parse(process, timeout=900):
    """
    Waits for server to be ready.
    Parses stdout for "Count of GPU blocks" and "Block size".
    Returns: (ready_bool, gpu_blocks, block_size, max_len_clamped, failure_reason)
    """
    start = time.time()
    gpu_blocks = 0
    block_size = 16 # default
    max_len_clamped = None
    
    logs = []
    failure_reason = None
    
    while time.time() - start < timeout:
        if process.poll() is not None:
            # Process died.
            for line in process.stdout:
                line_str = line.decode("utf-8", errors="replace").strip()
                logs.append(line_str)
            
            # SCAN FULL HISTORY if not found yet
            # Sometimes error was in previous lines or split
            if not failure_reason:
                full_log = "\n".join(logs)
                
                # Check 1: Sampler OOM
                if "warming up sampler" in full_log and "CUDA out of memory" in full_log:
                     failure_reason = "Sampler Warmup OOM"
                
                # Check 2: Explicit vLLM suggestion (Estimated)
                # "estimated maximum model length is 127120"
                elif "estimated maximum model length is" in full_log:
                     m = re.search(r"estimated maximum model length is (\d+)", full_log)
                     if m:
                         failure_reason = f"estimated maximum model length is {m.group(1)}"
                         
                # Check 3: Derived Max Model Len
                # "derived max_model_len (max_position_embeddings=131072.0 ...)"
                elif "derived max_model_len" in full_log:
                     failure_reason = "derived max_model_len detected"
                
                # Check 4: Capacity/Value Error
                elif "ValueError" in full_log and "maximum number of tokens" in full_log:
                     failure_reason = "Capacity Error (Found in history)"
                
                # Check 5: Generic OOM
                elif "CUDA out of memory" in full_log or "hipErrorOutOfMemory" in full_log:
                     failure_reason = "OOM detected"
            
            if not failure_reason:
                # Unexpected death! Dump logs to see why.
                log("CRITICAL: Process died unexpectedly! Dumping last 100 lines:")
                print("=== vLLM SERVER LOGS (LAST 100 LINES) ===")
                for l in logs[-100:]:
                    print(l)
                print("=============================================")
                    
            return False, 0, 0, None, failure_reason
            
        line = process.stdout.readline()
        if line:
            line_str = line.decode("utf-8", errors="replace").strip()
            logs.append(line_str)
            
            # 1. Parse Legacy "GPU blocks" (if present)
            m_blocks = re.search(r"# GPU blocks:\s*(\d+)", line_str)
            if m_blocks:
                gpu_blocks = int(m_blocks.group(1))
                block_size = 16 # assume default unless found
                log(f"  -> Found GPU blocks: {gpu_blocks} (Legacy)")

            # 2. Parse Newer "GPU KV cache size" (vLLM 0.11+)
            # "GPU KV cache size: 111,536 tokens"
            m_kv_tokens = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line_str)
            if m_kv_tokens:
                tokens_str = m_kv_tokens.group(1).replace(",", "")
                gpu_blocks = int(tokens_str) # We use 'gpu_blocks' variable to store total tokens now for simplicity
                block_size = 1 # Effectively 1 because we have the total count
                log(f"  -> Found GPU KV Cache tokens: {gpu_blocks}")

            # 3. Parse Block Size (optional, mostly for legacy)
            m_bs = re.search(r"block_size=(\d+)", line_str)
            if m_bs:
                block_size = int(m_bs.group(1))

            # Failure hints
            if "ValueError" in line_str and "maximum number of tokens" in line_str:
                failure_reason = line_str
            if "derived max_model_len" in line_str:
                failure_reason = line_str
            if "warming up sampler" in line_str and "CUDA out of memory" in line_str:
                failure_reason = "Sampler Warmup OOM"
            elif "CUDA out of memory" in line_str or "hipErrorOutOfMemory" in line_str:
                failure_reason = "OOM detected"

            # Check for startup
            if "Application startup complete" in line_str or "Uvicorn running on" in line_str:
                if gpu_blocks > 0:
                    log("  -> Server signal detected. Waiting 5s for socket stability...")
                    time.sleep(5)
                    return True, gpu_blocks, block_size, max_len_clamped, None
                else:
                    return False, 0, 0, None, "Parsed Success but Token/Block Count was 0"
                
    # Timeout case
    log("CRITICAL: Server startup timed out! Dumping last 100 lines:")
    print("=== vLLM SERVER LOGS (LAST 100 LINES) ===")
    for l in logs[-100:]:
        print(l)
    print("=============================================")
    return False, 0, 0, None, "Timeout"

def verify_context(model, context_len):
    """
    Sends a request to the server with length ~context_len to verify stability.
    """
    # Use dynamic host if set (by cluster logic), else localhost
    # But wait, the env var is set for the SERVER process, not necessarily us?
    # Actually, we (the client script) need to know where to send requests.
    # If we are on Head, localhost is fine for Head-based server. 
    # But if we use Ray, vLLM head usually binds to HOST IP.
    
    target_host = os.getenv("VLLM_HOST_IP", "127.0.0.1")
    url = f"http://{target_host}:{PORT}/v1/completions"
    
    # We use a simple "A " * N prompt.
    # Llama 3 tokenizer: "A" is usually 1 token.
    
    prompt = "A " * int(context_len * 0.5) # 50% fill to be safe/approx
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0
    }
    
    # Retry loop for connection refusals (race condition)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Increased timeout to 300s because prefilling 60k+ tokens takes time!
            r = requests.post(url, json=payload, timeout=300)
            if r.status_code == 200:
                return True, "Success"
            else:
                # If 500 or 400 error, maybe we shouldn't retry? Usually yes for 500 if transient.
                # But for now let's just fail or retry.
                # If we are OOMing, we will likely get a 500 or it will hang.
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                log(f"  -> Connection refused. Retrying verification ({attempt+1}/{max_retries})...")
                time.sleep(2)
            else:
                return False, "Connection Refused (Max Retries)"
        except Exception as e:
            return False, str(e)
            
    return False, "Unknown Error"

def run_probe(model, tp, util, max_seqs, start_limit=None, use_cluster=False):
    """
    Probes a specific configuration starting from the model's architectural limit.
    """
    trust_remote = MODEL_TABLE[model].get("trust_remote", False)
    # 1. Get the Advertised Limit (The "Smart" Way)
    arch_limit = get_hf_context_limit(model, trust_remote)
    
    # Intelligent Start: If we know a lower limit worked for lower concurrency, start there.
    target_len = arch_limit
    if start_limit:
        target_len = min(arch_limit, start_limit)
        log(f"  -> Smart Start: Capping initial probe at {target_len} (based on previous run)")
    
    result_data = {
        "model": model,
        "tp": tp,
        "util": util,
        "max_seqs": max_seqs,
        "model_limit": arch_limit,
        "configured_len": 0,
        "real_capacity": 0,
        "status": "fail",
        "error": ""
    }

    log(f"Probing {model} | TP={tp} | Util={util} | Seqs={max_seqs} | Model Limit={arch_limit}")
    
    # We loop until we succeed OR we drop below a useful context size.
    while target_len >= 2048:
        force_cleanup()
        
        cmd, env = get_vllm_server_cmd(model, tp, util, target_len, max_seqs, use_cluster=use_cluster)
        log(f"DEBUG: Cmd: {' '.join(cmd)}")
        
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            ready, blocks, block_size, _, fail_msg = wait_for_server_and_parse(proc)
            
            if ready:
                # Success - but let's VERIFY it actually answers
                total_capacity = blocks * block_size
                workable_len = min(target_len, total_capacity)
                
                # Verify with actual request
                # We cap verification at 4096 because we just want to know if it crashes, 
                # we don't need to wait for a 128k context fill just for a liveness check.
                verify_len = min(workable_len, 4096)
                log(f"  -> Server ready. Verifying stability with approx {int(verify_len * 0.5)} tokens (capped at 4k)...")
                v_ok, v_msg = verify_context(model, verify_len)
                
                if v_ok:
                    log(f"  -> Success! capacity={total_capacity}, configured={workable_len}")
                    log(f"  -> Verification passed: {v_msg}")
                    
                    # Cleanup SUCCESSFUL process immediately
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except: proc.kill()
                    
                    result_data["status"] = "success"
                    result_data["configured_len"] = target_len
                    result_data["real_capacity"] = total_capacity
                    result_data["max_context_1_user"] = workable_len
                    
                    return result_data
                else:
                    log(f"  -> Server started, but Verification FAILED: {v_msg}")
                    # Treat as a crash/failure, back off
                    fail_msg = "Verification Failed"
                    
                    # Capture any remaining logs if the process is dead or dying
                    # Or just read what's currently available non-blocking? 
                    # Simpler: just terminate and read output.
                    proc.terminate()
                    try: 
                        outs, errs = proc.communicate(timeout=5)
                        if outs:
                            print("=== vLLM SERVER LOGS (DURING VERIFICATION FAILURE) ===")
                            print(outs.decode('utf-8', errors='replace'))
                            print("======================================================")
                    except: 
                        proc.kill()

            
            # If we fall through here, ready=False OR verify=False
            log(f"  -> Attempt failed at {target_len}")
            if fail_msg: log(f"     Reason: {fail_msg}")
            result_data["error"] = fail_msg if fail_msg else "Process died or timed out"
                
            if fail_msg:
                # Case V: Verification Failed (Server up, but unstable inference)
                # User requests drop to 0.95 tier immediately.
                # Must check this FIRST to ensure we don't fall through.
                if "Verification Failed" in str(fail_msg):
                    log("  -> Verification Failed (Unstable). Aborting this Util, dropping to lower tier.")
                    break

                # Case S: Sampler Warmup OOM (Fatal for this Util)
                if "Sampler Warmup OOM" in fail_msg:
                    log("  -> Critical Sampler OOM. Utilization/Seqs too high. Aborting this configuration.")
                    break # Give up on this Util/Seq combo immediately

                # Case X: Dirty State / Zombie VRAM 
                # "Free memory on device (1.56/31.86 GiB) on startup is less than desired..."
                if "Free memory on device" in fail_msg and "less than desired" in fail_msg:
                        log("  -> Dirty VRAM detected (previous run didn't cleanup?). Retrying with HARD cleanup.")
                        force_cleanup(hard=True)
                        continue # Retry SAME target_len

                # Case A: VRAM Limit ("maximum number of tokens... is X")
                m_capacity = re.search(r"maximum number of tokens.*?KV cache is (\d+)", fail_msg)
                if m_capacity:
                    cap = int(m_capacity.group(1))
                    log(f"  -> Found Hardware Capacity: {cap}")
                    target_len = cap
                    continue # Retry Exact Cap

                # Case B: Model Limit mismatch 
                # "Value error, User-specified max_model_len (500000) is greater than the derived max_model_len (max_position_embeddings=131072.0 ...)"
                # We regex for 'derived max_model_len' and then look for numbers in the proximity.
                
                if "derived max_model_len" in fail_msg:
                    # Try to capture "max_position_embeddings=131072"
                    m_pos = re.search(r"max_position_embeddings=([\d\.]+)", fail_msg)
                    if m_pos:
                        limit = int(float(m_pos.group(1))) # handle 131072.0
                        log(f"  -> Found Model Limit: {limit}")
                        target_len = limit
                        continue
                        
                    # Fallback: look for simple parenthesis pattern if the above fails
                    m_derived = re.search(r"derived max_model_len\s*\((\d+)\)", fail_msg)
                    if m_derived:
                        limit = int(m_derived.group(1))
                        log(f"  -> Found Model Limit (Legacy): {limit}")
                        target_len = limit
                        continue

                # Case C: Estimated Max Length (New vLLM Safe Limit)
                # "estimated maximum model length is 111536"
                m_est = re.search(r"estimated maximum model length is (\d+)", fail_msg)
                if m_est:
                    limit = int(m_est.group(1))
                    log(f"  -> Found vLLM Estimated Limit: {limit}")
                    target_len = limit
                    continue

            # Case D: Generic OOM/Crash
            target_len = int(target_len * 0.8)
            log(f"  -> Backing off to: {target_len}")
                
            if target_len < 2048:
                log("  -> Give up (too small)")
                break
        finally:
            if proc:
                try: proc.terminate()
                except: pass
                try: proc.kill() 
                except: pass
                proc.wait() 
            force_cleanup()
                
    return result_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Filter to run only this model (substring match)")
    parser.add_argument("--steps", type=int, default=-1, help="Number of models to run (default: all)")
    parser.add_argument("--use-cluster", action="store_true", help="Enable testing TP>1 models over the Ray cluster if available")
    args = parser.parse_args()

    gpu_count = get_gpu_count(use_cluster=args.use_cluster)
    
    # 1. Load existing results to support RESUME
    results = []
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
            log(f"Loaded {len(results)} previous results. Resuming...")
        except Exception as e:
            log(f"Warning: Could not read existing results: {e}")

    count = 0
    for model in MODELS_TO_RUN:
        if args.model and args.model not in model:
            continue
            
        config = MODEL_TABLE[model]
        
        # KEY CHANGES:
        # We only want to test the MINIMUM required TP.
        # If model supports 1 and 2, we ONLY test 1 (local is faster/easier).
        # We only test 2 if model VALID_TP *starts* with 2 (or higher).
        
        valid_tps = config.get("valid_tp", [1])
        min_tp = min(valid_tps)
        
        if min_tp > gpu_count:
            log(f"Skipping {model}: Requires TP={min_tp} but only {gpu_count} GPUs available.")
            continue
            
        tps_to_test = [min_tp]
        
        for tp in tps_to_test:
            # Track successful seqs for this TP to skip lower utils
            # effectively: {seqs_count: max_working_util}
            # Since we iterate high-util -> low-util, if we succeeded already for this 'seqs', we skip.
            successful_seqs = set() 
            
            # Reset smart limit for each TP (TP2 should not inherit TP1's limit)
            last_working_len = None 
            
            for util in GPU_UTIL_STEPS:
                
                for seqs in CONCURRENCY_STEPS:
                    if seqs in successful_seqs:
                        log(f"Skipping {model} (TP={tp}, Util={util}, Seqs={seqs}) - Already succeeded at higher util.")
                        continue

                    # Check if we already have this result
                    existing_res = next((r for r in results 
                                         if r["model"] == model 
                                         and r["tp"] == tp 
                                         and str(r["util"]) == str(util) 
                                         and r["max_seqs"] == seqs), None)
                    
                    if existing_res:
                        res = existing_res
                        log(f"Skipping {model} (TP={tp}, Util={util}, Seqs={seqs}) - Found in results.")
                    else:
                        # New run
                        res = run_probe(model, tp, util, seqs, start_limit=last_working_len, use_cluster=args.use_cluster)
                        results.append(res)
                        
                        # Save immediately
                        with open(RESULTS_FILE, "w") as f:
                            json.dump(results, f, indent=2)

                    # Update logic for Resume OR New Run:
                    if res["status"] == "success":
                        last_working_len = res["configured_len"]
                        successful_seqs.add(seqs) # Mark this seq count as done for this TP

                    # Smart Break: If we failed at this concurrency level (capacity=0), 
                    # higher concurrency will also fail.
                    if res["real_capacity"] == 0 or res["status"] == "fail":
                        log(f"Stopping higher concurrency tests for {model} (failed at {seqs} seqs)")
                        break

        count += 1
        if args.steps != -1 and count >= args.steps and not args.model:
             break

    # generate_report(results) - Moved to separate script

if __name__ == "__main__":
    main()
